"""Assembling `TriggerInput` from what the stages actually measured.

The eight abstention triggers read fourteen facts. Every one of them was
measured somewhere upstream, and until now nobody carried them from
there to here — `tests/test_abstain.py` builds the bundle by hand and
names the gap in a docstring.

WHERE EACH FACT COMES FROM, because a trigger that fires on a fact
nobody can trace is worse than no trigger at all:

    live_hypotheses         ADJUDICATE — survived both hard gates
    all_hypotheses          ADJUDICATE — everything that was judged
    leading                 ADJUDICATE — the survivor holding the most
    leading_share           ADJUDICATE — its share of the residual
    missing_sources         the causal graph, via ADJUDICATE's verdict
    hypothesis_confidence   CONFIDENCE — one score per live hypothesis
    lane_kappa              CONFIDENCE — s3's own kappa, not recomputed
    independent_lanes       GATHER — which lanes reported on the leader
    low_reliability_share   the evidence engine's reliability profile
    case_type_accuracy      the calibration ledger, by case type
    source_staleness_hours  CONFIDENCE — s4's source health, in hours
    unavailable_sources     CONFIDENCE — held, required, and empty
    forced                  CONFIDENCE — triggers a cap forces

NOTHING IS RECOMPUTED HERE. Every value is read off an object an earlier
stage produced. If this module ever needs to calculate something, the
calculation belongs in the stage that owns the question — otherwise two
places compute the same number and the day they disagree, the trigger
and the confidence panel tell the reader different stories.

ONE EXPENSIVE CALL, AND WHY IT IS WORTH IT. Trigger T4 asks whether two
hypotheses are statistically indistinguishable, which needs a confidence
score PER live hypothesis, not just for the leader. `hypothesis_confidences`
scores each one. On a case with four survivors that is four scorings
instead of one. The alternative is to approximate T4 from attributed
shares, which would mean the trigger that decides "we cannot tell these
two apart" is itself a guess.
"""

from __future__ import annotations

from datetime import datetime

import duckdb

from engine.abstain.triggers import TriggerInput
from engine.adjudicate.gate import AdjudicationResult, HypothesisVerdict
from engine.confidence.gate import Score, score_case
from engine.contracts import Evidence
from engine.evidence import reliability_profile
from security.policy import User
from semantic_layer.schema import SemanticLayer, get_semantic_layer

#: Lanes GATHER can report from. Named here so `independent_lanes` counts
#: the same three things the gather result reports.
LANES = ("structured", "unstructured", "external")


def leading_hypothesis(
    adjudication: AdjudicationResult,
) -> HypothesisVerdict | None:
    """The survivor holding the most of the residual, or None.

    Ties break on the order ADJUDICATE judged them in, which is the order
    GATHER screened them into — the prior. A tie on attributed share is
    exactly the condition trigger T4 exists to catch, so it is not
    resolved quietly here; it is left for T4 to see.
    """
    surviving = adjudication.surviving
    if not surviving:
        return None
    return max(surviving, key=lambda item: item.attributed_share)


def hypothesis_confidences(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    adjudication: AdjudicationResult,
    *,
    case_type_for: dict[str, str],
    evidence: tuple[Evidence, ...] = (),
    structured_present: frozenset[str] = frozenset(),
    unstructured_present: frozenset[str] = frozenset(),
    layer: SemanticLayer | None = None,
    clock: datetime | None = None,
) -> dict[str, float]:
    """The confidence each live hypothesis would carry on its own.

    What trigger T4 compares. Scored the same way the published number is
    scored — same components, same caps, same map — because "these two are
    indistinguishable" is only meaningful if both were measured on the
    same instrument.
    """
    layer = layer or get_semantic_layer()
    scores: dict[str, float] = {}
    for verdict in adjudication.surviving:
        score = score_case(
            connection,
            user,
            adjudication,
            leading_tag=verdict.tag,
            case_type=case_type_for[verdict.tag],
            evidence=evidence,
            structured_present=structured_present,
            unstructured_present=unstructured_present,
            layer=layer,
            clock=clock,
        )
        scores[verdict.tag] = score.calibrated
    return scores


def independent_lanes(gather, leading_tag: str | None) -> int:
    """How many of GATHER's three lanes reported on the leading hypothesis.

    Not how many lanes RAN. A case where the unstructured lane ran and
    found nothing about this hypothesis has one reading, not two, and
    trigger T6 is entitled to know that.
    """
    if gather is None or leading_tag is None:
        return 0
    count = 0

    if gather.structured_for(leading_tag) is not None:
        count += 1

    unstructured = gather.unstructured
    if unstructured is not None and any(
        tag.hypothesis_tag == leading_tag for tag in unstructured.all_tags
    ):
        count += 1

    # The external lane reports per FEED, not per hypothesis. It counts as
    # an independent reading when at least one feed returned something for
    # the period — the feeds are subscribed precisely because they speak to
    # the hypotheses the graph cannot check internally.
    external = gather.external
    if external is not None and any(external.feeds.values()):
        count += 1

    return count


def low_reliability_share(
    evidence: tuple[Evidence, ...], layer: SemanticLayer | None = None
) -> float:
    """Share of the evidence weight sitting below the reliability floor.

    Read off `engine/evidence.py`'s own profile rather than recounted, so
    trigger T6 and the promotion check in the evidence engine can never
    disagree about whether a hypothesis rests on text.
    """
    if not evidence:
        return 0.0
    return reliability_profile(evidence, layer=layer).low_weight_share


def unavailable_sources(score: Score) -> tuple[str, ...]:
    """Sources the leading hypothesis requires, that the organisation
    holds, and that came back with nothing.

    THE DISTINCTION MATTERS. A source the organisation does not hold is
    trigger T3's business — it is a gap in what was ever bought. A source
    that is held and empty is trigger T8's — something broke this morning.
    Folding the two together would let a broken feed read as a
    procurement decision.
    """
    health = _source_health(score)
    return tuple(
        sorted(
            name
            for name, item in health.items()
            if item.held and not item.completeness and not item.history_weeks
        )
    )


def build_trigger_input(
    adjudication: AdjudicationResult,
    score: Score,
    *,
    gather=None,
    evidence: tuple[Evidence, ...] = (),
    hypothesis_confidence: dict[str, float] | None = None,
    layer: SemanticLayer | None = None,
) -> TriggerInput:
    """The bundle the eight triggers read, from the stages that measured it.

    `hypothesis_confidence` is passed in rather than computed, because
    computing it means one `score_case` per live hypothesis and the caller
    is the one that knows whether it can afford that. Omitted, the leader's
    published score is the only entry — which is honest but blinds T4, so
    the pipeline always supplies the full set.
    """
    layer = layer or get_semantic_layer()
    leading = leading_hypothesis(adjudication)
    leading_tag = leading.tag if leading is not None else None

    if hypothesis_confidence is None:
        hypothesis_confidence = (
            {leading_tag: score.calibrated} if leading_tag is not None else {}
        )

    leading_evidence = evidence or score.evidence
    accuracy, cases = score.calibration.accuracy_for(score.case_type)

    return TriggerInput(
        live_hypotheses=tuple(item.tag for item in adjudication.surviving),
        all_hypotheses=tuple(item.tag for item in adjudication.verdicts),
        leading=leading_tag,
        leading_share=leading.attributed_share if leading is not None else 0.0,
        missing_sources=tuple(leading.missing_sources) if leading is not None else (),
        hypothesis_confidence=dict(hypothesis_confidence),
        lane_kappa=_kappa(score),
        independent_lanes=independent_lanes(gather, leading_tag),
        low_reliability_share=low_reliability_share(leading_evidence, layer),
        case_type_accuracy=accuracy,
        case_type_cases=cases,
        source_staleness_hours=dict(score.source_staleness_hours),
        unavailable_sources=unavailable_sources(score),
        forced=tuple(score.forced_triggers),
    )


def _kappa(score: Score) -> float | None:
    """s3's own kappa. None when only one lane reported.

    Taken from the component's facts rather than recomputed: s3 already
    decided what agreement means between these two lanes, and a second
    opinion here would be a second definition.
    """
    try:
        return score.component("s3").facts.get("kappa")
    except KeyError:  # pragma: no cover - s3 is always scored
        return None


def _source_health(score: Score) -> dict:
    """Per-source health, as s4 measured it."""
    return score.sources or {}


__all__ = [
    "LANES",
    "build_trigger_input",
    "hypothesis_confidences",
    "independent_lanes",
    "leading_hypothesis",
    "low_reliability_share",
    "unavailable_sources",
]
