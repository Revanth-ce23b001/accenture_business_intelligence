"""The stage results, assembled into the frozen contracts.

`Adjudication` is what the narrator, the API and the UI all read. It is
assembled here and nowhere else, from what the four stages before it
produced.

WHAT THIS MODULE IS ALLOWED TO DO. Read values off stage results and put
them in the contract's shape. It picks evidence records out of the
ledgers by the id the emitting stage gave them, it does not recompute
them. Rule 3 holds throughout: every case-level quantity on the contract
is an `Evidence`, so the UI can click any number back to the statement
that produced it.

THE ONE RECORD IT MINTS. `Adjudication.materiality` is the KPI contract's
own limit, and QUALIFY does not emit it as a standalone record — Gate 5
emits the RATIO of residual to limit, which is a different number. So the
limit itself is minted here, through `EvidenceFactory` like everything
else, with `semantic_layer/kpis/<kpi>.yaml` as its source. It is a
contract value, not a measurement, and it says so: the source system is
the semantic layer and the method is a lookup.

WHY RECOMMEND RUNS INSIDE THIS STAGE. CLAUDE.md's pipeline is
"... ADJUDICATE -> VERDICT -> human decides". The recommendation is what
the human decides ON, so it is part of what VERDICT hands over rather
than a sixth stage the UI has to wait for separately. An abstained case
gets one too — a ranked, priced list of ways to find out — which is the
half of RECOMMEND that makes an abstention useful.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import duckdb

from engine.abstain.verdict import HeldResidual, VerdictDecision
from engine.abstain.verdict import to_contract as verdict_contract
from engine.adjudicate.gate import AdjudicationResult, to_contracts
from engine.confidence.gate import Score
from engine.confidence.gate import to_contract as confidence_contract
from engine.contracts import (
    Adjudication,
    ConfidenceBreakdown,
    Evidence,
    Recommendation,
    Verdict,
)
from engine.evidence import EvidenceFactory, EvidenceLedger
from engine.qualify.gate import QualifyResult
from engine.recommend.cost import CostBasis
from engine.recommend.gate import RecommendRequest, RecommendationResult, recommend
from engine.recommend.gate import to_contract as recommendation_contract
from engine.verdict.triggers import leading_hypothesis
from security.policy import User
from semantic_layer.schema import SemanticLayer, get_semantic_layer

#: Stage name for anything minted here. Keeps a VERDICT-minted record
#: distinguishable from a QUALIFY-minted one by its id alone.
EVIDENCE_PREFIX = "verdict"

#: Evidence ids QUALIFY gives the four case-level quantities. Named
#: constants rather than string literals at the point of use, so a rename
#: in QUALIFY breaks here loudly instead of producing a case file with a
#: missing headline.
HEADLINE_ID = "qualify.calendar.headline"
CALENDAR_ID = "qualify.calendar.calendar"
RESIDUAL_ID = "qualify.calendar.residual"
RESIDUAL_INR_ID = "qualify.calendar.residual_inr"
BAND_ID = "qualify.band.residual_band"

#: The evidence vocabulary's own words for "read from a contract" and
#: "looked up rather than computed". Both are validated against
#: `warehouse.yaml -> evidence` when the record is minted, so a typo here
#: fails at emit rather than becoming an unqueryable string in the table.
SEMANTIC_LAYER_ORIGIN = "semantic_layer"
LOOKUP_METHOD = "lookup"

#: The unit a monetary materiality limit must be declared in for the
#: verdict table's third condition to be evaluable.
INR_CRORE_UNIT = "INR_CR"

#: Statuses on `Adjudication`. A case that reached a verdict is closed;
#: one still being adjudicated is in progress.
STATUS_CLOSED = "closed"
STATUS_IN_PROGRESS = "in_progress"


class AssemblyError(RuntimeError):
    """A stage did not produce something the case file requires."""


@dataclass(frozen=True)
class CaseFile:
    """Everything one case is, as frozen contracts.

    This is the object the API serialises and the narrator is handed. It
    carries no engine types — a consumer of a case file never needs to
    import a stage.
    """

    case_id: str
    adjudication: Adjudication
    #: None when there is no verdict to give. A movement a gate killed
    #: never reached one — the gate chip carries that story — and a case
    #: still being adjudicated has not got there yet. Synthesising an
    #: INSUFFICIENT EVIDENCE for either would report an abstention nobody
    #: made, which is the one thing this project must not do.
    verdict: Verdict | None = None
    recommendation: Recommendation | None = None
    #: Kept alongside the contracts because the resolution panel renders
    #: the trace, and the trace is a property of the DECISION rather than
    #: of the verdict it reached.
    decision_trace: tuple[str, ...] = ()

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        return self.adjudication.evidence

    @property
    def confidence(self) -> ConfidenceBreakdown:
        return self.adjudication.confidence

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.adjudication.evidence}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble(
    *,
    case_id: str,
    qualify: QualifyResult,
    adjudication: AdjudicationResult,
    score: Score,
    decision: VerdictDecision,
    held: tuple[HeldResidual, ...],
    opened_at: datetime,
    decided_at: datetime,
    evidence: tuple[Evidence, ...],
    recommendation: Recommendation | None = None,
    elapsed_ms: float | None = None,
    layer: SemanticLayer | None = None,
) -> CaseFile:
    """Build the frozen case file from what the stages produced."""
    layer = layer or get_semantic_layer()
    request = qualify.request

    index = {item.evidence_id: item for item in evidence}
    materiality = _materiality_evidence(request.kpi, held, qualify.as_of, layer)
    if materiality.evidence_id not in index:
        evidence = (*evidence, materiality)

    contract = Adjudication(
        case_id=case_id,
        kpi=request.kpi,
        scope=request.scope,
        grain=request.grain,
        period=request.period,
        opened_at=opened_at,
        status=STATUS_CLOSED,
        headline_movement=_require(index, HEADLINE_ID, "the headline movement"),
        attributed=tuple(
            index[key] for key in (CALENDAR_ID,) if key in index
        ),
        qualified_residual=_require(index, RESIDUAL_ID, "the qualified residual"),
        materiality=materiality,
        empirical_band=index.get(BAND_ID),
        gates=qualify.gates,
        hypotheses=to_contracts(adjudication, layer),
        coverage=min(max(adjudication.coverage, 0.0), 1.0),
        confidence=confidence_contract(score, layer),
        triggers_fired=decision.triggers_fired,
        evidence=evidence,
        elapsed_ms=elapsed_ms,
    )

    return CaseFile(
        case_id=case_id,
        adjudication=contract,
        verdict=verdict_contract(decision, case_id, decided_at),
        recommendation=recommendation,
        decision_trace=decision.trace,
    )


def narrative_document(case_file: CaseFile) -> dict[str, Any]:
    """The frozen JSON document the narrator is handed.

    ONE DEFINITION, USED BY BOTH THE API AND THE FIXTURE RECORDER. The
    fixture key is a content hash of the request, and the request contains
    this document — so if the two ever build it differently, every fixture
    misses and the offline demo loses its narrative for a reason that
    takes an afternoon to find. It happened once; hence this function.

    The narrator has never known where its input came from, which is what
    makes it impossible for it to reach back into the engine and change
    something (rule 1). It gets a dump, after adjudication, and may render
    it and nothing else.

    THE RECOMMENDATION IS NOT IN IT. Action phrasing is a separate model
    task (`phrase_action`) working from playbook fields, and the action
    panel renders the `Recommendation` contract directly. Folding it in
    here would also make the document non-reproducible: RECOMMEND mints
    its evidence with a wall-clock `retrieved_at`, and this document is
    hashed to key the offline fixture.
    """
    payload = case_file.adjudication.model_dump(mode="json")
    if case_file.verdict is not None:
        payload["verdict"] = case_file.verdict.model_dump(mode="json")
    return payload


def held_residuals(
    adjudication: AdjudicationResult,
    *,
    kpi: str,
    layer: SemanticLayer | None = None,
) -> tuple[HeldResidual, ...]:
    """What each surviving hypothesis is left holding, and whether it can
    be checked.

    THE THIRD CONDITION OF THE VERDICT TABLE READS THIS. A hypothesis that
    is alive, cannot be verified with sources the organisation holds, and
    is holding more money than the KPI's materiality limit, blocks
    EXPLAINED. That is what makes #2451 *partially* explained, and it is
    the one condition most systems do not have.
    """
    layer = layer or get_semantic_layer()
    graph = layer.causal_graph.hypotheses
    limit = _materiality_inr(kpi, layer)
    residual_inr = abs(adjudication.request.residual_inr)
    total = abs(adjudication.request.residual_pt) or None

    held: list[HeldResidual] = []
    for verdict in adjudication.surviving:
        template = graph.get(verdict.tag)
        if template is None:
            continue
        # What this hypothesis holds is the residual it did NOT attribute
        # away — the money still resting on an argument nobody closed.
        share = (
            abs(verdict.attributed_pt) / total if total else verdict.attributed_share
        )
        held.append(
            HeldResidual(
                hypothesis=verdict.tag,
                label=template.label,
                verifiable=not template.is_structurally_unverifiable(),
                residual_inr=residual_inr * share,
                materiality_inr=limit,
                missing_sources=tuple(template.missing_sources()),
            )
        )
    return tuple(held)


def unattributed_held(
    adjudication: AdjudicationResult,
    *,
    kpi: str,
    hypothesis: str,
    layer: SemanticLayer | None = None,
) -> HeldResidual | None:
    """The residual no surviving hypothesis attributed, parked on `hypothesis`.

    #2451's shape: H1 takes 79% and the remaining INR 0.86 Cr rests on the
    competitor hypothesis, which is live and unverifiable. The money is
    not H1's, and it is not nobody's — the verdict table needs it attached
    to the argument that would explain it if anyone could check it.
    """
    layer = layer or get_semantic_layer()
    template = layer.causal_graph.hypotheses.get(hypothesis)
    if template is None:
        return None
    unattributed = abs(adjudication.unattributed_inr)
    if not unattributed:
        return None
    return HeldResidual(
        hypothesis=hypothesis,
        label=template.label,
        verifiable=not template.is_structurally_unverifiable(),
        residual_inr=unattributed,
        materiality_inr=_materiality_inr(kpi, layer),
        missing_sources=tuple(template.missing_sources()),
    )


def build_recommendation(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    *,
    case_id: str,
    qualify: QualifyResult,
    adjudication: AdjudicationResult,
    decision: VerdictDecision,
    score: Score,
    basis: CostBasis,
    indistinguishable_count: int = 0,
    layer: SemanticLayer | None = None,
    clock: datetime | None = None,
    open_linked_case: bool = True,
) -> tuple[Recommendation, RecommendationResult]:
    """Run RECOMMEND for this case and return both shapes.

    The contract goes on the case file; the result is kept because the
    resolution panel renders the investigation ranking and the refusal,
    which the contract flattens to strings.
    """
    layer = layer or get_semantic_layer()
    leading = leading_hypothesis(adjudication)
    accuracy, _ = score.calibration.accuracy_for(score.case_type)

    request = RecommendRequest(
        case_id=case_id,
        kpi=qualify.request.kpi,
        scope=qualify.request.scope,
        grain=qualify.request.grain,
        period=qualify.request.period,
        verdict=decision.value,
        leading_driver=leading.tag if leading is not None else None,
        unresolved_drivers=tuple(
            item.tag
            for item in adjudication.surviving
            if leading is None or item.tag != leading.tag
        ),
        attributable_inr=(
            abs(adjudication.request.residual_inr) * leading.attributed_share
            if leading is not None
            else 0.0
        ),
        residual_inr=abs(adjudication.request.residual_inr),
        basis=basis,
        triggers_fired=decision.triggers_fired,
        missing_sources=tuple(leading.missing_sources) if leading is not None else (),
        missing_for_hypothesis=leading.tag if leading is not None else None,
        indistinguishable_count=indistinguishable_count,
        case_type_accuracy=accuracy,
    )
    result = recommend(
        connection,
        user,
        request,
        layer=layer,
        clock=clock,
        open_linked_case=open_linked_case,
    )
    return recommendation_contract(result, layer), result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _require(index: dict[str, Evidence], key: str, what: str) -> Evidence:
    try:
        return index[key]
    except KeyError:
        raise AssemblyError(
            f"no evidence {key!r} in the case ledger, so {what} has nothing behind "
            "it. QUALIFY emits it on Gate 2; a case file cannot be assembled from a "
            "run that never got there."
        ) from None


def _materiality_inr(kpi: str, layer: SemanticLayer) -> float:
    """The KPI's materiality limit, in rupees.

    Declared in the KPI's own unit — crore for a revenue KPI, points for a
    conversion one. A held residual is compared in rupees, so a limit
    expressed in anything else has to be converted or refused. It is
    refused rather than guessed at: a conversion-rate limit of 0.5 points
    is not 0.5 of anything monetary, and inventing an exchange rate
    between them would put a made-up number in the one condition that
    decides EXPLAINED against PARTIALLY EXPLAINED.
    """
    materiality = layer.materiality(kpi)
    if materiality is None:
        raise AssemblyError(
            f"{kpi!r} declares no materiality limit, so nothing can be said about "
            "whether a held residual is above it"
        )
    units = layer.warehouse.units
    if materiality.unit == INR_CRORE_UNIT:
        return materiality.value * units.inr_per_crore
    raise AssemblyError(
        f"{kpi!r} declares its materiality in {materiality.unit!r}; a held residual "
        "is measured in rupees and there is no honest conversion between them. "
        "Declare an INR_CR limit on the contract, or the third verdict condition "
        "cannot be evaluated for this KPI."
    )


def _materiality_evidence(
    kpi: str,
    held: tuple[HeldResidual, ...],
    as_of: datetime,
    layer: SemanticLayer,
) -> Evidence:
    """Mint the KPI's own materiality limit as a record.

    A CONTRACT VALUE, NOT A MEASUREMENT, and the record says so: the
    source system is the semantic layer and the method is a lookup, so the
    UI badges it as read from a contract rather than computed from data.
    """
    materiality = layer.materiality(kpi)
    if materiality is None:
        raise AssemblyError(f"{kpi!r} declares no materiality limit")

    # `retrieved_at` pinned to the data timestamp: this record is a
    # contract lookup, not a measurement taken at a moment, and a wall
    # clock on it would make an otherwise reproducible case file differ
    # between two builds of the same inputs.
    factory = EvidenceFactory.for_stage(
        EVIDENCE_PREFIX, as_of, layer, retrieved_at=as_of
    )
    ledger = EvidenceLedger()
    record = factory.emit(
        "materiality.limit",
        kind="structured_query",
        label=f"Materiality limit for {kpi} ({materiality.display})",
        value=materiality.value,
        unit=materiality.unit,
        source_system=SEMANTIC_LAYER_ORIGIN,
        method=LOOKUP_METHOD,
        description=(
            f"the limit {kpi} declares in its contract, below which a movement is "
            "not worth an investigation"
        ),
        ref=f"semantic_layer/kpis/{kpi}.yaml::thresholds.materiality",
        notes=(
            f"{len(held)} surviving hypotheses were measured against it."
            if held
            else "No surviving hypothesis was left holding residual."
        ),
    )
    ledger.add(record)
    return record


__all__ = [
    "BAND_ID",
    "INR_CRORE_UNIT",
    "LOOKUP_METHOD",
    "SEMANTIC_LAYER_ORIGIN",
    "CALENDAR_ID",
    "EVIDENCE_PREFIX",
    "HEADLINE_ID",
    "RESIDUAL_ID",
    "RESIDUAL_INR_ID",
    "STATUS_CLOSED",
    "STATUS_IN_PROGRESS",
    "AssemblyError",
    "CaseFile",
    "assemble",
    "build_recommendation",
    "held_residuals",
    "narrative_document",
    "unattributed_held",
]
