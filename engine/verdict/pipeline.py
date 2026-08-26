"""The five stages, in order, emitting one event each.

    for event in run_case(connection, user, CaseRequest(...)):
        ...            # VALIDATE, QUALIFY, GATHER, ADJUDICATE, VERDICT
    case = event.run.casefile

A GENERATOR, NOT A FUNCTION THAT RETURNS A CASE. The API streams the
progress rail over SSE, and a rail that only updates when everything has
finished is a spinner with extra steps. Yielding per stage means the
caller sees GATHER land while ADJUDICATE is still running, which is the
whole point of showing the stages at all.

IT STOPS WHERE THE ENGINE STOPS. A movement killed at Gate 1 yields one
event and ends; #2470's shape. A movement that fails Gate 3 for want of
history yields two; #2471's. Nothing downstream is run "just to have
something to show", because a confidence score on a case that never
opened is a number about nothing.

THE FIFTH EVENT IS VERDICT, AND NARRATION IS NOT A STAGE HERE. CLAUDE.md
locks the five names, and the narrative is the model's rendering of a
decision the engine already made. `engine/` does not call the narrator:
the API adds a sixth NARRATE event after this generator is exhausted.
Keeping it out means the engine's stage list is the same five whether or
not there is a model available, which is what makes `MOCK_LLM=true` and
a model-free run the same code path.

WHAT EACH EVENT CARRIES. The stage name, the elapsed milliseconds, the
stage's own result object, and — when the stage ended the run — the
outcome code that stopped it. A consumer that wants to render a kill
chip has the code; one that wants the whole case waits for VERDICT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from time import perf_counter
from typing import Any, Callable, Iterator

import duckdb

from engine.abstain.triggers import evaluate_triggers
from engine.abstain.verdict import HeldResidual, VerdictDecision, decide
from engine.adjudicate.gate import AdjudicateRequest, AdjudicationResult, adjudicate
from engine.confidence.gate import Score, score_case
from engine.contracts import Evidence, Stage
from engine.db import warehouse_clock
from engine.gather.gate import GatherRequest, GatherResult, gather
from engine.qualify.gate import QualifyRequest, QualifyResult, qualify
from engine.qualify.restraint import register_case
from engine.recommend.cost import CostBasis
from engine.validate.checks import ValidationRequest
from engine.validate.gate import ValidationResult, validate
from engine.verdict.casefile import (
    CaseFile,
    assemble,
    build_recommendation,
    held_residuals,
    unattributed_held,
)
from engine.verdict.triggers import (
    build_trigger_input,
    hypothesis_confidences,
    leading_hypothesis,
)
from security.policy import User
from semantic_layer.schema import SemanticLayer, get_semantic_layer

#: The locked order (CLAUDE.md §"Architecture"). Taken from the contract's
#: own vocabulary so there is one copy of it in the repository.
STAGE_ORDER: tuple[Stage, ...] = ("VALIDATE", "QUALIFY", "GATHER", "ADJUDICATE", "VERDICT")

#: The same five, bound to names. Unpacked from the tuple rather than
#: retyped, so the order and the names cannot drift apart — and so the
#: code reads `VERDICT` rather than `VERDICT`, which is both
#: clearer and not a magic number for rule 2's scanner to trip over.
VALIDATE, QUALIFY, GATHER, ADJUDICATE, VERDICT = STAGE_ORDER


class PipelineError(RuntimeError):
    """The run cannot proceed as asked."""


@dataclass(frozen=True)
class CaseRequest:
    """What to investigate."""

    kpi: str
    scope: str
    grain: str
    period: str
    period_start: date
    period_end: date
    comparison_start: date
    comparison_end: date
    comparison_period: str | None = None
    case_id: str | None = None
    direction: str = "down"
    #: Run RECOMMEND and open the linked case the playbook chains to.
    #: Off in a read-only context, because opening a case is a write.
    open_linked_case: bool = True


@dataclass
class CaseRun:
    """Everything the run produced, accumulated stage by stage.

    Mutable while the generator is running and complete when it is
    exhausted. `stopped_at` names the stage that ended it, which is None
    only on a run that reached a verdict.
    """

    request: CaseRequest
    case_id: str
    user: User
    opened_at: datetime

    validation: ValidationResult | None = None
    qualification: QualifyResult | None = None
    gathering: GatherResult | None = None
    judgement: AdjudicationResult | None = None
    score: Score | None = None
    decision: VerdictDecision | None = None
    casefile: CaseFile | None = None

    held: tuple[HeldResidual, ...] = ()
    hypothesis_confidence: dict[str, float] = field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()

    stopped_at: Stage | None = None
    outcome_code: str | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def case_opened(self) -> bool:
        """Whether a case was actually opened, rather than a movement checked."""
        return self.qualification is not None and self.qualification.case_opened

    @property
    def reached_verdict(self) -> bool:
        return self.casefile is not None

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.evidence}


@dataclass(frozen=True)
class StageEvent:
    """One stage, finished."""

    stage: Stage
    ordinal: int
    duration_ms: float
    #: The stage's own result. Typed `Any` because the five are five
    #: different shapes and a consumer that wants one knows which.
    result: Any
    run: CaseRun
    #: Set when this stage ended the run. `None` on a stage that passed.
    outcome_code: str | None = None

    @property
    def terminal(self) -> bool:
        return self.outcome_code is not None or self.stage == VERDICT


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_case(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    request: CaseRequest,
    *,
    layer: SemanticLayer | None = None,
    clock: datetime | None = None,
    provider=None,
    monotonic: Callable[[], float] = perf_counter,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Iterator[StageEvent]:
    """Run the five stages, yielding one event each, stopping where they stop."""
    layer = layer or get_semantic_layer()
    clock = clock or warehouse_clock(connection, layer)

    ms_scale = layer.warehouse.units.milliseconds_per_second
    case_id = request.case_id or _case_id(request, now())
    run = CaseRun(request=request, case_id=case_id, user=user, opened_at=now())

    # --- 1. VALIDATE ------------------------------------------------------
    with _timed(run, VALIDATE, monotonic, ms_scale) as mark:
        run.validation = validate(
            connection,
            user,
            ValidationRequest(
                kpi=request.kpi,
                scope=request.scope,
                grain=request.grain,
                period=request.period,
            ),
            layer=layer,
            clock=clock,
        )
        run.evidence += run.validation.evidence
    yield _event(run, VALIDATE, mark, run.validation)
    if not run.validation.case_opened:
        run.stopped_at, run.outcome_code = VALIDATE, run.validation.outcome_code
        return

    # --- 2. QUALIFY -------------------------------------------------------
    with _timed(run, QUALIFY, monotonic, ms_scale) as mark:
        run.qualification = qualify(
            connection,
            user,
            QualifyRequest(
                kpi=request.kpi,
                scope=request.scope,
                grain=request.grain,
                period=request.period,
                comparison_period=request.comparison_period,
            ),
            layer=layer,
            clock=clock,
        )
        run.evidence += run.qualification.evidence
    yield _event(run, QUALIFY, mark, run.qualification)
    if not run.qualification.case_opened:
        run.stopped_at = QUALIFY
        run.outcome_code = run.qualification.outcome_code
        return

    # A case exists from here on, so it goes on the register before any
    # work is done in its name. A case whose evidence is written before
    # its row is a case the audit log cannot join.
    _register(connection, run, layer)

    decomposition = run.qualification.decomposition
    if decomposition is None:  # pragma: no cover - Gate 2 passed, so it exists
        raise PipelineError(
            f"{case_id} passed Gate 2 without a decomposition; there is no residual "
            "to investigate"
        )

    # --- 3. GATHER --------------------------------------------------------
    with _timed(run, GATHER, monotonic, ms_scale) as mark:
        run.gathering = gather(
            connection,
            user,
            GatherRequest(
                kpi=request.kpi,
                scope=request.scope,
                grain=request.grain,
                period=request.period,
                period_start=request.period_start,
                period_end=request.period_end,
                comparison_start=request.comparison_start,
                comparison_end=request.comparison_end,
                direction=request.direction,
            ),
            layer=layer,
            clock=clock,
            provider=provider,
        )
        run.evidence += run.gathering.evidence
    yield _event(run, GATHER, mark, run.gathering)

    # --- 4. ADJUDICATE ----------------------------------------------------
    with _timed(run, ADJUDICATE, monotonic, ms_scale) as mark:
        run.judgement = adjudicate(
            connection,
            user,
            AdjudicateRequest(
                kpi=request.kpi,
                scope=request.scope,
                grain=request.grain,
                period=request.period,
                period_start=request.period_start,
                period_end=request.period_end,
                comparison_start=request.comparison_start,
                comparison_end=request.comparison_end,
                residual_pt=decomposition.residual_pt,
                residual_inr=abs(decomposition.residual_inr),
                hypotheses=run.gathering.screening.tags,
                onset=_onset(run.gathering, request),
                volume_shares=_volume_shares(run.gathering),
                cause_magnitudes=_cause_magnitudes(run.gathering),
            ),
            layer=layer,
            clock=clock,
        )
        run.evidence += run.judgement.evidence
    yield _event(run, ADJUDICATE, mark, run.judgement)

    # --- 5. VERDICT -------------------------------------------------------
    with _timed(run, VERDICT, monotonic, ms_scale) as mark:
        _judge(connection, run, layer, clock)
    yield _event(run, VERDICT, mark, run.casefile)


def _judge(
    connection: duckdb.DuckDBPyConnection,
    run: CaseRun,
    layer: SemanticLayer,
    clock: datetime,
) -> None:
    """Score, evaluate the eight triggers, decide, recommend, assemble."""
    judgement = run.judgement
    qualification = run.qualification
    graph = layer.causal_graph.hypotheses
    case_types = {tag: template.case_type for tag, template in graph.items()}

    leading = leading_hypothesis(judgement)
    if leading is None:
        # Nothing survived both hard gates. T1's condition exactly, and the
        # case still gets a verdict — that is what abstention IS. Scored
        # against the first hypothesis judged so the components have
        # something to describe, and T1 does the rest.
        leading_tag = judgement.verdicts[0].tag if judgement.verdicts else None
        if leading_tag is None:
            raise PipelineError(
                f"{run.case_id} adjudicated no hypotheses at all; GATHER selected none"
            )
    else:
        leading_tag = leading.tag

    unstructured_present = _tagged_stores(run.gathering, leading_tag, layer)

    run.score = score_case(
        connection,
        run.user,
        judgement,
        leading_tag=leading_tag,
        case_type=case_types[leading_tag],
        evidence=judgement.evidence,
        unstructured_present=unstructured_present,
        layer=layer,
        clock=clock,
    )
    run.evidence += run.score.evidence

    run.hypothesis_confidence = hypothesis_confidences(
        connection,
        run.user,
        judgement,
        case_type_for=case_types,
        evidence=judgement.evidence,
        unstructured_present=unstructured_present,
        layer=layer,
        clock=clock,
    )

    triggers = evaluate_triggers(
        build_trigger_input(
            judgement,
            run.score,
            gather=run.gathering,
            evidence=judgement.evidence,
            hypothesis_confidence=run.hypothesis_confidence,
            layer=layer,
        ),
        layer,
    )

    run.held = _held(judgement, run.request.kpi, layer)

    run.decision = decide(
        coverage=judgement.coverage,
        confidence=run.score.calibrated,
        triggers=triggers,
        held=run.held,
        layer=layer,
    )

    # RECOMMEND runs on every case, decided or abstained — see the note in
    # casefile.py. `open_linked_case` governs only whether the playbook's
    # chained case is OPENED, which is a write and not always wanted.
    recommendation, _ = build_recommendation(
            connection,
            run.user,
            case_id=run.case_id,
            qualify=qualification,
            adjudication=judgement,
            decision=run.decision,
            score=run.score,
            basis=_cost_basis(run, layer),
            indistinguishable_count=len(run.hypothesis_confidence),
            layer=layer,
            clock=clock,
        open_linked_case=run.request.open_linked_case,
    )

    run.casefile = assemble(
        case_id=run.case_id,
        qualify=qualification,
        adjudication=judgement,
        score=run.score,
        decision=run.decision,
        held=run.held,
        opened_at=run.opened_at,
        decided_at=clock,
        evidence=run.evidence,
        recommendation=recommendation,
        layer=layer,
    )
    run.evidence = run.casefile.adjudication.evidence


# ---------------------------------------------------------------------------
# Reading the stages' output
# ---------------------------------------------------------------------------


def _held(
    judgement: AdjudicationResult, kpi: str, layer: SemanticLayer
) -> tuple[HeldResidual, ...]:
    """Every surviving hypothesis's holding, plus the unattributed remainder.

    The remainder is parked on the live UNVERIFIABLE hypothesis when there
    is exactly one, which is #2451's shape: H1 explains 79% and the rest
    rests on a competitor argument nobody can check. With none, or with
    more than one, the remainder is left unassigned — guessing which
    unverifiable argument holds it would be inventing the fact that
    decides the verdict.
    """
    held = list(held_residuals(judgement, kpi=kpi, layer=layer))
    unverifiable = [item for item in held if not item.verifiable]
    if len(unverifiable) == 1:
        extra = unattributed_held(
            judgement, kpi=kpi, hypothesis=unverifiable[0].hypothesis, layer=layer
        )
        if extra is not None:
            held = [item for item in held if item.hypothesis != extra.hypothesis]
            held.append(extra)
    return tuple(held)


def _tagged_stores(gathering: GatherResult, tag: str, layer: SemanticLayer) -> frozenset[str]:
    """Stores the classifier tagged for `tag`, for confidence component s3.

    s3 compares the two lanes. Handing it an empty set when the lane
    actually ran would score the case as though the lanes disagreed, which
    is a statement about the wiring rather than about the case.
    """
    unstructured = gathering.unstructured if gathering else None
    if unstructured is None:
        return frozenset()
    spec = layer.gather.unstructured.classification
    return frozenset(
        item.store_id
        for item in unstructured.all_tags
        if item.hypothesis_tag == tag and item.accepted(spec) and item.store_id
    )


def _volume_shares(gathering: GatherResult) -> dict[str, float]:
    """Per hypothesis, the share of volume its cause touches.

    TEST 2 CANNOT RUN WITHOUT THIS, AND TODAY IT IS ALWAYS EMPTY.
    `StructuredResult` carries one aggregate per hypothesis — mean
    availability, mean ASP — and neither the affected volume share nor
    the movement in the cause is among them. Producing them means a
    second window and a second query per template, which is GATHER's
    work and not something this module may invent: a volume share that
    the pipeline made up would decide a HARD GATE.

    So the map is read off the structured results if they ever carry the
    fields, and is empty otherwise. `test_sufficiency` answers an empty
    map with "not tested, not eliminated" and says so in its detail, which
    is the correct behaviour — an untested hypothesis is not a passed one.

    The visible consequence: on a live #2451, H3 (the price rise) is NOT
    eliminated on sufficiency, because sufficiency did not run. Test 1
    still eliminates H4 and H5 on precedence. Closing this needs
    `semantic_layer/gather.yaml` to declare a volume-share measure per
    template, which is a GATHER change.
    """
    shares: dict[str, float] = {}
    for result in gathering.structured:
        share = getattr(result, "volume_share", None)
        if share is not None:
            shares[result.hypothesis] = float(share)
    return shares


def _cause_magnitudes(gathering: GatherResult) -> dict[str, float]:
    """Per hypothesis, how far its cause moved. Test 2's other half.

    Empty for the same reason and with the same consequence — see
    `_volume_shares`. Both halves are needed: `test_sufficiency` refuses
    to bound an effect from one of them.
    """
    magnitudes: dict[str, float] = {}
    for result in gathering.structured:
        magnitude = getattr(result, "cause_magnitude", None)
        if magnitude is not None:
            magnitudes[result.hypothesis] = float(magnitude)
    return magnitudes


def _onset(gathering: GatherResult, request: CaseRequest) -> date:
    """When the movement began. Test 1 fits the pre-trend on the weeks before it.

    The period start, unless a structured result dates the effect itself.
    THE FALLBACK IS THE EARLIEST DEFENSIBLE DATE ON PURPOSE. Test 1 asks
    whether a cause preceded an effect; a late guess at the onset would
    make causes look like they came first, which is the exact error the
    test exists to catch. Erring early can only make the test harder to
    pass, never easier.
    """
    for result in gathering.structured:
        onset = getattr(result, "effect_onset", None)
        if onset is not None:
            return onset
    return request.period_start


def _cost_basis(run: CaseRun, layer: SemanticLayer) -> CostBasis:
    """The case's own quantities, for RECOMMEND to price against.

    Every field is measured or left at its default. A default is "this
    case did not measure that", and a playbook that needs it is refused
    rather than priced at zero — see the note on `CostBasis`.
    """
    judgement = run.judgement
    leading = leading_hypothesis(judgement) if judgement else None
    matching = leading.matching if leading is not None else None
    exposure = leading.exposure if leading is not None else None

    return CostBasis(
        treated_store_count=len(getattr(matching, "treated", ()) or ()),
        affected_store_count=len(getattr(exposure, "exposed", ()) or ()),
        revenue_at_risk_inr=abs(judgement.request.residual_inr) if judgement else 0.0,
    )


def _register(
    connection: duckdb.DuckDBPyConnection, run: CaseRun, layer: SemanticLayer
) -> None:
    """Put the case on the register. Idempotent for a caller that retries."""
    materiality = run.qualification.materiality
    connection.execute(
        "DELETE FROM case_registry WHERE case_id = ?", [run.case_id]
    )
    register_case(
        connection,
        case_id=run.case_id,
        kpi=run.request.kpi,
        scope=run.request.scope,
        grain=run.request.grain,
        period=run.request.period,
        opened_at=run.opened_at,
        materiality_multiple=materiality.multiple if materiality else None,
    )


def _case_id(request: CaseRequest, moment: datetime) -> str:
    """A readable id when the caller did not supply one.

    Scope and period rather than a UUID, because a case id is read aloud
    in a meeting. The timestamp keeps a re-run of the same movement from
    colliding with the case it is re-running.
    """
    stamp = moment.strftime("%Y%m%dT%H%M%S")
    return f"{request.kpi}.{request.scope}.{request.period}.{stamp}"


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class _Mark:
    """A stage's elapsed time, readable once the block has closed."""

    __slots__ = ("duration_ms",)

    def __init__(self) -> None:
        self.duration_ms = 0.0


class _timed:
    """Time a stage onto the run. Records the elapsed time even on a raise.

    A stage that raises still took time, and the event for a run that died
    in GATHER is the one worth having. The scale factor comes from the
    semantic layer because rule 2 admits no numbers in `engine/` at all,
    not even this one.
    """

    def __init__(
        self, run: CaseRun, stage: Stage, monotonic: Callable[[], float], scale: float
    ):
        self.run, self.stage, self.monotonic = run, stage, monotonic
        self.scale = scale
        self.mark = _Mark()

    def __enter__(self) -> _Mark:
        self.started = self.monotonic()
        return self.mark

    def __exit__(self, *exc_info) -> bool:
        self.mark.duration_ms = (self.monotonic() - self.started) * self.scale
        self.run.latency_ms[self.stage] = self.mark.duration_ms
        return False


def _event(run: CaseRun, stage: Stage, mark: _Mark, result: Any) -> StageEvent:
    outcome = None
    if stage == VALIDATE and run.validation is not None:
        outcome = None if run.validation.case_opened else run.validation.outcome_code
    elif stage == QUALIFY and run.qualification is not None:
        outcome = None if run.qualification.case_opened else run.qualification.outcome_code
    return StageEvent(
        stage=stage,
        ordinal=STAGE_ORDER.index(stage) + 1,
        duration_ms=mark.duration_ms,
        result=result,
        run=run,
        outcome_code=outcome,
    )


__all__ = [
    "STAGE_ORDER",
    "CaseRequest",
    "CaseRun",
    "PipelineError",
    "StageEvent",
    "run_case",
]
