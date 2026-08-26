"""Feedback in, and what it moved back out.

Every endpoint here returns the EFFECT, not an acknowledgement. A loop
whose response is `{"ok": true}` is a loop nobody can tell is running;
these say which prior moved and by how much, or which calibration entry
was written, or why nothing changed — because "nothing changed, and here
is why" is a real and frequent answer.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, Request, status

from api.deps import ConnectionDep, LayerDep, StoreDep, UserDep
from api.schemas import (
    CurveStanding,
    DriverFeedbackRequest,
    FeedbackOptions,
    FeedbackResult,
    OutcomeRequest,
    OutcomeResult,
    PriorStanding,
    PriorsResponse,
    VerdictFeedbackRequest,
)
from engine.learn import curves, feedback, loop, outcomes, priors

router = APIRouter(tags=["learning"])


# ---------------------------------------------------------------------------
# What a reader may say
# ---------------------------------------------------------------------------


@router.get("/api/feedback/options", response_model=FeedbackOptions)
def options(user: UserDep, layer: LayerDep) -> FeedbackOptions:
    """The vocabulary, from the contract.

    The UI renders its buttons from this rather than from a list in the
    frontend. Two copies of a closed set is one copy too many: the day
    somebody adds a fifth verdict option, the button appears without a
    deploy and, more importantly, nobody ships a button the engine will
    refuse.
    """
    spec = layer.learning.feedback
    return FeedbackOptions(
        verdict_actions={
            key: {
                "label": action.label,
                "description": action.description.strip(),
                "requires_reason": action.requires_reason,
                "scores_as_correct": action.scores_as_correct,
            }
            for key, action in spec.verdict_actions.items()
        },
        driver_actions={
            key: {"label": action.label, "description": action.description.strip()}
            for key, action in spec.driver_actions.items()
        },
        action_actions={
            key: {"label": action.label, "description": action.description.strip()}
            for key, action in spec.action_actions.items()
        },
        reason_codes=dict(spec.reason_codes),
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


@router.post(
    "/api/cases/{case_id}/feedback/verdict",
    response_model=FeedbackResult,
    status_code=status.HTTP_201_CREATED,
)
def verdict_feedback(
    case_id: str,
    body: VerdictFeedbackRequest,
    request: Request,
    user: UserDep,
    layer: LayerDep,
    connection: ConnectionDep,
    store: StoreDep,
) -> FeedbackResult:
    """Accept, Modify, Reject with reason, or Request investigation.

    Writes a calibration entry for the first three; the fourth writes
    none, because a request for more work is not a judgement on whether
    the published figure was right.

    The confidence AS PUBLISHED is read off the stored case and stamped on
    the event. A case re-run tomorrow may score differently, and the entry
    this produces is a claim about the figure the reader saw.
    """
    stored = _case(store, case_id)
    adjudication = stored.casefile.adjudication
    return _record_and_apply(
        connection,
        layer,
        user,
        overlay_path=_overlay_path(request),
        case_id=case_id,
        level=feedback.VERDICT_LEVEL,
        action=body.action,
        reason_code=body.reason_code,
        comment=body.comment,
        kpi=adjudication.kpi,
        verdict=stored.casefile.verdict.value,
        confidence=adjudication.confidence.calibrated,
        confidence_raw=adjudication.confidence.raw,
        case_type=_case_type(layer, stored),
    )


@router.post(
    "/api/cases/{case_id}/feedback/driver",
    response_model=FeedbackResult,
    status_code=status.HTTP_201_CREATED,
)
def driver_feedback(
    case_id: str,
    body: DriverFeedbackRequest,
    request: Request,
    user: UserDep,
    layer: LayerDep,
    connection: ConnectionDep,
    store: StoreDep,
) -> FeedbackResult:
    """Confirm or reject one driver. Moves its prior for this KPI.

    The response carries the prior before and after, so the reader can see
    that saying "no, it wasn't stock-outs" did something. It usually does
    something small — the declared prior is worth twenty observations —
    and small is the honest size of one person's opinion.
    """
    stored = _case(store, case_id)
    adjudication = stored.casefile.adjudication
    known = {item.hypothesis_id for item in adjudication.hypotheses}
    if body.driver not in known:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"{body.driver!r} is not a driver on case {case_id}; it adjudicated "
                f"{sorted(known)}"
            ),
        )
    return _record_and_apply(
        connection,
        layer,
        user,
        overlay_path=_overlay_path(request),
        case_id=case_id,
        level=feedback.DRIVER_LEVEL,
        action=body.action,
        target_id=body.driver,
        reason_code=body.reason_code,
        comment=body.comment,
        kpi=adjudication.kpi,
        verdict=stored.casefile.verdict.value,
        confidence=adjudication.confidence.calibrated,
        confidence_raw=adjudication.confidence.raw,
        case_type=_case_type(layer, stored),
    )


@router.post(
    "/api/cases/{case_id}/feedback/action",
    response_model=FeedbackResult,
    status_code=status.HTTP_201_CREATED,
)
def action_feedback(
    case_id: str,
    body: DriverFeedbackRequest,
    request: Request,
    user: UserDep,
    layer: LayerDep,
    connection: ConnectionDep,
    store: StoreDep,
) -> FeedbackResult:
    """Accept or reject the recommended action.

    Moves no curve. Accepting an action opens it for tracking; the curve
    moves at D+56 when the money is counted. A loop that moved a recovery
    curve on the accept would be learning from intentions.
    """
    stored = _case(store, case_id)
    adjudication = stored.casefile.adjudication
    return _record_and_apply(
        connection,
        layer,
        user,
        overlay_path=_overlay_path(request),
        case_id=case_id,
        level=feedback.ACTION_LEVEL,
        action=body.action,
        target_id=body.driver,
        reason_code=body.reason_code,
        comment=body.comment,
        kpi=adjudication.kpi,
        verdict=stored.casefile.verdict.value,
        confidence=adjudication.confidence.calibrated,
        confidence_raw=adjudication.confidence.raw,
        case_type=_case_type(layer, stored),
    )


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


@router.post(
    "/api/cases/{case_id}/outcome",
    response_model=OutcomeResult,
    status_code=status.HTTP_201_CREATED,
)
def outcome(
    case_id: str,
    body: OutcomeRequest,
    user: UserDep,
    layer: LayerDep,
    connection: ConnectionDep,
    store: StoreDep,
) -> OutcomeResult:
    """Record what happened, at D+14 or at D+56.

    WHAT HAPPENED OUTRANKS WHAT WE WERE TOLD. This supersedes whatever
    calibration entry the reader's feedback wrote for the same case: both
    rows are kept where they were written, and only one reaches the map.
    """
    stored = store.get(case_id)
    spec = layer.learning.outcomes

    if body.horizon_days not in {h.days for h in spec.horizons}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"D+{body.horizon_days} is not a declared horizon; learning.yaml "
                f"declares {[h.days for h in spec.horizons]}"
            ),
        )

    horizon = spec.horizon(body.horizon_days)
    recorded_at = body.recorded_at or datetime.now(UTC)

    if horizon.measures == "cause_confirmed":
        if body.cause_confirmed is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"D+{horizon.days} measures whether the cause held up; "
                "`cause_confirmed` is required",
            )
        result = outcomes.record_cause_check(
            connection,
            case_id=case_id,
            cause_confirmed=body.cause_confirmed,
            action_taken=body.action_taken,
            note=body.note,
            at=recorded_at,
            layer=layer,
        )
    else:
        if body.realised_recovery_inr is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"D+{horizon.days} measures the money; "
                "`realised_recovery_inr` is required",
            )
        low, high = _expected_band(stored, body)
        result = outcomes.record_recovery_check(
            connection,
            case_id=case_id,
            realised_inr=body.realised_recovery_inr,
            expected_low_inr=low,
            expected_high_inr=high,
            horizon_weeks=body.horizon_weeks,
            action_taken=body.action_taken,
            note=body.note,
            at=recorded_at,
            layer=layer,
        )
        _record_realisation(connection, layer, stored, body, low, high)

    ledger_entry_id = None
    if stored is not None:
        adjudication = stored.casefile.adjudication
        ledger_entry_id = loop.settle(
            connection,
            outcomes.for_case(connection, case_id),
            case_type=_case_type(layer, stored),
            confidence_raw=adjudication.confidence.raw,
            confidence_published=adjudication.confidence.calibrated,
            layer=layer,
        )

    return OutcomeResult(
        case_id=case_id,
        horizon_days=result.horizon_days,
        horizon_name=result.horizon_name,
        outcome=result.outcome,
        cause_confirmed=result.cause_confirmed,
        recovered=result.recovered,
        realised_recovery_inr=result.realised_recovery_inr,
        expected_low_inr=result.expected_low_inr,
        expected_high_inr=result.expected_high_inr,
        was_correct=result.was_correct,
        recorded_at=result.recorded_at,
        ledger_entry_id=ledger_entry_id,
        superseded_feedback=ledger_entry_id is not None,
    )


# ---------------------------------------------------------------------------
# What the loop has learned
# ---------------------------------------------------------------------------


@router.get("/api/priors", response_model=PriorsResponse)
def learned_priors(
    user: UserDep,
    layer: LayerDep,
    connection: ConnectionDep,
    kpi: str | None = Query(default=None),
) -> PriorsResponse:
    """Every prior feedback has moved, declared beside effective.

    The readable form of the runtime overlay. `overlay_path` names the
    file the same figures were last written to, so a reader can go and
    diff it — which is the point of generating it at all.
    """
    from semantic_layer.overlay import OVERLAY_PATH

    learned = (
        tuple(priors.effective_priors(connection, kpi=kpi, layer=layer).values())
        if kpi
        else priors.all_learned(connection, layer)
    )
    spec = layer.learning.priors
    return PriorsResponse(
        priors=tuple(
            PriorStanding(
                kpi=item.kpi,
                hypothesis=item.hypothesis,
                declared=item.declared,
                effective=item.effective,
                shift=item.shift,
                confirmed=item.confirmed,
                rejected=item.rejected,
                capped=abs(item.shift) >= spec.max_absolute_shift,
            )
            for item in learned
        ),
        strength=spec.strength,
        max_absolute_shift=spec.max_absolute_shift,
        overlay_path=str(OVERLAY_PATH),
        overlay_written=OVERLAY_PATH.exists(),
    )


@router.get("/api/recovery-curves", response_model=list[CurveStanding])
def recovery_curves(
    user: UserDep, layer: LayerDep, connection: ConnectionDep
) -> list[CurveStanding]:
    """Every playbook curve, declared beside where realisations have moved it."""
    return [
        CurveStanding(
            curve_ref=update.curve_ref,
            declared_p25=update.declared_p25,
            declared_p75=update.declared_p75,
            declared_sample_size=update.declared_sample_size,
            observed_p25=update.observed_p25,
            observed_p75=update.observed_p75,
            observations=update.observations,
            excluded_implausible=update.excluded,
            effective_p25=update.updated_p25,
            effective_p75=update.updated_p75,
            sample_size=update.sample_size,
            moved=update.moved,
        )
        for update in curves.all_updates(connection, layer)
    ]


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def _record_and_apply(
    connection, layer, user, *, overlay_path=None, **kwargs
) -> FeedbackResult:
    """Write the event, apply the loop, and report what moved."""
    try:
        entry = feedback.record(connection, user_id=user.user_id, persona=user.persona,
                                layer=layer, **kwargs)
    except feedback.FeedbackError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    try:
        effect = loop.apply(connection, entry, layer=layer, overlay_path=overlay_path)
    except loop.LoopError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return FeedbackResult(
        feedback_id=entry.feedback_id,
        case_id=entry.case_id,
        level=entry.level,
        action=entry.action,
        target_id=entry.target_id,
        recorded_at=entry.occurred_at,
        moved_calibration=effect.moved_calibration,
        ledger_entry_id=effect.ledger_entry_id,
        moved_prior=effect.moved_prior,
        prior_kpi=effect.prior_kpi,
        prior_hypothesis=effect.prior_hypothesis,
        prior_before=effect.prior_before,
        prior_after=effect.prior_after,
        prior_shift=effect.prior_shift,
        overlay_path=effect.overlay_path,
        notes=tuple(effect.notes),
    )


def _overlay_path(request: Request):
    """Where the runtime overlay is written for this app.

    Configurable so a test can point it at a temporary directory. A suite
    that wrote a generated file into the working tree would leave the
    repository dirty after `make test`, and the file would then look like
    a change somebody made.
    """
    return getattr(request.app.state, "overlay_path", None)


def _case(store, case_id: str):
    stored = store.get(case_id)
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"no case {case_id!r} in this process. Feedback is recorded against "
                "the case as published, and that object is held by the process that "
                "ran it."
            ),
        )
    return stored


def _case_type(layer, stored) -> str | None:
    """The calibration ledger's key for this case: its leading driver's type."""
    supported = [
        item
        for item in stored.casefile.adjudication.hypotheses
        if item.status != "eliminated"
    ]
    if not supported:
        return None
    leading = max(supported, key=lambda item: item.attributed_share or 0.0)
    template = layer.causal_graph.hypotheses.get(leading.hypothesis_id)
    return template.case_type if template is not None else None


def _expected_band(stored, body: OutcomeRequest) -> tuple[float, float]:
    """The band as PUBLISHED, from the case, or from the caller.

    Read off the recommendation rather than recomputed. A curve that has
    since moved would make every past case look better or worse than it
    was actually called at the time.
    """
    if body.expected_low_inr is not None and body.expected_high_inr is not None:
        return body.expected_low_inr, body.expected_high_inr
    if stored is None or stored.casefile.recommendation is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "this case published no recovery band, so there is nothing to "
                "measure the realised recovery against. Supply expected_low_inr "
                "and expected_high_inr explicitly."
            ),
        )
    recommendation = stored.casefile.recommendation
    return (
        float(recommendation.expected_recovery_low.value or 0.0),
        float(recommendation.expected_recovery_high.value or 0.0),
    )


def _record_realisation(connection, layer, stored, body, low, high) -> None:
    """Open a realisation, when the case published an action to realise."""
    if stored is None or stored.casefile.recommendation is None:
        return
    recommendation = stored.casefile.recommendation
    playbook = recommendation.playbook_ref
    curve_ref = _curve_for(layer, playbook)
    if curve_ref is None:
        return
    attributable = body.attributable_inr
    if attributable is None:
        return
    try:
        curves.record_realisation(
            connection,
            case_id=stored.case_id,
            playbook=playbook,
            curve_ref=curve_ref,
            attributable_inr=attributable,
            expected_low_inr=low,
            expected_high_inr=high,
            realised_inr=body.realised_recovery_inr or 0.0,
            horizon_weeks=body.horizon_weeks or 0,
            layer=layer,
        )
    except curves.CurveError:
        # A realisation that cannot be expressed as a share of attributed
        # money says nothing about the playbook. Recorded nowhere rather
        # than recorded wrongly.
        return


def _curve_for(layer, playbook_ref: str) -> str | None:
    name = playbook_ref.rsplit("/", 1)[-1].removesuffix(".yaml")
    playbook = layer.playbooks.get(name)
    return playbook.recovery_curve_ref if playbook is not None else None


__all__ = ["router"]
