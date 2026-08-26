"""Running an investigation, reading it back, and acting on it.

`POST /api/cases/run` is the only endpoint that computes anything. The
rest read what it produced.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from api.deps import ConnectionDep, LayerDep, ProviderDep, StoreDep, UserDep
from api.periods import PeriodError
from api.schemas import (
    CaseDetail,
    CaseRunRequest,
    Claim,
    FeedbackRequest,
    FeedbackResponse,
    NarrativeResponse,
)
from api.store import store_case
from api.streaming import DONE_EVENT, NARRATE_EVENT, Frame, iter_frames, stream, to_case_request
from engine.db import GovernanceError
from security.policy import PolicyError, resolve_policy

router = APIRouter(tags=["cases"])

#: SSE needs these two or a proxy will buffer the stream into one response
#: and the progress rail arrives all at once, at the end, which is the one
#: thing it exists not to do.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
SSE_MEDIA_TYPE = "text/event-stream"

FEEDBACK_INSERT = (
    "INSERT INTO feedback_event (feedback_id, case_id, user_id, persona, "
    "occurred_at, action, comment) VALUES (?, ?, ?, ?, ?, ?, ?)"
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@router.post("/api/cases/run")
def run(
    body: CaseRunRequest,
    request: Request,
    user: UserDep,
    layer: LayerDep,
    connection: ConnectionDep,
    store: StoreDep,
    provider: ProviderDep,
) -> StreamingResponse:
    """Investigate a movement, streaming one event per stage.

        VALIDATE  QUALIFY  GATHER  ADJUDICATE  VERDICT  NARRATE  done

    The first five are the locked stage names in the locked order
    (CLAUDE.md §"Architecture"). NARRATE follows because the narrative is
    the model rendering a decision the engine already reached — separate
    events so the moment the verdict landed is visible, rather than hidden
    inside the moment it was worded.

    A run that a gate stops emits the stages that ran and then `done`. It
    does not emit the stages that did not: a rail showing five green ticks
    for a case killed at Gate 1 is a lie about what the system did.

    AUTHORIZATION HAPPENS HERE, BEFORE ANYTHING IS READ. The persona is
    resolved against the KPI contract before the stream opens, so a caller
    who may not read the KPI gets a 403 with a body rather than an error
    frame inside a 200.
    """
    contract = layer.kpis.get(body.kpi)
    if contract is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no KPI contract named {body.kpi!r}; known: {sorted(layer.kpis)}",
        )
    try:
        resolve_policy(contract, user, layer.warehouse.governance)
    except PolicyError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    try:
        case_request = to_case_request(body, layer)
    except PeriodError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    persona = body.persona or user.persona

    def frames():
        return iter_frames(
            connection,
            user,
            case_request,
            layer=layer,
            provider=provider,
            on_complete=lambda run: _remember(store, run, user),
            narrate=lambda run: _narrate_frame(run, layer, provider, persona, store),
        )

    return StreamingResponse(
        stream(frames), media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS
    )


def _remember(store, run, user) -> None:
    """Put the finished case in the store, before the narration frame.

    Order matters: a client that reads `VERDICT` and immediately asks for
    the case must find it. Storing after narration would leave a window
    where the verdict has been announced and the case cannot be read.
    """
    if run.casefile is None:
        return
    store_case(
        store,
        run.casefile,
        user_id=user.user_id,
        persona=user.persona,
        latency_ms=run.latency_ms,
        decision_trace=run.decision_trace if hasattr(run, "decision_trace") else (),
    )


def _narrate_frame(run, layer, provider, persona: str, store) -> Frame | None:
    """The sixth event. A narration failure does not fail the case.

    The verdict is the engine's; the narrative is the model's rendering of
    it. A missing fixture or a refused call means the reader gets the case
    without prose, not that the investigation is discarded.
    """
    from time import perf_counter

    started = perf_counter()
    scale = layer.warehouse.units.milliseconds_per_second
    try:
        narrative = _narrate(run.casefile, persona, layer, provider, store)
    except Exception as exc:  # noqa: BLE001 - narration is not the case
        return Frame(
            NARRATE_EVENT,
            {
                "stage": NARRATE_EVENT,
                "ordinal": 6,
                "status": "failed",
                "duration_ms": round((perf_counter() - started) * scale, 3),
                "case_id": run.case_id,
                "detail": f"{type(exc).__name__}: {exc}",
                "summary": {},
            },
        )
    return Frame(
        NARRATE_EVENT,
        {
            "stage": NARRATE_EVENT,
            "ordinal": 6,
            "status": "ok",
            "duration_ms": round((perf_counter() - started) * scale, 3),
            "case_id": run.case_id,
            "detail": narrative.grounding,
            "summary": {
                "persona": narrative.persona,
                "text": narrative.text,
                "claims_checked": narrative.claims_checked,
                "claims_linked": narrative.claims_linked,
                "claims_stripped": narrative.claims_stripped,
            },
        },
    )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


@router.get("/api/cases/{case_id}", response_model=CaseDetail)
def get_case(case_id: str, user: UserDep, layer: LayerDep, store: StoreDep) -> CaseDetail:
    """The whole case.

    Everything the accept criterion names travels on the engine's own
    contracts rather than on a re-modelled subset:

        adjudication.gates                       every gate that ran
        adjudication.hypotheses[].tests          all six tests, per hypothesis
        adjudication.evidence                    the full ledger
        adjudication.confidence.components       s1 through s6
        adjudication.confidence.caps_applied     empty when none fired
        verdict.reason_text                      the string on the chip

    A persona may read a case only if the KPI's access policy would have
    let them run it. Otherwise a store manager could read a case opened by
    a regional head over stores they cannot see, which is the row filter
    defeated by a URL.
    """
    stored = store.get(case_id)
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"no case {case_id!r} in this process. Cases are held by the process "
                "that ran them; run it again with POST /api/cases/run."
            ),
        )
    _authorise(stored, user, layer)
    return CaseDetail(
        case_id=stored.case_id,
        adjudication=stored.casefile.adjudication,
        verdict=stored.casefile.verdict,
        recommendation=stored.casefile.recommendation,
        decision_trace=stored.casefile.decision_trace,
        latency_ms=stored.latency_ms,
        run_as_persona=stored.run_as_persona,
        stored_at=stored.stored_at,
        source=stored.source,
        config_ref=stored.config_ref,
        narrative_personas=tuple(sorted(layer.narrate.personas)),
    )


@router.get("/api/cases/{case_id}/narrative", response_model=NarrativeResponse)
def narrative(
    case_id: str,
    user: UserDep,
    layer: LayerDep,
    store: StoreDep,
    provider: ProviderDep,
    persona: str | None = Query(default=None),
) -> NarrativeResponse:
    """This case, written for one persona, with its grounding audit.

    The audit is not decoration. Every sentence carries the evidence ids
    it rests on, every numeric token in it was checked against the frozen
    adjudication object, and a sentence that failed either check was
    stripped. `grounding` is the line the UI prints underneath.

    Cached per case and persona: a narrative is two model calls, and
    re-rendering an unchanged case for the same reader is money for
    nothing.
    """
    stored = store.get(case_id)
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no case {case_id!r}"
        )
    _authorise(stored, user, layer)

    wanted = persona or user.persona
    if wanted not in layer.narrate.personas:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"no narrative persona {wanted!r}; narrate.yaml declares "
                f"{sorted(layer.narrate.personas)}"
            ),
        )
    try:
        return _narrate(stored.casefile, wanted, layer, provider, store)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"the narrator could not write this case: {type(exc).__name__}: {exc}. "
                "The case itself is unaffected — GET /api/cases/{id} still serves it."
            ),
        ) from exc


def _narrate(casefile, persona: str, layer, provider, store) -> NarrativeResponse:
    """Render, or return the cached rendering."""
    stored = store.get(casefile.case_id)
    if stored is not None and persona in stored.narratives:
        return stored.narratives[persona]

    from llm.narrate import narrate as write

    frozen = _frozen(casefile)
    story = write(frozen, persona, layer, provider=provider)
    response = NarrativeResponse(
        case_id=casefile.case_id,
        persona=persona,
        text=story.text,
        claims=tuple(
            Claim(sentence=item.sentence, evidence_ids=tuple(item.evidence_ids))
            for item in story.claims
        ),
        claims_checked=story.report.claims_checked,
        claims_linked=story.report.claims_linked,
        claims_stripped=story.report.claims_stripped,
        regenerated=story.regenerated,
        model=story.model,
        from_fixture=story.from_fixture,
        grounding=story.report.render(),
    )
    if stored is not None:
        stored.narratives[persona] = response
    return response


def _frozen(casefile) -> dict[str, Any]:
    """The case as the narrator receives it.

    Defined in `engine/verdict/casefile.py` and imported rather than built
    here, so the fixture recorder and this endpoint cannot disagree about
    the document — the fixture key is a hash of it.
    """
    from engine.verdict.casefile import narrative_document

    return narrative_document(casefile)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


@router.post(
    "/api/cases/{case_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def feedback(
    case_id: str,
    body: FeedbackRequest,
    user: UserDep,
    layer: LayerDep,
    connection: ConnectionDep,
    store: StoreDep,
) -> FeedbackResponse:
    """Record what a reader did about a case.

    THIS IS THE INPUT TO RECALIBRATION. CLAUDE.md's loop ends "human
    decides -> outcome logged -> recalibrate", and this is the logging
    step. A system that publishes a confidence and never learns whether it
    was right is publishing an opinion; the ledger this feeds is what
    turns 0.89 into 0.84 next quarter.

    Written to `feedback_event`, which is durable — unlike the assembled
    case object, this survives a restart, because what a person decided is
    a fact about the business and not about the process that served it.
    """
    stored = store.get(case_id)
    if stored is not None:
        _authorise(stored, user, layer)

    feedback_id = uuid.uuid4().hex
    occurred = datetime.now(UTC)
    from engine.db import as_stored_timestamp

    connection.execute(
        FEEDBACK_INSERT,
        [
            feedback_id,
            case_id,
            user.user_id,
            user.persona,
            as_stored_timestamp(occurred),
            body.action,
            body.comment,
        ],
    )
    return FeedbackResponse(
        feedback_id=feedback_id,
        case_id=case_id,
        action=body.action,
        recorded_at=occurred,
    )


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def _authorise(stored, user, layer) -> None:
    """A case is readable by a persona the KPI would have let run it."""
    contract = layer.kpis.get(stored.casefile.adjudication.kpi)
    if contract is None:  # pragma: no cover - a case cannot exist without one
        return
    try:
        resolve_policy(contract, user, layer.warehouse.governance)
    except PolicyError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"{user.persona} may not read {stored.casefile.adjudication.kpi}: {exc}"
            ),
        ) from exc


__all__ = ["SSE_HEADERS", "SSE_MEDIA_TYPE", "router"]
