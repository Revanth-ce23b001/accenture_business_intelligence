"""Running a case over Server-Sent Events.

    event: VALIDATE      event: QUALIFY   event: GATHER
    event: ADJUDICATE    event: VERDICT   event: NARRATE
    event: done

SIX EVENTS, AND WHY IT IS SIX. CLAUDE.md §"Architecture" locks the five
stage names as "the module names, the API event names and the UI progress
rail, in that order and with that spelling", so the first five are those
five, spelled that way. NARRATE is appended rather than substituted for
VERDICT: the narrative is the model rendering a decision the engine has
already made, and collapsing the two would hide the moment the verdict
was reached behind the moment it was worded.

THE ENGINE GENERATOR DOES NOT KNOW ABOUT SSE. `engine/verdict/pipeline.py`
yields `StageEvent`s; this module turns each into a frame and adds the
narration step afterwards. That split is what lets the same pipeline run
from a test, a script or a cron with no HTTP anywhere.

A STOPPED RUN IS A COMPLETE RUN. #2470 is killed at Gate 1 and #2471 at
Gate 3. Both emit their stages, then `done`. The stream does not fake the
stages that never ran — a rail that shows five green ticks for a case
that stopped at the first is lying about what the system did.

BLOCKING WORK, ON A THREAD. The pipeline is synchronous DuckDB and
statistics, and running it on the event loop would stall every other
request for the twelve seconds a case takes. It runs in a worker thread
and frames cross back over a queue.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Callable, Iterator

from api.schemas import CaseRunRequest, StageMessage
from engine.verdict.pipeline import (
    ADJUDICATE,
    GATHER,
    QUALIFY,
    STAGE_ORDER,
    VALIDATE,
    VERDICT,
    CaseRequest,
    StageEvent,
    run_case,
)

#: The event name for the sixth frame, and for the terminator. NARRATE is
#: not a `Stage` — it is deliberately outside the locked five.
NARRATE_EVENT = "NARRATE"
DONE_EVENT = "done"
ERROR_EVENT = "error"

#: Frames waiting to cross from the worker thread to the event loop. A
#: bound rather than an unbounded queue: a client that stops reading
#: should slow the producer, not grow the heap.
QUEUE_DEPTH = 32

#: Sentinel closing the queue.
_END = object()


@dataclass(frozen=True)
class Frame:
    """One SSE frame: an event name and a JSON body."""

    event: str
    data: dict[str, Any]

    def render(self) -> str:
        """The wire format. Two newlines terminate a frame."""
        body = json.dumps(self.data, default=_json_default, ensure_ascii=False)
        return f"event: {self.event}\ndata: {body}\n\n"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


# ---------------------------------------------------------------------------
# Stage summaries
# ---------------------------------------------------------------------------


def summarise(event: StageEvent) -> dict[str, Any]:
    """A small, stage-shaped digest for the progress rail.

    Deliberately small. The full evidence ledger and every test result go
    to `GET /api/cases/{id}`; an SSE frame is a progress update, and
    pushing seventy evidence records through five of them would make the
    rail the slowest part of the request it is reporting on.
    """
    result = event.result
    if event.stage == VALIDATE:
        return {
            "checks": len(result.checks),
            "failed": [check.check_id for check in result.failures],
            "passed": result.gate.passed,
        }
    if event.stage == QUALIFY:
        decomposition = result.decomposition
        return {
            "gates": [
                {"gate_id": gate.gate_id, "name": gate.name, "passed": gate.passed}
                for gate in result.gates
            ],
            "case_opened": result.case_opened,
            "headline_pt": _round(getattr(decomposition, "headline_pt", None)),
            "residual_pt": _round(getattr(decomposition, "residual_pt", None)),
            "region_strip": {k: _round(v) for k, v in result.region_strip.items()},
        }
    if event.stage == GATHER:
        return {
            "hypotheses": list(result.screening.tags),
            "lanes_run": list(result.lanes_run),
            "lane_failures": dict(result.lane_failures),
            "cache_hit_rate": _round(result.cache_hit_rate),
        }
    if event.stage == ADJUDICATE:
        return {
            "judged": len(result.verdicts),
            "surviving": [item.tag for item in result.surviving],
            "eliminated": {
                item.tag: item.elimination_reason for item in result.eliminated
            },
            "coverage": _round(result.coverage),
        }
    if event.stage == VERDICT and result is not None:
        verdict = result.verdict
        return {
            "verdict": verdict.value,
            "reason_code": verdict.reason_code,
            "reason_text": verdict.reason_text,
            "coverage": _round(verdict.coverage),
            "confidence": _round(verdict.confidence_calibrated),
            "triggers_fired": list(verdict.triggers_fired),
            "caps_applied": [
                cap.name for cap in result.adjudication.confidence.caps_applied
            ],
        }
    return {}


def _round(value: Any) -> Any:
    """Round for display only. The stored evidence keeps full precision."""
    return round(float(value), 4) if isinstance(value, (int, float)) else value


def to_message(event: StageEvent) -> StageMessage:
    return StageMessage(
        stage=event.stage,
        ordinal=event.ordinal,
        status="stopped" if event.outcome_code else "ok",
        duration_ms=round(event.duration_ms, 3),
        case_id=event.run.case_id,
        outcome_code=event.outcome_code,
        detail=_detail(event),
        summary=summarise(event),
    )


def _detail(event: StageEvent) -> str | None:
    """The one sentence a reader needs about this stage."""
    if event.outcome_code and event.stage == VALIDATE:
        failures = event.result.failures
        return failures[0].detail if failures else event.outcome_code
    if event.outcome_code and event.stage == QUALIFY:
        for gate in event.result.gates:
            if not gate.passed:
                return gate.detail
        restraint = event.result.restraint
        return restraint.detail if restraint is not None else event.outcome_code
    return None


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def to_case_request(body: CaseRunRequest, layer) -> CaseRequest:
    """Resolve the request body into the engine's own request.

    The four window dates are DERIVED from the period and the grain
    rather than accepted from the caller. A caller who could set the
    comparison window independently of the period could compare November
    against a fortnight in March, and the engine would dutifully report
    the difference as a movement.
    """
    from api.periods import resolve_window

    window = resolve_window(body.period, body.grain, body.comparison_period)
    return CaseRequest(
        kpi=body.kpi,
        scope=body.scope,
        grain=body.grain,
        period=body.period,
        period_start=window.period_start,
        period_end=window.period_end,
        comparison_start=window.comparison_start,
        comparison_end=window.comparison_end,
        comparison_period=window.comparison_period,
        case_id=body.case_id,
        direction=body.direction,
        open_linked_case=body.open_linked_case,
    )


def iter_frames(
    connection,
    user,
    request: CaseRequest,
    *,
    layer,
    provider=None,
    on_complete: Callable[[Any], None] | None = None,
    narrate: Callable[[Any], Frame | None] | None = None,
) -> Iterator[Frame]:
    """The whole stream, synchronously. Used directly by the tests.

    `on_complete` receives the finished `CaseRun` before the narration
    frame, so the case is in the store by the time a client that read
    `VERDICT` turns round and asks for it.
    """
    run = None
    try:
        for event in run_case(
            connection, user, request, layer=layer, provider=provider
        ):
            run = event.run
            yield Frame(event.stage, to_message(event).model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001 - every failure becomes a frame
        yield Frame(
            ERROR_EVENT,
            {
                "error": type(exc).__name__,
                "detail": str(exc),
                "case_id": getattr(run, "case_id", None),
            },
        )
        yield Frame(DONE_EVENT, _done(run, failed=True))
        return

    if on_complete is not None and run is not None:
        on_complete(run)

    if narrate is not None and run is not None and run.reached_verdict:
        frame = narrate(run)
        if frame is not None:
            yield frame

    yield Frame(DONE_EVENT, _done(run, failed=False))


def _done(run, *, failed: bool) -> dict[str, Any]:
    if run is None:
        return {"case_id": None, "status": "failed", "stages": []}
    return {
        "case_id": run.case_id,
        "status": "failed"
        if failed
        else ("stopped" if run.stopped_at else "complete"),
        "stopped_at": run.stopped_at,
        "outcome_code": run.outcome_code,
        "reached_verdict": run.reached_verdict,
        "stages": list(run.latency_ms),
        "latency_ms": {k: round(v, 3) for k, v in run.latency_ms.items()},
        "finished_at": datetime.now(UTC).isoformat(),
    }


async def stream(frames: Callable[[], Iterator[Frame]]) -> AsyncIterator[str]:
    """Run `frames()` on a worker thread and yield the wire text.

    The pipeline is twelve seconds of synchronous DuckDB and scipy. On the
    event loop that is twelve seconds during which nothing else in the
    process can be served, including the health check.
    """
    import anyio

    outbox: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)

    def produce() -> None:
        try:
            for frame in frames():
                outbox.put(frame)
        except Exception as exc:  # noqa: BLE001 - the stream must terminate
            outbox.put(Frame(ERROR_EVENT, {"error": type(exc).__name__, "detail": str(exc)}))
            outbox.put(Frame(DONE_EVENT, {"status": "failed"}))
        finally:
            outbox.put(_END)

    worker = threading.Thread(target=produce, name="casefile-run", daemon=True)
    worker.start()
    try:
        while True:
            item = await anyio.to_thread.run_sync(outbox.get)
            if item is _END:
                return
            yield item.render()
    finally:
        worker.join(timeout=0)


__all__ = [
    "DONE_EVENT",
    "ERROR_EVENT",
    "NARRATE_EVENT",
    "QUEUE_DEPTH",
    "Frame",
    "iter_frames",
    "stream",
    "summarise",
    "to_case_request",
    "to_message",
]
