"""The loop. One feedback event in, up to three things moved.

    apply(connection, feedback) -> LoopEffect

CLAUDE.md's pipeline ends "... VERDICT -> human decides -> outcome logged
-> recalibrate". This module is the arrow from the human back to the
engine, and it is deliberately the only one: nothing else in the codebase
writes `calibration_ledger`, `hypothesis_prior` or a realisation. One
door, for the same reason `execute_governed` is one door — so that "what
can move the engine's judgement" has a single answer.

WHICH LEVEL MOVES WHAT:

    verdict feedback  -> a calibration_ledger entry -> the isotonic map
    driver feedback   -> hypothesis_prior counts    -> the causal graph
    action feedback   -> nothing, yet               -> a realisation opens
                                                       at D+56, not now

The third is the one that surprises people. Accepting an action does not
move a recovery curve, because nothing has been recovered yet — it opens
a case for tracking and the curve moves when the money is counted at
D+56. A loop that moved the curve on the ACCEPT would be learning from
intentions.

WHAT DOES NOT HAPPEN HERE. No refit is triggered, no cache is warmed, no
background job is queued. `fit_calibration` reads the ledger on every
call, so the next read of `/api/calibration` sees the new row; the
effective priors are computed from the counts on every screening. Nothing
is precomputed, so nothing can be stale.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import duckdb

from engine.learn import curves, priors
from engine.learn.feedback import (
    ACTION_LEVEL,
    DRIVER_LEVEL,
    VERDICT_LEVEL,
    Feedback,
    FeedbackError,
)
from engine.learn.outcomes import Outcome, settled
from semantic_layer.schema import SemanticLayer, get_semantic_layer

LEDGER_INSERT = (
    "INSERT INTO calibration_ledger ("
    "entry_id, case_id, case_type, closed_at, confidence_raw, "
    "confidence_published, abstained, was_correct, notes) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

LEDGER_DELETE_FOR_CASE = "DELETE FROM calibration_ledger WHERE case_id = ?"

LEDGER_FOR_CASE = (
    "SELECT entry_id, case_type, confidence_raw, confidence_published, abstained, "
    "was_correct, notes FROM calibration_ledger WHERE case_id = ?"
)

#: Where a ledger entry came from. Written into `notes`, so a reader of
#: the ledger can tell a reported outcome from a reader's opinion without
#: joining anything.
FROM_FEEDBACK = "feedback"
FROM_OUTCOME = "outcome"


class LoopError(RuntimeError):
    """The feedback cannot be applied."""


@dataclass
class LoopEffect:
    """What one feedback event actually moved. Empty is a valid answer."""

    feedback_id: str
    case_id: str
    level: str
    action: str

    #: The calibration entry written, if any.
    ledger_entry_id: str | None = None
    superseded_entry_ids: tuple[str, ...] = ()
    #: The prior, before and after.
    prior_before: float | None = None
    prior_after: float | None = None
    prior_hypothesis: str | None = None
    prior_kpi: str | None = None
    #: Where the overlay was written, or None when it could not be.
    overlay_path: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def moved_calibration(self) -> bool:
        return self.ledger_entry_id is not None

    @property
    def moved_prior(self) -> bool:
        return (
            self.prior_before is not None
            and self.prior_after is not None
            and self.prior_after != self.prior_before
        )

    @property
    def prior_shift(self) -> float | None:
        if self.prior_before is None or self.prior_after is None:
            return None
        return self.prior_after - self.prior_before

    def render(self) -> str:
        parts = [f"{self.level}/{self.action} on {self.case_id}"]
        if self.moved_calibration:
            parts.append("wrote a calibration entry")
        if self.prior_hypothesis:
            parts.append(
                f"{self.prior_kpi}/{self.prior_hypothesis} "
                f"{self.prior_before:.4f} -> {self.prior_after:.4f}"
            )
        parts.extend(self.notes)
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# The one door
# ---------------------------------------------------------------------------


def apply(
    connection: duckdb.DuckDBPyConnection,
    entry: Feedback,
    *,
    layer: SemanticLayer | None = None,
    write_overlay: bool = True,
    overlay_path=None,
) -> LoopEffect:
    """Apply one feedback event. Returns exactly what it moved."""
    layer = layer or get_semantic_layer()
    effect = LoopEffect(
        feedback_id=entry.feedback_id,
        case_id=entry.case_id,
        level=entry.level,
        action=entry.action,
    )

    if entry.level == VERDICT_LEVEL:
        _apply_verdict(connection, entry, effect, layer)
    elif entry.level == DRIVER_LEVEL:
        _apply_driver(
            connection,
            entry,
            effect,
            layer,
            write_overlay=write_overlay,
            overlay_path=overlay_path,
        )
    elif entry.level == ACTION_LEVEL:
        _apply_action(connection, entry, effect, layer)
    else:  # pragma: no cover - `record` validates before this is reached
        raise LoopError(f"{entry.level!r} is not a feedback level")

    return effect


def _apply_verdict(
    connection, entry: Feedback, effect: LoopEffect, layer: SemanticLayer
) -> None:
    """Verdict feedback writes a calibration entry, or explains why not."""
    spec = layer.learning
    action = spec.feedback.verdict_actions[entry.action]

    if not spec.calibration.write_on_feedback:
        effect.notes.append("calibration.write_on_feedback is off")
        return
    if action.scores_as_correct is None:
        effect.notes.append(
            f"{action.label!r} is a request for work, not a judgement on "
            "correctness — no calibration entry written"
        )
        return
    if entry.confidence_raw is None or entry.confidence is None:
        effect.notes.append(
            "no published confidence was stamped on the feedback, so there is "
            "nothing for the map to be calibrated against"
        )
        return
    if not entry.case_type:
        effect.notes.append("no case type stamped; the entry would not be attributable")
        return

    effect.ledger_entry_id = write_ledger_entry(
        connection,
        case_id=entry.case_id,
        case_type=entry.case_type,
        confidence_raw=entry.confidence_raw,
        confidence_published=entry.confidence,
        was_correct=action.scores_as_correct,
        closed_at=entry.occurred_at,
        source=FROM_FEEDBACK,
        detail=f"{entry.action} by {entry.persona}"
        + (f" ({entry.reason_code})" if entry.reason_code else ""),
    )


def _apply_driver(
    connection,
    entry: Feedback,
    effect: LoopEffect,
    layer: SemanticLayer,
    *,
    write_overlay: bool,
    overlay_path=None,
) -> None:
    """Driver feedback moves the prior for that hypothesis on that KPI."""
    spec = layer.learning.feedback.driver_actions[entry.action]
    if not entry.kpi:
        raise LoopError(
            f"driver feedback on {entry.target_id!r} carries no KPI; a prior is "
            "learned per hypothesis PER KPI and there is nothing to attach it to"
        )
    if entry.target_id not in layer.causal_graph.hypotheses:
        raise LoopError(
            f"{entry.target_id!r} is not a hypothesis in the causal graph; a prior "
            "cannot be learned for something the graph does not declare"
        )

    before = priors.effective_prior(
        connection, kpi=entry.kpi, hypothesis=entry.target_id, layer=layer
    )
    priors.observe(
        connection,
        kpi=entry.kpi,
        hypothesis=entry.target_id,
        confirmed=spec.confirms,
        at=entry.occurred_at,
    )
    after = priors.effective_prior(
        connection, kpi=entry.kpi, hypothesis=entry.target_id, layer=layer
    )

    effect.prior_kpi = entry.kpi
    effect.prior_hypothesis = entry.target_id
    effect.prior_before = before.effective
    effect.prior_after = after.effective
    if abs(after.shift) >= layer.learning.priors.max_absolute_shift:
        effect.notes.append(
            "the shift cap is binding: further feedback will not move this prior, "
            "and the causal graph itself needs a person to look at it"
        )

    if write_overlay:
        path = priors.write_overlay(connection, path=overlay_path, layer=layer)
        effect.overlay_path = str(path) if path is not None else None
        if path is None:
            effect.notes.append(
                "the runtime overlay could not be written (read-only filesystem?); "
                "the counts are recorded and it regenerates on the next write"
            )


def _apply_action(
    connection, entry: Feedback, effect: LoopEffect, layer: SemanticLayer
) -> None:
    """Action feedback opens tracking. The curve moves at D+56, not now."""
    accepted = layer.learning.feedback.action_actions[entry.action].accepted
    effect.notes.append(
        (
            "action accepted — its recovery becomes trackable and the curve moves "
            "when the money is counted at D+56, not now"
        )
        if accepted
        else "action refused — there is no recovery to realise, so no curve moves"
    )


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def write_ledger_entry(
    connection: duckdb.DuckDBPyConnection,
    *,
    case_id: str,
    case_type: str,
    confidence_raw: float,
    confidence_published: float,
    was_correct: bool | None,
    closed_at: datetime,
    source: str,
    detail: str = "",
    abstained: bool = False,
    supersede: bool = True,
) -> str:
    """Write one calibration entry, replacing any earlier one for this case.

    ONE ENTRY PER CASE, ALWAYS. A case that a reader accepted and that
    later failed to recover would otherwise appear twice — once as a hit
    and once as a miss — and the map would be fitted on a population where
    the cases people cared about are counted twice. The `notes` column
    records which source won.
    """
    from engine.db import as_stored_timestamp

    if supersede:
        connection.execute(LEDGER_DELETE_FOR_CASE, [case_id])

    entry_id = uuid.uuid4().hex
    connection.execute(
        LEDGER_INSERT,
        [
            entry_id,
            case_id,
            case_type,
            as_stored_timestamp(closed_at),
            float(confidence_raw),
            float(confidence_published),
            abstained,
            was_correct,
            f"{source}: {detail}" if detail else source,
        ],
    )
    return entry_id


def apply_outcome(
    connection: duckdb.DuckDBPyConnection,
    outcome: Outcome,
    *,
    case_type: str,
    confidence_raw: float,
    confidence_published: float,
    layer: SemanticLayer | None = None,
) -> str | None:
    """Let a recorded outcome supersede whatever feedback said.

    WHAT HAPPENED OUTRANKS WHAT WE WERE TOLD, and this is where that is
    enforced. An unresolved outcome carries no judgement and supersedes
    nothing: silence is not evidence.
    """
    layer = layer or get_semantic_layer()
    if not layer.learning.calibration.outcome_supersedes_feedback:
        return None
    if outcome.was_correct is None:
        return None
    return write_ledger_entry(
        connection,
        case_id=outcome.case_id,
        case_type=case_type,
        confidence_raw=confidence_raw,
        confidence_published=confidence_published,
        was_correct=outcome.was_correct,
        closed_at=outcome.recorded_at,
        source=FROM_OUTCOME,
        detail=f"{outcome.horizon_name} at D+{outcome.horizon_days}: {outcome.outcome}",
    )


def ledger_for_case(connection: duckdb.DuckDBPyConnection, case_id: str) -> tuple[dict, ...]:
    """The entries currently standing for one case. Normally zero or one."""
    rows = connection.execute(LEDGER_FOR_CASE, [case_id]).fetchall()
    columns = (
        "entry_id", "case_type", "confidence_raw", "confidence_published",
        "abstained", "was_correct", "notes",
    )
    return tuple(dict(zip(columns, row, strict=True)) for row in rows)


def settle(
    connection: duckdb.DuckDBPyConnection,
    outcomes: tuple[Outcome, ...],
    *,
    case_type: str,
    confidence_raw: float,
    confidence_published: float,
    layer: SemanticLayer | None = None,
) -> str | None:
    """Apply the latest judged outcome for a case, if there is one."""
    final = settled(outcomes)
    if final is None:
        return None
    return apply_outcome(
        connection,
        final,
        case_type=case_type,
        confidence_raw=confidence_raw,
        confidence_published=confidence_published,
        layer=layer,
    )


__all__ = [
    "FROM_FEEDBACK",
    "FROM_OUTCOME",
    "LEDGER_INSERT",
    "LoopEffect",
    "LoopError",
    "apply",
    "apply_outcome",
    "ledger_for_case",
    "settle",
    "write_ledger_entry",
]
