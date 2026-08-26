"""What a reader said, recorded at the level they said it.

THREE LEVELS, BECAUSE A READER IS NOT A THUMB. Somebody can accept a
verdict and reject one driver inside it; somebody can agree with the
whole case and refuse the action it recommends. A system that records
only "accept" or "reject" flattens all three into one bit and then has to
guess which of the three things to update.

    verdict  the four options -> the isotonic calibration map
    driver   confirm / reject -> the causal graph priors
    action   accept / reject  -> the playbook recovery curves

WHAT IS STAMPED ON THE ROW, AND WHY. The verdict and both confidence
figures AS PUBLISHED. A case can be re-run tomorrow and reach a different
answer; what the reader was looking at when they clicked cannot be
reconstructed afterwards, and the calibration entry this row produces is
a claim about that figure and no other.

THE VOCABULARY IS CLOSED. Every action and every reason code comes from
`semantic_layer/learning.yaml`. An action this module has never heard of
is refused rather than stored: a feedback table containing a verb nobody
implemented is a table that cannot be aggregated.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb

from engine.learn.rows import fetch_dicts, moment, number, text
from semantic_layer.schema import SemanticLayer, get_semantic_layer

#: The three levels a reader can speak at.
VERDICT_LEVEL = "verdict"
DRIVER_LEVEL = "driver"
ACTION_LEVEL = "action"
LEVELS = (VERDICT_LEVEL, DRIVER_LEVEL, ACTION_LEVEL)

FEEDBACK_INSERT = (
    "INSERT INTO feedback_event ("
    "feedback_id, case_id, user_id, persona, occurred_at, action, comment, "
    "target_kind, target_id, reason_code, kpi, verdict_at_feedback, "
    "confidence_at_feedback, confidence_raw_at_feedback, case_type) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

FEEDBACK_FOR_CASE = (
    "SELECT feedback_id, case_id, user_id, persona, occurred_at, action, comment, "
    "target_kind, target_id, reason_code, kpi, verdict_at_feedback, "
    "confidence_at_feedback, confidence_raw_at_feedback, case_type "
    "FROM feedback_event WHERE case_id = ? ORDER BY occurred_at"
)


class FeedbackError(ValueError):
    """The feedback cannot be recorded as given."""


@dataclass(frozen=True)
class Feedback:
    """One thing a reader said about one part of one case."""

    case_id: str
    user_id: str
    persona: str
    level: str
    action: str
    occurred_at: datetime
    feedback_id: str
    #: The driver tag or the playbook ref. None at verdict level.
    target_id: str | None = None
    reason_code: str | None = None
    comment: str | None = None
    #: The case as published, stamped at the moment of the click.
    kpi: str | None = None
    verdict: str | None = None
    confidence: float | None = None
    confidence_raw: float | None = None
    case_type: str | None = None

    def as_row(self) -> tuple:
        from engine.db import as_stored_timestamp

        return (
            self.feedback_id,
            self.case_id,
            self.user_id,
            self.persona,
            as_stored_timestamp(self.occurred_at),
            self.action,
            self.comment,
            self.level,
            self.target_id,
            self.reason_code,
            self.kpi,
            self.verdict,
            self.confidence,
            self.confidence_raw,
            self.case_type,
        )


def validate(
    level: str,
    action: str,
    *,
    target_id: str | None,
    reason_code: str | None,
    layer: SemanticLayer | None = None,
) -> None:
    """Refuse anything the vocabulary does not contain, before it is stored.

    A rejection with no reason is refused here rather than stored with a
    null: "Reject with reason" is one of the four options CLAUDE.md names,
    and a rejection nobody had to explain is a different, weaker thing.
    """
    layer = layer or get_semantic_layer()
    spec = layer.learning.feedback

    if level not in LEVELS:
        raise FeedbackError(f"{level!r} is not a feedback level; expected {list(LEVELS)}")

    if level == VERDICT_LEVEL:
        declared = spec.verdict_actions
        if action not in declared:
            raise FeedbackError(
                f"{action!r} is not a verdict action; learning.yaml declares "
                f"{sorted(declared)}"
            )
        if declared[action].requires_reason and not reason_code:
            raise FeedbackError(
                f"{declared[action].label!r} requires a reason code; "
                f"learning.yaml declares {sorted(spec.reason_codes)}"
            )
        if target_id is not None:
            raise FeedbackError(
                "verdict feedback is about the whole case and takes no target"
            )
    elif level == DRIVER_LEVEL:
        if action not in spec.driver_actions:
            raise FeedbackError(
                f"{action!r} is not a driver action; learning.yaml declares "
                f"{sorted(spec.driver_actions)}"
            )
        if not target_id:
            raise FeedbackError("driver feedback names the driver it is about")
    else:
        if action not in spec.action_actions:
            raise FeedbackError(
                f"{action!r} is not an action-level action; learning.yaml declares "
                f"{sorted(spec.action_actions)}"
            )
        if not target_id:
            raise FeedbackError("action feedback names the playbook it is about")

    if reason_code is not None and reason_code not in spec.reason_codes:
        raise FeedbackError(
            f"{reason_code!r} is not a declared reason code; learning.yaml declares "
            f"{sorted(spec.reason_codes)}. Free text belongs in the comment, which "
            "travels but is not counted."
        )


def record(
    connection: duckdb.DuckDBPyConnection,
    *,
    case_id: str,
    user_id: str,
    persona: str,
    level: str,
    action: str,
    target_id: str | None = None,
    reason_code: str | None = None,
    comment: str | None = None,
    kpi: str | None = None,
    verdict: str | None = None,
    confidence: float | None = None,
    confidence_raw: float | None = None,
    case_type: str | None = None,
    occurred_at: datetime | None = None,
    layer: SemanticLayer | None = None,
) -> Feedback:
    """Validate and write one feedback event. Returns what was stored."""
    layer = layer or get_semantic_layer()
    validate(level, action, target_id=target_id, reason_code=reason_code, layer=layer)

    entry = Feedback(
        feedback_id=uuid.uuid4().hex,
        case_id=case_id,
        user_id=user_id,
        persona=persona,
        level=level,
        action=action,
        occurred_at=occurred_at or datetime.now(UTC),
        target_id=target_id,
        reason_code=reason_code,
        comment=comment,
        kpi=kpi,
        verdict=verdict,
        confidence=confidence,
        confidence_raw=confidence_raw,
        case_type=case_type,
    )
    connection.execute(FEEDBACK_INSERT, list(entry.as_row()))
    return entry


def for_case(
    connection: duckdb.DuckDBPyConnection, case_id: str
) -> tuple[Feedback, ...]:
    """Everything said about one case, oldest first."""
    rows = fetch_dicts(connection, FEEDBACK_FOR_CASE, [case_id])
    return tuple(
        Feedback(
            feedback_id=str(row["feedback_id"]),
            case_id=str(row["case_id"]),
            user_id=str(row["user_id"]),
            persona=str(row["persona"]),
            occurred_at=moment(row, "occurred_at"),
            action=str(row["action"]),
            comment=text(row, "comment"),
            # A row written before P16 has no level; it was verdict-level,
            # because that was the only level there was.
            level=text(row, "target_kind") or VERDICT_LEVEL,
            target_id=text(row, "target_id"),
            reason_code=text(row, "reason_code"),
            kpi=text(row, "kpi"),
            verdict=text(row, "verdict_at_feedback"),
            confidence=number(row, "confidence_at_feedback"),
            confidence_raw=number(row, "confidence_raw_at_feedback"),
            case_type=text(row, "case_type"),
        )
        for row in rows
    )


__all__ = [
    "ACTION_LEVEL",
    "DRIVER_LEVEL",
    "FEEDBACK_INSERT",
    "LEVELS",
    "VERDICT_LEVEL",
    "Feedback",
    "FeedbackError",
    "for_case",
    "record",
    "validate",
]
