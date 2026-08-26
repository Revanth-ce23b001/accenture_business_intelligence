"""What actually happened, at D+14 and at D+56.

TWO HORIZONS BECAUSE THERE ARE TWO QUESTIONS, and they have different
answers often enough that collapsing them would lose the interesting
cases.

    D+14  cause_check      did the driver we blamed turn out to be it?
    D+56  recovery_check   did the money come back, against the band?

A cause can be right and the recovery still fail — the action was taken
late, or the demand had already gone elsewhere. Recording one number for
both would make that case indistinguishable from a case where we blamed
the wrong thing, and those two demand opposite responses: one is a
playbook problem, the other is an adjudication problem.

WHAT SUPERSEDES WHAT. Verdict feedback writes a calibration entry
immediately — that is what the reader told us. The D+56 outcome writes
another for the same case, and it wins, because what happened outranks
what we were told. Both rows stay in `case_outcome` and `feedback_event`;
only one reaches the map. `learning.yaml -> calibration` says which.

SILENCE IS NOT SUCCESS. A case with no outcome past
`unresolved_after_days` is reported as unresolved and calibrates nothing.
"Nobody came back to us" is not evidence that we were right, and a loop
that treated it as evidence would learn fastest from the cases nobody
cared enough to check.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import duckdb

from engine.learn.rows import fetch_dicts, flag, integer, moment, number, text
from semantic_layer.schema import SemanticLayer, get_semantic_layer

OUTCOME_UPSERT = """
INSERT INTO case_outcome (
    case_id, horizon_days, horizon_name, recorded_at, action_taken, outcome,
    cause_confirmed, recovered, realised_recovery_inr, expected_low_inr,
    expected_high_inr, horizon_weeks, was_correct, note
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (case_id, horizon_days) DO UPDATE SET
    recorded_at           = excluded.recorded_at,
    action_taken          = excluded.action_taken,
    outcome               = excluded.outcome,
    cause_confirmed       = excluded.cause_confirmed,
    recovered             = excluded.recovered,
    realised_recovery_inr = excluded.realised_recovery_inr,
    expected_low_inr      = excluded.expected_low_inr,
    expected_high_inr     = excluded.expected_high_inr,
    horizon_weeks         = excluded.horizon_weeks,
    was_correct           = excluded.was_correct,
    note                  = excluded.note
"""

OUTCOMES_FOR_CASE = (
    "SELECT case_id, horizon_days, horizon_name, recorded_at, action_taken, outcome, "
    "cause_confirmed, recovered, realised_recovery_inr, expected_low_inr, "
    "expected_high_inr, horizon_weeks, was_correct, note "
    "FROM case_outcome WHERE case_id = ? ORDER BY horizon_days"
)

#: Outcome vocabulary. Closed, so the column can be grouped on.
CONFIRMED = "cause_confirmed"
DISCONFIRMED = "cause_disconfirmed"
RECOVERED = "recovered"
NOT_RECOVERED = "not_recovered"
UNRESOLVED = "unresolved"

OUTCOMES = (CONFIRMED, DISCONFIRMED, RECOVERED, NOT_RECOVERED, UNRESOLVED)


class OutcomeError(ValueError):
    """The outcome cannot be recorded as given."""


@dataclass(frozen=True)
class Outcome:
    """One case, at one horizon."""

    case_id: str
    horizon_days: int
    horizon_name: str
    recorded_at: datetime
    outcome: str
    action_taken: str | None = None
    cause_confirmed: bool | None = None
    recovered: bool | None = None
    realised_recovery_inr: float | None = None
    expected_low_inr: float | None = None
    expected_high_inr: float | None = None
    horizon_weeks: int | None = None
    was_correct: bool | None = None
    note: str | None = None

    @property
    def realised_share(self) -> float | None:
        """Realised against the LOW end of the published band."""
        if self.realised_recovery_inr is None or not self.expected_low_inr:
            return None
        return self.realised_recovery_inr / self.expected_low_inr

    def as_row(self) -> tuple:
        from engine.db import as_stored_timestamp

        return (
            self.case_id,
            self.horizon_days,
            self.horizon_name,
            as_stored_timestamp(self.recorded_at),
            self.action_taken,
            self.outcome,
            self.cause_confirmed,
            self.recovered,
            self.realised_recovery_inr,
            self.expected_low_inr,
            self.expected_high_inr,
            self.horizon_weeks,
            self.was_correct,
            self.note,
        )


def due(
    opened_at: datetime,
    *,
    now: datetime | None = None,
    layer: SemanticLayer | None = None,
) -> tuple[int, ...]:
    """Which horizons have come round for a case opened at `opened_at`.

    The check that turns "we should follow up" into a queue. A case opened
    twenty days ago is due its D+14 and not yet its D+56.
    """
    layer = layer or get_semantic_layer()
    moment = now or datetime.now(UTC)
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    elapsed = moment - opened_at
    return tuple(
        horizon.days
        for horizon in layer.learning.outcomes.horizons
        if elapsed >= timedelta(days=horizon.days)
    )


def record_cause_check(
    connection: duckdb.DuckDBPyConnection,
    *,
    case_id: str,
    cause_confirmed: bool,
    note: str | None = None,
    action_taken: str | None = None,
    at: datetime | None = None,
    layer: SemanticLayer | None = None,
) -> Outcome:
    """D+14. Did the driver hold up?

    `was_correct` is set here too: a case whose cause was wrong was wrong,
    whatever the money did afterwards. D+56 may overwrite it when the
    recovery says something different about the same case.
    """
    layer = layer or get_semantic_layer()
    horizon = _horizon_measuring(layer, "cause_confirmed")
    outcome = Outcome(
        case_id=case_id,
        horizon_days=horizon.days,
        horizon_name=horizon.name,
        recorded_at=at or datetime.now(UTC),
        outcome=CONFIRMED if cause_confirmed else DISCONFIRMED,
        action_taken=action_taken,
        cause_confirmed=cause_confirmed,
        was_correct=cause_confirmed,
        note=note,
    )
    connection.execute(OUTCOME_UPSERT, list(outcome.as_row()))
    return outcome


def record_recovery_check(
    connection: duckdb.DuckDBPyConnection,
    *,
    case_id: str,
    realised_inr: float,
    expected_low_inr: float,
    expected_high_inr: float,
    horizon_weeks: int | None = None,
    action_taken: str | None = None,
    note: str | None = None,
    at: datetime | None = None,
    layer: SemanticLayer | None = None,
) -> Outcome:
    """D+56. Did the money come back?

    THE BAR IS THE LOW END OF THE BAND, ON PURPOSE. A case that recovered
    the p25 estimate recovered; one that missed it did not. Missing the
    optimistic end of a range that was published as a range is not a
    failed recovery, and scoring it as one would teach the curves to
    quote narrower bands than the evidence supports.
    """
    layer = layer or get_semantic_layer()
    horizon = _horizon_measuring(layer, "recovered")
    bar = expected_low_inr * layer.learning.outcomes.recovered_at_share_of_low
    recovered = realised_inr >= bar

    outcome = Outcome(
        case_id=case_id,
        horizon_days=horizon.days,
        horizon_name=horizon.name,
        recorded_at=at or datetime.now(UTC),
        outcome=RECOVERED if recovered else NOT_RECOVERED,
        action_taken=action_taken,
        recovered=recovered,
        realised_recovery_inr=realised_inr,
        expected_low_inr=expected_low_inr,
        expected_high_inr=expected_high_inr,
        horizon_weeks=horizon_weeks,
        was_correct=recovered,
        note=note,
    )
    connection.execute(OUTCOME_UPSERT, list(outcome.as_row()))
    return outcome


def record_unresolved(
    connection: duckdb.DuckDBPyConnection,
    *,
    case_id: str,
    horizon_days: int,
    note: str | None = None,
    at: datetime | None = None,
    layer: SemanticLayer | None = None,
) -> Outcome:
    """Nobody came back. Recorded as unresolved, calibrating nothing."""
    layer = layer or get_semantic_layer()
    horizon = layer.learning.outcomes.horizon(horizon_days)
    outcome = Outcome(
        case_id=case_id,
        horizon_days=horizon.days,
        horizon_name=horizon.name,
        recorded_at=at or datetime.now(UTC),
        outcome=UNRESOLVED,
        was_correct=None,
        note=note or "no outcome reported within the follow-up window",
    )
    connection.execute(OUTCOME_UPSERT, list(outcome.as_row()))
    return outcome


def for_case(connection: duckdb.DuckDBPyConnection, case_id: str) -> tuple[Outcome, ...]:
    """Every horizon recorded for one case, earliest first."""
    return tuple(
        Outcome(
            case_id=str(row["case_id"]),
            horizon_days=int(row["horizon_days"]),
            horizon_name=str(row["horizon_name"]),
            recorded_at=moment(row, "recorded_at"),
            action_taken=text(row, "action_taken"),
            outcome=str(row["outcome"]),
            cause_confirmed=flag(row, "cause_confirmed"),
            recovered=flag(row, "recovered"),
            realised_recovery_inr=number(row, "realised_recovery_inr"),
            expected_low_inr=number(row, "expected_low_inr"),
            expected_high_inr=number(row, "expected_high_inr"),
            horizon_weeks=integer(row, "horizon_weeks"),
            was_correct=flag(row, "was_correct"),
            note=text(row, "note"),
        )
        for row in fetch_dicts(connection, OUTCOMES_FOR_CASE, [case_id])
    )


def settled(outcomes: tuple[Outcome, ...]) -> Outcome | None:
    """The outcome that calibrates, or None.

    The LATEST horizon with a judgement. D+56 outranks D+14 because what
    the money did is the later and better-informed answer about the same
    case; an unresolved horizon carries no judgement and is skipped.
    """
    judged = [item for item in outcomes if item.was_correct is not None]
    return max(judged, key=lambda item: item.horizon_days) if judged else None


def _horizon_measuring(layer: SemanticLayer, measures: str):
    for horizon in layer.learning.outcomes.horizons:
        if horizon.measures == measures:
            return horizon
    raise OutcomeError(
        f"no outcome horizon measures {measures!r}; learning.yaml declares "
        f"{[h.measures for h in layer.learning.outcomes.horizons]}"
    )


def new_outcome_id() -> str:
    return uuid.uuid4().hex


__all__ = [
    "CONFIRMED",
    "DISCONFIRMED",
    "NOT_RECOVERED",
    "OUTCOMES",
    "RECOVERED",
    "UNRESOLVED",
    "Outcome",
    "OutcomeError",
    "due",
    "for_case",
    "new_outcome_id",
    "record_cause_check",
    "record_recovery_check",
    "record_unresolved",
    "settled",
]
