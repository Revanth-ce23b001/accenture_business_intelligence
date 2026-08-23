"""VALIDATE — the first stage, and Gate 1.

CLAUDE.md §Architecture:

    Data -> VALIDATE -> QUALIFY -> GATHER -> ADJUDICATE -> VERDICT

Its job is to stop a case being opened. Five checks run against the
movement before anything else looks at it, and each returns a NAMED EXIT
rather than a boolean — DATA_INCIDENT, DEFINITION_CHANGE, ONE_OFF,
PENDING_RESTATEMENT — because "validation failed" tells an analyst nothing
and "25 of 140 store feeds did not load" tells them everything.

    from engine.validate import ValidationRequest, validate

    result = validate(connection, user, ValidationRequest(
        kpi="net_revenue", scope="West", grain="monthly",
        period="2025-11", comparison_period="2025-10",
    ))
    result.case_opened      # True: all five clean, on to QUALIFY
    result.outcome_code     # None, or the exit that killed it
    result.evidence         # every figure the checks produced
"""

from engine.validate.checks import (
    ValidationContext,
    ValidationError,
    ValidationRequest,
    Window,
    baseline_window,
    build_context,
    check_definition_drift,
    check_restatement_pending,
    check_row_count_delta,
    check_single_transaction_dominance,
    check_source_freshness,
    definition_hash,
    period_window,
)
from engine.validate.gate import ValidationResult, run_checks, validate

__all__ = [
    "ValidationContext",
    "ValidationError",
    "ValidationRequest",
    "ValidationResult",
    "Window",
    "baseline_window",
    "build_context",
    "check_definition_drift",
    "check_restatement_pending",
    "check_row_count_delta",
    "check_single_transaction_dominance",
    "check_source_freshness",
    "definition_hash",
    "period_window",
    "run_checks",
    "validate",
]
