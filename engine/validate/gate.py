"""Gate 1: run the five checks and decide whether a case opens.

The gate runs ALL FIVE checks, always, even after one has already failed.
A gate that stops at the first failure can only ever tell you one thing
about a movement, and the second thing is often the one that matters — a
period can be both mid-restatement and carried by a single order.

The headline `outcome_code` is the exit of the failing check with the
lowest declared order, so the priority is visible in
`semantic_layer/validate.yaml` rather than implied by the order the
functions happen to appear in this file.

Nothing here decides materiality, history or authorisation: those are
Gates 2 to 5 and they are not built yet. Gate 1 answers one question — is
this movement real enough to be worth explaining — and hands the case on
or kills it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import duckdb

from engine.contracts import CheckResult, Evidence, GateResult
from engine.db import warehouse_clock
from engine.validate.checks import (
    CHECKS,
    ValidationContext,
    ValidationRequest,
    build_context,
)
from security.policy import User
from semantic_layer.schema import SemanticLayer, get_semantic_layer


@dataclass(frozen=True)
class ValidationResult:
    """What Gate 1 decided, and everything it looked at to decide it."""

    request: ValidationRequest
    gate: GateResult
    checks: tuple[CheckResult, ...]
    evidence: tuple[Evidence, ...]
    as_of: datetime

    @property
    def case_opened(self) -> bool:
        """A case opens only when all five checks come back clean."""
        return self.gate.passed

    @property
    def outcome_code(self) -> str | None:
        return self.gate.outcome_code

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def check(self, check_id: str) -> CheckResult:
        for result in self.checks:
            if result.check_id == check_id:
                return result
        raise KeyError(f"Gate 1 ran no check called {check_id!r}")

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.evidence}


def validate(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    request: ValidationRequest,
    *,
    layer: SemanticLayer | None = None,
    clock: datetime | None = None,
) -> ValidationResult:
    """Run Gate 1 for `request`, as `user`.

    Every read is governed: the checks go through
    `engine/db.py::execute_governed` for anything holding a measure and
    `execute_metadata` for the two pipeline tables that hold none. A
    persona who cannot see the scope cannot validate it either, which is
    the correct behaviour and not an obstacle to work around.
    """
    layer = layer or get_semantic_layer()
    context = build_context(connection, user, request, layer=layer, clock=clock)
    return run_checks(context)


def run_checks(context: ValidationContext) -> ValidationResult:
    """Run all five against an already-built context."""
    spec = context.layer.validation

    results: list[CheckResult] = []
    evidence: list[Evidence] = []
    for check_id, run in CHECKS:
        result, produced = run(context)
        if result.check_id != check_id:  # pragma: no cover - wiring guard
            raise RuntimeError(
                f"check {run.__name__} reported id {result.check_id!r}, expected {check_id!r}"
            )
        results.append(result)
        evidence.extend(produced)

    ordered = tuple(sorted(results, key=lambda result: result.order))
    failures = [result for result in ordered if not result.passed]
    passed = not failures

    gate = GateResult(
        gate_id=spec.gate_id,
        name=spec.name,
        passed=passed,
        outcome_code=None if passed else failures[0].outcome,
        detail=_detail(context, ordered, failures),
        checks=ordered,
        evidence_ids=tuple(item.evidence_id for item in evidence),
    )
    return ValidationResult(
        request=context.request,
        gate=gate,
        checks=ordered,
        evidence=tuple(evidence),
        as_of=context.clock,
    )


def _detail(
    context: ValidationContext,
    results: tuple[CheckResult, ...],
    failures: list[CheckResult],
) -> str:
    """One line for the gate chip.

    On a kill it leads with the exit code and the reason, because that is
    what the analyst needs. On a pass it says how many checks were run, so
    "it passed" is a statement about five things rather than a shrug.
    """
    request = context.request
    subject = f"{request.kpi} / {request.scope} / {request.period} ({request.grain})"
    if not failures:
        return f"{subject}: all {len(results)} checks clean."

    lead = failures[0]
    others = failures[1:]
    text = f"{subject}: {lead.outcome} — {lead.detail}"
    if others:
        text += (
            " Also: "
            + "; ".join(f"{other.outcome} ({other.check_id})" for other in others)
            + "."
        )
    return text


def clock_for(connection: duckdb.DuckDBPyConnection) -> datetime:
    """The warehouse's `now`. Re-exported so a caller need not import db."""
    return warehouse_clock(connection)


__all__ = ["ValidationResult", "clock_for", "run_checks", "validate"]
