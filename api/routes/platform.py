"""What the system says about itself: calibration, telemetry, the audit log.

Three endpoints that answer three questions an evaluator asks and a
dashboard cannot:

    /api/calibration   is your confidence worth anything?
    /api/telemetry     what did it cost and how long did it take?
    /api/audit         what did it read, and what left the building?

None of them is decorative. The calibration endpoint is what makes the
0.89 -> 0.84 step defensible; the telemetry endpoint replaces an asserted
"11 minutes" with a measurement; the audit endpoint answers the question
an auditor actually asks, which is not "was there a policy" but "did this
number leave".
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from api.deps import ConnectionDep, LayerDep, UserDep
from api.schemas import (
    AuditEntry,
    AuditResponse,
    CalibrationBand,
    CalibrationResponse,
    TelemetryResponse,
)
from engine.confidence.calibration import CalibrationError, fit_calibration
from security.audit import COLUMN_SEPARATOR
from security.redaction import RELEASE_PURPOSE_PREFIX

router = APIRouter(tags=["platform"])

#: `audit_log` is not on the metadata allow-list and it is not a KPI
#: table, so it is read directly. That is deliberate: the log records
#: access to business data and is not itself business data, and routing it
#: through a KPI's row predicate would mean an auditor could only see the
#: log entries for the region they happen to manage.
AUDIT_SQL = """
SELECT audit_id, occurred_at, user_id, persona, kpi, statement_hash,
       row_predicate, rows_returned, rows_filtered, columns_masked,
       rows_released_to_llm, purpose
FROM audit_log
{where}
ORDER BY occurred_at DESC
LIMIT ?
"""

AUDIT_TOTALS = """
SELECT COUNT(*)                                        AS total,
       COALESCE(SUM(rows_filtered), 0)                 AS filtered,
       COALESCE(SUM(rows_released_to_llm), 0)          AS released,
       COALESCE(SUM(CASE WHEN rows_released_to_llm > 0 THEN 1 ELSE 0 END), 0) AS releases
FROM audit_log
{where}
"""

DEFAULT_AUDIT_LIMIT = 100
MAX_AUDIT_LIMIT = 1000


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@router.get("/api/calibration", response_model=CalibrationResponse)
def calibration(
    user: UserDep, layer: LayerDep, connection: ConnectionDep
) -> CalibrationResponse:
    """The organisation's own track record, fitted from closed cases.

    THE BAND TABLE IS COMPUTED, NOT DECLARED. CLAUDE.md asserts exactly
    two things about the ledger — the 0.8876 -> 0.84 step and an expected
    calibration error under 0.05 — and says everything else is "computed
    and printed from the seeded data". So this endpoint fits the map and
    reports what the fit found, including the bands too thin to score.

    `by_case_type` is what trigger T7 reads: a case type the organisation
    gets right 58% of the time does not get a published attribution,
    however confident an individual case looks.
    """
    try:
        fitted = fit_calibration(connection, user, layer=layer)
    except CalibrationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    by_type: dict[str, dict[str, Any]] = {}
    for template in layer.causal_graph.hypotheses.values():
        accuracy, cases = fitted.accuracy_for(template.case_type)
        by_type.setdefault(
            template.case_type,
            {
                "accuracy": accuracy,
                "cases": cases,
                "publication_floor": layer.adjudication.triggers["T7"].publication_floor,
                "below_floor": bool(
                    accuracy is not None
                    and cases >= layer.adjudication.triggers["T7"].min_closed_cases
                    and accuracy < layer.adjudication.triggers["T7"].publication_floor
                ),
            },
        )

    return CalibrationResponse(
        scored_cases=fitted.scored_cases,
        abstained_cases=fitted.abstained_cases,
        total_cases=fitted.total_cases,
        abstention_rate=fitted.abstention_rate,
        calibrating=fitted.calibrating,
        method=fitted.method,
        bands=tuple(
            CalibrationBand(
                label=band.label,
                low=band.low,
                high=band.high,
                cases=band.cases,
                mean_confidence=band.mean_confidence,
                accuracy=band.accuracy,
                gap=band.gap,
                thin=band.thin,
            )
            for band in fitted.bands
        ),
        ece_raw=fitted.ece_raw,
        ece_calibrated=fitted.ece_calibrated,
        by_case_type=by_type,
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@router.get("/api/telemetry", response_model=TelemetryResponse)
def telemetry(
    user: UserDep,
    layer: LayerDep,
    connection: ConnectionDep,
    case_id: str | None = Query(default=None),
) -> TelemetryResponse:
    """Measured cost and latency, against the two published targets.

    Every figure here comes out of `telemetry_request`, which is written
    per request by `telemetry/recorder.py`. Nothing is asserted: the P95
    is computed over the rows the process actually recorded and marked
    warm, and `cost_estimated` says whether the token counts behind the
    money were billed or estimated from text — which offline they always
    are, because a replayed fixture buys no tokens.

    An empty table is reported as an empty table. A telemetry endpoint
    that invents a plausible latency when nothing has run is worse than
    one that says nothing has run.
    """
    from telemetry.cost import check_cost, mean_cost_per_request, project_measured
    from telemetry.recorder import TelemetryError, check_latency, load_requests

    try:
        rows = load_requests(connection, case_id=case_id)
    except Exception as exc:  # noqa: BLE001 - a warehouse without the table
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"telemetry_request is not readable: {exc}. Run "
                "`engine.warehouse.load.create_schema(connection)` or `make seed`."
            ),
        ) from exc

    targets = layer.telemetry.targets
    mean_cost = mean_cost_per_request(rows)
    budget = check_cost(mean_cost, layer)

    measured_ms: float | None = None
    within: bool | None = None
    try:
        report = check_latency(rows, layer=layer)
        measured_ms, within = report.measured_ms, report.within
    except TelemetryError:
        # No warm rows yet. Reported as absent rather than as a pass.
        pass

    projection = project_measured(rows, layer=layer)
    return TelemetryResponse(
        requests=len(rows),
        warm_requests=sum(1 for row in rows if row.warm),
        p95_latency_ms=measured_ms,
        latency_budget_ms=targets.latency_budget_ms,
        latency_within_budget=within,
        mean_cost_inr=mean_cost,
        cost_ceiling_inr=budget.ceiling_inr,
        cost_within_ceiling=budget.within,
        cost_estimated=any(row.cost_estimated for row in rows),
        llm_calls=sum(row.llm_calls for row in rows),
        cache_hits=sum(row.cache_hits for row in rows),
        cache_misses=sum(row.cache_misses for row in rows),
        rows_released_to_llm=sum(row.rows_released_to_llm for row in rows),
        grounding_claims_checked=sum(row.grounding_claims_checked for row in rows),
        grounding_claims_stripped=sum(row.grounding_claims_stripped for row in rows),
        projection={
            "interactions_per_week": projection.interactions_per_week,
            "cost_per_interaction_inr": projection.cost_per_interaction_inr,
            "weekly_inr": projection.weekly_inr,
            "monthly_inr": projection.monthly_inr,
            "annual_inr": projection.annual_inr,
            "annual_inr_lakh": projection.annual_inr_lakh,
            "basis": projection.basis,
        },
    )


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


@router.get("/api/audit", response_model=AuditResponse)
def audit(
    user: UserDep,
    connection: ConnectionDep,
    kpi: str | None = Query(default=None),
    persona: str | None = Query(default=None),
    released_only: bool = Query(
        default=False,
        description="Only rows where warehouse rows crossed into a model prompt.",
    ),
    limit: int = Query(default=DEFAULT_AUDIT_LIMIT, ge=1, le=MAX_AUDIT_LIMIT),
) -> AuditResponse:
    """Every query that reached the warehouse, newest first.

    TWO KINDS OF ROW, AND THE LOG DISTINGUISHES THEM BY CONSTRUCTION. A
    READ is `execute_governed` returning rows to Python: it records what
    the policy did and releases nothing. A RELEASE is those rows being
    serialised into a model prompt — a different act against a different
    party — and gets its own row with its own count.

    `released_only=true` answers the question an auditor actually asks:
    not "was there a policy" but "did this leave the building".

    The statement itself is never returned. A hash is, which is enough to
    prove two calls ran the same query without copying customer-scoped SQL
    into a response with different access rules from the table it read.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if kpi:
        clauses.append("kpi = ?")
        params.append(kpi)
    if persona:
        clauses.append("persona = ?")
        params.append(persona)
    if released_only:
        clauses.append("rows_released_to_llm > 0")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    rows = connection.execute(
        AUDIT_SQL.format(where=where), [*params, limit]
    ).fetchall()
    totals = connection.execute(AUDIT_TOTALS.format(where=where), params).fetchone()

    entries = tuple(
        AuditEntry(
            audit_id=str(row[0]),
            occurred_at=row[1],
            user_id=str(row[2]),
            persona=str(row[3]),
            kpi=str(row[4]),
            statement_hash=str(row[5]),
            row_predicate=str(row[6]),
            rows_returned=int(row[7]),
            rows_filtered=int(row[8]),
            columns_masked=tuple(
                part for part in str(row[9] or "").split(COLUMN_SEPARATOR) if part
            ),
            rows_released_to_llm=int(row[10] or 0),
            purpose=str(row[11]) if row[11] is not None else None,
            released=int(row[10] or 0) > 0,
        )
        for row in rows
    )
    return AuditResponse(
        entries=entries,
        total=int(totals[0]),
        reads=int(totals[0]) - int(totals[3]),
        releases=int(totals[3]),
        rows_filtered_by_policy=int(totals[1]),
        rows_released_to_llm=int(totals[2]),
    )


__all__ = ["AUDIT_SQL", "DEFAULT_AUDIT_LIMIT", "MAX_AUDIT_LIMIT", "router"]
