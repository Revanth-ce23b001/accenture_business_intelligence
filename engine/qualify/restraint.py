"""Restraint — the three mechanics that stop a working investigator becoming
an unreadable one.

A stock-out moves net revenue, transactions, conversion rate and on-shelf
availability. That is four KPIs and ONE event, and an investigator that
opens four cases has not understood the business. An investigator that
opens forty a week gets ignored, and an ignored investigator is worth
nothing at all — so this module is not an optimisation. It is the
difference between a system that is used and a system that is muted.

  deduplication  one cause, one case. Correlated KPIs on the same scope and
                 period link to the case that is already open.
  suppression    while a case is open on a scope, later periods are covered
                 by it. A materially worse movement is an escalation and
                 reopens rather than being swallowed.
  owner load     three open cases per owner per week. Above the cap,
                 qualifying movements are DEFERRED to the digest and
                 reported as deferred — never dropped.

Every decision reads `case_registry` through `engine/db.py::execute_metadata`
and returns a named outcome, so "why did nothing happen" always has an
answer with a name on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb

from engine.db import execute_metadata
from security.policy import User
from semantic_layer.schema import KpiContract, RestraintSpec, SemanticLayer

#: `case_registry.status` for a case that has not been closed.
OPEN_STATUS = "in_progress"


@dataclass(frozen=True)
class OpenCase:
    """A row of `case_registry` that is still running."""

    case_id: str
    kpi: str
    scope: str
    grain: str
    period: str
    opened_at: datetime
    status: str
    materiality_multiple: float | None = None


@dataclass(frozen=True)
class RestraintVerdict:
    """Whether a qualifying movement is allowed to become a case."""

    outcome: str
    detail: str
    linked_case_id: str | None = None
    deferred_to: str | None = None
    open_cases_this_week: int = 0

    @property
    def allowed(self) -> bool:
        return self.outcome == "CLEAN"


def correlated_kpis(kpi: KpiContract, spec: RestraintSpec) -> tuple[str, ...]:
    """The KPIs a case on this one would duplicate.

    Declared in the KPI contracts as `drivers` and `related_kpis`, not
    inferred from the data. The business already wrote down which numbers
    move together; correlating them again from the series would find the
    same thing more slowly and less reliably.
    """
    names: set[str] = set()
    for source in spec.deduplication.link_via:
        names.update(getattr(kpi, source, ()) or ())
    names.discard(kpi.kpi)
    return tuple(sorted(names))


def open_cases(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
    *,
    kpis: tuple[str, ...] | None = None,
) -> tuple[OpenCase, ...]:
    """Every case still running, optionally restricted to some KPIs."""
    rows = execute_metadata(
        user,
        "case_registry",
        """
        SELECT case_id, kpi, scope, grain, period, opened_at, status,
               materiality_multiple
        FROM case_registry
        WHERE status = $open
        ORDER BY opened_at
        """,
        {"open": OPEN_STATUS},
        connection=connection,
        layer=layer,
        purpose="qualify.restraint.open_cases",
    )
    cases = tuple(
        OpenCase(
            case_id=str(row["case_id"]),
            kpi=str(row["kpi"]),
            scope=str(row["scope"]),
            grain=str(row["grain"]),
            period=str(row["period"]),
            opened_at=row["opened_at"],
            status=str(row["status"]),
            materiality_multiple=(
                float(row["materiality_multiple"])
                if row.get("materiality_multiple") is not None
                else None
            ),
        )
        for row in rows
    )
    if kpis is None:
        return cases
    wanted = set(kpis)
    return tuple(case for case in cases if case.kpi in wanted)


def assess_restraint(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    kpi: KpiContract,
    layer: SemanticLayer,
    *,
    scope: str,
    period: str,
    residual_multiple: float,
    now: datetime,
) -> RestraintVerdict:
    """Decide whether this movement may open a case.

    `residual_multiple` is how many times its materiality limit the
    residual is. Suppression uses it: a movement materially worse than the
    open case's is an escalation, not a repeat.
    """
    spec = layer.qualify.restraint
    running = open_cases(connection, user, layer)
    # The warehouse clock is timezone-aware; `case_registry.opened_at` is a
    # naive DuckDB TIMESTAMP holding UTC. Compare like with like.
    now = _naive(now)

    if spec.deduplication.enabled:
        family = (kpi.kpi, *correlated_kpis(kpi, spec))
        for case in running:
            if case.kpi not in family or case.kpi == kpi.kpi:
                continue
            if spec.deduplication.require_same_scope and case.scope != scope:
                continue
            if spec.deduplication.require_same_period and case.period != period:
                continue
            return RestraintVerdict(
                outcome=spec.deduplication.outcome_code,
                detail=(
                    f"{case.case_id} is already open on {case.kpi} for {scope} "
                    f"{period}. {kpi.kpi} and {case.kpi} move together — one cause, "
                    "one case."
                ),
                linked_case_id=case.case_id,
            )

    if spec.suppression.enabled:
        window = timedelta(days=spec.suppression.window_days)
        for case in running:
            if case.kpi != kpi.kpi or case.scope != scope or case.period == period:
                continue
            age = now - _naive(case.opened_at)
            if age > window:
                continue
            # An escalation, not a repeat: this movement is materially
            # worse than the one already being investigated. Measured
            # against the OPEN CASE's magnitude, not against materiality —
            # every case worth opening is several times materiality, so a
            # materiality comparison would suppress nothing.
            if case.materiality_multiple and (
                residual_multiple
                >= spec.suppression.escalation_multiple * case.materiality_multiple
            ):
                continue
            return RestraintVerdict(
                outcome=spec.suppression.outcome_code,
                detail=(
                    f"{case.case_id} has been open on {kpi.kpi} / {scope} since "
                    f"{_naive(case.opened_at).date()}, inside the "
                    f"{spec.suppression.window_days}-day window. This period is covered "
                    "by that investigation."
                ),
                linked_case_id=case.case_id,
            )

    if spec.owner_load.enabled:
        owner = getattr(kpi, spec.owner_load.owner_from.rsplit(".", maxsplit=1)[-1])
        owners = {name: contract.owner_role for name, contract in layer.kpis.items()}
        week_start = _week_start(now)
        this_week = [
            case
            for case in running
            if owners.get(case.kpi) == owner and _naive(case.opened_at) >= week_start
        ]
        if len(this_week) >= spec.owner_load.max_open_cases_per_owner_per_week:
            return RestraintVerdict(
                outcome=spec.owner_load.outcome_code,
                detail=(
                    f"{owner} already has {len(this_week)} cases open this week, at the "
                    f"cap of {spec.owner_load.max_open_cases_per_owner_per_week}. "
                    f"Deferred to the {spec.owner_load.over_cap_route}, not dropped."
                ),
                deferred_to=spec.owner_load.over_cap_route,
                open_cases_this_week=len(this_week),
            )

    return RestraintVerdict(
        outcome="CLEAN",
        detail="no open case covers this scope and period, and the owner is under cap.",
        open_cases_this_week=sum(
            1
            for case in running
            if layer.kpis.get(case.kpi) is not None
            and layer.kpis[case.kpi].owner_role == kpi.owner_role
            and _naive(case.opened_at) >= _week_start(now)
        ),
    )


def register_case(
    connection: duckdb.DuckDBPyConnection,
    *,
    case_id: str,
    kpi: str,
    scope: str,
    grain: str,
    period: str,
    opened_at: datetime,
    materiality_multiple: float | None = None,
    status: str = OPEN_STATUS,
    linked_from_case_id: str | None = None,
) -> None:
    """Write the `case_registry` row. A write, so it does not go through a reader.

    `linked_from_case_id` is set only by RECOMMEND, when a playbook's
    linked-case template raises the cause of the cause. A case a movement
    opened has no parent and leaves it null.
    """
    from engine.db import as_stored_timestamp

    connection.execute(
        """
        INSERT INTO case_registry
            (case_id, kpi, scope, grain, period, opened_at, status,
             materiality_multiple, linked_from_case_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            case_id,
            kpi,
            scope,
            grain,
            period,
            as_stored_timestamp(opened_at),
            status,
            materiality_multiple,
            linked_from_case_id,
        ],
    )


def _naive(moment: datetime) -> datetime:
    return moment.replace(tzinfo=None) if moment.tzinfo is not None else moment


def _week_start(moment: datetime) -> datetime:
    """Monday of the ISO week `moment` falls in, at midnight."""
    day = _naive(moment)
    midnight = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight - timedelta(days=day.weekday())


__all__ = [
    "OPEN_STATUS",
    "OpenCase",
    "RestraintVerdict",
    "assess_restraint",
    "correlated_kpis",
    "open_cases",
    "register_case",
]
