"""How much history the warehouse actually HOLDS for a KPI.

Counted from the data, never read off the contract. A KPI contract can
declare seventy-eight weeks of history and the warehouse hold seven, and
case #2471 is exactly that: quick-commerce fulfilment launched seven weeks
ago, its contract asks for twenty-six weekly points before a baseline
means anything, and the gap between the two is the whole case.

Everything here is in WEEKS, because that is the unit the KPI contracts
state their minimum in. A daily source is converted; a weekly source is
counted.
"""

from __future__ import annotations

import duckdb

from engine.db import execute_governed
from security.policy import User
from semantic_layer.schema import KpiContract, SemanticLayer, SourceLocation


class HistoryError(RuntimeError):
    """The KPI's history cannot be counted from the warehouse."""


def primary_source(kpi: KpiContract, layer: SemanticLayer) -> tuple[str, SourceLocation]:
    """The first required source that carries a date.

    A static master has no history to count; the KPI's history is the
    history of the feed that measures it.
    """
    registry = layer.warehouse.sources
    for name in kpi.confidence_rules.required_sources:
        location = registry.get(name)
        if location is not None and not location.is_static:
            return name, location
    raise HistoryError(
        f"{kpi.kpi} requires no dated source, so its history cannot be counted "
        f"(required: {kpi.confidence_rules.required_sources})"
    )


def observed_period_count(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    request,
    layer: SemanticLayer,
) -> int:
    """Weeks of history held for this KPI and scope."""
    kpi = layer.kpis[request.kpi]
    _name, location = primary_source(kpi, layer)

    columns = ["region", "store_id"] if location.scope_join != "none" else []
    where, params = _scope_filter(location, request.scope)
    selected = ", ".join([*columns, f'COUNT(DISTINCT "{location.date_column}") AS points'])
    grouped = f" GROUP BY {', '.join(str(i + 1) for i in range(len(columns)))}" if columns else ""

    rows, _filtered, _masked = execute_governed(
        user,
        request.kpi,
        f'SELECT {selected} FROM "{location.table}"{where}{grouped}',
        params,
        connection=connection,
        layer=layer,
        purpose="qualify.history",
    )
    if not rows:
        return 0

    points = max(int(row["points"] or 0) for row in rows)
    if location.grain == "weekly":
        return points
    return points // layer.warehouse.units.days_per_week


def _scope_filter(location: SourceLocation, scope: str) -> tuple[str, dict[str, str]]:
    if location.scope_column:
        return f' WHERE "{location.scope_column}" = $scope', {"scope": scope}
    if location.scope_join == "region":
        return " WHERE region = $scope", {"scope": scope}
    return "", {}


__all__ = ["HistoryError", "observed_period_count", "primary_source"]
