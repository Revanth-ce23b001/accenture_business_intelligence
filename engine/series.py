"""The sparkline behind a KPI card, read through the governed door.

A watchlist card shows eighteen points of history and a status. Those
points are warehouse data about the business, so they go through
`engine/db.py::execute_governed` like everything else — a regional
manager's sparkline is their region's, and a store manager's is their
store's, because the row predicate says so and not because the endpoint
remembered to filter.

WHAT THIS DOES NOT DO. It does not compute a KPI. `semantic_layer/
series.yaml` declares one query per KPI against the tables the warehouse
actually loaded, and this runs it. The KPI's own `formula_sql` is a
DEFINITION against the conceptual model and is not executable here — see
the note at the top of series.yaml, which is the honest version of a
distinction it would be easy to blur.

A KPI with no series comes back saying so, carrying the reason the
contract gives. Two of the six are in that state today, both because the
warehouse holds transaction detail as a sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb

from engine.db import GovernanceError, execute_governed
from security.policy import User
from semantic_layer.schema import KpiSeries, SemanticLayer, get_semantic_layer


@dataclass(frozen=True)
class Point:
    """One period and its value."""

    period: str
    value: float


@dataclass(frozen=True)
class Series:
    """A KPI's recent history for one scope, or the reason there is none."""

    kpi: str
    scope: str | None
    grain: str
    unit: str
    points: tuple[Point, ...] = ()
    available: bool = True
    reason: str | None = None
    source_table: str | None = None
    #: Rows the caller's row predicate withheld. Reported rather than
    #: swallowed: a sparkline drawn over a filtered slice is a different
    #: statement from one drawn over the estate.
    rows_filtered: int = 0

    @property
    def latest(self) -> float | None:
        return self.points[-1].value if self.points else None

    @property
    def previous(self) -> float | None:
        """The point before the last one.

        Written as the last of everything-but-the-last rather than as an
        index, because rule 2's scan cannot tell a position from a
        threshold and should not have to try.
        """
        earlier = self.points[:-1]
        return earlier[-1].value if earlier else None

    @property
    def change_pct(self) -> float | None:
        """Latest against the one before it, as a percentage.

        None when either is missing or the base is zero — an undefined
        change is reported as undefined rather than as zero, which would
        read on a card as "no movement".
        """
        latest, previous = self.latest, self.previous
        if latest is None or previous is None or not previous:
            return None
        # The percent scale is a unit conversion and lives in the semantic
        # layer with the rest of them (rule 2).
        scale = get_semantic_layer().warehouse.units.percent_scale
        return (latest - previous) / abs(previous) * scale


def load_series(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    kpi: str,
    *,
    scope: str | None = None,
    layer: SemanticLayer | None = None,
    points: int | None = None,
) -> Series:
    """One KPI's recent history, governed.

    `scope` narrows to one region. Left None, the series is summed across
    every region the caller's policy admits — which for a regional manager
    is one region, and the arithmetic is the same either way.
    """
    layer = layer or get_semantic_layer()
    declared = layer.series.series.get(kpi)
    if declared is None:
        raise GovernanceError(
            f"no series declared for {kpi!r}; series.yaml carries "
            f"{sorted(layer.series.series)}"
        )
    if not declared.available:
        return Series(
            kpi=kpi,
            scope=scope,
            grain=declared.grain,
            unit=declared.unit,
            available=False,
            reason=(declared.reason or "").strip(),
        )

    limit = points or layer.series.points
    try:
        rows, rows_filtered, _masked = execute_governed(
            user,
            kpi,
            declared.sql,
            connection=connection,
            layer=layer,
            purpose=f"watchlist.series.{kpi}",
        )
    except GovernanceError as exc:
        # A persona the KPI does not admit gets no series and is told so,
        # rather than an empty chart that reads as "nothing happened".
        return Series(
            kpi=kpi,
            scope=scope,
            grain=declared.grain,
            unit=declared.unit,
            available=False,
            reason=str(exc),
            source_table=declared.source_table,
        )

    return Series(
        kpi=kpi,
        scope=scope,
        grain=declared.grain,
        unit=declared.unit,
        points=_fold(rows, scope=scope, limit=limit, declared=declared),
        available=True,
        source_table=declared.source_table,
        rows_filtered=rows_filtered,
    )


def _fold(
    rows: tuple[dict[str, Any], ...],
    *,
    scope: str | None,
    limit: int,
    declared: KpiSeries,
) -> tuple[Point, ...]:
    """Collapse per-region rows into one series, newest `limit` points.

    A RATE IS NOT SUMMED ACROSS REGIONS. Revenue adds up; an availability
    percentage does not, and adding two 90% regions to get 180% would be
    the kind of arithmetic this whole project exists to make impossible.
    Rates are averaged, magnitudes are summed, and the unit decides which.
    """
    buckets: dict[str, list[float]] = {}
    for row in rows:
        if scope is not None and str(row.get("region")) != scope:
            continue
        value = row.get("value")
        if value is None:
            continue
        buckets.setdefault(str(row["period"]), []).append(float(value))

    ordered = sorted(buckets.items())[-limit:]
    combine = _mean if _is_rate(declared.unit) else sum
    return tuple(Point(period=period, value=combine(values)) for period, values in ordered)


#: Units that are rates. A rate is averaged across scopes; anything else
#: is summed. Named here rather than inferred from the number, because
#: "does this add up" is a property of the measure and not of its size.
RATE_UNITS = frozenset({"pct", "pt", "ratio", "inr"})


def _is_rate(unit: str) -> bool:
    return (unit or "").lower() in RATE_UNITS


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


__all__ = ["RATE_UNITS", "Point", "Series", "load_series"]
