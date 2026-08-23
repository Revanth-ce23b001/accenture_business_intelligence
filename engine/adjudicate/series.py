"""What the six tests read: series, panels and store covariates.

Fetched once per adjudication and shared, so no two tests can disagree
about what the data was. Every read goes through
`engine/db.py::execute_governed` (CLAUDE.md rule 5), and every statement
comes from `semantic_layer/adjudicate.yaml` — this module writes no SQL.

NO MODEL CALL HAPPENS ANYWHERE IN THIS STAGE. GATHER already asked the
model what a document was about. ADJUDICATE asks whether the evidence
supports the hypothesis, and that is arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from engine.db import execute_governed
from security.policy import User
from semantic_layer.schema import SemanticLayer, Series, Units, get_semantic_layer

#: The degree of the pre-trend fit. A line, so two points are the minimum
#: that determines one — which is why the guard below is written against
#: the degree rather than against a two nobody can trace.
SLOPE_DEGREE = 1


def units() -> Units:
    """Scale factors, from the layer. No module keeps its own hundred."""
    return get_semantic_layer().warehouse.units

#: Columns every configured statement returns.
POINT = "point"
VALUE = "value"
STORE = "store_id"


class SeriesError(RuntimeError):
    """A series the tests need is not available at this scope."""


@dataclass(frozen=True)
class DailySeries:
    """One scope-level series, one row per day."""

    label: str
    source_system: str
    direction: str | None
    frame: pd.DataFrame
    statement: str
    store_grain: bool

    @property
    def points(self) -> pd.Series:
        """Value indexed by day, summed or averaged across stores already."""
        return self.frame.set_index(POINT)[VALUE].sort_index()

    def by_store(self, start: date, end: date) -> pd.Series:
        """Mean value per store over a window. Empty when there is no store grain."""
        if not self.store_grain:
            return pd.Series(dtype=float)
        window = self.frame[
            (self.frame[POINT] >= pd.Timestamp(start))
            & (self.frame[POINT] <= pd.Timestamp(end))
        ]
        return window.groupby(STORE)[VALUE].mean()


@dataclass(frozen=True)
class StorePanel:
    """Every store in scope: covariates, pre and post revenue, pre-trend.

    There is deliberately NO treatment column. Which stores a cause
    reached is derived per hypothesis from that hypothesis's own cause
    series (`tests.exposed_stores`), because which stores were affected is
    a conclusion of the investigation and not an input to it.
    """

    frame: pd.DataFrame
    period_start: date
    period_end: date
    comparison_start: date
    comparison_end: date

    @property
    def scope_pre_inr(self) -> float:
        return float(self.frame["pre"].sum())

    def growth_pct(self, store_ids) -> np.ndarray:
        block = self.frame[self.frame[STORE].isin(list(store_ids))]
        return ((block["post"] / block["pre"] - 1.0) * units().percent_scale).to_numpy()

    @property
    def store_ids(self) -> list[str]:
        return [str(item) for item in self.frame[STORE]]


def load_series(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
    spec: Series,
    *,
    kpi: str,
    scope: str,
    window_start: date,
    window_end: date,
) -> DailySeries:
    """Run one configured statement and shape it into a daily series."""
    rows, _filtered, _masked = execute_governed(
        user,
        kpi,
        spec.sql,
        {"window_start": window_start, "window_end": window_end, "scope": scope},
        connection=connection,
        layer=layer,
        purpose="adjudicate.series",
    )
    if not rows:
        raise SeriesError(f"{spec.label}: no rows for {scope} over {window_start}..{window_end}")

    frame = pd.DataFrame(list(rows))
    frame[POINT] = pd.to_datetime(frame[POINT])
    store_grain = STORE in frame.columns

    if store_grain:
        daily = frame.groupby(POINT, as_index=False)[VALUE].mean()
    else:
        daily = frame.groupby(POINT, as_index=False)[VALUE].sum()

    return DailySeries(
        label=spec.label,
        source_system=spec.source_system,
        direction=spec.direction,
        frame=frame if store_grain else daily,
        statement=spec.sql.strip(),
        store_grain=store_grain,
    )


def daily_points(series: DailySeries, stores: tuple[str, ...] = ()) -> pd.Series:
    """Scope-level daily series, whatever grain it arrived at.

    `stores` narrows a store-grain series to a named set — the stores a
    cause actually reached — because averaging a fault across an estate it
    never touched flattens the step it is being searched for. A series
    with no store dimension ignores it: there is nothing to narrow.
    """
    if not series.store_grain:
        return series.frame.set_index(POINT)[VALUE].sort_index()
    frame = series.frame
    if stores:
        narrowed = frame[frame[STORE].isin(list(stores))]
        if not narrowed.empty:
            frame = narrowed
    return frame.groupby(POINT)[VALUE].mean().sort_index()


def load_panel(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
    *,
    kpi: str,
    scope: str,
    period_start: date,
    period_end: date,
    comparison_start: date,
    comparison_end: date,
    pre_trend_weeks: int,
    onset: date,
) -> StorePanel:
    """Store covariates, pre and post revenue, and the pre-trend slope.

    The pre-trend is fitted on the weeks BEFORE the onset, in logs, so a
    big store and a small one with the same growth rate match on it.
    """
    rows, _filtered, _masked = execute_governed(
        user,
        kpi,
        """
        SELECT
            d.region,
            d.store_id,
            d.store_format,
            d.catchment_type,
            d.sqft,
            d.staff_headcount,
            SUM(CASE WHEN f.txn_date BETWEEN $comparison_start AND $comparison_end
                     THEN f.net_revenue_inr ELSE 0 END) AS pre,
            SUM(CASE WHEN f.txn_date BETWEEN $period_start AND $period_end
                     THEN f.net_revenue_inr ELSE 0 END) AS post
        FROM fact_sales_daily AS f
        JOIN dim_store AS d USING (store_id)
        WHERE d.region = $scope
        GROUP BY 1, 2, 3, 4, 5, 6
        """,
        {
            "scope": scope,
            "period_start": period_start,
            "period_end": period_end,
            "comparison_start": comparison_start,
            "comparison_end": comparison_end,
        },
        connection=connection,
        layer=layer,
        purpose="adjudicate.panel",
    )
    if not rows:
        raise SeriesError(f"no stores visible in {scope}")
    frame = pd.DataFrame(list(rows))

    trend_start = onset - timedelta(weeks=pre_trend_weeks)
    trend_end = onset - timedelta(days=1)
    daily, _filtered, _masked = execute_governed(
        user,
        kpi,
        """
        SELECT region, store_id, txn_date, net_revenue_inr
        FROM fact_sales_daily
        WHERE region = $scope AND txn_date BETWEEN $trend_start AND $trend_end
        """,
        {"scope": scope, "trend_start": trend_start, "trend_end": trend_end},
        connection=connection,
        layer=layer,
        purpose="adjudicate.pre_trend",
    )
    history = pd.DataFrame(list(daily))
    history["txn_date"] = pd.to_datetime(history["txn_date"])

    slopes: dict[str, float] = {}
    levels: dict[str, float] = {}
    for store_id, block in history.groupby(STORE):
        ordered = block.sort_values("txn_date")
        elapsed = (ordered["txn_date"] - ordered["txn_date"].min()).dt.days.to_numpy(float)
        revenue = ordered["net_revenue_inr"].to_numpy(float)
        positive = revenue > 0
        if positive.sum() <= SLOPE_DEGREE:
            continue
        slopes[str(store_id)] = float(
            np.polyfit(elapsed[positive], np.log(revenue[positive]), SLOPE_DEGREE)[0]
        )
        levels[str(store_id)] = float(revenue.sum())

    frame["pre_trend_8wk"] = frame[STORE].map(slopes)
    frame["pre_level"] = frame[STORE].map(levels)
    frame = frame.dropna(subset=["pre_trend_8wk", "pre", "post"])
    frame = frame[frame["pre"] > 0].reset_index(drop=True)

    return StorePanel(
        frame=frame,
        period_start=period_start,
        period_end=period_end,
        comparison_start=comparison_start,
        comparison_end=comparison_end,
    )


__all__ = [
    "POINT",
    "SLOPE_DEGREE",
    "STORE",
    "VALUE",
    "DailySeries",
    "SeriesError",
    "StorePanel",
    "daily_points",
    "load_panel",
    "load_series",
    "units",
]
