"""The daily series QUALIFY works on, and the design matrix built from it.

One query, one place. Gates 2 to 5 all read the same daily regional series
and the same calendar features, so fetching it once means they cannot
disagree about what the data was.

Two things here are worth reading before the rest of the stage:

`festival_terms` builds the festival regressors from `dim_festival_window`
rather than from `dim_calendar.festival`. That column holds ONE label per
day and festivals overlap — Onam with Ganesh Chaturthi in South, Navratri
with Durga Puja in East — so a design built on it silently drops the
second festival and cannot fit either. The bridge table carries membership
properly, and the day's position in the ramp with it.

`design_matrix` contains no marketing spend and never will. A scheduled
promo window is a calendar event and belongs here; a spend LEVEL does not.
Put spend in the baseline and a marketing cut is absorbed into
"calendar-expected", which would make case #2451's H4 impossible to
eliminate on precedence — the single best moment in the demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import tau

import duckdb
import numpy as np
import pandas as pd

from engine.db import execute_governed
from security.policy import User
from semantic_layer.schema import CalendarRegressors, SemanticLayer, Units

#: Column names in the returned frame. Named so a typo is an ImportError
#: rather than an empty column.
REGION = "region"
DATE = "date"
REVENUE = "revenue"
PERIOD_MONTH = "period_month"


class SeriesError(RuntimeError):
    """The series QUALIFY needs is not available at this scope."""


@dataclass(frozen=True)
class RegionalSeries:
    """Daily net revenue by region, with the calendar beside it."""

    frame: pd.DataFrame
    festivals: pd.DataFrame
    origin: date

    @property
    def regions(self) -> tuple[str, ...]:
        return tuple(sorted(self.frame[REGION].unique()))

    def region(self, name: str) -> pd.DataFrame:
        block = self.frame[self.frame[REGION] == name]
        return block.sort_values(DATE).reset_index(drop=True)

    def period_total(self, name: str, period: str) -> float:
        block = self.region(name)
        return float(block.loc[block[PERIOD_MONTH] == period, REVENUE].sum())


def load_series(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    kpi_id: str,
    layer: SemanticLayer,
    *,
    scope: str | None = None,
) -> RegionalSeries:
    """Fetch the daily regional series and its calendar, governed.

    `scope` restricts to one region. Leaving it None fetches every region
    the caller's row policy allows, which is what Gate 4 needs — and if the
    policy allows only one, Gate 4 says so rather than concluding the
    movement is specific.
    """
    where = " WHERE s.region = $scope" if scope else ""
    params = {"scope": scope} if scope else {}

    rows, _filtered, _masked = execute_governed(
        user,
        kpi_id,
        f"""
        SELECT
            s.region,
            s.txn_date          AS date,
            SUM(s.net_revenue_inr) AS revenue,
            ANY_VALUE(c.dow)             AS dow,
            ANY_VALUE(c.is_payday)       AS is_payday,
            ANY_VALUE(c.is_month_end)    AS is_month_end,
            ANY_VALUE(c.is_promo_window) AS is_promo_window,
            ANY_VALUE(c.period_month)    AS period_month,
            ANY_VALUE(c.iso_year)        AS iso_year,
            ANY_VALUE(c.iso_week)        AS iso_week
        FROM fact_sales_daily AS s
        JOIN dim_calendar AS c ON c.date = s.txn_date{where}
        GROUP BY 1, 2
        """,
        params,
        connection=connection,
        layer=layer,
        purpose="qualify.series",
    )
    if not rows:
        raise SeriesError(
            f"no daily revenue visible for {scope or 'any region'} on {kpi_id}"
        )

    frame = pd.DataFrame(list(rows))
    frame[DATE] = pd.to_datetime(frame[DATE])
    frame = frame.sort_values([REGION, DATE]).reset_index(drop=True)

    festivals = _load_festivals(connection, user, kpi_id, layer)
    return RegionalSeries(
        frame=frame, festivals=festivals, origin=frame[DATE].min().date()
    )


def _load_festivals(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    kpi_id: str,
    layer: SemanticLayer,
) -> pd.DataFrame:
    """Festival membership per day. Pipeline metadata: no measure, no store."""
    from engine.db import execute_metadata

    rows = execute_metadata(
        user,
        "dim_festival_window",
        """
        SELECT date, festival, occurrence, phase, day_index, phase_days
        FROM dim_festival_window
        ORDER BY date, festival, phase
        """,
        connection=connection,
        layer=layer,
        purpose="qualify.festival_windows",
    )
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=["date", "festival", "phase", "day_index", "phase_days"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def festival_terms(
    dates: pd.Series, festivals: pd.DataFrame, spec: CalendarRegressors
) -> pd.DataFrame:
    """One column per (festival, phase, segment), multi-hot over dates.

    A festival window is split into `festival_window_segments` equal parts
    and its hangover into `festival_hangover_segments`, so the ramp is
    approximated by steps rather than assumed flat. Nothing here knows the
    shape of the ramp — only that there is one.
    """
    index = pd.Index(dates.to_numpy(), name="date")
    terms = pd.DataFrame(index=range(len(index)))
    if festivals.empty:
        return terms

    segments = {
        "window": spec.festival_window_segments,
        "hangover": spec.festival_hangover_segments,
    }
    marked = festivals.copy()
    marked["segment"] = [
        min(
            int(day_index * segments[phase] / max(phase_days, 1)),
            segments[phase] - 1,
        )
        for day_index, phase, phase_days in zip(
            marked["day_index"], marked["phase"], marked["phase_days"], strict=True
        )
    ]
    marked["term"] = (
        "fest_" + marked["festival"] + "_" + marked["phase"] + marked["segment"].astype(str)
    )

    position = pd.Series(np.arange(len(index)), index=index)
    for term, block in marked.groupby("term", sort=True):
        column = np.zeros(len(index))
        hits = position.reindex(block["date"].to_numpy()).dropna()
        if len(hits):
            column[hits.to_numpy().astype(int)] = 1.0
        terms[term] = column
    return terms


def design_matrix(
    block: pd.DataFrame,
    festivals: pd.DataFrame,
    spec: CalendarRegressors,
    origin: date,
    units: Units,
) -> pd.DataFrame:
    """The calendar design. Every column is something a calendar knows.

    `trading days in the period` is not a column: the model is daily and a
    period's prediction is the sum over the days that period actually has.
    November having one fewer day than October is therefore already in the
    answer, and on case #2451 it is very nearly all of it.
    """
    block = block.reset_index(drop=True)
    X = pd.DataFrame(index=block.index)
    X["const"] = 1.0

    if spec.day_of_week:
        # One dropped level: Monday is the baseline the constant absorbs.
        levels = sorted(set(block["dow"].to_numpy()))
        for level in levels[1:]:
            X[f"dow_{level}"] = (block["dow"] == level).astype(float)

    elapsed = (block[DATE] - pd.Timestamp(origin)).dt.days.to_numpy()
    if spec.trend:
        X["trend_years"] = elapsed / units.days_per_year

    day_of_year = block[DATE].dt.dayofyear.to_numpy()
    for harmonic in range(spec.annual_harmonics):
        order = harmonic + 1
        # tau is 2*pi. One full turn of the year per harmonic.
        angle = tau * order * day_of_year / units.days_per_year
        X[f"annual_sin_{order}"] = np.sin(angle)
        X[f"annual_cos_{order}"] = np.cos(angle)

    if spec.pay_cycle:
        X["pay_cycle"] = block["is_payday"].astype(float)
    if spec.month_end:
        X["month_end"] = block["is_month_end"].astype(float)
    if spec.promo_window:
        X["promo_window"] = block["is_promo_window"].astype(float)

    terms = festival_terms(block[DATE], festivals, spec)
    for column in terms.columns:
        X[column] = terms[column].to_numpy()
    return X


__all__ = [
    "DATE",
    "PERIOD_MONTH",
    "REGION",
    "REVENUE",
    "RegionalSeries",
    "SeriesError",
    "design_matrix",
    "festival_terms",
    "load_series",
]
