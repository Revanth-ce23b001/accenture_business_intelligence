"""Gate 3 — is there enough history, and is the residual outside the band?

Two questions, in that order, because the second is meaningless without
the first. Case #2471 stops here: a KPI that launched seven weeks ago has
no baseline, and twenty-six weekly points are required before one means
anything. It is monitored, not investigated.

Then the band. NOT sigma. Business series are heavy-tailed and
autocorrelated, so "two standard deviations" is not a 95% statement about
anything — it is a statement about a Gaussian nobody has seen. The band is
the empirical quantile of what this scope's residual actually does, over
the trailing lookback the KPI contract asks for.

STL(period=7, robust=True) runs on the CALENDAR-ADJUSTED series, not on
raw revenue. On raw revenue the residual is dominated by festivals — which
Gate 2 has already explained — and the band comes out several times too
wide, so a real movement hides inside it.

A consequence worth stating rather than hiding: the more a region's trade
depends on festivals, the less precisely the calendar model fits it, and
the wider its band. South's band is several times West's. That is the
honest answer — we know less about South — not a defect to tune away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from engine.qualify.calendar import CalendarFit
from engine.qualify.series import DATE, RegionalSeries
from semantic_layer.schema import BandGate, KpiContract, Units


class BandError(RuntimeError):
    """The band could not be computed as configured."""


@dataclass(frozen=True)
class HistoryCheck:
    """Whether this KPI has enough history to have an opinion."""

    observed_periods: int
    required_periods: int
    unit: str

    @property
    def sufficient(self) -> bool:
        return self.observed_periods >= self.required_periods


@dataclass(frozen=True)
class ResidualBand:
    """The normal size of what the calendar model cannot explain."""

    region: str
    band_pt: float
    quantile: float
    lookback_weeks: int
    observations: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp

    def breached_by(self, residual_pt: float) -> bool:
        return abs(residual_pt) > self.band_pt

    def direction_of(self, residual_pt: float) -> int:
        """-1, 0 or +1. Zero when the residual sits inside the band."""
        if not self.breached_by(residual_pt):
            return 0
        return 1 if residual_pt > 0 else -1


def check_history(
    observed_periods: int, kpi: KpiContract, grain: str
) -> HistoryCheck:
    """Count the periods the warehouse HOLDS against the contract's minimum.

    Counted from the data, never read off `history_weeks`: a contract can
    declare seventy-eight weeks of history and the warehouse hold seven,
    and #2471 is exactly that case.
    """
    required = kpi.baseline.min_history_weeks
    return HistoryCheck(
        observed_periods=observed_periods,
        required_periods=required,
        unit="weeks" if grain == "weekly" else grain,
    )


def residual_band(
    series: RegionalSeries,
    region: str,
    fit: CalendarFit,
    kpi: KpiContract,
    spec: BandGate,
    units: Units,
    *,
    window_end: pd.Timestamp | None = None,
) -> ResidualBand:
    """Empirical quantile band of the STL residual over the trailing lookback."""
    thresholds = kpi.thresholds
    if thresholds.band_lookback_weeks is None or thresholds.band_quantile is None:
        raise BandError(
            f"{kpi.kpi} declares no band lookback or quantile; it cannot open a case "
            "on a residual it has no band for"
        )

    residuals = fit.residuals
    if spec.stl.input != "calendar_residual":  # pragma: no cover - config guard
        raise BandError(f"unsupported band input {spec.stl.input!r}")
    if len(residuals) <= spec.stl.period:
        raise BandError(
            f"{region}: {len(residuals)} residual days, too few for STL at period "
            f"{spec.stl.period}"
        )

    decomposed = STL(
        residuals.to_numpy(), period=spec.stl.period, robust=spec.stl.robust
    ).fit()
    unexplained = pd.Series(decomposed.resid, index=residuals.index)

    end = window_end if window_end is not None else pd.Timestamp(residuals.index.max())
    start = end - pd.Timedelta(weeks=thresholds.band_lookback_weeks)
    window = unexplained[(unexplained.index > start) & (unexplained.index <= end)]
    if len(window) < spec.min_band_observations:
        raise BandError(
            f"{region}: {len(window)} residual days in the trailing "
            f"{thresholds.band_lookback_weeks} weeks, and {spec.min_band_observations} "
            "are needed for an empirical quantile to mean anything"
        )

    band = (
        float(np.quantile(np.abs(window.to_numpy()), thresholds.band_quantile))
        * units.percent_scale
    )
    return ResidualBand(
        region=region,
        band_pt=band,
        quantile=thresholds.band_quantile,
        lookback_weeks=thresholds.band_lookback_weeks,
        observations=len(window),
        window_start=start,
        window_end=end,
    )


def period_end(series: RegionalSeries, region: str, period: str) -> pd.Timestamp:
    """Last day of `period` that the warehouse actually holds."""
    block = series.region(region)
    days = block.loc[block["period_month"] == period, DATE]
    if days.empty:
        raise BandError(f"{region}: no days held for {period}")
    return pd.Timestamp(days.max())


__all__ = [
    "BandError",
    "HistoryCheck",
    "ResidualBand",
    "check_history",
    "period_end",
    "residual_band",
]
