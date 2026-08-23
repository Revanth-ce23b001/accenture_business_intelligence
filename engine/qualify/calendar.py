"""Gate 2 — the calendar decomposition.

An OLS on daily regional revenue in logs, fitted on history that stops
before the period under test, then asked what the calendar alone would
have delivered in M and in M-1.

DECOMPOSE, DO NOT ADJUST. Nothing is subtracted from the series and no
adjusted series is emitted. The gate reports how many points of the
movement the calendar accounts for and hands the untouched numbers on,
because a reader cannot tell an adjustment from a fact and an adjusted
series cannot be audited.

The identity is exact by construction:

    headline_pt = calendar_pt + residual_pt

with every term measured against the ACTUAL level of M-1. The calendar
effect is the prediction for M against that actual level, not against the
prediction for M-1: folding the model's fit error on a clean month into
"calendar" would credit the calendar with something it did not do. The
prediction for M-1 is emitted anyway, so the fit error is visible.

On case #2451 the answer is almost arithmetic. West's festival amplitude
is small and November has thirty days against October's thirty-one:
30/31 - 1 = -3.2%, which is the whole calendar effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm

from engine.qualify.series import (
    DATE,
    PERIOD_MONTH,
    REVENUE,
    RegionalSeries,
    design_matrix,
)
from semantic_layer.schema import CalendarGate, SemanticLayer, Units


class CalendarError(RuntimeError):
    """The calendar baseline could not be fitted."""


@dataclass(frozen=True)
class CalendarFit:
    """A fitted baseline, and what it had to leave out to be one."""

    region: str
    fit_days: int
    fit_months: tuple[str, ...]
    excluded_months: tuple[str, ...]
    r_squared: float
    parameters: int
    residuals: pd.Series
    _model: object

    def predict(
        self,
        block: pd.DataFrame,
        series: RegionalSeries,
        spec: CalendarGate,
        units: Units,
    ) -> np.ndarray:
        """Predicted daily revenue, back on the rupee scale."""
        X = design_matrix(
            block, series.festivals, spec.model.regressors, series.origin, units
        )
        return np.exp(np.asarray(self._model.predict(X)))


@dataclass(frozen=True)
class CalendarDecomposition:
    """How much of the movement the calendar accounts for."""

    region: str
    period: str
    comparison_period: str
    actual_period_inr: float
    actual_comparison_inr: float
    predicted_period_inr: float
    predicted_comparison_inr: float
    headline_pt: float
    calendar_pt: float
    residual_pt: float
    fit: CalendarFit
    #: Carried so the percentage arithmetic has a source. Rule 2 leaves the
    #: engine no numbers of its own, not even a hundred.
    percent_scale: float

    @property
    def residual_inr(self) -> float:
        """The residual back in rupees, on the comparison period's base."""
        return self.residual_pt / self.percent_scale * self.actual_comparison_inr

    @property
    def calendar_inr(self) -> float:
        return self.calendar_pt / self.percent_scale * self.actual_comparison_inr

    @property
    def comparison_fit_error_pt(self) -> float:
        """How far the baseline missed a period nobody is disputing.

        Emitted rather than absorbed. A big number here means the whole
        decomposition should be read with suspicion.
        """
        if not self.actual_comparison_inr:
            return 0.0
        return (
            (self.predicted_comparison_inr - self.actual_comparison_inr)
            / self.actual_comparison_inr
            * self.percent_scale
        )

    def reconciles(self, tolerance: float) -> bool:
        return abs(self.headline_pt - self.calendar_pt - self.residual_pt) <= tolerance


def fit_calendar(
    series: RegionalSeries,
    region: str,
    spec: CalendarGate,
    units: Units,
    *,
    before_period: str,
) -> CalendarFit:
    """Fit the baseline on history strictly before `before_period`.

    Two passes. The first fits everything; months whose mean residual
    exceeds the configured multiple of the monthly residual spread are
    dropped; the second refits without them. A calendar baseline is not
    fitted through a month the business already knows was abnormal, and
    the months it dropped are reported rather than quietly removed.
    """
    block = series.region(region)
    history = block[block[PERIOD_MONTH] < before_period].reset_index(drop=True)
    if len(history) < spec.model.min_fit_days:
        raise CalendarError(
            f"{region}: {len(history)} days of history before {before_period}, and "
            f"{spec.model.min_fit_days} are needed to fit a calendar baseline"
        )

    X = design_matrix(
        history, series.festivals, spec.model.regressors, series.origin, units
    )
    y = np.log(history[REVENUE].to_numpy())

    first = sm.OLS(y, X).fit()
    by_month = pd.DataFrame(
        {"month": history[PERIOD_MONTH].to_numpy(), "residual": first.resid}
    ).groupby("month")["residual"].mean()
    spread = float(by_month.std())
    limit = spec.model.anomaly_exclusion.sigma * spread
    excluded = tuple(sorted(by_month[np.abs(by_month) > limit].index.tolist()))

    keep = ~history[PERIOD_MONTH].isin(excluded).to_numpy()
    model = sm.OLS(y[keep], X[keep]).fit() if excluded else first

    return CalendarFit(
        region=region,
        fit_days=int(keep.sum()),
        fit_months=tuple(sorted(set(history[PERIOD_MONTH][keep]))),
        excluded_months=excluded,
        r_squared=float(model.rsquared),
        parameters=int(X.shape[1]),
        residuals=pd.Series(
            np.asarray(y - model.predict(X)), index=history[DATE], name="residual"
        ),
        _model=model,
    )


def decompose(
    series: RegionalSeries,
    region: str,
    period: str,
    comparison_period: str,
    spec: CalendarGate,
    units: Units,
    *,
    fit: CalendarFit | None = None,
) -> CalendarDecomposition:
    """Split the movement into calendar and residual. Nothing is adjusted."""
    fit = fit or fit_calendar(series, region, spec, units, before_period=period)
    block = series.region(region)

    predicted: dict[str, float] = {}
    actual: dict[str, float] = {}
    for name in (period, comparison_period):
        window = block[block[PERIOD_MONTH] == name].reset_index(drop=True)
        if window.empty:
            raise CalendarError(f"{region}: no daily revenue for period {name}")
        predicted[name] = float(fit.predict(window, series, spec, units).sum())
        actual[name] = float(window[REVENUE].sum())

    base = actual[comparison_period]
    if not base:
        raise CalendarError(f"{region}: {comparison_period} is empty; no base to move from")

    scale = units.percent_scale
    headline_pt = (actual[period] - base) / base * scale
    if spec.decomposition.denominator == "actual_previous_period":
        calendar_pt = (predicted[period] - base) / base * scale
    else:
        calendar_pt = (predicted[period] - predicted[comparison_period]) / base * scale
    residual_pt = headline_pt - calendar_pt

    return CalendarDecomposition(
        region=region,
        period=period,
        comparison_period=comparison_period,
        actual_period_inr=actual[period],
        actual_comparison_inr=base,
        predicted_period_inr=predicted[period],
        predicted_comparison_inr=predicted[comparison_period],
        headline_pt=headline_pt,
        calendar_pt=calendar_pt,
        residual_pt=residual_pt,
        fit=fit,
        percent_scale=scale,
    )


def previous_month(period: str) -> str:
    """`2025-11` -> `2025-10`. Months only; other grains carry their own."""
    stamp = pd.Timestamp(f"{period}-01") - pd.DateOffset(months=1)
    return stamp.strftime("%Y-%m")


def trading_days(series: RegionalSeries, region: str, period: str) -> int:
    block = series.region(region)
    return int((block[PERIOD_MONTH] == period).sum())


def decompose_all_regions(
    series: RegionalSeries,
    period: str,
    comparison_period: str,
    spec: CalendarGate,
    units: Units,
) -> dict[str, CalendarDecomposition]:
    """The same method, region by region. Gate 4 compares the results.

    A region whose history is too short to fit is left out rather than
    given a zero, so a peer that could not be modelled never counts as a
    peer that did not move.
    """
    out: dict[str, CalendarDecomposition] = {}
    for region in series.regions:
        try:
            out[region] = decompose(
                series, region, period, comparison_period, spec, units
            )
        except CalendarError:
            continue
    return out


__all__ = [
    "CalendarDecomposition",
    "CalendarError",
    "CalendarFit",
    "decompose",
    "decompose_all_regions",
    "fit_calendar",
    "previous_month",
    "trading_days",
]
