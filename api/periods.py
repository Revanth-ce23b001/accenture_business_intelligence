"""Turning a period label into the four dates the engine needs.

`2025-11` at monthly grain means 1 Nov to 30 Nov, compared against 1 Oct
to 31 Oct. `2025-11-12` at daily grain means that day against the one
before it.

WHY THE CALLER DOES NOT SUPPLY THE WINDOWS. `CaseRequest` carries four
dates, and a caller free to set them independently could ask for November
against a fortnight in March. The engine would answer honestly and the
answer would be meaningless. So the API accepts a PERIOD and derives the
window, and the only thing a caller may override is which period to
compare against — a whole period, never a pair of loose dates.

No calendar cleverness lives here. Month lengths come from the standard
library and the comparison period is the previous one at the same grain,
which is what every KPI contract's `comparison` says. A KPI that wanted
year-on-year would say so in its contract, and this would read it.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

MONTH_PATTERN = re.compile(r"^(\d{4})-(\d{2})$")
DAY_PATTERN = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
WEEK_PATTERN = re.compile(r"^(\d{4})-W(\d{2})$")

#: ISO weekday of Monday, and the number of days a week spans. Named
#: because `weekday() == 0` at a call site is a puzzle.
MONDAY = 1
DAYS_IN_WEEK = 7


class PeriodError(ValueError):
    """The period label does not parse at the grain given."""


@dataclass(frozen=True)
class Window:
    """A period and the one it is compared against."""

    period: str
    period_start: date
    period_end: date
    comparison_period: str
    comparison_start: date
    comparison_end: date


def resolve_window(
    period: str, grain: str, comparison_period: str | None = None
) -> Window:
    """The four dates, derived. Raises `PeriodError` on a label that does
    not match the grain."""
    start, end, label = _bounds(period, grain)
    if comparison_period:
        prior_start, prior_end, prior_label = _bounds(comparison_period, grain)
    else:
        prior_start, prior_end, prior_label = _previous(start, grain)
    return Window(
        period=label,
        period_start=start,
        period_end=end,
        comparison_period=prior_label,
        comparison_start=prior_start,
        comparison_end=prior_end,
    )


def _bounds(period: str, grain: str) -> tuple[date, date, str]:
    if grain == "monthly":
        match = MONTH_PATTERN.match(period)
        if not match:
            raise PeriodError(
                f"{period!r} is not a monthly period; monthly periods look like '2025-11'"
            )
        year, month = int(match.group(1)), int(match.group(2))
        if not 1 <= month <= 12:
            raise PeriodError(f"{period!r} names month {month}")
        last = calendar.monthrange(year, month)[1]
        return date(year, month, 1), date(year, month, last), period

    if grain == "daily":
        match = DAY_PATTERN.match(period)
        if not match:
            raise PeriodError(
                f"{period!r} is not a daily period; daily periods look like '2025-11-12'"
            )
        day = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return day, day, period

    if grain == "weekly":
        match = WEEK_PATTERN.match(period)
        if not match:
            raise PeriodError(
                f"{period!r} is not a weekly period; weekly periods look like '2025-W47'"
            )
        year, week = int(match.group(1)), int(match.group(2))
        try:
            start = date.fromisocalendar(year, week, MONDAY)
        except ValueError as exc:
            raise PeriodError(f"{period!r}: {exc}") from exc
        return start, start + timedelta(days=DAYS_IN_WEEK - 1), period

    raise PeriodError(f"unknown grain {grain!r}; expected daily, weekly or monthly")


def _previous(start: date, grain: str) -> tuple[date, date, str]:
    """The period before `start`, at the same grain."""
    if grain == "monthly":
        year, month = (start.year - 1, 12) if start.month == 1 else (start.year, start.month - 1)
        last = calendar.monthrange(year, month)[1]
        return date(year, month, 1), date(year, month, last), f"{year:04d}-{month:02d}"

    if grain == "daily":
        day = start - timedelta(days=1)
        return day, day, day.isoformat()

    if grain == "weekly":
        prior = start - timedelta(days=DAYS_IN_WEEK)
        iso = prior.isocalendar()
        return (
            prior,
            prior + timedelta(days=DAYS_IN_WEEK - 1),
            f"{iso.year:04d}-W{iso.week:02d}",
        )

    raise PeriodError(f"unknown grain {grain!r}")


__all__ = ["DAYS_IN_WEEK", "MONDAY", "PeriodError", "Window", "resolve_window"]
