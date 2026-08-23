"""Lane 3 — the external lane.

The calendar, the weather and the competitor news feed. Small, and it
matters twice.

It is where the confounders live. Test 6 screens promotions, weather and
competitor activity across the treated and control groups, and it cannot
screen what nobody fetched.

And it is where an absence is the finding. `ext_competitor_news` carries
a headline and a body and no price and no footfall, because the
organisation holds neither — which is what makes `competitor_action`
structurally unverifiable rather than merely unproven, and what fires
trigger T3 on case #2467. The lane reports the columns the feed does NOT
have, because that is the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from statistics import fmean

import duckdb

from engine.contracts import Evidence
from engine.db import GovernanceError, execute_governed, execute_metadata
from engine.evidence import EvidenceFactory
from engine.gather.guard import NO_GUARD, Guard
from security.policy import User
from semantic_layer.schema import ExternalFeed, SemanticLayer

QUERY_KIND = "structured_query"
NEWS_KIND = "news_item"

_AGGREGATES = {"mean": fmean, "sum": sum, "count": len}


@dataclass(frozen=True)
class FeedResult:
    """One external feed, and what it could and could not say."""

    feed: str
    available: bool
    reason: str
    rows: int = 0
    value: float | None = None
    unit: str | None = None
    absent_columns: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()


@dataclass
class ExternalResult:
    feeds: dict[str, FeedResult] = field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()


def gather_external(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
    factory: EvidenceFactory,
    *,
    kpi: str,
    scope: str,
    period_start: date,
    period_end: date,
    guard: Guard = NO_GUARD,
) -> ExternalResult:
    """Fetch every declared feed for the period."""
    spec = layer.gather.external
    result = ExternalResult()
    evidence: list[Evidence] = []

    for name, feed in sorted(spec.feeds.items()):
        found = _FETCHERS[name](
            connection, user, layer, factory, feed,
            kpi=kpi, scope=scope,
            period_start=period_start, period_end=period_end, guard=guard,
        )
        result.feeds[name] = found
        evidence.extend(found.evidence)

    result.evidence = tuple(evidence)
    return result


def _calendar(
    connection, user, layer, factory, feed: ExternalFeed, *,
    kpi: str, scope: str, period_start: date, period_end: date, guard: Guard = NO_GUARD,
) -> FeedResult:
    """Festival and promo windows over the period. Pipeline metadata."""
    statement = """
        SELECT
            COUNT(*)                                                  AS days,
            SUM(CASE WHEN is_festival THEN 1 ELSE 0 END)              AS festival_days,
            SUM(CASE WHEN is_promo_window THEN 1 ELSE 0 END)          AS promo_days
        FROM dim_calendar
        WHERE date BETWEEN $period_start AND $period_end
    """
    with guard():
        rows = execute_metadata(
            user, feed.table, statement,
            {"period_start": period_start, "period_end": period_end},
            connection=connection, layer=layer, purpose="gather.external.calendar",
        )
    row = rows[0]
    festival_days = int(row["festival_days"] or 0)
    promo_days = int(row["promo_days"] or 0)
    item = factory.emit(
        "external.calendar.festival_days",
        kind=QUERY_KIND,
        label=f"Festival days in the period — {scope}",
        value=festival_days,
        unit="count",
        source_system=feed.source_system,
        method="sql",
        description=f"days flagged is_festival between {period_start} and {period_end}",
        ref="semantic_layer/gather.yaml::external.feeds.calendar",
        inputs=(feed.table,),
        statement=statement,
        notes=(
            f"{promo_days} promotional days in the same window. Gate 2 has already "
            "removed the calendar from the residual; this is here so Test 6 can screen "
            "it as a confounder."
        ),
    )
    return FeedResult(
        feed="calendar", available=True,
        reason=f"{int(row['days'])} days, {festival_days} of them festival days",
        rows=int(row["days"]), value=festival_days, unit="count", evidence=(item,),
    )


def _weather(
    connection, user, layer, factory, feed: ExternalFeed, *,
    kpi: str, scope: str, period_start: date, period_end: date, guard: Guard = NO_GUARD,
) -> FeedResult:
    statement = """
        SELECT region, city, AVG(rain_mm) AS rain_mm, AVG(temp_c) AS temp_c
        FROM ext_weather_daily
        WHERE weather_date BETWEEN $period_start AND $period_end
          AND region = $scope
        GROUP BY 1, 2
    """
    try:
        with guard():
            rows, _filtered, _masked = execute_governed(
                user, kpi, statement,
                {"period_start": period_start, "period_end": period_end, "scope": scope},
                connection=connection, layer=layer, purpose="gather.external.weather",
            )
    except GovernanceError as exc:
        return FeedResult(
            feed="weather", available=False,
            reason=f"not readable at this persona's scope: {exc}",
        )
    values = [float(row["rain_mm"]) for row in rows if row.get("rain_mm") is not None]
    if not values:
        return FeedResult(
            feed="weather", available=True,
            reason=f"no weather rows for {scope} over the period", rows=len(rows),
        )
    value = round(
        _AGGREGATES[feed.aggregate or "mean"](values),
        layer.warehouse.units.crore_places,
    )
    item = factory.emit(
        "external.weather.rain_mm",
        kind=QUERY_KIND,
        label=f"Mean daily rainfall across {len(rows)} cities — {scope}",
        value=value,
        unit=feed.unit or "mm",
        source_system=feed.source_system,
        method="sql",
        description=f"mean rain_mm by city, {period_start} to {period_end}",
        ref="semantic_layer/gather.yaml::external.feeds.weather",
        inputs=(feed.table,),
        statement=statement,
        notes="Fetched so Test 6 can screen weather as a confounder, not to explain anything.",
    )
    return FeedResult(
        feed="weather", available=True, reason=f"{len(rows)} cities",
        rows=len(rows), value=value, unit=feed.unit, evidence=(item,),
    )


#: Columns a competitor feed would need in order to test the hypothesis
#: that depends on it. Their absence IS the finding.
_COMPETITOR_COLUMNS_NEEDED = ("price", "footfall")


def _competitor_news(
    connection, user, layer, factory, feed: ExternalFeed, *,
    kpi: str, scope: str, period_start: date, period_end: date, guard: Guard = NO_GUARD,
) -> FeedResult:
    """News items, and the two columns the feed does not have.

    A count of headlines is not evidence of a competitor's effect on this
    region's revenue, and the lane says so on the evidence rather than
    letting the count stand as if it were.
    """
    statement = """
        SELECT news_id, published_date, competitor, headline, source
        FROM ext_competitor_news
        WHERE published_date BETWEEN $period_start AND $period_end
    """
    # The news feed is a business feed with no store dimension, so it goes
    # through the governed path and a scoped persona simply cannot read it.
    try:
        with guard():
            rows, _filtered, _masked = execute_governed(
                user, kpi, statement,
                {"period_start": period_start, "period_end": period_end},
                connection=connection, layer=layer,
                purpose="gather.external.competitor_news",
            )
    except GovernanceError as exc:
        return FeedResult(
            feed="competitor_news", available=False,
            reason=(
                "the news feed carries no region, so a scoped persona cannot read it: "
                f"{exc}"
            ),
            absent_columns=_COMPETITOR_COLUMNS_NEEDED,
        )

    with guard():
        columns = {
            str(row[0])
            for row in connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'ext_competitor_news'"
            ).fetchall()
        }
    absent = tuple(
        name for name in _COMPETITOR_COLUMNS_NEEDED
        if not any(name in column for column in columns)
    )
    item = factory.emit(
        "external.competitor_news.items",
        kind=NEWS_KIND,
        label=f"Competitor news items published in the period — {scope}",
        value=len(rows),
        unit="count",
        source_system=feed.source_system,
        method="count",
        description=f"rows in ext_competitor_news between {period_start} and {period_end}",
        ref="semantic_layer/gather.yaml::external.feeds.competitor_news",
        inputs=(feed.table,),
        statement=statement,
        notes=(
            "The feed holds no "
            + " and no ".join(absent)
            + " column. A headline count cannot establish dose or timing at store "
            "level, which is why competitor_action is structurally unverifiable "
            "rather than merely unproven."
        ) if absent else "The feed carries every column the hypothesis would need.",
    )
    return FeedResult(
        feed="competitor_news", available=True,
        reason=f"{len(rows)} items, missing {list(absent)}",
        rows=len(rows), value=len(rows), unit="count",
        absent_columns=absent, evidence=(item,),
    )


_FETCHERS = {
    "calendar": _calendar,
    "weather": _weather,
    "competitor_news": _competitor_news,
}


__all__ = ["ExternalResult", "FeedResult", "gather_external"]
