"""Gate 1's five checks.

Each one answers a different question about whether the movement in front
of us is a movement at all, and each returns a NAMED EXIT rather than a
boolean. That distinction is the whole design. A gate that returns False
tells an analyst to go and look; a gate that returns DATA_INCIDENT and
names twenty-five stores tells them what happened.

  1  source_freshness              a feed is outside its SLA
  2  row_count_delta               stores loaded far fewer rows than usual
  3  definition_drift              the KPI means something else than it did
  4  single_transaction_dominance  one bill is carrying the movement
  5  restatement_pending           finance has reopened the period

Case #2470 is why this stage exists: a -18.0% daily movement in West that
never happened, because 25 of 140 store feeds did not load. Check 2
catches it, names the stores, and no case is opened. Case #2451 passes all
five and goes on to QUALIFY.

Every read goes through `engine/db.py` — `execute_governed` for anything
holding a measure, `execute_metadata` for the two pipeline tables that
hold none (CLAUDE.md rule 5). Every figure leaves as an `Evidence` object
(rule 3), and no threshold is written here (rule 2): they all come from
`semantic_layer/validate.yaml` and the KPI contract.
"""

from __future__ import annotations

import calendar as calendar_module
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Callable, Mapping

import duckdb

from engine.contracts import CheckResult, Evidence, Grain
from engine.db import (
    GovernanceError,
    as_stored_timestamp,
    execute_governed,
    execute_metadata,
    warehouse_clock,
)
from engine.evidence import EvidenceFactory
from security.policy import User
from semantic_layer.schema import KpiContract, SemanticLayer, get_semantic_layer

#: Prefix for every evidence id this stage mints.
EVIDENCE_PREFIX = "validate"

SOURCE_REF = "engine/validate/checks.py"

QUERY_KIND = "structured_query"
DERIVED_KIND = "derived_estimate"

#: `Evidence.source_system` values this stage uses. Declared in
#: semantic_layer/warehouse.yaml; named here so a typo is an ImportError.
POS_SOURCE = "pos"
RETURNS_SOURCE = "returns"
SEMANTIC_LAYER = "semantic_layer"
WAREHOUSE = "warehouse"
DERIVED = "derived"

#: Seconds in an hour, without writing 3600 (CLAUDE.md rule 2 admits no
#: numbers in engine/, and this is arithmetic rather than policy).
_ONE_HOUR_SECONDS = timedelta(hours=1).total_seconds()

#: One day less than a week. Same reason.
_WEEK_MINUS_A_DAY = timedelta(weeks=1) - timedelta(days=1)

#: Scope value meaning "the whole estate" — no region filter.
NATIONAL_SCOPE = "All-India"

_HASHES: Mapping[str, Callable[[bytes], Any]] = {"sha256": hashlib.sha256}


class ValidationError(RuntimeError):
    """The gate could not be evaluated as asked."""


# ---------------------------------------------------------------------------
# Request and period arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A closed date range: both ends are inside it."""

    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def __str__(self) -> str:
        return f"{self.start.isoformat()}..{self.end.isoformat()}"


@dataclass(frozen=True)
class ValidationRequest:
    """What is being validated, before anything has looked at it."""

    kpi: str
    scope: str
    grain: Grain
    period: str
    comparison_period: str | None = None

    @property
    def is_national(self) -> bool:
        return self.scope == NATIONAL_SCOPE


def period_window(period: str, grain: Grain) -> Window:
    """Resolve a period label to a closed date range.

    Monthly is `YYYY-MM`; daily and weekly are `YYYY-MM-DD`, weekly being
    the first day of the week. Computed rather than read from
    `dim_calendar` so the boundaries of a period do not depend on the
    warehouse holding that period.
    """
    parts = period.split("-")
    if grain == "monthly":
        try:
            year, month = parts
        except ValueError as exc:
            raise ValidationError(
                f"monthly period {period!r} is not YYYY-MM"
            ) from exc
        start = date.fromisoformat(f"{year}-{month}-01")
        _, length = calendar_module.monthrange(start.year, start.month)
        return Window(start, start.replace(day=length))

    try:
        start = date.fromisoformat(period)
    except ValueError as exc:
        raise ValidationError(
            f"{grain} period {period!r} is not YYYY-MM-DD"
        ) from exc
    if grain == "weekly":
        return Window(start, start + _WEEK_MINUS_A_DAY)
    return Window(start, start)


def baseline_window(period: Window, lookback_weeks: int) -> Window:
    """The lookback immediately before the period, not overlapping it."""
    end = period.start - timedelta(days=1)
    return Window(end - timedelta(weeks=lookback_weeks) + timedelta(days=1), end)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class ValidationContext:
    """Everything the five checks share, resolved once."""

    connection: duckdb.DuckDBPyConnection
    user: User
    request: ValidationRequest
    layer: SemanticLayer
    clock: datetime
    period: Window
    baseline: Window
    comparison: Window | None
    factory: EvidenceFactory

    @property
    def kpi(self) -> KpiContract:
        return self.layer.kpis[self.request.kpi]

    @property
    def spec(self):
        return self.layer.validation.checks

    def governed(
        self, sql: str, params: Mapping[str, Any] | None = None, *, purpose: str
    ) -> tuple[dict[str, Any], ...]:
        rows, _filtered, _masked = execute_governed(
            self.user,
            self.request.kpi,
            sql,
            params,
            connection=self.connection,
            layer=self.layer,
            purpose=purpose,
        )
        return rows

    def metadata(
        self, table: str, sql: str, params: Mapping[str, Any] | None = None, *, purpose: str
    ) -> tuple[dict[str, Any], ...]:
        return execute_metadata(
            self.user,
            table,
            sql,
            params,
            connection=self.connection,
            layer=self.layer,
            purpose=purpose,
        )

    def scope_clause(self, column: str = "region") -> str:
        """`AND region = $scope`, or nothing at all for a national scope."""
        return "" if self.request.is_national else f" AND {column} = $scope"

    def scope_params(self) -> dict[str, Any]:
        return {} if self.request.is_national else {"scope": self.request.scope}


def build_context(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    request: ValidationRequest,
    *,
    layer: SemanticLayer | None = None,
    clock: datetime | None = None,
) -> ValidationContext:
    layer = layer or get_semantic_layer()
    if request.kpi not in layer.kpis:
        raise ValidationError(
            f"no KPI contract named {request.kpi!r}; known: {sorted(layer.kpis)}"
        )
    if request.grain not in layer.kpis[request.kpi].grain.supported:
        raise ValidationError(
            f"{request.kpi!r} is not computed at {request.grain} grain "
            f"(supported: {layer.kpis[request.kpi].grain.supported})"
        )

    period = period_window(request.period, request.grain)
    lookback = layer.validation.checks.row_count_delta.lookback_weeks
    comparison = (
        period_window(request.comparison_period, request.grain)
        if request.comparison_period
        else None
    )
    resolved_clock = clock or warehouse_clock(connection, layer)
    return ValidationContext(
        connection=connection,
        user=user,
        request=request,
        layer=layer,
        clock=resolved_clock,
        period=period,
        baseline=baseline_window(period, lookback),
        comparison=comparison,
        # The stage's data timestamp is the warehouse clock, fixed once for
        # the whole run so every figure agrees about how current the data
        # was. It is NOT `now()`, which the factory records separately.
        factory=EvidenceFactory.for_stage(EVIDENCE_PREFIX, resolved_clock, layer),
    )


# ---------------------------------------------------------------------------
# Evidence helper
# ---------------------------------------------------------------------------


def _evidence(
    ctx: ValidationContext,
    check_id: str,
    name: str,
    *,
    label: str,
    value: float | int | str | None,
    unit: str,
    source_system: str,
    kind: str = QUERY_KIND,
    operation: str = "sql",
    description: str,
    inputs: tuple[str, ...] = (),
    ref: str,
    statement: str | None = None,
    completeness: float = 1.0,
    freshness_hours: float = 0.0,
    notes: str | None = None,
    assumptions: tuple[str, ...] = (),
) -> Evidence:
    """Mint one record through the evidence engine.

    Nothing here chooses a reliability weight, an id prefix or a
    timestamp — `engine/evidence.py` does, and it refuses anything it
    cannot make chaseable.
    """
    return ctx.factory.emit(
        f"{check_id}.{name}",
        kind=kind,
        label=label,
        value=value,
        unit=unit,
        source_system=source_system,
        method=operation,
        description=description,
        ref=ref,
        inputs=inputs,
        statement=statement,
        freshness_hours=freshness_hours,
        completeness=completeness,
        assumptions=assumptions,
        notes=notes,
    )


@dataclass
class _Outcome:
    """What a check produced, before it becomes a contract object."""

    outcome: str
    detail: str
    subjects: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)


def _result(
    check_id: str, spec, produced: _Outcome
) -> tuple[CheckResult, tuple[Evidence, ...]]:
    """Wrap a check's finding in the contract the gate reports."""
    return (
        CheckResult(
            check_id=check_id,
            name=spec.name,
            order=spec.order,
            outcome=produced.outcome,
            detail=produced.detail,
            subjects=produced.subjects,
            evidence_ids=tuple(item.evidence_id for item in produced.evidence),
        ),
        produced.evidence,
    )


# ---------------------------------------------------------------------------
# 1. Source freshness against SLA
# ---------------------------------------------------------------------------


def check_source_freshness(ctx: ValidationContext) -> tuple[CheckResult, tuple[Evidence, ...]]:
    """Is every source the KPI depends on inside its refresh SLA?

    The limit is the KPI contract's own `refresh.sla_hours` — different
    KPIs tolerate different lag, and the contract is where its owner said
    so. Staleness is measured against the warehouse clock rather than
    today's date: a fixed 18-month extract is not stale for being an
    extract.

    A static master has no arrival time and is reported as static, not as
    fresh. A source with no store dimension cannot be read by a scoped
    persona, and that is reported as unverified rather than assumed clean.
    """
    spec = ctx.spec.source_freshness
    kpi = ctx.kpi
    registry = ctx.layer.warehouse.sources
    sla_hours = kpi.refresh.sla_hours * spec.sla_multiplier

    wanted = (
        set(kpi.confidence_rules.required_sources)
        if spec.scope == "required_sources"
        else {system.name for system in kpi.source_systems}
    )
    names = sorted(name for name in wanted)

    evidence: list[Evidence] = []
    stale: list[str] = []
    unverified: list[str] = []
    static: list[str] = []

    for name in names:
        location = registry[name]
        if location.is_static:
            static.append(name)
            continue

        last_loaded, statement = _source_last_loaded(ctx, name, location)
        if last_loaded is None:
            unverified.append(name)
            continue

        # A returns ledger posts credits after the sale, so its newest row
        # can be dated later than the clock. That is early, not stale.
        elapsed = (ctx.clock - last_loaded).total_seconds() / _ONE_HOUR_SECONDS
        hours = max(elapsed, 0.0)
        if hours > sla_hours:
            stale.append(name)

        evidence.append(
            _evidence(
                ctx,
                "source_freshness",
                name,
                label=f"{name} feed staleness against a {sla_hours:g}h SLA",
                value=round(hours, ctx.layer.warehouse.units.pct_places),
                unit="hours",
                source_system=name,
                description=(
                    f"MAX({location.date_column}) over {location.table}, against the "
                    "warehouse clock"
                ),
                inputs=(location.table,),
                ref=f"semantic_layer/warehouse.yaml::sources.{name}",
                statement=statement,
                freshness_hours=hours,
                notes=location.description.strip(),
            )
        )

    if stale:
        return _result(
            "source_freshness",
            spec,
            _Outcome(
                outcome=spec.exit_code,
                detail=(
                    f"{len(stale)} of {len(names)} required sources are outside the "
                    f"{sla_hours:g}h SLA for {kpi.kpi}: {', '.join(stale)}."
                ),
                subjects=tuple(stale),
                evidence=tuple(evidence),
            ),
        )

    parts = [f"{len(evidence)} dated source(s) inside the {sla_hours:g}h SLA"]
    if static:
        parts.append(f"{len(static)} static master(s) ({', '.join(static)})")
    if unverified:
        parts.append(
            f"{len(unverified)} not readable at this scope ({', '.join(unverified)})"
        )
    return _result(
        "source_freshness",
        spec,
        _Outcome(
            outcome="CLEAN",
            detail="; ".join(parts) + ".",
            subjects=tuple(static + unverified),
            evidence=tuple(evidence),
        ),
    )


def _source_last_loaded(
    ctx: ValidationContext, name: str, location
) -> tuple[datetime | None, str | None]:
    """Newest row in a source, and the statement that found it.

    The statement comes back so it can be kept VERBATIM on the evidence:
    a reader who doubts a staleness figure should be able to run the query
    that produced it, not a description of it.
    """
    if location.scope_join == "region":
        sql = (
            f'SELECT region, store_id, MAX("{location.date_column}") AS last_loaded '
            f'FROM "{location.table}" GROUP BY 1, 2'
        )
    elif location.scope_join == "store_id":
        sql = (
            f'SELECT d.region, s.store_id, MAX(s."{location.date_column}") AS last_loaded '
            f'FROM "{location.table}" AS s JOIN dim_store AS d USING (store_id) '
            "GROUP BY 1, 2"
        )
    else:
        # National source with no store dimension. A scoped persona's row
        # predicate has nothing to bind to, which is a refusal, not a pass.
        sql = f'SELECT MAX("{location.date_column}") AS last_loaded FROM "{location.table}"'

    try:
        rows = ctx.governed(sql, purpose=f"validate.source_freshness.{name}")
    except GovernanceError:
        return None, sql

    values = [row["last_loaded"] for row in rows if row.get("last_loaded") is not None]
    if not values:
        return None, sql
    newest = max(values)
    if isinstance(newest, datetime):
        return (newest.replace(tzinfo=UTC) if newest.tzinfo is None else newest), sql
    return datetime(newest.year, newest.month, newest.day, tzinfo=UTC), sql


# ---------------------------------------------------------------------------
# 2. Row-count delta per store against the 8-week median
# ---------------------------------------------------------------------------


def check_row_count_delta(ctx: ValidationContext) -> tuple[CheckResult, tuple[Evidence, ...]]:
    """Did every store's feed carry its usual number of rows?

    Per store: the mean rows loaded per day over the period, against the
    median of the same statistic over the preceding lookback. A median,
    not a mean, because a quiet Tuesday is not a signal and a zero is.

    This is what catches #2470, and the reason it catches it at daily
    grain and not at monthly is arithmetic: a store that loses one day in
    thirty moves its mean by three percent; a store that loses its only
    day moves it by a hundred.
    """
    spec = ctx.spec.row_count_delta
    units = ctx.layer.warehouse.units

    statement = f"""
        SELECT
            region,
            store_id,
            AVG(CASE WHEN feed_date BETWEEN $p_start AND $p_end
                     THEN "{spec.measure}" END)                       AS period_mean,
            COUNT(CASE WHEN feed_date BETWEEN $p_start AND $p_end
                       THEN 1 END)                                     AS period_days,
            MEDIAN(CASE WHEN feed_date BETWEEN $b_start AND $b_end
                        THEN "{spec.measure}" END)                     AS baseline,
            COUNT(CASE WHEN feed_date BETWEEN $b_start AND $b_end
                       THEN 1 END)                                     AS baseline_days,
            SUM(CASE WHEN feed_date BETWEEN $p_start AND $p_end
                          AND status <> $loaded THEN 1 ELSE 0 END)     AS failed_days
        FROM "{spec.source_table}"
        WHERE feed_date BETWEEN $b_start AND $p_end{ctx.scope_clause()}
        GROUP BY 1, 2
    """
    rows = ctx.governed(
        statement,
        {
            "p_start": ctx.period.start,
            "p_end": ctx.period.end,
            "b_start": ctx.baseline.start,
            "b_end": ctx.baseline.end,
            "loaded": spec.loaded_status,
            **ctx.scope_params(),
        },
        purpose="validate.row_count_delta",
    )

    if not rows:
        return _result(
            "row_count_delta",
            spec,
            _Outcome(
                outcome="INSUFFICIENT_BASELINE",
                detail=(
                    f"{spec.source_table} holds no rows for {ctx.request.scope} over "
                    f"{ctx.baseline}..{ctx.period.end.isoformat()}."
                ),
            ),
        )

    baseline_days = max((int(row["baseline_days"] or 0) for row in rows), default=0)
    if baseline_days < spec.min_lookback_days:
        return _result(
            "row_count_delta",
            spec,
            _Outcome(
                outcome="INSUFFICIENT_BASELINE",
                detail=(
                    f"only {baseline_days} days of feed history before {ctx.period.start}, "
                    f"and {spec.min_lookback_days} are needed for a {spec.statistic} worth "
                    "comparing against."
                ),
            ),
        )

    breached: list[str] = []
    failed_feeds: list[str] = []
    for row in rows:
        baseline = float(row["baseline"] or 0.0)
        observed = float(row["period_mean"] or 0.0)
        if int(row["failed_days"] or 0):
            failed_feeds.append(str(row["store_id"]))
        if not baseline:
            continue
        if (baseline - observed) / baseline > spec.max_store_shortfall:
            breached.append(str(row["store_id"]))

    breached.sort()
    failed_feeds.sort()
    store_count = len(rows)
    share = len(breached) / store_count if store_count else 0.0
    revenue_share = _affected_revenue_share(ctx, breached)

    evidence = (
        _evidence(
            ctx,
            "row_count_delta",
            "stores_checked",
            source_system=POS_SOURCE,
            statement=statement,
            label=f"Stores with feed history in {ctx.request.scope}",
            value=store_count,
            unit="count",
            description=(
                f"{spec.statistic} of daily {spec.measure} over {ctx.baseline}, per store"
            ),
            inputs=(spec.source_table,),
            ref="semantic_layer/validate.yaml::checks.row_count_delta",
        ),
        _evidence(
            ctx,
            "row_count_delta",
            "stores_short",
            source_system=POS_SOURCE,
            statement=statement,
            label=(
                f"Stores loading more than {spec.max_store_shortfall:.0%} below their "
                f"own {spec.lookback_weeks}-week median"
            ),
            value=len(breached),
            unit="count",
            description=(
                f"period mean over {ctx.period} against the {spec.statistic} over "
                f"{ctx.baseline}"
            ),
            inputs=(spec.source_table,),
            ref="semantic_layer/validate.yaml::checks.row_count_delta",
            notes=f"Stores: {', '.join(breached) if breached else 'none'}",
        ),
        _evidence(
            ctx,
            "row_count_delta",
            "store_share",
            source_system=DERIVED,
            label="Share of the scope's stores whose feed fell short",
            value=round(share * units.percent_scale, units.pct_places),
            unit="pct",
            kind=DERIVED_KIND,
            operation="ratio",
            description="short stores divided by stores with feed history",
            inputs=(
                f"{EVIDENCE_PREFIX}.row_count_delta.stores_short",
                f"{EVIDENCE_PREFIX}.row_count_delta.stores_checked",
            ),
            ref="semantic_layer/validate.yaml::checks.row_count_delta",
        ),
        _evidence(
            ctx,
            "row_count_delta",
            "revenue_share",
            source_system=DERIVED,
            label="Share of the scope's revenue sitting behind those feeds",
            value=round(revenue_share * units.percent_scale, units.pct_places),
            unit="pct",
            kind=DERIVED_KIND,
            operation="ratio",
            description=(
                "net revenue of the affected stores over the period, divided by the "
                "scope's net revenue over the period"
            ),
            inputs=("fact_sales_daily", f"{EVIDENCE_PREFIX}.row_count_delta.stores_short"),
            ref="semantic_layer/validate.yaml::checks.row_count_delta",
            notes=(
                "This is the size of the hole in the headline: the movement is computed "
                "without these stores."
            ),
        ),
    )

    if share > spec.max_affected_store_share:
        return _result(
            "row_count_delta",
            spec,
            _Outcome(
                outcome=spec.exit_code,
                detail=(
                    f"{len(breached)} of {store_count} {ctx.request.scope} store feeds loaded "
                    f"below half their {spec.lookback_weeks}-week median over {ctx.period} "
                    f"({share * units.percent_scale:.1f}% of stores, "
                    f"{revenue_share * units.percent_scale:.1f}% of revenue). "
                    f"{len(failed_feeds)} reported FAILED."
                ),
                subjects=tuple(breached),
                evidence=evidence,
            ),
        )

    return _result(
        "row_count_delta",
        spec,
        _Outcome(
            outcome="CLEAN",
            detail=(
                f"{len(breached)} of {store_count} {ctx.request.scope} store feeds fell short, "
                f"{share * units.percent_scale:.1f}% of the scope, within the "
                f"{spec.max_affected_store_share:.0%} limit."
            ),
            subjects=tuple(breached),
            evidence=evidence,
        ),
    )


def _affected_revenue_share(ctx: ValidationContext, store_ids: list[str]) -> float:
    """What share of the scope's period revenue sits behind these stores."""
    if not store_ids:
        return 0.0
    rows = ctx.governed(
        f"""
        SELECT region, store_id, SUM(net_revenue_inr) AS revenue
        FROM fact_sales_daily
        WHERE txn_date BETWEEN $p_start AND $p_end{ctx.scope_clause()}
        GROUP BY 1, 2
    """,
        {"p_start": ctx.period.start, "p_end": ctx.period.end, **ctx.scope_params()},
        purpose="validate.row_count_delta.revenue_share",
    )
    affected = set(store_ids)
    total = sum(float(row["revenue"] or 0.0) for row in rows)
    held = sum(float(row["revenue"] or 0.0) for row in rows if row["store_id"] in affected)
    return held / total if total else 0.0


# ---------------------------------------------------------------------------
# 3. Definition drift against the last run
# ---------------------------------------------------------------------------


def definition_hash(kpi: KpiContract, fields: list[str], algorithm: str) -> str:
    """Fingerprint of what the KPI MEANS.

    Only the fields that decide the meaning are hashed, so reformatting a
    comment does not raise a false alarm and changing a scope exclusion
    always does.
    """
    try:
        digest = _HASHES[algorithm]
    except KeyError as exc:
        raise ValidationError(f"unknown hash algorithm {algorithm!r}") from exc
    payload = "\n".join(f"{name}={getattr(kpi, name)!r}" for name in fields)
    return digest(payload.encode("utf-8")).hexdigest()


def check_definition_drift(ctx: ValidationContext) -> tuple[CheckResult, tuple[Evidence, ...]]:
    """Has the KPI's definition changed since the last run?

    A movement computed under a new definition is not comparable with the
    period before it, and the difference looks exactly like a real change.
    The first run of a KPI records its hash and passes: there is nothing
    to have drifted from.
    """
    spec = ctx.spec.definition_drift
    kpi = ctx.kpi
    current = definition_hash(kpi, spec.hashed_fields, spec.algorithm)

    previous = ctx.metadata(
        spec.log_table,
        f"""
        SELECT formula_hash, kpi_version, hashed_fields, first_seen_at, last_seen_at
        FROM "{spec.log_table}"
        WHERE kpi = $kpi
        ORDER BY last_seen_at DESC
        """,
        {"kpi": kpi.kpi},
        purpose="validate.definition_drift",
    )
    _record_definition(ctx, spec, kpi, current)

    evidence = (
        _evidence(
            ctx,
            "definition_drift",
            "current_hash",
            source_system=SEMANTIC_LAYER,
            label=f"{kpi.kpi} definition fingerprint",
            value=current,
            unit="sha256",
            kind=DERIVED_KIND,
            operation="hash",
            description=(
                f"{spec.algorithm} over {', '.join(spec.hashed_fields)} of the KPI contract"
            ),
            inputs=(f"semantic_layer/kpis/{kpi.kpi}.yaml",),
            ref="semantic_layer/validate.yaml::checks.definition_drift",
            notes=(
                "Only the fields that decide what the number MEANS are hashed. "
                "Reformatting a comment does not raise this; changing a scope "
                "exclusion always does."
            ),
        ),
    )

    if not previous:
        return _result(
            "definition_drift",
            spec,
            _Outcome(
                outcome="CLEAN",
                detail=(
                    f"first run for {kpi.kpi} v{kpi.version}; fingerprint recorded, "
                    "nothing to have drifted from."
                ),
                evidence=evidence,
            ),
        )

    latest = previous[0]
    if str(latest["formula_hash"]) == current:
        return _result(
            "definition_drift",
            spec,
            _Outcome(
                outcome="CLEAN",
                detail=(
                    f"{kpi.kpi} v{kpi.version} matches the definition in force since "
                    f"{latest['first_seen_at']}."
                ),
                evidence=evidence,
            ),
        )

    return _result(
        "definition_drift",
        spec,
        _Outcome(
            outcome=spec.exit_code,
            detail=(
                f"{kpi.kpi} changed definition since the last run: "
                f"v{latest['kpi_version']} {_short(latest['formula_hash'], spec)} became "
                f"v{kpi.version} {_short(current, spec)}. Movements across the change are "
                "not comparable."
            ),
            subjects=(str(latest["formula_hash"]), current),
            evidence=evidence,
        ),
    )


def _short(digest: object, spec) -> str:
    """Enough fingerprint to tell two definitions apart by eye."""
    text = str(digest)
    return f"{text[: spec.display_hash_chars]}\u2026"


def _record_definition(ctx: ValidationContext, spec, kpi: KpiContract, digest: str) -> None:
    """Upsert the fingerprint. A write, so it does not go through a reader."""
    ctx.connection.execute(
        f"""
        INSERT INTO "{spec.log_table}"
            (kpi, formula_hash, kpi_version, hashed_fields, first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (kpi, formula_hash) DO UPDATE SET last_seen_at = EXCLUDED.last_seen_at
        """,
        [
            kpi.kpi,
            digest,
            kpi.version,
            ",".join(spec.hashed_fields),
            as_stored_timestamp(ctx.clock),
            as_stored_timestamp(ctx.clock),
        ],
    )


# ---------------------------------------------------------------------------
# 4. Single-transaction dominance
# ---------------------------------------------------------------------------


def check_single_transaction_dominance(
    ctx: ValidationContext,
) -> tuple[CheckResult, tuple[Evidence, ...]]:
    """Is one transaction carrying the movement?

    Top bill divided by the movement, CLAUDE.md's own framing. Above the
    limit, the movement is one order and a case would end up investigating
    a customer rather than the business.

    `fact_bill_lines` is a deterministic 2% sample, so the largest bill
    OBSERVED is a lower bound on the largest bill that exists. The check
    therefore under-fires rather than over-fires, and its evidence carries
    the sample rate as completeness so nobody reads a clean result as
    proof there was no such bill.
    """
    spec = ctx.spec.single_transaction_dominance
    units = ctx.layer.warehouse.units

    movement, basis, movement_sql = _movement_inr(ctx)
    if abs(movement) < spec.min_movement_inr:
        return _result(
            "single_transaction_dominance",
            spec,
            _Outcome(
                outcome="CLEAN",
                detail=(
                    f"movement of {abs(movement) / units.inr_per_crore:.4f} INR Cr is below "
                    "the floor at which a single-transaction ratio means anything; "
                    "materiality is Gate 2's question, not this one."
                ),
            ),
        )

    statement = f"""
        SELECT
            region,
            store_id,
            arg_max(bill_id, bill_net) AS top_bill_id,
            MAX(bill_net)              AS top_bill_inr,
            MAX(sample_rate)           AS sample_rate
        FROM (
            SELECT
                d.region,
                store_id,
                "{spec.transaction_key}"  AS bill_id,
                MAX(sample_rate)          AS sample_rate,
                {spec.amount_expression}  AS bill_net
            FROM "{spec.source_table}" JOIN dim_store AS d USING (store_id)
            WHERE txn_ts >= $p_start AND txn_ts < $p_next{ctx.scope_clause("d.region")}
            GROUP BY 1, 2, 3
        )
        GROUP BY 1, 2
    """
    rows = ctx.governed(
        statement,
        {
            "p_start": datetime.combine(ctx.period.start, datetime.min.time()),
            "p_next": datetime.combine(
                ctx.period.end + timedelta(days=1), datetime.min.time()
            ),
            **ctx.scope_params(),
        },
        purpose="validate.single_transaction_dominance",
    )

    if not rows:
        return _result(
            "single_transaction_dominance",
            spec,
            _Outcome(
                outcome="CLEAN",
                detail=(
                    f"no sampled bill lines for {ctx.request.scope} over {ctx.period}; "
                    "no transaction can be shown to dominate."
                ),
            ),
        )

    top = max(rows, key=lambda row: float(row["top_bill_inr"] or 0.0))
    top_inr = float(top["top_bill_inr"] or 0.0)
    sample_rate = float(top["sample_rate"] or 1.0)
    share = top_inr / abs(movement)

    evidence = (
        _evidence(
            ctx,
            "single_transaction_dominance",
            "movement",
            source_system=POS_SOURCE,
            statement=movement_sql,
            label=f"Movement being validated ({basis})",
            value=round(movement / units.inr_per_crore, units.crore_places),
            unit="INR_CR",
            description=basis,
            inputs=("fact_sales_daily",),
            ref="semantic_layer/validate.yaml::checks.single_transaction_dominance",
        ),
        _evidence(
            ctx,
            "single_transaction_dominance",
            "top_transaction",
            source_system=RETURNS_SOURCE,
            statement=statement,
            label=f"Largest single bill in {ctx.request.scope} over the period",
            value=round(top_inr / units.inr_per_lakh, units.crore_places),
            unit="INR_L",
            description=(
                f"{spec.amount_expression} grouped by {spec.transaction_key} over "
                f"{spec.source_table}"
            ),
            inputs=(spec.source_table,),
            ref="semantic_layer/validate.yaml::checks.single_transaction_dominance",
            completeness=sample_rate,
            notes=(
                f"Bill {top['top_bill_id']} at store {top['store_id']}. "
                f"{spec.source_table} is a {sample_rate:.0%} sample, so this is a LOWER "
                "BOUND on the largest bill that exists: the check under-fires."
            ),
        ),
        _evidence(
            ctx,
            "single_transaction_dominance",
            "share_of_movement",
            source_system=DERIVED,
            label="Largest bill as a share of the movement",
            value=round(share * units.percent_scale, units.pct_places),
            unit="pct",
            kind=DERIVED_KIND,
            operation="ratio",
            description="top bill divided by the absolute movement",
            inputs=(
                f"{EVIDENCE_PREFIX}.single_transaction_dominance.top_transaction",
                f"{EVIDENCE_PREFIX}.single_transaction_dominance.movement",
            ),
            ref="semantic_layer/validate.yaml::checks.single_transaction_dominance",
            completeness=sample_rate,
        ),
    )

    if share > spec.max_top_transaction_share:
        return _result(
            "single_transaction_dominance",
            spec,
            _Outcome(
                outcome=spec.exit_code,
                detail=(
                    f"one bill ({top['top_bill_id']}, store {top['store_id']}) carries "
                    f"{share * units.percent_scale:.1f}% of the movement, above the "
                    f"{spec.max_top_transaction_share:.0%} limit. This is an order, not a trend."
                ),
                subjects=(str(top["top_bill_id"]), str(top["store_id"])),
                evidence=evidence,
            ),
        )

    return _result(
        "single_transaction_dominance",
        spec,
        _Outcome(
            outcome="CLEAN",
            detail=(
                f"largest single bill is {share * units.percent_scale:.1f}% of the movement, "
                f"under the {spec.max_top_transaction_share:.0%} limit."
            ),
            evidence=evidence,
        ),
    )


def _movement_inr(ctx: ValidationContext) -> tuple[float, str, str]:
    """The movement in rupees, and one line saying what it was measured against.

    With a comparison period, it is period-over-period. Without one, it is
    the period against its own calendar expectation — which is what a
    daily monitor computes, and is the only comparison available for a
    single day that has no obvious predecessor.
    """
    comparison = ctx.comparison
    lo = min(ctx.period.start, comparison.start) if comparison else ctx.period.start
    hi = max(ctx.period.end, comparison.end) if comparison else ctx.period.end

    statement = f"""
        SELECT
            region,
            store_id,
            SUM(CASE WHEN txn_date BETWEEN $p_start AND $p_end
                     THEN net_revenue_inr ELSE 0 END)       AS period_inr,
            SUM(CASE WHEN txn_date BETWEEN $p_start AND $p_end
                     THEN calendar_expected_inr ELSE 0 END) AS expected_inr,
            SUM(CASE WHEN txn_date BETWEEN $c_start AND $c_end
                     THEN net_revenue_inr ELSE 0 END)       AS comparison_inr
        FROM fact_sales_daily
        WHERE txn_date BETWEEN $lo AND $hi{ctx.scope_clause()}
        GROUP BY 1, 2
    """
    rows = ctx.governed(
        statement,
        {
            "p_start": ctx.period.start,
            "p_end": ctx.period.end,
            "c_start": comparison.start if comparison else ctx.period.start,
            "c_end": comparison.end if comparison else ctx.period.end,
            "lo": lo,
            "hi": hi,
            **ctx.scope_params(),
        },
        purpose="validate.movement",
    )

    period_total = sum(float(row["period_inr"] or 0.0) for row in rows)
    if comparison:
        against = sum(float(row["comparison_inr"] or 0.0) for row in rows)
        basis = f"{ctx.request.period} against {ctx.request.comparison_period}"
    else:
        against = sum(float(row["expected_inr"] or 0.0) for row in rows)
        basis = f"{ctx.request.period} against its calendar expectation"
    return period_total - against, basis, statement


# ---------------------------------------------------------------------------
# 5. Restatement flagged within the window
# ---------------------------------------------------------------------------


def check_restatement_pending(
    ctx: ValidationContext,
) -> tuple[CheckResult, tuple[Evidence, ...]]:
    """Has finance reopened this period?

    A restatement is DECLARED, not inferred. Late-arriving returns make a
    young period incomplete, and incompleteness is not a restatement:
    somebody has to say the published figure is wrong and will be
    reissued. So this reads the register, and a period nobody has flagged
    passes however young its data is.
    """
    spec = ctx.spec.restatement_pending
    kpi = ctx.kpi
    window_days = kpi.refresh.restatement_window_days

    statement = f"""
        SELECT restatement_id, scope, period, flagged_at, status, reason
        FROM "{spec.register_table}"
        WHERE kpi = $kpi
          AND period = $period
          AND status = $open
          AND scope IN ($scope, $wildcard)
        ORDER BY flagged_at
    """
    rows = ctx.metadata(
        spec.register_table,
        statement,
        {
            "kpi": kpi.kpi,
            "period": ctx.request.period,
            "open": spec.open_status,
            "scope": ctx.request.scope,
            "wildcard": spec.scope_wildcard,
        },
        purpose="validate.restatement_pending",
    )

    evidence = (
        _evidence(
            ctx,
            "restatement_pending",
            "open_flags",
            source_system=WAREHOUSE,
            statement=statement,
            label=f"Open restatement flags covering {ctx.request.scope} {ctx.request.period}",
            value=len(rows),
            unit="count",
            description=(
                f"{spec.register_table} filtered to kpi, period, status={spec.open_status} "
                f"and scope in (this scope, {spec.scope_wildcard})"
            ),
            inputs=(spec.register_table,),
            ref="semantic_layer/validate.yaml::checks.restatement_pending",
            notes=(
                f"The contract's restatement window is {window_days} day(s). A flag is a "
                "declaration by finance, not an inference from how young the data is."
            ),
        ),
    )

    if not rows:
        return _result(
            "restatement_pending",
            spec,
            _Outcome(
                outcome="CLEAN",
                detail=(
                    f"no open restatement covers {kpi.kpi} / {ctx.request.scope} / "
                    f"{ctx.request.period}."
                ),
                evidence=evidence,
            ),
        )

    ids = tuple(str(row["restatement_id"]) for row in rows)
    first = rows[0]
    return _result(
        "restatement_pending",
        spec,
        _Outcome(
            outcome=spec.exit_code,
            detail=(
                f"{len(rows)} open restatement flag(s) cover this period "
                f"({', '.join(ids)}); {first['scope']} {first['period']} was flagged on "
                f"{first['flagged_at']} and the figure will be reissued."
            ),
            subjects=ids,
            evidence=evidence,
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CHECKS: tuple[tuple[str, Callable[[ValidationContext], Any]], ...] = (
    ("source_freshness", check_source_freshness),
    ("row_count_delta", check_row_count_delta),
    ("definition_drift", check_definition_drift),
    ("single_transaction_dominance", check_single_transaction_dominance),
    ("restatement_pending", check_restatement_pending),
)


__all__ = [
    "CHECKS",
    "NATIONAL_SCOPE",
    "ValidationContext",
    "ValidationError",
    "ValidationRequest",
    "Window",
    "baseline_window",
    "build_context",
    "check_definition_drift",
    "check_restatement_pending",
    "check_row_count_delta",
    "check_single_transaction_dominance",
    "check_source_freshness",
    "definition_hash",
    "period_window",
]
