"""The four disagreements, measured and named.

Real warehouses do not have one number for anything. Two teams publish two
revenue figures, two systems key on two different columns, one feed
arrives weekly and everything it is compared against is daily, and the
fiscal calendar does not line up with the ISO one. The usual response is
to pick a winner in the ETL and never mention the others.

This module does the opposite. It computes every figure, reports the gap,
and writes a row to `data_gap_register` for each problem it found. Nothing
is repaired: `engine/warehouse/load.py` loaded the raw rows as they came,
and a case that touches an affected scope cites the register rather than
working around it.

Everything it emits is an `Evidence` object (CLAUDE.md rule 3), so every
figure below is clickable through to the query that produced it. The
`flat_intraweek` allocation carries its assumption on the evidence itself
rather than in a footnote, because a footnote does not survive the trip to
the screen.

Rule 2: no thresholds, no unit conversions and no table names live here.
Every one of them is read off `semantic_layer/warehouse.yaml`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping

import duckdb

from engine.contracts import Evidence, LineageStep
from semantic_layer.schema import (
    CalendarMismatch,
    DefinitionConflict,
    EntityKeyMismatch,
    GrainMismatch,
    SemanticLayer,
    Units,
    get_semantic_layer,
)

#: Prefix for every evidence id this module mints.
EVIDENCE_PREFIX = "recon"

#: Where these numbers came from, for `Evidence.source_ref`.
SOURCE_REF = "engine/warehouse/reconcile.py"

#: A warehouse query is the top reliability tier; a figure computed from
#: two of them is a derived estimate. Both weights come from the semantic
#: layer, never from here.
QUERY_KIND = "structured_query"
DERIVED_KIND = "derived_estimate"


class ReconciliationError(RuntimeError):
    """The warehouse cannot be reconciled as loaded."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataGap:
    """One row of `data_gap_register`: a disagreement, not a failure."""

    gap_id: str
    detected_at: datetime
    gap_code: str
    severity: str
    subject: str
    scope: str | None
    measure: str
    value_numeric: float | None
    unit: str | None
    detail: str
    resolution_hint: str | None = None

    def as_row(self) -> tuple:
        return (
            self.gap_id,
            self.detected_at,
            self.gap_code,
            self.severity,
            self.subject,
            self.scope,
            self.measure,
            self.value_numeric,
            self.unit,
            self.detail,
            self.resolution_hint,
        )


@dataclass(frozen=True)
class DefinitionConflictResult:
    """Three revenue figures for one period, and the gap between two of them."""

    totals_inr_cr: Mapping[str, float]
    numerator: str
    denominator: str
    gap_pct: float
    excluded_transfers_inr_cr: float
    evidence: tuple[Evidence, ...]
    gap: DataGap

    @property
    def arbiter_inr_cr(self) -> float:
        return self.totals_inr_cr["contract"]


@dataclass(frozen=True)
class EntityKeyResult:
    """What the POS-to-operations join could not resolve."""

    unmapped_keys: tuple[str, ...]
    unmapped_rows: int
    total_rows: int
    quarantined_inr_cr: float
    total_inr_cr: float
    quarantined_share: float
    evidence: tuple[Evidence, ...]
    gap: DataGap

    @property
    def unmapped_key_count(self) -> int:
        return len(self.unmapped_keys)


@dataclass(frozen=True)
class GrainResult:
    """Weekly marketing spend, allocated to days, and what that assumed."""

    assumption_tag: str
    weeks: int
    allocated_days: int
    short_weeks: int
    weekly_total_inr_cr: float
    allocated_total_inr_cr: float
    evidence: tuple[Evidence, ...]
    gap: DataGap

    @property
    def reconciles(self) -> bool:
        """The allocation must move spend, never create or destroy it."""
        return math.isclose(self.weekly_total_inr_cr, self.allocated_total_inr_cr)


@dataclass(frozen=True)
class FestivalOverlap:
    """How one festival window falls across one kind of period.

    Keyed by OCCURRENCE, not just by name. Diwali happens once a fiscal
    year and moves eleven days earlier each time, so pooling both
    occurrences would report a window that spans three months and no such
    window exists.
    """

    festival: str
    occurrence: str
    period_type: str
    periods_spanned: int
    max_overlap_share: float


@dataclass(frozen=True)
class CalendarResult:
    """Where the fiscal calendar and the ISO calendar disagree."""

    compared: tuple[str, str]
    aligned_day_share: float
    overlaps: tuple[FestivalOverlap, ...]
    divergent_festivals: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    gap: DataGap

    def overlap(
        self, festival: str, period_type: str, occurrence: str | None = None
    ) -> FestivalOverlap | None:
        for row in self.overlaps:
            if row.festival != festival or row.period_type != period_type:
                continue
            if occurrence is None or row.occurrence == occurrence:
                return row
        return None


@dataclass(frozen=True)
class ReconciliationReport:
    """Everything the warehouse disagrees with itself about."""

    as_of: datetime
    definition_conflict: DefinitionConflictResult
    entity_key_mismatch: EntityKeyResult
    grain_mismatch: GrainResult
    calendar_mismatch: CalendarResult
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)

    @property
    def gaps(self) -> tuple[DataGap, ...]:
        return (
            self.definition_conflict.gap,
            self.entity_key_mismatch.gap,
            self.grain_mismatch.gap,
            self.calendar_mismatch.gap,
        )

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.evidence}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def reconcile(
    connection: duckdb.DuckDBPyConnection,
    *,
    layer: SemanticLayer | None = None,
    as_of: datetime | None = None,
) -> ReconciliationReport:
    """Run all four checks against the loaded warehouse."""
    layer = layer or get_semantic_layer()
    as_of = as_of or warehouse_clock(connection, layer)

    definition = check_definition_conflict(connection, layer, as_of)
    keys = check_entity_keys(connection, layer, as_of)
    grain = check_grain_allocation(connection, layer, as_of)
    calendar = check_calendar_alignment(connection, layer, as_of)

    evidence = (
        definition.evidence + keys.evidence + grain.evidence + calendar.evidence
    )
    return ReconciliationReport(
        as_of=as_of,
        definition_conflict=definition,
        entity_key_mismatch=keys,
        grain_mismatch=grain,
        calendar_mismatch=calendar,
        evidence=evidence,
    )


def write_gaps(
    connection: duckdb.DuckDBPyConnection, report: ReconciliationReport
) -> int:
    """Record the report in `data_gap_register`. Returns rows written."""
    rows = [gap.as_row() for gap in report.gaps]
    connection.executemany(GAP_INSERT, rows)
    return len(rows)


def warehouse_clock(
    connection: duckdb.DuckDBPyConnection, layer: SemanticLayer | None = None
) -> datetime:
    """The latest day the warehouse holds data for.

    Freshness is measured against this, not against the wall clock. The
    warehouse is a fixed 18-month extract; measuring its age against today
    would make every source look stale for a reason that says nothing
    about the data.
    """
    layer = layer or get_semantic_layer()
    calendar = layer.warehouse.reconciliation.calendar_mismatch
    value = connection.execute(
        f'SELECT MAX("{calendar.date_column}") FROM "{calendar.calendar_table}"'
    ).fetchone()[0]
    if value is None:
        raise ReconciliationError(
            f"{calendar.calendar_table} is empty; load the warehouse before reconciling"
        )
    return datetime(value.year, value.month, value.day, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 1. Definition conflict
# ---------------------------------------------------------------------------


def check_definition_conflict(
    connection: duckdb.DuckDBPyConnection,
    layer: SemanticLayer,
    as_of: datetime,
    *,
    scope: str | None = None,
) -> DefinitionConflictResult:
    """Compute every rival revenue figure and the gap between two of them.

    The expressions are the semantic layer's, not this module's. Changing
    what a team means by "revenue" is a change to `warehouse.yaml`, and
    nothing here has to be edited to follow it.
    """
    spec: DefinitionConflict = layer.warehouse.reconciliation.definition_conflict
    units = layer.warehouse.units

    names = sorted(spec.definitions)
    selected = ", ".join(
        f'{spec.definitions[name].expression} AS "{name}"' for name in names
    )
    where, params = _scope_filter(scope)
    statement = (
        f"SELECT {selected}, "
        f'SUM("{spec.shared_exclusion.column}") AS "__excluded" '
        f'FROM "{spec.fact_table}"{where}'
    )
    row = _one(connection, statement, params)

    totals = {name: _crore(float(row[name]), units) for name in names}
    excluded = _crore(float(row["__excluded"]), units)

    numerator = totals[spec.gap.numerator]
    denominator = totals[spec.gap.denominator]
    if not denominator:
        raise ReconciliationError(
            f"the {spec.gap.denominator!r} revenue definition totals zero; the gap is undefined"
        )
    gap_pct = (numerator - denominator) / denominator * units.percent_scale

    reliability = layer.reliability_weight(QUERY_KIND)
    derived_reliability = layer.reliability_weight(DERIVED_KIND)
    scope_label = scope or "All-India"

    evidence: list[Evidence] = []
    for name in names:
        definition = spec.definitions[name]
        evidence.append(
            Evidence(
                evidence_id=_eid("definition", name, scope),
                kind=QUERY_KIND,
                produced_by="code",
                label=f"{definition.label} — {scope_label}",
                value=totals[name],
                unit="INR_CR",
                reliability=reliability,
                source_ref=f"{SOURCE_REF}::check_definition_conflict",
                as_of=as_of,
                freshness_hours=0.0,
                completeness=1.0,
                lineage=(
                    LineageStep(
                        step=0,
                        operation="sql",
                        description=(
                            f"{definition.expression} over {spec.fact_table}"
                            f" — owned by {definition.owner}"
                        ),
                        inputs=(spec.fact_table,),
                        ref=f"semantic_layer/warehouse.yaml::definitions.{name}",
                    ),
                ),
                notes=definition.rationale.strip(),
            )
        )

    evidence.append(
        Evidence(
            evidence_id=_eid("definition", "gap", scope),
            kind=DERIVED_KIND,
            produced_by="code",
            label=(
                f"{spec.definitions[spec.gap.numerator].label} exceeds "
                f"{spec.definitions[spec.gap.denominator].label} — {scope_label}"
            ),
            value=round(gap_pct, units.pct_places),
            unit=spec.gap.unit,
            reliability=derived_reliability,
            source_ref=f"{SOURCE_REF}::check_definition_conflict",
            as_of=as_of,
            freshness_hours=0.0,
            completeness=1.0,
            lineage=(
                LineageStep(
                    step=0,
                    operation="ratio",
                    description=(
                        f"({spec.gap.numerator} - {spec.gap.denominator}) / "
                        f"{spec.gap.denominator}"
                    ),
                    inputs=(
                        _eid("definition", spec.gap.numerator, scope),
                        _eid("definition", spec.gap.denominator, scope),
                    ),
                    ref="semantic_layer/warehouse.yaml::definition_conflict.gap",
                ),
            ),
            notes=(
                f"Neither figure is wrong. {spec.definitions[spec.arbiter].label} is the "
                "arbiter because the KPI contract says so; the others are recorded, "
                "not corrected."
            ),
        )
    )

    gap = DataGap(
        gap_id=_gid(spec.gap_code, scope),
        detected_at=as_of,
        gap_code=spec.gap_code,
        severity=spec.severity,
        subject=spec.fact_table,
        scope=scope,
        measure="definition_gap_pct",
        value_numeric=round(gap_pct, units.pct_places),
        unit=spec.gap.unit,
        detail=(
            f"{len(names)} revenue definitions computed over {spec.fact_table}. "
            f"{spec.definitions[spec.gap.numerator].label} and "
            f"{spec.definitions[spec.gap.denominator].label} differ by "
            f"{gap_pct:.2f}{spec.gap.unit}. Arbiter: {spec.definitions[spec.arbiter].label}."
        ),
        resolution_hint=(
            f"Quote {spec.arbiter} on the case and show the gap. Do not average them."
        ),
    )

    return DefinitionConflictResult(
        totals_inr_cr=totals,
        numerator=spec.gap.numerator,
        denominator=spec.gap.denominator,
        gap_pct=gap_pct,
        excluded_transfers_inr_cr=excluded,
        evidence=tuple(evidence),
        gap=gap,
    )


# ---------------------------------------------------------------------------
# 2. Entity key mismatch
# ---------------------------------------------------------------------------


def check_entity_keys(
    connection: duckdb.DuckDBPyConnection, layer: SemanticLayer, as_of: datetime
) -> EntityKeyResult:
    """Count what the POS key could not be resolved to an operations key.

    The unmapped rows are QUARANTINED — held, counted and reported. They
    are not dropped, and they are not rescued by joining on the surrogate
    key that happens to sit beside them: a silent fallback is how a broken
    integration survives for years.
    """
    spec: EntityKeyMismatch = layer.warehouse.reconciliation.entity_key_mismatch
    units = layer.warehouse.units

    row = _one(
        connection,
        f"""
        SELECT
            COUNT(*)                                          AS total_rows,
            SUM(CASE WHEN x."{spec.ops_key}" IS NULL THEN 1 ELSE 0 END) AS unmapped_rows,
            SUM(f."{spec.measure}")                           AS total_measure,
            SUM(CASE WHEN x."{spec.ops_key}" IS NULL
                     THEN f."{spec.measure}" ELSE 0 END)      AS unmapped_measure
        FROM "{spec.fact_table}" AS f
        LEFT JOIN "{spec.bridge_table}" AS x
               ON f."{spec.pos_key}" = x."{spec.pos_key}"
        """,
    )

    keys = tuple(
        str(record[0])
        for record in connection.execute(
            f"""
            SELECT DISTINCT f."{spec.pos_key}"
            FROM "{spec.fact_table}" AS f
            LEFT JOIN "{spec.bridge_table}" AS x
                   ON f."{spec.pos_key}" = x."{spec.pos_key}"
            WHERE x."{spec.ops_key}" IS NULL
            ORDER BY 1
            """
        ).fetchall()
    )

    total_rows = int(row["total_rows"])
    unmapped_rows = int(row["unmapped_rows"])
    total = _crore(float(row["total_measure"]), units)
    quarantined = _crore(float(row["unmapped_measure"]), units)
    share = quarantined / total if total else 0.0

    reliability = layer.reliability_weight(QUERY_KIND)
    completeness = (total_rows - unmapped_rows) / total_rows if total_rows else 0.0
    common = dict(
        kind=QUERY_KIND,
        produced_by="code",
        reliability=reliability,
        source_ref=f"{SOURCE_REF}::check_entity_keys",
        as_of=as_of,
        freshness_hours=0.0,
        completeness=completeness,
    )
    lineage = (
        LineageStep(
            step=0,
            operation="sql",
            description=(
                f"LEFT JOIN {spec.fact_table} to {spec.bridge_table} on {spec.pos_key}; "
                f"rows with no {spec.ops_key} are quarantined"
            ),
            inputs=(spec.fact_table, spec.bridge_table),
            ref="engine/warehouse/views.sql::v_sales_daily_quarantined",
        ),
    )

    evidence = (
        Evidence(
            evidence_id=_eid("keys", "unmapped_count"),
            label=f"POS {spec.pos_key} values with no row in {spec.bridge_table}",
            value=len(keys),
            unit="count",
            lineage=lineage,
            notes=f"Unmapped keys: {', '.join(keys) if keys else 'none'}",
            **common,
        ),
        Evidence(
            evidence_id=_eid("keys", "quarantined_revenue"),
            label="Revenue held in quarantine, unresolvable to an operations key",
            value=quarantined,
            unit="INR_CR",
            lineage=lineage,
            **common,
        ),
        Evidence(
            evidence_id=_eid("keys", "quarantined_share"),
            label="Quarantined revenue as a share of all revenue",
            value=_pct(share, units),
            unit="pct",
            lineage=lineage,
            **common,
        ),
    )

    gap = DataGap(
        gap_id=_gid(spec.gap_code),
        detected_at=as_of,
        gap_code=spec.gap_code,
        severity=spec.severity,
        subject=spec.fact_table,
        scope=None,
        measure="quarantined_revenue_share",
        value_numeric=_pct(share, units),
        unit="pct",
        detail=(
            f"{len(keys)} {spec.pos_key} values and {unmapped_rows} rows do not resolve "
            f"through {spec.bridge_table}, holding "
            f"{quarantined:.2f} INR Cr ({share * units.percent_scale:.2f}%) of revenue. "
            f"Policy: {spec.policy}."
        ),
        resolution_hint=(
            f"Ask store operations to issue {spec.bridge_table} rows for the re-fasciad "
            "stores. Until then, exclude the quarantined revenue and say so."
        ),
    )

    return EntityKeyResult(
        unmapped_keys=keys,
        unmapped_rows=unmapped_rows,
        total_rows=total_rows,
        quarantined_inr_cr=quarantined,
        total_inr_cr=total,
        quarantined_share=share,
        evidence=evidence,
        gap=gap,
    )


# ---------------------------------------------------------------------------
# 3. Grain mismatch
# ---------------------------------------------------------------------------


def check_grain_allocation(
    connection: duckdb.DuckDBPyConnection, layer: SemanticLayer, as_of: datetime
) -> GrainResult:
    """Allocate weekly spend to days, and tag what that assumed.

    The allocation is flat, which is almost certainly wrong: weekend spend
    exceeds Tuesday spend. It is used anyway because there is no daily
    feed, and every value derived from it carries `flat_intraweek` on the
    evidence so a reader can see which figures rest on it.
    """
    spec: GrainMismatch = layer.warehouse.reconciliation.grain_mismatch
    units = layer.warehouse.units

    weekly = _one(
        connection,
        f'SELECT COUNT(*) AS weeks, SUM(spend_inr) AS total FROM "{spec.source_table}"',
    )
    allocated = _one(
        connection,
        """
        SELECT COUNT(*) AS days,
               SUM(allocated_spend_inr) AS total,
               COUNT(DISTINCT CASE WHEN days_in_week <> $nominal
                                   THEN (region, campaign, iso_year, iso_week) END) AS short
        FROM v_marketing_spend_daily
        """,
        {"nominal": spec.days_per_week},
    )

    weeks = int(weekly["weeks"])
    weekly_total = _crore(float(weekly["total"]), units)
    allocated_total = _crore(float(allocated["total"]), units)
    allocated_days = int(allocated["days"])
    short_weeks = int(allocated["short"])

    evidence = (
        Evidence(
            evidence_id=_eid("grain", "allocated_spend"),
            kind=DERIVED_KIND,
            produced_by="code",
            label=(
                f"Marketing spend allocated from {spec.source_grain} to "
                f"{spec.target_grain} grain"
            ),
            value=allocated_total,
            unit="INR_CR",
            reliability=layer.reliability_weight(DERIVED_KIND),
            source_ref=f"{SOURCE_REF}::check_grain_allocation",
            as_of=as_of,
            freshness_hours=0.0,
            completeness=1.0,
            lineage=(
                LineageStep(
                    step=0,
                    operation="allocate",
                    description=(
                        f"{spec.method}: each week's spend divided evenly across the days "
                        "of that ISO week present in dim_calendar"
                    ),
                    inputs=(spec.source_table, "dim_calendar"),
                    ref="engine/warehouse/views.sql::v_marketing_spend_daily",
                ),
            ),
            assumptions=(spec.assumption_tag,),
            notes=spec.note.strip(),
        ),
        Evidence(
            evidence_id=_eid("grain", "short_weeks"),
            kind=QUERY_KIND,
            produced_by="code",
            label=(
                f"Weeks allocated across fewer than {spec.days_per_week} days "
                "(series edges)"
            ),
            value=short_weeks,
            unit="count",
            reliability=layer.reliability_weight(QUERY_KIND),
            source_ref=f"{SOURCE_REF}::check_grain_allocation",
            as_of=as_of,
            freshness_hours=0.0,
            completeness=1.0,
            lineage=(
                LineageStep(
                    step=0,
                    operation="sql",
                    description=(
                        "count of region/campaign/ISO-week groups whose day count differs "
                        f"from the nominal {spec.days_per_week}"
                    ),
                    inputs=("v_marketing_spend_daily",),
                    ref="engine/warehouse/views.sql::v_marketing_spend_daily",
                ),
            ),
            assumptions=(spec.assumption_tag,),
        ),
    )

    gap = DataGap(
        gap_id=_gid(spec.gap_code),
        detected_at=as_of,
        gap_code=spec.gap_code,
        severity=spec.severity,
        subject=spec.source_table,
        scope=None,
        measure="allocated_days",
        value_numeric=float(allocated_days),
        unit="count",
        detail=(
            f"{weeks} {spec.source_grain} rows allocated to {allocated_days} "
            f"{spec.target_grain} rows by {spec.method}. {short_weeks} of those weeks are "
            f"shorter than {spec.days_per_week} days at the series edges. Every derived "
            f"value is tagged '{spec.assumption_tag}'."
        ),
        resolution_hint=(
            "Request a daily spend feed. Until it exists the intra-week shape is assumed, "
            "not measured."
        ),
    )

    return GrainResult(
        assumption_tag=spec.assumption_tag,
        weeks=weeks,
        allocated_days=allocated_days,
        short_weeks=short_weeks,
        weekly_total_inr_cr=weekly_total,
        allocated_total_inr_cr=allocated_total,
        evidence=evidence,
        gap=gap,
    )


# ---------------------------------------------------------------------------
# 4. Calendar mismatch
# ---------------------------------------------------------------------------


def check_calendar_alignment(
    connection: duckdb.DuckDBPyConnection, layer: SemanticLayer, as_of: datetime
) -> CalendarResult:
    """Measure where the fiscal and ISO calendars put the same festival.

    For each festival and each kind of period, this computes how many
    periods the festival window spans and the largest share of any single
    period it covers. A festival that fills one fiscal week but splits
    across two ISO weeks is a festival whose trade lands in a different
    place depending on which calendar the report was run on.
    """
    spec: CalendarMismatch = layer.warehouse.reconciliation.calendar_mismatch
    units = layer.warehouse.units
    left, right = spec.compare

    aligned = _one(
        connection,
        """
        SELECT AVG(CASE WHEN same_week_start THEN 1.0 ELSE 0.0 END) AS aligned
        FROM v_calendar_alignment
        """,
    )
    aligned_share = float(aligned["aligned"])

    overlaps: list[FestivalOverlap] = []
    for period_type, period in spec.periods.items():
        rows = connection.execute(
            f"""
            WITH per_period AS (
                SELECT
                    "{spec.festival_column}" AS festival,
                    fiscal_year_label        AS occurrence,
                    "{period.column}"        AS period_key,
                    COUNT(*)                 AS festival_days
                FROM "{spec.calendar_table}"
                WHERE "{spec.festival_flag_column}"
                GROUP BY 1, 2, 3
            ),
            period_size AS (
                SELECT "{period.column}" AS period_key, COUNT(*) AS period_days
                FROM "{spec.calendar_table}"
                GROUP BY 1
            )
            SELECT
                p.festival,
                p.occurrence,
                COUNT(*)                                   AS periods_spanned,
                MAX(p.festival_days * 1.0 / s.period_days)  AS max_overlap_share
            FROM per_period AS p
            JOIN period_size AS s USING (period_key)
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).fetchall()
        for festival, occurrence, spanned, share in rows:
            overlaps.append(
                FestivalOverlap(
                    festival=str(festival),
                    occurrence=str(occurrence),
                    period_type=period_type,
                    periods_spanned=int(spanned),
                    max_overlap_share=float(share),
                )
            )

    # A festival "diverges" when the two calendars cut its window into a
    # different number of periods — which is to say, when a report run on
    # one calendar splits its trade and a report run on the other does not.
    spans: dict[tuple[str, str], dict[str, int]] = {}
    for row in overlaps:
        spans.setdefault((row.festival, row.occurrence), {})[row.period_type] = (
            row.periods_spanned
        )
    divergent = tuple(
        sorted(
            f"{festival} {occurrence}"
            for (festival, occurrence), by_period in spans.items()
            if by_period.get(left) != by_period.get(right)
        )
    )

    reliability = layer.reliability_weight(QUERY_KIND)
    common = dict(
        kind=QUERY_KIND,
        produced_by="code",
        reliability=reliability,
        source_ref=f"{SOURCE_REF}::check_calendar_alignment",
        as_of=as_of,
        freshness_hours=0.0,
        completeness=1.0,
        lineage=(
            LineageStep(
                step=0,
                operation="sql",
                description=(
                    f"festival window days per period, for {sorted(spec.periods)}, "
                    f"over {spec.calendar_table}"
                ),
                inputs=(spec.calendar_table,),
                ref="engine/warehouse/views.sql::v_calendar_alignment",
            ),
        ),
    )

    evidence = (
        Evidence(
            evidence_id=_eid("calendar", "aligned_days"),
            label=(
                f"Days on which the {spec.periods[left].label} and the "
                f"{spec.periods[right].label} begin together"
            ),
            value=_pct(aligned_share, units),
            unit="pct",
            notes=(
                "The fiscal week counts from the start of the fiscal year whatever "
                "weekday that is; the ISO week always starts on a Monday."
            ),
            **common,
        ),
        Evidence(
            evidence_id=_eid("calendar", "divergent_festivals"),
            label=(
                f"Festivals spanning a different number of {spec.periods[left].label}s "
                f"than {spec.periods[right].label}s"
            ),
            value=len(divergent),
            unit="count",
            notes=f"Divergent: {', '.join(divergent) if divergent else 'none'}",
            **common,
        ),
    )

    gap = DataGap(
        gap_id=_gid(spec.gap_code),
        detected_at=as_of,
        gap_code=spec.gap_code,
        severity=spec.severity,
        subject=spec.calendar_table,
        scope=None,
        measure="aligned_day_share_pct",
        value_numeric=_pct(aligned_share, units),
        unit="pct",
        detail=(
            f"{spec.periods[left].label} and {spec.periods[right].label} begin on the same "
            f"day for {aligned_share * units.percent_scale:.2f}% of days. "
            f"{len(divergent)} festival windows span a different number of periods under "
            f"the two calendars: {', '.join(divergent) if divergent else 'none'}."
        ),
        resolution_hint=(
            "State which calendar a period-over-period comparison was run on. A festival "
            "moves between periods when the calendar changes, and the movement is not real."
        ),
    )

    return CalendarResult(
        compared=(left, right),
        aligned_day_share=aligned_share,
        overlaps=tuple(overlaps),
        divergent_festivals=divergent,
        evidence=evidence,
        gap=gap,
    )


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

GAP_INSERT = (
    "INSERT INTO data_gap_register (gap_id, detected_at, gap_code, severity, subject, "
    "scope, measure, value_numeric, unit, detail, resolution_hint) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _crore(value: float, units: Units) -> float:
    """Quote a rupee figure in crore, at the semantic layer's precision."""
    return round(value / units.inr_per_crore, units.crore_places)


def _pct(share: float, units: Units) -> float:
    """Quote a ratio as a percentage, at the semantic layer's precision."""
    return round(share * units.percent_scale, units.pct_places)


def _one(
    connection: duckdb.DuckDBPyConnection,
    statement: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cursor = connection.execute(statement, dict(params)) if params else connection.execute(
        statement
    )
    columns = [description[0] for description in cursor.description]
    row = cursor.fetchone()
    if row is None:  # pragma: no cover - aggregates always return a row
        raise ReconciliationError(f"query returned no rows: {statement}")
    return dict(zip(columns, row, strict=True))


def _scope_filter(scope: str | None) -> tuple[str, dict[str, Any]]:
    if scope is None:
        return "", {}
    return " WHERE region = $scope", {"scope": scope}


def _eid(*parts: str | None) -> str:
    return ".".join([EVIDENCE_PREFIX, *[part for part in parts if part]])


def _gid(*parts: str | None) -> str:
    return "-".join([part for part in parts if part])


__all__ = [
    "CalendarResult",
    "DataGap",
    "DefinitionConflictResult",
    "EntityKeyResult",
    "FestivalOverlap",
    "GrainResult",
    "ReconciliationError",
    "ReconciliationReport",
    "check_calendar_alignment",
    "check_definition_conflict",
    "check_entity_keys",
    "check_grain_allocation",
    "reconcile",
    "warehouse_clock",
    "write_gaps",
]
