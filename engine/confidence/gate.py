"""Score the case: six components, four caps, one isotonic map.

    conf_raw -> apply caps -> calibrate -> publish

Three numbers leave here and all three are shown. The raw score is what
the engine thought. The capped score is what it is allowed to think. The
published score is what the organisation's own track record says a score
like that has been worth. A system that showed only the last of those
would be asking to be trusted; showing all three is what makes it
checkable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import duckdb

from engine.adjudicate import AdjudicationResult, HypothesisVerdict
from engine.confidence.calibration import Calibration, fit_calibration
from engine.confidence.caps import Cap, apply_caps, evaluate_caps, forced_triggers
from engine.confidence.components import (
    Component,
    ScoringContext,
    SourceHealth,
    compute_components,
    weighted_sum,
)
from engine.contracts import CapApplied, ConfidenceBreakdown, ConfidenceComponent, Evidence
from engine.db import execute_governed, warehouse_clock
from engine.evidence import EvidenceFactory, EvidenceLedger
from security.policy import User
from semantic_layer.schema import SemanticLayer, get_semantic_layer

EVIDENCE_PREFIX = "confidence"
DERIVED_KIND = "derived_estimate"
DERIVED = "derived"



class ConfidenceError(RuntimeError):
    """The case cannot be scored as asked."""


@dataclass(frozen=True)
class Score:
    """The whole derivation, raw through published."""

    components: tuple[Component, ...]
    raw: float
    caps: tuple[Cap, ...]
    after_caps: float
    calibrated: float
    calibration: Calibration
    case_type: str
    forced_triggers: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    as_of: datetime

    @property
    def calibrating(self) -> bool:
        return self.calibration.calibrating

    @property
    def shift(self) -> float:
        """How far the map moved the score. Negative means it cooled it."""
        return self.calibrated - self.after_caps

    #: Per required source, its age in hours. What trigger T8 reads.
    source_staleness_hours: dict = field(default_factory=dict)

    def component(self, key: str) -> Component:
        for item in self.components:
            if item.key == key:
                return item
        raise KeyError(f"no component {key!r}")

    def render(self) -> str:
        lines = [
            f"confidence  raw {self.raw:.4f}"
            + (f" -> capped {self.after_caps:.4f}" if self.caps else "")
            + f" -> published {self.calibrated:.2f}"
        ]
        for item in self.components:
            lines.append(
                f"  {item.key} {item.name:<20} {item.value:.4f} "
                f"x {item.weight:.2f} = {item.contribution:.4f}   {item.detail}"
            )
        if not self.caps:
            lines.append("  no cap fired.")
        for cap in self.caps:
            lines.append(f"  CAP {cap.name} -> {cap.ceiling:.2f}: {cap.detail}")
        lines.append(
            f"  calibration: {self.shift:+.4f} from {self.calibration.scored_cases} "
            f"closed cases"
            + (" (CALIBRATING)" if self.calibrating else "")
        )
        return "\n".join(lines)


def score_case(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    adjudication: AdjudicationResult,
    *,
    leading_tag: str,
    case_type: str,
    evidence: tuple[Evidence, ...] = (),
    structured_present: frozenset[str] = frozenset(),
    unstructured_present: frozenset[str] = frozenset(),
    regime_change: bool = False,
    layer: SemanticLayer | None = None,
    clock: datetime | None = None,
) -> Score:
    """Score one adjudicated case."""
    layer = layer or get_semantic_layer()
    clock = clock or warehouse_clock(connection, layer)
    request = adjudication.request
    leading = adjudication.verdict(leading_tag)

    stores = _stores_in_scope(connection, user, layer, request)
    sources = source_health(
        connection, user, layer, request, leading_tag, clock=clock
    )
    context = ScoringContext(
        adjudication=adjudication,
        leading=leading,
        layer=layer,
        sources=sources,
        structured_present=frozenset(structured_present) or _exposed(leading),
        unstructured_present=frozenset(unstructured_present),
        stores_in_scope=stores,
        regime_change=regime_change,
    )

    components = compute_components(context)
    raw = weighted_sum(components)
    caps = evaluate_caps(leading, evidence, layer)
    after_caps = apply_caps(raw, caps)

    calibration = _calibration_for(connection, user, layer, case_type)
    calibrated = calibration.map(after_caps)

    factory = EvidenceFactory.for_stage(EVIDENCE_PREFIX, clock, layer)
    ledger = EvidenceLedger()
    _emit(factory, ledger, layer, components, raw, caps, after_caps, calibrated, calibration)

    return Score(
        components=components,
        raw=raw,
        caps=caps,
        after_caps=after_caps,
        calibrated=calibrated,
        calibration=calibration,
        case_type=case_type,
        forced_triggers=forced_triggers(caps),
        evidence=tuple(ledger),
        as_of=clock,
        source_staleness_hours={
            name: health.staleness_hours
            for name, health in sources.items()
            if health.held
        },
    )


def _calibration_for(connection, user, layer, case_type) -> Calibration:
    """The map is fitted over the WHOLE ledger, never per case type.

    A per-type map was built first and it was wrong. Isotonic regression
    needs range: it learns what a 0.65 has been worth and what a 0.95 has
    been worth, and interpolates between them. Slice the ledger by case
    type and each slice occupies a narrow band of raw confidence — the
    types the organisation is good at cluster high, the ones it is bad at
    cluster low — so the per-type fit has almost no range to learn over
    and clips everything outside its own cluster to an endpoint. On the
    first run of this module a 0.74 on an availability case came back as
    0.00, because every availability case in the ledger scored above 0.85
    and the map had nothing to say below that.

    Case type still matters, just not here. It is what trigger T7 reads,
    and a per-type ACCURACY is a rate over a slice rather than a map
    across a range, which a slice can support.
    """
    return fit_calibration(connection, user, layer=layer)


def _exposed(leading: HypothesisVerdict) -> frozenset[str]:
    """The structured lane's stores, from ADJUDICATE's own exposure."""
    if leading.exposure is None:
        return frozenset()
    return frozenset(leading.exposure.exposed)


def _stores_in_scope(connection, user, layer, request) -> frozenset[str]:
    rows, _filtered, _masked = execute_governed(
        user, request.kpi,
        "SELECT region, store_id FROM dim_store WHERE region = $scope",
        {"scope": request.scope},
        connection=connection, layer=layer, purpose="confidence.scope",
    )
    return frozenset(str(row["store_id"]) for row in rows)


def source_health(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
    request,
    leading_tag: str,
    *,
    clock: datetime,
) -> dict[str, SourceHealth]:
    """Freshness, completeness and history depth per required source.

    The sources are the ones the CAUSAL GRAPH says this hypothesis
    requires, not the ones the KPI contract lists: the question is what
    this argument rests on, and a hypothesis about stock-outs rests on the
    inventory feed whether or not net revenue does.

    A source the organisation does not hold scores zero on everything.
    That is not a missing measurement; it is a measurement of nothing.
    """
    template = layer.causal_graph.hypotheses.get(leading_tag)
    if template is None:
        return {}
    registry = layer.warehouse.sources
    contract = layer.kpis[request.kpi]
    sla_hours = contract.refresh.sla_hours
    zero_at = layer.adjudication.confidence.components["s4"].zero_at_sla_multiple
    reference_weeks = layer.adjudication.confidence.components["s5"].reference_weeks
    days_per_week = layer.warehouse.units.days_per_week

    health: dict[str, SourceHealth] = {}
    for name in template.required_sources:
        declared = layer.causal_graph.sources.get(name)
        if declared is not None and not declared.held:
            health[name] = SourceHealth(name, 0.0, 0.0, 0.0, held=False)
            continue

        location = registry.get(name)
        if location is None:
            health[name] = SourceHealth(name, 0.0, 0.0, 0.0, held=False)
            continue

        if location.date_column is None:
            # A static master has no arrival time, so it cannot be stale
            # and it cannot be short of history. Reported as static rather
            # than given a freshness of zero and called stale.
            health[name] = SourceHealth(name, 1.0, 1.0, float(reference_weeks))
            continue

        rows, _filtered, _masked = execute_governed(
            user, request.kpi,
            f"SELECT MIN({location.date_column}) AS first_day, "  # noqa: S608 - from the registry
            f"MAX({location.date_column}) AS last_day, "
            f"COUNT(*) AS rows_held FROM {location.table}",
            {},
            connection=connection, layer=layer,
            purpose=f"confidence.source_health.{name}",
        )
        row = rows[0] if rows else {}
        last_day, first_day = row.get("last_day"), row.get("first_day")
        if last_day is None or first_day is None:
            health[name] = SourceHealth(name, 0.0, 0.0, 0.0)
            continue

        hours = max(
            (clock - _as_datetime(last_day, clock)).total_seconds()
            / layer.warehouse.units.seconds_per_hour,
            0.0,
        )
        freshness = max(0.0, min(1.0, 1.0 - hours / (sla_hours * zero_at)))
        weeks = (_as_datetime(last_day, clock) - _as_datetime(first_day, clock)).days / days_per_week
        health[name] = SourceHealth(
            name,
            freshness,
            _completeness(connection, user, layer, request, location),
            weeks,
            staleness_hours=hours,
        )
    return health


def _completeness(connection, user, layer, request, location) -> float:
    """The share of the scope's days the source actually carries.

    Days, not rows. A source that holds every day but half the stores is a
    different failure from one that holds every store but skipped a week,
    and the second is the one that breaks a time series.
    """
    if location.scope_join not in {"region", "store_id"}:
        return 1.0
    rows, _filtered, _masked = execute_governed(
        user, request.kpi,
        f"SELECT COUNT(DISTINCT {location.date_column}) AS days_held "  # noqa: S608
        f"FROM {location.table} "
        f"WHERE {location.date_column} BETWEEN $start AND $end",
        {"start": request.period_start, "end": request.period_end},
        connection=connection, layer=layer, purpose="confidence.completeness",
    )
    held = int(rows[0]["days_held"]) if rows else 0
    expected = (request.period_end - request.period_start).days + 1
    return max(0.0, min(1.0, held / expected)) if expected else 1.0


def _as_datetime(value, clock):
    from datetime import datetime as _dt

    if isinstance(value, _dt):
        return value if value.tzinfo else value.replace(tzinfo=clock.tzinfo)
    return _dt(value.year, value.month, value.day, tzinfo=clock.tzinfo)


def _emit(factory, ledger, layer, components, raw, caps, after_caps, calibrated, calibration):
    places = layer.warehouse.units.crore_places
    for item in components:
        ledger.add(
            factory.emit(
                item.key,
                kind=DERIVED_KIND,
                label=item.name,
                value=round(item.value, places),
                unit="ratio",
                source_system=DERIVED,
                method="ratio",
                description=item.detail,
                ref=f"semantic_layer/adjudication.yaml::confidence.components.{item.key}",
                inputs=("adjudicate.coverage",),
                notes=f"weight {item.weight:.2f}; contributes {item.contribution:.4f}",
            )
        )
    ledger.add(
        factory.emit(
            "raw",
            kind=DERIVED_KIND,
            label="Confidence, weighted sum before caps",
            value=round(raw, places),
            unit="ratio",
            source_system=DERIVED,
            method="ratio",
            description=" + ".join(
                f"{item.weight:.2f}x{item.value:.4f}" for item in components
            ),
            ref="semantic_layer/adjudication.yaml::confidence.components",
            inputs=tuple(f"{EVIDENCE_PREFIX}.{item.key}" for item in components),
        )
    )
    if caps:
        ledger.add(
            factory.emit(
                "capped",
                kind=DERIVED_KIND,
                label="Confidence after caps",
                value=round(after_caps, places),
                unit="ratio",
                source_system=DERIVED,
                method="compare",
                description="; ".join(
                    f"{cap.name} at {cap.ceiling:.2f}" for cap in caps
                ),
                ref="semantic_layer/adjudication.yaml::confidence.caps",
                inputs=(f"{EVIDENCE_PREFIX}.raw",),
                notes="A cap is applied after the sum and cannot be outvoted.",
            )
        )
    ledger.add(
        factory.emit(
            "published",
            kind=DERIVED_KIND,
            label="Confidence as published",
            value=round(calibrated, places),
            unit="ratio",
            source_system=DERIVED,
            method="lookup",
            description=(
                f"isotonic map fitted to {calibration.scored_cases} closed cases; "
                f"moves {after_caps:.4f} to {calibrated:.4f}"
                + (" (CALIBRATING: too few cases, map is the identity)"
                   if calibration.calibrating else "")
            ),
            ref="engine/confidence/calibration.py::fit_calibration",
            inputs=(f"{EVIDENCE_PREFIX}.{'capped' if caps else 'raw'}",),
            notes=(
                "The organisation's own record on cases like this one. It is not a "
                "correction to the model; it is what a score like this has been worth."
            ),
        )
    )


def to_contract(score: Score, layer: SemanticLayer) -> ConfidenceBreakdown:
    """The score as the frozen contract the narrative layer receives."""
    specs = layer.adjudication.confidence.components
    return ConfidenceBreakdown(
        components=tuple(
            ConfidenceComponent(
                key=item.key,
                name=item.name,
                value=min(max(item.value, 0.0), 1.0),
                weight=specs[item.key].weight,
                detail=item.detail,
                evidence_ids=(f"{EVIDENCE_PREFIX}.{item.key}",),
            )
            for item in score.components
        ),
        raw=min(max(score.raw, 0.0), 1.0),
        caps_applied=tuple(
            CapApplied(
                name=cap.name,
                condition=cap.condition,
                ceiling=cap.ceiling,
                forced_trigger=cap.forced_trigger,
            )
            for cap in score.caps
        ),
        after_caps=min(max(score.after_caps, 0.0), 1.0),
        calibrated=min(max(score.calibrated, 0.0), 1.0),
        calibration_method=score.calibration.method,
        calibration_sample_size=score.calibration.scored_cases,
    )


__all__ = ["ConfidenceError", "Score", "score_case", "source_health", "to_contract"]
