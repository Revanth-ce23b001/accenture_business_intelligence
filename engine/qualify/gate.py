"""QUALIFY — run Gates 2 to 5, then the restraint, and decide.

VALIDATE asked whether the movement is real. QUALIFY asks whether it is
worth explaining, and stops at the first gate that says no. Unlike Gate 1,
these gates are SEQUENTIAL: there is no point measuring a residual against
a band when there is no history to build a band from, and no point asking
whether a movement is material before knowing what the residual is.

Every gate emits `Evidence` (CLAUDE.md rule 3) and returns a named exit
(never a boolean), and the whole stage returns the untouched series
alongside its decomposition — decompose, do not adjust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import duckdb
from engine.contracts import CheckResult, Evidence, GateResult
from engine.db import warehouse_clock
from engine.evidence import EvidenceFactory, EvidenceLedger
from engine.qualify.band import (
    BandError,
    HistoryCheck,
    ResidualBand,
    check_history,
    period_end,
    residual_band,
)
from engine.qualify.history import observed_period_count
from engine.qualify.calendar import (
    CalendarDecomposition,
    CalendarError,
    CalendarFit,
    decompose,
    fit_calendar,
    previous_month,
    trading_days,
)
from engine.qualify.materiality import MaterialityVerdict, assess_materiality
from engine.qualify.restraint import RestraintVerdict, assess_restraint
from engine.qualify.series import RegionalSeries, SeriesError, load_series
from engine.qualify.specificity import SpecificityVerdict, assess_specificity
from security.policy import User
from semantic_layer.schema import SemanticLayer, get_semantic_layer

EVIDENCE_PREFIX = "qualify"
SOURCE_REF = "engine/qualify"
QUERY_KIND = "structured_query"
DERIVED_KIND = "derived_estimate"

#: `Evidence.source_system` values this stage uses, declared in
#: semantic_layer/warehouse.yaml. Named here so a typo is an ImportError
#: rather than a string nobody can query.
POS_SOURCE = "pos"
SEMANTIC_LAYER = "semantic_layer"
DERIVED = "derived"

class QualifyError(RuntimeError):
    """QUALIFY could not be run as asked."""


@dataclass(frozen=True)
class QualifyRequest:
    """What is being qualified. Mirrors the VALIDATE request."""

    kpi: str
    scope: str
    grain: str
    period: str
    comparison_period: str | None = None

    def resolved_comparison(self) -> str:
        if self.comparison_period:
            return self.comparison_period
        if self.grain == "monthly":
            return previous_month(self.period)
        raise QualifyError(
            f"{self.grain} grain has no default comparison period; give one explicitly"
        )


@dataclass(frozen=True)
class QualifyResult:
    """Everything QUALIFY decided, and everything it looked at."""

    request: QualifyRequest
    gates: tuple[GateResult, ...]
    checks: tuple[CheckResult, ...]
    evidence: tuple[Evidence, ...]
    as_of: datetime
    decomposition: CalendarDecomposition | None = None
    history: HistoryCheck | None = None
    band: ResidualBand | None = None
    specificity: SpecificityVerdict | None = None
    materiality: MaterialityVerdict | None = None
    restraint: RestraintVerdict | None = None
    peer_decompositions: dict[str, CalendarDecomposition] = field(default_factory=dict)

    @property
    def case_opened(self) -> bool:
        """All four gates clean, and restraint allowed it.

        Restraint is not a gate. Gates 2 to 5 ask whether the movement
        deserves a case; restraint asks whether opening one more case is
        useful to the person who would have to read it. A movement can
        pass every gate and still be a duplicate.
        """
        if not all(gate.passed for gate in self.gates):
            return False
        return self.restraint is None or self.restraint.allowed

    @property
    def outcome_code(self) -> str | None:
        for gate in self.gates:
            if not gate.passed:
                return gate.outcome_code
        if self.restraint is not None and not self.restraint.allowed:
            return self.restraint.outcome
        return None

    @property
    def region_strip(self) -> dict[str, float]:
        return {
            region: decomposition.residual_pt
            for region, decomposition in sorted(self.peer_decompositions.items())
        }

    def gate(self, gate_id: int) -> GateResult:
        for candidate in self.gates:
            if candidate.gate_id == gate_id:
                return candidate
        raise KeyError(f"QUALIFY ran no gate {gate_id}")

    def check(self, check_id: str) -> CheckResult:
        for candidate in self.checks:
            if candidate.check_id == check_id:
                return candidate
        raise KeyError(f"QUALIFY ran no check called {check_id!r}")

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.evidence}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def qualify(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    request: QualifyRequest,
    *,
    layer: SemanticLayer | None = None,
    clock: datetime | None = None,
    series: RegionalSeries | None = None,
) -> QualifyResult:
    """Run Gates 2 to 5 and the restraint, stopping at the first refusal."""
    layer = layer or get_semantic_layer()
    clock = clock or warehouse_clock(connection, layer)
    spec = layer.qualify
    kpi = layer.kpis.get(request.kpi)
    if kpi is None:
        raise QualifyError(
            f"no KPI contract named {request.kpi!r}; known: {sorted(layer.kpis)}"
        )

    builder = _Builder(request, kpi, layer, clock)

    # --- Gate 3a: history, before anything is fitted -----------------------
    observed = _observed_periods(connection, user, request, layer)
    history = check_history(observed, kpi, request.grain)
    if not history.sufficient:
        builder.history_gate(history)
        return builder.finish(history=history)

    if series is None:
        try:
            series = load_series(connection, user, request.kpi, layer)
        except SeriesError as exc:
            raise QualifyError(str(exc)) from exc

    comparison = request.resolved_comparison()

    # --- Gate 2: calendar ---------------------------------------------------
    try:
        units = layer.warehouse.units
        subject_fit = fit_calendar(
            series, request.scope, spec.calendar, units, before_period=request.period
        )
        decomposition = decompose(
            series, request.scope, request.period, comparison, spec.calendar, units,
            fit=subject_fit,
        )
    except (CalendarError, KeyError) as exc:
        builder.calendar_unfit(str(exc))
        return builder.finish(history=history)
    builder.calendar_gate(decomposition, series)

    # --- Gate 3b: residual band --------------------------------------------
    try:
        band = residual_band(
            series,
            request.scope,
            subject_fit,
            kpi,
            spec.band,
            layer.warehouse.units,
            window_end=period_end(series, request.scope, comparison),
        )
    except BandError as exc:
        builder.band_unavailable(str(exc), history)
        return builder.finish(history=history, decomposition=decomposition)
    builder.band_gate(history, band, decomposition)
    if not band.breached_by(decomposition.residual_pt):
        return builder.finish(
            history=history, band=band, decomposition=decomposition
        )

    # --- Gate 4: specificity ------------------------------------------------
    peers, peer_bands = _peers(
        series, request, comparison, kpi, spec, subject_fit, band, layer.warehouse.units
    )
    verdict = assess_specificity(request.scope, peers, peer_bands, spec.specificity)
    builder.specificity_gate(verdict)

    # --- Gate 5: materiality ------------------------------------------------
    material = assess_materiality(
        kpi,
        decomposition.residual_pt,
        decomposition.residual_inr,
        spec.materiality,
        layer.warehouse.units,
    )
    builder.materiality_gate(material)

    # --- restraint ----------------------------------------------------------
    restraint = assess_restraint(
        connection,
        user,
        kpi,
        layer,
        scope=request.scope,
        period=request.period,
        residual_multiple=material.multiple,
        now=clock,
    )
    builder.restraint_check(restraint)

    return builder.finish(
        history=history,
        band=band,
        decomposition=decomposition,
        specificity=verdict,
        materiality=material,
        restraint=restraint,
        peers=peers,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _observed_periods(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    request: QualifyRequest,
    layer: SemanticLayer,
) -> int:
    """Weeks of history the warehouse holds for this KPI and scope.

    Counted from the data, never read off the contract's `history_weeks`.
    Case #2471 is exactly the difference: the contract can declare
    seventy-eight weeks and the warehouse hold seven.
    """
    return observed_period_count(connection, user, request, layer)


def _peers(series, request, comparison, kpi, spec, subject_fit, subject_band, units):
    """Decompose every visible peer scope by exactly the same method."""
    peers: dict[str, CalendarDecomposition] = {}
    bands: dict[str, ResidualBand] = {}
    for region in series.regions:
        try:
            fit = (
                subject_fit
                if region == request.scope
                else fit_calendar(
                    series, region, spec.calendar, units, before_period=request.period
                )
            )
            peers[region] = decompose(
                series, region, request.period, comparison, spec.calendar, units, fit=fit
            )
            bands[region] = (
                subject_band
                if region == request.scope
                else residual_band(
                    series,
                    region,
                    fit,
                    kpi,
                    spec.band,
                    units,
                    window_end=period_end(series, region, comparison),
                )
            )
        except (CalendarError, BandError):
            continue
    return peers, bands


class _Builder:
    """Accumulates gate results and evidence in the order the gates ran."""

    def __init__(self, request, kpi, layer, clock):
        self.request = request
        self.kpi = kpi
        self.layer = layer
        self.clock = clock
        self.spec = layer.qualify
        self.gates: list[GateResult] = []
        self.checks: list[CheckResult] = []
        # The stage's data timestamp is the warehouse clock, fixed once for
        # the whole run. It is NOT `now()`; the factory records that
        # separately as `retrieved_at`.
        self.factory = EvidenceFactory.for_stage(EVIDENCE_PREFIX, clock, layer)
        self.ledger = EvidenceLedger()

    @property
    def evidence(self) -> list[Evidence]:
        return list(self.ledger)

    # -- evidence ---------------------------------------------------------

    def _evidence(
        self, gate: str, name: str, *, label, value, unit, description, ref,
        source_system=DERIVED, kind=QUERY_KIND, operation="sql", inputs=(),
        statement=None, notes=None,
    ) -> Evidence:
        """Mint one record through the evidence engine and keep it.

        Nothing here picks a reliability weight, an id prefix or a
        timestamp — `engine/evidence.py` does, and refuses anything it
        cannot make chaseable.
        """
        item = self.factory.emit(
            f"{gate}.{name}",
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
        )
        self.ledger.add(item)
        return item

    def _record(self, gate_id, name, check_id, outcome, detail, subjects=(), ids=()):
        check = CheckResult(
            check_id=check_id, name=name, order=gate_id, outcome=outcome,
            detail=detail, subjects=subjects, evidence_ids=ids,
        )
        self.checks.append(check)
        self.gates.append(
            GateResult(
                gate_id=gate_id,
                name=name,
                passed=outcome == "CLEAN",
                outcome_code=None if outcome == "CLEAN" else outcome,
                detail=detail,
                checks=(check,),
                evidence_ids=ids,
            )
        )

    # -- gates ------------------------------------------------------------

    def history_gate(self, history: HistoryCheck) -> None:
        spec = self.spec.band
        item = self._evidence(
            "band", "observed_history",
            source_system=POS_SOURCE,
            label=f"{self.kpi.kpi} periods held by the warehouse",
            value=history.observed_periods, unit="count",
            description=(
                f"count of distinct {self.request.grain} periods present for "
                f"{self.request.scope}"
            ),
            ref="semantic_layer/qualify.yaml::band.history",
            notes=(
                f"{history.required_periods} required by "
                f"{self.kpi.kpi}.baseline.min_history_weeks. Counted from the data, "
                "not read off the contract's history_weeks."
            ),
        )
        self._record(
            spec.gate_id, spec.name, "history", spec.history_outcome_code,
            (
                f"{history.observed_periods} {history.unit} of history against the "
                f"{history.required_periods} the contract requires. Monitoring only; "
                "no baseline can be fitted."
            ),
            ids=(item.evidence_id,),
        )

    def calendar_unfit(self, reason: str) -> None:
        spec = self.spec.calendar
        self._record(spec.gate_id, spec.name, "calendar", spec.outcome_code, reason)

    def calendar_gate(self, d: CalendarDecomposition, series: RegionalSeries) -> None:
        spec = self.spec.calendar
        units = self.layer.warehouse.units
        gate = "calendar"
        ref = "semantic_layer/qualify.yaml::calendar"
        days_m = trading_days(series, d.region, d.period)
        days_prev = trading_days(series, d.region, d.comparison_period)

        ids = tuple(
            item.evidence_id
            for item in (
                self._evidence(
                    gate, "headline",
                    source_system=POS_SOURCE,
                    label=f"{d.region} {d.period} against {d.comparison_period}",
                    value=round(d.headline_pt, units.pct_places), unit="pt",
                    description="actual period total against actual comparison total",
                    inputs=("fact_sales_daily",), ref=ref,
                ),
                self._evidence(
                    gate, "calendar",
                    source_system=DERIVED,
                    label="Movement the calendar accounts for",
                    value=round(d.calendar_pt, units.pct_places), unit="pt",
                    kind=DERIVED_KIND, operation="ols",
                    description=(
                        f"OLS baseline predicted {d.period} against the actual "
                        f"{d.comparison_period} level"
                    ),
                    inputs=("fact_sales_daily", "dim_calendar", "dim_festival_window"),
                    ref=ref,
                    notes=(
                        f"{days_m} trading days against {days_prev}. The day count "
                        "enters by construction: the model is daily and the period "
                        "prediction is the sum over the days the period has."
                    ),
                ),
                self._evidence(
                    gate, "residual",
                    source_system=DERIVED,
                    label="Qualified residual — what the calendar does not explain",
                    value=round(d.residual_pt, units.pct_places), unit="pt",
                    kind=DERIVED_KIND, operation="difference",
                    description="headline minus calendar; the series is not adjusted",
                    inputs=(
                        f"{EVIDENCE_PREFIX}.calendar.headline",
                        f"{EVIDENCE_PREFIX}.calendar.calendar",
                    ),
                    ref=ref,
                ),
                self._evidence(
                    gate, "residual_inr",
                    source_system=DERIVED,
                    label="Qualified residual in rupees",
                    value=round(d.residual_inr / units.inr_per_crore, units.crore_places),
                    unit="INR_CR", kind=DERIVED_KIND, operation="scale",
                    description=(
                        "residual points applied to the comparison period's actual level"
                    ),
                    inputs=(f"{EVIDENCE_PREFIX}.calendar.residual",), ref=ref,
                ),
                self._evidence(
                    gate, "comparison_fit_error",
                    source_system=DERIVED,
                    label=f"Baseline error on {d.comparison_period}, a month nobody disputes",
                    value=round(d.comparison_fit_error_pt, units.pct_places), unit="pt",
                    kind=DERIVED_KIND, operation="ratio",
                    description="predicted comparison total against actual comparison total",
                    inputs=("fact_sales_daily",), ref=ref,
                    notes=(
                        "Emitted rather than absorbed. The calendar effect is measured "
                        "against the ACTUAL previous level, so this error is not folded "
                        "into it — but a large value here means the whole decomposition "
                        "should be read with suspicion."
                    ),
                ),
                self._evidence(
                    gate, "fit_quality",
                    source_system=DERIVED,
                    label="Baseline fit on the history before the period",
                    value=round(d.fit.r_squared, units.crore_places), unit="ratio",
                    kind=DERIVED_KIND, operation="ols",
                    description=(
                        f"{d.fit.fit_days} days, {d.fit.parameters} parameters, fitted "
                        f"on log revenue and excluding {d.period} itself"
                    ),
                    inputs=("fact_sales_daily",), ref=ref,
                    notes=(
                        "Months excluded as already-known anomalies: "
                        + (", ".join(d.fit.excluded_months) or "none")
                    ),
                ),
            )
        )
        self._record(
            spec.gate_id, spec.name, "calendar", "CLEAN",
            (
                f"{d.headline_pt:+.1f} pt headline decomposes into {d.calendar_pt:+.1f} pt "
                f"calendar and {d.residual_pt:+.1f} pt residual "
                f"({d.residual_inr / units.inr_per_crore:+.2f} INR Cr). "
                "The series is not adjusted."
            ),
            ids=ids,
        )

    def band_unavailable(self, reason: str, history: HistoryCheck) -> None:
        spec = self.spec.band
        self._record(
            spec.gate_id, spec.name, "band", spec.history_outcome_code, reason
        )

    def band_gate(self, history, band: ResidualBand, d: CalendarDecomposition) -> None:
        spec = self.spec.band
        units = self.layer.warehouse.units
        ref = "semantic_layer/qualify.yaml::band"
        item = self._evidence(
            "band", "residual_band",
            source_system=DERIVED,
            label=(
                f"{band.region} empirical residual band, {band.lookback_weeks}-week "
                f"trailing, q{band.quantile:g}"
            ),
            value=round(band.band_pt, units.pct_places), unit="pt",
            kind=DERIVED_KIND, operation="stl",
            description=(
                f"STL(period={spec.stl.period}, robust={spec.stl.robust}) on the "
                f"calendar residual; empirical quantile of |residual| over "
                f"{band.observations} days"
            ),
            inputs=("fact_sales_daily",), ref=ref,
            notes=(
                "An empirical quantile, not a sigma. Business series are heavy-tailed "
                "and autocorrelated, so two standard deviations is a statement about a "
                "Gaussian nobody has seen."
            ),
        )
        breached = band.breached_by(d.residual_pt)
        outcome = "CLEAN" if breached else spec.within_band_outcome_code
        detail = (
            f"residual {d.residual_pt:+.1f} pt against a {band.band_pt:.1f} pt band "
            f"({band.observations} days, q{band.quantile:g}); "
            + ("outside the band." if breached else "inside the band — not a case.")
        )
        self._record(
            spec.gate_id, spec.name, "band", outcome, detail, ids=(item.evidence_id,)
        )

    def specificity_gate(self, verdict: SpecificityVerdict) -> None:
        spec = self.spec.specificity
        units = self.layer.warehouse.units
        strip = " · ".join(
            f"{region} {value:+.1f}" for region, value in sorted(verdict.strip.items())
        )
        item = self._evidence(
            "specificity", "peers_breaching",
            source_system=DERIVED,
            label=f"Peer {spec.peer_dimension}s breaching their own band, same direction",
            value=len(verdict.breaching), unit="count",
            kind=DERIVED_KIND, operation="compare",
            description=(
                f"each {spec.peer_dimension} decomposed by the same method and judged "
                "against its own band"
            ),
            inputs=("fact_sales_daily",),
            ref="semantic_layer/qualify.yaml::specificity",
            notes=f"Regional residual strip: {strip}",
        )
        if not verdict.peers_visible:
            self._record(
                spec.gate_id, spec.name, "specificity", spec.unknown_outcome_code,
                (
                    f"only {len(verdict.peers)} {spec.peer_dimension}(s) visible at this "
                    "scope, so the gate cannot rule. Not visible is not the same as "
                    "not moving."
                ),
                ids=(item.evidence_id,),
            )
            return
        if verdict.is_market_case:
            self._record(
                spec.gate_id, spec.name, "specificity", spec.outcome_code,
                (
                    f"{len(verdict.breaching)} of {len(verdict.peers)} regions breach in "
                    f"the same direction ({', '.join(verdict.breaching)}). This is the "
                    f"market, not {verdict.subject}. Escalated to "
                    f"{spec.escalate_to_role.upper()}: one case, not "
                    f"{len(verdict.breaching)}."
                ),
                subjects=verdict.breaching, ids=(item.evidence_id,),
            )
            return
        self._record(
            spec.gate_id, spec.name, "specificity", "CLEAN",
            (
                f"{len(verdict.breaching)} of {len(verdict.peers)} regions breach, below "
                f"the {spec.min_peers_breaching} that would make this a market movement. "
                f"Strip: {strip}."
            ),
            subjects=verdict.breaching, ids=(item.evidence_id,),
        )

    def materiality_gate(self, verdict: MaterialityVerdict) -> None:
        spec = self.spec.materiality
        item = self._evidence(
            "materiality", "residual_against_limit",
            source_system=SEMANTIC_LAYER,
            label=f"Residual against the {verdict.owner_role}'s materiality limit",
            value=round(verdict.multiple, self.layer.warehouse.units.crore_places),
            unit="ratio", kind=DERIVED_KIND, operation="ratio",
            description=(
                f"|residual| {verdict.residual:.4f} {verdict.unit} over the contract's "
                f"{verdict.threshold} {verdict.unit}"
            ),
            inputs=(f"{EVIDENCE_PREFIX}.calendar.residual_inr",),
            ref=f"semantic_layer/kpis/{verdict.kpi}.yaml::thresholds.materiality",
        )
        if verdict.material:
            self._record(
                spec.gate_id, spec.name, "materiality", "CLEAN",
                (
                    f"residual is {verdict.multiple:.1f}x the {verdict.display} limit "
                    f"{verdict.owner_role} set. Worth an investigation."
                ),
                ids=(item.evidence_id,),
            )
            return
        self._record(
            spec.gate_id, spec.name, "materiality", spec.outcome_code,
            (
                f"residual is {verdict.multiple:.2f}x the {verdict.display} limit — below "
                f"it. Routed to the {verdict.route}, not dropped."
            ),
            ids=(item.evidence_id,),
        )

    def restraint_check(self, verdict: RestraintVerdict) -> None:
        check = CheckResult(
            check_id="restraint",
            name="Restraint",
            # After the last gate. Restraint is not a gate: it asks whether
            # one more case helps the person who has to read it.
            order=max(self.spec.gate_ids()) + 1,
            outcome=verdict.outcome,
            detail=verdict.detail,
            subjects=(verdict.linked_case_id,) if verdict.linked_case_id else (),
        )
        self.checks.append(check)

    def finish(self, **parts) -> QualifyResult:
        peers = parts.pop("peers", {}) or {}
        return QualifyResult(
            request=self.request,
            gates=tuple(self.gates),
            checks=tuple(self.checks),
            evidence=tuple(self.ledger),
            as_of=self.clock,
            peer_decompositions=peers,
            **parts,
        )


__all__ = ["QualifyError", "QualifyRequest", "QualifyResult", "qualify"]
