"""ADJUDICATE — run the six tests against every hypothesis and decide.

    Data -> VALIDATE -> QUALIFY -> GATHER -> ADJUDICATE -> VERDICT

GATHER brought the evidence back without an opinion. This forms one, and
it forms it with statistics rather than assertion. Nothing here calls a
model — a static test asserts that nothing under `engine/adjudicate/`
imports `llm/`.

The order matters. The two hard gates run first, and a failure eliminates
the hypothesis outright with a machine-readable reason: an eliminated
hypothesis does not get to cast a weighted vote on how confident we are
about something else. The three weighted tests then run on what survives,
and the confounder screen caps rather than kills, because a confounded
hypothesis may still be true.

Attribution is deliberately conservative. A DiD is an estimate with a
standard error, and claiming the point estimate claims a precision the
data has not got. Each surviving hypothesis is attributed the near end of
its interval; whatever is left over stays unattributed, where the verdict
table can see it and where case #2451's ₹0.86 Cr comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import duckdb
import pandas as pd

from engine.adjudicate.series import (
    STORE,
    DailySeries,
    SeriesError,
    StorePanel,
    load_panel,
    load_series,
)
from engine.adjudicate.tests import (
    ConfounderResult,
    DidEstimate,
    Exposure,
    Finding,
    Matching,
    balance,
    estimate_did,
    exposed_stores,
    fit_elasticity,
    match_controls,
    parallel_trends,
    test_confounders,
    test_did,
    test_dose_response,
    test_precedence,
    test_specificity,
    test_sufficiency,
)
from engine.contracts import Evidence, Hypothesis, TestResult
from engine.db import GovernanceError, execute_governed, warehouse_clock
from engine.evidence import EvidenceFactory, EvidenceLedger
from security.policy import PolicyError, User, resolve_policy
from semantic_layer.schema import SemanticLayer, get_semantic_layer

EVIDENCE_PREFIX = "adjudicate"
DERIVED_KIND = "derived_estimate"
DERIVED = "derived"


class AdjudicationError(RuntimeError):
    """ADJUDICATE could not be run as asked."""


@dataclass(frozen=True)
class AdjudicateRequest:
    """The movement, the candidates, and the window they are judged in."""

    kpi: str
    scope: str
    grain: str
    period: str
    period_start: date
    period_end: date
    comparison_start: date
    comparison_end: date
    residual_pt: float
    residual_inr: float
    hypotheses: tuple[str, ...]
    #: When the movement began. The pre-trend and the pre-test are fitted
    #: on the weeks before it.
    onset: date
    #: Per hypothesis, the share of volume its cause touches and how far
    #: that cause moved. GATHER's structured lane supplies them; a
    #: hypothesis without them is not sufficiency-testable.
    volume_shares: dict[str, float] = field(default_factory=dict)
    cause_magnitudes: dict[str, float] = field(default_factory=dict)


@dataclass
class HypothesisVerdict:
    """One hypothesis, its six tests, and what it was allowed to claim."""

    tag: str
    label: str
    status: str
    elimination_reason: str | None
    findings: tuple[Finding, ...]
    attributed_pt: float = 0.0
    attributed_share: float = 0.0
    did: DidEstimate | None = None
    matching: Matching | None = None
    exposure: Exposure | None = None
    confounders: tuple[ConfounderResult, ...] = ()
    verifiable: bool = True
    missing_sources: tuple[str, ...] = ()

    def finding(self, test_id: int) -> Finding | None:
        for item in self.findings:
            if item.test_id == test_id:
                return item
        return None

    @property
    def eliminated(self) -> bool:
        return self.status == "eliminated"

    @property
    def unresolved_confounders(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.confounders if not item.resolved)


@dataclass
class AdjudicationResult:
    """Every hypothesis, judged, and what the whole slate accounts for."""

    request: AdjudicateRequest
    verdicts: tuple[HypothesisVerdict, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    as_of: datetime | None = None

    def verdict(self, tag: str) -> HypothesisVerdict:
        for item in self.verdicts:
            if item.tag == tag:
                return item
        raise KeyError(f"no hypothesis {tag!r} was adjudicated")

    @property
    def surviving(self) -> tuple[HypothesisVerdict, ...]:
        return tuple(item for item in self.verdicts if not item.eliminated)

    @property
    def eliminated(self) -> tuple[HypothesisVerdict, ...]:
        return tuple(item for item in self.verdicts if item.eliminated)

    @property
    def coverage(self) -> float:
        """Share of the qualified residual the surviving hypotheses account for."""
        return min(sum(item.attributed_share for item in self.surviving), 1.0)

    @property
    def unattributed_pt(self) -> float:
        return abs(self.request.residual_pt) - sum(
            abs(item.attributed_pt) for item in self.surviving
        )

    @property
    def unattributed_inr(self) -> float:
        if not self.request.residual_pt:
            return 0.0
        return (
            self.unattributed_pt
            / abs(self.request.residual_pt)
            * abs(self.request.residual_inr)
        )

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.evidence}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def adjudicate(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    request: AdjudicateRequest,
    *,
    layer: SemanticLayer | None = None,
    clock: datetime | None = None,
) -> AdjudicationResult:
    """Judge every candidate against the six tests."""
    layer = layer or get_semantic_layer()
    if request.kpi not in layer.kpis:
        raise AdjudicationError(
            f"no KPI contract named {request.kpi!r}; known: {sorted(layer.kpis)}"
        )
    try:
        resolve_policy(layer.kpis[request.kpi], user, layer.warehouse.governance)
    except PolicyError as exc:
        raise GovernanceError(str(exc)) from exc

    clock = clock or warehouse_clock(connection, layer)
    spec = layer.adjudicate
    scored = layer.adjudication.tests
    factory = EvidenceFactory.for_stage(EVIDENCE_PREFIX, clock, layer)
    ledger = EvidenceLedger()

    window_start = request.onset - pd.Timedelta(
        weeks=spec.precedence.window_weeks_before
    ).to_pytimedelta()
    window_end = request.period_end

    panel = load_panel(
        connection, user, layer,
        kpi=request.kpi, scope=request.scope,
        period_start=request.period_start, period_end=request.period_end,
        comparison_start=request.comparison_start,
        comparison_end=request.comparison_end,
        pre_trend_weeks=spec.did.pretest.lookback_weeks,
        onset=request.onset,
    )
    effect = load_series(
        connection, user, layer, spec.precedence.effect,
        kpi=request.kpi, scope=request.scope,
        window_start=window_start, window_end=window_end,
    )

    history = _pre_history(
        connection, user, layer, request, spec.did.pretest.lookback_weeks
    )

    verdicts: list[HypothesisVerdict] = []
    for tag in request.hypotheses:
        verdicts.append(
            _judge(
                connection, user, layer, factory, ledger, request, spec, scored,
                tag=tag, panel=panel, effect=effect,
                history=history, window_start=window_start, window_end=window_end,
            )
        )

    _attribute(verdicts, request, spec)
    result = AdjudicationResult(
        request=request, verdicts=tuple(verdicts), as_of=clock
    )
    _emit_summary(factory, ledger, result, layer)
    result.evidence = tuple(ledger)
    return result


# ---------------------------------------------------------------------------
# One hypothesis
# ---------------------------------------------------------------------------


def _judge(
    connection, user, layer, factory, ledger, request, spec, scored, *,
    tag, panel, effect, history, window_start, window_end,
) -> HypothesisVerdict:
    template = layer.causal_graph.hypotheses.get(tag)
    label = template.label if template else tag
    applicable = set(template.applicable_tests) if template else set()

    cause = None
    cause_spec = spec.precedence.causes.get(tag)
    if cause_spec is not None:
        try:
            cause = load_series(
                connection, user, layer, cause_spec,
                kpi=request.kpi, scope=request.scope,
                window_start=window_start, window_end=window_end,
            )
        except (SeriesError, GovernanceError):
            cause = None

    # WHICH STORES THIS CAUSE REACHED. Derived from this hypothesis's own
    # cause series, so a price rise cannot borrow a stock-out's treatment
    # group and claim its difference-in-differences.
    dose = (
        cause.by_store(request.period_start, request.period_end)
        if cause is not None
        else pd.Series(dtype=float)
    )
    direction = cause.direction if cause is not None else None
    exposure = exposed_stores(dose, spec.exposure, direction=direction)
    matching = (
        match_controls(panel, spec.did, list(exposure.exposed))
        if exposure.testable
        else None
    )

    findings: list[Finding] = []

    # --- Test 1, hard gate --------------------------------------------------
    precedence = test_precedence(
        cause, effect, spec.precedence,
        reason=spec.precedence.causes.get(tag).label if cause_spec else "none configured",
        # Only a split the engine is prepared to stand behind. A
        # two-store "exposed" group that failed its own size floor is
        # not a place to measure an onset.
        on_stores=exposure.exposed if exposure.testable else (),
    )
    findings.append(precedence)

    # --- Test 2, hard gate --------------------------------------------------
    sufficiency = _sufficiency_for(
        connection, user, layer, request, spec, scored, tag, label,
        window_start=window_start, window_end=window_end,
    )
    findings.append(sufficiency)

    hard_failures = [item for item in findings if item.testable and not item.passed]
    if hard_failures:
        # EVERY gate that failed, not the first. A hypothesis that arrived
        # too late AND could not have moved the number this far has failed
        # twice, and a case file that reports one of those is hiding the
        # other.
        names = {spec.precedence.test_id: "precedence", spec.sufficiency.test_id: "sufficiency"}
        reason = ", ".join(names[item.test_id] for item in hard_failures)
        verdict = HypothesisVerdict(
            tag=tag, label=label, status="eliminated",
            elimination_reason=reason,
            findings=tuple(findings),
            verifiable=not (template.missing_sources() if template else []),
            missing_sources=tuple(template.missing_sources()) if template else (),
        )
        # The confounder screen still runs on an eliminated hypothesis: what
        # else could have produced its signature is worth recording even
        # when the hypothesis itself is dead.
        verdict.exposure = exposure
        verdict.confounders = _screen(
            connection, user, layer, request, spec, template, matching,
            window_start=window_start, window_end=window_end,
        )
        findings.append(test_confounders(list(verdict.confounders), spec.confounder_screen))
        verdict.findings = tuple(findings)
        _emit_hypothesis(factory, ledger, request, verdict, layer)
        return verdict

    # --- Tests 3, 4, 5 ------------------------------------------------------
    if "dose_response" in applicable:
        findings.append(
            test_dose_response(
                dose, panel, spec.dose_response,
                min_r=scored["dose_response"].min_r or 0.0, direction=direction,
            )
        )
    if "specificity" in applicable:
        findings.append(
            test_specificity(
                exposure, panel, spec.specificity,
                max_p=scored["specificity"].max_p_value or 1.0,
            )
        )

    estimate = None
    if "did" in applicable and matching is not None and matching.pairs:
        estimate = estimate_did(panel, matching, spec.did)
        findings.append(
            test_did(
                estimate, matching,
                parallel_trends(history, matching, spec.did, request.onset),
                spec.did,
                max_p=scored["did"].max_p_value or 1.0,
            )
        )

    # --- Test 6, cap --------------------------------------------------------
    confounders = _screen(
        connection, user, layer, request, spec, template, matching,
        window_start=window_start, window_end=window_end,
    )
    findings.append(test_confounders(list(confounders), spec.confounder_screen))

    verdict = HypothesisVerdict(
        tag=tag, label=label,
        status="supported" if estimate is not None else "live",
        elimination_reason=None,
        findings=tuple(findings),
        did=estimate,
        matching=matching,
        exposure=exposure,
        confounders=confounders,
        verifiable=not (template.missing_sources() if template else []),
        missing_sources=tuple(template.missing_sources()) if template else (),
    )
    _emit_hypothesis(factory, ledger, request, verdict, layer)
    return verdict


def _sufficiency_for(
    connection, user, layer, request, spec, scored, tag, label, *, window_start, window_end
) -> Finding:
    series = spec.sufficiency.hypotheses.get(tag)
    minimum = scored["sufficiency"].min_residual_share or 0.0
    if series is None:
        return test_sufficiency(
            float("nan"), 0.0, 0,
            volume_share=None, cause_magnitude_pct=None,
            residual_pt=request.residual_pt, spec=spec.sufficiency,
            min_residual_share=minimum, label=label,
        )
    lookback_start = request.comparison_end - pd.Timedelta(
        weeks=spec.sufficiency.elasticity.lookback_weeks
    ).to_pytimedelta()
    try:
        rows, _filtered, _masked = execute_governed(
            user, request.kpi, series.sql,
            {
                "window_start": lookback_start,
                "window_end": request.comparison_end,
                "scope": request.scope,
            },
            connection=connection, layer=layer, purpose=f"adjudicate.elasticity.{tag}",
        )
    except GovernanceError as exc:
        return test_sufficiency(
            float("nan"), 0.0, 0,
            volume_share=None, cause_magnitude_pct=None,
            residual_pt=request.residual_pt, spec=spec.sufficiency,
            min_residual_share=minimum, label=f"{label} ({exc})",
        )

    elasticity, r_squared, observations = fit_elasticity(
        pd.DataFrame(list(rows)), spec.sufficiency
    )
    return test_sufficiency(
        elasticity, r_squared, observations,
        volume_share=request.volume_shares.get(tag),
        cause_magnitude_pct=request.cause_magnitudes.get(tag),
        residual_pt=request.residual_pt,
        spec=spec.sufficiency,
        min_residual_share=minimum,
        label=label,
        declared=series.declared_elasticity,
        declared_source=series.declared_source,
    )


def _screen(
    connection, user, layer, request, spec, template, matching, *, window_start, window_end
) -> tuple[ConfounderResult, ...]:
    """Every confounder the graph declares for this hypothesis."""
    if template is None:
        return ()
    screen = spec.confounder_screen
    results: list[ConfounderResult] = []

    for name in template.confounders:
        measure = screen.measures.get(name)
        if measure is None:
            results.append(
                ConfounderResult(
                    name, "unmeasured", None,
                    "the causal graph declares it and nothing measures it.",
                )
            )
            continue
        if measure.scope_level == "region" and screen.region_level_resolves:
            results.append(
                ConfounderResult(
                    name, "by_construction", None,
                    f"{measure.label}: {measure.note.strip()}",
                )
            )
            continue
        if measure.scope_level == "matched":
            if matching is None:
                results.append(
                    ConfounderResult(
                        name, "unscreened", None,
                        f"{measure.label}: this would be removed by matching on "
                        f"{measure.resolved_by}, and this hypothesis has no matched "
                        "control group to remove it with. Its cause has no store "
                        "dimension, so there is nothing to match on.",
                    )
                )
                continue
            results.append(
                ConfounderResult(
                    name, "by_matching", None,
                    f"{measure.label}: removed by matching on "
                    f"{measure.resolved_by}. {measure.note.strip()}",
                )
            )
            continue
        if matching is None:
            results.append(
                ConfounderResult(
                    name, "unscreened", None,
                    f"{measure.label}: measurable per store, but this hypothesis has "
                    "no treated and control groups to balance it across. Unscreened "
                    "is not resolved.",
                )
            )
            continue
        try:
            rows, _filtered, _masked = execute_governed(
                user, request.kpi, measure.sql,
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "scope": request.scope,
                },
                connection=connection, layer=layer,
                purpose=f"adjudicate.confounder.{name}",
            )
        except GovernanceError as exc:
            results.append(
                ConfounderResult(name, "unmeasured", None, f"{measure.label}: {exc}")
            )
            continue
        frame = pd.DataFrame(list(rows))
        if frame.empty or STORE not in frame.columns:
            results.append(
                ConfounderResult(
                    name, "unmeasured", None,
                    f"{measure.label}: the measure returned no store-level rows.",
                )
            )
            continue
        results.append(
            balance(
                frame.groupby(STORE)["value"].mean(), matching, screen, name, measure.label
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def _attribute(verdicts, request, spec) -> None:
    """Give each survivor the near end of its interval, and no more."""
    residual = abs(request.residual_pt)
    if not residual:
        return
    remaining = residual
    for verdict in verdicts:
        if verdict.eliminated or verdict.did is None:
            continue
        claim = min(abs(verdict.did.attributable_pt), remaining)
        verdict.attributed_pt = -claim if request.residual_pt < 0 else claim
        verdict.attributed_share = claim / residual
        remaining -= claim


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _emit_hypothesis(factory, ledger, request, verdict, layer) -> None:
    units = layer.warehouse.units
    for finding in verdict.findings:
        if finding.statistic is None and finding.p_value is None:
            continue
        section = _section_for(layer, finding.test_id)
        ledger.add(
            factory.emit(
                f"{verdict.tag}.test{finding.test_id}",
                kind=DERIVED_KIND,
                label=f"{finding.name} — {verdict.label}",
                value=(
                    round(float(finding.statistic), units.crore_places)
                    if finding.statistic is not None
                    else round(float(finding.p_value), units.crore_places)
                ),
                unit=TEST_UNITS[section],
                source_system=DERIVED,
                method=TEST_METHODS[section],
                description=finding.detail,
                ref=f"semantic_layer/adjudicate.yaml::{section}",
                inputs=("fact_sales_daily",),
                notes=(
                    None if finding.testable
                    else "NOT TESTABLE. Not a pass and not a failure."
                ),
            )
        )
    if verdict.did is not None:
        ledger.add(
            factory.emit(
                f"{verdict.tag}.attributable",
                kind=DERIVED_KIND,
                label=f"Residual {verdict.label} may claim",
                value=round(verdict.did.attributable_pt, units.crore_places),
                unit="pt",
                source_system=DERIVED,
                method="did",
                description=(
                    f"the {verdict.did.confidence:.0%} interval's near end of a "
                    f"{verdict.did.point_pt:+.2f} pt estimate "
                    f"(se {verdict.did.standard_error_pt:.2f})"
                ),
                ref="semantic_layer/adjudicate.yaml::did.attribution",
                inputs=(f"{EVIDENCE_PREFIX}.{verdict.tag}.test5",),
                notes=(
                    "The near end of the interval, not the point estimate. Attributing "
                    "the point estimate would claim a precision the data has not got."
                ),
            )
        )


def _emit_summary(factory, ledger, result, layer) -> None:
    units = layer.warehouse.units
    ledger.add(
        factory.emit(
            "coverage",
            kind=DERIVED_KIND,
            label="Share of the qualified residual the surviving hypotheses account for",
            value=round(result.coverage, units.crore_places),
            unit="ratio",
            source_system=DERIVED,
            method="ratio",
            description=(
                f"{len(result.surviving)} surviving of {len(result.verdicts)} candidates; "
                f"{result.unattributed_pt:.2f} pt left unattributed"
            ),
            ref="semantic_layer/adjudicate.yaml::did.attribution",
            inputs=tuple(
                f"{EVIDENCE_PREFIX}.{item.tag}.attributable"
                for item in result.surviving
                if item.did is not None
            ),
        )
    )
    ledger.add(
        factory.emit(
            "unattributed",
            kind=DERIVED_KIND,
            label="Residual no surviving hypothesis accounts for",
            value=round(result.unattributed_inr / units.inr_per_crore, units.crore_places),
            unit="INR_CR",
            source_system=DERIVED,
            method="difference",
            description="the qualified residual less everything attributed",
            ref="semantic_layer/adjudicate.yaml::did.attribution",
            inputs=(f"{EVIDENCE_PREFIX}.coverage",),
            notes=(
                "This is the number the verdict table compares against materiality. "
                "It is what an unverifiable hypothesis is left holding."
            ),
        )
    )


#: What each test's headline statistic IS, and how it was made. Keyed by
#: the test's name in `adjudicate.yaml`, never by a number: the test ids
#: themselves live in the semantic layer, and a dict keyed 1..6 here would
#: be a second copy of them.
TEST_UNITS = {
    "precedence": "days",
    "sufficiency": "pt",
    "dose_response": "ratio",
    "specificity": "t_statistic",
    "did": "pt",
    "confounder_screen": "count",
}
TEST_METHODS = {
    "precedence": "changepoint",
    "sufficiency": "elasticity",
    "dose_response": "ols",
    "specificity": "welch_t",
    "did": "did",
    "confounder_screen": "balance",
}


def _section_for(spec: SemanticLayer, test_id: int) -> str:
    """The `adjudicate.yaml` section a test id belongs to."""
    for name, identifier in spec.adjudicate.test_ids().items():
        if identifier == test_id:
            return name
    raise AdjudicationError(f"no adjudicate.yaml section carries test id {test_id}")


def _pre_history(connection, user, layer, request, weeks) -> pd.DataFrame:
    start = request.onset - pd.Timedelta(weeks=weeks).to_pytimedelta()
    rows, _filtered, _masked = execute_governed(
        user, request.kpi,
        """
        SELECT region, store_id, txn_date, net_revenue_inr
        FROM fact_sales_daily
        WHERE region = $scope AND txn_date BETWEEN $start AND $end
        """,
        {"scope": request.scope, "start": start, "end": request.onset},
        connection=connection, layer=layer, purpose="adjudicate.parallel_trends",
    )
    return pd.DataFrame(list(rows))


def to_contracts(result: AdjudicationResult, layer: SemanticLayer) -> tuple[Hypothesis, ...]:
    """The adjudication as `Hypothesis` contracts, for the case file."""
    scored = layer.adjudication.tests
    by_id = {spec.test_id: (name, spec) for name, spec in scored.items()}
    out: list[Hypothesis] = []
    for verdict in result.verdicts:
        tests = tuple(
            TestResult(
                test_id=finding.test_id,
                name=finding.name,
                type=by_id[finding.test_id][1].type,
                passed=finding.passed,
                statistic=finding.statistic,
                p_value=finding.p_value,
                weight=by_id[finding.test_id][1].weight,
                detail=finding.detail,
            )
            for finding in verdict.findings
        )
        out.append(
            Hypothesis(
                hypothesis_id=verdict.tag,
                label=verdict.label,
                description=verdict.label,
                status=verdict.status,
                elimination_reason=verdict.elimination_reason,
                tests=tests,
                required_sources=(),
                missing_sources=verdict.missing_sources,
                verifiable=verdict.verifiable,
                attributed_share=min(abs(verdict.attributed_share), 1.0),
            )
        )
    return tuple(out)


__all__ = [
    "AdjudicateRequest",
    "AdjudicationError",
    "AdjudicationResult",
    "HypothesisVerdict",
    "adjudicate",
    "to_contracts",
]
