"""The five frozen adjudication objects the narrator is handed.

WHAT THIS IS, AND WHAT IT IS NOT. P13 wires the stages together, and from
then on this object arrives from the pipeline: VALIDATE through VERDICT,
computed. Until then the narrator still has to be built and checked
against all five scenarios, so this module assembles the same contract
from the generator's own scenario configs.

RULE 4 IS THE REASON IT READS THE CONFIGS. Every quantity below comes out
of `data/generator/config/scenario_*.yaml` — the same files
`tests/test_number_registry.py` asserts the generator reproduces. Nothing
here is a literal. Change a target in the generator and the narrative
changes with it, which is precisely the property that would be lost if
these objects were written out by hand.

WHAT IT DOES NOT PROVE. It does not prove the engine computes these
figures; the stage test suites do that, scenario by scenario. It proves
the narrator and the grounding validator behave correctly on the shape and
the values a real case carries. When P13 lands, `build_case` is replaced
by the pipeline's own output and nothing downstream changes — the
narrator has never known where the object came from, which is the point of
handing it a frozen JSON document.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

from semantic_layer.schema import SemanticLayer, get_semantic_layer

#: The five CLAUDE.md §"Number Registry" scenarios, in the order the demo
#: walks them.
CASE_IDS = ("2451", "2467", "2470", "2471", "2472")

#: The data timestamp every case is quoted as of. The end of the 18-month
#: window, not the moment this module runs — a case file whose numbers
#: shift with the wall clock cannot be replayed offline.
AS_OF = datetime(2025, 11, 30, 12, 0, tzinfo=UTC)

DERIVED_KIND = "derived_estimate"
STRUCTURED_KIND = "structured_query"
DERIVED = "derived"
CONTRACT = "semantic_layer"


class CaseFileError(RuntimeError):
    """A scenario config does not carry what the narrator needs."""


def scenarios() -> dict[str, dict[str, Any]]:
    """The generator's parsed scenario configs. The source of every number."""
    from data.generator.model import load_all_configs

    return load_all_configs()["scenarios"]


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------


@dataclass
class _Builder:
    """Emits evidence through the real factory, so the ids and lineage are real."""

    layer: SemanticLayer
    case_id: str

    def __post_init__(self) -> None:
        from engine.evidence import EvidenceFactory, EvidenceLedger

        # `retrieved_at` is pinned, and it is the one place this builder
        # departs from what the engine does. The factory defaults it to
        # `datetime.now()` — correctly, because a real case wants the data
        # timestamp and the read timestamp kept apart and both true.
        #
        # A case file that has to be REPLAYED cannot have either of them
        # move: the frozen JSON is hashed into the narrator's request
        # fingerprint, so a wall clock in the object means a new fixture
        # key every second and an offline demo that never replays
        # (rule 8). Pinned here; live in the pipeline.
        self.factory = EvidenceFactory.for_stage(
            "case", AS_OF, self.layer, retrieved_at=AS_OF
        )
        self.ledger = EvidenceLedger()

    def emit(
        self,
        key: str,
        label: str,
        value: float | int | str | None,
        unit: str,
        *,
        method: str = "sql",
        kind: str = STRUCTURED_KIND,
        source_system: str = "pos",
        description: str,
        ref: str,
    ):
        record = self.factory.emit(
            key,
            kind=kind,
            label=label,
            value=value,
            unit=unit,
            source_system=source_system,
            method=method,
            description=description,
            ref=ref,
        )
        self.ledger.add(record)
        return record

    def derived(self, key: str, label: str, value, unit: str, *, description: str, ref: str):
        return self.emit(
            key,
            label,
            value,
            unit,
            method="ratio",
            kind=DERIVED_KIND,
            source_system=DERIVED,
            description=description,
            ref=ref,
        )

    def threshold(self, key: str, label: str, value, unit: str, *, description: str, ref: str):
        return self.emit(
            key,
            label,
            value,
            unit,
            method="lookup",
            kind=DERIVED_KIND,
            source_system=CONTRACT,
            description=description,
            ref=ref,
        )


def _test(layer: SemanticLayer, name: str, *, passed: bool, statistic=None, p_value=None, detail: str):
    from engine.contracts import TestResult

    spec = layer.adjudication.tests[name]
    return TestResult(
        test_id=spec.test_id,
        name=spec.name,
        type=spec.type,
        passed=passed,
        statistic=statistic,
        p_value=p_value,
        weight=spec.weight,
        detail=detail,
        evidence_ids=(),
    )


def _confidence(layer: SemanticLayer, scores: Mapping[str, float]):
    """The six components, weighted from the semantic layer."""
    from engine.contracts import ConfidenceBreakdown, ConfidenceComponent

    specs = layer.adjudication.confidence.components
    components = tuple(
        ConfidenceComponent(
            key=key,
            name=specs[key].name,
            value=float(scores[key]),
            weight=specs[key].weight,
            detail=None,
            evidence_ids=(),
        )
        for key in ("s1", "s2", "s3", "s4", "s5", "s6")
        if key in scores
    )
    raw = float(scores["raw"])
    return ConfidenceBreakdown(
        components=components,
        raw=raw,
        caps_applied=(),
        after_caps=raw,
        calibrated=float(scores["calibrated"]),
        calibration_method="isotonic",
        calibration_sample_size=None,
    )


def _empty_confidence():
    """A case that never reached scoring. Zeroes, not absences.

    A gate kill has no confidence, and the honest rendering of that is a
    zero the narrator can cite rather than a missing field it has to write
    around.
    """
    from engine.contracts import ConfidenceBreakdown

    return ConfidenceBreakdown(
        components=(),
        raw=0.0,
        caps_applied=(),
        after_caps=0.0,
        calibrated=0.0,
        calibration_method="isotonic",
        calibration_sample_size=None,
    )


# ---------------------------------------------------------------------------
# The five cases
# ---------------------------------------------------------------------------


def build_case(case_id: str, layer: SemanticLayer | None = None):
    """Assemble one scenario's frozen `Adjudication`."""
    layer = layer or get_semantic_layer()
    configs = scenarios()
    if case_id not in configs:
        raise CaseFileError(
            f"no scenario config for case {case_id!r}; the generator carries "
            f"{sorted(configs)}"
        )
    builders = {
        "2451": _build_2451,
        "2467": _build_2467,
        "2470": _build_2470,
        "2471": _build_2471,
        "2472": _build_2472,
    }
    return builders[case_id](configs[case_id], layer)


def build_all(layer: SemanticLayer | None = None) -> dict[str, Any]:
    layer = layer or get_semantic_layer()
    return {case_id: build_case(case_id, layer) for case_id in CASE_IDS}


def frozen(case_id: str, layer: SemanticLayer | None = None) -> dict[str, Any]:
    """The JSON form the narrator receives."""
    return build_case(case_id, layer).model_dump(mode="json")


def _shell(config: Mapping[str, Any], **overrides) -> dict[str, Any]:
    """The fields every case carries, read off its own config."""
    return {
        "case_id": str(config["case_id"]),
        "kpi": config["kpi"],
        "scope": config["scope"],
        "grain": config["grain"],
        "period": str(config["period"]),
        "opened_at": AS_OF,
        **overrides,
    }


def _build_2451(config, layer):
    """West, net revenue, Nov 2025 -> PARTIALLY EXPLAINED."""
    from engine.contracts import Adjudication, Hypothesis

    t = config["targets"]
    m = config["mechanism"]
    e = config["evidence"]
    measured = config["measured_targets"]
    competing = config["competing_hypotheses"]
    b = _Builder(layer, "2451")
    materiality = layer.kpis[config["kpi"]].thresholds.materiality

    headline = b.derived(
        "headline",
        "West net revenue, month on month",
        t["headline_movement_pct"],
        "pct",
        description="November against October, net of returns, discount and tax",
        ref="semantic_layer/kpis/net_revenue.yaml::formula_sql",
    )
    decline = b.derived(
        "decline",
        "Absolute decline",
        t["absolute_decline_inr_cr"],
        "INR_CR",
        description="October less November",
        ref="semantic_layer/kpis/net_revenue.yaml::formula_sql",
    )
    calendar = b.derived(
        "calendar",
        "Calendar-attributed movement",
        t["calendar_attributed_inr_cr"],
        "INR_CR",
        description="STL plus calendar regression; expected, not unexplained",
        ref="semantic_layer/kpis/net_revenue.yaml::baseline",
    )
    residual = b.derived(
        "residual",
        "Qualified residual",
        t["qualified_residual_inr_cr"],
        "INR_CR",
        description="the decline the calendar does not explain",
        ref="engine/qualify/gate.py",
    )
    residual_pt = b.derived(
        "residual_pt",
        "Qualified residual, points",
        abs(t["qualified_residual_pt"]),
        "pt",
        description="the same residual expressed against October",
        ref="engine/qualify/gate.py",
    )
    limit = b.threshold(
        "materiality",
        "Materiality threshold",
        materiality.value,
        materiality.unit,
        description=f"below {materiality.display} a residual is not worth a case",
        ref="semantic_layer/kpis/net_revenue.yaml::thresholds.materiality",
    )
    band = b.derived(
        "band",
        "West 8-week empirical band",
        t["west_empirical_band_pt"],
        "pt",
        description="residual quantile band over the trailing eight weeks",
        ref="engine/qualify/band.py",
    )
    treated = b.emit(
        "treated_stores",
        "Treated stores",
        m["treated_store_count"],
        "count",
        method="count",
        description="West stores the allocation fault reached",
        ref="engine/adjudicate/tests.py::exposed_stores",
    )
    controls = b.emit(
        "matched_controls",
        "Matched control stores",
        m["matched_control_count"],
        "count",
        method="count",
        description="covariate-matched untreated stores",
        ref="engine/adjudicate/tests.py::match_controls",
    )
    before = b.emit(
        "availability_before",
        "Top-20 availability, treated, before",
        e["top20_availability_treated_before_pct"],
        "pct",
        source_system="wms",
        description="12:00 daily mean over the pre-period",
        ref="semantic_layer/kpis/on_shelf_availability.yaml",
    )
    after = b.emit(
        "availability_after",
        "Top-20 availability, treated, after",
        e["top20_availability_treated_after_pct"],
        "pct",
        source_system="wms",
        description="12:00 daily mean over the incident window",
        ref="semantic_layer/kpis/on_shelf_availability.yaml",
    )
    tickets = b.emit(
        "tickets",
        "Support tickets, size not available",
        e["support_tickets_count"],
        "count",
        kind="ticket_aggregate",
        source_system="support_tickets",
        method="count",
        description="ticket aggregate over the incident window",
        ref="engine/gather/unstructured.py",
    )
    notes = b.emit(
        "store_notes",
        "Treated stores whose notes flag unavailability",
        e["store_notes_flagging_treated"],
        "count",
        kind="corroborated_unstructured",
        source_system="store_notes",
        method="count",
        description="classifier output over shift notes",
        ref="llm/classify.py",
    )
    did = b.derived(
        "did",
        "Matched-control difference-in-differences",
        abs(measured["matched_control_did_pt"]),
        "pt",
        description="34 treated against 34 matched controls",
        ref="engine/adjudicate/tests.py::test_did",
    )
    dose = b.derived(
        "dose_response",
        "Dose-response correlation",
        measured["dose_response_r"],
        "r",
        description="OLS of store-level decline on store-level coverage gap",
        ref="engine/adjudicate/tests.py::test_dose_response",
    )
    attributed = b.derived(
        "h1_attributed",
        "Attributed to availability collapse",
        measured["h1_attribution_inr_cr"],
        "INR_CR",
        description="the share of the qualified residual this hypothesis accounts for",
        ref="engine/adjudicate/gate.py",
    )
    unattributed = b.derived(
        "unattributed",
        "Unattributed residual",
        measured["unattributed_residual_inr_cr"],
        "INR_CR",
        description="the residual no surviving hypothesis accounts for",
        ref="engine/adjudicate/gate.py",
    )
    coverage = b.derived(
        "coverage",
        "Share of the qualified residual accounted for",
        measured["h1_attribution_share"],
        "ratio",
        description="what the surviving hypotheses together explain",
        ref="engine/adjudicate/gate.py::coverage",
    )
    published = b.derived(
        "confidence_published",
        "Confidence, as published",
        config["confidence"]["calibrated"],
        "ratio",
        description=(
            "the weighted score after the isotonic map fitted on closed cases; "
            "the organisation's own record on cases like this one"
        ),
        ref="engine/confidence/calibration.py::fit_calibration",
    )
    volume_share = b.derived(
        "affected_volume_share",
        "Affected SKU share of West volume",
        m["affected_sku_volume_share"],
        "ratio",
        description="demand-weighted share of the SKUs that lost availability",
        ref="engine/gather/structured.py",
    )

    h1 = Hypothesis(
        hypothesis_id="H1",
        label="Availability collapse on top SKUs",
        description=(
            "Treated stores lost shelf availability on the top-20 SKUs, so demand "
            "that arrived could not convert."
        ),
        status="supported",
        tests=(
            _test(layer, "precedence", passed=True, detail="cause onset precedes effect onset"),
            _test(
                layer, "sufficiency", passed=True,
                statistic=m["affected_sku_volume_share"],
                detail="affected volume share times observed elasticity clears the residual",
            ),
            _test(
                layer, "dose_response", passed=True,
                statistic=measured["dose_response_r"],
                detail="store-level coverage gap against store-level decline",
            ),
            _test(layer, "specificity", passed=True, detail="present against absent, Welch's t"),
            _test(
                layer, "did", passed=True,
                statistic=measured["matched_control_did_pt"],
                p_value=measured["matched_control_did_max_p"],
                detail="matched-control estimate, parallel-trends pre-test passed",
            ),
            _test(
                layer, "confounder_screen", passed=True,
                detail="promo, weather, competitor openings and staffing balanced across groups",
            ),
        ),
        required_sources=("wms", "erp"),
        missing_sources=(),
        verifiable=True,
        attributed_share=measured["h1_attribution_share"],
        attributed=attributed,
        residual_held=None,
        evidence_ids=(
            did.evidence_id, dose.evidence_id, before.evidence_id, after.evidence_id,
            treated.evidence_id, controls.evidence_id, tickets.evidence_id,
            notes.evidence_id, attributed.evidence_id, volume_share.evidence_id,
        ),
    )
    h2 = Hypothesis(
        hypothesis_id="H2",
        label="Competitor promotion",
        description=(
            "A competitor may have run a promotion in the same catchments. The two "
            "feeds that would test it are not held."
        ),
        status="live",
        tests=(
            _test(layer, "precedence", passed=False, detail="no competitor series to locate an onset in"),
        ),
        required_sources=("competitor_pricing", "competitor_footfall"),
        missing_sources=("competitor_pricing", "competitor_footfall"),
        verifiable=False,
        attributed_share=None,
        attributed=None,
        residual_held=unattributed,
        evidence_ids=(unattributed.evidence_id,),
    )
    h3 = Hypothesis(
        hypothesis_id="H3",
        label="Price increase",
        description="A price rise on some sub-categories suppressed volume.",
        status="eliminated",
        elimination_reason="sufficiency",
        tests=(
            _test(layer, "precedence", passed=True, detail="the price event precedes the decline"),
            _test(
                layer, "sufficiency", passed=False,
                statistic=competing["H3_price"]["volume_share"],
                detail=(
                    "modelled maximum impact falls short of the residual it would have "
                    "to explain"
                ),
            ),
        ),
        required_sources=("pricing", "pos"),
        missing_sources=(),
        verifiable=True,
        attributed_share=None,
        evidence_ids=(residual.evidence_id,),
    )
    h4 = Hypothesis(
        hypothesis_id="H4",
        label="Marketing spend cut",
        description=(
            "Regional marketing spend was cut. The strongest correlation in the data "
            "and it arrives after the decline has started."
        ),
        status="eliminated",
        elimination_reason="precedence",
        tests=(
            _test(
                layer, "precedence", passed=False,
                statistic=competing["H4_marketing"]["onset_offset_days"],
                detail="the cut's changepoint falls after the decline's changepoint",
            ),
        ),
        required_sources=("promo_calendar", "marketing_spend"),
        missing_sources=(),
        verifiable=True,
        attributed_share=None,
        evidence_ids=(residual.evidence_id,),
    )
    h5 = Hypothesis(
        hypothesis_id="H5",
        label="Customer complaints",
        description="Service and quality complaints rose in West over the same window.",
        status="eliminated",
        elimination_reason="precedence_and_confounder",
        tests=(
            _test(
                layer, "precedence", passed=False,
                statistic=competing["H5_complaints"]["onset_offset_days"],
                detail="the complaint changepoint falls after the decline's",
            ),
            _test(
                layer, "confounder_screen", passed=False,
                detail="an empty shelf produces the same complaint signature",
            ),
        ),
        required_sources=("competitor_pricing", "competitor_footfall"),
        missing_sources=("competitor_pricing", "competitor_footfall"),
        verifiable=False,
        attributed_share=None,
        evidence_ids=(residual.evidence_id,),
    )

    return Adjudication(
        **_shell(config),
        status="closed",
        headline_movement=headline,
        attributed=(calendar,),
        qualified_residual=residual,
        materiality=limit,
        empirical_band=band,
        gates=(),
        hypotheses=(h1, h2, h3, h4, h5),
        coverage=measured["h1_attribution_share"],
        confidence=_confidence(layer, config["confidence"]),
        triggers_fired=(),
        evidence=tuple(b.ledger),
    )


def _build_2467(config, layer):
    """East, conversion rate, Oct 2025 -> INSUFFICIENT EVIDENCE."""
    from engine.contracts import Adjudication, Hypothesis

    t = config["targets"]
    e = config["evidence"]
    cal = config["calibration"]
    b = _Builder(layer, "2467")
    materiality = layer.kpis[config["kpi"]].thresholds.materiality

    headline = b.derived(
        "headline",
        "East conversion rate, month on month",
        t["headline_movement_pt"],
        "pt",
        description=(
            f"{t['conversion_before_pt']} down to {t['conversion_after_pt']}, "
            "instrumented stores only"
        ),
        ref="semantic_layer/kpis/conversion_rate.yaml::formula_sql",
    )
    before = b.emit(
        "conversion_before",
        "East conversion rate, September",
        t["conversion_before_pt"],
        "pt",
        description="transactions over footfall, instrumented stores",
        ref="semantic_layer/kpis/conversion_rate.yaml::formula_sql",
    )
    after = b.emit(
        "conversion_after",
        "East conversion rate, October",
        t["conversion_after_pt"],
        "pt",
        description="transactions over footfall, instrumented stores",
        ref="semantic_layer/kpis/conversion_rate.yaml::formula_sql",
    )
    attributed_pt = b.derived(
        "calendar_mix",
        "Calendar and mix attributed",
        abs(t["calendar_and_mix_attributed_pt"]),
        "pt",
        description="the part of the movement the calendar and mix account for",
        ref="engine/contribution/decomposition.py",
    )
    residual = b.derived(
        "residual",
        "Qualified residual",
        abs(t["qualified_residual_pt"]),
        "pt",
        description="the movement neither calendar nor mix accounts for",
        ref="engine/qualify/gate.py",
    )
    equivalent = b.derived(
        "residual_revenue",
        "Qualified residual, revenue equivalent",
        t["qualified_residual_revenue_equivalent_inr_cr"],
        "INR_CR",
        description="the residual conversion points priced against East revenue",
        ref="engine/qualify/materiality.py",
    )
    limit = b.threshold(
        "materiality",
        "Materiality threshold",
        materiality.value,
        materiality.unit,
        description=f"the residual clears {materiality.display} comfortably",
        ref="semantic_layer/kpis/conversion_rate.yaml::thresholds.materiality",
    )
    revenue = b.emit(
        "east_revenue",
        "East net revenue, monthly",
        t["east_net_revenue_monthly_inr_cr"],
        "INR_CR",
        description=f"{t['east_store_count']} stores",
        ref="semantic_layer/kpis/net_revenue.yaml::formula_sql",
    )
    stores = b.emit(
        "east_stores",
        "Stores in scope",
        t["east_store_count"],
        "count",
        method="count",
        source_system="erp",
        description="every East store",
        ref="engine/db.py::execute_governed",
    )
    staleness = b.emit(
        "marketing_staleness",
        "Marketing feed staleness",
        e["marketing_feed_staleness_hours"],
        "hours",
        source_system="marketing_spend",
        method="compare",
        description="hours since the marketing feed last landed",
        ref="engine/validate/checks.py",
    )
    news = b.emit(
        "competitor_news",
        "Competitor news items",
        e["competitor_news_items"],
        "count",
        kind="news_item",
        source_system="competitor_news",
        method="count",
        description="external mentions of competitor activity in the catchments",
        ref="engine/gather/external.py",
    )
    mentions = b.emit(
        "review_mentions",
        "Competitor mentions in reviews",
        e["competitor_review_mentions"],
        "count",
        kind="social_mention",
        source_system="review_feed",
        method="count",
        description="public review and social mentions",
        ref="engine/gather/external.py",
    )
    closures = b.emit(
        "road_closures",
        "Road closures in store notes",
        e["road_closures_in_notes"],
        "count",
        kind="store_note",
        source_system="store_notes",
        method="count",
        description="shift notes reporting a closed road in the catchment",
        ref="llm/classify.py",
    )
    accuracy = b.derived(
        "calibration_accuracy",
        "Historical accuracy, competitor-attribution cases",
        cal["competitor_attribution_accuracy"],
        "ratio",
        description=f"over {cal['competitor_attribution_sample']} closed cases",
        ref="engine/confidence/calibration.py",
    )
    floor = b.threshold(
        "publication_floor",
        "Publication floor",
        cal["publication_floor"],
        "ratio",
        description="below this, a case type is not published",
        ref="semantic_layer/adjudication.yaml::triggers.T7",
    )
    sample = b.derived(
        "calibration_sample",
        "Closed cases behind that accuracy",
        cal["competitor_attribution_sample"],
        "count",
        description="competitor-attribution cases in the trailing twelve months",
        ref="engine/confidence/calibration.py",
    )

    h2 = Hypothesis(
        hypothesis_id="H2",
        label="Competitor promotion",
        description=(
            "A competitor promotion took share in the East catchments. Neither the "
            "pricing feed nor the catchment footfall feed is held."
        ),
        status="live",
        tests=(
            _test(layer, "precedence", passed=False, detail="no competitor series to locate an onset in"),
        ),
        required_sources=("competitor_pricing", "competitor_footfall"),
        missing_sources=("competitor_pricing", "competitor_footfall"),
        verifiable=False,
        attributed_share=None,
        residual_held=equivalent,
        evidence_ids=(news.evidence_id, mentions.evidence_id, equivalent.evidence_id),
    )
    h6 = Hypothesis(
        hypothesis_id="H6",
        label="Catchment footfall decline",
        description=(
            "Fewer people entered the catchments. Counters cover only part of the "
            "estate, so the test would run on a subset."
        ),
        status="live",
        tests=(
            _test(layer, "precedence", passed=False, detail="counter coverage is too thin to locate an onset"),
        ),
        required_sources=("footfall",),
        missing_sources=(),
        verifiable=True,
        attributed_share=None,
        evidence_ids=(closures.evidence_id, residual.evidence_id),
    )

    return Adjudication(
        **_shell(config),
        status="closed",
        headline_movement=headline,
        attributed=(attributed_pt,),
        qualified_residual=residual,
        materiality=limit,
        empirical_band=None,
        gates=(),
        hypotheses=(h2, h6),
        coverage=0.0,
        confidence=_empty_confidence(),
        triggers_fired=tuple(config["triggers_expected"]),
        evidence=tuple(b.ledger),
    )


def _gate_case(config, layer, *, detail: str, subjects: tuple[str, ...], evidence):
    """A case that a gate killed. The narrative is the kill, not a verdict."""
    from engine.contracts import Adjudication, CheckResult, GateResult

    gate = config["gate"]
    spec = layer.adjudication.gates[gate["gate_id"]]
    result = GateResult(
        gate_id=gate["gate_id"],
        name=spec.name,
        passed=False,
        outcome_code=gate["outcome_code"],
        detail=detail,
        checks=(
            CheckResult(
                check_id=gate["outcome_code"].lower(),
                name=spec.name,
                order=1,
                outcome=gate["outcome_code"],
                detail=detail,
                subjects=subjects,
            ),
        ),
        evidence_ids=tuple(item.evidence_id for item in evidence),
    )
    return result


def _build_2470(config, layer):
    """West, 12 Nov 2025, daily -> Gate 1 kill. No case opened."""
    from engine.contracts import Adjudication

    t = config["targets"]
    b = _Builder(layer, "2470")
    materiality = layer.kpis[config["kpi"]].thresholds.materiality

    headline = b.derived(
        "headline",
        "Apparent West movement, 12 Nov",
        t["apparent_movement_pct"],
        "pct",
        description="what the daily feed showed before the incident was found",
        ref="semantic_layer/kpis/net_revenue.yaml::formula_sql",
    )
    failed = b.emit(
        "failed_feeds",
        "Store feeds that failed to load",
        t["failed_store_feeds"],
        "count",
        source_system="pos",
        method="count",
        description="feed_status rows reporting zero rows loaded on the incident date",
        ref="engine/validate/checks.py",
    )
    total = b.emit(
        "west_stores",
        "West stores expected",
        t["total_west_stores"],
        "count",
        source_system="erp",
        method="count",
        description="every West store the feed should carry",
        ref="engine/db.py::execute_governed",
    )
    share = b.derived(
        "failed_share",
        "Revenue share behind the failed feeds",
        config["incident"]["target_revenue_share"],
        "ratio",
        description="the share of West daily revenue the missing stores carry",
        ref="engine/validate/checks.py",
    )
    limit = b.threshold(
        "materiality",
        "Materiality threshold",
        materiality.value,
        materiality.unit,
        description="never reached; the gate killed the case first",
        ref="semantic_layer/kpis/net_revenue.yaml::thresholds.materiality",
    )
    detail = (
        f"{t['failed_store_feeds']} of {t['total_west_stores']} West store feeds "
        "loaded zero rows on this date. The movement is in the feed, not in the "
        "business, and the day is recoverable on reload."
    )
    gate = _gate_case(
        config, layer, detail=detail, subjects=("fact_feed_status",),
        evidence=(failed, total, share),
    )
    return Adjudication(
        **_shell(config),
        status="closed",
        headline_movement=headline,
        attributed=(),
        qualified_residual=share,
        materiality=limit,
        empirical_band=None,
        gates=(gate,),
        hypotheses=(),
        coverage=0.0,
        confidence=_empty_confidence(),
        triggers_fired=(),
        evidence=tuple(b.ledger),
    )


def _build_2471(config, layer):
    """All-India, Q-commerce fulfilment, weekly -> Gate 3. Monitoring only."""
    from engine.contracts import Adjudication

    t = config["targets"]
    b = _Builder(layer, "2471")
    contract = layer.kpis[config["kpi"]]

    available = b.emit(
        "weekly_points",
        "Weekly points available",
        t["weekly_points_available"],
        "count",
        source_system="qcomm_oms",
        method="count",
        description=f"weeks of history since launch on {t['launch_date']}",
        ref="semantic_layer/kpis/qcomm_fulfilment_rate.yaml::history_weeks",
    )
    required = b.threshold(
        "weekly_points_required",
        "Weekly points required",
        t["weekly_points_required"],
        "count",
        description="a seasonal baseline needs this much history before it means anything",
        ref="semantic_layer/kpis/qcomm_fulfilment_rate.yaml::baseline.min_history_weeks",
    )
    rate = b.emit(
        "fulfilment_rate",
        "Q-commerce fulfilment rate",
        config["series"]["fulfilment_rate_base_pct"],
        "pct",
        source_system="qcomm_oms",
        description=f"across {config['series']['dark_stores']} dark stores",
        ref="semantic_layer/kpis/qcomm_fulfilment_rate.yaml::formula_sql",
    )
    dark = b.emit(
        "dark_stores",
        "Dark stores in scope",
        config["series"]["dark_stores"],
        "count",
        source_system="erp",
        method="count",
        description="fulfilment nodes serving the quick-commerce channel",
        ref="engine/db.py::execute_governed",
    )
    # A KPI with no agreed materiality limit cannot open a case at all,
    # which is a second, independent reason this one is monitoring only.
    limit = b.threshold(
        "materiality",
        "Materiality threshold",
        0.0,
        "count",
        description=(
            "none agreed for this KPI. A movement with no limit to clear cannot "
            "open a case even once it has the history."
        ),
        ref="semantic_layer/kpis/qcomm_fulfilment_rate.yaml::thresholds",
    )
    detail = (
        f"{t['weekly_points_available']} weekly points against the "
        f"{t['weekly_points_required']} a baseline requires. Monitoring only until "
        "the history is there."
    )
    gate = _gate_case(
        config, layer, detail=detail, subjects=(contract.kpi,),
        evidence=(available, required),
    )
    return Adjudication(
        **_shell(config),
        status="closed",
        headline_movement=rate,
        attributed=(),
        qualified_residual=available,
        materiality=limit,
        empirical_band=None,
        gates=(gate,),
        hypotheses=(),
        coverage=0.0,
        confidence=_empty_confidence(),
        triggers_fired=(),
        evidence=tuple(b.ledger),
    )


def _build_2472(config, layer):
    """South, net revenue, Sep 2025 -> CASE OPENED, adjudication in progress.

    The subtle one. Flat is not fine when the calendar expected a festival
    rise, and this is the scenario that shows why a residual opens a case
    and a headline does not.
    """
    from engine.contracts import Adjudication

    t = config["targets"]
    b = _Builder(layer, "2472")
    materiality = layer.kpis[config["kpi"]].thresholds.materiality

    headline = b.derived(
        "headline",
        "South net revenue, month on month",
        t["headline_movement_pct"],
        "pct",
        description="flat against August, which is what a dashboard would show",
        ref="semantic_layer/kpis/net_revenue.yaml::formula_sql",
    )
    expected = b.derived(
        "calendar_expected",
        "Calendar-expected movement",
        t["calendar_expected_movement_pct"],
        "pct",
        description=(
            f"{t['festival']} fell in September; the calendar model expected a rise"
        ),
        ref="semantic_layer/kpis/net_revenue.yaml::baseline.calendar_regressors",
    )
    residual = b.derived(
        "residual",
        "Qualified residual",
        abs(t["qualified_residual_pt"]),
        "pt",
        description="flat against an expected rise is a residual of that size",
        ref="engine/qualify/gate.py",
    )
    limit = b.threshold(
        "materiality",
        "Materiality threshold",
        materiality.value,
        materiality.unit,
        description=f"the residual clears {materiality.display}",
        ref="semantic_layer/kpis/net_revenue.yaml::thresholds.materiality",
    )
    return Adjudication(
        **_shell(config),
        status=config["gate"]["adjudication_status"],
        headline_movement=headline,
        attributed=(expected,),
        qualified_residual=residual,
        materiality=limit,
        empirical_band=None,
        gates=(),
        hypotheses=(),
        coverage=0.0,
        confidence=_empty_confidence(),
        triggers_fired=(),
        evidence=tuple(b.ledger),
    )


__all__ = [
    "AS_OF",
    "CASE_IDS",
    "CaseFileError",
    "build_all",
    "build_case",
    "frozen",
    "scenarios",
]
