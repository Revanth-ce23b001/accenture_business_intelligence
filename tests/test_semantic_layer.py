"""The semantic layer loads, validates, and rejects malformed contracts."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from semantic_layer.schema import (
    MANDATORY_PLAYBOOK_FIELDS,
    SemanticLayerError,
    load_semantic_layer,
    load_timed,
)

ROOT = Path(__file__).resolve().parents[1]
LAYER_DIR = ROOT / "semantic_layer"

EXPECTED_KPIS = {
    "net_revenue",
    "transactions",
    "conversion_rate",
    "avg_selling_price",
    "on_shelf_availability",
    "qcomm_fulfilment_rate",
}

EXPECTED_HYPOTHESES = {
    "stock_out",
    "price_increase",
    "promo_lapse",
    "competitor_action",
    "staffing_shortfall",
    "footfall_decline",
    "mix_shift",
    "channel_cannibalisation",
    "weather",
    "transport_disruption",
    "data_incident",
    "calendar_shift",
    "store_closure",
    "assortment_change",
}


@pytest.fixture(scope="module")
def layer():
    return load_semantic_layer()


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A writable copy of the real semantic layer, for mutation tests."""
    destination = tmp_path / "semantic_layer"
    shutil.copytree(
        LAYER_DIR,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.py"),
    )
    return destination


def _rewrite(path: Path, mutate) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


# --- structure --------------------------------------------------------------


def test_six_kpi_contracts(layer):
    assert set(layer.kpis) == EXPECTED_KPIS


def test_fourteen_hypothesis_templates(layer):
    assert set(layer.causal_graph.hypotheses) == EXPECTED_HYPOTHESES


#: P2 wrote five. P11 added `manager_call_down`, the cheapest resolution
#: path there is and the one #2467 ranks first. Named rather than counted,
#: so adding a playbook has to be a deliberate edit here.
EXPECTED_PLAYBOOKS = {
    "availability_recovery",
    "competitor_intelligence",
    "footfall_instrumentation",
    "manager_call_down",
    "price_correction",
    "staffing_uplift",
}


def test_the_playbook_set_is_the_expected_one(layer):
    assert set(layer.playbooks) == EXPECTED_PLAYBOOKS


def test_every_playbook_driver_is_a_real_hypothesis(layer):
    for playbook in layer.playbooks.values():
        assert playbook.driver in layer.causal_graph.hypotheses


def test_every_playbook_recovery_curve_resolves(layer):
    for playbook in layer.playbooks.values():
        assert playbook.recovery_curve_ref in layer.recovery_curves.curves


# --- net_revenue, per the brief --------------------------------------------


def test_net_revenue_materiality(layer):
    materiality = layer.kpis["net_revenue"].thresholds.materiality
    assert materiality is not None
    assert materiality.value == pytest.approx(0.50)
    assert materiality.unit == "INR_CR"


def test_net_revenue_baseline_and_history(layer):
    baseline = layer.kpis["net_revenue"].baseline
    assert baseline.method == "stl_plus_calendar_regression"
    assert baseline.min_history_weeks == 26


def test_net_revenue_definition_excludes_b2b_and_transfers(layer):
    kpi = layer.kpis["net_revenue"]
    combined = f"{kpi.definition}\n{kpi.formula_sql}"
    assert "B2B" in combined
    assert "TRANSFER" in combined
    assert "channel <> 'B2B'" in kpi.formula_sql
    assert "txn_type <> 'TRANSFER'" in kpi.formula_sql


def test_net_revenue_definition_nets_off_returns_discount_and_tax(layer):
    definition = layer.kpis["net_revenue"].definition.lower()
    for term in ("return", "discount", "tax"):
        assert term in definition
    for column in ("returns_amount", "discount_amount", "tax_amount"):
        assert column in layer.kpis["net_revenue"].formula_sql


def test_conversion_rate_materiality(layer):
    materiality = layer.kpis["conversion_rate"].thresholds.materiality
    assert materiality is not None
    assert materiality.value == pytest.approx(0.5)
    assert materiality.unit == "pt"


# --- the deliberate gap, case #2471 ----------------------------------------


def test_qcomm_has_seven_weeks_of_history(layer):
    assert layer.kpis["qcomm_fulfilment_rate"].history_weeks == 7


def test_qcomm_has_no_baseline_method(layer):
    assert layer.kpis["qcomm_fulfilment_rate"].baseline.method is None


def test_qcomm_has_no_materiality(layer):
    assert layer.kpis["qcomm_fulfilment_rate"].thresholds.materiality is None


def test_qcomm_fails_the_history_gate(layer):
    kpi = layer.kpis["qcomm_fulfilment_rate"]
    assert kpi.has_sufficient_history() is False
    assert kpi.history_weeks < kpi.baseline.min_history_weeks


def test_qcomm_cannot_open_a_case(layer):
    assert layer.kpis["qcomm_fulfilment_rate"].can_open_a_case() is False


def test_every_other_kpi_can_open_a_case(layer):
    for name, kpi in layer.kpis.items():
        if name == "qcomm_fulfilment_rate":
            continue
        assert kpi.can_open_a_case() is True, f"{name} cannot open a case"


# --- T3 determinism, case #2467 --------------------------------------------


def test_competitor_action_requires_two_unheld_sources(layer):
    hypothesis = layer.causal_graph.hypotheses["competitor_action"]
    assert set(hypothesis.required_sources) == {"competitor_pricing", "competitor_footfall"}


def test_competitor_action_only_has_news(layer):
    hypothesis = layer.causal_graph.hypotheses["competitor_action"]
    assert hypothesis.available_sources == ["competitor_news"]


def test_competitor_action_excludes_sufficiency_and_did(layer):
    tests = layer.causal_graph.hypotheses["competitor_action"].applicable_tests
    assert "sufficiency" not in tests
    assert "did" not in tests
    assert set(tests) == {"precedence", "dose_response", "specificity", "confounder_screen"}


def test_competitor_action_carries_the_calibration_note(layer):
    note = layer.causal_graph.hypotheses["competitor_action"].calibration_note
    assert note == "58% accurate on n=12 closed cases — below publication floor"


def test_t3_is_a_set_difference_not_a_judgement(layer):
    """The missing-source computation is what makes T3 deterministic."""
    hypothesis = layer.causal_graph.hypotheses["competitor_action"]
    assert hypothesis.missing_sources() == ["competitor_pricing", "competitor_footfall"]
    assert hypothesis.is_structurally_unverifiable() is True


def test_competitor_action_is_the_only_unverifiable_hypothesis(layer):
    assert layer.unverifiable_hypotheses() == ["competitor_action"]


def test_unheld_sources_are_declared_unheld(layer):
    sources = layer.causal_graph.sources
    assert sources["competitor_pricing"].held is False
    assert sources["competitor_footfall"].held is False
    assert sources["competitor_news"].held is True


# --- causal graph integrity -------------------------------------------------


def test_every_hypothesis_affects_known_kpis(layer):
    for name, hypothesis in layer.causal_graph.hypotheses.items():
        for kpi in hypothesis.affects:
            assert kpi in layer.kpis, f"{name} affects unknown kpi {kpi}"


def test_every_hypothesis_names_known_tests(layer):
    for name, hypothesis in layer.causal_graph.hypotheses.items():
        for test in hypothesis.applicable_tests:
            assert test in layer.adjudication.tests, f"{name} names unknown test {test}"


def test_priors_are_probabilities_but_not_a_partition(layer):
    priors = [h.prior for h in layer.causal_graph.hypotheses.values()]
    assert all(0.0 < p < 1.0 for p in priors)
    # Hypotheses are not mutually exclusive; the priors deliberately do not
    # sum to 1.0. Asserting that they do would be wrong.
    assert sum(priors) != pytest.approx(1.0)


# --- adjudication constants -------------------------------------------------


def test_confidence_weights_sum_to_one(layer):
    weights = [c.weight for c in layer.adjudication.confidence.components.values()]
    assert sum(weights) == pytest.approx(1.0)


def test_confidence_weights_match_claude_md(layer):
    components = layer.adjudication.confidence.components
    assert components["s1"].weight == pytest.approx(0.28)
    assert components["s2"].weight == pytest.approx(0.20)
    assert components["s3"].weight == pytest.approx(0.16)
    assert components["s4"].weight == pytest.approx(0.14)
    assert components["s5"].weight == pytest.approx(0.10)
    assert components["s6"].weight == pytest.approx(0.12)


def test_all_seven_reliability_tiers_have_weights(layer):
    weights = layer.adjudication.reliability.weights
    assert weights["structured_query"] == pytest.approx(0.95)
    assert weights["derived_estimate"] == pytest.approx(0.85)
    assert weights["corroborated_unstructured"] == pytest.approx(0.80)
    assert weights["ticket_aggregate"] == pytest.approx(0.75)
    assert weights["store_note"] == pytest.approx(0.55)
    assert weights["news_item"] == pytest.approx(0.45)
    assert weights["social_mention"] == pytest.approx(0.35)


def test_eight_triggers_are_defined(layer):
    assert set(layer.adjudication.triggers) == {f"T{i}" for i in range(1, 9)}


def test_six_tests_are_defined(layer):
    assert len(layer.adjudication.tests) == 6
    assert {t.test_id for t in layer.adjudication.tests.values()} == set(range(1, 7))


def test_hard_gates_are_tests_one_and_two(layer):
    tests = layer.adjudication.tests
    assert tests["precedence"].type == "hard_gate"
    assert tests["sufficiency"].type == "hard_gate"
    assert tests["confounder_screen"].type == "cap"


def test_did_carries_the_dominant_weight(layer):
    assert layer.adjudication.tests["did"].weight == pytest.approx(0.45)


def test_missing_source_cap_forces_t3(layer):
    cap = layer.adjudication.confidence.caps["required_source_missing"]
    assert cap.ceiling == pytest.approx(0.45)
    assert cap.forced_trigger == "T3"


# --- malformed contracts are rejected ---------------------------------------


@pytest.mark.parametrize("field", MANDATORY_PLAYBOOK_FIELDS)
def test_playbook_missing_a_mandatory_field_is_rejected(sandbox, field):
    """Accept criterion: validation fails if any mandatory field is missing."""
    path = sandbox / "playbooks" / "availability_recovery.yaml"
    _rewrite(path, lambda p: p.pop(field))
    with pytest.raises(SemanticLayerError) as excinfo:
        load_semantic_layer(sandbox)
    assert field in str(excinfo.value)


def test_kpi_with_unknown_default_grain_is_rejected(sandbox):
    path = sandbox / "kpis" / "net_revenue.yaml"
    _rewrite(path, lambda p: p["grain"].__setitem__("default", "hourly"))
    with pytest.raises(SemanticLayerError):
        load_semantic_layer(sandbox)


def test_kpi_referencing_an_unknown_kpi_is_rejected(sandbox):
    path = sandbox / "kpis" / "net_revenue.yaml"
    _rewrite(path, lambda p: p["drivers"].append("gross_margin"))
    with pytest.raises(SemanticLayerError, match="unknown kpi"):
        load_semantic_layer(sandbox)


def test_kpi_with_unknown_key_is_rejected(sandbox):
    path = sandbox / "kpis" / "net_revenue.yaml"
    _rewrite(path, lambda p: p.__setitem__("matriality", 0.5))
    with pytest.raises(SemanticLayerError):
        load_semantic_layer(sandbox)


def test_access_policy_default_persona_must_exist(sandbox):
    path = sandbox / "kpis" / "net_revenue.yaml"
    _rewrite(path, lambda p: p["access_policy"].__setitem__("default_persona", "ghost"))
    with pytest.raises(SemanticLayerError, match="default_persona"):
        load_semantic_layer(sandbox)


def test_hypothesis_with_unknown_source_is_rejected(sandbox):
    path = sandbox / "causal_graph.yaml"
    _rewrite(
        path,
        lambda p: p["hypotheses"]["stock_out"]["required_sources"].append("crystal_ball"),
    )
    with pytest.raises(SemanticLayerError, match="unknown source"):
        load_semantic_layer(sandbox)


def test_hypothesis_cannot_claim_an_unheld_source_is_available(sandbox):
    """The guard that stops T3 being quietly deleted."""
    path = sandbox / "causal_graph.yaml"
    _rewrite(
        path,
        lambda p: p["hypotheses"]["competitor_action"]["available_sources"].append(
            "competitor_pricing"
        ),
    )
    with pytest.raises(SemanticLayerError, match="does not hold"):
        load_semantic_layer(sandbox)


def test_hypothesis_affecting_an_unknown_kpi_is_rejected(sandbox):
    path = sandbox / "causal_graph.yaml"
    _rewrite(path, lambda p: p["hypotheses"]["stock_out"]["affects"].append("footfall"))
    with pytest.raises(SemanticLayerError, match="unknown kpi"):
        load_semantic_layer(sandbox)


def test_hypothesis_naming_an_unknown_test_is_rejected(sandbox):
    path = sandbox / "causal_graph.yaml"
    _rewrite(path, lambda p: p["hypotheses"]["stock_out"]["applicable_tests"].append("vibes"))
    with pytest.raises(SemanticLayerError, match="unknown test"):
        load_semantic_layer(sandbox)


def test_hypothesis_cannot_confound_itself(sandbox):
    path = sandbox / "causal_graph.yaml"
    _rewrite(path, lambda p: p["hypotheses"]["stock_out"]["confounders"].append("stock_out"))
    with pytest.raises(SemanticLayerError, match="its own confounder"):
        load_semantic_layer(sandbox)


def test_prior_outside_zero_to_one_is_rejected(sandbox):
    path = sandbox / "causal_graph.yaml"
    _rewrite(path, lambda p: p["hypotheses"]["stock_out"].__setitem__("prior", 1.4))
    with pytest.raises(SemanticLayerError):
        load_semantic_layer(sandbox)


def test_dangling_recovery_curve_reference_is_rejected(sandbox):
    path = sandbox / "playbooks" / "availability_recovery.yaml"
    _rewrite(path, lambda p: p.__setitem__("recovery_curve_ref", "wishful_thinking"))
    with pytest.raises(SemanticLayerError, match="unknown recovery curve"):
        load_semantic_layer(sandbox)


def test_playbook_with_unknown_driver_is_rejected(sandbox):
    path = sandbox / "playbooks" / "availability_recovery.yaml"
    _rewrite(path, lambda p: p.__setitem__("driver", "bad_luck"))
    with pytest.raises(SemanticLayerError, match="not a"):
        load_semantic_layer(sandbox)


def test_playbook_monitoring_an_unknown_kpi_is_rejected(sandbox):
    path = sandbox / "playbooks" / "availability_recovery.yaml"
    _rewrite(path, lambda p: p["monitoring_plan"].__setitem__("primary_kpi", "morale"))
    with pytest.raises(SemanticLayerError, match="unknown kpi"):
        load_semantic_layer(sandbox)


def test_checkpoint_beyond_the_horizon_is_rejected(sandbox):
    path = sandbox / "playbooks" / "availability_recovery.yaml"
    _rewrite(
        path,
        lambda p: p["monitoring_plan"]["checkpoints"].append(
            {"at_week": 99, "expect": "miracle"}
        ),
    )
    with pytest.raises(SemanticLayerError, match="horizon"):
        load_semantic_layer(sandbox)


def test_recovery_curve_with_inverted_quartiles_is_rejected(sandbox):
    path = sandbox / "recovery_curves.yaml"
    _rewrite(path, lambda p: p["curves"]["availability_restock"].__setitem__("p25", 0.99))
    with pytest.raises(SemanticLayerError):
        load_semantic_layer(sandbox)


def test_recovery_above_full_attributable_loss_is_rejected(sandbox):
    path = sandbox / "recovery_curves.yaml"
    _rewrite(path, lambda p: p["curves"]["availability_restock"].__setitem__("p75", 1.4))
    with pytest.raises(SemanticLayerError):
        load_semantic_layer(sandbox)


def test_confidence_weights_that_do_not_sum_to_one_are_rejected(sandbox):
    path = sandbox / "adjudication.yaml"
    _rewrite(
        path,
        lambda p: p["confidence"]["components"]["s1"].__setitem__("weight", 0.50),
    )
    with pytest.raises(SemanticLayerError, match="sum to"):
        load_semantic_layer(sandbox)


def test_hard_gate_carrying_a_weight_is_rejected(sandbox):
    path = sandbox / "adjudication.yaml"
    _rewrite(path, lambda p: p["tests"]["precedence"].__setitem__("weight", 0.3))
    with pytest.raises(SemanticLayerError, match="not voted on"):
        load_semantic_layer(sandbox)


def test_cap_forcing_an_unknown_trigger_is_rejected(sandbox):
    path = sandbox / "adjudication.yaml"
    _rewrite(
        path,
        lambda p: p["confidence"]["caps"]["required_source_missing"].__setitem__(
            "forced_trigger", "T99"
        ),
    )
    with pytest.raises(SemanticLayerError, match="unknown trigger"):
        load_semantic_layer(sandbox)


def test_verdict_branches_must_be_ordered(sandbox):
    path = sandbox / "adjudication.yaml"
    _rewrite(path, lambda p: p["verdict"]["explained"].__setitem__("min_coverage", 0.10))
    with pytest.raises(SemanticLayerError, match="less coverage"):
        load_semantic_layer(sandbox)


def test_empty_file_is_rejected(sandbox):
    (sandbox / "causal_graph.yaml").write_text("", encoding="utf-8")
    with pytest.raises(SemanticLayerError, match="empty"):
        load_semantic_layer(sandbox)


def test_unparseable_yaml_is_rejected(sandbox):
    (sandbox / "causal_graph.yaml").write_text("a:\n  - b\n c: [\n", encoding="utf-8")
    with pytest.raises(SemanticLayerError, match="not valid YAML"):
        load_semantic_layer(sandbox)


def test_missing_directory_is_rejected(tmp_path):
    with pytest.raises(SemanticLayerError, match="missing semantic-layer directory"):
        load_semantic_layer(tmp_path)


# --- performance ------------------------------------------------------------


def test_loading_all_files_is_under_200ms():
    """Accept criterion: loading all files takes < 200 ms.

    THE BAR IS UNCHANGED. What changed is the estimator, and only after it
    failed twice on code that passes: the same load measured five times in
    one run came back [108, 119, 250, 286, 440] ms, a four-fold spread on
    identical work. The suite holds a 178 MB warehouse and the generated
    world resident by the time this runs, and pydantic validation
    allocates heavily, so every sample carries whatever garbage collection
    and page faults happened to land on it.

    The MINIMUM is the right statistic for a microbenchmark on a busy
    machine, and it is what `timeit` documents for the same reason: noise
    only ever adds time, never removes it, so the fastest sample is the
    one least polluted by everything that is not the measurement. A median
    over five samples is a median over five different amounts of
    interference.

    Measured alone, this load takes 25-45 ms. If the minimum over fifteen
    samples cannot get under 200 ms, the loader really has become four
    times slower and the failure is a true one.
    """
    load_semantic_layer()  # warm the yaml/pydantic import path
    timings = sorted(load_timed()[1] for _ in range(15))
    assert timings[0] < 200.0, (
        f"fastest load {timings[0]:.1f} ms over {len(timings)} samples "
        f"(slowest {timings[-1]:.1f} ms)"
    )
