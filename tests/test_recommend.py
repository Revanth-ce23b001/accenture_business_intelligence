"""P11 — RECOMMEND.

The acceptance criteria, and the Number Registry values this stage owns.

    #2451  cost INR 18 L  ·  recovery INR 2.3-3.1 Cr from the curve  ·
           owner "Regional Supply Chain Lead" from the KPI contract  ·
           the DC-allocation linked case created
    #2467  three ranked investigations  ·  the INR 1.2 Cr refusal, with
           its cost and its error rate
    a playbook missing a mandatory field fails validation

REGISTRY GAPS. Two figures CLAUDE.md quotes for #2467 do not reproduce
from East's own data, and neither is adjusted here (rule 10). Both are
pinned by a `test_*_does_not_reproduce` that asserts the MEASURED value
and states why, so closing the gap fails the test rather than drifting
past it. See the two at the bottom of this file.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine.contracts import Recommendation
from engine.qualify.restraint import register_case
from engine.recommend import (
    CostBasis,
    CostError,
    MatchError,
    RecommendationResult,
    RecommendError,
    RecommendRequest,
    acquisition_playbook,
    candidates,
    compute_cost,
    compute_recovery,
    confidence_label,
    error_rate,
    match_playbook,
    plan_call_down,
    rank_investigations,
    recommend,
    refuse,
    resolve_owner,
    to_contract,
)
from security.policy import User
from semantic_layer.schema import SemanticLayerError, load_semantic_layer

CR = 0.02  # INR crore tolerance
LAKH = 0.1  # INR lakh tolerance

CLOCK = datetime(2025, 11, 30, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyst() -> User:
    return User(user_id="u.recommend", persona="cxo")


@pytest.fixture
def registry(warehouse):
    """A clean case registry and gap register for one test.

    Both are session-scoped warehouse tables and this module writes to
    them, so each test clears what it is about to use rather than
    inheriting rows from the last one.
    """
    warehouse.execute("DELETE FROM case_registry")
    warehouse.execute("DELETE FROM data_gap_register")
    yield warehouse
    warehouse.execute("DELETE FROM case_registry")
    warehouse.execute("DELETE FROM data_gap_register")


def crore(value: float, layer) -> float:
    return value * layer.warehouse.units.inr_per_crore


@pytest.fixture
def basis_2451(layer) -> CostBasis:
    """#2451's measured quantities. 34 treated of 140 West stores."""
    return CostBasis(
        treated_store_count=34,
        affected_store_count=140,
        revenue_at_risk_inr=crore(4.10, layer),
    )


@pytest.fixture
def request_2451(basis_2451, layer) -> RecommendRequest:
    return RecommendRequest(
        case_id="2451",
        kpi="net_revenue",
        scope="West",
        grain="monthly",
        period="2025-11",
        verdict="PARTIALLY_EXPLAINED",
        leading_driver="stock_out",
        attributable_inr=crore(3.24, layer),
        residual_inr=crore(4.10, layer),
        basis=basis_2451,
    )


@pytest.fixture
def east_uninstrumented(warehouse) -> int:
    """East's OWN uninstrumented count, measured, not the registry's 109."""
    return int(
        warehouse.execute(
            "SELECT COUNT(*) FROM dim_store "
            "WHERE region = 'East' AND NOT has_footfall_counter"
        ).fetchone()[0]
    )


@pytest.fixture
def request_2467(east_uninstrumented, layer) -> RecommendRequest:
    return RecommendRequest(
        case_id="2467",
        kpi="conversion_rate",
        scope="East",
        grain="monthly",
        period="2025-10",
        verdict="INSUFFICIENT_EVIDENCE",
        leading_driver=None,
        unresolved_drivers=("competitor_action", "footfall_decline", "price_increase"),
        residual_inr=crore(1.88, layer),
        basis=CostBasis(
            affected_store_count=58,
            uninstrumented_store_count=east_uninstrumented,
            revenue_at_risk_inr=crore(1.88, layer),
        ),
        triggers_fired=("T3", "T4", "T7"),
        missing_sources=("competitor_pricing", "competitor_footfall"),
        missing_for_hypothesis="competitor_action",
        indistinguishable_count=2,
        case_type_accuracy=0.58,
    )


@pytest.fixture
def result_2451(registry, analyst, request_2451, layer) -> RecommendationResult:
    register_case(
        registry,
        case_id="2451",
        kpi="net_revenue",
        scope="West",
        grain="monthly",
        period="2025-11",
        opened_at=CLOCK,
    )
    return recommend(registry, analyst, request_2451, layer=layer, clock=CLOCK)


@pytest.fixture
def result_2467(registry, analyst, request_2467, layer) -> RecommendationResult:
    register_case(
        registry,
        case_id="2467",
        kpi="conversion_rate",
        scope="East",
        grain="monthly",
        period="2025-10",
        opened_at=CLOCK,
    )
    return recommend(registry, analyst, request_2467, layer=layer, clock=CLOCK)


# ===========================================================================
# Matching — the recommendation attaches to the ADJUDICATED driver
# ===========================================================================


def test_playbook_matches_the_adjudicated_driver(layer):
    assert match_playbook("stock_out", layer).playbook == "availability_recovery"


def test_a_recovering_lever_beats_an_information_lever(layer):
    """`competitor_action` carries two playbooks. Only one recovers.

    Neither does, in fact — both are information levers — so the tie
    falls to lead time and the call-down (2 days) wins over the feed
    (2 weeks). The ordering rule is recommend.yaml's, not the module's.
    """
    ordered = [p.playbook for p in candidates("competitor_action", layer)]
    assert ordered == ["manager_call_down", "competitor_intelligence"]


def test_a_driver_with_no_playbook_is_reported_not_improvised(layer):
    with pytest.raises(MatchError, match="NO_PLAYBOOK_FOR_DRIVER"):
        match_playbook("data_incident", layer)


# ===========================================================================
# Ownership — from the KPI contract, never hardcoded
# ===========================================================================


def test_owner_is_the_regional_supply_chain_lead(layer):
    """Accept criterion: owner "Regional Supply Chain Lead" from the contract."""
    contract = layer.kpis["net_revenue"]
    owner = resolve_owner(contract, "stock_out")
    assert owner.title == "Regional Supply Chain Lead"
    assert owner.role == "regional_supply_chain_lead"
    assert owner.resolved_from == "kpi_contract.driver_ownership"


def test_the_contract_overrides_the_playbook_and_says_so(layer):
    """The playbook owns the lever; the contract owns the KPI. Contract wins."""
    contract = layer.kpis["net_revenue"]
    playbook = layer.playbooks["availability_recovery"]
    owner = resolve_owner(contract, "stock_out", playbook=playbook)
    assert playbook.owner_role == "supply_chain_director"
    assert owner.role == "regional_supply_chain_lead"
    assert owner.overrides_playbook
    assert "supply_chain_director" in owner.render()


def test_an_unnamed_driver_falls_back_to_the_kpi_owner(layer):
    owner = resolve_owner(layer.kpis["net_revenue"], "channel_cannibalisation")
    assert owner.role == "ecommerce_lead"
    owner = resolve_owner(layer.kpis["net_revenue"], None)
    assert owner.role == layer.kpis["net_revenue"].owner_role
    assert owner.resolved_from.endswith("default")


def test_a_contract_with_no_driver_ownership_falls_through(layer):
    """transactions has not been asked the question. Reported, not guessed."""
    contract = layer.kpis["transactions"]
    assert contract.driver_ownership is None
    owner = resolve_owner(contract, "stock_out")
    assert owner.role == contract.owner_role
    assert owner.resolved_from == "kpi_contract.owner_role"


def test_owner_is_not_hardcoded_anywhere_in_the_engine():
    """Rule 2's sibling: the title appears in YAML and nowhere in Python."""
    from pathlib import Path

    engine = Path(__file__).resolve().parents[1] / "engine"
    offenders = [
        path.name
        for path in engine.rglob("*.py")
        if "Regional Supply Chain Lead" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"the owner title is hardcoded in {offenders}"


# ===========================================================================
# Cost — a playbook rate times a measured quantity
# ===========================================================================


def test_2451_action_costs_18_lakh(layer, basis_2451):
    """Accept criterion: #2451 -> cost INR 18 L.

    INR 53,000 per store from the playbook, 34 treated stores from the
    case. 18.02, which displays as 18 — see the rounding gap pinned in
    tests/test_number_registry.py.
    """
    cost = compute_cost(layer.playbooks["availability_recovery"], basis_2451, layer)
    assert cost.lakh(layer) == pytest.approx(18.0, abs=LAKH)
    assert cost.rate_inr == 53000
    assert cost.quantity == 34
    assert cost.basis == "treated_store_count"


def test_cost_needs_both_halves(layer):
    """A playbook rate with no measured quantity is refused, not zeroed.

    A lever quoted at nothing wins every ranking, which is exactly why
    this has to raise.
    """
    with pytest.raises(CostError, match="measured as zero"):
        compute_cost(layer.playbooks["availability_recovery"], CostBasis(), layer)


def test_annual_subscription_does_not_scale_with_the_case(layer):
    small = compute_cost(
        layer.playbooks["competitor_intelligence"], CostBasis(affected_store_count=3), layer
    )
    large = compute_cost(
        layer.playbooks["competitor_intelligence"],
        CostBasis(affected_store_count=300),
        layer,
    )
    assert small.amount_inr == large.amount_inr
    assert small.lakh(layer) == pytest.approx(4.0, abs=LAKH)


def test_call_down_is_nine_managers_and_two_hours_on_2451(layer):
    """Registry: "9 managers, 2 hours" — computed, from 34 treated stores."""
    plan = plan_call_down(34, layer)
    assert plan.managers == 9
    assert plan.hours == pytest.approx(2.0, abs=1e-9)
    assert plan.render() == "9 managers, 2 hours"


def test_call_down_moves_with_the_store_count(layer):
    """It is arithmetic, not a constant: change the stores, both move."""
    assert plan_call_down(4, layer).managers == 1
    assert plan_call_down(5, layer).managers == 2
    assert plan_call_down(140, layer).managers == 35


# ===========================================================================
# Recovery — the curve, over prior comparable cases
# ===========================================================================


def test_2451_expected_recovery_is_23_to_31_crore(layer, basis_2451):
    """Accept criterion: recovery INR 2.3-3.1 Cr, computed from the curve."""
    playbook = layer.playbooks["availability_recovery"]
    cost = compute_cost(playbook, basis_2451, layer)
    recovery = compute_recovery(playbook, crore(3.24, layer), layer, cost=cost)
    assert recovery.low_crore(layer) == pytest.approx(2.3, abs=CR)
    assert recovery.high_crore(layer) == pytest.approx(3.1, abs=CR)
    assert (recovery.p25, recovery.p75) == (0.71, 0.96)


def test_recovery_never_exceeds_what_was_attributed(layer, basis_2451):
    """Resolved defect 5, enforced rather than remembered."""
    playbook = layer.playbooks["availability_recovery"]
    attributable = crore(3.24, layer)
    recovery = compute_recovery(playbook, attributable, layer)
    assert recovery.high_inr <= attributable
    assert all(curve.p75 <= 1.0 for curve in layer.recovery_curves.curves.values())


def test_recovery_confidence_comes_from_the_sample_size(layer):
    """n=3 -> Medium. Not from how tight the band looks."""
    assert confidence_label(3, layer) == "Medium"
    assert confidence_label(9, layer) == "High"
    assert confidence_label(0, layer) == "Low"
    assert layer.recovery_curves.curves["availability_restock"].sample_size == 3


def test_every_curve_label_agrees_with_its_sample_size(layer):
    """The loader cross-checks this; asserted here so the reason is visible."""
    for name, curve in layer.recovery_curves.curves.items():
        assert curve.confidence_label == confidence_label(curve.sample_size, layer), name


def test_2451_roi_is_13_to_17x(result_2451, layer):
    recovery = result_2451.recovery
    assert recovery.roi_low == pytest.approx(13, abs=0.5)
    assert recovery.roi_high == pytest.approx(17, abs=0.5)


def test_an_information_lever_is_credited_with_no_recovery(layer):
    """Buying a feed recovers nothing, and no ROI is quoted for it."""
    playbook = layer.playbooks["competitor_intelligence"]
    cost = compute_cost(playbook, CostBasis(), layer)
    recovery = compute_recovery(playbook, crore(1.88, layer), layer, cost=cost)
    assert recovery.low_inr == 0.0 and recovery.high_inr == 0.0
    assert recovery.roi_low is None and recovery.roi_high is None
    assert not recovery.recovers


# ===========================================================================
# #2451 end to end
# ===========================================================================


def test_2451_produces_the_availability_playbook(result_2451):
    assert result_2451.playbook.playbook == "availability_recovery"
    assert result_2451.owner.title == "Regional Supply Chain Lead"
    assert result_2451.call_down.render() == "9 managers, 2 hours"


def test_2451_action_text_is_the_playbooks_own_words(result_2451, layer):
    """Rule 1: the model may re-phrase this. It may not invent it."""
    assert result_2451.action == layer.playbooks["availability_recovery"].action.strip()


def test_2451_escalates_to_the_playbooks_role(result_2451):
    """Escalation IS the playbook's: whoever authorised the freight premium."""
    assert result_2451.escalation.role == "retail_operations_lead"


def test_2451_linked_case_is_created(result_2451, registry):
    """Accept criterion: the DC-allocation linked case is created."""
    linked = result_2451.linked_case
    assert linked is not None
    assert linked.driver == "transport_disruption"
    assert linked.kpi == "on_shelf_availability"
    assert linked.scope == "West DC"
    assert linked.parent_case_id == "2451"
    assert linked.rationale == "the cause of the cause"

    row = registry.execute(
        "SELECT kpi, scope, status, linked_from_case_id FROM case_registry "
        "WHERE case_id = ?",
        [linked.case_id],
    ).fetchone()
    assert row == ("on_shelf_availability", "West DC", "open", "2451")


def test_2451_linked_case_id_is_allocated_not_invented(result_2451):
    assert result_2451.linked_case.case_id == "2452"


def test_2451_writes_no_data_gap(result_2451, registry):
    """T3 did not fire on #2451. Nothing goes in the register."""
    assert result_2451.gaps == ()
    assert registry.execute("SELECT COUNT(*) FROM data_gap_register").fetchone()[0] == 0


def test_2451_contract_carries_every_number_as_evidence(result_2451, layer):
    contract = to_contract(result_2451, layer)
    assert isinstance(contract, Recommendation)
    assert contract.playbook_ref == "availability_recovery"
    assert contract.cost.value == pytest.approx(18.0, abs=LAKH)
    assert contract.expected_recovery_low.value == pytest.approx(2.3, abs=CR)
    assert contract.expected_recovery_high.value == pytest.approx(3.1, abs=CR)
    assert contract.recovery_confidence == "Medium"
    assert contract.sample_size == 3
    assert contract.effort == "9 managers, 2 hours"
    assert contract.linked_case_id == "2452"


def test_every_emitted_number_is_produced_by_code(result_2451):
    """Rule 1. Nothing on this panel came from a model."""
    assert result_2451.evidence
    assert all(item.produced_by == "code" for item in result_2451.evidence)
    assert all(item.lineage for item in result_2451.evidence)


def test_a_decided_case_offers_no_investigations(result_2451):
    """The two shapes are not the same object relabelled."""
    assert result_2451.investigations == ()
    assert result_2451.not_recommended == ()


# ===========================================================================
# #2467 — an abstention outputs investigations, not an action
# ===========================================================================


def test_2467_offers_three_ranked_investigations(result_2467):
    """Accept criterion: three ranked investigations."""
    ranked = result_2467.investigations
    assert len(ranked) == 3
    assert {item.playbook for item in ranked} == {
        "manager_call_down",
        "footfall_instrumentation",
        "competitor_intelligence",
    }


def test_2467_investigations_descend_by_value_per_rupee(result_2467):
    ranked = result_2467.investigations
    ratios = [item.value_per_rupee for item in ranked]
    assert ratios == sorted(ratios, reverse=True)
    assert ranked[0].playbook == "manager_call_down"


def test_2467_value_uses_the_causal_graphs_prior(result_2467, layer):
    """Both halves computed: residual at stake x the graph's prior."""
    feed = next(i for i in result_2467.investigations if i.playbook == "competitor_intelligence")
    assert feed.prior == layer.causal_graph.hypotheses["competitor_action"].prior
    assert feed.value_inr == pytest.approx(
        abs(crore(1.88, layer)) * feed.prior, rel=1e-9
    )


def test_2467_offers_no_action(result_2467):
    assert result_2467.playbook is None
    assert result_2467.recovery is None


def test_a_recovering_lever_is_never_listed_as_an_investigation(result_2467):
    """A price cut is not a cheap way to look into a price hypothesis."""
    assert "price_correction" not in {i.playbook for i in result_2467.investigations}


def test_2467_refuses_the_price_response_at_12_crore(result_2467, layer):
    """Accept criterion: the INR 1.2 Cr refusal, with cost and error rate."""
    refused = result_2467.not_recommended
    assert len(refused) == 1
    entry = refused[0]
    assert entry.playbook == "price_correction"
    assert entry.cost.crore(layer) == pytest.approx(1.2, abs=CR)
    assert entry.error_rate == pytest.approx(0.50, abs=1e-9)
    rendered = entry.render(layer)
    assert "1.20 Cr" in rendered and "50%" in rendered


def test_the_refusal_is_priced_at_margin_because_no_price_event_was_observed(
    result_2467, layer
):
    """There is no rise to give back, so the lever is quoted at what it can lose."""
    entry = result_2467.not_recommended[0]
    assert entry.cost.depth_source == "unobserved_depth"
    assert entry.cost.depth == layer.recommend.economics.gross_margin_rate


def test_the_error_rate_is_the_worse_of_two_computed_sources(layer):
    """T4's coin flip (0.50) beats calibration's 0.42, and the worse wins."""
    rate, source, _detail = error_rate(
        layer, indistinguishable_count=2, case_type_accuracy=0.58
    )
    assert (rate, source) == (0.5, "indistinguishable_tie")

    rate, source, _detail = error_rate(layer, case_type_accuracy=0.58)
    assert rate == pytest.approx(0.42, abs=1e-9)
    assert source == "case_type_calibration"


def test_a_refusal_without_an_error_rate_is_refused(layer):
    with pytest.raises(ValueError, match="must carry an error rate"):
        error_rate(layer)


def test_an_abstention_that_refuses_nothing_is_an_error(
    registry, analyst, request_2467, layer
):
    """recommend.yaml requires a refusal; a shopping list is not a resolution."""
    register_case(
        registry,
        case_id="2467",
        kpi="conversion_rate",
        scope="East",
        grain="monthly",
        period="2025-10",
        opened_at=CLOCK,
    )
    stripped = RecommendRequest(
        **{
            **request_2467.__dict__,
            "unresolved_drivers": ("competitor_action", "footfall_decline"),
        }
    )
    with pytest.raises(RecommendError, match="refused nothing"):
        recommend(registry, analyst, stripped, layer=layer, clock=CLOCK)


def test_2467_contract_leads_with_its_top_investigation(result_2467, layer):
    contract = to_contract(result_2467, layer)
    assert contract.playbook_ref == "manager_call_down"
    assert contract.expected_recovery_low.value == 0.0
    assert contract.expected_recovery_high.value == 0.0
    assert contract.roi_low is None
    assert len(contract.not_recommended) == 1
    assert "1.20 Cr" in contract.not_recommended[0]


# ===========================================================================
# The data gap register — written whenever T3 fires
# ===========================================================================


def test_t3_writes_both_missing_sources_to_the_register(result_2467, registry):
    assert result_2467.gaps_written == 2
    rows = registry.execute(
        "SELECT gap_code, severity, subject, scope, measure, value_numeric, "
        "resolution_hint FROM data_gap_register ORDER BY subject"
    ).fetchall()
    assert [row[2] for row in rows] == ["competitor_footfall", "competitor_pricing"]
    for row in rows:
        assert row[0] == "SOURCE_NOT_HELD"
        assert row[1] == "ERROR"
        assert row[3] == "East"
        assert row[5] == 0.0


def test_the_gap_hint_is_the_playbook_that_acquires_the_source(result_2467, layer):
    """Not the cheapest lever — the one that can actually close it."""
    hints = {gap.source: gap.acquisition_playbook for gap in result_2467.gaps}
    assert set(hints.values()) == {"competitor_intelligence"}
    assert acquisition_playbook(
        "competitor_pricing", "competitor_action", layer
    ).playbook == "competitor_intelligence"
    assert "4.00 L" in result_2467.gaps[0].resolution_hint


def test_a_call_down_does_not_close_a_source_gap(layer):
    """It is the cheapest investigation and it acquires nothing."""
    assert layer.playbooks["manager_call_down"].lever != (
        layer.recommend.data_gap.acquisition_lever
    )


def test_the_register_is_keyed_by_source_and_scope_not_by_case(
    registry, analyst, request_2467, layer
):
    """A second case in the same region updates the entry, never duplicates it."""
    register_case(
        registry,
        case_id="2467",
        kpi="conversion_rate",
        scope="East",
        grain="monthly",
        period="2025-10",
        opened_at=CLOCK,
    )
    recommend(registry, analyst, request_2467, layer=layer, clock=CLOCK)
    recommend(registry, analyst, request_2467, layer=layer, clock=CLOCK)
    assert registry.execute("SELECT COUNT(*) FROM data_gap_register").fetchone()[0] == 2


def test_no_t3_means_no_register_write(registry, analyst, request_2467, layer):
    register_case(
        registry,
        case_id="2467",
        kpi="conversion_rate",
        scope="East",
        grain="monthly",
        period="2025-10",
        opened_at=CLOCK,
    )
    without_t3 = RecommendRequest(
        **{**request_2467.__dict__, "triggers_fired": ("T4", "T7")}
    )
    result = recommend(registry, analyst, without_t3, layer=layer, clock=CLOCK)
    assert result.gaps == ()
    assert registry.execute("SELECT COUNT(*) FROM data_gap_register").fetchone()[0] == 0


# ===========================================================================
# Validation — a playbook missing a mandatory field does not load
# ===========================================================================


@pytest.fixture
def sandbox(tmp_path):
    import shutil
    from pathlib import Path

    destination = tmp_path / "semantic_layer"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "semantic_layer",
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.py"),
    )
    return destination


def _rewrite(path, mutate):
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize(
    "field",
    ["driver", "lever", "action", "cost_model", "lead_time", "owner_role",
     "recovery_curve_ref", "monitoring_plan"],
)
def test_a_playbook_missing_a_mandatory_field_fails_validation(sandbox, field):
    """Accept criterion, asserted on the playbook P11 added."""
    path = sandbox / "playbooks" / "manager_call_down.yaml"
    _rewrite(path, lambda payload: payload.pop(field))
    with pytest.raises(SemanticLayerError) as excinfo:
        load_semantic_layer(sandbox)
    assert field in str(excinfo.value)


def test_a_playbook_scaling_with_an_unknown_basis_fails_validation(sandbox):
    path = sandbox / "playbooks" / "availability_recovery.yaml"
    _rewrite(
        path, lambda p: p["cost_model"].__setitem__("scales_with", "vibes")
    )
    with pytest.raises(SemanticLayerError, match="not a cost basis"):
        load_semantic_layer(sandbox)


def test_a_recovery_curve_whose_label_contradicts_its_sample_size_fails(sandbox):
    path = sandbox / "recovery_curves.yaml"
    _rewrite(
        path,
        lambda p: p["curves"]["availability_restock"].__setitem__(
            "confidence_label", "High"
        ),
    )
    with pytest.raises(SemanticLayerError, match="fitted on 3"):
        load_semantic_layer(sandbox)


def test_a_kpi_owning_a_driver_that_is_not_in_the_graph_fails(sandbox):
    path = sandbox / "kpis" / "net_revenue.yaml"
    _rewrite(
        path,
        lambda p: p["driver_ownership"]["by_driver"].__setitem__(
            "gremlins", {"role": "cxo", "title": "CXO"}
        ),
    )
    with pytest.raises(SemanticLayerError, match="not a\\s+hypothesis"):
        load_semantic_layer(sandbox)


def test_a_kpi_may_not_own_a_driver_that_cannot_move_it(sandbox):
    """Dead config reads as a decision somebody made.

    Weather cannot move on-shelf availability, so no case on that KPI
    could ever reach a `weather` ownership row. Naming an owner there is
    not harmless: it looks like the business decided who answers for it.
    """
    path = sandbox / "kpis" / "on_shelf_availability.yaml"
    _rewrite(
        path,
        lambda p: p["driver_ownership"]["by_driver"].__setitem__(
            "weather", {"role": "retail_operations_lead", "title": "Retail Operations Lead"}
        ),
    )
    with pytest.raises(SemanticLayerError, match="does not affect it"):
        load_semantic_layer(sandbox)


def test_every_owned_driver_actually_affects_its_kpi(layer):
    for name, contract in layer.kpis.items():
        if contract.driver_ownership is None:
            continue
        for driver in contract.driver_ownership.by_driver:
            assert name in layer.causal_graph.hypotheses[driver].affects, (
                f"{name} owns {driver}, which cannot move it"
            )


def test_one_role_may_not_carry_two_titles(sandbox):
    path = sandbox / "kpis" / "conversion_rate.yaml"
    _rewrite(
        path,
        lambda p: p["driver_ownership"]["by_driver"]["stock_out"].__setitem__(
            "title", "Supply Chain Person"
        ),
    )
    with pytest.raises(SemanticLayerError, match="is titled"):
        load_semantic_layer(sandbox)


# ===========================================================================
# REGISTRY GAPS — measured, pinned, and NOT adjusted (CLAUDE.md rule 10)
# ===========================================================================


def test_2467_call_down_does_not_reproduce_nine_managers(result_2467):
    """GAP — CLAUDE.md quotes "call-down 9 managers ~2 hours" on #2467.

    Nine is #2451's figure: 34 treated West stores at four to a cluster
    manager. #2467 is an EAST case with no treated group — it abstained,
    so no mechanism was localised — and its call-down therefore covers all
    58 East stores, which is fifteen managers.

    The two cases were given the same effort line in the Round 1 material.
    Only one of them can be right, and the arithmetic that produces #2451's
    nine produces fifteen here. Reported, not reconciled: closing this
    means deciding whether an abstained case rings the whole region or a
    sample of it, which is a business decision, not a coding one.
    """
    assert result_2467.call_down.stores == 58
    assert result_2467.call_down.managers == 15
    assert result_2467.call_down.managers != 9
    assert result_2467.call_down.hours == pytest.approx(2.0, abs=1e-9)


def test_2467_footfall_instrumentation_does_not_reproduce_12_lakh(
    result_2467, east_uninstrumented, layer
):
    """GAP — CLAUDE.md quotes "footfall counters INR 12 L, 6 wks" on #2467.

    INR 12 L is 109 stores at INR 11,000, and 109 is WEST's uninstrumented
    count (140 less 31). It inherits the denominator defect already pinned
    in tests/test_number_registry.py as GAP 5, where the same "31 of 140"
    is quoted on an East case.

    East holds 58 stores, 26 of them uninstrumented, so instrumenting
    East's own estate costs INR 2.86 L. The consequence is a REORDERING:
    at INR 12 L the counters rank below the INR 4 L competitor feed, which
    is the order CLAUDE.md lists. At East's real price they rank above it.
    The engine publishes the computed order.
    """
    footfall = next(
        item
        for item in result_2467.investigations
        if item.playbook == "footfall_instrumentation"
    )
    assert east_uninstrumented == 26
    assert footfall.cost.lakh(layer) == pytest.approx(2.86, abs=LAKH)
    assert footfall.cost.lakh(layer) != pytest.approx(12.0, abs=LAKH)

    order = [item.playbook for item in result_2467.investigations]
    assert order == [
        "manager_call_down",
        "footfall_instrumentation",
        "competitor_intelligence",
    ]
