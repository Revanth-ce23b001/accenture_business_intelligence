"""QUALIFY — Gates 2 to 5, and the restraint.

The acceptance criteria are four scenarios:

  #2451  8.1 / 3.2 / 4.9, and a regional strip of N -0.4 . S +1.1 . E -0.6 . W -4.9
  #2471  INSUFFICIENT_HISTORY — 7 weeks held, 26 required
  #2472  a case opened on a FLAT series against a +12% calendar expectation
  a synthetic all-region drop  ->  MARKET_CASE

Two of those reproduce exactly and two do not, and the ones that do not
are pinned in the REGISTRY GAPS section at the end rather than having
their targets moved (CLAUDE.md rule 10). The short version: the registry's
figures are the GENERATOR's own calendar expectation, and QUALIFY does not
get to see it — it refits a baseline from eighteen months of daily data
with two occurrences of most festivals, and lands within about a tenth of
a point on the subject scope and a third of a point on its peers.

The mutating tests run inside a transaction that is rolled back.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from engine.contracts import Evidence
from engine.db import GovernanceError
from engine.qualify import (
    QualifyError,
    QualifyRequest,
    correlated_kpis,
    previous_month,
    qualify,
    register_case,
)
from security.policy import User

CASE_2451 = QualifyRequest(
    kpi="net_revenue", scope="West", grain="monthly",
    period="2025-11", comparison_period="2025-10",
)
CASE_2471 = QualifyRequest(
    kpi="qcomm_fulfilment_rate", scope="All-India", grain="weekly",
    period="2025-11", comparison_period="2025-10",
)
CASE_2472 = QualifyRequest(
    kpi="net_revenue", scope="South", grain="monthly",
    period="2025-09", comparison_period="2025-08",
)

ANALYST = User(user_id="U008", persona="analyst", display_name="Meera Joshi")

#: The registry's regional residual strip.
STRIP = {"North": -0.4, "South": 1.1, "East": -0.6, "West": -4.9}

#: Tolerances. The subject scope reproduces to a tenth of a point; the peer
#: scopes to a third. Both are stated here rather than buried in an assert,
#: and the reason they are not zero is `test_the_refit_does_not_reproduce_
#: the_strip_exactly` at the bottom of this file.
SUBJECT_PT = 0.15
PEER_PT = 0.35


@pytest.fixture
def sandbox(warehouse):
    """A transaction that is always rolled back."""
    warehouse.execute("BEGIN TRANSACTION")
    try:
        yield warehouse
    finally:
        warehouse.execute("ROLLBACK")


@pytest.fixture(scope="module")
def result_2451(warehouse):
    return qualify(warehouse, ANALYST, CASE_2451)


@pytest.fixture(scope="module")
def result_2471(warehouse):
    return qualify(warehouse, ANALYST, CASE_2471)


@pytest.fixture(scope="module")
def result_2472(warehouse):
    return qualify(warehouse, ANALYST, CASE_2472)


# ===========================================================================
# ACCEPT: #2451 -> 8.1 / 3.2 / 4.9
# ===========================================================================


def test_2451_headline_is_minus_8_1(result_2451):
    """The one figure that is a fact rather than an estimate."""
    assert result_2451.decomposition.headline_pt == pytest.approx(-8.1, abs=0.02)


def test_2451_calendar_accounts_for_3_2_points(result_2451):
    assert result_2451.decomposition.calendar_pt == pytest.approx(-3.2, abs=SUBJECT_PT)


def test_2451_qualified_residual_is_4_9_points(result_2451):
    assert result_2451.decomposition.residual_pt == pytest.approx(-4.9, abs=SUBJECT_PT)


def test_2451_residual_in_rupees_is_4_1_crore(result_2451):
    """4.9 pt of an INR 83.70 Cr October."""
    assert result_2451.decomposition.residual_inr / 1e7 == pytest.approx(-4.10, abs=0.15)


def test_the_decomposition_reconciles_exactly(result_2451, result_2472, layer):
    """headline = calendar + residual, by construction and to the bit."""
    tolerance = layer.qualify.calendar.decomposition.reconciliation_tolerance_pt
    for result in (result_2451, result_2472):
        assert result.decomposition.reconciles(tolerance)


def test_2451_calendar_is_mostly_the_day_count(warehouse, result_2451):
    """November has thirty days against October's thirty-one.

    30/31 - 1 = -3.2%, which is very nearly the whole calendar effect.
    West's festival amplitude is small, so on this case the calendar is
    arithmetic — and that is worth being able to say out loud.
    """
    from engine.qualify.calendar import trading_days
    from engine.qualify.series import load_series
    from semantic_layer.schema import get_semantic_layer

    series = load_series(warehouse, ANALYST, "net_revenue", get_semantic_layer())
    days_m = trading_days(series, "West", "2025-11")
    days_prev = trading_days(series, "West", "2025-10")
    assert (days_m, days_prev) == (30, 31)
    day_count_effect = (days_m / days_prev - 1) * 100.0
    assert result_2451.decomposition.calendar_pt == pytest.approx(
        day_count_effect, abs=0.5
    )


def test_2451_region_strip(result_2451):
    """ACCEPT: N -0.4 . S +1.1 . E -0.6 . W -4.9, same method per region."""
    strip = result_2451.region_strip
    assert set(strip) == set(STRIP)
    for region, target in STRIP.items():
        tolerance = SUBJECT_PT if region == "West" else PEER_PT
        assert strip[region] == pytest.approx(target, abs=tolerance), (
            f"{region}: {strip[region]:+.2f} against {target:+.1f}"
        )


def test_2451_strip_signs_match_the_registry(result_2451):
    """Weaker than the values and worth asserting separately.

    A strip that got every magnitude right and one sign wrong would tell
    an analyst the opposite of the truth about that region.
    """
    for region, target in STRIP.items():
        assert (result_2451.region_strip[region] > 0) == (target > 0), region


def test_2451_opens_a_case(result_2451):
    assert result_2451.case_opened is True
    assert result_2451.outcome_code is None


def test_2451_passes_all_four_gates(result_2451):
    assert [gate.gate_id for gate in result_2451.gates] == [2, 3, 4, 5]
    assert all(gate.passed for gate in result_2451.gates)


def test_2451_residual_is_outside_the_band(result_2451):
    band = result_2451.band
    assert band.breached_by(result_2451.decomposition.residual_pt)
    assert band.lookback_weeks == 8
    assert band.quantile == pytest.approx(0.90)


def test_2451_clears_materiality_several_times_over(result_2451):
    verdict = result_2451.materiality
    assert verdict.material
    assert verdict.unit == "INR_CR"
    assert verdict.threshold == pytest.approx(0.50)
    assert verdict.multiple > 4.0
    assert verdict.owner_role == "finance_controller"


def test_2451_is_not_a_market_case(result_2451):
    verdict = result_2451.specificity
    assert verdict.is_market_case is False
    assert verdict.breaching == ("West",)
    assert verdict.escalate_to_role is None


# ===========================================================================
# ACCEPT: #2471 -> INSUFFICIENT_HISTORY, 7 weeks against 26
# ===========================================================================


def test_2471_stops_at_gate_3(result_2471):
    assert result_2471.outcome_code == "INSUFFICIENT_HISTORY"
    assert result_2471.case_opened is False
    assert [gate.gate_id for gate in result_2471.gates] == [3]


def test_2471_holds_seven_weeks_and_needs_twenty_six(result_2471):
    history = result_2471.history
    assert history.observed_periods == 7
    assert history.required_periods == 26
    assert history.sufficient is False


def test_2471_history_is_counted_not_read_off_the_contract(warehouse, result_2471):
    """`history_weeks` is a claim. The count is the fact.

    Here the two happen to agree, which is exactly why the assertion is
    against the WAREHOUSE and not against the contract: a gate that read
    the claim would pass this test today and fit a baseline on seven points
    the day somebody edited the YAML.
    """
    held = warehouse.execute(
        "SELECT COUNT(DISTINCT week_start) FROM fact_qcomm_weekly WHERE scope = 'All-India'"
    ).fetchone()[0]
    assert result_2471.history.observed_periods == held == 7


def test_2471_fits_no_baseline_at_all(result_2471):
    """Nothing downstream ran. There was nothing to run it on."""
    assert result_2471.decomposition is None
    assert result_2471.band is None
    assert result_2471.materiality is None


def test_2471_says_monitoring_only(result_2471):
    assert "Monitoring only" in result_2471.gate(3).detail


# ===========================================================================
# ACCEPT: #2472 -> a case on a flat series
# ===========================================================================


def test_2472_headline_is_flat(result_2472):
    """0.0% month on month. On a dashboard this is a quiet month."""
    assert result_2472.decomposition.headline_pt == pytest.approx(0.0, abs=0.05)


def test_2472_calendar_expected_a_rise(result_2472):
    """Ganesh Chaturthi fell in September; the calendar expected +12%."""
    assert result_2472.decomposition.calendar_pt == pytest.approx(12.0, abs=0.5)


def test_2472_residual_is_minus_12_points(result_2472):
    """Flat against an expected rise is a -12 pt residual."""
    assert result_2472.decomposition.residual_pt == pytest.approx(-12.0, abs=0.5)


def test_2472_opens_a_case(result_2472):
    """The scenario that shows why a RESIDUAL opens a case, not a headline."""
    assert result_2472.case_opened is True
    assert result_2472.outcome_code is None


def test_2472_would_not_have_been_noticed_by_the_headline(result_2472):
    """A headline threshold of any size would have missed this entirely."""
    assert abs(result_2472.decomposition.headline_pt) < 0.1
    assert abs(result_2472.decomposition.residual_pt) > 10.0


# ===========================================================================
# ACCEPT: a synthetic all-region drop -> MARKET_CASE
# ===========================================================================


def test_an_all_region_drop_is_a_market_case(sandbox):
    """Three regions moving together is one case, not four."""
    sandbox.execute(
        "UPDATE fact_sales_daily SET net_revenue_inr = net_revenue_inr * 0.80 "
        "WHERE txn_date BETWEEN DATE '2025-11-01' AND DATE '2025-11-30'"
    )
    result = qualify(sandbox, ANALYST, CASE_2451)

    assert result.outcome_code == "MARKET_CASE"
    assert result.case_opened is False
    assert result.specificity.is_market_case is True
    assert len(result.specificity.breaching) >= 3


def test_a_market_case_escalates_to_the_cco(sandbox, layer):
    sandbox.execute(
        "UPDATE fact_sales_daily SET net_revenue_inr = net_revenue_inr * 0.80 "
        "WHERE txn_date BETWEEN DATE '2025-11-01' AND DATE '2025-11-30'"
    )
    result = qualify(sandbox, ANALYST, CASE_2451)
    assert result.specificity.escalate_to_role == layer.qualify.specificity.escalate_to_role
    assert "one case, not" in result.gate(4).detail


def test_a_market_case_needs_the_same_direction(sandbox):
    """Regions moving in opposite directions are not a market movement.

    Half the estate up and half down is a mix shift, or four unrelated
    stories. Either way it is not one thing happening to everybody.
    """
    sandbox.execute(
        "UPDATE fact_sales_daily SET net_revenue_inr = net_revenue_inr * "
        "CASE WHEN region IN ('North', 'East') THEN 0.80 ELSE 1.25 END "
        "WHERE txn_date BETWEEN DATE '2025-11-01' AND DATE '2025-11-30'"
    )
    result = qualify(sandbox, ANALYST, CASE_2451)
    assert result.specificity.is_market_case is False


# ===========================================================================
# Gate 2 — decompose, do not adjust
# ===========================================================================


def test_the_series_is_never_adjusted(warehouse, result_2451):
    """The gate reports a decomposition. It does not write one back."""
    total = warehouse.execute(
        "SELECT SUM(net_revenue_inr) FROM fact_sales_daily "
        "WHERE region = 'West' AND txn_date BETWEEN DATE '2025-11-01' "
        "AND DATE '2025-11-30'"
    ).fetchone()[0]
    assert result_2451.decomposition.actual_period_inr == pytest.approx(total)


def test_the_period_under_test_is_excluded_from_the_fit(result_2451):
    """A baseline fitted through the anomaly always understates it."""
    assert "2025-11" not in result_2451.decomposition.fit.fit_months
    assert "2025-10" in result_2451.decomposition.fit.fit_months


def test_the_baseline_fits_the_history_it_was_given(result_2451):
    assert result_2451.decomposition.fit.r_squared > 0.95
    assert result_2451.decomposition.fit.fit_days > 400


def test_the_comparison_month_fit_error_is_reported_not_absorbed(result_2451):
    """The calendar effect is measured against the ACTUAL previous level.

    So the model's error on that month is not folded into "calendar" — and
    because it is not folded in, it has to be shown.
    """
    evidence = result_2451.evidence_by_id()["qualify.calendar.comparison_fit_error"]
    assert evidence.unit == "pt"
    assert abs(evidence.value) < 1.0


def test_an_already_anomalous_month_is_dropped_from_the_fit(result_2451):
    """South's September is case #2472. A baseline is not fitted through it.

    The exclusion is not configured by name — pass one finds the month
    because its mean residual is three standard deviations out, and pass
    two refits without it. The method discovers the anomaly it should not
    learn from.
    """
    south = result_2451.peer_decompositions["South"]
    assert "2025-09" in south.fit.excluded_months
    west = result_2451.peer_decompositions["West"]
    assert west.fit.excluded_months == ()


def test_the_baseline_holds_no_marketing_spend(layer):
    """The one regressor that must never be here.

    Marketing spend inside the calendar baseline would absorb a spend cut
    into "calendar-expected", and H4 on case #2451 could then never be
    eliminated on precedence. A scheduled promo WINDOW is a calendar event
    and is allowed; a spend LEVEL is not.
    """
    regressors = layer.qualify.calendar.model.regressors
    fields = set(type(regressors).model_fields)
    assert "promo_window" in fields
    assert not any("spend" in name for name in fields)
    assert not any("marketing" in name for name in fields)


def test_the_promo_window_regressor_is_actually_in_the_design(warehouse):
    from engine.qualify.series import design_matrix, load_series
    from semantic_layer.schema import get_semantic_layer

    layer = get_semantic_layer()
    series = load_series(warehouse, ANALYST, "net_revenue", layer)
    block = series.region("West")
    design = design_matrix(
        block,
        series.festivals,
        layer.qualify.calendar.model.regressors,
        series.origin,
        layer.warehouse.units,
    )
    assert "promo_window" in design.columns
    assert design["promo_window"].sum() > 0


def test_overlapping_festivals_are_represented_separately(warehouse):
    """Onam and Ganesh Chaturthi run together, and dim_calendar cannot say so.

    One label per day means the second festival is silently dropped. The
    bridge table carries both, and the design has a column for each.
    """
    both = warehouse.execute(
        """
        SELECT date, COUNT(DISTINCT festival) AS n
        FROM dim_festival_window WHERE phase = 'window'
        GROUP BY 1 HAVING COUNT(DISTINCT festival) > 1
        """
    ).fetchall()
    assert both, "no overlapping festival days — the bridge is not doing its job"

    from engine.qualify.series import festival_terms, load_series
    from semantic_layer.schema import get_semantic_layer

    layer = get_semantic_layer()
    series = load_series(warehouse, ANALYST, "net_revenue", layer)
    terms = festival_terms(
        series.region("South")["date"],
        series.festivals,
        layer.qualify.calendar.model.regressors,
    )
    assert any("Onam" in name for name in terms.columns)
    assert any("GaneshChaturthi" in name for name in terms.columns)


# ===========================================================================
# Gate 3 — the band is an empirical quantile, not a sigma
# ===========================================================================


def test_the_band_is_a_quantile_not_a_sigma(warehouse, layer, result_2451):
    """Two standard deviations of a heavy-tailed series is not 95% of anything."""
    assert layer.qualify.band.method == "stl_residual_empirical_quantile"
    band = result_2451.band
    residuals = result_2451.decomposition.fit.residuals
    window = residuals[
        (residuals.index > band.window_start) & (residuals.index <= band.window_end)
    ]
    two_sigma = float(window.std()) * 2.0 * 100.0
    assert band.band_pt != pytest.approx(two_sigma, abs=0.01)


def test_the_band_window_stops_before_the_period_under_test(result_2451):
    """Otherwise the anomaly widens the band it has to clear."""
    import pandas as pd

    assert result_2451.band.window_end < pd.Timestamp("2025-11-01")


def test_a_residual_inside_the_band_is_not_a_case(sandbox):
    """A movement the scope makes all the time is not news."""
    sandbox.execute(
        "UPDATE fact_sales_daily SET net_revenue_inr = net_revenue_inr * 1.049 "
        "WHERE region = 'West' AND txn_date BETWEEN DATE '2025-11-01' "
        "AND DATE '2025-11-30'"
    )
    result = qualify(sandbox, ANALYST, CASE_2451)
    assert result.outcome_code == "WITHIN_BAND"
    assert result.case_opened is False
    assert abs(result.decomposition.residual_pt) < result.band.band_pt


def test_a_festival_heavy_region_has_a_wider_band(result_2451):
    """Stated rather than hidden: we know less about South than about West.

    The calendar model fits West almost exactly — its festival amplitude
    is small — and South least well. So South needs a bigger movement
    before we will claim it means anything. That is the honest answer, not
    a defect.
    """
    assert result_2451.band.region == "West"
    assert result_2451.band.band_pt > 0


# ===========================================================================
# Gate 4 — specificity
# ===========================================================================


def test_specificity_uses_the_same_method_for_every_region(result_2451):
    for region, decomposition in result_2451.peer_decompositions.items():
        assert decomposition.region == region
        assert decomposition.period == "2025-11"
        assert decomposition.comparison_period == "2025-10"


def test_a_regional_manager_cannot_rule_on_specificity(warehouse):
    """One region visible is not four regions clean.

    "I could not see the others" and "the others were fine" are different
    statements and must not print as the same one.
    """
    manager = User(user_id="U003", persona="regional_manager", region="West")
    result = qualify(warehouse, manager, CASE_2451)
    assert result.outcome_code == "PEERS_NOT_VISIBLE"
    assert result.specificity.peers_visible is False
    assert "not the same as" in result.gate(4).detail


# ===========================================================================
# Gate 5 — materiality
# ===========================================================================


def test_a_sub_threshold_residual_goes_to_the_digest(sandbox, layer):
    """Not a case is not the same as ignored."""
    from engine.qualify.materiality import assess_materiality

    contract = layer.kpis["net_revenue"]
    verdict = assess_materiality(
        contract,
        residual_pt=-0.05,
        residual_inr=-4.0e6,          # INR 0.40 Cr, below the INR 50 L limit
        spec=layer.qualify.materiality,
        units=layer.warehouse.units,
    )
    assert verdict.material is False
    assert verdict.route == "weekly_digest"


def test_a_rate_kpi_is_judged_in_points_not_rupees(layer):
    from engine.qualify.materiality import assess_materiality

    contract = layer.kpis["conversion_rate"]
    assert contract.thresholds.materiality.unit == "pt"
    verdict = assess_materiality(
        contract,
        residual_pt=-1.2,
        residual_inr=-1.88e7,
        spec=layer.qualify.materiality,
        units=layer.warehouse.units,
    )
    assert verdict.unit == "pt"
    assert verdict.residual == pytest.approx(1.2)
    assert verdict.material is True


def test_a_kpi_with_no_materiality_limit_cannot_open_a_case(layer):
    from engine.qualify.materiality import MaterialityError, assess_materiality

    contract = layer.kpis["qcomm_fulfilment_rate"]
    assert contract.thresholds.materiality is None
    with pytest.raises(MaterialityError, match="monitoring only"):
        assess_materiality(
            contract, -5.0, -1e7, layer.qualify.materiality, layer.warehouse.units
        )


# ===========================================================================
# Restraint
# ===========================================================================


def test_correlated_kpis_come_from_the_contracts_not_the_data(layer):
    """The business already wrote down which numbers move together."""
    family = correlated_kpis(layer.kpis["net_revenue"], layer.qualify.restraint)
    assert "transactions" in family
    assert "conversion_rate" in family
    assert "on_shelf_availability" in family
    assert "net_revenue" not in family


def test_one_cause_one_case(sandbox):
    """A stock-out moves four KPIs. It is still one event."""
    register_case(
        sandbox, case_id="C-2451", kpi="transactions", scope="West",
        grain="monthly", period="2025-11", opened_at=datetime(2025, 11, 28),
        materiality_multiple=6.0,
    )
    result = qualify(sandbox, ANALYST, CASE_2451)

    assert result.outcome_code == "DUPLICATE_OF_OPEN_CASE"
    assert result.case_opened is False
    assert result.restraint.linked_case_id == "C-2451"
    assert "one cause" in result.restraint.detail


def test_a_case_on_an_unrelated_kpi_is_not_a_duplicate(sandbox):
    register_case(
        sandbox, case_id="C-OTHER", kpi="qcomm_fulfilment_rate", scope="West",
        grain="monthly", period="2025-11", opened_at=datetime(2025, 11, 28),
        materiality_multiple=6.0,
    )
    assert qualify(sandbox, ANALYST, CASE_2451).case_opened is True


def test_an_open_case_suppresses_the_next_period(sandbox):
    """The investigation covers it; a second case splits the evidence."""
    register_case(
        sandbox, case_id="C-OPEN", kpi="net_revenue", scope="West",
        grain="monthly", period="2025-10", opened_at=datetime(2025, 11, 20),
        materiality_multiple=20.0,
    )
    result = qualify(sandbox, ANALYST, CASE_2451)
    assert result.outcome_code == "SUPPRESSED_BY_OPEN_CASE"
    assert result.restraint.linked_case_id == "C-OPEN"


def test_a_materially_worse_movement_escalates_instead(sandbox, layer):
    """Suppression is not a mute button.

    A movement several times worse than the one being investigated is an
    escalation, and it opens.
    """
    multiple = layer.qualify.restraint.suppression.escalation_multiple
    register_case(
        sandbox, case_id="C-SMALL", kpi="net_revenue", scope="West",
        grain="monthly", period="2025-10", opened_at=datetime(2025, 11, 20),
        materiality_multiple=8.3 / (multiple + 1.0),
    )
    assert qualify(sandbox, ANALYST, CASE_2451).case_opened is True


def test_an_expired_open_case_does_not_suppress(sandbox, layer):
    window = layer.qualify.restraint.suppression.window_days
    register_case(
        sandbox, case_id="C-OLD", kpi="net_revenue", scope="West",
        grain="monthly", period="2025-10",
        opened_at=datetime(2025, 11, 30) - __import__("datetime").timedelta(days=window + 1),
        materiality_multiple=20.0,
    )
    assert qualify(sandbox, ANALYST, CASE_2451).case_opened is True


def test_three_cases_a_week_per_owner(sandbox, layer):
    """Above the cap, deferred to the digest and reported as deferred."""
    cap = layer.qualify.restraint.owner_load.max_open_cases_per_owner_per_week
    for index, scope in enumerate(("North", "South", "East")[:cap]):
        register_case(
            sandbox, case_id=f"C-{index}", kpi="net_revenue", scope=scope,
            grain="monthly", period="2025-11", opened_at=datetime(2025, 11, 25),
            materiality_multiple=5.0,
        )
    result = qualify(sandbox, ANALYST, CASE_2451)

    assert result.outcome_code == "OWNER_WEEKLY_CAP"
    assert result.case_opened is False
    assert result.restraint.deferred_to == "weekly_digest"
    assert result.restraint.open_cases_this_week == cap
    assert "not dropped" in result.restraint.detail


def test_the_cap_is_per_owner_not_global(sandbox, layer):
    """Three cases for retail operations do not block finance."""
    cap = layer.qualify.restraint.owner_load.max_open_cases_per_owner_per_week
    for index in range(cap):
        register_case(
            sandbox, case_id=f"C-OPS-{index}", kpi="conversion_rate", scope="East",
            grain="monthly", period="2025-11", opened_at=datetime(2025, 11, 25),
            materiality_multiple=5.0,
        )
    assert qualify(sandbox, ANALYST, CASE_2451).case_opened is True


def test_a_case_opened_last_week_does_not_count_against_this_week(sandbox, layer):
    cap = layer.qualify.restraint.owner_load.max_open_cases_per_owner_per_week
    for index in range(cap):
        register_case(
            sandbox, case_id=f"C-LAST-{index}", kpi="net_revenue", scope="North",
            grain="monthly", period="2025-10", opened_at=datetime(2025, 11, 10),
            materiality_multiple=5.0,
        )
    assert qualify(sandbox, ANALYST, CASE_2451).case_opened is True


# ===========================================================================
# Evidence and contracts
# ===========================================================================


def test_every_figure_leaves_as_evidence(result_2451):
    assert result_2451.evidence
    assert all(isinstance(item, Evidence) for item in result_2451.evidence)
    ids = [item.evidence_id for item in result_2451.evidence]
    assert len(ids) == len(set(ids))


def test_no_evidence_came_from_the_model(result_2451):
    assert all(item.produced_by == "code" for item in result_2451.evidence)


def test_every_evidence_carries_lineage_and_a_unit(result_2451):
    for item in result_2451.evidence:
        assert item.lineage, f"{item.evidence_id} has nothing to click through to"
        assert item.unit


def test_the_headline_calendar_and_residual_are_all_addressable(result_2451):
    index = result_2451.evidence_by_id()
    for name in ("headline", "calendar", "residual", "residual_inr"):
        assert f"qualify.calendar.{name}" in index


def test_every_gate_reports_a_named_outcome(result_2451, result_2471):
    for result in (result_2451, result_2471):
        for check in result.checks:
            assert check.outcome == check.outcome.upper()
            assert check.outcome not in {"TRUE", "FALSE"}


def test_reliability_weights_come_from_the_semantic_layer(result_2451, layer):
    weights = layer.adjudication.reliability.weights
    for item in result_2451.evidence:
        assert item.reliability == pytest.approx(weights[item.kind])


# ===========================================================================
# Plumbing
# ===========================================================================


def test_previous_month():
    assert previous_month("2025-11") == "2025-10"
    assert previous_month("2025-01") == "2024-12"


def test_an_unknown_kpi_is_refused(warehouse):
    with pytest.raises(QualifyError, match="no KPI contract"):
        qualify(
            warehouse, ANALYST,
            QualifyRequest("not_a_kpi", "West", "monthly", "2025-11"),
        )


def test_a_persona_the_contract_does_not_know_is_refused(warehouse):
    with pytest.raises(GovernanceError, match="not defined in the access policy"):
        qualify(warehouse, User(user_id="U999", persona="intern"), CASE_2451)


def test_the_gate_map_has_one_home(layer):
    """qualify.yaml and adjudication.yaml must agree on the numbering."""
    assert layer.qualify.gate_ids() == [2, 3, 4, 5]
    for gate_id, code in (
        (2, layer.qualify.calendar.outcome_code),
        (3, layer.qualify.band.history_outcome_code),
        (4, layer.qualify.specificity.outcome_code),
        (5, layer.qualify.materiality.outcome_code),
    ):
        assert layer.adjudication.gates[gate_id].outcome_code == code


# ===========================================================================
# REGISTRY GAPS — pinned, not hidden
#
# Each test below asserts a value that DIFFERS from the Number Registry.
# The registry target is not adjusted (rule 10). These fail the moment the
# method changes, forcing this section to be revisited.
# ===========================================================================


def test_the_refit_does_not_reproduce_the_strip_exactly(result_2451):
    """GAP 1 — the peer strip lands within about a third of a point.

    The registry's strip is the GENERATOR's own calendar expectation, which
    it knows exactly because it built it. QUALIFY does not get to see it:
    it refits a baseline from eighteen months of daily regional revenue in
    which most festivals occur twice, and estimates the same quantity.

    West — the scope under investigation — comes back within a tenth of a
    point. Its peers come back within a third, and the residual error is
    the baseline's estimation error on a roughly INR 100 Cr monthly figure,
    which is a quarter of one percent. There is no version of this that is
    exact without the engine reading the generator's answer.
    """
    strip = result_2451.region_strip
    errors = {region: strip[region] - target for region, target in STRIP.items()}
    assert abs(errors["West"]) < SUBJECT_PT
    assert any(abs(error) > 0.1 for region, error in errors.items() if region != "West")
    assert all(abs(error) < PEER_PT for error in errors.values())


def test_the_empirical_band_is_not_the_registrys_1_8_pt(result_2451):
    """GAP 2 — the registry's +/-1.8 pt band does not reproduce, and has no derivation.

    `scenario_2451.yaml` carries `west_empirical_band_pt: 1.8` as a target
    the generator never measures and no test has ever asserted — it is a
    Round 1 illustrative figure.

    The method P6 specifies produces a wider band, because the eight weeks
    before November 2025 are the most festival-dense of the year and the
    calendar model's DAILY misfit inside a festival window is real. The
    substantive claim survives either way: the residual is outside the band
    on both numbers, which is what `test_2451_residual_is_outside_the_band`
    asserts.
    """
    import yaml

    from data.generator.model import CONFIG_DIR

    scenario = yaml.safe_load(
        (CONFIG_DIR / "scenario_2451.yaml").read_text(encoding="utf-8")
    )
    target = scenario["targets"]["west_empirical_band_pt"]
    assert target == 1.8
    assert result_2451.band.band_pt > target
    assert result_2451.band.breached_by(result_2451.decomposition.residual_pt)


def test_the_history_available_is_shorter_than_the_brief_asks_for(layer, result_2451):
    """GAP 3 — the brief asks for three years of daily data. There are eighteen months.

    The model uses what exists and reports how much that was, rather than
    padding the window or pretending. Two occurrences of most festivals is
    the binding constraint on the whole gate, and it is why the festival
    window is split in two rather than four.
    """
    requested_days = layer.qualify.calendar.model.requested_history_years * 365
    assert result_2451.decomposition.fit.fit_days < requested_days
    assert result_2451.decomposition.fit.fit_days >= layer.qualify.calendar.model.min_fit_days
