"""The definition of done for the project.

Every value in CLAUDE.md §"Number Registry" is asserted here against what
the generator actually produces. Tolerances are the ones the P3 brief
specifies:

    ₹          ±0.02 Cr
    pt         ±0.05
    percentage ±0.5
    p-values   to the stated bound

RULE 10. Where a registry value cannot be reproduced, the target is NOT
adjusted. Instead the discrepancy is pinned by a test named
`test_*_does_not_reproduce` that asserts the measured value and states
why. Those tests pass while the gap exists and FAIL the moment someone
changes the generator to close it, which forces this file to be updated
rather than letting the gap drift out of sight.

See the module docstring section "REGISTRY GAPS" at the bottom for the
full list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from data.generator.build import _region_month_cr, build_world
from semantic_layer.schema import load_semantic_layer

CR = 0.02      # ₹ crore tolerance
PT = 0.05      # percentage-point tolerance
PCT = 0.5      # percentage tolerance


@pytest.fixture(scope="module")
def world():
    return build_world()


@pytest.fixture(scope="module")
def layer():
    return load_semantic_layer()


@pytest.fixture(scope="module")
def m(world):
    """Measured quantities, read back off the generated world."""
    return world.diagnostics


@pytest.fixture(scope="module")
def panel(world):
    """Store-level October and November revenue, for the DiD family."""
    period = world.calendar["period_month"].to_numpy()
    pre = world.actual[:, period == "2025-10"].sum(axis=1)
    post = world.actual[:, period == "2025-11"].sum(axis=1)
    return pre, post, (post - pre) / pre * 100.0


# ===========================================================================
# Entity
# ===========================================================================


def test_store_count(world):
    assert len(world.stores) == 412


@pytest.mark.parametrize(
    "region,count", [("North", 118), ("South", 96), ("East", 58), ("West", 140)]
)
def test_stores_per_region(world, region, count):
    assert int((world.stores["region"] == region).sum()) == count


def test_timeline_is_18_months_ending_30_nov_2025(world):
    assert world.calendar["date"].max().date().isoformat() == "2025-11-30"
    assert len(world.calendar) == 548


def test_current_period_is_november_2025(world):
    assert world.entity["timeline"]["current_period"] == "2025-11"


def test_annual_net_revenue_is_about_3400_cr(world):
    years = len(world.calendar) / 365.25
    total = world.actual.sum() / 1e7 / years
    assert total == pytest.approx(3400.0, abs=120.0)


def test_west_annual_revenue_is_about_1000_cr(world):
    years = len(world.calendar) / 365.25
    rows = world.stores["region"].to_numpy() == "West"
    assert world.actual[rows].sum() / 1e7 / years == pytest.approx(1000.0, abs=60.0)


def test_diwali_moves_eleven_days_year_on_year():
    from data.generator.model import FESTIVALS

    (start_2024, _), (start_2025, _) = FESTIVALS["Diwali"]
    # Compare on a common year so the 2024 leap day does not add a day.
    assert (start_2024.replace(year=start_2025.year) - start_2025).days == 11


# ===========================================================================
# Case #2451 — West, net revenue, Nov 2025
# ===========================================================================


def test_west_net_revenue_october(m):
    assert m["west_oct_cr"] == pytest.approx(83.70, abs=CR)


def test_west_net_revenue_november(m):
    assert m["west_nov_cr"] == pytest.approx(76.92, abs=CR)


def test_november_is_october_times_0919(m):
    assert m["west_nov_cr"] == pytest.approx(m["west_oct_cr"] * 0.919, abs=CR)


def test_headline_movement(m):
    assert m["headline_pct"] == pytest.approx(-8.1, abs=PCT)


def test_absolute_decline(m):
    assert m["absolute_decline_cr"] == pytest.approx(6.78, abs=CR)


def test_calendar_attributed_pt(m):
    assert m["calendar_pt"] == pytest.approx(-3.2, abs=PT)


def test_calendar_attributed_rupees(m):
    assert m["calendar_cr"] == pytest.approx(2.68, abs=CR)


def test_qualified_residual_pt(m):
    assert m["residual_pt"] == pytest.approx(-4.9, abs=PT)


def test_qualified_residual_rupees(m):
    assert m["residual_cr"] == pytest.approx(4.10, abs=CR)


def test_reconciliation_calendar_plus_residual_equals_decline(m):
    """The identity CLAUDE.md requires to hold to ±₹0.02 Cr."""
    assert m["calendar_cr"] + m["residual_cr"] == pytest.approx(
        m["absolute_decline_cr"], abs=CR
    )


def test_reconciliation_against_the_registry_figures(m):
    assert 2.68 + 4.10 == pytest.approx(6.78, abs=CR)
    assert m["calendar_cr"] + m["residual_cr"] == pytest.approx(6.78, abs=CR)


def test_reconciliation_in_points(m):
    assert m["calendar_pt"] + m["residual_pt"] == pytest.approx(m["headline_pct"], abs=PT)


@pytest.mark.parametrize(
    "region,expected",
    [("North", -0.4), ("South", 1.1), ("East", -0.6), ("West", -4.9)],
)
def test_regional_residual_strip(m, region, expected):
    assert m[f"residual_pt_{region}"] == pytest.approx(expected, abs=PT)


def test_materiality_threshold_comes_from_the_semantic_layer(layer):
    materiality = layer.kpis["net_revenue"].thresholds.materiality
    assert materiality is not None
    assert materiality.value == pytest.approx(0.50, abs=1e-9)
    assert materiality.display == "₹50 L"


# --- the treated group ------------------------------------------------------


def test_west_has_140_stores(world):
    assert int((world.stores["region"] == "West").sum()) == 140


def test_34_treated_stores(world):
    assert int(world.mechanism.treated_mask.sum()) == 34


def test_34_matched_controls(world):
    assert int(world.mechanism.control_mask.sum()) == 34


def test_106_untreated_west_stores(world):
    west = world.stores["region"].to_numpy() == "West"
    assert int((west & ~world.mechanism.treated_mask).sum()) == 106


def test_treated_and_control_are_disjoint(world):
    assert not (world.mechanism.treated_mask & world.mechanism.control_mask).any()


def test_controls_are_all_west_and_untreated(world):
    west = world.stores["region"].to_numpy() == "West"
    assert (world.mechanism.control_mask <= (west & ~world.mechanism.treated_mask)).all()


# --- evidence ---------------------------------------------------------------


def test_top20_availability_before(m):
    assert m["avail_treated_before_pct"] == pytest.approx(94.0, abs=PCT)


def test_top20_availability_after(m):
    assert m["avail_treated_after_pct"] == pytest.approx(71.0, abs=PCT)


def test_affected_sku_share_of_west_volume(world):
    share = world.scenarios["2451"]["mechanism"]["affected_sku_volume_share"]
    assert share == pytest.approx(0.61, abs=1e-9)


def test_store_notes_flagging_unavailability(world, tmp_path):
    from data.generator import sources_ops

    sources_ops.emit_store_notes(world, tmp_path)
    notes = pd.read_csv(tmp_path / "store_ops" / "store_notes.csv")
    flagged = notes[notes["body"].str.contains(
        "not available|stock|khatam|nil hai|Stock out|empty", case=False, regex=True
    )]
    store_ids = world.stores["store_id"].to_numpy()
    treated = set(store_ids[world.mechanism.treated_mask])
    control = set(store_ids[world.mechanism.control_mask])

    onset = world.scenarios["2451"]["mechanism"]["onset_date"]
    during = flagged[flagged["note_date"] >= onset]
    assert during[during["store_id"].isin(treated)]["store_id"].nunique() == 21
    assert during[during["store_id"].isin(control)]["store_id"].nunique() == 2


def test_store_notes_chi_square_is_significant(world):
    """21 of 34 against 2 of 34, p < 0.001."""
    table = np.array([[21, 34 - 21], [2, 34 - 2]])
    _, p, _, _ = stats.chi2_contingency(table)
    assert p < 0.001


def test_support_tickets(world, tmp_path):
    from data.generator import sources_ops

    sources_ops.emit_tickets(world, tmp_path)
    tickets = pd.read_csv(tmp_path / "store_ops" / "tickets.csv")
    from data.generator.sources_ops import TICKET_SIZE_BODIES

    size_issue = tickets[tickets["body"].isin(TICKET_SIZE_BODIES)]
    november = size_issue[size_issue["created_date"].str.startswith("2025-11")]
    october = size_issue[size_issue["created_date"].str.startswith("2025-10")]
    assert len(november) == pytest.approx(340, abs=12)
    increase = (len(november) / len(october) - 1.0) * 100.0
    assert increase == pytest.approx(186, abs=15)


# --- the statistics that must EMERGE ---------------------------------------


def test_matched_control_did(world, panel):
    """DiD expressed against West revenue, the registry's -4.3 pt."""
    pre, post, growth = panel
    T, C = world.mechanism.treated_mask, world.mechanism.control_mask
    west = world.stores["region"].to_numpy() == "West"
    period = world.calendar["period_month"].to_numpy()
    west_oct = world.actual[np.ix_(west, period == "2025-10")].sum()

    counterfactual = pre[T] * (1.0 + growth[C].mean() / 100.0)
    loss = (post[T] - counterfactual).sum()
    assert loss / west_oct * 100.0 == pytest.approx(-4.3, abs=PT)


def test_matched_control_did_is_significant(world, panel):
    """p < 0.01, the stated bound."""
    _, _, growth = panel
    T, C = world.mechanism.treated_mask, world.mechanism.control_mask
    _, p = stats.ttest_ind(growth[T], growth[C], equal_var=False)
    assert p < 0.01


def test_parallel_trends_pretest_passes(world):
    """The stated bound is that it PASSES. See the gap test for 0.41."""
    T, C = world.mechanism.treated_mask, world.mechanism.control_mask
    dates = world.calendar["date"].to_numpy()
    onset = np.datetime64(world.scenarios["2451"]["mechanism"]["onset_date"])
    week = ((dates - onset) / np.timedelta64(7, "D")).astype(int)

    rows = [
        (k, world.actual[np.ix_(T, week == k)].sum(), world.actual[np.ix_(C, week == k)].sum())
        for k in range(-8, 0)
    ]
    frame = pd.DataFrame(rows, columns=["week", "treated", "control"])
    ratio = np.log(frame["treated"] / frame["control"])
    _, _, _, p_value, _ = stats.linregress(frame["week"], ratio)
    assert p_value > 0.05


def test_dose_response_r(world):
    """r = 0.71, emergent from per-store gap heterogeneity."""
    from data.generator.build import _dose_response_r

    r = _dose_response_r(
        world.actual, world.expected, world.stores, world.calendar, world.mechanism
    )
    assert r == pytest.approx(0.71, abs=PT)


def test_the_effect_is_causal_not_painted_in(world):
    """Setting availability_after equal to availability_before must erase it.

    This is the test that answers "did you just paint this in?". If the
    decline survived a world where availability never fell, it would be
    coming from somewhere other than the mechanism.
    """
    import copy

    from data.generator.model import apply_availability_mechanism

    scenario = copy.deepcopy(world.scenarios["2451"])
    scenario["mechanism"]["availability_after_pct"] = scenario["mechanism"][
        "availability_before_pct"
    ]
    neutral, mech = apply_availability_mechanism(
        world.expected, world.stores, world.calendar, scenario, world.streams
    )
    assert np.allclose(neutral, world.expected, rtol=1e-12)
    assert np.allclose(mech.units_multiplier, 1.0, atol=1e-12)


def test_mechanism_leaves_october_untouched(world):
    """The fault starts in November. October must be exactly the baseline."""
    period = world.calendar["period_month"].to_numpy()
    oct_cols = period == "2025-10"
    assert np.allclose(
        world.mechanism.units_multiplier[:, oct_cols], 1.0, atol=1e-12
    )


def test_mechanism_never_touches_untreated_stores(world):
    untreated = ~world.mechanism.treated_mask
    assert np.allclose(world.mechanism.units_multiplier[untreated, :], 1.0, atol=1e-12)


# --- competing hypotheses ---------------------------------------------------


def test_h3_price_parameters(world):
    h3 = world.scenarios["2451"]["competing_hypotheses"]["H3_price"]
    assert h3["asp_increase_pct"] == pytest.approx(4.1, abs=PCT)
    assert h3["volume_share"] == pytest.approx(0.08, abs=1e-9)
    assert h3["max_impact_pt"] == pytest.approx(0.4, abs=PT)
    assert h3["eliminated_on"] == "sufficiency"


def test_h3_max_impact_is_below_materiality(world, layer, m):
    """0.4 pt of ₹83.70 Cr is ₹0.33 Cr, under the ₹50 L limit."""
    h3 = world.scenarios["2451"]["competing_hypotheses"]["H3_price"]
    impact_cr = h3["max_impact_pt"] / 100.0 * m["west_oct_cr"]
    assert impact_cr < layer.kpis["net_revenue"].thresholds.materiality.value


def test_h4_marketing_cut_lands_after_the_decline_began(world):
    h4 = world.scenarios["2451"]["competing_hypotheses"]["H4_marketing"]
    assert h4["spend_change_pct"] == pytest.approx(-22, abs=PCT)
    assert h4["onset_offset_days"] == 11
    assert h4["eliminated_on"] == "precedence"


def test_h4_is_visible_in_the_marketing_feed(world, tmp_path):
    """The cut must be real in the data, not just asserted in config."""
    from data.generator import sources_ops

    sources_ops.emit_marketing_spend(world, tmp_path)
    spend = pd.read_csv(tmp_path / "context" / "marketing_spend.csv")
    west = spend[spend["region"] == "West"]
    onset = pd.Timestamp(world.scenarios["2451"]["mechanism"]["onset_date"])
    cut_from = onset + pd.Timedelta(days=11)

    before = west[pd.to_datetime(west["week_start"]) < cut_from]["spend_inr"].mean()
    after = west[pd.to_datetime(west["week_start"]) >= cut_from]["spend_inr"].mean()
    assert (after / before - 1.0) * 100.0 == pytest.approx(-22, abs=6.0)


def test_h5_complaints_onset(world):
    h5 = world.scenarios["2451"]["competing_hypotheses"]["H5_complaints"]
    assert h5["complaint_increase_pct"] == pytest.approx(31, abs=PCT)
    assert h5["onset_offset_days"] == 6


def test_both_eliminated_hypotheses_start_after_the_decline(world):
    """Test 1 eliminates H4 and H5 on precedence alone."""
    hypotheses = world.scenarios["2451"]["competing_hypotheses"]
    assert hypotheses["H4_marketing"]["onset_offset_days"] > 0
    assert hypotheses["H5_complaints"]["onset_offset_days"] > 0


# --- confidence -------------------------------------------------------------


def test_confidence_raw_is_the_weighted_sum(world, layer):
    """0.28·s1 + 0.20·s2 + 0.16·s3 + 0.14·s4 + 0.10·s5 + 0.12·s6 = 0.8876."""
    components = layer.adjudication.confidence.components
    scores = world.scenarios["2451"]["confidence"]
    total = sum(components[k].weight * scores[k] for k in ("s1", "s2", "s3", "s4", "s5", "s6"))
    assert total == pytest.approx(0.8876, abs=1e-6)
    assert round(total, 2) == pytest.approx(0.89, abs=1e-9)


def test_confidence_publishes_at_084(world):
    assert world.scenarios["2451"]["confidence"]["calibrated"] == pytest.approx(0.84, abs=0.01)


def test_no_confidence_cap_fires_on_2451(world, layer):
    """The critical clarification: Test 6 PASSES on #2451, so 0.85 must not fire."""
    scores = world.scenarios["2451"]["confidence"]
    assert scores["s2"] == pytest.approx(1.00, abs=1e-9)
    assert scores["raw"] == pytest.approx(0.8876, abs=1e-6)
    assert scores["raw"] > layer.adjudication.confidence.caps["confounder_screen_failed"].ceiling


# --- recommendation ---------------------------------------------------------


def test_expected_recovery_range(world, layer):
    """₹2.3–3.1 Cr = the recovery curve applied to ₹3.24 Cr attributable."""
    curve = layer.recovery_curves.curves["availability_restock"]
    attributable = world.scenarios["2451"]["measured_targets"]["h1_attribution_inr_cr"]
    assert attributable * curve.p25 == pytest.approx(2.3, abs=CR)
    assert attributable * curve.p75 == pytest.approx(3.1, abs=CR)


def test_recovery_curve_quartiles(layer):
    curve = layer.recovery_curves.curves["availability_restock"]
    assert curve.p25 == pytest.approx(0.71, abs=1e-9)
    assert curve.p75 == pytest.approx(0.96, abs=1e-9)
    assert curve.sample_size == 3
    assert curve.confidence_label == "Medium"


def test_roi_range(world):
    rec = world.scenarios["2451"]["recommendation"]
    cost_cr = rec["action_cost_inr_l"] / 100.0
    assert rec["expected_recovery_low_inr_cr"] / cost_cr == pytest.approx(13, abs=0.5)
    assert rec["expected_recovery_high_inr_cr"] / cost_cr == pytest.approx(17, abs=0.5)


# ===========================================================================
# Case #2467 — East, conversion rate, Oct 2025
# ===========================================================================


def test_east_store_count(world):
    assert int((world.stores["region"] == "East").sum()) == 58


def test_east_monthly_net_revenue(world):
    period = world.calendar["period_month"].to_numpy()
    months = pd.unique(period)
    rows = world.stores["region"].to_numpy() == "East"
    monthly = [world.actual[np.ix_(rows, period == mo)].sum() / 1e7 for mo in months]
    assert float(np.mean(monthly)) == pytest.approx(34.5, abs=CR)


def test_conversion_headline(world):
    t = world.scenarios["2467"]["targets"]
    assert t["conversion_before_pt"] - t["conversion_after_pt"] == pytest.approx(2.4, abs=PT)


def test_conversion_residual_and_materiality(world, layer):
    t = world.scenarios["2467"]["targets"]
    assert t["qualified_residual_pt"] == pytest.approx(-1.2, abs=PT)
    materiality = layer.kpis["conversion_rate"].thresholds.materiality
    assert abs(t["qualified_residual_pt"]) > materiality.value


def test_conversion_residual_revenue_equivalent(world):
    t = world.scenarios["2467"]["targets"]
    equivalent = (
        abs(t["qualified_residual_pt"]) / t["conversion_before_pt"]
    ) * t["east_net_revenue_monthly_inr_cr"]
    assert equivalent == pytest.approx(1.88, abs=CR)


def test_marketing_feed_is_stale(world, layer):
    hours = world.scenarios["2467"]["evidence"]["marketing_feed_staleness_hours"]
    assert hours == 74
    assert hours > layer.adjudication.triggers["T8"].max_staleness_hours


def test_competitor_calibration_below_publication_floor(world, layer):
    cal = world.scenarios["2467"]["calibration"]
    assert cal["competitor_attribution_accuracy"] == pytest.approx(0.58, abs=0.005)
    assert cal["competitor_attribution_sample"] == 12
    assert cal["competitor_attribution_accuracy"] < layer.adjudication.triggers["T7"].publication_floor


def test_competitor_hypothesis_is_structurally_unverifiable(layer):
    """T3 is a set difference over the causal graph, not a judgement."""
    hypothesis = layer.causal_graph.hypotheses["competitor_action"]
    assert hypothesis.missing_sources() == ["competitor_pricing", "competitor_footfall"]


def test_competitor_news_holds_no_pricing_or_footfall(world, tmp_path):
    from data.generator import sources_ops

    sources_ops.emit_competitor_news(world, tmp_path)
    news = pd.read_csv(tmp_path / "context" / "competitor_news.csv")
    assert not any(c in news.columns for c in ("price", "pricing", "footfall"))
    assert len(news) % 6 == 0


def test_2467_fires_t3_t4_t7(world):
    assert world.scenarios["2467"]["triggers_expected"] == ["T3", "T4", "T7"]


def test_price_response_is_explicitly_not_recommended(world):
    block = world.scenarios["2467"]["explicitly_not_recommended"]
    assert block["margin_at_risk_inr_cr"] == pytest.approx(1.2, abs=CR)
    assert block["chance_of_being_wrong"] == pytest.approx(0.50, abs=1e-9)


# ===========================================================================
# Case #2470 — West, 12 Nov 2025, daily -> Gate 1
# ===========================================================================


def test_2470_is_a_daily_check(world):
    assert world.scenarios["2470"]["grain"] == "daily"
    assert world.scenarios["2470"]["period"] == "2025-11-12"


def test_25_of_140_feeds_failed(world, tmp_path):
    """On the incident date, and only on it.

    feed_status carries one row per store per day for the whole window —
    Gate 1's row-count check needs an eight-week median to compare
    against, and a feed that failed is only visible as a failure next to a
    normal week. So the count is taken on 12 Nov 2025, not over the file.
    """
    from data.generator import sources

    sources.emit_feed_status(world, tmp_path)
    status = pd.read_csv(tmp_path / "pos_erp" / "feed_status.csv")
    incident = status[status["feed_date"] == world.scenarios["2470"]["period"]]

    west = incident[incident["region"] == "West"]
    assert len(west) == 140
    assert int((west["status"] == "FAILED").sum()) == 25
    assert (west[west["status"] == "FAILED"]["rows_loaded"] == 0).all()

    # Every other day in the window loaded for every store.
    other_days = status[status["feed_date"] != world.scenarios["2470"]["period"]]
    assert int((other_days["status"] == "FAILED").sum()) == 0
    assert len(status) == len(world.stores) * len(world.calendar)


def test_failed_feeds_carry_18_percent_of_west_daily_revenue(world, tmp_path):
    from data.generator import sources

    sources.emit_feed_status(world, tmp_path)
    assert world.diagnostics["incident_revenue_share"] * 100.0 == pytest.approx(
        18.0, abs=PCT
    )


def test_2470_opens_no_case(world):
    assert world.scenarios["2470"]["gate"]["case_opened"] is False
    assert world.scenarios["2470"]["gate"]["outcome_code"] == "DATA_INCIDENT"


def test_2470_does_not_disturb_2451(m):
    """The feed broke; the sales did not. #2451's month must still reconcile."""
    assert m["west_nov_cr"] == pytest.approx(76.92, abs=CR)
    assert m["calendar_cr"] + m["residual_cr"] == pytest.approx(6.78, abs=CR)


# ===========================================================================
# Case #2471 — All-India, Q-commerce, weekly -> Gate 3
# ===========================================================================


def test_qcomm_has_seven_weekly_points(world, tmp_path):
    from data.generator import sources_ops

    sources_ops.emit_qcomm_weekly(world, tmp_path)
    weekly = pd.read_csv(tmp_path / "context" / "qcomm_weekly.csv")
    assert len(weekly) == 7


def test_qcomm_requires_twenty_six_weeks(layer):
    assert layer.kpis["qcomm_fulfilment_rate"].baseline.min_history_weeks == 26


def test_qcomm_fails_the_history_gate(layer):
    kpi = layer.kpis["qcomm_fulfilment_rate"]
    assert kpi.has_sufficient_history() is False
    assert kpi.can_open_a_case() is False


def test_2471_opens_no_case(world):
    assert world.scenarios["2471"]["gate"]["case_opened"] is False
    assert world.scenarios["2471"]["gate"]["outcome_code"] == "INSUFFICIENT_HISTORY"


# ===========================================================================
# Case #2472 — South, Sep 2025 -> CASE OPENED
# ===========================================================================


def test_south_september_is_flat(world):
    prev = _region_month_cr(world.actual, world.stores, world.calendar, "South", "2025-08")
    curr = _region_month_cr(world.actual, world.stores, world.calendar, "South", "2025-09")
    assert (curr / prev - 1.0) * 100.0 == pytest.approx(0.0, abs=PCT)


def test_south_september_was_expected_to_rise_12_percent(world):
    prev = _region_month_cr(world.expected, world.stores, world.calendar, "South", "2025-08")
    curr = _region_month_cr(world.expected, world.stores, world.calendar, "South", "2025-09")
    assert (curr / prev - 1.0) * 100.0 == pytest.approx(12.0, abs=PCT)


def test_south_september_residual_is_minus_12_pt(world):
    prev_a = _region_month_cr(world.actual, world.stores, world.calendar, "South", "2025-08")
    curr_a = _region_month_cr(world.actual, world.stores, world.calendar, "South", "2025-09")
    prev_e = _region_month_cr(world.expected, world.stores, world.calendar, "South", "2025-08")
    curr_e = _region_month_cr(world.expected, world.stores, world.calendar, "South", "2025-09")
    actual_growth = (curr_a / prev_a - 1.0) * 100.0
    expected_growth = (curr_e / prev_e - 1.0) * 100.0
    assert actual_growth - expected_growth == pytest.approx(-12.0, abs=PT)


def test_2472_opens_a_case_and_stays_in_progress(world):
    assert world.scenarios["2472"]["gate"]["case_opened"] is True
    assert world.scenarios["2472"]["gate"]["adjudication_status"] == "in_progress"


# ===========================================================================
# Scenario separation
# ===========================================================================


def test_scenario_scopes_and_periods(world):
    expected = {
        "2451": ("West", "2025-11", "monthly"),
        "2467": ("East", "2025-10", "monthly"),
        "2470": ("West", "2025-11-12", "daily"),
        "2471": ("All-India", "2025-11", "weekly"),
        "2472": ("South", "2025-09", "monthly"),
    }
    for case, (scope, period, grain) in expected.items():
        scenario = world.scenarios[case]
        assert (scenario["scope"], scenario["period"], scenario["grain"]) == (
            scope,
            period,
            grain,
        )


def test_no_two_monthly_scenarios_share_a_scope_and_period(world):
    seen = set()
    for case, scenario in world.scenarios.items():
        key = (scenario["scope"], scenario["period"], scenario["grain"])
        assert key not in seen, f"{case} duplicates {key}"
        seen.add(key)


# ===========================================================================
# Footfall instrumentation
# ===========================================================================


def test_west_has_31_counters_and_109_nulls(world):
    west = world.stores["region"].to_numpy() == "West"
    counters = world.stores["has_footfall_counter"].to_numpy()
    assert int((west & counters).sum()) == 31
    assert int((west & ~counters).sum()) == 109


def test_footfall_is_null_for_uninstrumented_stores(world, tmp_path):
    from data.generator import sources_ops

    sources_ops.emit_footfall(world, tmp_path)
    footfall = pd.read_csv(tmp_path / "store_ops" / "footfall_daily.csv")
    uninstrumented = footfall[~footfall["counter_installed"]]
    assert uninstrumented["footfall"].isna().all()
    assert footfall[footfall["counter_installed"]]["footfall"].notna().all()


# ===========================================================================
# Deliberate defects
# ===========================================================================


def test_returns_are_restated_up_to_three_days_later(world, tmp_path):
    from data.generator import sources

    sources.emit_bill_lines(world, tmp_path)
    lines = pd.read_csv(tmp_path / "pos_erp" / "bill_lines.csv")
    returned = lines[lines["returns_amount"] > 0].copy()
    lag = (
        pd.to_datetime(returned["posted_date"])
        - pd.to_datetime(returned["txn_ts"]).dt.normalize()
    ).dt.days
    assert lag.max() == 3
    assert lag.min() == 0
    assert (lag > 0).any(), "no return was actually restated late"


def test_b2b_orders_sit_in_the_same_table(world, tmp_path):
    from data.generator import sources

    sources.emit_bill_lines(world, tmp_path)
    lines = pd.read_csv(tmp_path / "pos_erp" / "bill_lines.csv")
    assert "B2B" in set(lines["channel"])
    assert "TRANSFER" in set(lines["txn_type"])


def test_three_stores_change_store_code_mid_period(world, tmp_path):
    from data.generator import sources

    sources.emit_bill_lines(world, tmp_path)
    lines = pd.read_csv(tmp_path / "pos_erp" / "bill_lines.csv")
    per_store = lines.groupby("store_id")["store_code"].nunique()
    assert int((per_store > 1).sum()) == 3


def test_ticket_category_is_unreliable(world, tmp_path):
    from data.generator import sources_ops

    sources_ops.emit_tickets(world, tmp_path)
    tickets = pd.read_csv(tmp_path / "store_ops" / "tickets.csv")
    from data.generator.sources_ops import TICKET_SIZE_BODIES

    size_bodies = tickets[tickets["body"].isin(TICKET_SIZE_BODIES)]
    mislabelled = (size_bodies["category"] != "SIZE_ISSUE").mean()
    assert 0.15 < mislabelled < 0.45, "the category field must be wrong sometimes"


def test_store_notes_carry_no_tags(world, tmp_path):
    from data.generator import sources_ops

    sources_ops.emit_store_notes(world, tmp_path)
    notes = pd.read_csv(tmp_path / "store_ops" / "store_notes.csv")
    for forbidden in ("tag", "label", "category", "is_stockout", "topic"):
        assert forbidden not in notes.columns


def test_store_notes_are_english_and_hinglish(world, tmp_path):
    from data.generator import sources_ops

    sources_ops.emit_store_notes(world, tmp_path)
    notes = pd.read_csv(tmp_path / "store_ops" / "store_notes.csv")
    bodies = " ".join(notes["body"].tolist())
    assert "nahi" in bodies or "khatam" in bodies or "Aaj" in bodies
    assert "Routine day" in bodies or "Customer" in bodies


# ===========================================================================
# REGISTRY GAPS — pinned, not hidden
#
# Each test below asserts a value that DIFFERS from the Number Registry.
# The registry target is not adjusted (rule 10). These tests fail the
# moment the generator changes, forcing this file to be revisited.
# ===========================================================================


def test_store_level_did_does_not_reproduce_minus_4_3(world, panel):
    """GAP 1 — the registry's -4.3 pt is not a store-level figure.

    34 treated stores hold ~23% of West revenue, so a store-level DiD of
    -4.3 pt would scale to about -1.0 pt of West, not the -3.87 pt the
    registry attributes to H1. Measured at store level the effect is
    roughly -18 pt; expressed against West revenue it is -4.3 pt, which
    is what `test_matched_control_did` asserts.

    CLAUDE.md does not say which level the DiD is quoted at. This pins
    the store-level value so the ambiguity stays visible.
    """
    _, _, growth = panel
    T, C = world.mechanism.treated_mask, world.mechanism.control_mask
    store_level = growth[T].mean() - growth[C].mean()
    assert store_level < -12.0, f"store-level DiD measured {store_level:.2f} pt"
    assert not np.isclose(store_level, -4.3, atol=1.0)


def test_h1_attribution_of_3_87_pt_does_not_reproduce(world, m):
    """GAP 2 — 3.87 pt requires an undefined shrinkage step.

    The registry gives DiD -4.3 pt AND H1 attribution 3.87 pt, with
    3.87 = 0.79 x 4.9. The ratio 3.87 / 4.3 = 0.90 implies a shrinkage
    factor applied between the DiD point estimate and the attributed
    figure. CLAUDE.md never defines it.

    The mechanism here delivers the DiD target of -4.3 pt. Attributing
    3.87 pt from it requires that 0.90 factor, which is a decision, not
    a computation, so it is not implemented.
    """
    measured = world.scenarios["2451"]["measured_targets"]
    implied_shrinkage = measured["h1_attribution_pt"] / abs(
        measured["matched_control_did_pt"]
    )
    assert implied_shrinkage == pytest.approx(0.90, abs=0.01)
    assert measured["h1_attribution_pt"] == pytest.approx(
        0.79 * abs(m["residual_pt"]), abs=PT
    )


def test_unattributed_residual_is_arithmetic_not_measured(world, m):
    """GAP 2 continued — 1.03 pt and ₹0.86 Cr follow from 3.87, not data."""
    measured = world.scenarios["2451"]["measured_targets"]
    assert abs(m["residual_pt"]) - measured["h1_attribution_pt"] == pytest.approx(
        measured["unattributed_residual_pt"], abs=PT
    )
    assert m["residual_cr"] - measured["h1_attribution_inr_cr"] == pytest.approx(
        measured["unattributed_residual_inr_cr"], abs=CR
    )


def test_unattributed_residual_exceeds_materiality(world, layer, m):
    """This is what forces PARTIALLY EXPLAINED, and it does hold."""
    unattributed = world.scenarios["2451"]["measured_targets"][
        "unattributed_residual_inr_cr"
    ]
    assert unattributed > layer.kpis["net_revenue"].thresholds.materiality.value


def test_parallel_trends_p_of_0_41_does_not_reproduce(world):
    """GAP 3 — p = 0.41 exactly is not a reproducible target.

    Under the null of parallel pre-trends the p-value is uniform on
    [0, 1]; any particular value is a property of the noise draw, not of
    the mechanism. The substantive claim — that the pre-test PASSES — is
    asserted by `test_parallel_trends_pretest_passes`. Hitting 0.41
    exactly would require searching seeds, which would tell a reader
    nothing true about the data.
    """
    T, C = world.mechanism.treated_mask, world.mechanism.control_mask
    dates = world.calendar["date"].to_numpy()
    onset = np.datetime64(world.scenarios["2451"]["mechanism"]["onset_date"])
    week = ((dates - onset) / np.timedelta64(7, "D")).astype(int)
    rows = [
        (k, world.actual[np.ix_(T, week == k)].sum(), world.actual[np.ix_(C, week == k)].sum())
        for k in range(-8, 0)
    ]
    frame = pd.DataFrame(rows, columns=["week", "treated", "control"])
    _, _, _, p_value, _ = stats.linregress(
        frame["week"], np.log(frame["treated"] / frame["control"])
    )
    assert p_value > 0.05
    assert not np.isclose(p_value, 0.41, atol=0.05)


def test_action_cost_of_18_lakh_is_a_rounded_display(world, layer):
    """GAP 4 — ₹18 L is not exactly reproducible from a round unit cost.

    The playbook charges ₹53,000 per treated store. Over 34 stores that
    is ₹18.02 L, which displays as ₹18 L. Landing exactly ₹18,00,000
    would need ₹52,941.18 per store. P3 needs a stated tolerance for
    this value, or a unit cost from the business.
    """
    playbook = layer.playbooks["availability_recovery"]
    treated = int(world.mechanism.treated_mask.sum())
    cost_lakh = playbook.cost_model.unit_cost_inr * treated / 1e5
    assert cost_lakh == pytest.approx(18.0, abs=0.1)
    assert not np.isclose(cost_lakh, 18.0, atol=1e-6)


def test_2467_footfall_denominator_is_wrong_in_the_registry(world):
    """GAP 5 — "31 of 140 stores" is quoted on an EAST case.

    East has 58 stores; 140 is West's count. The generator emits East's
    own instrumented share and records the claimed figures separately.
    """
    evidence = world.scenarios["2467"]["evidence"]
    assert evidence["footfall_counters_claimed_denominator"] == 140
    east_stores = int((world.stores["region"] == "East").sum())
    assert east_stores == 58
    assert evidence["footfall_counters_claimed_denominator"] != east_stores
