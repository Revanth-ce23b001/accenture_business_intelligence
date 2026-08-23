"""ADJUDICATE — six tests, four hypotheses, and one number nobody may adjust.

The acceptance criteria are four, and they are checked exactly:

  H1 stock_out           passes all six. DiD significant at p < 0.01,
                         parallel-trends pre-test PASSES, dose-response
                         r = 0.71, coverage 0.79.
  H3 price_increase      ELIMINATED on sufficiency: a modelled maximum of
                         -0.4 pt against a -4.9 pt residual.
  H4 promo_lapse         ELIMINATED on precedence: the cut arrives after
                         the decline it is supposed to have caused.
  H5 competitor_action   ELIMINATED on precedence, and Test 6 records the
                         stock-out it cannot rule out.

Two of these are the product. H4 has the strongest raw correlation with
revenue in the whole warehouse, and an LLM asked "why did revenue fall?"
picks it. Test 1 eliminates it in one line, on arithmetic, before anything
is asked to reason.

Where a measured value differs from the Number Registry it is reported and
pinned HERE, at the value the engine actually produces, with the reason.
CLAUDE.md rule 10: never adjust the target to match the implementation —
and never quietly adjust the implementation to match the target either.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.adjudicate import (
    AdjudicateRequest,
    AdjudicationError,
    Pair,
    adjudicate,
    changepoint,
    exposed_stores,
    match_controls,
    signed,
    to_contracts,
)
from engine.db import GovernanceError
from security.policy import User

ANALYST = User(user_id="U008", persona="analyst", display_name="Meera Joshi")

#: Case #2451 as QUALIFY hands it over: West, November against October,
#: a -4.9 pt qualified residual worth INR 4.10 Cr, and the four candidates
#: GATHER brought back.
CASE_2451 = AdjudicateRequest(
    kpi="net_revenue",
    scope="West",
    grain="monthly",
    period="2025-11",
    period_start=date(2025, 11, 1),
    period_end=date(2025, 11, 30),
    comparison_start=date(2025, 10, 1),
    comparison_end=date(2025, 10, 31),
    residual_pt=-4.9,
    residual_inr=4.10e7,
    hypotheses=("stock_out", "competitor_action", "price_increase", "promo_lapse"),
    onset=date(2025, 11, 4),
    volume_shares={"stock_out": 0.61, "price_increase": 0.08},
    cause_magnitudes={"stock_out": -23.0, "price_increase": 4.1},
)

PRECEDENCE, SUFFICIENCY, DOSE, SPECIFICITY, DID, CONFOUNDERS = 1, 2, 3, 4, 5, 6


@pytest.fixture(scope="module")
def case(warehouse):
    """One adjudication of #2451, shared across the module."""
    return adjudicate(warehouse, ANALYST, CASE_2451)


# ===========================================================================
# ACCEPT 1 — H1 passes all six
# ===========================================================================


def test_h1_survives_every_test(case):
    h1 = case.verdict("stock_out")
    assert not h1.eliminated
    failures = [
        f"T{item.test_id} {item.name}: {item.detail}"
        for item in h1.findings
        if item.testable and not item.passed
    ]
    assert failures == []


def test_h1_did_is_significant_and_negative(case):
    estimate = case.verdict("stock_out").did
    assert estimate is not None
    assert estimate.point_pt < 0
    assert estimate.p_value < 0.01
    assert estimate.matched_pairs == 34


def test_h1_parallel_trends_pretest_passes(case):
    """A HIGH p-value is the pass: no pre-trend was DETECTED.

    It can never prove parallel trends — only fail to reject them — and
    the evidence says so rather than claiming the design is validated.
    """
    finding = case.verdict("stock_out").finding(DID)
    assert "passes" in finding.detail
    assert finding.facts["pretest_p"] > 0.05


def test_h1_dose_response_r_is_zero_point_seven_one(case):
    """The registry's r = 0.71, reproduced to two places."""
    finding = case.verdict("stock_out").finding(DOSE)
    assert finding.passed
    assert finding.statistic == pytest.approx(0.71, abs=0.02)
    assert finding.facts["stores"] == 140, "every store in scope, not the treated ones"


def test_h1_specificity_splits_thirty_four_against_one_hundred_and_six(case):
    finding = case.verdict("stock_out").finding(SPECIFICITY)
    assert finding.passed
    assert finding.facts["present_n"] == 34
    assert finding.facts["absent_n"] == 106
    assert finding.p_value < 0.001


def test_h1_confounder_screen_passes(case):
    """CLAUDE.md is explicit: no cap fires on #2451.

    Test 6 screens promo, weather, competitor action and staffing across
    the treated and control groups and finds no difference. H2 is a
    COMPETING HYPOTHESIS for the residual, not a confounder of H1, and it
    is handled by the verdict table rather than by a confidence cap.
    """
    finding = case.verdict("stock_out").finding(CONFOUNDERS)
    assert finding.passed
    assert case.verdict("stock_out").unresolved_confounders == ()


def test_coverage_is_seventy_nine_percent(case):
    """0.79, and it falls out rather than being aimed at.

    The attributed figure is the near end of the 95% interval, and 95% is
    the conventional level. Coverage is |lower bound| / 4.9.
    """
    assert case.coverage == pytest.approx(0.79, abs=0.01)


def test_the_unattributed_residual_clears_materiality(case):
    """INR 0.84 Cr against a INR 0.50 Cr limit. THIS is what forces
    PARTIALLY EXPLAINED, and it is computed, not asserted."""
    assert case.unattributed_pt == pytest.approx(1.03, abs=0.05)
    assert case.unattributed_inr / 1e7 == pytest.approx(0.86, abs=0.05)
    assert case.unattributed_inr / 1e7 > 0.50


def test_attribution_is_the_interval_and_never_the_point_estimate(case):
    """Claiming the point estimate claims a precision the data has not got."""
    h1 = case.verdict("stock_out")
    assert abs(h1.attributed_pt) < abs(h1.did.point_pt)
    # The NEAR end of the interval — the end closer to zero — because that
    # is the part of the movement the estimate can defend.
    near = min(abs(h1.did.lower_pt), abs(h1.did.upper_pt))
    assert abs(h1.attributed_pt) == pytest.approx(near, abs=1e-9)


# ===========================================================================
# ACCEPT 2 — H3 eliminated on sufficiency
# ===========================================================================


def test_h3_is_eliminated_and_sufficiency_is_named(case):
    h3 = case.verdict("price_increase")
    assert h3.eliminated
    assert "sufficiency" in h3.elimination_reason


def test_h3_modelled_maximum_is_four_tenths_of_a_point(case):
    """8% of volume x 4.1% x elasticity gives at most -0.4 pt against -4.9.

    The registry's figure, reproduced. A true hypothesis, too small to
    matter, eliminated on arithmetic rather than on correlation.
    """
    finding = case.verdict("price_increase").finding(SUFFICIENCY)
    assert not finding.passed
    assert finding.statistic == pytest.approx(0.4, abs=0.05)
    assert finding.facts["volume_share"] == 0.08
    assert finding.facts["cause_magnitude_pct"] == 4.1


def test_a_declared_elasticity_says_so_and_carries_a_margin(case):
    """This warehouse holds no price experiment, so no elasticity fits.

    The bound falls back to the published category figure, the finding
    says DECLARED in as many words, names where it came from, and the
    elimination has to survive that figure being wrong by 3x. It does, by
    an order of magnitude.
    """
    finding = case.verdict("price_increase").finding(SUFFICIENCY)
    assert finding.facts["elasticity_provenance"] == "declared"
    assert "DECLARED, not fitted" in finding.detail
    assert "Merchandising" in finding.detail
    margin = finding.facts["elimination_margin"]
    assert finding.statistic * margin < finding.facts["required_pt"]


# ===========================================================================
# ACCEPT 3 and 4 — H4 and H5 eliminated on precedence
# ===========================================================================


def test_h4_marketing_cut_arrives_after_the_decline(case):
    """The strongest correlation in the warehouse, killed by a date.

    An LLM asked "why did revenue fall?" on this data picks this one.
    """
    h4 = case.verdict("promo_lapse")
    assert h4.eliminated
    assert "precedence" in h4.elimination_reason
    finding = h4.finding(PRECEDENCE)
    assert finding.facts["lag_days"] > 0
    assert finding.facts["cause_onset"] > finding.facts["effect_onset"]
    assert "A cause cannot follow its effect." in finding.detail


def test_h5_complaints_arrive_after_the_decline(case):
    h5 = case.verdict("competitor_action")
    assert h5.eliminated
    assert "precedence" in h5.elimination_reason
    finding = h5.finding(PRECEDENCE)
    assert finding.facts["cause_onset"] == date(2025, 11, 10), (
        "six days after the fault began, which is where the generator put it"
    )
    assert finding.facts["lag_days"] > 0


def test_h5_carries_an_unresolved_stock_out_confounder(case):
    """A complaint spike is what an empty shelf produces too.

    The screen cannot rule it out — this hypothesis has no store-level
    cause, so it has no matched control group to balance across — and
    UNSCREENED IS NOT RESOLVED. Test 6 fails and says which one.
    """
    h5 = case.verdict("competitor_action")
    assert "stock_out" in h5.unresolved_confounders
    assert not h5.finding(CONFOUNDERS).passed


def test_every_eliminated_hypothesis_names_a_machine_readable_reason(case):
    for verdict in case.eliminated:
        assert verdict.elimination_reason
        for reason in verdict.elimination_reason.split(", "):
            assert reason in {"precedence", "sufficiency"}


def test_an_eliminated_hypothesis_is_attributed_nothing(case):
    for verdict in case.eliminated:
        assert verdict.attributed_pt == 0.0
        assert verdict.attributed_share == 0.0


# ===========================================================================
# The hard gates run FIRST, and they run alone
# ===========================================================================


def test_an_eliminated_hypothesis_never_reaches_the_weighted_tests(case):
    """A dead hypothesis does not get to vote on how confident we are."""
    weighted = {DOSE, SPECIFICITY, DID}
    for verdict in case.eliminated:
        assert {item.test_id for item in verdict.findings} & weighted == set()


def test_a_hypothesis_cannot_borrow_another_ones_treatment_group(case):
    """Exposure is derived PER HYPOTHESIS from its own cause series.

    An earlier implementation matched once, off a scenario column, and
    handed the same treated group to every candidate — at which point the
    price rise reported the stock-out's difference-in-differences as its
    own and coverage came out at 100%.
    """
    h1 = case.verdict("stock_out")
    others = [
        verdict for verdict in case.verdicts
        if verdict.tag != "stock_out" and verdict.did is not None
    ]
    for verdict in others:
        assert verdict.did.point_pt != h1.did.point_pt


def test_the_engine_reads_no_scenario_flag():
    """`is_treated_2451` is a fact about the generator, not about the world.

    Which stores an event reached is a CONCLUSION of Tests 3, 4 and 5. An
    engine that reads it off a column has assumed its own answer.
    """
    for path in (Path("engine") / "adjudicate").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "is_treated_2451" not in source, path
        assert "is_matched_control_2451" not in source, path


# ===========================================================================
# NO MODEL CALL TOUCHES THIS STAGE
# ===========================================================================


def test_nothing_under_adjudicate_imports_the_model_layer():
    """CLAUDE.md: no LLM call may touch this module. Checked statically.

    GATHER already asked the model what a document was ABOUT. ADJUDICATE
    asks whether the evidence supports the hypothesis, and that is
    arithmetic all the way down.
    """
    offenders: list[str] = []
    for path in sorted((Path("engine") / "adjudicate").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "llm" or name.startswith("llm."):
                    offenders.append(f"{path}:{node.lineno} imports {name}")
    assert offenders == [], "\n".join(offenders)


def test_every_finding_is_produced_by_code(case):
    for evidence in case.evidence:
        assert evidence.produced_by == "code"


# ===========================================================================
# Governance — rule 5 holds here too
# ===========================================================================


def test_a_persona_outside_the_contract_is_refused(warehouse):
    with pytest.raises(GovernanceError):
        adjudicate(warehouse, User(user_id="U999", persona="intern"), CASE_2451)


def test_an_unknown_kpi_is_refused(warehouse):
    with pytest.raises(AdjudicationError):
        adjudicate(warehouse, ANALYST, replace(CASE_2451, kpi="not_a_kpi"))


# ===========================================================================
# The pieces, on their own
# ===========================================================================


def test_changepoint_finds_a_step_where_one_was_put(layer):
    index = pd.date_range("2025-01-01", periods=60, freq="D")
    values = np.concatenate([np.full(30, 100.0), np.full(30, 60.0)])
    series = pd.Series(values, index=index)
    assert changepoint(series, layer.adjudicate.precedence) == date(2025, 1, 31)


def test_changepoint_finds_nothing_in_a_flat_series(layer):
    index = pd.date_range("2025-01-01", periods=60, freq="D")
    series = pd.Series(np.full(60, 100.0), index=index)
    assert changepoint(series, layer.adjudicate.precedence) is None


def test_signed_points_a_falling_cause_the_same_way_as_the_decline():
    values = np.array([1.0, -2.0])
    assert list(signed(values, "down")) == [-1.0, 2.0]
    assert list(signed(values, "up")) == [1.0, -2.0]


def test_exposure_selects_nobody_when_nothing_happened(layer):
    dose = pd.Series(np.full(100, 94.0), index=[f"S{i:03d}" for i in range(100)])
    exposure = exposed_stores(dose, layer.adjudicate.exposure, direction="down")
    assert not exposure.testable
    assert exposure.exposed == ()


def test_exposure_can_never_select_a_majority(layer):
    """A threshold set from the median cannot put most of the scope on one
    side of it. This is why there is no "too many exposed" ceiling in the
    config: it would be unreachable code pretending to be a safeguard.

    The case it would have caught — a movement that reached the whole
    estate — is caught upstream by QUALIFY's Gate 4, which escalates it as
    MARKET_CASE before ADJUDICATE is asked anything.
    """
    rng = np.random.default_rng(20260822)
    for low in (10, 40, 60, 90):
        values = np.concatenate([
            rng.normal(60.0, 3.0, low), rng.normal(94.0, 1.0, 100 - low)
        ])
        dose = pd.Series(values, index=[f"S{i:03d}" for i in range(100)])
        exposure = exposed_stores(dose, layer.adjudicate.exposure, direction="down")
        assert exposure.share <= 0.5, f"{low} low stores gave {exposure.share:.0%}"


def test_exposure_needs_stores_on_both_sides(layer):
    """A split with nobody on one side is not a split."""
    rng = np.random.default_rng(20260822)
    values = np.concatenate([rng.uniform(93.0, 95.0, 98), np.array([20.0, 21.0])])
    dose = pd.Series(values, index=[f"S{i:03d}" for i in range(100)])
    exposure = exposed_stores(dose, layer.adjudicate.exposure, direction="down")
    assert not exposure.testable
    assert "are needed on each side" in exposure.reason


def test_exposure_finds_the_thirty_four(warehouse, layer):
    """34 of 140, from the data, with no flag consulted."""
    rows = warehouse.execute(
        """
        SELECT i.store_id,
               100.0 * SUM(CASE WHEN i.on_hand_units > 0 THEN 1 ELSE 0 END) / COUNT(*)
        FROM fact_inventory_snapshot AS i
        JOIN dim_store AS d USING (store_id)
        JOIN dim_sku   AS s USING (sku_id)
        WHERE s.is_top20 AND i.snapshot_hour_ist = 12 AND d.region = 'West'
          AND i.snapshot_date BETWEEN DATE '2025-11-01' AND DATE '2025-11-30'
        GROUP BY 1
        """
    ).fetchall()
    dose = pd.Series({row[0]: row[1] for row in rows})
    exposure = exposed_stores(dose, layer.adjudicate.exposure, direction="down")
    assert exposure.testable
    assert len(exposure.exposed) == 34
    assert len(exposure.unexposed) == 106

    flagged = {
        row[0] for row in warehouse.execute(
            "SELECT store_id FROM dim_store WHERE region = 'West' AND is_treated_2451"
        ).fetchall()
    }
    assert set(exposure.exposed) == flagged, (
        "the derived exposure and the generator's own treatment agree, "
        "which is the point: the engine found it rather than being told"
    )


def test_matching_is_optimal_and_not_order_dependent(warehouse, layer):
    """The pairing minimises TOTAL distance, so the sort cannot change it.

    Greedy nearest-neighbour was the first implementation: whichever
    treated store sorted first took the best control, and the estimate
    moved when the store ids did.
    """
    from engine.adjudicate.series import load_panel

    panel = load_panel(
        warehouse, ANALYST, layer, kpi="net_revenue", scope="West",
        period_start=date(2025, 11, 1), period_end=date(2025, 11, 30),
        comparison_start=date(2025, 10, 1), comparison_end=date(2025, 10, 31),
        pre_trend_weeks=8, onset=date(2025, 11, 4),
    )
    treated = [
        row[0] for row in warehouse.execute(
            "SELECT store_id FROM dim_store WHERE region = 'West' AND is_treated_2451"
        ).fetchall()
    ]
    forward = match_controls(panel, layer.adjudicate.did, treated)
    backward = match_controls(panel, layer.adjudicate.did, list(reversed(treated)))
    assert forward.pairs == backward.pairs
    assert len(forward.pairs) == 34
    assert forward.dropped == ()
    assert forward.worst_distance <= layer.adjudicate.did.matching.max_distance


def test_a_matched_pair_is_a_record_and_not_a_positional_tuple():
    pair = Pair("S001", "S002", 0.5)
    assert pair.treated == "S001"
    assert pair.control == "S002"
    assert pair.distance == 0.5


def test_the_adjudication_renders_as_contracts(case, layer):
    hypotheses = to_contracts(case, layer)
    assert len(hypotheses) == len(case.verdicts)
    by_id = {item.hypothesis_id: item for item in hypotheses}
    assert by_id["stock_out"].status == "supported"
    assert by_id["promo_lapse"].status == "eliminated"
    assert 0.0 <= by_id["stock_out"].attributed_share <= 1.0


# ===========================================================================
# REGISTRY GAPS — pinned, not hidden
#
# Each test below asserts a value that DIFFERS from the Number Registry.
# The registry target is NOT adjusted (rule 10), and the implementation is
# not quietly bent to meet it either. These fail the moment the method or
# the generator changes, which forces this section to be revisited.
# ===========================================================================


def test_the_did_does_not_reproduce_minus_4_3_exactly(case):
    """GAP 1 — the registry says -4.3 pt; the engine measures -4.77.

    The whole difference is WHICH 34 of the 106 untreated stores become
    controls:

        the engine's optimally matched 34 grew   -2.66% in November
        the generator's own control set grew     -5.13%
        the whole untreated pool grew            -3.32%

    Both control sets are draws either side of the pool, on opposite
    sides. The standard error on the estimate is about 0.45 pt, so -4.77
    and -4.33 are roughly one standard error apart: statistically the same
    number, reported to a precision neither of them has.

    The registry's figure is one particular matched draw — the
    generator's. Tuning the matcher until it reproduced that draw would be
    fitting the method to the answer, which is the one thing this project
    may not do. It is pinned here instead.
    """
    estimate = case.verdict("stock_out").did
    assert estimate.point_pt == pytest.approx(-4.77, abs=0.15)
    assert abs(estimate.point_pt - (-4.3)) < 1.5 * estimate.standard_error_pt


def test_the_parallel_trends_p_value_is_not_the_registrys_0_41(case):
    """GAP 2 — the registry says p = 0.41; the engine measures about 0.74.

    Both PASS, which is the substantive claim: no pre-trend was detected
    over the eight weeks before the onset. The particular p-value depends
    on which stores are in the control group, and GAP 1 above is why those
    differ. A pre-test p-value is not a quantity anybody acts on; the
    pass is.
    """
    finding = case.verdict("stock_out").finding(DID)
    assert finding.facts["pretest_p"] == pytest.approx(0.74, abs=0.15)
    assert finding.facts["pretest_p"] > 0.05


def test_the_attribution_chain_closes_without_the_undefined_shrinkage(case):
    """GAP 3, RESOLVED — and worth recording because it was open.

    `tests/test_number_registry.py` pins an open gap: the registry gives a
    DiD of -4.3 pt AND an H1 attribution of 3.87 pt, and 3.87 / 4.3 = 0.90
    implies a shrinkage factor CLAUDE.md never defines.

    ADJUDICATE closes it without one. The attributed figure is the near
    end of the 95% interval on the engine's own -4.77 pt estimate, which
    is 3.89 pt — the registry's 3.87, to two places. The undefined
    shrinkage was a confidence interval all along, and the rest of the
    chain follows from it:

        attribution   3.89 pt   registry 3.87 pt
        coverage      0.795     registry 0.79
        unattributed  1.01 pt   registry 1.03 pt
        unattributed  INR 0.84 Cr   registry INR 0.86 Cr

    all of it computed, none of it entered.
    """
    h1 = case.verdict("stock_out")
    assert abs(h1.attributed_pt) == pytest.approx(3.87, abs=0.05)
    assert case.coverage == pytest.approx(0.79, abs=0.01)
    assert case.unattributed_pt == pytest.approx(1.03, abs=0.05)
    assert case.unattributed_inr / 1e7 == pytest.approx(0.86, abs=0.05)


def test_no_price_elasticity_can_be_fitted_from_this_warehouse(case):
    """GAP 4 — Test 2's elasticity is DECLARED for H3, not observed.

    `adjudicate.yaml` says elasticities are fitted from this warehouse's
    own history, on the argument that a sufficiency test run on a textbook
    elasticity is a test of the textbook. For the price hypothesis it
    cannot be: prices in this data move with mix and never against a
    demand curve, and every specification tried — store x day, store x
    week, within-store, within-SKU, within store x SKU — explains under
    4% of variance.

    So the bound falls back to the published category figure and says
    DECLARED in the finding itself, and the elimination has to survive
    that figure being wrong by 3x. It survives it by an order of
    magnitude, which is the only reason drawing the conclusion is
    defensible.
    """
    finding = case.verdict("price_increase").finding(SUFFICIENCY)
    assert finding.facts["elasticity_provenance"] == "declared"
    assert finding.facts["r_squared"] < 0.05
