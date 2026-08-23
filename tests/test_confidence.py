"""Confidence — six components, four caps, and the 0.89 -> 0.84 step.

The acceptance criteria are:

  the weighted sum is exactly the formula in CLAUDE.md
  every cap has an isolated test, and NO CAP FIRES ON #2451
  isotonic_map(0.8876) == 0.84 +/- 0.01
  expected calibration error <= 0.05
  the band table is computed from the seeded cases, not written down
  below the minimum closed cases the map is the identity and says so

The cap tests are the ones to read. CLAUDE.md carries an explicit warning
that an implementation which reads H2's unverifiability as a confounder of
H1 produces 0.85 and is wrong, and
`test_no_cap_fires_on_2451` is what stops that happening again.

Where the engine's own components differ from the Number Registry's
declared column, the difference is pinned in the REGISTRY GAPS section at
the end rather than tuned away (rule 10).
"""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from engine.abstain import EXPLAINED, PARTIALLY_EXPLAINED, HeldResidual, decide
from engine.adjudicate import AdjudicateRequest, adjudicate
from engine.confidence import (
    CONFOUNDER,
    KEYS,
    LOW_RELIABILITY,
    MISSING_SOURCE,
    PRECEDENCE,
    apply_caps,
    evaluate_caps,
    fit_calibration,
    forced_triggers,
    score_case,
    to_contract,
)
from engine.confidence.components import Component
from security.policy import User

ANALYST = User(user_id="U008", persona="analyst", display_name="Meera Joshi")

CASE_2451 = AdjudicateRequest(
    kpi="net_revenue", scope="West", grain="monthly", period="2025-11",
    period_start=date(2025, 11, 1), period_end=date(2025, 11, 30),
    comparison_start=date(2025, 10, 1), comparison_end=date(2025, 10, 31),
    residual_pt=-4.9, residual_inr=4.10e7,
    hypotheses=("stock_out", "competitor_action", "price_increase", "promo_lapse"),
    onset=date(2025, 11, 4),
    volume_shares={"stock_out": 0.61, "price_increase": 0.08},
    cause_magnitudes={"stock_out": -23.0, "price_increase": 4.1},
)

#: The Number Registry's declared component column for #2451, from
#: `data/generator/config/scenario_2451.yaml`. Used to check the FORMULA,
#: which is a separate question from whether the engine reproduces the
#: components.
REGISTRY_COMPONENTS = {
    "s1": 0.93, "s2": 1.00, "s3": 0.84, "s4": 0.90, "s5": 0.72, "s6": 0.79,
}
REGISTRY_RAW = 0.8876
REGISTRY_PUBLISHED = 0.84


@pytest.fixture(scope="module")
def adjudication(warehouse):
    return adjudicate(warehouse, ANALYST, CASE_2451)


@pytest.fixture(scope="module")
def tagged_stores(warehouse, layer):
    """Stores the classifier tagged for stock_out, from a real GATHER run.

    s3 compares the two lanes, so it needs both. Running only ADJUDICATE
    and scoring the result gives an empty unstructured lane and a source
    agreement of zero, which is a statement about the test harness rather
    than about the case.
    """
    from engine.gather import GatherRequest, gather

    result = gather(
        warehouse, ANALYST,
        GatherRequest(
            kpi=CASE_2451.kpi, scope=CASE_2451.scope, grain=CASE_2451.grain,
            period=CASE_2451.period,
            period_start=CASE_2451.period_start, period_end=CASE_2451.period_end,
            comparison_start=CASE_2451.comparison_start,
            comparison_end=CASE_2451.comparison_end,
        ),
    )
    spec = layer.gather.unstructured.classification
    return frozenset(
        tag.store_id
        for tag in result.unstructured.all_tags
        if tag.hypothesis_tag == "stock_out" and tag.accepted(spec) and tag.store_id
    )


@pytest.fixture(scope="module")
def score(warehouse, adjudication, tagged_stores):
    return score_case(
        warehouse, ANALYST, adjudication,
        leading_tag="stock_out", case_type="availability_attribution",
        evidence=adjudication.evidence,
        unstructured_present=tagged_stores,
    )


@pytest.fixture(scope="module")
def calibration(warehouse):
    return fit_calibration(warehouse, ANALYST)


# ===========================================================================
# The formula
# ===========================================================================


def test_the_weights_are_the_ones_claude_md_specifies(layer):
    weights = {
        key: spec.weight
        for key, spec in layer.adjudication.confidence.components.items()
    }
    assert weights == {
        "s1": 0.28, "s2": 0.20, "s3": 0.16, "s4": 0.14, "s5": 0.10, "s6": 0.12
    }
    assert sum(weights.values()) == pytest.approx(1.0)


def test_the_formula_reproduces_0_8876_from_the_registry_components(layer):
    """The arithmetic, checked on its own.

    Given the six values CLAUDE.md declares for #2451, the weighted sum
    is 0.8876 exactly. That is a statement about the FORMULA, and it is
    true. Whether this warehouse's data produces those six values is a
    different question, answered in REGISTRY GAPS below.
    """
    specs = layer.adjudication.confidence.components
    components = tuple(
        Component(key, specs[key].name, specs[key].weight, value, "", specs[key].method)
        for key, value in REGISTRY_COMPONENTS.items()
    )
    from engine.confidence import weighted_sum

    assert weighted_sum(components) == pytest.approx(REGISTRY_RAW, abs=1e-9)


def test_the_weighted_sum_is_the_sum_of_the_contributions(score):
    assert score.raw == pytest.approx(
        sum(item.contribution for item in score.components), abs=1e-12
    )


def test_all_six_components_are_present_and_in_order(score):
    assert tuple(item.key for item in score.components) == KEYS


def test_every_component_explains_itself(score):
    for item in score.components:
        assert item.detail.strip(), item.key
        assert 0.0 <= item.value <= 1.0, item.key


# ===========================================================================
# THE CAPS — and the one that must not fire
# ===========================================================================


def test_no_cap_fires_on_2451(score, adjudication):
    """CLAUDE.md's critical clarification, enforced.

    "On #2451 no cap fires. H2 (competitor promotion) is a COMPETING
    HYPOTHESIS for the residual, not a confounder of H1. Test 6 screened
    promo, weather, competitor openings and staffing across the treated
    and control groups and found no difference — it PASSES. If your
    implementation applies the 0.85 cap here and produces 0.85, it is
    wrong."

    So: no cap, and in particular not that one.
    """
    assert score.caps == ()
    assert score.after_caps == score.raw
    assert CONFOUNDER not in {cap.name for cap in score.caps}
    assert score.after_caps != pytest.approx(0.85, abs=1e-9)


def test_the_competing_hypothesis_is_handled_by_the_verdict_not_a_cap(score, layer):
    """H2 changes the VERDICT and does not touch the confidence.

    The two questions look alike and are not. A confounder asks whether
    something else produced THIS hypothesis's signature in the treated
    group. A competitor asks whether something else accounts for the part
    of the residual this hypothesis does not.
    """
    assert score.caps == ()
    blocked = decide(
        coverage=0.79, confidence=score.calibrated, triggers=(),
        held=(
            HeldResidual(
                "competitor_action", "Competitor promotion", verifiable=False,
                residual_inr=0.86e7, materiality_inr=0.50e7,
                missing_sources=("competitor_pricing",),
            ),
        ),
        layer=layer,
    )
    assert blocked.value == PARTIALLY_EXPLAINED
    assert blocked.reason_code == "live_unverifiable_above_materiality"


def test_the_confounder_cap_fires_when_a_confounder_of_this_hypothesis_is_open(
    adjudication, layer
):
    """Isolated: an unresolved confounder OF THIS HYPOTHESIS caps at 0.85."""
    verdict = replace(
        adjudication.verdict("stock_out"),
        confounders=adjudication.verdict("competitor_action").confounders,
    ) if False else adjudication.verdict("competitor_action")
    caps = evaluate_caps(verdict, (), layer)
    names = {cap.name for cap in caps}
    assert CONFOUNDER in names
    cap = next(item for item in caps if item.name == CONFOUNDER)
    assert cap.ceiling == 0.85
    assert apply_caps(0.99, [cap]) == pytest.approx(0.85)


def test_the_missing_source_cap_fires_and_forces_t3(adjudication, layer):
    """Isolated: a source the organisation does not hold caps at 0.45.

    competitor_action requires competitor_pricing and competitor_footfall
    and the organisation holds neither.
    """
    verdict = adjudication.verdict("competitor_action")
    assert verdict.missing_sources
    caps = evaluate_caps(verdict, (), layer)
    cap = next(item for item in caps if item.name == MISSING_SOURCE)
    assert cap.ceiling == 0.45
    assert cap.forced_trigger == "T3"
    assert apply_caps(0.99, [cap]) == pytest.approx(0.45)
    assert forced_triggers(caps) == ("T3",)


def test_the_precedence_cap_zeroes_and_eliminates(adjudication, layer):
    """Isolated: a cause that arrived after its effect has no confidence.

    promo_lapse failed temporal precedence by ten days.
    """
    verdict = adjudication.verdict("promo_lapse")
    caps = evaluate_caps(verdict, (), layer)
    cap = next(item for item in caps if item.name == PRECEDENCE)
    assert cap.ceiling == 0.0
    assert cap.eliminates
    assert apply_caps(0.99, [cap]) == pytest.approx(0.0)


def test_the_low_reliability_cap_fires_on_evidence_that_is_mostly_text(
    adjudication, layer
):
    """Isolated: evidence dominated by weights below the floor caps at 0.65."""
    from tests.conftest import make_evidence

    text_only = tuple(
        make_evidence(
            f"ev.note.{index}", kind="store_note", reliability=0.55,
            source_system="store_notes", method="lookup",
        )
        for index in range(10)
    )
    caps = evaluate_caps(adjudication.verdict("stock_out"), text_only, layer)
    cap = next(item for item in caps if item.name == LOW_RELIABILITY)
    assert cap.ceiling == 0.65
    assert apply_caps(0.99, [cap]) == pytest.approx(0.65)


def test_the_lowest_ceiling_wins_when_several_caps_fire(adjudication, layer):
    """A cap cannot be outvoted, including by another cap."""
    from tests.conftest import make_evidence

    text_only = tuple(
        make_evidence(
            f"ev.note.{i}", kind="social_mention", reliability=0.35,
            source_system="review_feed",
        )
        for i in range(10)
    )
    caps = evaluate_caps(adjudication.verdict("competitor_action"), text_only, layer)
    assert len(caps) >= 2
    assert apply_caps(0.99, caps) == pytest.approx(min(cap.ceiling for cap in caps))


def test_every_cap_that_fires_is_reported_not_only_the_binding_one(
    adjudication, layer
):
    from tests.conftest import make_evidence

    text_only = tuple(
        make_evidence(
            f"ev.note.{i}", kind="social_mention", reliability=0.35,
            source_system="review_feed",
        )
        for i in range(10)
    )
    caps = evaluate_caps(adjudication.verdict("competitor_action"), text_only, layer)
    names = {cap.name for cap in caps}
    assert MISSING_SOURCE in names and LOW_RELIABILITY in names, (
        "a case file that reported only the lowest ceiling would hide the "
        "second thing that went wrong"
    )


# ===========================================================================
# CALIBRATION — the 0.89 -> 0.84 step
# ===========================================================================


def test_the_ledger_holds_213_closed_cases(calibration):
    assert calibration.total_cases == 213
    assert calibration.scored_cases == 145
    assert calibration.abstained_cases == 68


def test_the_abstention_rate_is_about_a_third(calibration):
    assert calibration.abstention_rate == pytest.approx(0.32, abs=0.01)


def test_the_isotonic_map_sends_0_8876_to_0_84(calibration):
    """The step the whole demo turns on, asserted to CLAUDE.md's tolerance.

    "the model scored this 89%; our track record on similar cases says we
    run about five points hot, so we publish 84%."
    """
    assert calibration.map(REGISTRY_RAW) == pytest.approx(REGISTRY_PUBLISHED, abs=0.01)
    assert calibration.shift(REGISTRY_RAW) < 0, "the map cools the score, it does not warm it"


def test_the_expected_calibration_error_is_within_five_points(calibration, layer):
    """The second and only other thing CLAUDE.md asserts about the ledger.

    Measured over the RAW scores, because that is what the map exists to
    correct. Measured after the map it is near zero in-sample and says
    nothing.
    """
    limit = layer.adjudication.confidence.calibration.max_expected_calibration_error
    assert calibration.ece_raw <= limit
    assert calibration.ece_calibrated <= calibration.ece_raw


def test_the_band_table_is_computed_and_not_declared(calibration):
    """Every number in the printed table comes off the seeded rows.

    CLAUDE.md: "Everything else in the ledger is computed and printed from
    the seeded data — do not hardcode the band table."
    """
    assert calibration.bands
    for band in calibration.bands:
        assert band.cases > 0
        if not band.thin:
            assert 0.0 <= band.accuracy <= 1.0
            assert band.low <= band.mean_confidence <= band.high
    assert sum(band.cases for band in calibration.bands) == calibration.scored_cases


def test_the_top_bands_run_hot_which_is_what_the_map_corrects(calibration):
    top = [band for band in calibration.bands if band.low >= 0.8 and not band.thin]
    assert top
    assert all(band.gap > 0 for band in top), (
        "the demo's claim is that the engine runs hot where it is most confident"
    )


def test_the_map_never_reorders_two_cases(calibration):
    """The one thing a calibration map must not do.

    If the engine scores A above B, the published figures have to keep
    that order or the ranking a human reads is not the ranking the engine
    produced. Isotonic regression is the least-squares fit subject to
    exactly this constraint.
    """
    probes = [index / 100 for index in range(101)]
    mapped = [calibration.map(value) for value in probes]
    assert mapped == sorted(mapped)


def test_below_the_minimum_closed_cases_the_map_is_the_identity(warehouse, layer):
    """A map fitted to a dozen cases is a map fitted to noise.

    competitor_attribution has 12 published cases, below the 25 the fit
    needs, so the map is the identity and the result says CALIBRATING and
    carries n.
    """
    thin = fit_calibration(
        warehouse, ANALYST, layer=layer, case_type="competitor_attribution"
    )
    assert thin.calibrating
    assert thin.scored_cases < layer.adjudication.confidence.calibration.min_cases_for_fit
    for probe in (0.30, 0.62, 0.8876, 0.97):
        assert thin.map(probe) == pytest.approx(probe)
    assert "CALIBRATING" in thin.render()
    assert str(thin.scored_cases) in thin.render()


def test_the_case_type_record_is_what_trigger_t7_reads(calibration):
    """Two types decide two scenarios."""
    competitor, competitor_n = calibration.accuracy_for("competitor_attribution")
    availability, availability_n = calibration.accuracy_for("availability_attribution")
    assert competitor == pytest.approx(0.58, abs=0.01)
    assert competitor_n == 12
    assert availability > 0.70
    assert availability_n == 38


def test_an_unknown_case_type_has_no_record_rather_than_a_bad_one(calibration):
    accuracy, cases = calibration.accuracy_for("no_such_case_type")
    assert accuracy is None
    assert cases == 0


def test_abstained_cases_are_excluded_from_the_fit(calibration):
    """An abstention published nothing, so there is nothing it got wrong.

    Counting one as a miss would punish the engine for the behaviour this
    project exists to encourage.
    """
    assert calibration.scored_cases + calibration.abstained_cases == 213
    assert sum(band.cases for band in calibration.bands) == calibration.scored_cases


# ===========================================================================
# The score, end to end on #2451
# ===========================================================================


def test_the_score_is_ordered_raw_then_capped_then_published(score):
    assert score.after_caps <= score.raw
    assert 0.0 <= score.calibrated <= 1.0
    assert score.calibration.scored_cases == 145


def test_the_published_figure_is_the_map_of_the_capped_one(score):
    assert score.calibrated == pytest.approx(
        score.calibration.map(score.after_caps), abs=1e-12
    )


def test_the_score_emits_evidence_for_every_component(score):
    ids = {item.evidence_id for item in score.evidence}
    for key in KEYS:
        assert f"confidence.{key}" in ids
    assert "confidence.raw" in ids
    assert "confidence.published" in ids


def test_no_bare_float_leaves_the_module(score):
    for evidence in score.evidence:
        assert evidence.produced_by == "code"
        assert evidence.unit
        assert evidence.lineage


def test_the_score_renders_as_the_frozen_contract(score, layer):
    contract = to_contract(score, layer)
    assert len(contract.components) == len(KEYS)
    assert contract.raw == pytest.approx(min(max(score.raw, 0.0), 1.0))
    assert contract.caps_applied == ()
    assert contract.calibration_sample_size == 145


def test_nothing_under_confidence_or_abstain_imports_the_model_layer():
    """Rules 1 and 6. A score is arithmetic; an abstention is a boolean."""
    offenders: list[str] = []
    for package in ("confidence", "abstain"):
        for path in sorted((Path("engine") / package).rglob("*.py")):
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


# ===========================================================================
# REGISTRY GAPS — pinned, not hidden
#
# The registry declares a value for each of s1..s6 on #2451. The formula
# reproduces 0.8876 from those six exactly, and that is asserted above.
# What the ENGINE computes from this warehouse differs on four of them,
# and the reasons are recorded here rather than tuned away (rule 10).
# ===========================================================================


def test_two_components_reproduce_the_registry_exactly(score):
    """s2 and s6 come out on the nose."""
    assert score.component("s2").value == pytest.approx(REGISTRY_COMPONENTS["s2"], abs=0.01)
    assert score.component("s6").value == pytest.approx(REGISTRY_COMPONENTS["s6"], abs=0.01)


def test_s1_saturates_because_this_dids_evidence_is_far_stronger_than_0_93(score):
    """GAP 1 — registry 0.93, engine ~1.00, and the registry is self-consistent.

    A logistic of |t| gives 0.93 at |t| ~ 2.6, which is p ~ 0.01 — exactly
    the "p < 0.01" the registry states for its own DiD. This warehouse's
    DiD lands at |t| ~ 10.5, p ~ 1e-15, because the generator's mechanism
    is clean and the effect is enormous. Evidence strength saturates.

    Nothing is wrong with either number. The registry's 0.93 describes a
    DiD that is just past significance; the engine is scoring one that is
    ten standard errors past it.
    """
    component = score.component("s1")
    assert component.facts["t_statistic"] > 8.0
    assert component.value > 0.99
    assert component.value > REGISTRY_COMPONENTS["s1"]


def test_s3_is_lower_than_the_registry_because_kappa_is_not_raw_agreement(score):
    """GAP 2 — registry 0.84, engine ~0.60.

    The two lanes agree on about 81% of the estate where chance alone
    would give 54%, so raw agreement would score near the registry's
    figure and Cohen's kappa scores 0.60. Kappa is the right statistic:
    two lanes that both find nothing in 80 of 140 stores agree about most
    of the estate without either of them having found anything.
    """
    component = score.component("s3")
    assert component.facts["kappa"] == pytest.approx(component.value, abs=1e-9)
    assert component.facts["observed_agreement"] > component.value
    assert component.value < REGISTRY_COMPONENTS["s3"]


def test_s4_is_higher_than_the_registry_because_nothing_is_stale(score):
    """GAP 3 — registry 0.90, engine 1.00.

    Both sources #2451's leading hypothesis requires are current to the
    warehouse clock and complete over the period. There is no defect for
    the score to find. The registry's 0.90 is not reproducible without
    inventing one.
    """
    component = score.component("s4")
    assert component.value == pytest.approx(1.0, abs=0.01)
    assert all(
        health["freshness"] == pytest.approx(1.0, abs=0.01)
        for health in component.facts["sources"].values()
    )


def test_s5_is_far_lower_than_the_registry_because_the_wms_feed_is_short(score):
    """GAP 4 — registry 0.72, engine ~0.24, and this one is about the data.

    The generator emits inventory snapshots for about thirteen weeks. The
    hypothesis rests on that feed, so measured over the sources it
    requires, historical depth is 13/52. The registry's 0.72 implies about
    37 weeks of the shortest source, which this warehouse does not have.

    Measured over the KPI's own series instead — 78 weeks of sales — s5
    would be 1.00 and overshoot. Neither reading lands on 0.72. The
    required-sources reading is kept because it is the conservative one
    and because it measures something s4 does not: s4 asks whether the
    feed arrived, s5 asks how far back it goes.
    """
    component = score.component("s5")
    assert component.facts["shortest_source"] == "wms"
    assert component.facts["history_weeks"] < 20
    assert component.value < REGISTRY_COMPONENTS["s5"]


def test_the_engines_raw_score_is_pinned_where_it_lands(score):
    """The four gaps partly cancel: 0.8356 against the registry's 0.8876.

    s1 and s4 come out above the registry, s3 and s5 below it, and the
    weighted sum lands about five points low. Published, that is 0.81
    rather than 0.84.

    Not tuned. Four components would have to be bent to reach 0.8876, and
    each of them is measuring something real about this warehouse.
    """
    assert score.raw == pytest.approx(0.836, abs=0.02)
    assert score.raw < REGISTRY_RAW
    assert score.calibrated == pytest.approx(0.81, abs=0.02)


def test_the_verdict_is_unaffected_by_the_gap(score, layer):
    """What the gap does NOT change, which is the part that matters.

    0.81 and 0.84 are on the same side of every threshold in the verdict
    table. The case is PARTIALLY EXPLAINED either way, for the same
    reason, and no trigger's answer moves.
    """
    held = (
        HeldResidual(
            "competitor_action", "Competitor promotion", verifiable=False,
            residual_inr=0.84e7, materiality_inr=0.50e7,
        ),
    )
    for confidence in (score.calibrated, REGISTRY_PUBLISHED):
        decision = decide(
            coverage=0.79, confidence=confidence, triggers=(), held=held, layer=layer
        )
        assert decision.value == PARTIALLY_EXPLAINED
        assert decision.reason_code == "live_unverifiable_above_materiality"
    assert decide(
        coverage=0.79, confidence=score.calibrated, triggers=(), held=(), layer=layer
    ).value == EXPLAINED
