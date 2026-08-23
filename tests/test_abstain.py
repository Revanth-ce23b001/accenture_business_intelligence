"""Abstention — eight named booleans and the three-condition decision table.

The acceptance criteria are:

  every trigger has an isolated test, fired and not fired
  the decision table is tested over a truth grid
  #2451 -> PARTIALLY EXPLAINED, reason live-unverifiable above materiality
  #2467 -> T3, T4 and T7 fire, INSUFFICIENT EVIDENCE, all three named

Rule 6: abstention is architectural, not a prompt instruction. Nothing in
`engine/abstain/` asks a model anything, and a static test in
`tests/test_confidence.py` says so.

The truth grid is the part worth reading. The decision table has three
inputs and four exits, and the exits are not symmetric: INSUFFICIENT
EVIDENCE is reached twice for two different reasons, and the grid checks
that the two are distinguishable.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime

import pytest

from engine.abstain import (
    BELOW_ALL_THRESHOLDS,
    COVERAGE_BELOW,
    EXPLAINED,
    IDS,
    INSUFFICIENT_EVIDENCE,
    LIVE_UNVERIFIABLE,
    PARTIALLY_EXPLAINED,
    TRIGGERS_FIRED,
    HeldResidual,
    Trigger,
    TriggerInput,
    decide,
    evaluate_triggers,
    fired,
    to_contract,
)
from engine.contracts import TriggerId
from security.policy import User

ANALYST = User(user_id="U008", persona="analyst", display_name="Meera Joshi")

CRORE = 1e7
MATERIALITY_INR = 0.50 * CRORE
_NOW = datetime(2025, 11, 30, 12, 0, tzinfo=UTC)


def clean(**overrides) -> TriggerInput:
    """A case where nothing is wrong. Each test breaks exactly one thing."""
    payload = dict(
        live_hypotheses=("stock_out",),
        all_hypotheses=("stock_out", "price_increase", "promo_lapse"),
        leading="stock_out",
        leading_share=0.79,
        missing_sources=(),
        hypothesis_confidence={"stock_out": 0.84},
        lane_kappa=0.60,
        independent_lanes=2,
        low_reliability_share=0.06,
        case_type_accuracy=0.89,
        case_type_cases=38,
        source_staleness_hours={"wms": 6.0, "erp": 0.0},
        unavailable_sources=(),
        forced=(),
    )
    payload.update(overrides)
    return TriggerInput(**payload)


def only(triggers, trigger_id: str) -> Trigger:
    return next(item for item in triggers if item.id == trigger_id)


# ===========================================================================
# The set itself
# ===========================================================================


def test_all_eight_triggers_are_evaluated_every_time(layer):
    """Including the ones that did not fire.

    A UI that showed only the fired triggers would leave a reader unable
    to tell "we checked and it was fine" from "we did not check".
    """
    triggers = evaluate_triggers(clean(), layer)
    assert tuple(item.id for item in triggers) == IDS
    assert len(triggers) == 8


def test_the_trigger_ids_match_the_contract_and_the_config(layer):
    assert set(IDS) == set(TriggerId.__args__)
    assert set(IDS) == set(layer.adjudication.triggers)


def test_every_trigger_explains_itself_whether_it_fired_or_not(layer):
    for triggers in (evaluate_triggers(clean(), layer),
                     evaluate_triggers(clean(missing_sources=("competitor_pricing",)), layer)):
        for item in triggers:
            assert item.detail.strip(), item.id
            assert item.name.strip(), item.id


def test_nothing_fires_on_a_clean_case(layer):
    assert fired(evaluate_triggers(clean(), layer)) == ()


# ===========================================================================
# T1 — no hypothesis passes both hard gates
# ===========================================================================


def test_t1_fires_when_nothing_survived_the_hard_gates(layer):
    triggers = evaluate_triggers(clean(live_hypotheses=()), layer)
    assert only(triggers, "T1").fired
    assert "none of 3 candidates" in only(triggers, "T1").detail


def test_t1_is_quiet_when_something_survived(layer):
    assert not only(evaluate_triggers(clean(), layer), "T1").fired


# ===========================================================================
# T2 — the best hypothesis explains too little
# ===========================================================================


def test_t2_fires_below_the_thirty_percent_floor(layer):
    triggers = evaluate_triggers(clean(leading_share=0.22), layer)
    assert only(triggers, "T2").fired
    assert only(triggers, "T2").facts["floor"] == 0.30


def test_t2_is_quiet_at_the_floor_exactly(layer):
    """At the floor is not below it. Boundaries are where rules get read
    wrong, so the boundary is asserted rather than assumed."""
    assert not only(evaluate_triggers(clean(leading_share=0.30), layer), "T2").fired


def test_t2_is_quiet_on_2451s_seventy_nine_percent(layer):
    assert not only(evaluate_triggers(clean(), layer), "T2").fired


# ===========================================================================
# T3 — a source the organisation does not hold
# ===========================================================================


def test_t3_fires_and_names_the_feed(layer):
    triggers = evaluate_triggers(
        clean(missing_sources=("competitor_pricing", "competitor_footfall")), layer
    )
    trigger = only(triggers, "T3")
    assert trigger.fired
    assert "competitor_pricing" in trigger.detail
    assert "competitor_footfall" in trigger.detail


def test_t3_is_quiet_when_every_required_source_is_held(layer):
    assert not only(evaluate_triggers(clean(), layer), "T3").fired


def test_t3_can_be_forced_by_the_missing_source_cap(layer):
    """The cap and the trigger can never disagree, because the cap forces it."""
    triggers = evaluate_triggers(clean(forced=("T3",)), layer)
    trigger = only(triggers, "T3")
    assert trigger.fired
    assert trigger.forced
    assert "forced by a confidence cap" in trigger.detail


# ===========================================================================
# T4 — two hypotheses the data cannot separate
# ===========================================================================


def test_t4_fires_when_two_hypotheses_are_within_the_resolution_floor(layer):
    triggers = evaluate_triggers(
        clean(hypothesis_confidence={"competitor_action": 0.62, "footfall_decline": 0.58}),
        layer,
    )
    trigger = only(triggers, "T4")
    assert trigger.fired
    assert trigger.facts["separation"] == pytest.approx(0.04, abs=1e-9)
    assert "choosing, not concluding" in trigger.detail


def test_t4_is_quiet_when_one_hypothesis_clearly_leads(layer):
    triggers = evaluate_triggers(
        clean(hypothesis_confidence={"stock_out": 0.84, "price_increase": 0.31}), layer
    )
    assert not only(triggers, "T4").fired


def test_t4_needs_two_hypotheses_before_it_can_fire(layer):
    """One hypothesis cannot be indistinguishable from anything."""
    triggers = evaluate_triggers(clean(hypothesis_confidence={"stock_out": 0.84}), layer)
    assert not only(triggers, "T4").fired
    assert "no pair" in only(triggers, "T4").detail


# ===========================================================================
# T5 — the lanes contradict each other
# ===========================================================================


def test_t5_fires_when_the_lanes_agree_worse_than_chance(layer):
    triggers = evaluate_triggers(clean(lane_kappa=-0.22), layer)
    trigger = only(triggers, "T5")
    assert trigger.fired
    assert "different stores" in trigger.detail


def test_t5_is_quiet_when_the_lanes_agree(layer):
    assert not only(evaluate_triggers(clean(lane_kappa=0.60), layer), "T5").fired


def test_t5_needs_two_independent_lanes_before_it_can_fire(layer):
    """One lane cannot contradict itself."""
    triggers = evaluate_triggers(clean(independent_lanes=1, lane_kappa=-0.5), layer)
    assert not only(triggers, "T5").fired


# ===========================================================================
# T6 — the case rests on text
# ===========================================================================


def test_t6_fires_when_most_of_the_evidence_weight_is_below_the_floor(layer):
    triggers = evaluate_triggers(clean(low_reliability_share=0.72), layer)
    trigger = only(triggers, "T6")
    assert trigger.fired
    assert "Text alone does not reach a verdict" in trigger.detail


def test_t6_is_quiet_on_evidence_backed_by_a_matched_control(layer):
    assert not only(evaluate_triggers(clean(), layer), "T6").fired


# ===========================================================================
# T7 — our own record is below the publication floor
# ===========================================================================


def test_t7_fires_on_a_case_type_we_get_wrong(layer):
    """Case #2467's competitor-attribution record: 58% over 12 cases."""
    triggers = evaluate_triggers(
        clean(case_type_accuracy=0.58, case_type_cases=12), layer
    )
    trigger = only(triggers, "T7")
    assert trigger.fired
    assert "58%" in trigger.detail
    assert "12 closed cases" in trigger.detail


def test_t7_is_quiet_on_a_case_type_we_get_right(layer):
    assert not only(evaluate_triggers(clean(), layer), "T7").fired


def test_t7_does_not_fire_on_too_few_cases(layer):
    """No record is not a bad record, and it is acted on differently."""
    triggers = evaluate_triggers(
        clean(case_type_accuracy=0.20, case_type_cases=3), layer
    )
    assert not only(triggers, "T7").fired
    assert "No record is not a bad record" in only(triggers, "T7").detail


def test_t7_does_not_fire_when_there_is_no_record_at_all(layer):
    triggers = evaluate_triggers(
        clean(case_type_accuracy=None, case_type_cases=0), layer
    )
    assert not only(triggers, "T7").fired


# ===========================================================================
# T8 — a required source is stale
# ===========================================================================


def test_t8_fires_on_a_stale_required_source(layer):
    triggers = evaluate_triggers(
        clean(source_staleness_hours={"wms": 74.0, "erp": 0.0}), layer
    )
    trigger = only(triggers, "T8")
    assert trigger.fired
    assert "wms 74h" in trigger.detail


def test_t8_fires_on_an_unavailable_source(layer):
    triggers = evaluate_triggers(clean(unavailable_sources=("footfall",)), layer)
    assert only(triggers, "T8").fired


def test_t8_ignores_a_stale_feed_the_case_does_not_use(layer):
    """Case #2467's marketing feed is 74 hours stale and T8 does not fire.

    The leading hypothesis is a competitor promotion; it does not rest on
    the marketing feed. A stale feed nobody is relying on is a data-quality
    ticket, not a reason to withhold an answer that does not use it.
    """
    triggers = evaluate_triggers(
        clean(source_staleness_hours={"competitor_news": 4.0}), layer
    )
    assert not only(triggers, "T8").fired


def test_t8_is_quiet_inside_the_staleness_limit(layer):
    assert not only(evaluate_triggers(clean(), layer), "T8").fired


# ===========================================================================
# THE DECISION TABLE — over a truth grid
# ===========================================================================


def _held(verifiable: bool, inr: float) -> tuple[HeldResidual, ...]:
    return (
        HeldResidual(
            "competitor_action", "Competitor promotion or price action",
            verifiable=verifiable, residual_inr=inr, materiality_inr=MATERIALITY_INR,
            missing_sources=("competitor_pricing",),
        ),
    )


def _trigger(fires: bool) -> tuple[Trigger, ...]:
    return (Trigger("T3", "a source we do not hold", fires, "grid"),)


GRID_CASES = list(
    itertools.product(
        (False, True),          # a trigger fired
        (0.85, 0.50, 0.10),     # coverage: over 0.70, over 0.30, under both
        (0.84, 0.65, 0.40),     # confidence: over 0.70, over 0.60, under both
        (True, False),          # the competing hypothesis is verifiable
        (0.86 * CRORE, 0.10 * CRORE),  # what it holds, against a 0.50 Cr limit
    )
)


def _expected(trigger, coverage, confidence, verifiable, held_inr):
    """The table from CLAUDE.md, written out again independently.

    Deliberately a second implementation rather than a call into the one
    under test: a grid checked against the code it is testing checks
    nothing.
    """
    if trigger:
        return INSUFFICIENT_EVIDENCE, TRIGGERS_FIRED
    blocks = (not verifiable) and held_inr > MATERIALITY_INR
    if coverage >= 0.70 and confidence >= 0.70 and not blocks:
        return EXPLAINED, None
    if coverage >= 0.30 and confidence >= 0.60:
        return PARTIALLY_EXPLAINED, (LIVE_UNVERIFIABLE if blocks else COVERAGE_BELOW)
    return INSUFFICIENT_EVIDENCE, BELOW_ALL_THRESHOLDS


@pytest.mark.parametrize(
    ("trigger", "coverage", "confidence", "verifiable", "held_inr"), GRID_CASES
)
def test_the_decision_table_over_a_truth_grid(
    trigger, coverage, confidence, verifiable, held_inr, layer
):
    decision = decide(
        coverage=coverage, confidence=confidence,
        triggers=_trigger(trigger), held=_held(verifiable, held_inr), layer=layer,
    )
    value, reason = _expected(trigger, coverage, confidence, verifiable, held_inr)
    assert decision.value == value
    assert decision.reason_code == reason


def test_the_grid_reaches_every_exit(layer):
    """A grid that never reaches an exit has not tested it."""
    reached = set()
    for case in GRID_CASES:
        decision = decide(
            coverage=case[1], confidence=case[2],
            triggers=_trigger(case[0]), held=_held(case[3], case[4]), layer=layer,
        )
        reached.add((decision.value, decision.reason_code))
    assert (EXPLAINED, None) in reached
    assert (PARTIALLY_EXPLAINED, LIVE_UNVERIFIABLE) in reached
    assert (PARTIALLY_EXPLAINED, COVERAGE_BELOW) in reached
    assert (INSUFFICIENT_EVIDENCE, TRIGGERS_FIRED) in reached
    assert (INSUFFICIENT_EVIDENCE, BELOW_ALL_THRESHOLDS) in reached


def test_a_trigger_short_circuits_before_the_thresholds_are_read(layer):
    """Even a perfect case abstains if a trigger fired."""
    decision = decide(
        coverage=1.0, confidence=1.0, triggers=_trigger(True),
        held=(), layer=layer,
    )
    assert decision.value == INSUFFICIENT_EVIDENCE
    assert decision.reason_code == TRIGGERS_FIRED
    assert "never reached" in decision.trace[0]


def test_the_two_insufficient_evidence_exits_are_different_statements(layer):
    """"We could not run this" and "we ran it and it fell short" are not
    the same, and the resolution panel acts on them differently."""
    broken = decide(
        coverage=0.79, confidence=0.84, triggers=_trigger(True), held=(), layer=layer
    )
    fell_short = decide(
        coverage=0.10, confidence=0.20, triggers=_trigger(False), held=(), layer=layer
    )
    assert broken.reason_code != fell_short.reason_code
    assert broken.triggers_fired == ("T3",)
    assert fell_short.triggers_fired == ()


def test_every_trigger_that_fired_is_named_not_just_the_first(layer):
    triggers = (
        Trigger("T3", "a source we do not hold", True, "x"),
        Trigger("T4", "indistinguishable", True, "y"),
        Trigger("T7", "below the floor", True, "z"),
    )
    decision = decide(
        coverage=0.5, confidence=0.5, triggers=triggers, held=(), layer=layer
    )
    assert decision.triggers_fired == ("T3", "T4", "T7")
    for name in ("T3", "T4", "T7"):
        assert name in decision.reason_text


# ===========================================================================
# ACCEPT — #2451 lands PARTIALLY EXPLAINED for the right reason
# ===========================================================================


def test_2451_is_partially_explained_on_the_third_condition(layer):
    """Coverage and confidence both clear the EXPLAINED bar.

    What stops it is the third condition, which is the one most systems do
    not have: a competing explanation nobody can check, holding more money
    than the business said it cares about.
    """
    decision = decide(
        coverage=0.79, confidence=0.84, triggers=(),
        held=_held(verifiable=False, inr=0.86 * CRORE), layer=layer,
    )
    assert decision.value == PARTIALLY_EXPLAINED
    assert decision.reason_code == LIVE_UNVERIFIABLE
    assert "live-unverifiable hypothesis above materiality" in decision.reason_text
    assert "0.86 Cr" in decision.reason_text
    assert "0.50 Cr" in decision.reason_text


def test_2451_would_be_explained_if_the_competitor_could_be_checked(layer):
    """The same case, with the competitor feed bought. One condition."""
    decision = decide(
        coverage=0.79, confidence=0.84, triggers=(),
        held=_held(verifiable=True, inr=0.86 * CRORE), layer=layer,
    )
    assert decision.value == EXPLAINED


def test_2451_would_be_explained_if_the_holding_were_immaterial(layer):
    """And with the holding below the limit. The other half of the condition."""
    decision = decide(
        coverage=0.79, confidence=0.84, triggers=(),
        held=_held(verifiable=False, inr=0.10 * CRORE), layer=layer,
    )
    assert decision.value == EXPLAINED


def test_the_reason_string_renders_on_the_chip(layer):
    decision = decide(
        coverage=0.79, confidence=0.84, triggers=(),
        held=_held(verifiable=False, inr=0.86 * CRORE), layer=layer,
    )
    contract = to_contract(decision, "2451", _NOW)
    assert contract.value == PARTIALLY_EXPLAINED
    assert contract.reason_text
    assert contract.reason_code == LIVE_UNVERIFIABLE


# ===========================================================================
# ACCEPT — #2467 abstains on T3, T4 and T7
# ===========================================================================


@pytest.fixture(scope="module")
def case_2467(warehouse, layer):
    """Case #2467's trigger input, built from the layer and the warehouse.

    Two of the three triggers are computed from real state and one is
    supplied:

      T3  from `causal_graph.yaml` — competitor_action requires
          competitor_pricing and competitor_footfall, and both are
          declared `held: false`.
      T7  from `calibration_ledger` — 12 closed competitor-attribution
          cases, 58% right, against a 70% publication floor.
      T4  the two hypothesis confidences are supplied. Producing them
          would mean adjudicating East conversion end to end, and the
          precedence effect series is currently revenue-shaped for every
          KPI. That is a wiring job, and it is P11's.
    """
    from engine.confidence import fit_calibration

    template = layer.causal_graph.hypotheses["competitor_action"]
    calibration = fit_calibration(warehouse, ANALYST, layer=layer)
    accuracy, cases = calibration.accuracy_for("competitor_attribution")
    return TriggerInput(
        live_hypotheses=("competitor_action", "footfall_decline"),
        all_hypotheses=("competitor_action", "footfall_decline", "mix_shift"),
        leading="competitor_action",
        leading_share=0.61,
        missing_sources=tuple(template.missing_sources()),
        hypothesis_confidence={"competitor_action": 0.44, "footfall_decline": 0.41},
        lane_kappa=0.18,
        independent_lanes=2,
        low_reliability_share=0.38,
        case_type_accuracy=accuracy,
        case_type_cases=cases,
        source_staleness_hours={},
        unavailable_sources=(),
    )


def test_2467_fires_exactly_t3_t4_and_t7(case_2467, layer):
    triggers = evaluate_triggers(case_2467, layer)
    assert fired(triggers) == ("T3", "T4", "T7")


def test_2467s_t3_names_the_two_feeds_the_organisation_does_not_hold(case_2467, layer):
    trigger = only(evaluate_triggers(case_2467, layer), "T3")
    assert trigger.fired
    assert set(trigger.facts["missing_sources"]) == {
        "competitor_pricing", "competitor_footfall"
    }


def test_2467s_t7_reads_the_seeded_ledger_not_a_constant(case_2467, layer):
    trigger = only(evaluate_triggers(case_2467, layer), "T7")
    assert trigger.fired
    assert trigger.facts["cases"] == 12
    assert trigger.facts["accuracy"] == pytest.approx(0.58, abs=0.01)
    assert trigger.facts["floor"] == 0.70


def test_2467_is_insufficient_evidence_with_all_three_named(case_2467, layer):
    triggers = evaluate_triggers(case_2467, layer)
    decision = decide(
        coverage=0.61, confidence=0.44, triggers=triggers, held=(), layer=layer
    )
    assert decision.value == INSUFFICIENT_EVIDENCE
    assert decision.reason_code == TRIGGERS_FIRED
    assert decision.triggers_fired == ("T3", "T4", "T7")
    for name in ("T3", "T4", "T7"):
        assert name in decision.reason_text
    assert decision.abstained


def test_2467_abstains_on_evidence_and_not_on_materiality(case_2467, layer):
    """Its residual clears the limit comfortably. That is not what stops it.

    CLAUDE.md: "This case abstains on EVIDENCE, not on materiality." The
    trigger short-circuit is what makes that true — the coverage and
    confidence thresholds are never reached.
    """
    triggers = evaluate_triggers(case_2467, layer)
    decision = decide(
        coverage=0.95, confidence=0.99, triggers=triggers, held=(), layer=layer
    )
    assert decision.value == INSUFFICIENT_EVIDENCE
