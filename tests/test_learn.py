"""P16 — feedback and the calibration loop.

The three accept criteria, each with a section:

  1. submitting feedback measurably moves a reliability bin, and the
     change is returned by /api/calibration on the next call
  2. rejecting a driver updates its prior in the runtime overlay
  3. ECE recomputes

Plus the four verdict options, the two outcome horizons, and the bounds
that stop a loop wired to a button from being moved by whoever clicks
most.

THE WAREHOUSE THESE TESTS WRITE TO IS THEIR OWN. `calibration_ledger`
arrives seeded with 213 closed cases and the fit depends on every one of
them; a test that left a row behind would move the map for every test
after it. Each section that writes clears what it wrote.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from engine.learn import curves, feedback, loop, outcomes, priors
from engine.warehouse.migrate import MigrationError, pending
from semantic_layer.overlay import read_priors_overlay

NOW = datetime(2025, 11, 30, 12, 0, tzinfo=UTC)

ANALYST = {"user_id": "U008", "persona": "analyst"}

#: A KPI and a driver that exist, for the prior tests.
KPI = "net_revenue"
DRIVER = "stock_out"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wh(warehouse):
    """The session warehouse with P16's tables present."""
    from engine.warehouse.load import create_schema

    create_schema(warehouse)
    return warehouse


@pytest.fixture
def clean(wh):
    """A warehouse with no feedback, no outcomes and no learned priors.

    The seeded calibration ledger is left alone: it is the organisation's
    track record and the fit depends on it. Entries these tests write are
    keyed by case ids that do not appear in the seed, and are removed.
    """
    wh.execute("DELETE FROM feedback_event")
    wh.execute("DELETE FROM case_outcome")
    wh.execute("DELETE FROM hypothesis_prior")
    wh.execute("DELETE FROM recovery_realisation")
    wh.execute("DELETE FROM calibration_ledger WHERE case_id LIKE 'TEST-%'")
    yield wh
    wh.execute("DELETE FROM calibration_ledger WHERE case_id LIKE 'TEST-%'")
    wh.execute("DELETE FROM hypothesis_prior")
    wh.execute("DELETE FROM recovery_realisation")


def submit(connection, layer, **kwargs):
    """Record one feedback event and apply the loop to it."""
    payload = {
        "case_id": "TEST-1",
        "user_id": ANALYST["user_id"],
        "persona": ANALYST["persona"],
        "occurred_at": NOW,
        "layer": layer,
    }
    payload.update(kwargs)
    entry = feedback.record(connection, **payload)
    return entry, loop.apply(connection, entry, layer=layer, write_overlay=False)


def bins(connection, user, layer):
    """The band table, keyed by label, as /api/calibration would return it."""
    from engine.confidence.calibration import fit_calibration

    fitted = fit_calibration(connection, user, layer=layer)
    return {band.label: band for band in fitted.bands}, fitted


@pytest.fixture(scope="module")
def analyst_user():
    from security.policy import User

    return User(user_id="U008", persona="analyst", display_name="Meera Joshi")


# ===========================================================================
# The vocabulary — four options, three levels
# ===========================================================================


def test_the_four_verdict_options_are_the_four(layer):
    assert set(layer.learning.feedback.verdict_actions) == {
        "accept", "modify", "reject", "request_investigation"
    }


def test_reject_demands_a_reason(layer, clean):
    with pytest.raises(feedback.FeedbackError, match="requires a reason"):
        feedback.validate("verdict", "reject", target_id=None, reason_code=None, layer=layer)


def test_accept_does_not_demand_a_reason(layer):
    feedback.validate("verdict", "accept", target_id=None, reason_code=None, layer=layer)


def test_an_undeclared_action_is_refused(layer):
    with pytest.raises(feedback.FeedbackError, match="not a verdict action"):
        feedback.validate("verdict", "shrug", target_id=None, reason_code=None, layer=layer)


def test_an_undeclared_reason_is_refused(layer):
    """Free text is a comment. A comment cannot be counted."""
    with pytest.raises(feedback.FeedbackError, match="not a declared reason code"):
        feedback.validate(
            "verdict", "reject", target_id=None, reason_code="just because", layer=layer
        )


def test_driver_feedback_names_its_driver(layer):
    with pytest.raises(feedback.FeedbackError, match="names the driver"):
        feedback.validate("driver", "reject", target_id=None, reason_code=None, layer=layer)


def test_verdict_feedback_takes_no_target(layer):
    with pytest.raises(feedback.FeedbackError, match="takes no target"):
        feedback.validate("verdict", "accept", target_id=DRIVER, reason_code=None, layer=layer)


def test_all_three_levels_round_trip(clean, layer):
    submit(clean, layer, level="verdict", action="accept",
           case_type="availability_attribution", confidence=0.84, confidence_raw=0.8876)
    submit(clean, layer, level="driver", action="reject", target_id=DRIVER, kpi=KPI)
    submit(clean, layer, level="action", action="accept", target_id="availability_recovery")

    stored = feedback.for_case(clean, "TEST-1")
    assert {item.level for item in stored} == {"verdict", "driver", "action"}


# ===========================================================================
# ACCEPT 1 — feedback moves a reliability bin
# ===========================================================================


def test_accepting_a_case_moves_its_bin(clean, layer, analyst_user):
    """ACCEPT: submitting feedback measurably moves a reliability bin.

    The case scores 0.84 raw, so it lands in the 80-89% band. Accepting it
    adds one case to that band and moves its accuracy towards 100%.
    """
    before, _ = bins(clean, analyst_user, layer)
    band = "80%-90%"
    cases_before = before[band].cases
    accuracy_before = before[band].accuracy

    submit(
        clean, layer, case_id="TEST-BIN", level="verdict", action="accept",
        case_type="availability_attribution", confidence=0.84, confidence_raw=0.84,
    )

    after, _ = bins(clean, analyst_user, layer)
    assert after[band].cases == cases_before + 1, "the band did not gain the case"
    assert after[band].accuracy != accuracy_before, "the band's accuracy did not move"
    assert after[band].accuracy > accuracy_before, "an accepted case is a hit"


def test_rejecting_a_case_moves_the_bin_the_other_way(clean, layer, analyst_user):
    before, _ = bins(clean, analyst_user, layer)
    band = "80%-90%"
    accuracy_before = before[band].accuracy

    submit(
        clean, layer, case_id="TEST-BIN", level="verdict", action="reject",
        reason_code="wrong_driver", case_type="availability_attribution",
        confidence=0.84, confidence_raw=0.84,
    )

    after, _ = bins(clean, analyst_user, layer)
    assert after[band].accuracy < accuracy_before, "a rejected case is a miss"


def test_modify_scores_as_a_miss(clean, layer):
    """The one thing a calibration map must not learn is that nearly right
    is right."""
    _, effect = submit(
        clean, layer, case_id="TEST-MOD", level="verdict", action="modify",
        reason_code="wrong_magnitude", case_type="availability_attribution",
        confidence=0.84, confidence_raw=0.84,
    )
    assert effect.moved_calibration
    (entry,) = loop.ledger_for_case(clean, "TEST-MOD")
    assert entry["was_correct"] is False


def test_request_investigation_writes_no_entry(clean, layer):
    """Not a judgement on correctness. Scoring it either way would record
    an opinion the reader did not give."""
    _, effect = submit(
        clean, layer, case_id="TEST-REQ", level="verdict",
        action="request_investigation", case_type="availability_attribution",
        confidence=0.84, confidence_raw=0.84,
    )
    assert not effect.moved_calibration
    assert loop.ledger_for_case(clean, "TEST-REQ") == ()
    assert any("request for work" in note for note in effect.notes)


def test_one_case_never_holds_two_ledger_entries(clean, layer):
    """A reader who changes their mind does not get counted twice."""
    for action in ("accept", "reject"):
        submit(
            clean, layer, case_id="TEST-TWICE", level="verdict", action=action,
            reason_code="wrong_driver" if action == "reject" else None,
            case_type="availability_attribution", confidence=0.84, confidence_raw=0.84,
        )
    entries = loop.ledger_for_case(clean, "TEST-TWICE")
    assert len(entries) == 1
    assert entries[0]["was_correct"] is False, "the later opinion stands"


def test_feedback_with_no_confidence_stamped_calibrates_nothing(clean, layer):
    _, effect = submit(
        clean, layer, case_id="TEST-NOCONF", level="verdict", action="accept",
        case_type="availability_attribution",
    )
    assert not effect.moved_calibration
    assert any("nothing for the map" in note for note in effect.notes)


# ===========================================================================
# ACCEPT 3 — ECE recomputes
# ===========================================================================


def test_the_expected_calibration_error_recomputes(clean, layer, analyst_user):
    """ACCEPT: ECE recomputes.

    Twelve rejections of high-confidence cases make the map measurably
    worse calibrated at the raw end. The point is not the direction — it is
    that the number is a function of the ledger and moves when the ledger
    does, rather than being a figure printed once.
    """
    _, before = bins(clean, analyst_user, layer)
    for index in range(12):
        submit(
            clean, layer, case_id=f"TEST-ECE-{index}", level="verdict",
            action="reject", reason_code="wrong_driver",
            case_type="availability_attribution", confidence=0.95, confidence_raw=0.95,
        )
    _, after = bins(clean, analyst_user, layer)

    assert after.scored_cases == before.scored_cases + 12
    assert after.ece_raw != before.ece_raw, "ECE did not move with the ledger"
    assert after.ece_raw > before.ece_raw, (
        "twelve confident misses should make the raw scores look worse calibrated"
    )


def test_the_map_still_corrects_after_feedback(clean, layer, analyst_user):
    """The isotonic map is refitted, not merely re-read."""
    from engine.confidence.calibration import fit_calibration

    before = fit_calibration(clean, analyst_user, layer=layer)
    for index in range(20):
        submit(
            clean, layer, case_id=f"TEST-MAP-{index}", level="verdict",
            action="reject", reason_code="wrong_driver",
            case_type="availability_attribution", confidence=0.9, confidence_raw=0.9,
        )
    after = fit_calibration(clean, analyst_user, layer=layer)
    assert after.map(0.9) < before.map(0.9), (
        "twenty misses at 0.9 should cool what 0.9 publishes as"
    )


# ===========================================================================
# ACCEPT 2 — rejecting a driver updates the runtime overlay
# ===========================================================================


def test_rejecting_a_driver_moves_its_prior(clean, layer):
    """ACCEPT: rejecting a driver updates its prior."""
    declared = layer.causal_graph.hypotheses[DRIVER].prior
    before = priors.effective_prior(clean, kpi=KPI, hypothesis=DRIVER, layer=layer)
    assert before.effective == declared, "an unjudged driver sits on its declared prior"

    _, effect = submit(
        clean, layer, level="driver", action="reject", target_id=DRIVER, kpi=KPI
    )

    assert effect.moved_prior
    assert effect.prior_after < effect.prior_before, "a rejection lowers the prior"
    assert effect.prior_hypothesis == DRIVER and effect.prior_kpi == KPI


def test_confirming_a_driver_raises_its_prior(clean, layer):
    _, effect = submit(
        clean, layer, level="driver", action="confirm", target_id=DRIVER, kpi=KPI
    )
    assert effect.prior_after > effect.prior_before


def test_the_overlay_file_records_the_move(clean, layer, tmp_path):
    """ACCEPT: the change lands in causal_graph.yaml's runtime overlay.

    Written to a temporary path here so the suite does not leave a
    generated file in the working tree — the writer is the same one the
    API calls.
    """
    overlay = tmp_path / "priors.yaml"
    priors.observe(clean, kpi=KPI, hypothesis=DRIVER, confirmed=False, at=NOW)
    written = priors.write_overlay(clean, path=overlay, layer=layer, generated_at=NOW)

    assert written == overlay
    payload = read_priors_overlay(overlay)
    entry = payload["priors"][KPI][DRIVER]
    assert entry["rejected"] == 1
    assert entry["effective"] < entry["declared"]
    assert entry["shift"] < 0
    assert "do not edit" in overlay.read_text(encoding="utf-8")


def test_the_overlay_names_the_rule_it_applied(clean, layer, tmp_path):
    overlay = tmp_path / "priors.yaml"
    priors.observe(clean, kpi=KPI, hypothesis=DRIVER, confirmed=False, at=NOW)
    priors.write_overlay(clean, path=overlay, layer=layer, generated_at=NOW)
    payload = read_priors_overlay(overlay)
    assert payload["rule"]["strength"] == layer.learning.priors.strength
    assert payload["source"] == "hypothesis_prior"


def test_the_declared_prior_is_never_written_to(clean, layer):
    """`causal_graph.yaml` is the contract. Nothing in the loop edits it."""
    from semantic_layer.schema import PACKAGE_ROOT

    graph = (PACKAGE_ROOT / "causal_graph.yaml").read_text(encoding="utf-8")
    submit(clean, layer, level="driver", action="reject", target_id=DRIVER, kpi=KPI)
    assert (PACKAGE_ROOT / "causal_graph.yaml").read_text(encoding="utf-8") == graph


def test_one_click_moves_the_prior_a_little_and_not_a_lot(clean, layer):
    """The correct weight for one person's opinion.

    A loop whose first lesson is invisible is a loop nobody believes in; a
    loop that a single click can swing is a loop nobody should.
    """
    declared = layer.causal_graph.hypotheses[DRIVER].prior
    submit(clean, layer, level="driver", action="reject", target_id=DRIVER, kpi=KPI)
    after = priors.effective_prior(clean, kpi=KPI, hypothesis=DRIVER, layer=layer)

    assert after.effective < declared
    assert abs(after.shift) < declared / 10.0, "one click moved it more than 10%"


def test_the_shift_cap_binds(clean, layer):
    """A hypothesis cannot be argued out of existence by volume of clicks."""
    cap = layer.learning.priors.max_absolute_shift
    for _ in range(500):
        priors.observe(clean, kpi=KPI, hypothesis=DRIVER, confirmed=False, at=NOW)
    after = priors.effective_prior(clean, kpi=KPI, hypothesis=DRIVER, layer=layer)
    assert abs(after.shift) <= cap + 1e-9
    assert after.effective >= layer.learning.priors.floor


def test_the_prior_is_learned_per_kpi(clean, layer):
    """Stock-outs explain availability far more often than they explain
    average selling price. One pooled count would blur the two."""
    submit(clean, layer, level="driver", action="reject", target_id=DRIVER, kpi=KPI)
    other = priors.effective_prior(
        clean, kpi="on_shelf_availability", hypothesis=DRIVER, layer=layer
    )
    assert other.observations == 0
    assert other.effective == layer.causal_graph.hypotheses[DRIVER].prior


def test_a_driver_the_graph_does_not_declare_is_refused(clean, layer):
    entry = feedback.record(
        clean, case_id="TEST-1", user_id="U008", persona="analyst",
        level="driver", action="reject", target_id="gremlins", kpi=KPI, layer=layer,
    )
    with pytest.raises(loop.LoopError, match="not a hypothesis"):
        loop.apply(clean, entry, layer=layer, write_overlay=False)


def test_driver_feedback_without_a_kpi_is_refused(clean, layer):
    """A prior is learned per hypothesis PER KPI. There is nothing to
    attach a KPI-less observation to."""
    entry = feedback.record(
        clean, case_id="TEST-1", user_id="U008", persona="analyst",
        level="driver", action="reject", target_id=DRIVER, layer=layer,
    )
    with pytest.raises(loop.LoopError, match="carries no KPI"):
        loop.apply(clean, entry, layer=layer, write_overlay=False)


# ===========================================================================
# Screening reads the learned prior
# ===========================================================================


def test_the_learned_prior_reaches_the_screening(clean, layer):
    """The loop is only real if it changes what the engine does next.

    `Candidate.prior` is what the screen sorts on, and it comes from the
    posterior once anyone has judged the driver.
    """
    from engine.gather.hypotheses import _learned_priors

    declared = layer.causal_graph.hypotheses[DRIVER].prior
    for _ in range(5):
        priors.observe(clean, kpi=KPI, hypothesis=DRIVER, confirmed=False, at=NOW)

    resolved = _learned_priors(clean, layer, KPI)
    assert resolved[DRIVER].effective < declared


def test_screening_falls_back_to_the_contract_when_the_table_is_missing(layer):
    """Fail open, to the declared prior. A warehouse without the table
    screens exactly as it did before the loop existed."""
    from engine.db import IN_MEMORY, connect
    from engine.gather.hypotheses import _learned_priors

    assert _learned_priors(connect(IN_MEMORY), layer, KPI) == {}


# ===========================================================================
# Outcomes — D+14 and D+56
# ===========================================================================


def test_the_two_horizons_are_the_two(layer):
    assert [h.days for h in layer.learning.outcomes.horizons] == [14, 56]


def test_a_horizon_comes_due_when_it_comes_due(layer):
    opened = NOW - timedelta(days=20)
    assert outcomes.due(opened, now=NOW, layer=layer) == (14,)
    assert outcomes.due(NOW - timedelta(days=60), now=NOW, layer=layer) == (14, 56)
    assert outcomes.due(NOW, now=NOW, layer=layer) == ()


def test_d14_records_whether_the_cause_held_up(clean, layer):
    result = outcomes.record_cause_check(
        clean, case_id="TEST-OUT", cause_confirmed=True, at=NOW, layer=layer
    )
    assert result.horizon_days == 14
    assert result.cause_confirmed is True
    assert result.was_correct is True
    assert result.outcome == outcomes.CONFIRMED


def test_d56_records_whether_the_money_came_back(clean, layer):
    result = outcomes.record_recovery_check(
        clean, case_id="TEST-OUT", realised_inr=2.4e7,
        expected_low_inr=2.3e7, expected_high_inr=3.1e7,
        horizon_weeks=8, at=NOW, layer=layer,
    )
    assert result.horizon_days == 56
    assert result.recovered is True
    assert result.outcome == outcomes.RECOVERED


def test_the_bar_is_the_low_end_of_the_band(clean, layer):
    """Missing the optimistic end of a range published as a range is not a
    failed recovery."""
    missed = outcomes.record_recovery_check(
        clean, case_id="TEST-LOW", realised_inr=2.0e7,
        expected_low_inr=2.3e7, expected_high_inr=3.1e7, at=NOW, layer=layer,
    )
    assert missed.recovered is False

    made = outcomes.record_recovery_check(
        clean, case_id="TEST-MID", realised_inr=2.5e7,
        expected_low_inr=2.3e7, expected_high_inr=3.1e7, at=NOW, layer=layer,
    )
    assert made.recovered is True


def test_both_horizons_coexist_for_one_case(clean, layer):
    outcomes.record_cause_check(clean, case_id="TEST-BOTH", cause_confirmed=True,
                                at=NOW, layer=layer)
    outcomes.record_recovery_check(
        clean, case_id="TEST-BOTH", realised_inr=1.0e7,
        expected_low_inr=2.3e7, expected_high_inr=3.1e7, at=NOW, layer=layer,
    )
    recorded = outcomes.for_case(clean, "TEST-BOTH")
    assert [item.horizon_days for item in recorded] == [14, 56]
    assert recorded[0].cause_confirmed is True
    assert recorded[1].recovered is False, "the cause held and the money did not come back"


def test_an_unresolved_case_calibrates_nothing(clean, layer):
    """Silence is not evidence that we were right."""
    result = outcomes.record_unresolved(
        clean, case_id="TEST-QUIET", horizon_days=56, at=NOW, layer=layer
    )
    assert result.was_correct is None
    assert outcomes.settled((result,)) is None


def test_the_later_horizon_is_the_one_that_settles(clean, layer):
    early = outcomes.record_cause_check(clean, case_id="TEST-SETTLE",
                                        cause_confirmed=True, at=NOW, layer=layer)
    late = outcomes.record_recovery_check(
        clean, case_id="TEST-SETTLE", realised_inr=1.0,
        expected_low_inr=2.3e7, expected_high_inr=3.1e7, at=NOW, layer=layer,
    )
    assert outcomes.settled((early, late)) == late


def test_an_outcome_supersedes_the_feedback_entry(clean, layer):
    """What happened outranks what we were told."""
    submit(
        clean, layer, case_id="TEST-SUP", level="verdict", action="accept",
        case_type="availability_attribution", confidence=0.84, confidence_raw=0.84,
    )
    (told,) = loop.ledger_for_case(clean, "TEST-SUP")
    assert told["was_correct"] is True
    assert told["notes"].startswith("feedback")

    happened = outcomes.record_recovery_check(
        clean, case_id="TEST-SUP", realised_inr=1.0,
        expected_low_inr=2.3e7, expected_high_inr=3.1e7, at=NOW, layer=layer,
    )
    loop.apply_outcome(
        clean, happened, case_type="availability_attribution",
        confidence_raw=0.84, confidence_published=0.84, layer=layer,
    )

    entries = loop.ledger_for_case(clean, "TEST-SUP")
    assert len(entries) == 1, "the case is counted once, not twice"
    assert entries[0]["was_correct"] is False
    assert entries[0]["notes"].startswith("outcome")


# ===========================================================================
# Recovery curves — realised against expected
# ===========================================================================


def test_a_realisation_is_a_share_of_what_was_attributed(clean, layer):
    result = curves.record_realisation(
        clean, case_id="TEST-R", playbook="availability_recovery",
        curve_ref="availability_restock", attributable_inr=3.24e7,
        expected_low_inr=2.3e7, expected_high_inr=3.1e7,
        realised_inr=2.43e7, horizon_weeks=8, at=NOW, layer=layer,
    )
    assert result.realised_share == pytest.approx(0.75)
    assert result.implausible is False


def test_an_implausible_recovery_is_recorded_and_excluded(clean, layer):
    """Recovering three times the attributed loss means something else
    moved. Kept where anyone can see it, kept out of the update."""
    curves.record_realisation(
        clean, case_id="TEST-R", playbook="availability_recovery",
        curve_ref="availability_restock", attributable_inr=1.0e7,
        expected_low_inr=0.7e7, expected_high_inr=0.9e7,
        realised_inr=3.0e7, horizon_weeks=8, at=NOW, layer=layer,
    )
    usable, excluded = curves.shares_for(clean, "availability_restock")
    assert usable == ()
    assert excluded == 1


def test_a_curve_below_the_minimum_does_not_move(clean, layer):
    curves.record_realisation(
        clean, case_id="TEST-R1", playbook="availability_recovery",
        curve_ref="availability_restock", attributable_inr=1.0e7,
        expected_low_inr=0.7e7, expected_high_inr=0.9e7,
        realised_inr=0.5e7, horizon_weeks=8, at=NOW, layer=layer,
    )
    update = curves.update_for(clean, "availability_restock", layer=layer)
    assert update.observations == 1
    assert not update.moved, "one realisation is a coincidence"


def test_enough_realisations_move_the_curve(clean, layer):
    """Six disappointing recoveries pull the band down."""
    declared = layer.recovery_curves.curves["availability_restock"]
    for index in range(6):
        curves.record_realisation(
            clean, case_id=f"TEST-R{index}", playbook="availability_recovery",
            curve_ref="availability_restock", attributable_inr=1.0e7,
            expected_low_inr=0.71e7, expected_high_inr=0.96e7,
            realised_inr=0.40e7, horizon_weeks=8, at=NOW, layer=layer,
        )
    update = curves.update_for(clean, "availability_restock", layer=layer)
    assert update.observations == 6
    assert update.moved
    assert update.updated_p25 < declared.p25
    assert update.updated_p75 < declared.p75
    assert update.sample_size == declared.sample_size + 6


def test_the_curve_shift_is_capped(clean, layer):
    cap = layer.learning.recovery.max_absolute_shift
    declared = layer.recovery_curves.curves["availability_restock"]
    for index in range(50):
        curves.record_realisation(
            clean, case_id=f"TEST-C{index}", playbook="availability_recovery",
            curve_ref="availability_restock", attributable_inr=1.0e7,
            expected_low_inr=0.71e7, expected_high_inr=0.96e7,
            realised_inr=0.0, horizon_weeks=8, at=NOW, layer=layer,
        )
    update = curves.update_for(clean, "availability_restock", layer=layer)
    assert declared.p25 - update.updated_p25 <= cap + 1e-9
    assert update.updated_p25 <= update.updated_p75, "a band cannot have negative width"


def test_accepting_an_action_moves_no_curve(clean, layer):
    """A loop that moved a recovery curve on the accept would be learning
    from intentions."""
    _, effect = submit(
        clean, layer, level="action", action="accept", target_id="availability_recovery"
    )
    assert not effect.moved_calibration
    assert not effect.moved_prior
    assert any("D+56" in note for note in effect.notes)


# ===========================================================================
# The migration
# ===========================================================================


def test_the_warehouse_needs_no_migration_after_create_schema(wh):
    assert pending(wh) == ()


def test_a_migration_that_would_destroy_data_refuses():
    """A prototype's convenience is not worth a silent DROP TABLE."""
    from engine.db import IN_MEMORY, connect
    from engine.warehouse.load import create_schema

    old = connect(IN_MEMORY)
    old.execute(
        "CREATE TABLE case_outcome (case_id VARCHAR PRIMARY KEY, "
        "recorded_at TIMESTAMP NOT NULL, action_taken VARCHAR, "
        "outcome VARCHAR NOT NULL, realised_recovery_inr DOUBLE, "
        "horizon_weeks INTEGER, was_correct BOOLEAN, note VARCHAR)"
    )
    old.execute(
        "INSERT INTO case_outcome VALUES ('C1', now(), 'x', 'recovered', 1.0, 8, true, null)"
    )
    with pytest.raises(MigrationError, match="would drop them"):
        create_schema(old)
    assert old.execute("SELECT COUNT(*) FROM case_outcome").fetchone()[0] == 1


def test_a_stale_warehouse_migrates():
    from engine.db import IN_MEMORY, connect
    from engine.warehouse.load import create_schema
    from engine.warehouse.migrate import columns_of

    old = connect(IN_MEMORY)
    old.execute(
        "CREATE TABLE case_outcome (case_id VARCHAR PRIMARY KEY, "
        "recorded_at TIMESTAMP NOT NULL, action_taken VARCHAR, "
        "outcome VARCHAR NOT NULL, realised_recovery_inr DOUBLE, "
        "horizon_weeks INTEGER, was_correct BOOLEAN, note VARCHAR)"
    )
    create_schema(old)
    assert "horizon_days" in columns_of(old, "case_outcome")
    assert pending(old) == ()


# ===========================================================================
# Through the API — the accept criteria as a caller sees them
# ===========================================================================


@pytest.fixture(scope="module")
def api(wh, layer, tmp_path_factory):
    """An app over the session warehouse, with one case in its store.

    The case is ASSEMBLED rather than investigated. A real run is thirteen
    seconds and proves the pipeline, which `tests/test_api.py` already
    does; what these tests need is a published case with a known
    confidence to give feedback about.
    """
    from fastapi.testclient import TestClient

    from api.app import create_app
    from api.auth import HEADER, mint
    from api.store import CaseStore, store_case
    from engine.contracts import Adjudication
    from engine.verdict.casefile import CaseFile
    from llm.provider import MockProvider
    from tests.conftest import (
        FIXED_TS,
        make_adjudication,
        make_evidence,
        make_hypothesis,
        make_verdict,
    )

    published = make_adjudication().model_copy(
        update={
            "case_id": "TEST-API",
            "kpi": KPI,
            # A real driver, so its prior is one the causal graph declares.
            "hypotheses": (
                make_hypothesis().model_copy(
                    update={"hypothesis_id": DRIVER, "attributed_share": 0.79}
                ),
            ),
            "confidence": make_adjudication().confidence.model_copy(
                update={"raw": 0.84, "after_caps": 0.84, "calibrated": 0.84}
            ),
        }
    )
    store = CaseStore()
    store_case(
        store,
        CaseFile(
            case_id="TEST-API",
            adjudication=published,
            verdict=make_verdict().model_copy(update={"case_id": "TEST-API"}),
        ),
        user_id="U008",
        persona="analyst",
    )
    app = create_app(
        connection=wh,
        layer=layer,
        provider=MockProvider(),
        store=store,
        # Redirected: a suite run must not leave a generated file in the
        # working tree, where it would read as a change somebody made.
        overlay_path=tmp_path_factory.mktemp("overlay") / "priors.yaml",
    )
    return TestClient(app), {HEADER: mint("U008", "analyst")}


def test_the_options_come_from_the_contract(api, layer):
    client, headers = api
    body = client.get("/api/feedback/options", headers=headers).json()
    assert set(body["verdict_actions"]) == set(layer.learning.feedback.verdict_actions)
    assert set(body["reason_codes"]) == set(layer.learning.feedback.reason_codes)
    assert body["verdict_actions"]["reject"]["requires_reason"] is True


def test_feedback_moves_the_bin_and_calibration_returns_it(api, clean, layer):
    """ACCEPT, end to end: submitting feedback measurably moves a
    reliability bin, and the change is returned by /api/calibration on the
    next call."""
    client, headers = api
    band = "80%-90%"

    before = client.get("/api/calibration", headers=headers).json()
    row_before = next(b for b in before["bands"] if b["label"] == band)

    posted = client.post(
        "/api/cases/TEST-API/feedback/verdict",
        json={"action": "accept"},
        headers=headers,
    )
    assert posted.status_code == 201, posted.text
    assert posted.json()["moved_calibration"] is True

    after = client.get("/api/calibration", headers=headers).json()
    row_after = next(b for b in after["bands"] if b["label"] == band)

    assert row_after["cases"] == row_before["cases"] + 1
    assert row_after["accuracy"] != row_before["accuracy"]
    assert after["ece_raw"] != before["ece_raw"], "ECE recomputed"


def test_rejecting_a_driver_is_reported_with_the_prior_that_moved(api, clean, layer):
    """ACCEPT, end to end: rejecting a driver updates its prior, and the
    response says what it moved rather than acknowledging receipt."""
    client, headers = api

    posted = client.post(
        "/api/cases/TEST-API/feedback/driver",
        json={"driver": DRIVER, "action": "reject", "reason_code": "wrong_driver"},
        headers=headers,
    )
    assert posted.status_code == 201, posted.text
    body = posted.json()

    assert body["moved_prior"] is True
    assert body["prior_hypothesis"] == DRIVER
    assert body["prior_kpi"] == KPI
    assert body["prior_after"] < body["prior_before"]
    assert body["prior_shift"] < 0

    standing = client.get(f"/api/priors?kpi={KPI}", headers=headers).json()
    entry = next(item for item in standing["priors"] if item["hypothesis"] == DRIVER)
    assert entry["rejected"] == 1
    assert entry["effective"] < entry["declared"]


def test_a_driver_not_on_the_case_is_a_404(api, clean):
    client, headers = api
    response = client.post(
        "/api/cases/TEST-API/feedback/driver",
        json={"driver": "weather", "action": "reject"},
        headers=headers,
    )
    assert response.status_code == 404


def test_a_rejection_without_a_reason_is_a_422(api, clean):
    client, headers = api
    response = client.post(
        "/api/cases/TEST-API/feedback/verdict",
        json={"action": "reject"},
        headers=headers,
    )
    assert response.status_code == 422
    assert "reason" in response.json()["detail"]


def test_request_investigation_says_what_it_did_not_do(api, clean):
    client, headers = api
    body = client.post(
        "/api/cases/TEST-API/feedback/verdict",
        json={"action": "request_investigation"},
        headers=headers,
    ).json()
    assert body["moved_calibration"] is False
    assert body["notes"]


def test_an_outcome_supersedes_through_the_api(api, clean, layer):
    client, headers = api
    client.post(
        "/api/cases/TEST-API/feedback/verdict", json={"action": "accept"}, headers=headers
    )
    response = client.post(
        "/api/cases/TEST-API/outcome",
        json={
            "horizon_days": 56,
            "realised_recovery_inr": 1.0,
            "expected_low_inr": 2.3e7,
            "expected_high_inr": 3.1e7,
            "horizon_weeks": 8,
        },
        headers=headers,
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["recovered"] is False
    assert body["superseded_feedback"] is True

    entries = loop.ledger_for_case(clean, "TEST-API")
    assert len(entries) == 1
    assert entries[0]["was_correct"] is False


def test_the_wrong_field_for_a_horizon_is_a_422(api, clean):
    client, headers = api
    response = client.post(
        "/api/cases/TEST-API/outcome",
        json={"horizon_days": 14, "realised_recovery_inr": 1.0},
        headers=headers,
    )
    assert response.status_code == 422
    assert "cause_confirmed" in response.json()["detail"]


def test_an_undeclared_horizon_is_a_422(api, clean):
    client, headers = api
    response = client.post(
        "/api/cases/TEST-API/outcome",
        json={"horizon_days": 30, "cause_confirmed": True},
        headers=headers,
    )
    assert response.status_code == 422


def test_the_curve_standing_is_served(api, clean, layer):
    client, headers = api
    body = client.get("/api/recovery-curves", headers=headers).json()
    assert body
    refs = {item["curve_ref"] for item in body}
    assert refs == set(layer.recovery_curves.curves)
    for item in body:
        assert item["effective_p25"] <= item["effective_p75"]


def test_feedback_needs_a_signed_header(api):
    client, _ = api
    assert client.post(
        "/api/cases/TEST-API/feedback/verdict", json={"action": "accept"}
    ).status_code == 401
    assert client.get("/api/priors").status_code == 401
