"""Gate 1 — VALIDATE.

The two acceptance criteria are scenarios:

  * #2470 exits DATA_INCIDENT, names the 25 failed feeds, and opens no case
  * #2451 passes all five checks clean

and then every exit path has a test of its own, because a named exit that
has never been produced is a string in a YAML file, not a behaviour.

The exit-path tests mutate the warehouse inside a transaction that is
rolled back afterwards. The session warehouse is shared and copying a
155 MB file per test is not worth it; a ROLLBACK is exact and instant.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from engine.contracts import Evidence, GateResult
from engine.db import GovernanceError
from engine.validate import (
    ValidationError,
    ValidationRequest,
    baseline_window,
    definition_hash,
    period_window,
    validate,
)
from security.policy import User

# The scenarios, as the generator config declares them.
CASE_2451 = ValidationRequest(
    kpi="net_revenue",
    scope="West",
    grain="monthly",
    period="2025-11",
    comparison_period="2025-10",
)
CASE_2470 = ValidationRequest(
    kpi="net_revenue",
    scope="West",
    grain="daily",
    period="2025-11-12",
)

ANALYST = User(user_id="U008", persona="analyst", display_name="Meera Joshi")

CHECK_IDS = (
    "source_freshness",
    "row_count_delta",
    "definition_drift",
    "single_transaction_dominance",
    "restatement_pending",
)


@pytest.fixture
def sandbox(warehouse):
    """A transaction that is always rolled back.

    Everything the exit-path tests insert or delete — including the audit
    rows the checks write on the way past — disappears at the end.
    """
    warehouse.execute("BEGIN TRANSACTION")
    try:
        yield warehouse
    finally:
        warehouse.execute("ROLLBACK")


@pytest.fixture(scope="module")
def result_2451(warehouse):
    return validate(warehouse, ANALYST, CASE_2451)


@pytest.fixture(scope="module")
def result_2470(warehouse):
    return validate(warehouse, ANALYST, CASE_2470)


# ===========================================================================
# ACCEPT: #2470 exits DATA_INCIDENT and opens no case
# ===========================================================================


def test_2470_does_not_open_a_case(result_2470):
    assert result_2470.case_opened is False
    assert result_2470.gate.passed is False


def test_2470_exits_data_incident(result_2470):
    assert result_2470.outcome_code == "DATA_INCIDENT"
    assert result_2470.gate.gate_id == 1


def test_2470_names_the_twenty_five_failed_feeds(result_2470):
    """"Twenty-five stores" is a number. Naming them is what can be acted on."""
    check = result_2470.check("row_count_delta")
    assert check.outcome == "DATA_INCIDENT"
    assert len(check.subjects) == 25
    assert len(set(check.subjects)) == 25
    assert all(store_id.startswith("S") for store_id in check.subjects)
    assert "25 of 140" in check.detail


def test_2470_reports_the_revenue_behind_the_missing_feeds(result_2470):
    """The registry's 18.0%: the size of the hole in the headline."""
    evidence = result_2470.evidence_by_id()["validate.row_count_delta.revenue_share"]
    assert evidence.value == pytest.approx(18.0, abs=0.1)
    assert evidence.unit == "pct"


def test_2470_reports_the_share_of_stores_too(result_2470):
    evidence = result_2470.evidence_by_id()["validate.row_count_delta.store_share"]
    assert evidence.value == pytest.approx(25 / 140 * 100.0, abs=0.01)


def test_2470_is_caught_by_exactly_one_check(result_2470):
    """The other four are clean, and the gate reports all five."""
    failures = {check.check_id for check in result_2470.failures}
    assert failures == {"row_count_delta"}
    assert len(result_2470.checks) == len(CHECK_IDS)


def test_2470_gate_detail_leads_with_the_reason(result_2470):
    detail = result_2470.gate.detail
    assert "DATA_INCIDENT" in detail
    assert "25 of 140" in detail


# ===========================================================================
# ACCEPT: #2451 passes all five clean
# ===========================================================================


def test_2451_opens_a_case(result_2451):
    assert result_2451.case_opened is True
    assert result_2451.outcome_code is None


@pytest.mark.parametrize("check_id", CHECK_IDS)
def test_2451_passes_every_check(result_2451, check_id):
    check = result_2451.check(check_id)
    assert check.outcome == "CLEAN", f"{check_id}: {check.detail}"


def test_2451_ran_all_five_and_says_so(result_2451):
    assert len(result_2451.checks) == len(CHECK_IDS)
    assert {check.check_id for check in result_2451.checks} == set(CHECK_IDS)
    assert "all 5 checks clean" in result_2451.gate.detail


def test_2451_the_incident_day_does_not_fail_the_month(result_2451):
    """The point of #2470 being a DAILY check inside #2451's month.

    Twenty-five stores lose one day in thirty. That is a three percent
    move in their monthly mean, nowhere near the fifty percent shortfall
    that makes a store an incident — so the same fault that kills the
    daily check leaves the monthly one clean.
    """
    check = result_2451.check("row_count_delta")
    assert check.outcome == "CLEAN"
    assert check.subjects == ()


def test_2451_movement_is_the_registry_decline(result_2451):
    """Gate 1 computes a movement only to divide by it, but it must be right."""
    evidence = result_2451.evidence_by_id()[
        "validate.single_transaction_dominance.movement"
    ]
    assert evidence.value == pytest.approx(-6.78, abs=0.02)
    assert evidence.unit == "INR_CR"


# ===========================================================================
# Every exit path
# ===========================================================================


def test_exit_data_incident_from_a_stale_feed(sandbox):
    """Check 1: the POS feed stops arriving, ten days before the clock."""
    sandbox.execute("DELETE FROM fact_sales_daily WHERE txn_date > DATE '2025-11-20'")
    result = validate(sandbox, ANALYST, CASE_2451)

    check = result.check("source_freshness")
    assert check.outcome == "DATA_INCIDENT"
    assert "pos" in check.subjects
    assert result.outcome_code == "DATA_INCIDENT"
    assert result.case_opened is False

    staleness = result.evidence_by_id()["validate.source_freshness.pos"]
    assert staleness.unit == "hours"
    assert staleness.value > 6  # the contract's SLA for net_revenue


def test_exit_data_incident_from_missing_rows(result_2470):
    """Check 2: #2470 itself, which is the canonical instance."""
    assert result_2470.check("row_count_delta").outcome == "DATA_INCIDENT"


def test_exit_definition_change(sandbox, layer):
    """Check 3: the last run computed this KPI under a different definition."""
    spec = layer.validation.checks.definition_drift
    # The log holds exactly one prior definition for this KPI, and it is
    # not the one in force. Clearing first matters: another test in this
    # module has already recorded the current fingerprint.
    sandbox.execute(f'DELETE FROM "{spec.log_table}" WHERE kpi = ?', ["net_revenue"])
    sandbox.execute(
        f"""
        INSERT INTO "{spec.log_table}"
            (kpi, formula_hash, kpi_version, hashed_fields, first_seen_at, last_seen_at)
        VALUES ('net_revenue', 'a-definition-that-is-no-longer-in-force', 1, ?, ?, ?)
        """,
        [",".join(spec.hashed_fields), datetime(2025, 6, 1), datetime(2025, 11, 29)],
    )
    result = validate(sandbox, ANALYST, CASE_2451)

    check = result.check("definition_drift")
    assert check.outcome == "DEFINITION_CHANGE"
    assert "a-definition-" in check.subjects[0]
    assert result.outcome_code == "DEFINITION_CHANGE"
    assert result.case_opened is False


def test_exit_one_off(sandbox, layer):
    """Check 4: a single bill carrying more than a quarter of the movement."""
    spec = layer.validation.checks.single_transaction_dominance
    # #2451's movement is ~INR 6.78 Cr, so a quarter of it is ~INR 1.70 Cr.
    dominant = 2.5e7
    sandbox.execute(
        """
        INSERT INTO fact_bill_lines VALUES
        ('B999999999', 1, 'S0273', 'WE-0273', 'K001',
         TIMESTAMP '2025-11-14 11:00:00', DATE '2025-11-14',
         'RETAIL', 'SALE', 1, ?, 0.0, 0.0, 0.0, 0.02)
        """,
        [dominant],
    )
    result = validate(sandbox, ANALYST, CASE_2451)

    check = result.check("single_transaction_dominance")
    assert check.outcome == "ONE_OFF"
    assert "B999999999" in check.subjects
    assert result.outcome_code == "ONE_OFF"
    assert result.case_opened is False

    share = result.evidence_by_id()[
        "validate.single_transaction_dominance.share_of_movement"
    ]
    assert share.value > spec.max_top_transaction_share * 100.0


def test_exit_pending_restatement(sandbox, layer):
    """Check 5: finance has reopened the period."""
    spec = layer.validation.checks.restatement_pending
    sandbox.execute(
        f"""
        INSERT INTO "{spec.register_table}" VALUES
        ('RS9001', 'net_revenue', 'West', 'monthly', '2025-11',
         DATE '2025-11-28', NULL, ?, 'Two Pune stores double-counted a bulk order.')
        """,
        [spec.open_status],
    )
    result = validate(sandbox, ANALYST, CASE_2451)

    check = result.check("restatement_pending")
    assert check.outcome == "PENDING_RESTATEMENT"
    assert check.subjects == ("RS9001",)
    assert result.outcome_code == "PENDING_RESTATEMENT"
    assert result.case_opened is False


def test_exit_insufficient_baseline(warehouse):
    """Check 2 with nothing to compare against says so, rather than passing.

    June 2024 is the first month in the series, so its eight-week lookback
    lies entirely before the data starts. A check with no baseline that
    returns CLEAN is worse than useless: it looks like a verified pass.
    """
    result = validate(
        warehouse,
        ANALYST,
        ValidationRequest("net_revenue", "West", "monthly", "2024-06", "2024-05"),
    )
    check = result.check("row_count_delta")
    assert check.outcome == "INSUFFICIENT_BASELINE"
    assert result.case_opened is False


def test_a_wildcard_restatement_covers_a_region(sandbox, layer):
    """An All-India flag covers West. A West flag would not cover All-India."""
    spec = layer.validation.checks.restatement_pending
    sandbox.execute(
        f"""
        INSERT INTO "{spec.register_table}" VALUES
        ('RS9002', 'net_revenue', ?, 'monthly', '2025-11',
         DATE '2025-11-27', NULL, ?, 'National tax-rate correction.')
        """,
        [spec.scope_wildcard, spec.open_status],
    )
    result = validate(sandbox, ANALYST, CASE_2451)
    assert result.check("restatement_pending").outcome == "PENDING_RESTATEMENT"


def test_a_resolved_restatement_does_not_fire(warehouse, layer):
    """The seeded register is historical and closed, and must stay quiet."""
    seeded = warehouse.execute(
        f"SELECT status FROM {layer.validation.checks.restatement_pending.register_table}"
    ).fetchall()
    assert seeded and all(row[0] == "RESOLVED" for row in seeded)


# ===========================================================================
# Named exits, not booleans
# ===========================================================================


def test_every_check_returns_a_named_outcome(result_2451, result_2470):
    """The design point: no check reports True or False."""
    for result in (result_2451, result_2470):
        for check in result.checks:
            assert isinstance(check.outcome, str)
            assert check.outcome == check.outcome.upper()
            assert check.outcome != "TRUE" and check.outcome != "FALSE"


def test_the_gate_priority_comes_from_the_semantic_layer(layer):
    from engine.validate.checks import CHECKS

    declared = [spec.order for spec in layer.validation.checks.in_order()]
    assert declared == sorted(declared)
    assert [key for key, _ in CHECKS] == [
        name
        for name, _ in sorted(
            (
                (key, getattr(layer.validation.checks, key).order)
                for key in layer.validation.checks.__class__.model_fields
            ),
            key=lambda pair: pair[1],
        )
    ]


def test_the_gate_cannot_pass_while_a_check_failed():
    """The contract refuses the inconsistency rather than trusting callers."""
    from engine.contracts import CheckResult

    failed = CheckResult(
        check_id="row_count_delta",
        name="Row-count delta",
        order=2,
        outcome="DATA_INCIDENT",
        detail="25 feeds did not load.",
    )
    with pytest.raises(ValueError, match="reports passed=True"):
        GateResult(
            gate_id=1,
            name="Data completeness",
            passed=True,
            outcome_code=None,
            detail="fine",
            checks=(failed,),
        )


def test_a_passing_gate_carries_no_outcome_code():
    from engine.contracts import CheckResult

    clean = CheckResult(
        check_id="row_count_delta",
        name="Row-count delta",
        order=2,
        outcome="CLEAN",
        detail="nothing short.",
    )
    with pytest.raises(ValueError, match="still carries"):
        GateResult(
            gate_id=1,
            name="Data completeness",
            passed=True,
            outcome_code="DATA_INCIDENT",
            detail="fine",
            checks=(clean,),
        )


# ===========================================================================
# Evidence, not bare results (rule 3)
# ===========================================================================


def test_every_check_that_measured_something_emitted_evidence(result_2451):
    with_evidence = {
        check.check_id for check in result_2451.checks if check.evidence_ids
    }
    assert with_evidence == set(CHECK_IDS)


def test_all_evidence_is_typed_and_unique(result_2451):
    assert result_2451.evidence
    assert all(isinstance(item, Evidence) for item in result_2451.evidence)
    ids = [item.evidence_id for item in result_2451.evidence]
    assert len(ids) == len(set(ids))


def test_no_evidence_came_from_the_model(result_2451):
    """Rule 1. Gate 1 does not call an LLM at all, and this pins it."""
    assert all(item.produced_by == "code" for item in result_2451.evidence)


def test_every_evidence_carries_lineage_and_a_unit(result_2451):
    for item in result_2451.evidence:
        assert item.lineage, f"{item.evidence_id} has nothing to click through to"
        assert item.unit


def test_the_sampled_bill_table_declares_its_sample_rate(result_2451, layer):
    """A clean ONE_OFF result must not be read as proof of no such bill."""
    top = result_2451.evidence_by_id()[
        "validate.single_transaction_dominance.top_transaction"
    ]
    assert top.completeness == pytest.approx(0.02)
    assert "LOWER BOUND" in (top.notes or "")


def test_reliability_weights_come_from_the_semantic_layer(result_2451, layer):
    weights = layer.adjudication.reliability.weights
    for item in result_2451.evidence:
        assert item.reliability == pytest.approx(weights[item.kind])


# ===========================================================================
# Period arithmetic
# ===========================================================================


def test_monthly_period_spans_the_whole_month():
    window = period_window("2025-11", "monthly")
    assert window.start == date(2025, 11, 1)
    assert window.end == date(2025, 11, 30)
    assert window.days == 30


def test_february_in_a_leap_year():
    assert period_window("2024-02", "monthly").end == date(2024, 2, 29)


def test_daily_period_is_one_day():
    window = period_window("2025-11-12", "daily")
    assert window.start == window.end == date(2025, 11, 12)
    assert window.days == 1


def test_weekly_period_is_seven_days():
    window = period_window("2025-11-10", "weekly")
    assert window.days == 7
    assert window.end == date(2025, 11, 16)


def test_baseline_ends_the_day_before_the_period_and_does_not_overlap():
    period = period_window("2025-11", "monthly")
    baseline = baseline_window(period, 8)
    assert baseline.end == period.start - timedelta(days=1)
    assert baseline.days == 8 * 7
    assert baseline.end < period.start


def test_a_malformed_period_is_refused():
    with pytest.raises(ValidationError, match="not YYYY-MM"):
        period_window("2025-11-12", "monthly")


def test_an_unknown_kpi_is_refused(warehouse):
    with pytest.raises(ValidationError, match="no KPI contract"):
        validate(
            warehouse,
            ANALYST,
            ValidationRequest("not_a_kpi", "West", "monthly", "2025-11"),
        )


def test_a_grain_the_contract_does_not_support_is_refused(warehouse, layer):
    """qcomm_fulfilment_rate is weekly only; a monthly ask is a mistake."""
    contract = layer.kpis["qcomm_fulfilment_rate"]
    unsupported = next(
        grain
        for grain in ("daily", "weekly", "monthly")
        if grain not in contract.grain.supported
    )
    with pytest.raises(ValidationError, match="not computed at"):
        validate(
            warehouse,
            ANALYST,
            ValidationRequest("qcomm_fulfilment_rate", "All-India", unsupported, "2025-11"),
        )


# ===========================================================================
# The definition fingerprint
# ===========================================================================


def test_the_fingerprint_covers_the_fields_that_decide_meaning(layer):
    spec = layer.validation.checks.definition_drift
    assert set(spec.hashed_fields) == {"version", "definition", "formula_sql"}


def test_changing_the_formula_changes_the_fingerprint(layer):
    spec = layer.validation.checks.definition_drift
    contract = layer.kpis["net_revenue"]
    before = definition_hash(contract, spec.hashed_fields, spec.algorithm)
    changed = contract.model_copy(
        update={"formula_sql": contract.formula_sql + "\n-- and now excludes B2C"}
    )
    assert definition_hash(changed, spec.hashed_fields, spec.algorithm) != before


def test_the_fingerprint_is_stable_across_runs(layer):
    spec = layer.validation.checks.definition_drift
    contract = layer.kpis["net_revenue"]
    first = definition_hash(contract, spec.hashed_fields, spec.algorithm)
    second = definition_hash(contract, spec.hashed_fields, spec.algorithm)
    assert first == second


def test_the_first_run_records_and_passes(sandbox, layer):
    spec = layer.validation.checks.definition_drift
    sandbox.execute(f'DELETE FROM "{spec.log_table}" WHERE kpi = \'net_revenue\'')
    result = validate(sandbox, ANALYST, CASE_2451)

    assert result.check("definition_drift").outcome == "CLEAN"
    assert "first run" in result.check("definition_drift").detail
    recorded = sandbox.execute(
        f'SELECT COUNT(*) FROM "{spec.log_table}" WHERE kpi = \'net_revenue\''
    ).fetchone()[0]
    assert recorded == 1


# ===========================================================================
# Governance holds through the gate (rule 5)
# ===========================================================================


def test_every_check_read_is_audited(sandbox):
    sandbox.execute("DELETE FROM audit_log")
    validate(sandbox, ANALYST, CASE_2451)
    rows = sandbox.execute(
        "SELECT purpose, kpi FROM audit_log ORDER BY purpose"
    ).fetchall()
    purposes = {row[0] for row in rows}
    assert any(purpose.startswith("validate.source_freshness") for purpose in purposes)
    assert "validate.row_count_delta" in purposes
    assert "validate.definition_drift" in purposes
    assert "validate.restatement_pending" in purposes


def test_a_regional_manager_can_validate_their_own_region(warehouse):
    user = User(user_id="U003", persona="regional_manager", region="West")
    result = validate(warehouse, user, CASE_2451)
    assert result.case_opened is True


def test_a_regional_manager_cannot_validate_another_region(warehouse):
    """East's manager asking about West sees no stores, and is told so.

    The row predicate removes every West row before the check sees it, so
    there is no baseline. That is the honest answer: not "West is fine",
    and not a crash — no visible history to check.
    """
    user = User(user_id="U004", persona="regional_manager", region="East")
    result = validate(warehouse, user, CASE_2451)
    assert result.check("row_count_delta").outcome == "INSUFFICIENT_BASELINE"
    assert result.case_opened is False


def test_a_persona_the_contract_does_not_know_is_refused(warehouse):
    with pytest.raises(GovernanceError, match="not defined in the access policy"):
        validate(warehouse, User(user_id="U999", persona="intern"), CASE_2451)


def test_metadata_reads_cannot_be_pointed_at_the_facts(warehouse):
    from engine.db import execute_metadata

    with pytest.raises(GovernanceError, match="not a metadata table"):
        execute_metadata(
            ANALYST,
            "fact_sales_daily",
            "SELECT COUNT(*) FROM fact_sales_daily",
            connection=warehouse,
        )


def test_the_metadata_allow_list_holds_no_business_tables(layer):
    """The allow-list applies no row predicate, so this is the whole guard."""
    for table in layer.warehouse.governance.metadata_tables:
        assert not table.startswith(("fact_", "doc_", "ext_"))
