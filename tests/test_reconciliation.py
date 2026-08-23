"""The four reconciliation problems, each visible rather than fixed.

The P4 acceptance criteria are two numbers:

  * the two revenue definitions differ by 4.1% and BOTH compute
  * quarantine = 3 stores / 0.4% of revenue

Both are asserted below against the loaded warehouse, not against the
generator's manifest — the point is that the warehouse reproduces them
from the rows it holds.

The other two problems have no target number in CLAUDE.md, so what is
asserted about them is the property that matters: the grain allocation
conserves spend and tags its assumption, and the calendar mismatch is
measured rather than declared.
"""

from __future__ import annotations

import pytest

from engine.contracts import Evidence
from engine.warehouse.reconcile import (
    check_definition_conflict,
    reconcile,
    warehouse_clock,
    write_gaps,
)

#: Tolerances. The registry quotes 4.1% and 0.4%; both are generator
#: targets solved to four decimal places, so a basis point either way is
#: generous rather than tight.
PCT = 0.05
SHARE_PCT = 0.02


# ===========================================================================
# 1. Definition conflict — the accept criterion
# ===========================================================================


def test_both_revenue_definitions_compute(reconciliation):
    """Neither figure is a placeholder: both are real sums over real rows."""
    totals = reconciliation.definition_conflict.totals_inr_cr
    assert set(totals) == {"contract", "pos_ledger", "marketing_reported"}
    for name, value in totals.items():
        assert value > 0, f"{name} computed to {value}"


def test_the_two_definitions_differ_by_4_1_percent(reconciliation):
    """ACCEPT: POS-including-B2B against marketing-excluding-returns."""
    assert reconciliation.definition_conflict.gap_pct == pytest.approx(4.1, abs=PCT)


def test_the_gap_is_the_b2b_book_less_the_returns_credit(warehouse, reconciliation):
    """The gap is arithmetic, not a fitted constant.

    pos_ledger - marketing = (net + b2b) - (net + returns) = b2b - returns,
    and the denominator is the marketing figure. If this identity ever
    stops holding, one of the two definitions has drifted from the
    semantic layer.
    """
    b2b, returns, net = warehouse.execute(
        "SELECT SUM(b2b_net_revenue_inr), SUM(returns_amount_inr), SUM(net_revenue_inr) "
        "FROM fact_sales_daily"
    ).fetchone()
    expected = (b2b - returns) / (net + returns) * 100.0
    assert reconciliation.definition_conflict.gap_pct == pytest.approx(expected, abs=1e-6)


def test_the_contract_figure_sits_between_the_two_rivals(reconciliation):
    """The arbiter is not simply the largest or the smallest.

    Both rival definitions overstate the contract's figure, from opposite
    directions: the POS ledger by the B2B book it does not exclude, and
    marketing by the returns it does not deduct. That is what makes this a
    conflict rather than one team being wrong.
    """
    totals = reconciliation.definition_conflict.totals_inr_cr
    assert totals["contract"] < totals["marketing_reported"]
    assert totals["contract"] < totals["pos_ledger"]


def test_the_definitions_come_from_the_semantic_layer(layer, warehouse):
    """Rule 2: the expressions are YAML, and the module follows them.

    Recomputing each definition straight from the YAML expression must
    reproduce what `check_definition_conflict` reported, or the module has
    a private copy of what a team means by revenue.
    """
    spec = layer.warehouse.reconciliation.definition_conflict
    units = layer.warehouse.units
    result = check_definition_conflict(warehouse, layer, warehouse_clock(warehouse))
    for name, definition in spec.definitions.items():
        direct = warehouse.execute(
            f"SELECT {definition.expression} FROM {spec.fact_table}"
        ).fetchone()[0]
        # Quoted at the semantic layer's precision, so compare at it too.
        assert result.totals_inr_cr[name] == pytest.approx(
            direct / units.inr_per_crore, abs=10.0**-units.crore_places
        )


def test_the_definition_gap_holds_for_the_case_scope(warehouse, layer):
    """West is where case #2451 lives, and the gap follows it there.

    The B2B book is lumpy, so a single region will not land on 4.1% to the
    basis point. It has to stay in the same neighbourhood, or a case
    opened on West would be arguing about a different disagreement from
    the one the warehouse reported.
    """
    result = check_definition_conflict(
        warehouse, layer, warehouse_clock(warehouse), scope="West"
    )
    assert result.gap_pct == pytest.approx(4.1, abs=1.0)
    assert result.totals_inr_cr["contract"] > 0


def test_transfers_are_excluded_by_every_definition(reconciliation):
    """The one thing all three teams agree on, carried so it can be shown."""
    assert reconciliation.definition_conflict.excluded_transfers_inr_cr > 0


# ===========================================================================
# 2. Entity key mismatch — the accept criterion
# ===========================================================================


def test_three_store_keys_do_not_resolve(reconciliation):
    """ACCEPT, first half: 3 stores unmapped after the re-fascia."""
    assert reconciliation.entity_key_mismatch.unmapped_key_count == 3


def test_quarantined_revenue_is_0_4_percent(reconciliation):
    """ACCEPT, second half: 0.4% of revenue."""
    share_pct = reconciliation.entity_key_mismatch.quarantined_share * 100.0
    assert share_pct == pytest.approx(0.4, abs=SHARE_PCT)


def test_the_quarantine_is_held_not_dropped(warehouse, reconciliation):
    """Quarantined rows are still in the warehouse and still countable."""
    keyed, conformed, quarantined = (
        warehouse.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        for view in (
            "v_sales_daily_keyed",
            "v_sales_daily_conformed",
            "v_sales_daily_quarantined",
        )
    )
    assert conformed + quarantined == keyed
    assert quarantined == reconciliation.entity_key_mismatch.unmapped_rows


def test_the_unmapped_keys_are_the_re_fascia_codes(reconciliation):
    """They are new POS codes, and the operations master has never seen one."""
    keys = reconciliation.entity_key_mismatch.unmapped_keys
    assert len(keys) == len(set(keys))
    assert all(key.startswith("RC-") for key in keys), keys


def test_the_quarantine_starts_at_the_re_fascia_and_not_before(warehouse):
    """Each store is fine until its fascia changes, then stops resolving.

    That is what makes this a mid-series mapping failure rather than three
    stores that were never onboarded.
    """
    rows = warehouse.execute(
        "SELECT store_id, MIN(txn_date), MAX(txn_date) FROM v_sales_daily_quarantined "
        "GROUP BY store_id ORDER BY store_id"
    ).fetchall()
    assert len(rows) == 3
    for store_id, first, last in rows:
        earlier = warehouse.execute(
            "SELECT COUNT(*) FROM v_sales_daily_conformed "
            "WHERE store_id = ? AND txn_date < ?",
            [store_id, first],
        ).fetchone()[0]
        assert earlier > 0, f"{store_id} never resolved at all"
        assert first < last


def test_store_id_is_not_used_to_rescue_the_quarantine(warehouse):
    """The surrogate key would resolve every row, and is deliberately unused.

    If the conformance view ever falls back to store_id, the quarantine
    empties and the mapping gap becomes invisible. This is the test that
    fails when somebody 'fixes' it.
    """
    rescuable = warehouse.execute(
        "SELECT COUNT(DISTINCT q.store_id) FROM v_sales_daily_quarantined AS q "
        "JOIN dim_store AS d USING (store_id)"
    ).fetchone()[0]
    assert rescuable == 3, "the quarantined stores DO exist in dim_store — by store_id"
    still_quarantined = warehouse.execute(
        "SELECT COUNT(*) FROM v_sales_daily_quarantined"
    ).fetchone()[0]
    assert still_quarantined > 0, "the view rescued them anyway"


# ===========================================================================
# 3. Grain mismatch
# ===========================================================================


def test_weekly_spend_is_allocated_to_days(reconciliation):
    grain = reconciliation.grain_mismatch
    assert grain.weeks > 0
    assert grain.allocated_days > grain.weeks


def test_the_allocation_conserves_spend(reconciliation):
    """Allocation moves money between rows. It must not create any."""
    grain = reconciliation.grain_mismatch
    assert grain.allocated_total_inr_cr == pytest.approx(
        grain.weekly_total_inr_cr, rel=1e-9
    )
    assert grain.reconciles


def test_every_derived_value_carries_the_flat_intraweek_tag(reconciliation, layer):
    """The assumption rides on the Evidence, not in a footnote."""
    tag = layer.warehouse.reconciliation.grain_mismatch.assumption_tag
    assert tag == "flat_intraweek"
    for item in reconciliation.grain_mismatch.evidence:
        assert tag in item.assumptions, f"{item.evidence_id} lost its assumption tag"


def test_only_the_allocated_evidence_is_tagged(reconciliation, layer):
    """A tag on everything would be a tag on nothing."""
    tag = layer.warehouse.reconciliation.grain_mismatch.assumption_tag
    untagged = [
        item
        for item in reconciliation.evidence
        if tag in item.assumptions
        and item not in reconciliation.grain_mismatch.evidence
    ]
    assert not untagged


def test_short_weeks_at_the_series_edges_are_counted(reconciliation):
    """A truncated week divides by its real day count, and says how many.

    Dividing an edge week by a nominal seven would understate its daily
    spend without anything on screen to say so.
    """
    assert reconciliation.grain_mismatch.short_weeks > 0


def test_the_allocation_divides_by_the_days_actually_present(warehouse):
    inconsistent = warehouse.execute(
        "SELECT COUNT(*) FROM v_marketing_spend_daily "
        "WHERE ABS(allocated_spend_inr * days_in_week - week_spend_inr) > 1e-6"
    ).fetchone()[0]
    assert inconsistent == 0


# ===========================================================================
# 4. Calendar mismatch
# ===========================================================================


def test_the_two_calendars_do_not_agree(reconciliation):
    """Fiscal weeks count from 1 April; ISO weeks start on a Monday."""
    share = reconciliation.calendar_mismatch.aligned_day_share
    assert 0.0 < share < 1.0, "the calendars either always or never align"


def test_alignment_is_measured_off_the_calendar_table(warehouse, reconciliation):
    measured = warehouse.execute(
        "SELECT AVG(CASE WHEN iso_week_start = fiscal_week_start THEN 1.0 ELSE 0.0 END) "
        "FROM dim_calendar"
    ).fetchone()[0]
    assert reconciliation.calendar_mismatch.aligned_day_share == pytest.approx(measured)


def test_festival_overlap_is_computed_per_period(reconciliation, layer):
    """Every festival occurrence gets a figure under every period type."""
    periods = set(layer.warehouse.reconciliation.calendar_mismatch.periods)
    overlaps = reconciliation.calendar_mismatch.overlaps
    assert overlaps
    by_occurrence: dict[tuple[str, str], set[str]] = {}
    for row in overlaps:
        by_occurrence.setdefault((row.festival, row.occurrence), set()).add(row.period_type)
    for key, seen in by_occurrence.items():
        assert seen == periods, f"{key} is missing period types {sorted(periods - seen)}"


def test_overlap_shares_are_shares(reconciliation):
    for row in reconciliation.calendar_mismatch.overlaps:
        assert 0.0 < row.max_overlap_share <= 1.0
        assert row.periods_spanned >= 1


def test_at_least_one_festival_lands_differently_under_the_two_calendars(reconciliation):
    """The whole point: which period a festival's trade belongs to moves.

    A festival that fills one fiscal week and splits across two ISO weeks
    produces two different week-on-week comparisons from the same data,
    and neither of them is wrong.
    """
    assert reconciliation.calendar_mismatch.divergent_festivals


def test_diwali_2025_fills_a_fiscal_week_but_not_an_iso_week(reconciliation):
    """A concrete instance, so the divergence is not just a count."""
    fiscal = reconciliation.calendar_mismatch.overlap("Diwali", "fiscal_week", "FY25-26")
    iso = reconciliation.calendar_mismatch.overlap("Diwali", "iso_week", "FY25-26")
    assert fiscal is not None and iso is not None
    assert fiscal.max_overlap_share > iso.max_overlap_share


# ===========================================================================
# The register
# ===========================================================================


def test_every_problem_produces_a_gap(reconciliation, layer):
    codes = {gap.gap_code for gap in reconciliation.gaps}
    assert codes == set(layer.warehouse.reconciliation.gap_codes())


def test_gaps_are_written_to_the_register(warehouse, reconciliation):
    warehouse.execute("DELETE FROM data_gap_register")
    written = write_gaps(warehouse, reconciliation)
    rows = warehouse.execute(
        "SELECT gap_code, severity, measure, value_numeric, unit, detail, resolution_hint "
        "FROM data_gap_register ORDER BY gap_code"
    ).fetchall()
    assert written == len(rows) == len(reconciliation.gaps)
    for _code, severity, measure, value, _unit, detail, hint in rows:
        assert severity in {"ASSUMPTION", "WARNING", "ERROR"}
        assert measure and detail and hint
        assert value is not None


def test_severity_comes_from_the_semantic_layer(reconciliation, layer):
    spec = layer.warehouse.reconciliation
    by_code = {gap.gap_code: gap.severity for gap in reconciliation.gaps}
    assert by_code[spec.entity_key_mismatch.gap_code] == "ERROR"
    assert by_code[spec.grain_mismatch.gap_code] == "ASSUMPTION"


# ===========================================================================
# Everything it emits is Evidence (rule 3)
# ===========================================================================


def test_no_bare_floats_leave_the_module(reconciliation):
    assert reconciliation.evidence
    assert all(isinstance(item, Evidence) for item in reconciliation.evidence)


def test_every_evidence_id_is_unique(reconciliation):
    ids = [item.evidence_id for item in reconciliation.evidence]
    assert len(ids) == len(set(ids))


def test_every_value_is_produced_by_code(reconciliation):
    """Rule 1: the model is never the source of a number."""
    assert all(item.produced_by == "code" for item in reconciliation.evidence)


def test_every_evidence_carries_lineage_and_a_unit(reconciliation):
    for item in reconciliation.evidence:
        assert item.lineage, f"{item.evidence_id} has no lineage to click through to"
        assert item.unit, f"{item.evidence_id} has no unit"


def test_reliability_weights_come_from_the_semantic_layer(reconciliation, layer):
    weights = layer.adjudication.reliability.weights
    for item in reconciliation.evidence:
        assert item.reliability == pytest.approx(weights[item.kind])


def test_freshness_is_measured_against_the_warehouse_clock(warehouse, reconciliation):
    """Not against today. A fixed extract is not stale for being old."""
    latest = warehouse.execute("SELECT MAX(date) FROM dim_calendar").fetchone()[0]
    assert reconciliation.as_of.date() == latest
