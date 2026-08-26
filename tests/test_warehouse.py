"""The warehouse builds from data/raw, and it builds honestly.

What "honestly" means here is narrow and checkable: the loaders copy the
raw rows in without repairing them. The B2B column arrives, the returns
column arrives, the store_codes that resolve to nothing arrive, and the
weekly marketing feed stays weekly. Everything a later stage needs in
order to *notice* a problem has to survive the load.

The reconciliation numbers themselves are `tests/test_reconciliation.py`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from engine.warehouse.load import (
    RAW_DIR,
    WarehouseError,
    build_warehouse,
    row_counts,
    table_names,
    view_names,
)

ROOT = Path(__file__).resolve().parents[1]

#: Every table the P4 brief names, plus the source tables the generator
#: emits that the brief's list does not enumerate. A warehouse missing one
#: of the latter is not usable by P5, so they are asserted too.
REQUIRED_TABLES = {
    # dimensions
    "dim_store",
    "dim_sku",
    "dim_calendar",
    "dim_store_xref",
    "dim_user",
    "dim_festival_window",
    # facts
    "fact_sales_daily",
    "fact_inventory_snapshot",
    "fact_footfall_daily",
    "fact_marketing_spend_weekly",
    # documents
    "doc_store_notes",
    "doc_tickets",
    "doc_reviews",
    # external
    "ext_competitor_news",
    "ext_weather_daily",
    # operational registers
    "restatement_register",
    # case artefacts
    "case_registry",
    "case_hypothesis",
    "evidence",
    "test_result",
    "recommendation",
    "feedback_event",
    "case_outcome",
    "calibration_ledger",
    "telemetry_event",
    "telemetry_request",
    "audit_log",
    "data_gap_register",
    "kpi_definition_log",
    "llm_cache",
}

#: Loaded because P5 needs them, though the brief's table list omits them.
SUPPORTING_TABLES = {
    "fact_sales_daily_sku",
    "fact_bill_lines",
    "fact_feed_status",
    "fact_qcomm_weekly",
}

#: Artefact tables. The LOAD creates them and leaves them empty — nothing
#: in data/raw belongs in any of them. Two are filled during P4 itself, by
#: an action rather than by the load:
#:
#:   audit_log          engine/db.py::execute_governed, on every query
#:                      -> tests/test_governed_access.py
#:   data_gap_register  engine/warehouse/reconcile.py::write_gaps
#:                      -> tests/test_reconciliation.py
#:   kpi_definition_log engine/validate/checks.py, on the first run of a KPI
#:                      -> tests/test_validate.py
#:
#: The rest wait for a case to produce them.
EMPTY_AFTER_LOAD = {
    "case_registry",
    "case_hypothesis",
    "evidence",
    "test_result",
    "recommendation",
    "feedback_event",
    "case_outcome",
    "calibration_ledger",
    "telemetry_event",
    "telemetry_request",
    "audit_log",
    "data_gap_register",
    # Written by engine/validate when Gate 1 first runs a KPI.
    "kpi_definition_log",
    # Written by llm/classify.py the first time a document is tagged.
    "llm_cache",
}


# --- structure --------------------------------------------------------------


def test_every_required_table_exists(warehouse):
    missing = REQUIRED_TABLES - set(table_names(warehouse))
    assert not missing, f"warehouse is missing tables: {sorted(missing)}"


def test_supporting_source_tables_are_loaded_too(warehouse):
    missing = SUPPORTING_TABLES - set(table_names(warehouse))
    assert not missing, f"source tables not loaded: {sorted(missing)}"


def test_conformance_views_exist(warehouse):
    views = set(view_names(warehouse))
    assert {
        "v_sales_daily_keyed",
        "v_sales_daily_conformed",
        "v_sales_daily_quarantined",
        "v_revenue_definitions",
        "v_marketing_spend_daily",
        "v_calendar_alignment",
    } <= views


@pytest.mark.parametrize("table", sorted(REQUIRED_TABLES - EMPTY_AFTER_LOAD))
def test_every_source_table_carries_rows(warehouse, table):
    """A silently empty table passes every other test in this file."""
    count = warehouse.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    assert count > 0, f"{table} loaded no rows"


@pytest.mark.parametrize("table", sorted(EMPTY_AFTER_LOAD))
def test_artefact_tables_are_created_by_the_load_and_left_empty(tmp_path, table):
    """Built fresh, so a test that wrote to one earlier cannot mask this.

    The session `warehouse` fixture is shared and the audit and gap tests
    write to it; this builds a throwaway warehouse with the schema only.
    """
    from engine.db import IN_MEMORY, connect
    from engine.warehouse.load import create_schema

    connection = connect(IN_MEMORY)
    try:
        create_schema(connection)
        count = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        assert count == 0
    finally:
        connection.close()


# --- the load matches the raw files ----------------------------------------

RAW_TO_TABLE = [
    ("dim/dim_store.csv", "dim_store"),
    ("dim/dim_sku.csv", "dim_sku"),
    ("dim/dim_store_xref.csv", "dim_store_xref"),
    ("dim/dim_user.csv", "dim_user"),
    ("context/calendar.csv", "dim_calendar"),
    ("context/festival_windows.csv", "dim_festival_window"),
    ("pos_erp/sales_daily.csv", "fact_sales_daily"),
    ("pos_erp/feed_status.csv", "fact_feed_status"),
    ("store_ops/footfall_daily.csv", "fact_footfall_daily"),
    ("store_ops/store_notes.csv", "doc_store_notes"),
    ("store_ops/tickets.csv", "doc_tickets"),
    ("store_ops/reviews.csv", "doc_reviews"),
    ("pos_erp/restatements.csv", "restatement_register"),
    ("context/marketing_spend.csv", "fact_marketing_spend_weekly"),
    ("context/competitor_news.csv", "ext_competitor_news"),
    ("context/weather_daily.csv", "ext_weather_daily"),
]


@pytest.mark.parametrize("relative,table", RAW_TO_TABLE, ids=lambda v: str(v))
def test_row_count_matches_the_raw_file(warehouse, relative, table):
    """Every raw row lands. Nothing is filtered on the way in."""
    with (RAW_DIR / relative).open(encoding="utf-8", newline="") as handle:
        raw_rows = sum(1 for _ in csv.reader(handle)) - 1
    loaded = warehouse.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    assert loaded == raw_rows


# --- what the load must NOT do ---------------------------------------------


def test_b2b_and_returns_survive_the_load(warehouse):
    """Both are excluded by the KPI contract and both must still be here.

    A loader that applied the contract's exclusions would make the
    definition conflict uncomputable, which is the usual way this gap
    disappears.
    """
    row = warehouse.execute(
        "SELECT SUM(b2b_net_revenue_inr), SUM(returns_amount_inr), "
        "SUM(transfer_amount_inr) FROM fact_sales_daily"
    ).fetchone()
    assert all(value > 0 for value in row)


def test_unresolvable_store_codes_are_loaded_not_dropped(warehouse):
    quarantined = warehouse.execute(
        "SELECT COUNT(*) FROM v_sales_daily_quarantined"
    ).fetchone()[0]
    assert quarantined > 0, "the unmapped rows were dropped on load, not quarantined"


def test_marketing_spend_stays_weekly(warehouse):
    """The daily figure is a view, not a column. The raw grain is untouched."""
    columns = {
        row[0]
        for row in warehouse.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'fact_marketing_spend_weekly'"
        ).fetchall()
    }
    assert "week_start" in columns
    assert not {"spend_date", "allocated_spend_inr"} & columns


def test_footfall_nulls_survive_the_load(warehouse):
    """109 of 140 West stores have no counter. The nulls are the evidence."""
    nulls = warehouse.execute(
        "SELECT COUNT(*) FROM fact_footfall_daily WHERE footfall IS NULL"
    ).fetchone()[0]
    assert nulls > 0
    contradictions = warehouse.execute(
        "SELECT COUNT(*) FROM fact_footfall_daily "
        "WHERE counter_installed AND footfall IS NULL"
    ).fetchone()[0]
    assert contradictions == 0


def test_the_netting_identity_holds_in_the_warehouse(warehouse):
    """gross - returns - discount - tax = net, exactly, on every row."""
    worst = warehouse.execute(
        "SELECT MAX(ABS(gross_amount_inr - returns_amount_inr - discount_amount_inr "
        "- tax_amount_inr - net_revenue_inr)) FROM fact_sales_daily"
    ).fetchone()[0]
    assert worst == pytest.approx(0.0, abs=0.01)


def test_dim_store_does_not_carry_the_pos_key(warehouse):
    """The xref is the only bridge; a second copy here could not go stale."""
    columns = {
        row[0]
        for row in warehouse.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'dim_store'"
        ).fetchall()
    }
    assert "outlet_id" in columns
    assert "store_code" not in columns


def test_personas_in_dim_user_are_known_to_every_kpi_contract(warehouse, layer):
    """A persona no contract knows about cannot be resolved at query time."""
    personas = {
        row[0] for row in warehouse.execute("SELECT DISTINCT persona FROM dim_user").fetchall()
    }
    for name, kpi in layer.kpis.items():
        unknown = personas - set(kpi.access_policy.personas)
        assert not unknown, f"{name} has no access policy for {sorted(unknown)}"


def test_scoped_personas_carry_the_attribute_their_predicate_needs(warehouse):
    """A regional manager with no region would silently see the estate."""
    rows = warehouse.execute(
        "SELECT user_id, persona, region, store_id FROM dim_user "
        "WHERE persona = 'regional_manager' AND region IS NULL"
    ).fetchall()
    assert not rows, f"regional managers with no region: {rows}"
    rows = warehouse.execute(
        "SELECT user_id FROM dim_user WHERE persona = 'store_manager' AND store_id IS NULL"
    ).fetchall()
    assert not rows, f"store managers with no store: {rows}"


# --- build behaviour --------------------------------------------------------


def test_row_counts_are_read_back_not_assumed(warehouse):
    counts = row_counts(warehouse)
    assert counts["dim_store"] == 412
    assert set(counts) == set(table_names(warehouse))


def test_a_missing_raw_directory_is_a_clear_error(tmp_path):
    from engine.db import IN_MEMORY, connect

    connection = connect(IN_MEMORY)
    try:
        with pytest.raises(WarehouseError, match="raw directory not found"):
            build_warehouse(raw_dir=tmp_path / "nowhere", connection=connection)
    finally:
        connection.close()


def test_schema_is_ddl_not_python():
    """Rule 2 again: the shape of the warehouse lives in SQL, readable."""
    schema = (ROOT / "engine" / "warehouse" / "schema.sql").read_text(encoding="utf-8")
    assert schema.count("CREATE TABLE IF NOT EXISTS") >= len(REQUIRED_TABLES)
