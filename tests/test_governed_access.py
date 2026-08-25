"""Rule 5: authorization happens before retrieval, and there is one door.

Two things are checked here.

The first is static, and it is the one that matters most: no module
outside `engine/db.py` calls `duckdb.connect`. A governed query path is
worth nothing if any module can open its own handle, and the only way to
know that has not happened is to look at every file.

The second is behavioural: `execute_governed` filters rows, drops masked
columns, counts what it withheld, refuses what it cannot resolve, and
writes an audit row every time.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from engine.db import GovernanceError, execute_governed, to_duckdb_params
from security.policy import PolicyError, User, bind_names, resolve_policy

ROOT = Path(__file__).resolve().parents[1]

#: The one module allowed to open a connection.
CONNECTION_OWNER = ROOT / "engine" / "db.py"

#: Pruned from the walk, not filtered out of it. `.git` and `data/raw`
#: hold hundreds of megabytes between them and neither holds Python.
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "raw",
    "frontend",
}

#: A store query that exercises both halves of the policy: `region` is
#: what a regional manager's predicate binds to, and `staff_headcount` is
#: masked for the scoped personas.
STORE_QUERY = "SELECT store_id, region, city, staff_headcount FROM dim_store"


def python_sources() -> list[Path]:
    """Every Python file in the repository, skipping what cannot hold one."""
    found: list[Path] = []
    for directory, subdirectories, files in os.walk(ROOT):
        subdirectories[:] = [name for name in subdirectories if name not in SKIP_DIRS]
        found.extend(
            Path(directory) / name for name in files if name.endswith(".py")
        )
    return sorted(found)


# ===========================================================================
# The static rule
# ===========================================================================


def test_there_are_sources_to_scan():
    """Guard against the scan passing because it found nothing."""
    assert len(python_sources()) > 10


def _connect_calls(path: Path) -> list[int]:
    """Lines calling `duckdb.connect`, however duckdb was imported."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    aliases = {"duckdb"}
    direct: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "duckdb":
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "duckdb":
            for alias in node.names:
                if alias.name == "connect":
                    direct.add(alias.asname or alias.name)

    found: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "connect":
            if isinstance(func.value, ast.Name) and func.value.id in aliases:
                found.append(node.lineno)
        elif isinstance(func, ast.Name) and func.id in direct:
            found.append(node.lineno)
    return found


def test_engine_db_is_the_only_module_that_opens_a_connection():
    """CLAUDE.md rule 5, enforced across the whole repository."""
    offenders = {
        str(path.relative_to(ROOT)): _connect_calls(path)
        for path in python_sources()
        if path != CONNECTION_OWNER and _connect_calls(path)
    }
    assert not offenders, (
        "duckdb.connect may only be called in engine/db.py — every other module must go "
        f"through engine.db.connect / execute_governed. Found: {offenders}"
    )


def test_engine_db_does_open_one():
    """The complement: the owner must actually own it."""
    assert _connect_calls(CONNECTION_OWNER), "engine/db.py no longer opens the connection"


def test_no_module_imports_duckdb_to_query_the_warehouse_behind_the_policy():
    """Importing duckdb is fine; using it to open a second door is not.

    This is the same assertion as above stated the other way round, and it
    exists so the failure message names the file rather than a line number.
    """
    for path in python_sources():
        if path == CONNECTION_OWNER:
            continue
        assert not _connect_calls(path), f"{path.relative_to(ROOT)} opens its own connection"


# ===========================================================================
# Parameter rewriting
# ===========================================================================


def test_named_parameters_are_rewritten_for_duckdb():
    assert to_duckdb_params("region = :user_region") == "region = $user_region"


def test_casts_are_not_mistaken_for_parameters():
    """`::DATE` is a cast. Rewriting it would corrupt the query."""
    assert to_duckdb_params("txn_date::DATE >= :period_start") == (
        "txn_date::DATE >= $period_start"
    )
    assert bind_names("txn_date::DATE >= :period_start") == {"period_start"}


# ===========================================================================
# Row filtering
# ===========================================================================


@pytest.fixture
def users(warehouse):
    rows = warehouse.execute(
        "SELECT user_id, display_name, email, persona, region, store_id FROM dim_user"
    ).fetchall()
    columns = ["user_id", "display_name", "email", "persona", "region", "store_id"]
    return {
        row[0]: User.from_row(dict(zip(columns, row, strict=True)))
        for row in rows
    }


def by_persona(users, persona):
    return next(user for user in users.values() if user.persona == persona)


def test_an_unrestricted_persona_sees_everything(warehouse, users):
    rows, filtered, masked = execute_governed(
        by_persona(users, "cxo"), "net_revenue", STORE_QUERY, connection=warehouse
    )
    assert len(rows) == 412
    assert filtered == 0
    assert masked == ()


def test_a_regional_manager_sees_only_their_region(warehouse, users):
    user = next(
        u for u in users.values() if u.persona == "regional_manager" and u.region == "West"
    )
    rows, filtered, _masked = execute_governed(
        user, "net_revenue", STORE_QUERY, connection=warehouse
    )
    assert len(rows) == 140
    assert filtered == 412 - 140
    assert {row["region"] for row in rows} == {"West"}


def test_the_withheld_count_is_reported_not_hidden(warehouse, users):
    """140 rows and no count reads as 'that is the estate'. It is not."""
    user = next(
        u for u in users.values() if u.persona == "regional_manager" and u.region == "East"
    )
    rows, filtered, _ = execute_governed(
        user, "net_revenue", STORE_QUERY, connection=warehouse
    )
    assert len(rows) + filtered == 412


def test_a_store_manager_sees_one_store(warehouse, users):
    user = by_persona(users, "store_manager")
    rows, filtered, _ = execute_governed(
        user, "net_revenue", STORE_QUERY, connection=warehouse
    )
    assert len(rows) == 1
    assert rows[0]["store_id"] == user.store_id
    assert filtered == 412 - 1


# ===========================================================================
# Column masking
# ===========================================================================


def test_masked_columns_are_dropped_from_the_result(warehouse, users, layer):
    user = by_persona(users, "regional_manager")
    rows, _filtered, masked = execute_governed(
        user, "net_revenue", STORE_QUERY, connection=warehouse
    )
    policy = layer.kpis["net_revenue"].access_policy.personas["regional_manager"]
    assert "staff_headcount" in policy.masked_columns
    assert masked == ("staff_headcount",)
    assert all("staff_headcount" not in row for row in rows)
    assert all("city" in row for row in rows)


def test_only_masked_columns_present_in_the_result_are_reported(warehouse, users, layer):
    """The CCO masks `staff_id`, which this query never selected.

    Repointed at the CCO in P13. It used to run as the analyst, whose mask
    list is now empty, so it would have passed because there was nothing
    to report rather than because nothing selected was masked — a test
    that holds for the wrong reason is worse than one that fails.
    """
    policy = layer.kpis["net_revenue"].access_policy.personas["cco"]
    assert policy.masked_columns == ["staff_id"]
    assert "staff_id" not in STORE_QUERY

    _rows, _filtered, masked = execute_governed(
        by_persona(users, "cco"), "net_revenue", STORE_QUERY, connection=warehouse
    )
    assert masked == ()


def test_the_visibility_column_never_reaches_the_caller(warehouse, users, layer):
    rows, _filtered, _masked = execute_governed(
        by_persona(users, "cxo"), "net_revenue", STORE_QUERY, connection=warehouse
    )
    flag = layer.warehouse.governance.visibility_column
    assert all(flag not in row for row in rows)


def test_the_callers_own_query_semantics_survive_the_wrapper(warehouse, users):
    """A GROUP BY must not be rewritten into something else."""
    rows, _filtered, _masked = execute_governed(
        by_persona(users, "cxo"),
        "net_revenue",
        "SELECT region, COUNT(*) AS stores FROM dim_store GROUP BY region",
        connection=warehouse,
    )
    assert {row["region"]: row["stores"] for row in rows} == {
        "North": 118,
        "South": 96,
        "East": 58,
        "West": 140,
    }


def test_caller_parameters_are_still_bound(warehouse, users):
    rows, _filtered, _masked = execute_governed(
        by_persona(users, "cxo"),
        "net_revenue",
        "SELECT store_id, region FROM dim_store WHERE region = :region",
        {"region": "East"},
        connection=warehouse,
    )
    assert len(rows) == 58


# ===========================================================================
# Refusals — all of them before anything is read
# ===========================================================================


def test_a_caller_cannot_supply_a_reserved_binding(warehouse, users):
    """The attack: overwrite your own row filter on the way in."""
    user = next(
        u for u in users.values() if u.persona == "regional_manager" and u.region == "East"
    )
    with pytest.raises(GovernanceError, match="reserved"):
        execute_governed(
            user,
            "net_revenue",
            STORE_QUERY,
            {"user_region": "West"},
            connection=warehouse,
        )


def test_an_unknown_kpi_is_refused(warehouse, users):
    with pytest.raises(GovernanceError, match="no KPI contract"):
        execute_governed(
            by_persona(users, "cxo"), "not_a_kpi", STORE_QUERY, connection=warehouse
        )


def test_a_persona_the_contract_does_not_know_is_refused(warehouse):
    with pytest.raises(GovernanceError, match="not defined in the access policy"):
        execute_governed(
            User(user_id="U999", persona="intern"),
            "net_revenue",
            STORE_QUERY,
            connection=warehouse,
        )


def test_a_scoped_persona_with_no_scope_is_refused_not_widened(warehouse):
    """The failure mode worth naming: a manager with no region seeing all."""
    with pytest.raises(GovernanceError, match="Refusing rather than widening"):
        execute_governed(
            User(user_id="U998", persona="regional_manager", region=None),
            "net_revenue",
            STORE_QUERY,
            connection=warehouse,
        )


def test_resolve_policy_raises_before_any_query_is_built(layer):
    kpi = layer.kpis["net_revenue"]
    governance = layer.warehouse.governance
    with pytest.raises(PolicyError):
        resolve_policy(kpi, User(user_id="U997", persona="nobody"), governance)


# ===========================================================================
# Audit
# ===========================================================================


def test_every_call_writes_an_audit_row(warehouse, users):
    warehouse.execute("DELETE FROM audit_log")
    user = next(
        u for u in users.values() if u.persona == "regional_manager" and u.region == "West"
    )
    execute_governed(user, "net_revenue", STORE_QUERY, connection=warehouse, purpose="P4 test")

    row = warehouse.execute(
        "SELECT user_id, persona, kpi, row_predicate, rows_returned, rows_filtered, "
        "columns_masked, purpose FROM audit_log"
    ).fetchall()
    assert len(row) == 1
    user_id, persona, kpi, predicate, returned, filtered, masked, purpose = row[0]
    assert user_id == user.user_id
    assert persona == "regional_manager"
    assert kpi == "net_revenue"
    assert ":user_region" in predicate
    assert returned == 140
    assert filtered == 412 - 140
    assert masked == "staff_headcount"
    assert purpose == "P4 test"


def test_the_audit_records_a_hash_not_the_statement(warehouse, users):
    warehouse.execute("DELETE FROM audit_log")
    execute_governed(
        by_persona(users, "cxo"), "net_revenue", STORE_QUERY, connection=warehouse
    )
    stored = warehouse.execute("SELECT statement_hash FROM audit_log").fetchone()[0]
    assert len(stored) == 64
    assert "dim_store" not in stored


def test_a_failing_query_is_audited_too(warehouse, users):
    """A query that errored still reached the warehouse, so it is recorded."""
    warehouse.execute("DELETE FROM audit_log")
    with pytest.raises(Exception):
        execute_governed(
            by_persona(users, "cxo"),
            "net_revenue",
            "SELECT * FROM no_such_table",
            connection=warehouse,
        )
    count = warehouse.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    assert count == 1


def test_a_refused_call_writes_no_audit_row(warehouse):
    """It never reached the warehouse. Recording it would overstate access."""
    warehouse.execute("DELETE FROM audit_log")
    with pytest.raises(GovernanceError):
        execute_governed(
            User(user_id="U996", persona="intern"),
            "net_revenue",
            STORE_QUERY,
            connection=warehouse,
        )
    assert warehouse.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0
