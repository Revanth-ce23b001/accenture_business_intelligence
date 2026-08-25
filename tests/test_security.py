"""P13 — the access policy, the trust boundary, and the audit log.

The accept criteria this file owns:

    Rhea querying South returns 0 rows, rows_released_to_llm = 0, and an
        audit row is written
    the same question as CCO returns all four regions
    masked columns are provably absent from the LLM payload, asserted on
        the serialised request
    a static test asserts no bypass path exists

THE THIRD ONE ONLY MEANS SOMETHING BECAUSE THE COLUMNS ARE REAL.
`dim_store.staff_cost` and `dim_store.staff_id` exist in the warehouse, so
a query can select them, the mask has to fire, and the assertion that they
are absent from the prompt is a test rather than a tautology. Six of the
other declared masks still name columns that do not exist anywhere; that
gap is pinned at the bottom of this file rather than left to be
discovered by whoever next writes a test against one of them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from engine.db import execute_governed
from llm.provider import LLMMessage, LLMRequest
from security.audit import AuditRecord, statement_hash
from security.policy import PolicyError, User, resolve_policy
from security.redaction import (
    RELEASE_PURPOSE_PREFIX,
    RedactionError,
    build_payload,
    guard_serialised,
    release_to_llm,
)

ROOT = Path(__file__).resolve().parents[1]

#: The question. Selects the two masked columns on purpose — a query that
#: never asks for them cannot demonstrate that they were withheld.
STAFF_QUERY = """
SELECT region, store_id, staff_headcount, staff_cost, staff_id
FROM dim_store
"""

SOUTH_QUERY = STAFF_QUERY + "WHERE region = 'South'"

REGION_QUERY = """
SELECT region, COUNT(*) AS stores, SUM(staff_cost) AS staff_cost
FROM dim_store
GROUP BY region
"""


@pytest.fixture(scope="module")
def people(warehouse) -> dict[str, User]:
    """The three P13 names, read from `dim_user` rather than constructed.

    Built from the warehouse so the test cannot pass against a user the
    seeded roster does not actually contain.
    """
    rows = warehouse.execute(
        "SELECT user_id, persona, region, store_id, display_name, email "
        "FROM dim_user ORDER BY user_id"
    ).fetchall()
    by_name = {}
    for user_id, persona, region, store_id, display_name, email in rows:
        by_name[display_name.split()[0]] = User(
            user_id=user_id,
            persona=persona,
            region=region,
            store_id=store_id,
            display_name=display_name,
            email=email,
        )
    return by_name


@pytest.fixture
def audit(warehouse):
    """A clean audit log for one test. It is a session-scoped table."""
    warehouse.execute("DELETE FROM audit_log")
    yield warehouse
    warehouse.execute("DELETE FROM audit_log")


def audit_rows(connection):
    return connection.execute(
        "SELECT user_id, persona, kpi, row_predicate, rows_returned, rows_filtered, "
        "columns_masked, rows_released_to_llm, purpose FROM audit_log "
        "ORDER BY occurred_at, purpose"
    ).fetchall()


# ===========================================================================
# The roster
# ===========================================================================


def test_the_three_named_users_exist_with_the_right_personas(people):
    assert people["Rhea"].persona == "regional_head_west"
    assert people["Rhea"].region == "West"
    assert people["Vikram"].persona == "cco"
    assert people["Vikram"].region is None
    assert people["Arjun"].persona == "analyst"
    assert people["Arjun"].region is None


def test_each_persona_masks_what_the_brief_says(layer):
    personas = layer.kpis["net_revenue"].access_policy.personas
    assert personas["regional_head_west"].masked_columns == ["staff_cost", "staff_id"]
    assert personas["cco"].masked_columns == ["staff_id"]
    assert personas["analyst"].masked_columns == []


def test_the_regional_head_is_scoped_and_the_other_two_are_not(layer):
    personas = layer.kpis["net_revenue"].access_policy.personas
    assert personas["regional_head_west"].row_predicate == "region = :user_region"
    assert personas["cco"].row_predicate == "TRUE"
    assert personas["analyst"].row_predicate == "TRUE"


def test_every_kpi_contract_knows_all_three(layer):
    """A persona one contract has never heard of is a hard error at query time."""
    for name, kpi in layer.kpis.items():
        for persona in ("regional_head_west", "cco", "analyst"):
            assert persona in kpi.access_policy.personas, f"{name} lacks {persona}"


def test_a_regional_head_with_no_region_is_refused_not_widened(layer):
    """The predicate binds the user's own region. No region, no query."""
    stateless = User(user_id="U999", persona="regional_head_west")
    with pytest.raises(PolicyError, match="Refusing rather than widening"):
        resolve_policy(
            layer.kpis["net_revenue"], stateless, layer.warehouse.governance
        )


# ===========================================================================
# Accept criterion 1 — Rhea querying South
# ===========================================================================


def test_rhea_querying_south_returns_nothing(audit, people):
    """Accept criterion.

    She asks a question about a region that is not hers. The predicate is
    hers, so every row the question selects fails it: 0 returned, 96
    withheld, and the withheld count is REPORTED rather than hidden — she
    is told there were rows she could not see, which is the honest
    failure mode.
    """
    rows, filtered, masked = execute_governed(
        people["Rhea"], "net_revenue", SOUTH_QUERY, connection=audit
    )
    assert rows == ()
    assert filtered == 96
    assert masked == ("staff_cost", "staff_id")


def test_rhea_querying_south_releases_nothing_and_is_audited(audit, people):
    """Accept criterion: rows_released_to_llm = 0, and an audit row is written."""
    execute_governed(people["Rhea"], "net_revenue", SOUTH_QUERY, connection=audit)

    recorded = audit_rows(audit)
    assert len(recorded) == 1
    (user_id, persona, kpi, predicate, returned, filtered, masked, released, purpose) = (
        recorded[0]
    )
    assert user_id == people["Rhea"].user_id
    assert persona == "regional_head_west"
    assert kpi == "net_revenue"
    assert predicate == "region = :user_region"
    assert returned == 0
    assert filtered == 96
    assert masked == "staff_cost,staff_id"
    assert released == 0
    assert purpose is None


def test_rhea_sees_her_own_region(audit, people):
    """The complement. A refusal that refused everything would prove nothing."""
    rows, _filtered, _masked = execute_governed(
        people["Rhea"], "net_revenue", STAFF_QUERY, connection=audit
    )
    assert len(rows) == 140
    assert {row["region"] for row in rows} == {"West"}


# ===========================================================================
# Accept criterion 2 — the same question as CCO
# ===========================================================================


def test_the_same_question_as_cco_returns_all_four_regions(audit, people):
    """Accept criterion.

    One question, two readers, two answers, and neither of them is wrong.
    """
    rhea, _f, _m = execute_governed(
        people["Rhea"], "net_revenue", REGION_QUERY, connection=audit
    )
    vikram, _f, _m = execute_governed(
        people["Vikram"], "net_revenue", REGION_QUERY, connection=audit
    )
    assert {row["region"] for row in rhea} == {"West"}
    assert {row["region"] for row in vikram} == {"North", "South", "East", "West"}


def test_the_cco_still_does_not_see_the_roster_key(audit, people):
    """Unscoped on rows is not unmasked on columns."""
    rows, _filtered, masked = execute_governed(
        people["Vikram"], "net_revenue", STAFF_QUERY, connection=audit
    )
    assert len(rows) == 412
    assert masked == ("staff_id",)
    assert all("staff_id" not in row for row in rows)
    # And the column the CCO IS allowed is still there.
    assert all("staff_cost" in row for row in rows)


def test_the_analyst_sees_everything(audit, people):
    """Every region, every column. The investigation needs both."""
    rows, filtered, masked = execute_governed(
        people["Arjun"], "net_revenue", STAFF_QUERY, connection=audit
    )
    assert len(rows) == 412
    assert filtered == 0
    assert masked == ()
    assert all("staff_cost" in row and "staff_id" in row for row in rows)


# ===========================================================================
# Accept criterion 3 — provably absent from the LLM payload
# ===========================================================================


def build_prompt(payload) -> LLMRequest:
    """A real request, built the way a real caller would build one."""
    return LLMRequest.for_task(
        "narrate",
        system="You summarise a case.",
        messages=(LLMMessage(role="user", content=payload.as_json()),),
        max_tokens=256,
    )


def test_masked_columns_are_absent_from_the_serialised_request(audit, people, layer):
    """Accept criterion, asserted on the bytes that would go over the wire.

    Not on the dicts, and not on a claim that the mask ran. The check is a
    search of the serialised request for the column NAMES, because that is
    the one assertion that does not care which code path put them there.
    """
    policy = resolve_policy(
        layer.kpis["net_revenue"], people["Rhea"], layer.warehouse.governance
    )
    rows, _filtered, _masked = execute_governed(
        people["Rhea"], "net_revenue", STAFF_QUERY, connection=audit
    )
    payload = build_payload(rows, policy)
    request = build_prompt(payload)

    serialised = request.model_dump_json()
    assert "staff_cost" not in serialised
    assert "staff_id" not in serialised
    # And the columns she IS entitled to did make it, so the assertion is
    # not passing because the payload is empty.
    assert "staff_headcount" in serialised
    assert payload.row_count == 140

    guard_serialised(serialised, policy)


def test_the_guard_catches_a_masked_column_arriving_by_another_route(
    people, layer
):
    """The threat is not a broken mask. It is a second route to the model.

    These rows never went through `execute_governed` — they were built by
    hand, the way a caller assembling a payload from two results would
    build them. The structural lock catches it.
    """
    policy = resolve_policy(
        layer.kpis["net_revenue"], people["Rhea"], layer.warehouse.governance
    )
    smuggled = [{"region": "West", "staff_cost": 140000}]
    with pytest.raises(RedactionError, match="staff_cost"):
        build_payload(smuggled, policy)


def test_the_byte_guard_catches_what_the_structure_guard_cannot(people, layer):
    """A masked name in prose, not in a field. Same leak, different shape."""
    policy = resolve_policy(
        layer.kpis["net_revenue"], people["Rhea"], layer.warehouse.governance
    )
    prompt = "The staff_cost for this store is high."
    with pytest.raises(RedactionError, match="staff_cost"):
        guard_serialised(prompt, policy)


def test_the_guard_does_not_trip_on_a_longer_name(people, layer):
    """Word-bounded. `staff_costing_model` is not `staff_cost`."""
    policy = resolve_policy(
        layer.kpis["net_revenue"], people["Rhea"], layer.warehouse.governance
    )
    guard_serialised("the staff_costing_model was reviewed", policy)


def test_the_analyst_payload_carries_the_columns_the_others_lose(
    audit, people, layer
):
    """Nothing masked means nothing withheld, and the guard agrees."""
    policy = resolve_policy(
        layer.kpis["net_revenue"], people["Arjun"], layer.warehouse.governance
    )
    rows, _f, _m = execute_governed(
        people["Arjun"], "net_revenue", STAFF_QUERY, connection=audit
    )
    payload = build_payload(rows, policy)
    serialised = build_prompt(payload).model_dump_json()
    assert "staff_cost" in serialised
    guard_serialised(serialised, policy)  # masks nothing, so nothing to refuse


def test_the_visibility_column_never_reaches_a_payload(audit, people, layer):
    policy = resolve_policy(
        layer.kpis["net_revenue"], people["Vikram"], layer.warehouse.governance
    )
    flag = layer.warehouse.governance.visibility_column
    with pytest.raises(RedactionError, match=flag):
        build_payload([{"region": "West", flag: True}], policy, visibility_column=flag)


# ===========================================================================
# The audit log
# ===========================================================================


def test_a_release_is_a_separate_row_with_its_own_count(audit, people, layer):
    """A read releases nothing; a release says how much left."""
    policy = resolve_policy(
        layer.kpis["net_revenue"], people["Vikram"], layer.warehouse.governance
    )
    rows, _f, _m = execute_governed(
        people["Vikram"], "net_revenue", REGION_QUERY, connection=audit
    )
    payload = build_payload(rows, policy)
    release_to_llm(audit, people["Vikram"], policy, payload, purpose="narrate")

    recorded = audit_rows(audit)
    assert len(recorded) == 2
    read = next(row for row in recorded if row[-1] is None)
    released = next(row for row in recorded if row[-1] is not None)
    assert read[7] == 0
    assert released[7] == 4
    assert released[-1] == f"{RELEASE_PURPOSE_PREFIX}.narrate"


def test_the_release_count_is_what_left_not_what_was_read(audit, people, layer):
    """Read 412, send 3, and the log says 3."""
    policy = resolve_policy(
        layer.kpis["net_revenue"], people["Arjun"], layer.warehouse.governance
    )
    rows, _f, _m = execute_governed(
        people["Arjun"], "net_revenue", STAFF_QUERY, connection=audit
    )
    payload = build_payload(rows[:3], policy)
    record = release_to_llm(audit, people["Arjun"], policy, payload, purpose="classify")
    assert len(rows) == 412
    assert record.rows_released_to_llm == 3
    assert record.released


def test_every_governed_call_is_audited_even_when_it_returns_nothing(audit, people):
    for user in ("Rhea", "Vikram", "Arjun"):
        execute_governed(people[user], "net_revenue", SOUTH_QUERY, connection=audit)
    assert len(audit_rows(audit)) == 3


def test_the_audit_stores_a_hash_not_the_statement(audit, people):
    execute_governed(people["Rhea"], "net_revenue", SOUTH_QUERY, connection=audit)
    stored = audit.execute("SELECT statement_hash FROM audit_log").fetchone()[0]
    assert stored == statement_hash(SOUTH_QUERY)
    assert "dim_store" not in stored


def test_an_audit_record_round_trips_its_row():
    record = AuditRecord.for_query(
        user_id="U009",
        persona="regional_head_west",
        kpi="net_revenue",
        sql="SELECT 1",
        row_predicate="region = :user_region",
        rows_returned=0,
        rows_filtered=96,
        columns_masked=("staff_cost", "staff_id"),
        rows_released_to_llm=0,
    )
    row = record.as_row()
    assert row[7] == 0  # rows_returned
    assert row[8] == 96  # rows_filtered
    assert row[9] == "staff_cost,staff_id"
    assert row[10] == 0  # rows_released_to_llm


# ===========================================================================
# Accept criterion 4 — no bypass path
# ===========================================================================


def python_sources() -> list[Path]:
    return sorted(
        path
        for directory in ("engine", "llm", "security", "api", "telemetry", "data")
        for path in (ROOT / directory).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _calls_named(path: Path, name: str) -> list[int]:
    """Lines where `name` is called or referenced as a bare name."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == name
    ]


def test_there_are_sources_to_scan():
    assert len(python_sources()) > 20


def test_only_two_modules_write_to_the_audit_log():
    """`AUDIT_INSERT` is the only statement that appends to the log.

    engine/db.py owns it and writes the READ rows; security/redaction.py
    writes the RELEASE rows. A third writer would mean a row nobody could
    attribute to a policy decision.
    """
    allowed = {
        ROOT / "engine" / "db.py",
        ROOT / "security" / "redaction.py",
    }
    offenders = {
        str(path.relative_to(ROOT)): _calls_named(path, "AUDIT_INSERT")
        for path in python_sources()
        if path not in allowed and _calls_named(path, "AUDIT_INSERT")
    }
    assert not offenders, f"only engine/db.py and security/redaction.py may write audit rows: {offenders}"


def test_the_two_allowed_writers_do_write():
    """The complement, so the scan cannot pass by finding nothing."""
    assert _calls_named(ROOT / "engine" / "db.py", "AUDIT_INSERT")
    assert _calls_named(ROOT / "security" / "redaction.py", "AUDIT_INSERT")


def test_no_module_builds_a_policy_without_the_resolver():
    """`AppliedPolicy` is constructed in exactly one place.

    Anything that could build its own would be choosing its own row
    predicate and its own mask list, which is the definition of a bypass.
    """
    offenders = {
        str(path.relative_to(ROOT)): _calls_named(path, "AppliedPolicy")
        for path in python_sources()
        if path != ROOT / "security" / "policy.py" and _calls_named(path, "AppliedPolicy")
    }
    # Type annotations are fine; construction is not. Filter to calls.
    real = {}
    for name, _lines in offenders.items():
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        constructed = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AppliedPolicy"
        ]
        if constructed:
            real[name] = constructed
    assert not real, f"AppliedPolicy is constructed outside security/policy.py: {real}"


def test_security_never_reaches_the_warehouse_on_its_own():
    """policy.py decides; it does not read. The split is what makes it testable.

    `resolve_policy` takes a contract and a user and returns a predicate.
    It has no connection, so there is no arrangement of its arguments that
    returns data — which is why an access decision can be unit-tested
    without a database at all.
    """
    source = (ROOT / "security" / "policy.py").read_text(encoding="utf-8")
    assert "duckdb" not in source
    assert "execute" not in source


def test_the_engine_never_masks_by_hand():
    """Masking happens in one function. A second implementation would drift."""
    offenders = []
    for path in python_sources():
        if path in {ROOT / "engine" / "db.py", ROOT / "security" / "redaction.py"}:
            continue
        source = path.read_text(encoding="utf-8")
        if "masked_columns" in source and "policy" not in source.lower():
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"masked_columns handled outside the policy path: {offenders}"


# ===========================================================================
# REGISTRY GAP — masks over columns that do not exist
# ===========================================================================


DECLARED_BUT_ABSENT = {
    "cost_price",
    "courier_cost",
    "margin_pct",
    "rider_id",
    "supplier_cost",
    "supplier_name",
}

#: Columns that would carry personal data if they ever existed. None of
#: them does, and after P13 none of them is masked either — emptying the
#: analyst's list per the brief removed the only declaration of
#: `customer_phone` and `customer_email` anywhere in the semantic layer.
#: That is inert while the columns are absent and a real hole the moment
#: one appears, so the tripwire below watches for exactly that.
CUSTOMER_PII = {"customer_phone", "customer_email", "customer_name", "customer_id"}


def test_the_two_masks_p13_names_are_real_columns(warehouse):
    """Without this, accept criterion 3 is a tautology.

    A mask over a column that does not exist never fires, so asserting the
    column is absent from a payload proves only that it was never there.
    P13 makes `staff_cost` and `staff_id` real for exactly this reason.
    """
    columns = {
        row[0]
        for row in warehouse.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'dim_store'"
        ).fetchall()
    }
    assert {"staff_cost", "staff_id"} <= columns


def test_the_other_declared_masks_do_not_reproduce(warehouse, layer):
    """GAP — eight declared masks name columns the warehouse does not hold.

    `customer_phone`, `margin_pct`, `rider_id` and five others are masked
    by one persona or another in the KPI contracts, and no table has them.
    Each is a mask that cannot fire and a promise nothing keeps.

    Not fixed here: making them real means adding customer PII, supplier
    terms and courier data to the generator, which is a data-model change
    several times the size of P13. Pinned instead, so that ADDING a
    phantom fails this test and MAKING ONE REAL also fails it — either way
    the list has to be updated deliberately.

    Two of them left the file entirely in P13. `customer_phone` and
    `customer_email` were declared only by `analyst`, whose mask list the
    brief empties, so the semantic layer no longer mentions them at all.
    Inert today; the tripwire below is what makes it stay that way.
    """
    held = {
        row[0]
        for row in warehouse.execute(
            "SELECT DISTINCT column_name FROM information_schema.columns "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }
    declared = {
        column
        for kpi in layer.kpis.values()
        for persona in kpi.access_policy.personas.values()
        for column in persona.masked_columns
    }
    phantom = {column for column in declared if column not in held}
    assert phantom == DECLARED_BUT_ABSENT, (
        "the set of masks over non-existent columns changed. If one became "
        "real, remove it from DECLARED_BUT_ABSENT. If a new one appeared, it "
        f"is a mask that will never fire. Now: {sorted(phantom)}"
    )


def test_customer_pii_is_either_absent_or_masked(warehouse, layer):
    """The tripwire the analyst change installs.

    P13 empties the analyst's mask list because the brief says so, and
    `customer_phone` and `customer_email` were declared nowhere else, so
    the semantic layer no longer mentions customer PII at all. That is
    harmless while no such column exists and a real hole the first day one
    does — an unscoped, unmasked persona reading a table with a phone
    number in it.

    So the invariant is stated as a property rather than a promise: for
    every column that would carry personal data, either the warehouse does
    not have it, or every unscoped persona masks it. Adding
    `customer_phone` to a table fails this test until somebody decides who
    is allowed to see it, which is the decision that should not be made by
    accident.
    """
    held = {
        row[0]
        for row in warehouse.execute(
            "SELECT DISTINCT column_name FROM information_schema.columns "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }
    present = CUSTOMER_PII & held
    if not present:
        return  # nothing to protect yet — the state P13 leaves it in

    for name, kpi in layer.kpis.items():
        for persona, rules in kpi.access_policy.personas.items():
            if rules.row_predicate.strip().upper() != "TRUE":
                continue
            unmasked = present - set(rules.masked_columns)
            assert not unmasked, (
                f"{name}/{persona} reads every row and does not mask "
                f"{sorted(unmasked)}, which now exist in the warehouse"
            )


def test_no_persona_masks_a_column_it_cannot_see_anyway(layer):
    """A mask on a persona with no rows would be theatre.

    Every persona that masks something must also be able to return rows,
    or the mask is protecting nothing from nobody.
    """
    for name, kpi in layer.kpis.items():
        for persona, rules in kpi.access_policy.personas.items():
            if rules.masked_columns:
                assert rules.row_predicate, f"{name}/{persona}"
