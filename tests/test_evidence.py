"""The evidence engine.

CLAUDE.md rule 3 — every number in the UI is clickable to its evidence, so
no bare float leaves the engine. This file checks the three things that
make that true rather than aspirational:

  * `engine/evidence.py` is the ONLY place an `Evidence` is constructed.
    A static scan enforces it, the same way P4's scan enforces that only
    `engine/db.py` opens a connection.
  * every persisted row carries a source system, a method, a data
    timestamp and a lineage — and the data timestamp is the DATA's, never
    the moment it was read.
  * the floor rule holds: a hypothesis whose evidence is dominated by
    weights below the floor cannot be promoted to EXPLAINED.
"""

from __future__ import annotations

import ast
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from engine.contracts import Evidence
from engine.evidence import (
    EvidenceError,
    EvidenceFactory,
    EvidenceLedger,
    as_row,
    can_promote_to_explained,
    lineage_json,
    load,
    persist,
    reliability_floor,
    reliability_profile,
    reliability_weight,
)
from engine.qualify import QualifyRequest, qualify
from engine.validate import ValidationRequest, validate
from security.policy import User

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"

#: The one module allowed to build an Evidence.
EVIDENCE_OWNER = ENGINE / "evidence.py"

ANALYST = User(user_id="U008", persona="analyst", display_name="Meera Joshi")

#: A fixed data timestamp. It is a DATE the warehouse holds, not a clock
#: reading, which is the whole point of the field.
DATA_AS_OF = datetime(2025, 11, 30, tzinfo=UTC)


@pytest.fixture
def factory(layer):
    return EvidenceFactory.for_stage("test", DATA_AS_OF, layer)


@pytest.fixture
def sandbox(warehouse):
    warehouse.execute("BEGIN TRANSACTION")
    try:
        yield warehouse
    finally:
        warehouse.execute("ROLLBACK")


def make(factory, key="sample", **overrides):
    payload = dict(
        kind="structured_query",
        label="West net revenue, November 2025",
        value=76.92,
        unit="INR_CR",
        source_system="pos",
        method="sql",
        description="SUM(net_revenue_inr) over fact_sales_daily for West",
        ref="semantic_layer/kpis/net_revenue.yaml",
        inputs=("fact_sales_daily",),
        statement="SELECT SUM(net_revenue_inr) FROM fact_sales_daily WHERE region = 'West'",
    )
    payload.update(overrides)
    return factory.emit(key, **payload)


# ===========================================================================
# The reliability weight table
# ===========================================================================

#: CLAUDE.md §"Reliability weights for evidence", transcribed. The engine
#: reads these from the semantic layer; this is the independent copy that
#: catches a YAML edit nobody meant.
CLAUDE_MD_WEIGHTS = {
    "structured_query": 0.95,
    "derived_estimate": 0.85,
    "corroborated_unstructured": 0.80,
    "ticket_aggregate": 0.75,
    "store_note": 0.55,
    "news_item": 0.45,
    "social_mention": 0.35,
}


@pytest.mark.parametrize("kind,weight", sorted(CLAUDE_MD_WEIGHTS.items()))
def test_the_weight_table_matches_claude_md(layer, kind, weight):
    assert reliability_weight(kind, layer) == pytest.approx(weight)


def test_the_floor_is_0_6(layer):
    """The line below which evidence cannot carry a hypothesis alone."""
    assert reliability_floor(layer) == pytest.approx(0.60)


def test_every_evidence_kind_has_a_weight(layer):
    from engine.contracts import EvidenceKind

    for kind in EvidenceKind.__args__:
        assert reliability_weight(kind, layer) > 0


def test_an_unknown_kind_is_refused(layer):
    with pytest.raises(EvidenceError, match="no reliability weight"):
        reliability_weight("rumour", layer)


def test_the_caller_cannot_choose_its_own_weight(factory):
    """A caller that could pick its weight could choose to be believed."""
    record = make(factory, kind="social_mention")
    assert record.reliability == pytest.approx(CLAUDE_MD_WEIGHTS["social_mention"])
    assert "reliability" not in EvidenceFactory.emit.__code__.co_varnames


# ===========================================================================
# The floor rule
# ===========================================================================


def test_text_alone_does_not_reach_a_verdict(factory, layer):
    """Three social mentions and a store note. All below the floor."""
    weak = [
        make(factory, f"weak-{index}", kind=kind, value=None, unit="count")
        for index, kind in enumerate(
            ("social_mention", "social_mention", "news_item", "store_note")
        )
    ]
    check = can_promote_to_explained(weak, layer)

    assert check.allowed is False
    assert check.trigger == "T6"
    assert check.profile.dominated is True
    assert "Text alone" in check.detail


def test_text_plus_a_matched_control_does(factory, layer):
    """CLAUDE.md's own sentence, made executable.

    A store note on its own is 0.55 and cannot carry a hypothesis. The
    same note beside a matched-control estimate can, because the estimate
    is worth more and the rule weighs rather than counts.
    """
    note_only = [make(factory, "note", kind="store_note", value=None, unit="count")]
    assert can_promote_to_explained(note_only, layer).allowed is False

    with_control = [
        make(factory, "note-2", kind="store_note", value=None, unit="count"),
        make(factory, "did", kind="derived_estimate"),
    ]
    check = can_promote_to_explained(with_control, layer)

    assert check.allowed is True
    assert check.trigger is None
    assert check.profile.dominated is False


def test_one_strong_record_does_not_outweigh_two_weak_ones(factory, layer):
    """Where the rule bites, pinned so a later tuning cannot move it quietly.

    A matched-control estimate (0.85) against a store note (0.55) and a
    news item (0.45): the weak pair carries 54% of the weight, over the
    50% limit, and the hypothesis stays unpromotable. One good number does
    not launder two bad ones.
    """
    records = [
        make(factory, "did-2", kind="derived_estimate"),
        make(factory, "note-3", kind="store_note", value=None, unit="count"),
        make(factory, "news-2", kind="news_item", value=None, unit="count"),
    ]
    check = can_promote_to_explained(records, layer)

    assert check.allowed is False
    assert check.trigger == "T6"
    assert check.profile.low_weight_share > check.profile.max_low_share


def test_the_floor_rule_weighs_rather_than_counts(factory, layer):
    """One structured query outvotes two social mentions.

    By count the weak evidence is the majority; by weight it is not, and
    weight is what the tiers exist to express.
    """
    records = [
        make(factory, "query", kind="structured_query"),
        make(factory, "mention-a", kind="social_mention", value=None, unit="count"),
        make(factory, "mention-b", kind="social_mention", value=None, unit="count"),
    ]
    profile = reliability_profile(records, layer)

    assert profile.low_count_share > 0.5
    assert profile.low_weight_share < 0.5
    assert profile.dominated is False


def test_a_hypothesis_with_no_evidence_is_dominated(layer):
    """Nothing supporting it is not the same as nothing against it."""
    check = can_promote_to_explained([], layer)
    assert check.allowed is False
    assert "no evidence at all" in check.profile.explain()


def test_the_floor_rule_carries_the_confidence_ceiling(factory, layer):
    """T6 fires and the cap comes with it, both from the semantic layer."""
    weak = [make(factory, "mention", kind="social_mention", value=None, unit="count")]
    check = can_promote_to_explained(weak, layer)
    cap = layer.adjudication.confidence.caps["low_reliability_dominated"]
    assert check.confidence_ceiling == pytest.approx(cap.ceiling)


def test_the_floor_and_the_share_come_from_the_semantic_layer(factory, layer):
    weak = [make(factory, "mention", kind="social_mention", value=None, unit="count")]
    profile = reliability_profile(weak, layer)
    trigger = layer.adjudication.triggers["T6"]
    assert profile.floor == pytest.approx(trigger.reliability_floor)
    assert profile.max_low_share == pytest.approx(trigger.max_low_reliability_share)


def test_corroborated_notes_clear_the_floor(factory, layer):
    """Twenty independent notes promote to a tier that can carry a case.

    CLAUDE.md's corroboration rule: a single store note is 0.55 and cannot;
    twenty of them corroborate to 0.80 and can. #2451's twenty-one treated
    stores filing notes is exactly this.
    """
    rule = layer.adjudication.reliability.corroboration
    assert reliability_weight(rule.promotes_from, layer) < reliability_floor(layer)
    assert reliability_weight(rule.promotes_to, layer) > reliability_floor(layer)

    promoted = [make(factory, "notes", kind=rule.promotes_to, value=None, unit="count")]
    assert can_promote_to_explained(promoted, layer).allowed is True


# ===========================================================================
# Minting: what the factory refuses
# ===========================================================================


def test_an_unknown_source_system_is_refused(factory):
    with pytest.raises(EvidenceError, match="not a known evidence origin"):
        make(factory, source_system="somebodys_spreadsheet")


def test_an_unknown_method_is_refused(factory):
    with pytest.raises(EvidenceError, match="not a declared method"):
        make(factory, method="vibes")


def test_every_record_gets_a_lineage(factory):
    record = make(factory)
    assert record.lineage
    assert record.lineage[0].description


def test_a_record_with_no_lineage_cannot_be_built():
    """Rule 3 at the contract level, not just at the factory."""
    from tests.conftest import make_evidence

    with pytest.raises(Exception):
        make_evidence(lineage=())


def test_the_model_may_not_be_the_source_of_a_number(factory):
    """Rule 1, enforced where the record is made."""
    with pytest.raises(Exception):
        make(factory, produced_by="model", value=76.92)


def test_the_model_may_still_carry_text(factory):
    record = make(
        factory, "narrative", produced_by="model", value=None, unit="text",
        source_system="semantic_layer", method="lookup",
    )
    assert record.produced_by == "model"
    assert record.value is None


def test_produced_by_defaults_to_code(factory):
    assert make(factory).produced_by == "code"


def test_ids_are_prefixed_by_the_stage(layer):
    validate_factory = EvidenceFactory.for_stage("validate", DATA_AS_OF, layer)
    qualify_factory = EvidenceFactory.for_stage("qualify", DATA_AS_OF, layer)
    assert make(validate_factory, "movement").evidence_id == "validate.movement"
    assert make(qualify_factory, "movement").evidence_id == "qualify.movement"


# ===========================================================================
# Verbatim SQL
# ===========================================================================


def test_the_statement_is_kept_verbatim_not_hashed(factory):
    """A reader who doubts a number should be able to run the query."""
    sql = "SELECT SUM(net_revenue_inr) FROM fact_sales_daily WHERE region = 'West'"
    record = make(factory, statement=sql)
    assert record.lineage[0].statement == sql


def test_the_statement_survives_the_json_round_trip(factory):
    import json

    sql = "SELECT 1 AS answer\nFROM fact_sales_daily\nWHERE region = 'West'"
    record = make(factory, statement=sql)
    restored = json.loads(lineage_json(record))
    assert restored[0]["statement"] == sql


def test_indentation_from_the_source_file_is_stripped(factory):
    """The query is kept verbatim. The Python indentation around it is not."""
    record = make(
        factory,
        statement="""
        SELECT region, SUM(net_revenue_inr)
        FROM fact_sales_daily
        GROUP BY 1
        """,
    )
    statement = record.lineage[0].statement
    assert statement.startswith("SELECT region")
    assert "\nFROM fact_sales_daily" in statement


def test_evidence_keeps_the_statement_where_audit_log_keeps_a_hash(sandbox):
    """Two tables, two answers, and both are deliberate.

    `audit_log` records a hash because it has different access rules from
    the table it describes. Evidence records the statement, because a
    number you cannot re-derive is not evidence.
    """
    sandbox.execute("DELETE FROM audit_log")
    result = validate(
        sandbox, ANALYST,
        ValidationRequest("net_revenue", "West", "monthly", "2025-11", "2025-10"),
    )
    hashes = sandbox.execute("SELECT DISTINCT statement_hash FROM audit_log").fetchall()
    assert hashes and all(len(row[0]) == 64 for row in hashes)

    statements = [
        step.statement
        for item in result.evidence
        for step in item.lineage
        if step.statement
    ]
    assert statements
    assert any("SELECT" in statement.upper() for statement in statements)


# ===========================================================================
# source_as_of is the DATA timestamp
# ===========================================================================


def test_source_as_of_is_the_data_timestamp_not_the_clock(factory):
    record = make(factory)
    assert record.source_as_of == DATA_AS_OF
    assert record.retrieved_at is not None
    assert record.retrieved_at > record.source_as_of


def test_the_two_timestamps_are_separate_fields(factory):
    """A system with one timestamp always ends up showing the wrong one."""
    record = make(factory)
    assert record.source_as_of != record.retrieved_at


def test_data_cannot_be_newer_than_the_moment_it_was_read(layer):
    impossible = EvidenceFactory(
        layer=layer,
        stage="test",
        source_as_of=datetime(2026, 1, 1, tzinfo=UTC),
        retrieved_at=datetime(2025, 11, 30, tzinfo=UTC),
    )
    with pytest.raises(Exception, match="before source_as_of"):
        make(impossible)


def test_a_whole_stage_agrees_about_how_current_the_data_is(warehouse):
    """One clock per run, so no two figures disagree about the date."""
    result = validate(
        warehouse, ANALYST,
        ValidationRequest("net_revenue", "West", "monthly", "2025-11", "2025-10"),
    )
    stamps = {item.source_as_of for item in result.evidence}
    assert len(stamps) == 1
    assert stamps.pop() == result.as_of


def test_the_stage_clock_is_the_warehouse_clock_not_today(warehouse):
    """A fixed extract is not stale for being an extract.

    The data ends on 30 November 2025. Stamping today's date on it would
    say the reader knows more than they do.
    """
    from engine.db import warehouse_clock

    result = validate(
        warehouse, ANALYST,
        ValidationRequest("net_revenue", "West", "monthly", "2025-11", "2025-10"),
    )
    assert result.as_of == warehouse_clock(warehouse)
    assert result.as_of < datetime.now(UTC) - timedelta(days=1)


# ===========================================================================
# Persistence — the accept criterion
# ===========================================================================


def test_no_persisted_row_has_a_null_source_system_as_of_method_or_lineage(sandbox):
    """ACCEPT. Every stage's evidence, written and read back."""
    sandbox.execute("DELETE FROM evidence")
    written = 0
    written += persist(
        sandbox,
        validate(
            sandbox, ANALYST,
            ValidationRequest("net_revenue", "West", "monthly", "2025-11", "2025-10"),
        ).evidence,
        case_id="CASE-2451",
    )
    written += persist(
        sandbox,
        qualify(
            sandbox, ANALYST,
            QualifyRequest("net_revenue", "West", "monthly", "2025-11", "2025-10"),
        ).evidence,
        case_id="CASE-2451",
    )
    assert written > 0

    nulls = sandbox.execute(
        "SELECT COUNT(*) FROM evidence WHERE source_system IS NULL "
        "OR source_as_of IS NULL OR method IS NULL OR lineage_json IS NULL "
        "OR lineage_json = '[]'"
    ).fetchone()[0]
    assert nulls == 0

    rows = load(sandbox, "CASE-2451")
    assert len(rows) == written
    for row in rows:
        assert row["source_system"]
        assert row["method"]
        assert row["source_as_of"] is not None
        assert row["lineage_json"] not in (None, "", "[]")


def test_the_persisted_as_of_is_the_data_date(sandbox):
    sandbox.execute("DELETE FROM evidence")
    result = validate(
        sandbox, ANALYST,
        ValidationRequest("net_revenue", "West", "monthly", "2025-11", "2025-10"),
    )
    persist(sandbox, result.evidence, case_id="CASE-2451")
    stamps = sandbox.execute(
        "SELECT DISTINCT source_as_of, retrieved_at FROM evidence"
    ).fetchall()
    assert len(stamps) == 1
    source_as_of, retrieved_at = stamps[0]
    assert source_as_of.date() == datetime(2025, 11, 30).date()
    assert retrieved_at > source_as_of


def test_a_number_lands_in_the_numeric_column_and_text_in_the_text_one(factory):
    numeric = as_row(make(factory, "number", value=76.92), "CASE-1")
    text = as_row(
        make(factory, "text", value="a1b2c3", unit="sha256", method="hash",
             source_system="semantic_layer"),
        "CASE-1",
    )
    assert numeric[5] == pytest.approx(76.92)
    assert numeric[6] is None
    assert text[5] is None
    assert text[6] == "a1b2c3"


def test_the_persisted_lineage_carries_the_statement(sandbox, factory):
    import json

    sandbox.execute("DELETE FROM evidence")
    sql = "SELECT COUNT(*) FROM fact_sales_daily"
    persist(sandbox, [make(factory, statement=sql)], case_id="CASE-1")
    stored = sandbox.execute("SELECT lineage_json FROM evidence").fetchone()[0]
    assert json.loads(stored)[0]["statement"] == sql


# ===========================================================================
# The ledger
# ===========================================================================


def test_two_numbers_cannot_share_an_id(factory):
    ledger = EvidenceLedger()
    ledger.add(make(factory, "same"))
    with pytest.raises(EvidenceError, match="already registered"):
        ledger.add(make(factory, "same"))


def test_the_ledger_keeps_order_and_lookup(factory):
    ledger = EvidenceLedger()
    for index in range(3):
        ledger.add(make(factory, f"item-{index}"))
    assert ledger.ids == ("test.item-0", "test.item-1", "test.item-2")
    assert ledger["test.item-1"].label
    assert "test.item-9" not in ledger
    assert len(ledger) == 3


def test_a_dangling_lineage_input_is_findable(factory):
    """A broken link is a click that lands nowhere."""
    ledger = EvidenceLedger()
    ledger.add(make(factory, "derived", inputs=("test.does-not-exist",)))
    assert ledger.unresolved_inputs() == ("test.does-not-exist",)


def test_the_stages_leave_no_dangling_links(warehouse):
    result = qualify(
        warehouse, ANALYST,
        QualifyRequest("net_revenue", "West", "monthly", "2025-11", "2025-10"),
    )
    ledger = EvidenceLedger()
    ledger.extend(result.evidence)
    assert ledger.unresolved_inputs() == ()


# ===========================================================================
# One way to make a number public
# ===========================================================================

SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".ruff_cache", "raw", "frontend",
}


def engine_sources() -> list[Path]:
    found: list[Path] = []
    for directory, subdirectories, files in os.walk(ENGINE):
        subdirectories[:] = [name for name in subdirectories if name not in SKIP_DIRS]
        found.extend(Path(directory) / name for name in files if name.endswith(".py"))
    return sorted(found)


def _constructs_evidence(path: Path) -> list[int]:
    """Lines calling `Evidence(...)` directly."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Evidence"
    ]


def test_there_are_engine_sources_to_scan():
    assert len(engine_sources()) > 5


def test_only_the_evidence_engine_constructs_evidence():
    """The same shape as P4's rule for `duckdb.connect`, for the same reason.

    A record built outside the factory has skipped every refusal it makes:
    it can name a source system nobody has heard of, a method that means
    nothing, or a reliability weight it chose for itself.
    """
    offenders = {
        str(path.relative_to(ROOT)): _constructs_evidence(path)
        for path in engine_sources()
        if path.name != "contracts.py" and _constructs_evidence(path)
    }
    offenders.pop(str(EVIDENCE_OWNER.relative_to(ROOT)), None)
    assert not offenders, (
        "Evidence must be minted through engine/evidence.py::EvidenceFactory, which "
        f"refuses records that cannot be chased. Found: {offenders}"
    )


# ===========================================================================
# The retrofit — P5 and P6 both go through the engine
# ===========================================================================


@pytest.mark.parametrize("stage", ["validate", "qualify"])
def test_every_stage_record_names_a_known_origin_and_method(warehouse, layer, stage):
    request = (
        ValidationRequest("net_revenue", "West", "monthly", "2025-11", "2025-10")
        if stage == "validate"
        else QualifyRequest("net_revenue", "West", "monthly", "2025-11", "2025-10")
    )
    runner = validate if stage == "validate" else qualify
    result = runner(warehouse, ANALYST, request)

    origins = layer.warehouse.known_origins()
    methods = set(layer.warehouse.evidence.methods)
    assert result.evidence
    for item in result.evidence:
        assert item.source_system in origins, item.evidence_id
        assert item.method in methods, item.evidence_id
        assert item.evidence_id.startswith(f"{stage}.")


def test_reconciliation_goes_through_the_engine_too(reconciliation, layer):
    origins = layer.warehouse.known_origins()
    methods = set(layer.warehouse.evidence.methods)
    for item in reconciliation.evidence:
        assert item.source_system in origins, item.evidence_id
        assert item.method in methods, item.evidence_id
        assert item.lineage


def test_no_engine_stage_returns_a_bare_float(warehouse):
    """Rule 3, stated as a property of what the stages hand back."""
    for result in (
        validate(
            warehouse, ANALYST,
            ValidationRequest("net_revenue", "West", "monthly", "2025-11", "2025-10"),
        ),
        qualify(
            warehouse, ANALYST,
            QualifyRequest("net_revenue", "West", "monthly", "2025-11", "2025-10"),
        ),
    ):
        assert result.evidence
        assert all(isinstance(item, Evidence) for item in result.evidence)
