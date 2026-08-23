"""GATHER — three lanes, and the line the model does not cross.

The acceptance criteria are three:

  21 of 34 treated stores filed a note tagged stock_out, 2 of 34 controls,
  chi-square p < 0.001
  cache hit rate >= 90% on a second run
  no numeric fact in the pipeline originates from a model response

The third is the one worth reading the tests for. It is checked three
ways, because "the model did not produce this number" is easy to claim and
easy to get wrong:

  * every `Evidence` carrying a number is `produced_by="code"`
  * every count and every p-value is RECOMPUTED here, in the test, from
    the tag list alone — if the pipeline's numbers came from anywhere but
    arithmetic over tags, they would not match
  * the classifier's own confidences are perturbed and the counts do not
    move. A number that changes when the model's numbers change came from
    the model.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pytest

from engine.contracts import Evidence
from engine.db import GovernanceError
from engine.gather import (
    Bm25,
    GatherError,
    GatherRequest,
    HybridIndex,
    chi_square,
    count_by_group,
    gather,
    screen,
    tokenise,
)
from engine.qualify import register_case
from llm.classify import Document, DocumentTag, build_request, parse_response
from llm.provider import LLMResponse, MockProvider
from security.policy import User

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "llm" / "fixtures"

ANALYST = User(user_id="U008", persona="analyst", display_name="Meera Joshi")

CASE_2451 = GatherRequest(
    kpi="net_revenue",
    scope="West",
    grain="monthly",
    period="2025-11",
    period_start=date(2025, 11, 1),
    period_end=date(2025, 11, 30),
    comparison_start=date(2025, 10, 1),
    comparison_end=date(2025, 10, 31),
)

#: The registry's unstructured finding is about STORE NOTES.
NOTES = "store_notes"
STOCK_OUT = "stock_out"


@pytest.fixture
def sandbox(warehouse):
    warehouse.execute("BEGIN TRANSACTION")
    try:
        yield warehouse
    finally:
        warehouse.execute("ROLLBACK")


@pytest.fixture(scope="module")
def result(warehouse):
    """One GATHER run for case #2451, shared across the module."""
    return gather(warehouse, ANALYST, CASE_2451)


@pytest.fixture(scope="module")
def store_groups(warehouse):
    """The treated and matched-control stores, straight from the master."""
    rows = warehouse.execute(
        "SELECT store_id, is_treated_2451, is_matched_control_2451 FROM dim_store "
        "WHERE region = 'West'"
    ).fetchall()
    return {
        "treated": {row[0] for row in rows if row[1]},
        "control": {row[0] for row in rows if row[2]},
    }


# ===========================================================================
# ACCEPT: 21 of 34 treated, 2 of 34 controls, chi-square p < 0.001
# ===========================================================================


def test_twenty_one_treated_stores_filed_a_stock_out_note(result):
    treated, _control, _test = result.unstructured.comparison(NOTES, STOCK_OUT)
    assert treated.stores == 34
    assert treated.flagged == 21


def test_at_most_two_control_stores_did(result):
    _treated, control, _test = result.unstructured.comparison(NOTES, STOCK_OUT)
    assert control.stores == 34
    assert control.flagged <= 2


def test_the_chi_square_is_significant(result):
    _treated, _control, test = result.unstructured.comparison(NOTES, STOCK_OUT)
    assert test.p_value < 0.001
    assert test.degrees_of_freedom == 1
    assert test.yates is True
    assert test.reliable


def test_the_comparison_is_never_pooled_across_corpora(result):
    """A shift note is the store's report; a ticket is a customer's complaint.

    They have different base rates and different meanings. Pooling them
    would produce a number about neither, and the registry's finding is
    explicitly about notes.
    """
    keys = set(result.unstructured.group_counts)
    assert (NOTES, STOCK_OUT) in keys
    assert ("tickets", STOCK_OUT) in keys
    notes = result.unstructured.comparison(NOTES, STOCK_OUT)
    tickets = result.unstructured.comparison("tickets", STOCK_OUT)
    assert notes[0].flagged != tickets[0].flagged


# ===========================================================================
# ACCEPT: cache hit rate >= 90% on a second run
# ===========================================================================


def test_a_second_run_is_served_from_the_cache(sandbox, layer):
    # An earlier test in this module has already warmed the cache on the
    # shared warehouse. Start cold, so "first" means first.
    sandbox.execute(
        f"DELETE FROM {layer.gather.unstructured.classification.cache_table}"
    )
    first = gather(sandbox, ANALYST, CASE_2451)
    second = gather(sandbox, ANALYST, CASE_2451)

    assert first.unstructured.batches > 0
    assert second.unstructured.batches == 0
    assert second.cache_hit_rate >= 0.90


def test_the_cache_is_keyed_on_the_document_not_the_batch(result, layer):
    """The corpora repeat, and that is what makes the cache worth having.

    "Routine day, footfall normal, no issues to report" is one document
    however many stores filed it. Hashing the batch would make every one
    of them a miss.
    """
    run = result.unstructured
    assert run.cache_lookups > 500
    assert run.distinct_documents < 60
    assert run.batches <= 3


def test_the_cache_key_covers_what_could_change_the_answer(layer):
    spec = layer.gather.unstructured.classification
    document = Document("N1", "Size 8 not available since morning.", "doc_store_notes")

    base = document.content_hash(("stock_out",), spec, "claude-haiku-4-5")
    assert base != document.content_hash(("price_increase",), spec, "claude-haiku-4-5")
    assert base != document.content_hash(("stock_out",), spec, "claude-sonnet-5")
    assert base != replace(document, text="Something else").content_hash(
        ("stock_out",), spec, "claude-haiku-4-5"
    )
    # NOT the document id: the same note from two stores is one classification.
    assert base == replace(document, document_id="N2").content_hash(
        ("stock_out",), spec, "claude-haiku-4-5"
    )


def test_the_cache_survives_the_process(sandbox, layer):
    gather(sandbox, ANALYST, CASE_2451)
    rows = sandbox.execute(
        f"SELECT COUNT(*) FROM {layer.gather.unstructured.classification.cache_table}"
    ).fetchone()[0]
    assert rows > 0


# ===========================================================================
# ACCEPT: no numeric fact originates from a model response
# ===========================================================================


def test_no_numeric_evidence_came_from_the_model(result):
    """Rule 1, end to end."""
    numeric = [
        item for item in result.evidence if isinstance(item.value, (int, float))
    ]
    assert numeric
    assert all(item.produced_by == "code" for item in numeric)


def test_the_counts_are_reproducible_from_the_tags_alone(result, layer, store_groups):
    """Recomputed here, in the test, from nothing but the tag list.

    If the pipeline's counts came from anywhere other than arithmetic over
    tags, these would not match.
    """
    spec = layer.gather.unstructured
    tags = result.unstructured.corpora[NOTES].tags
    treated, control, test = result.unstructured.comparison(NOTES, STOCK_OUT)

    again_treated, again_control = count_by_group(
        tags, store_groups, STOCK_OUT, spec.classification
    )
    assert again_treated.flagged == treated.flagged
    assert again_control.flagged == control.flagged

    again_test = chi_square(again_treated, again_control, spec.statistics)
    assert again_test.p_value == pytest.approx(test.p_value)
    assert again_test.statistic == pytest.approx(test.statistic)


def test_the_document_counts_are_arithmetic_over_the_tags(result, layer):
    """Every per-corpus count, recomputed the obvious way."""
    spec = layer.gather.unstructured.classification
    for name, corpus in result.unstructured.corpora.items():
        for tag, count in corpus.tag_counts.items():
            recomputed = sum(
                1
                for item in corpus.tags
                if item.hypothesis_tag == tag and item.accepted(spec)
            )
            assert recomputed == count, f"{name}/{tag}"


def test_the_counts_do_not_move_when_the_models_confidences_do(
    warehouse, layer, store_groups, result
):
    """The strongest form: perturb the model's numbers, keep its tags.

    Every confidence is replaced with a different value, all still above
    the acceptance floor. A count that came from the model's arithmetic
    would move. These do not.
    """
    from llm.classify import classify_documents
    from engine.gather.unstructured import load_documents

    spec = layer.gather.unstructured
    corpus = spec.corpora[NOTES]
    documents = load_documents(
        warehouse, ANALYST, layer, corpus,
        kpi="net_revenue", scope="West",
        period_start=CASE_2451.period_start, period_end=CASE_2451.period_end,
    )
    # The same slate the run used, so the cache keys and the fixtures line up.
    tags = result.screening.tags

    warehouse.execute("BEGIN TRANSACTION")
    try:
        honest = classify_documents(
            documents, tags, spec.classification, connection=warehouse
        )
        warehouse.execute(f"DELETE FROM {spec.classification.cache_table}")
        loud = classify_documents(
            documents, tags, spec.classification,
            connection=warehouse,
            provider=_ConfidenceShifter(FIXTURES, floor=spec.classification.min_confidence),
        )
    finally:
        warehouse.execute("ROLLBACK")

    before = count_by_group(
        tuple(honest.tags), store_groups, STOCK_OUT, spec.classification
    )
    after = count_by_group(
        tuple(loud.tags), store_groups, STOCK_OUT, spec.classification
    )

    confidences_changed = {tag.confidence for tag in honest.tags} != {
        tag.confidence for tag in loud.tags
    }
    assert confidences_changed, "the perturbation did not perturb anything"
    assert before[0].flagged == after[0].flagged
    assert before[1].flagged == after[1].flagged


class _ConfidenceShifter:
    """Replays the fixtures with every confidence changed, tags untouched."""

    name = "confidence-shifter"

    def __init__(self, fixtures_dir: Path, floor: float) -> None:
        self._mock = MockProvider(fixtures_dir)
        self._floor = floor

    def complete(self, request):
        response = self._mock.complete(request)
        payload = parse_response(response.text)
        for entry in payload:
            entry["confidence"] = round(min(self._floor + 0.07, 1.0), 4)
        return LLMResponse(
            text=json.dumps(payload, ensure_ascii=False),
            model=response.model,
            request_fingerprint=response.request_fingerprint,
            from_fixture=True,
        )


def test_the_classifier_is_never_asked_for_a_number(layer):
    """The prompt forbids it in as many words."""
    from llm.classify import SYSTEM_PROMPT

    assert "Never invent a count" in SYSTEM_PROMPT
    assert "not being asked how many" in SYSTEM_PROMPT


def test_the_long_tail_generator_is_never_asked_for_a_number():
    from llm.hypothesise import SYSTEM_PROMPT

    assert "Do NOT return a probability" in SYSTEM_PROMPT


def test_a_document_tag_is_not_evidence():
    """The types keep them apart, so no refactor can quietly merge them."""
    assert not issubclass(DocumentTag, Evidence)
    assert "value" not in DocumentTag.__dataclass_fields__


# ===========================================================================
# Hypotheses — three sources, screened to five
# ===========================================================================


def test_five_candidates_survive_the_screen(result, layer):
    assert len(result.hypotheses) == layer.gather.hypotheses.top_n == 5


def test_the_registry_drivers_are_all_in_the_slate(result):
    """#2451's H1, H3, H4 and H2/H5 map to four causal-graph drivers."""
    tags = set(result.screening.tags)
    assert {"stock_out", "price_increase", "promo_lapse", "competitor_action"} <= tags


def test_stock_out_leads_on_prior_times_applicability(result):
    assert result.hypotheses[0].tag == "stock_out"
    assert result.hypotheses[0].score == pytest.approx(0.18)


def test_an_unverifiable_hypothesis_is_not_screened_out(result):
    """The screen must not delete case #2451's conclusion.

    H2 (competitor promotion) cannot be tested, holds INR 0.86 Cr of the
    residual, and is exactly why the verdict is PARTIALLY EXPLAINED.
    Penalising a hypothesis for being unverifiable would drop it here and
    the case would come out EXPLAINED.
    """
    competitor = result.screening.by_tag()["competitor_action"]
    assert competitor.verifiable is False
    assert competitor.missing_sources
    assert competitor.applicability == pytest.approx(1.0)


def test_a_hypothesis_an_earlier_stage_removed_is_not_offered(result):
    """The residual is calendar-free by construction. Do not explain it twice."""
    excluded = result.screening.excluded
    assert "Gate 2" in excluded["calendar_shift"]
    assert "Gate 1" in excluded["data_incident"]
    assert "calendar_shift" not in result.screening.tags


def test_every_exclusion_carries_a_reason(result):
    assert result.screening.excluded
    assert all(reason.strip() for reason in result.screening.excluded.values())


def test_the_long_tail_is_offered_and_tagged_as_generated(result, layer):
    offered = result.screening.long_tail_offered
    prefix = layer.gather.hypotheses.sources.long_tail.tag_prefix
    assert offered
    assert all(candidate.tag.startswith(prefix) for candidate in offered)


def test_the_long_tail_does_not_beat_a_well_covered_graph(result):
    """The correct outcome for net revenue: the graph already has better answers."""
    assert all(not candidate.generated for candidate in result.hypotheses)


def test_the_long_tail_fills_the_slate_when_the_graph_is_thin(warehouse, layer):
    """Quick-commerce fulfilment has three graph drivers and five slots."""
    screening = screen(
        warehouse, ANALYST, layer,
        kpi="qcomm_fulfilment_rate", scope="All-India", grain="weekly",
        period="2025-11", direction="down", now=datetime(2025, 11, 30),
    )
    graph_candidates = [c for c in screening.selected if not c.generated]
    generated = [c for c in screening.selected if c.generated]
    assert len(graph_candidates) < layer.gather.hypotheses.top_n
    assert generated, "the long tail should fill the slate the graph cannot"


def test_case_history_lifts_a_driver_that_has_explained_this_before(sandbox, layer):
    """A driver that explained a movement here twice is a better bet."""
    driver = "mix_shift"
    baseline = screen(
        sandbox, ANALYST, layer,
        kpi="net_revenue", scope="West", grain="monthly",
        period="2025-11", direction="down", now=datetime(2025, 11, 30),
    ).by_tag()
    assert driver not in baseline

    for index in range(4):
        case_id = f"C-HIST-{index}"
        register_case(
            sandbox, case_id=case_id, kpi="net_revenue", scope="West",
            grain="monthly", period=f"2025-0{index + 1}",
            opened_at=datetime(2025, index + 1, 15), materiality_multiple=5.0,
        )
        sandbox.execute(
            "UPDATE case_registry SET status = 'closed' WHERE case_id = ?", [case_id]
        )
        sandbox.execute(
            "INSERT INTO case_hypothesis (case_id, hypothesis_id, label, description, "
            "status, verifiable) VALUES (?, ?, 'Mix shift', 'x', 'supported', TRUE)",
            [case_id, driver],
        )

    lifted = screen(
        sandbox, ANALYST, layer,
        kpi="net_revenue", scope="West", grain="monthly",
        period="2025-11", direction="down", now=datetime(2025, 11, 30),
    ).by_tag()
    assert driver in lifted, "four prior cases should lift it into the slate"
    assert lifted[driver].prior_cases == 4
    assert lifted[driver].history_bonus == pytest.approx(
        layer.gather.hypotheses.sources.case_history.max_bonus
    )


def test_the_slate_is_reproducible(warehouse, layer):
    """Same inputs, same five, same order — ties broken by name."""
    runs = [
        screen(
            warehouse, ANALYST, layer,
            kpi="net_revenue", scope="West", grain="monthly",
            period="2025-11", direction="down", now=datetime(2025, 11, 30),
        ).tags
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


# ===========================================================================
# Lane 1 — structured
# ===========================================================================


def test_every_candidate_gets_a_structured_answer(result):
    assert {item.hypothesis for item in result.structured} == set(
        result.screening.tags
    )


def test_a_hypothesis_with_no_template_says_why(result):
    """"We have no query" and "we forgot to write one" look identical when absent."""
    competitor = result.structured_for("competitor_action")
    assert competitor.available is False
    assert "not held" in competitor.reason
    assert competitor.evidence == ()


def test_the_structured_lane_keeps_its_sql_verbatim(result):
    stock_out = result.structured_for("stock_out")
    assert stock_out.available
    assert stock_out.evidence
    statement = stock_out.evidence[0].lineage[0].statement
    assert "fact_inventory_snapshot" in statement
    assert "snapshot_hour_ist = 12" in statement


def test_the_structured_lane_runs_through_the_governed_door(sandbox):
    """A persona who cannot see a scope cannot gather evidence about it."""
    east_manager = User(user_id="U004", persona="regional_manager", region="East")
    result = gather(sandbox, east_manager, CASE_2451)
    stock_out = result.structured_for("stock_out")
    assert stock_out.value is None


def test_the_lane_gathers_and_does_not_judge(result):
    stock_out = result.structured_for("stock_out")
    assert "ADJUDICATE" in (stock_out.evidence[0].notes or "")


# ===========================================================================
# Lane 2 — retrieval
# ===========================================================================


def test_bm25_finds_the_lexical_match():
    documents = [
        "Routine day, footfall normal, no issues to report.",
        "Customer asked for size 8 in the running range, not available since morning.",
        "New window display installed as per visual merchandising guide.",
    ]
    from semantic_layer.schema import get_semantic_layer

    spec = get_semantic_layer().gather.unstructured.retrieval
    scores = Bm25(documents, spec).scores("size not available")
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_the_embeddings_find_what_the_words_do_not(layer):
    """Hinglish and English say the same thing with no shared terms."""
    documents = [
        "Popular sizes stock mein nahi hai, customer wapas chala gaya.",
        "Staff training conducted for new POS screen.",
        "Shelf for top sellers empty again, replenishment truck did not come.",
    ]
    index = HybridIndex(["a", "b", "c"], documents, layer.gather.unstructured.retrieval)
    ranked = index.search("shelf empty no stock")
    assert ranked[0].document_id in {"a", "c"}
    assert ranked[0].score > 0


def test_the_blend_uses_both_rankers(layer):
    spec = layer.gather.unstructured.retrieval
    assert spec.bm25_weight > 0 and spec.embedding_weight > 0
    assert spec.bm25_weight + spec.embedding_weight == pytest.approx(1.0)


def test_ranking_decides_what_is_shown_not_what_is_counted(result, layer):
    """Every in-scope document is classified, not just the top-ranked ones.

    A store whose note the ranker happened to miss would otherwise count as
    a store that filed nothing, and the chi-square is a statement about 68
    stores rather than about the twelve documents that scored highest.
    """
    notes = result.unstructured.corpora[NOTES]
    assert len(notes.ranked) == layer.gather.unstructured.retrieval.top_k_per_corpus
    assert len(notes.tags) == notes.documents
    assert notes.documents > len(notes.ranked)


def test_tokenisation_handles_both_languages():
    assert tokenise("Size 8 khatam, nahi hai") == ["size", "8", "khatam", "nahi", "hai"]


# ===========================================================================
# Lane 3 — external
# ===========================================================================


def test_the_external_lane_fetches_every_declared_feed(result, layer):
    assert set(result.external.feeds) == set(layer.gather.external.feeds)


def test_the_competitor_feed_reports_the_columns_it_does_not_have(result):
    """The absence IS the evidence, and it is what fires T3 on #2467."""
    news = result.external.feeds["competitor_news"]
    assert news.available
    assert set(news.absent_columns) == {"price", "footfall"}
    assert "structurally unverifiable" in (news.evidence[0].notes or "")


def test_the_external_lane_gathers_the_confounders(result):
    """Test 6 cannot screen what nobody fetched."""
    assert result.external.feeds["weather"].available
    assert result.external.feeds["calendar"].available


# ===========================================================================
# Execution
# ===========================================================================


def test_all_three_lanes_ran(result):
    assert set(result.lanes_run) == {"structured", "unstructured", "external"}
    assert result.lane_failures == {}


def test_the_lanes_run_concurrently(layer):
    assert layer.gather.execution.parallel_lanes is True
    assert layer.gather.execution.max_workers >= 3


def test_a_failing_lane_does_not_fail_the_stage(sandbox, layer):
    """An adjudication that knows a lane is missing beats one that never ran."""
    assert layer.gather.execution.fail_open is True
    sandbox.execute("DROP TABLE doc_store_notes")
    result = gather(sandbox, ANALYST, CASE_2451)
    assert "unstructured" in result.lane_failures
    assert "structured" in result.lanes_run
    assert result.evidence


def test_an_unknown_kpi_is_refused(warehouse):
    with pytest.raises(GatherError, match="no KPI contract"):
        gather(warehouse, ANALYST, replace(CASE_2451, kpi="not_a_kpi"))


def test_a_persona_the_contract_does_not_know_is_refused(warehouse):
    with pytest.raises(GovernanceError, match="not defined in the access policy"):
        gather(warehouse, User(user_id="U999", persona="intern"), CASE_2451)


# ===========================================================================
# Evidence
# ===========================================================================


def test_every_figure_leaves_as_evidence(result):
    assert result.evidence
    assert all(isinstance(item, Evidence) for item in result.evidence)
    ids = [item.evidence_id for item in result.evidence]
    assert len(ids) == len(set(ids))


def test_the_document_counts_carry_the_corpus_reliability(result, layer):
    """A review is 0.35 and a warehouse query is 0.95, and the tier says so."""
    weights = layer.adjudication.reliability.weights
    for item in result.evidence:
        assert item.reliability == pytest.approx(weights[item.kind])
    review_counts = [
        item for item in result.evidence if item.evidence_id.startswith(
            "gather.unstructured.reviews."
        ) and item.kind == "social_mention"
    ]
    assert review_counts
    assert all(item.reliability < 0.6 for item in review_counts)


def test_the_chi_square_evidence_says_where_the_number_came_from(result):
    item = result.evidence_by_id()[
        f"gather.unstructured.{NOTES}.{STOCK_OUT}.chi_square_p"
    ]
    assert item.produced_by == "code"
    assert item.method == "compare"
    assert "no part of this number" in (item.notes or "")


# ===========================================================================
# Offline
# ===========================================================================


def test_the_suite_runs_with_the_model_mocked():
    """CLAUDE.md rule 8: the demo runs offline, from day one."""
    from llm.provider import get_provider, mock_enabled

    assert mock_enabled()
    assert isinstance(get_provider(), MockProvider)


def test_the_offline_fixtures_say_they_are_not_recordings():
    """Honesty about what an offline fixture proves, in the fixture itself."""
    from llm.record_fixtures import SYNTHETIC_NOTE

    synthetic = []
    for path in FIXTURES.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("synthetic"):
            synthetic.append(path)
            assert payload["note"] == SYNTHETIC_NOTE
    assert synthetic, "no fixtures are labelled; re-run llm.record_fixtures"


def test_the_stand_in_is_not_reachable_from_the_engine():
    """The keyword matcher lives in the recorder and nowhere else."""
    import ast

    for path in (ROOT / "engine").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                assert not any(
                    "record_fixtures" in name for name in names
                ), f"{path.relative_to(ROOT)} imports the fixture stand-in"


def test_a_batch_request_is_deterministic(layer):
    """The fingerprint is the cache key and the fixture key."""
    spec = layer.gather.unstructured.classification
    documents = (Document("N1", "Size 8 not available.", "doc_store_notes"),)
    first = build_request(documents, ("stock_out",), spec)
    second = build_request(documents, ("stock_out",), spec)
    assert first.fingerprint() == second.fingerprint()
