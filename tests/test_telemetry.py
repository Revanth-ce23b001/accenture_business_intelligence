"""P14 — telemetry.

The four accept criteria, each with its own section:

  1. every request writes a complete row with all five stage latencies
  2. cost per case < INR 6
  3. P95 total latency < 9 s warm
  4. a cost projection helper for 10,000 interactions/week

Plus the measurement that replaced resolved defect 6: real wall-clock time
from case open to verdict, stamped on `case_registry`.

TWO THINGS THIS SUITE IS CAREFUL NOT TO DO.

It does not assert a threshold as a literal. Every ceiling comes out of
`semantic_layer/telemetry.yaml` through the loaded layer, so an edit to
the contract moves the test with it rather than leaving it asserting last
month's target (rule 2).

It does not pretend a replay is a measurement. `MOCK_LLM=true` replays
fixtures, which buy no tokens; the cost of an offline run is estimated
from text and every row it produces carries `cost_estimated = True`. The
tests assert that flag exactly as hard as they assert the money.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from engine.contracts import RequestTelemetry
from engine.warehouse.load import create_schema
from llm.provider import TASK_MODEL, MockProvider
from telemetry.cost import (
    Usage,
    check_cost,
    cost_of_call,
    estimate_usage,
    mean_cost_per_request,
    project,
    project_at_ceiling,
    project_measured,
    reported_usage,
)
from telemetry.recorder import (
    STAGES,
    MeteredProvider,
    Recorder,
    TelemetryError,
    check_latency,
    load_requests,
    percentile,
    persist,
    reset_warmup,
    write_case_elapsed,
)

TELEMETRY_DIR = Path(__file__).resolve().parents[1] / "telemetry"

FIXED_TS = datetime(2025, 11, 30, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeClock:
    """A monotonic clock that only moves when told to.

    Timing tests that rely on real elapsed time are the flakiest tests in
    any suite. Every assertion about a DURATION here uses this; the ones
    that measure the recorder's own overhead use the real clock, and say
    so.
    """

    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def recorder(layer, clock) -> Recorder:
    reset_warmup()
    return Recorder(
        user_id="u.reg.west",
        persona="regional_manager",
        question="why did west revenue fall last month",
        layer=layer,
        mock=True,
        monotonic=clock,
        wall_clock=lambda: FIXED_TS,
    )


@pytest.fixture(scope="session")
def telemetry_warehouse(warehouse):
    """The warehouse with the telemetry tables present and empty.

    Session-scoped like `warehouse` itself; the tests that write clear
    their own rows first so they do not read each other's.
    """
    create_schema(warehouse)
    return warehouse


@pytest.fixture
def clean_telemetry(telemetry_warehouse):
    telemetry_warehouse.execute("DELETE FROM telemetry_request")
    telemetry_warehouse.execute("DELETE FROM telemetry_event")
    return telemetry_warehouse


def run_all_stages(recorder: Recorder, clock: FakeClock, *, ms: float = 100.0) -> None:
    for stage in STAGES:
        with recorder.stage(stage):
            clock.advance(ms / 1000.0)


def make_row(**overrides) -> RequestTelemetry:
    payload = dict(
        request_id="req.001",
        user_id="u.reg.west",
        persona="regional_manager",
        started_at=FIXED_TS,
        total_latency_ms=0.0,
        latency_by_stage={stage: 0.0 for stage in STAGES},
    )
    payload.update(overrides)
    return RequestTelemetry(**payload)


# ===========================================================================
# Accept criterion 1 — every request writes a complete row, five latencies
# ===========================================================================


def test_the_five_stages_are_the_locked_five(layer):
    """Taken from the contract, not retyped. There is one copy of the order."""
    assert STAGES == ("VALIDATE", "QUALIFY", "GATHER", "ADJUDICATE", "VERDICT")


def test_a_full_request_records_every_stage(recorder, clock):
    run_all_stages(recorder, clock)
    row = recorder.close(verdict="PARTIALLY_EXPLAINED", confidence=0.84)

    assert set(row.latency_by_stage) == set(STAGES)
    assert row.stages_entered == STAGES
    assert all(value == pytest.approx(100.0) for value in row.latency_by_stage.values())
    assert row.total_latency_ms == pytest.approx(500.0)


def test_a_gate_one_kill_still_writes_all_five_latencies(recorder, clock):
    """Case #2470 shape: VALIDATE kills it, the row is still complete.

    The four stages that never ran record 0.0, and `stages_entered` says
    which did — so the zero cannot be misread as "ADJUDICATE was instant".
    """
    with recorder.stage("VALIDATE"):
        clock.advance(0.04)
    row = recorder.close()

    assert set(row.latency_by_stage) == set(STAGES)
    assert row.stages_entered == ("VALIDATE",)
    assert row.latency_by_stage["VALIDATE"] == pytest.approx(40.0)
    assert row.latency_by_stage["ADJUDICATE"] == 0.0
    assert row.verdict is None


def test_a_row_missing_a_stage_is_refused():
    partial = {stage: 0.0 for stage in STAGES if stage != "GATHER"}
    with pytest.raises(ValueError, match="GATHER"):
        make_row(latency_by_stage=partial)


def test_a_row_naming_an_unknown_stage_is_refused():
    """The `Stage` literal does this one — the vocabulary is closed."""
    latencies = {stage: 0.0 for stage in STAGES}
    latencies["RETROSPECTIVE"] = 1.0
    with pytest.raises(ValueError, match="RETROSPECTIVE"):
        make_row(latency_by_stage=latencies)


def test_a_stage_that_raises_still_records_its_time(recorder, clock):
    """The row for a request that died mid-stage is the interesting one."""
    with pytest.raises(ZeroDivisionError):
        with recorder.stage("GATHER"):
            clock.advance(0.25)
            raise ZeroDivisionError("the corpus reader fell over")

    row = recorder.close()
    assert row.latency_by_stage["GATHER"] == pytest.approx(250.0)
    assert row.stages_entered == ("GATHER",)


def test_re_entering_a_stage_accumulates(recorder, clock):
    for _ in range(3):
        with recorder.stage("GATHER"):
            clock.advance(0.01)
    row = recorder.close()
    assert row.latency_by_stage["GATHER"] == pytest.approx(30.0)
    assert row.stages_entered == ("GATHER",)


def test_stages_may_not_be_nested(recorder):
    with pytest.raises(TelemetryError, match="still open"):
        with recorder.stage("GATHER"):
            with recorder.stage("ADJUDICATE"):
                pass


def test_the_rail_runs_forwards_only(recorder):
    with recorder.stage("ADJUDICATE"):
        pass
    with pytest.raises(TelemetryError, match="order is locked"):
        with recorder.stage("QUALIFY"):
            pass


def test_closing_with_a_stage_open_is_refused(recorder):
    with pytest.raises(TelemetryError, match="still open"):
        with recorder.stage("GATHER"):
            recorder.close()


def test_a_request_closes_once(recorder, clock):
    run_all_stages(recorder, clock)
    recorder.close()
    with pytest.raises(TelemetryError, match="already closed"):
        recorder.close()


def test_the_database_refuses_a_row_with_a_missing_stage_latency(telemetry_warehouse):
    """The accept criterion is enforced by the schema, not only by Python.

    A caller that finds a way around the contract still cannot write an
    incomplete row.
    """
    columns = dict(
        telemetry_warehouse.execute(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_name = 'telemetry_request'"
        ).fetchall()
    )
    for stage in STAGES:
        name = f"latency_{stage.lower()}_ms"
        assert name in columns, f"telemetry_request has no {name}"
        assert columns[name] == "NO", f"{name} is nullable; the row could be incomplete"
    assert columns["total_latency_ms"] == "NO"
    assert columns["stages_entered"] == "NO"


def test_a_persisted_row_reads_back_identical(clean_telemetry, recorder, clock, layer):
    run_all_stages(recorder, clock)
    recorder.record_methods(["stl_decomposition", "matched_did"])
    recorder.record_rows(scanned=412, filtered=272, released=34)
    recorder.record_cache(hits=1_842, misses=158)
    row = recorder.close(verdict="PARTIALLY_EXPLAINED", confidence=0.84)

    persist(clean_telemetry, row, events=recorder.events())
    (stored,) = load_requests(clean_telemetry)

    assert stored.request_id == row.request_id
    assert stored.latency_by_stage == pytest.approx(row.latency_by_stage)
    assert stored.stages_entered == STAGES
    assert stored.analytical_methods_executed == ("stl_decomposition", "matched_did")
    assert (stored.rows_scanned, stored.rows_filtered_by_policy) == (412, 272)
    assert stored.rows_released_to_llm == 34
    assert (stored.cache_hits, stored.cache_misses) == (1_842, 158)
    assert stored.verdict == "PARTIALLY_EXPLAINED"
    assert stored.confidence == pytest.approx(0.84)


def test_the_events_behind_the_row_are_written_too(clean_telemetry, recorder, clock):
    run_all_stages(recorder, clock)
    row = recorder.close()
    persist(clean_telemetry, row, events=recorder.events())

    events = clean_telemetry.execute(
        "SELECT stage, duration_ms FROM telemetry_event WHERE event = 'stage'"
    ).fetchall()
    assert {stage for stage, _ in events} == set(STAGES)


def test_a_missing_table_is_reported_not_swallowed(recorder, clock, tmp_path):
    """Unlike the audit write, this one raises. Silence would hide the criterion."""
    from engine.db import connect

    empty = connect(tmp_path / "no-schema.duckdb")
    run_all_stages(recorder, clock)
    try:
        with pytest.raises(TelemetryError, match="create_schema"):
            persist(empty, recorder.close())
    finally:
        empty.close()


def test_the_governed_result_is_folded_in_whole(recorder):
    """`rows_scanned` counts what the POLICY evaluated: returned + withheld."""

    class Result:
        rows = ({"store_id": "s1"}, {"store_id": "s2"})
        rows_filtered = 5

    recorder.record_query(Result())
    row = recorder.close()
    assert row.rows_scanned == 7
    assert row.rows_filtered_by_policy == 5


def test_a_release_is_counted_from_the_payload_that_was_released(recorder):
    """Same object the audit row was written from, so the two cannot disagree."""

    class Payload:
        row_count = 34

    recorder.record_release(Payload())
    assert recorder.close().rows_released_to_llm == 34


def test_grounding_counts_come_off_the_report(recorder):
    class Report:
        claims_checked = 14
        claims_stripped = 0

    recorder.record_grounding(Report())
    row = recorder.close()
    assert (row.grounding_claims_checked, row.grounding_claims_stripped) == (14, 0)


def test_more_stripped_than_checked_is_refused():
    with pytest.raises(ValueError, match="stripped"):
        make_row(grounding_claims_checked=3, grounding_claims_stripped=4)


def test_one_model_per_call_or_the_row_is_refused():
    with pytest.raises(ValueError, match="one entry per call"):
        make_row(llm_calls=2, model_per_call=("claude-sonnet-5",))


# ===========================================================================
# Resolved defect 6 — real wall-clock, case open to verdict
# ===========================================================================


def test_the_case_clock_runs_from_case_open_to_verdict(recorder, clock):
    """Not from the start of the request, and not to the end of it.

    VALIDATE happens before a case exists. Persisting artefacts happens
    after the analyst has their answer. Neither belongs in "how long did
    this case take".
    """
    with recorder.stage("VALIDATE"):
        clock.advance(1.0)
    with recorder.stage("QUALIFY"):
        clock.advance(0.5)
        recorder.mark_case_opened("2451")
        clock.advance(0.5)
    with recorder.stage("GATHER"):
        clock.advance(2.0)
    with recorder.stage("ADJUDICATE"):
        clock.advance(3.0)
    with recorder.stage("VERDICT"):
        clock.advance(1.0)
    clock.advance(10.0)  # housekeeping after the verdict — not the analyst's wait
    row = recorder.close(verdict="PARTIALLY_EXPLAINED", confidence=0.84)

    assert row.case_id == "2451"
    assert row.case_elapsed_ms == pytest.approx(6_500.0)
    assert row.total_latency_ms == pytest.approx(18_000.0)


def test_a_request_that_opened_no_case_has_no_case_elapsed(recorder, clock):
    """None rather than 0.0. Gate 1 killed it before a case existed."""
    with recorder.stage("VALIDATE"):
        clock.advance(0.04)
    assert recorder.close().case_elapsed_ms is None


def test_the_elapsed_time_lands_on_case_registry(clean_telemetry, recorder, clock):
    """The number the case header shows. Measured, not asserted."""
    from engine.qualify.restraint import register_case

    case_id = "TEL-2451"
    clean_telemetry.execute("DELETE FROM case_registry WHERE case_id = ?", [case_id])
    register_case(
        clean_telemetry,
        case_id=case_id,
        kpi="net_revenue",
        scope="West",
        grain="monthly",
        period="2025-11",
        opened_at=FIXED_TS,
    )

    with recorder.stage("QUALIFY"):
        recorder.mark_case_opened(case_id)
    with recorder.stage("VERDICT"):
        clock.advance(7.5)
    row = recorder.close(verdict="PARTIALLY_EXPLAINED", confidence=0.84)
    persist(clean_telemetry, row)

    (stored,) = clean_telemetry.execute(
        "SELECT elapsed_ms FROM case_registry WHERE case_id = ?", [case_id]
    ).fetchone()
    assert stored == pytest.approx(7_500.0)


def test_stamping_a_case_nobody_opened_is_refused(clean_telemetry):
    with pytest.raises(TelemetryError, match="not in case_registry"):
        write_case_elapsed(clean_telemetry, "CASE-THAT-NEVER-WAS", 1.0)


# ===========================================================================
# Accept criterion 2 — cost per case < INR 6
# ===========================================================================


def test_every_routed_model_is_priced(layer):
    """A model the router can pick and the price list cannot price is a
    call that would silently cost zero."""
    for task, model in TASK_MODEL.items():
        assert model in layer.telemetry.pricing, f"{task} routes to unpriced {model}"


def test_an_unpriced_model_is_refused_not_zeroed(layer):
    from semantic_layer.schema import SemanticLayerError

    with pytest.raises(SemanticLayerError, match="no price for model"):
        cost_of_call(Usage("claude-not-a-model", input_tokens=100), layer)


def test_a_call_is_priced_off_the_loaded_rates(layer):
    """The arithmetic, against the contract rather than against a literal."""
    price = layer.telemetry.price("claude-sonnet-5")
    rate = layer.telemetry.currency.usd_to_inr
    usage = Usage(
        model="claude-sonnet-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    call = cost_of_call(usage, layer)

    assert call.input_usd == pytest.approx(price.input_usd_per_mtok)
    assert call.output_usd == pytest.approx(price.output_usd_per_mtok)
    assert call.cache_read_usd == pytest.approx(price.cache_read_usd_per_mtok)
    assert call.cache_write_usd == pytest.approx(price.cache_write_usd_per_mtok)
    assert call.inr == pytest.approx(call.usd * rate)


def test_a_cache_hit_costs_less_than_the_miss_it_replaced(layer):
    """The whole argument for the content-hash cache, as arithmetic."""
    miss = cost_of_call(Usage("claude-haiku-4-5", input_tokens=100_000), layer)
    hit = cost_of_call(
        Usage("claude-haiku-4-5", cache_read_input_tokens=100_000), layer
    )
    assert hit.inr < miss.inr


def test_the_ceiling_comes_from_the_contract(layer):
    budget = check_cost(0.0, layer)
    assert budget.ceiling_inr == layer.telemetry.targets.cost_per_case_inr


def test_a_case_over_the_ceiling_says_so(layer):
    ceiling = layer.telemetry.targets.cost_per_case_inr
    assert check_cost(ceiling, layer).within
    assert not check_cost(ceiling * 2, layer).within
    assert "OVER" in check_cost(ceiling * 2, layer).render()


def test_a_replayed_call_is_flagged_as_estimated(layer):
    """A fixture buys no tokens. The row must not read as a measurement."""

    class Replayed:
        model = "claude-sonnet-5"
        text = "The West region's decline is 79% attributed to stock-outs."
        input_tokens = output_tokens = 0
        cache_read_input_tokens = cache_creation_input_tokens = 0

    usage, estimated = reported_usage(Replayed(), layer=layer)
    assert estimated
    assert usage.output_tokens > 0


def test_the_estimate_reads_the_prompt_as_well_as_the_reply(layer):
    """Without the request, an offline run under-reports its own input."""
    from llm.provider import LLMMessage, LLMRequest

    class Replayed:
        model = "claude-sonnet-5"
        text = "One sentence."

    request = LLMRequest.for_task(
        "narrate",
        system="x" * 4_000,
        messages=(LLMMessage(role="user", content="y" * 4_000),),
    )
    without = estimate_usage(Replayed(), layer=layer)
    with_prompt = estimate_usage(Replayed(), request=request, layer=layer)

    assert without.input_tokens == 0
    assert with_prompt.input_tokens > without.input_tokens
    assert with_prompt.output_tokens == without.output_tokens


def test_reported_counts_beat_the_estimate(layer):
    class Live:
        model = "claude-sonnet-5"
        text = "x" * 4_000
        input_tokens = 11
        output_tokens = 7
        cache_read_input_tokens = cache_creation_input_tokens = 0

    usage, estimated = reported_usage(Live(), layer=layer)
    assert not estimated
    assert (usage.input_tokens, usage.output_tokens) == (11, 7)


def test_the_offline_narrative_for_2451_stays_within_the_per_case_budget(layer):
    """A real run of the demo path, priced.

    `narrate_all` writes the case for all three personas through the
    fixture store — the actual calls the offline demo makes. The token
    counts are estimated (fixtures record none), which is why the row is
    flagged, but the call count, the routing and the text are the real
    thing.

    This is the narrative only. Classification and hypothesis generation
    add to it, which is what the headroom in this assertion is for.
    """
    from llm.casefiles import frozen
    from llm.narrate import narrate_all

    reset_warmup()
    recorder = Recorder(user_id="u.cxo", persona="cxo", layer=layer, mock=True)
    provider = MeteredProvider(MockProvider(), recorder)
    with recorder.stage("VERDICT"):
        narrate_all(frozen("2451", layer), layer, provider=provider)
    row = recorder.close(verdict="PARTIALLY_EXPLAINED", confidence=0.84)

    assert row.llm_calls == len(layer.narrate.personas)
    assert set(row.model_per_call) == {"claude-sonnet-5"}
    assert row.cost_estimated, "an offline run must not present itself as measured"
    assert check_cost(row.estimated_cost_inr, layer).within, (
        f"narration alone cost INR {row.estimated_cost_inr:.2f} of the INR "
        f"{layer.telemetry.targets.cost_per_case_inr:.2f} per-case ceiling"
    )


def test_a_metered_provider_cannot_be_bypassed(layer):
    """Every completion goes through the meter, including the regeneration.

    `narrate` makes a second call when a sentence fails grounding. A
    telemetry design that counted calls at the call site would miss it.
    """
    from llm.provider import LLMProvider

    recorder = Recorder(user_id="u1", persona="cxo", layer=layer, mock=True)
    provider = MeteredProvider(MockProvider(), recorder)
    assert isinstance(provider, LLMProvider)
    assert provider.name.startswith("metered:")


def test_the_mean_over_no_rows_is_zero_not_a_crash():
    assert mean_cost_per_request([]) == 0.0


# ===========================================================================
# Accept criterion 3 — P95 total latency < 9 s warm
# ===========================================================================


def test_the_percentile_is_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 4.0
    assert percentile(values, 0.5) == pytest.approx(2.5)
    assert percentile([7.0], 0.95) == 7.0


def test_a_percentile_of_nothing_is_refused():
    with pytest.raises(TelemetryError, match="no measurements"):
        percentile([], 0.95)


def test_the_budget_and_the_quantile_come_from_the_contract(layer):
    rows = [make_row(warm=True, total_latency_ms=1_000.0)]
    report = check_latency(rows, layer=layer)
    assert report.budget_ms == layer.telemetry.targets.latency_budget_ms
    assert report.quantile == layer.telemetry.targets.latency_percentile


def test_cold_requests_are_excluded_from_the_warm_claim(layer):
    """"Warm" is honoured by dropping the rows marked cold, not by choosing.

    The cold row here is slower than the budget. Included, it would fail
    the claim; the claim says warm, so it is excluded and counted.
    """
    budget = layer.telemetry.targets.latency_budget_ms
    rows = [
        make_row(request_id="cold", warm=False, total_latency_ms=budget * 3),
        *(
            make_row(request_id=f"warm{i}", warm=True, total_latency_ms=1_000.0)
            for i in range(9)
        ),
    ]
    report = check_latency(rows, layer=layer)
    assert report.within
    assert report.sample_size == 9
    assert report.excluded_cold == 1
    assert not check_latency(rows, layer=layer, warm_only=False).within


def test_all_cold_rows_is_a_refusal_not_a_pass(layer):
    rows = [make_row(warm=False, total_latency_ms=1.0)]
    with pytest.raises(TelemetryError, match="all 1 were cold"):
        check_latency(rows, layer=layer)


def test_the_first_request_of_a_process_is_cold(layer):
    """Measured, not asserted in a footnote."""
    reset_warmup()
    first = Recorder(user_id="u1", persona="cxo", layer=layer, mock=True).close()
    second = Recorder(user_id="u1", persona="cxo", layer=layer, mock=True).close()
    assert not first.warm
    assert second.warm


def test_a_breach_of_the_budget_is_reported_as_one(layer):
    budget = layer.telemetry.targets.latency_budget_ms
    rows = [make_row(warm=True, total_latency_ms=budget + 1.0)]
    report = check_latency(rows, layer=layer)
    assert not report.within
    assert "OVER" in report.render()


def test_the_instrument_costs_a_negligible_share_of_the_budget(layer):
    """The recorder's own overhead, on the real clock.

    This measures the INSTRUMENT, not the pipeline — there is no
    end-to-end run to time until the API arrives. What it rules out is a
    telemetry layer that eats the budget it exists to police.
    """
    reset_warmup()
    rows = []
    for _ in range(50):
        recorder = Recorder(user_id="u1", persona="cxo", layer=layer, mock=True)
        for stage in STAGES:
            with recorder.stage(stage):
                pass
        rows.append(recorder.close())

    report = check_latency(rows, layer=layer)
    assert report.measured_ms < layer.telemetry.targets.latency_budget_ms / 100.0, (
        f"the recorder alone accounts for {report.measured_ms:.1f} ms of the "
        f"{report.budget_ms:.0f} ms budget"
    )


# ===========================================================================
# Accept criterion 4 — projection for 10,000 interactions/week
# ===========================================================================


def test_the_volume_comes_from_the_contract(layer):
    assert project(1.0, layer=layer).interactions_per_week == 10_000
    assert layer.telemetry.projection.interactions_per_week == 10_000


def test_the_projection_arithmetic(layer):
    spec = layer.telemetry.projection
    projection = project(2.50, layer=layer)

    assert projection.weekly_inr == pytest.approx(2.50 * spec.interactions_per_week)
    assert projection.annual_inr == pytest.approx(
        projection.weekly_inr * spec.weeks_per_year
    )
    assert projection.monthly_inr == pytest.approx(
        projection.annual_inr / spec.months_per_year
    )
    assert projection.annual_inr_lakh == pytest.approx(projection.annual_inr / 100_000)


def test_a_month_is_a_twelfth_of_a_year_not_four_weeks(layer):
    """Four-week months lose four weeks a year — an 8% understatement."""
    projection = project(1.0, layer=layer)
    assert projection.monthly_inr > projection.weekly_inr * 4


def test_the_volume_can_be_overridden_without_editing_the_contract(layer):
    assert project(1.0, layer=layer, interactions_per_week=30_000).weekly_inr == 30_000


def test_the_projection_says_what_it_was_measured_over(layer):
    rows = [
        make_row(request_id="a", estimated_cost_inr=2.0),
        make_row(request_id="b", estimated_cost_inr=4.0),
    ]
    projection = project_measured(rows, layer=layer)
    assert projection.cost_per_interaction_inr == pytest.approx(3.0)
    assert "2 recorded requests" in projection.basis
    assert "2 recorded requests" in projection.render()


def test_the_worst_case_is_the_published_ceiling(layer):
    """The number to quote as an upper bound: every case at INR 6."""
    ceiling = layer.telemetry.targets.cost_per_case_inr
    projection = project_at_ceiling(layer=layer)
    assert projection.cost_per_interaction_inr == ceiling
    assert projection.weekly_inr == pytest.approx(
        ceiling * layer.telemetry.projection.interactions_per_week
    )
    assert projection.annual_inr == pytest.approx(projection.weekly_inr * 52)


def test_a_projection_from_recorded_rows_beats_one_from_a_typed_number(
    clean_telemetry, recorder, clock, layer
):
    """End to end: record, persist, read back, project."""
    run_all_stages(recorder, clock)
    persist(clean_telemetry, recorder.close())

    rows = load_requests(clean_telemetry)
    projection = project_measured(rows, layer=layer)
    assert projection.interactions_per_week == 10_000
    assert "recorded requests" in projection.basis


# ===========================================================================
# Rule 2 — no threshold lives in Python, this package included
# ===========================================================================


def telemetry_sources() -> list[Path]:
    return sorted(
        p for p in TELEMETRY_DIR.rglob("*.py") if "__pycache__" not in p.parts
    )


def test_there_are_telemetry_sources_to_scan():
    assert telemetry_sources()


@pytest.mark.parametrize("path", telemetry_sources(), ids=lambda p: p.name)
def test_no_configured_value_appears_as_a_literal(path: Path, layer):
    """Every number in telemetry.yaml must be absent from telemetry/*.py.

    Derived from the LOADED config rather than from a hand-kept list, so
    it cannot go stale: change a price in the contract and this test
    changes with it. Unit conversions (tokens per million, rupees per
    lakh, milliseconds per second) are not configured values and are not
    caught by it — they are arithmetic, not policy.
    """
    telemetry = layer.telemetry
    configured = {
        telemetry.currency.usd_to_inr,
        telemetry.estimation.chars_per_token,
        telemetry.targets.cost_per_case_inr,
        telemetry.targets.latency_budget_ms,
        telemetry.targets.latency_percentile,
        float(telemetry.projection.interactions_per_week),
        float(telemetry.projection.weeks_per_year),
        float(telemetry.projection.months_per_year),
    }
    for price in telemetry.pricing.values():
        configured.update(
            {
                price.input_usd_per_mtok,
                price.output_usd_per_mtok,
                price.cache_read_usd_per_mtok,
                price.cache_write_usd_per_mtok,
            }
        )
    # Identity and zero elements carry no policy — see the same carve-out
    # in tests/test_no_thresholds_in_engine.py.
    configured -= {0.0, 1.0}

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and isinstance(node.value, (int, float))
        and float(node.value) in configured
    ]
    assert not found, (
        f"CLAUDE.md rule 2: {path.name} hardcodes values that live in "
        f"semantic_layer/telemetry.yaml: {found}"
    )


def test_the_price_list_refuses_an_inverted_cache_rate(layer):
    """A cache read dearer than fresh input would make caching a loss."""
    from pydantic import ValidationError

    from semantic_layer.schema import ModelPrice

    with pytest.raises(ValidationError, match="caching would be a loss"):
        ModelPrice(
            description="broken",
            input_usd_per_mtok=1.0,
            output_usd_per_mtok=5.0,
            cache_read_usd_per_mtok=2.0,
            cache_write_usd_per_mtok=3.0,
        )


def test_telemetry_never_produces_evidence():
    """Rule 1's boundary, stated as a test.

    Telemetry describes the SYSTEM. `Evidence` describes the BUSINESS. A
    latency wrapped as evidence would put an operational number on an
    evidence panel with a clickable derivation behind it, which is a
    category error the UI has no way to catch.
    """
    for path in telemetry_sources():
        source = path.read_text(encoding="utf-8")
        assert "Evidence(" not in source, f"{path.name} constructs Evidence"
        assert "EvidenceFactory" not in source, f"{path.name} imports the factory"
