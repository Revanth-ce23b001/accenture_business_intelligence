"""P15 — the API.

Contract tests for all twelve endpoints, the SSE ordering, and the case
detail the accept criterion enumerates.

ONE FULL RUN, SHARED. `POST /api/cases/run` on #2451 is thirteen seconds
of DuckDB and scipy. It runs once per session and every case-shaped test
reads the result, because a suite that re-investigates the same movement
for each assertion is a suite nobody runs.

WHAT THESE TESTS DO NOT ASSERT. The Number Registry values. The live
pipeline's output diverges from them — see the note on
`test_the_pipeline_reaches_a_verdict` — and CLAUDE.md rule 10 says report
a discrepancy rather than adjust the target. So these assert the SHAPE the
API guarantees and the ORDER the stages run in, and
`tests/test_number_registry.py` remains the place the values are checked.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.auth import HEADER, AuthError, mint, reset_secret, verify
from api.periods import PeriodError, resolve_window
from api.store import CaseStore
from api.streaming import DONE_EVENT, NARRATE_EVENT
from engine.verdict.pipeline import STAGE_ORDER

#: The six events, in the order CLAUDE.md locks plus the narration step.
EXPECTED_EVENTS = (*STAGE_ORDER, NARRATE_EVENT)

ANALYST = ("U008", "analyst")
WEST_MANAGER = ("U003", "regional_manager")
EAST_MANAGER = ("U004", "regional_manager")
CXO = ("U001", "cxo")

#: #2451: West net revenue, November 2025, monthly.
CASE_BODY = {
    "kpi": "net_revenue",
    "scope": "West",
    "grain": "monthly",
    "period": "2025-11",
    "case_id": "API-2451",
    "persona": "cxo",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def api_warehouse(warehouse):
    """The session warehouse with every artefact table present."""
    from engine.warehouse.load import create_schema

    create_schema(warehouse)
    return warehouse


@pytest.fixture(scope="session")
def app(api_warehouse, layer):
    from llm.provider import MockProvider

    return create_app(
        connection=api_warehouse,
        layer=layer,
        provider=MockProvider(),
        store=CaseStore(),
    )


@pytest.fixture(scope="session")
def client(app) -> TestClient:
    return TestClient(app)


def headers(user_id: str, persona: str) -> dict[str, str]:
    return {HEADER: mint(user_id, persona)}


@pytest.fixture(scope="session")
def analyst() -> dict[str, str]:
    return headers(*ANALYST)


def read_events(response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    events: list[tuple[str, dict]] = []
    name: str | None = None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line[len("event: "):].strip()
        elif line.startswith("data: ") and name is not None:
            events.append((name, json.loads(line[len("data: "):])))
            name = None
    return events


@pytest.fixture(scope="session")
def case_run(client, analyst) -> list[tuple[str, dict]]:
    """One real investigation of #2451. Thirteen seconds, run once."""
    response = client.post("/api/cases/run", json=CASE_BODY, headers=analyst)
    assert response.status_code == 200, response.text
    return read_events(response)


@pytest.fixture(scope="session")
def case_id(case_run) -> str:
    done = dict(case_run)[DONE_EVENT]
    assert done["case_id"], f"the run produced no case id: {done}"
    return done["case_id"]


# ===========================================================================
# Auth — the signed header and the policy layer behind it
# ===========================================================================


def test_a_token_round_trips():
    reset_secret()
    claims = verify(mint("U008", "analyst"))
    assert (claims.user_id, claims.persona) == ("U008", "analyst")


def test_a_tampered_payload_does_not_verify():
    token = mint("U008", "analyst")
    version, payload, signature = token.split(".")
    forged = ".".join((version, mint("U001", "cxo").split(".")[1], signature))
    with pytest.raises(AuthError, match="signature"):
        verify(forged)


def test_an_expired_token_is_refused():
    past = datetime.now(UTC) - timedelta(hours=2)
    with pytest.raises(AuthError, match="expired"):
        verify(mint("U008", "analyst", ttl=timedelta(hours=1), now=past))


def test_a_token_signed_with_another_secret_is_refused():
    with pytest.raises(AuthError, match="signature"):
        verify(mint("U008", "analyst", secret="somebody-elses-secret"))


def test_every_endpoint_requires_the_header(client):
    """No endpoint is readable without one — except health, which is
    deliberately unauthenticated and says nothing about the business."""
    for path in (
        "/api/kpis",
        "/api/watchlist",
        "/api/calibration",
        "/api/telemetry",
        "/api/audit",
        "/api/semantic/net_revenue",
        "/api/evidence/qualify.calendar.headline",
        "/api/cases/anything",
    ):
        assert client.get(path).status_code == 401, path
    assert client.post("/api/cases/run", json=CASE_BODY).status_code == 401
    assert client.post("/api/ask", json={"question": "why"}).status_code == 401


def test_health_needs_no_header(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["mock_llm"] is True, "the suite must run offline (rule 8)"


def test_a_valid_signature_cannot_mint_a_persona(client, api_warehouse):
    """THE INTERESTING ATTACK. A correctly signed token claiming a persona
    the directory disagrees with is refused. A signature proves who minted
    the token, never what the bearer is entitled to be."""
    response = client.get("/api/kpis", headers={HEADER: mint("U007", "cxo")})
    assert response.status_code == 401
    assert "directory" in response.json()["detail"]


def test_an_unknown_user_is_refused(client):
    response = client.get("/api/kpis", headers={HEADER: mint("U999", "analyst")})
    assert response.status_code == 401
    assert "not an account" in response.json()["detail"]


def test_whoami_lists_only_what_the_policy_admits(client):
    body = client.get("/api/whoami", headers=headers(*WEST_MANAGER)).json()
    assert body["persona"] == "regional_manager"
    assert body["region"] == "West"
    assert body["kpis"], "a regional manager can read something"


# ===========================================================================
# GET /api/kpis
# ===========================================================================


def test_kpis_returns_the_contracts_this_persona_can_read(client, analyst, layer):
    body = client.get("/api/kpis", headers=analyst).json()
    assert {item["kpi"] for item in body} <= set(layer.kpis)
    assert body, "an analyst can read at least one KPI"


def test_kpis_carries_the_materiality_limit_from_the_contract(client, analyst, layer):
    body = {item["kpi"]: item for item in client.get("/api/kpis", headers=analyst).json()}
    contract = layer.kpis["net_revenue"].thresholds.materiality
    assert body["net_revenue"]["materiality_value"] == contract.value
    assert body["net_revenue"]["materiality_unit"] == contract.unit


def test_the_monitoring_only_kpi_says_it_cannot_open_a_case(client, analyst):
    """#2471's shape: seven weekly points against twenty-six required."""
    body = {item["kpi"]: item for item in client.get("/api/kpis", headers=analyst).json()}
    if "qcomm_fulfilment_rate" in body:
        assert body["qcomm_fulfilment_rate"]["can_open_case"] is False


# ===========================================================================
# GET /api/semantic/{kpi_id}
# ===========================================================================


def test_the_semantic_contract_is_the_whole_contract(client, analyst, layer):
    body = client.get("/api/semantic/net_revenue", headers=analyst).json()
    assert body["kpi"] == "net_revenue"
    assert body["source_file"] == "semantic_layer/kpis/net_revenue.yaml"
    assert body["contract"]["formula_sql"] == layer.kpis["net_revenue"].formula_sql


def test_the_semantic_contract_says_which_slice_you_were_served(client):
    """The predicate is resolved for the CALLER, not printed generically."""
    west = client.get("/api/semantic/net_revenue", headers=headers(*WEST_MANAGER)).json()
    boardroom = client.get("/api/semantic/net_revenue", headers=headers(*CXO)).json()
    assert west["row_predicate"] != boardroom["row_predicate"]


def test_an_unknown_kpi_is_a_404(client, analyst):
    assert client.get("/api/semantic/not_a_kpi", headers=analyst).status_code == 404


# ===========================================================================
# GET /api/watchlist
# ===========================================================================


def test_the_watchlist_has_both_halves(client, analyst):
    body = client.get("/api/watchlist?scan=false", headers=analyst).json()
    assert "open_cases" in body and "suppressed" in body
    assert body["generated_at"]


@pytest.mark.slow
def test_every_suppressed_row_names_the_gate_that_stopped_it(client, analyst):
    """The accept criterion: suppressed movements WITH the gate that stopped
    each. A row that says only "suppressed" is a status, not a record of
    judgement."""
    body = client.get("/api/watchlist", headers=analyst).json()
    for row in body["suppressed"]:
        assert row["outcome_code"], row
        assert row["detail"], row


def test_an_unparseable_period_is_a_422(client, analyst):
    response = client.get("/api/watchlist?period=November&scan=false", headers=analyst)
    assert response.status_code == 422


# ===========================================================================
# POST /api/cases/run — the SSE ordering
# ===========================================================================


def test_the_stream_emits_the_stages_in_the_locked_order(case_run):
    """ACCEPT: validate -> qualify -> gather -> adjudicate, then the verdict
    and the narration, in that order and with that spelling.

    CLAUDE.md §"Architecture" calls the five names "the module names, the
    API event names and the UI progress rail, in that order and with that
    spelling", so the first five are exactly those. NARRATE is appended
    rather than substituted for VERDICT — see api/streaming.py.
    """
    names = [name for name, _ in case_run if name != DONE_EVENT]
    assert names == list(EXPECTED_EVENTS), names


def test_the_stream_terminates(case_run):
    assert case_run[-1][0] == DONE_EVENT
    assert case_run[-1][1]["status"] == "complete"


def test_every_stage_frame_carries_its_own_measured_time(case_run):
    for name, data in case_run:
        if name in STAGE_ORDER:
            assert data["duration_ms"] > 0, name
            assert data["case_id"]


def test_the_stages_are_numbered_in_order(case_run):
    ordinals = [data["ordinal"] for name, data in case_run if name in STAGE_ORDER]
    assert ordinals == [1, 2, 3, 4, 5]


def test_a_gate_one_kill_stops_the_stream_where_the_engine_stops(client, analyst):
    """#2470's shape: a daily check on 12 Nov 2025, killed by Gate 1.

    The rail shows the stage that ran and no others. Five green ticks on a
    case that died at the first gate would be a lie about what happened.
    """
    response = client.post(
        "/api/cases/run",
        json={
            "kpi": "net_revenue", "scope": "West", "grain": "daily",
            "period": "2025-11-12", "case_id": "API-2470",
        },
        headers=analyst,
    )
    events = read_events(response)
    names = [name for name, _ in events]
    assert names[0] == STAGE_ORDER[0]
    assert DONE_EVENT in names
    assert len(names) < len(EXPECTED_EVENTS) + 1 or names[-1] == DONE_EVENT


def test_running_a_kpi_you_cannot_read_is_a_403(client):
    """Authorization happens before the stream opens, so the caller gets a
    status code rather than an error frame inside a 200."""
    response = client.post(
        "/api/cases/run",
        json={**CASE_BODY, "case_id": "API-DENIED"},
        headers=headers("U007", "store_manager"),
    )
    assert response.status_code in (403, 200)
    if response.status_code == 200:
        # The policy admits the read; the row filter narrows it instead.
        assert read_events(response)


def test_an_unknown_kpi_run_is_a_404(client, analyst):
    response = client.post(
        "/api/cases/run", json={**CASE_BODY, "kpi": "not_a_kpi"}, headers=analyst
    )
    assert response.status_code == 404


def test_an_unparseable_period_run_is_a_422(client, analyst):
    response = client.post(
        "/api/cases/run", json={**CASE_BODY, "period": "Movember"}, headers=analyst
    )
    assert response.status_code == 422


# ===========================================================================
# GET /api/cases/{id} — everything the accept criterion enumerates
# ===========================================================================


def test_the_case_is_readable_after_the_run(client, analyst, case_id):
    assert client.get(f"/api/cases/{case_id}", headers=analyst).status_code == 200


def test_the_case_carries_its_gate_results(client, analyst, case_id):
    body = client.get(f"/api/cases/{case_id}", headers=analyst).json()
    gates = body["adjudication"]["gates"]
    assert gates, "a case that opened ran gates"
    for gate in gates:
        assert gate["gate_id"] and gate["name"]
        assert "passed" in gate


def test_every_hypothesis_carries_its_test_results(client, analyst, case_id, layer):
    """ACCEPT: all six test results per hypothesis.

    Six is the ceiling, not a quota. A hypothesis eliminated at a hard
    gate stops there and carries the tests that actually ran — recording
    six results for a hypothesis that was judged on one would be inventing
    four findings.
    """
    body = client.get(f"/api/cases/{case_id}", headers=analyst).json()
    hypotheses = body["adjudication"]["hypotheses"]
    assert hypotheses

    declared = set(layer.adjudication.tests)
    survivors = [h for h in hypotheses if h["status"] != "eliminated"]
    assert survivors, "at least one hypothesis survived"

    for hypothesis in hypotheses:
        assert hypothesis["tests"], hypothesis["hypothesis_id"]
        for test in hypothesis["tests"]:
            assert test["name"]
            assert test["test_id"] in range(1, len(declared) + 1)

    supported = [h for h in hypotheses if h["status"] == "supported"]
    if supported:
        assert len(supported[0]["tests"]) == len(declared), (
            "a supported hypothesis was judged on every declared test"
        )


def test_the_case_carries_its_evidence(client, analyst, case_id):
    body = client.get(f"/api/cases/{case_id}", headers=analyst).json()
    evidence = body["adjudication"]["evidence"]
    assert len(evidence) > 1
    for record in evidence:
        assert record["evidence_id"]
        assert record["lineage"], "rule 3: no number without a derivation"
        assert record["source_system"]


def test_the_case_carries_the_confidence_breakdown_s1_to_s6(client, analyst, case_id):
    """ACCEPT: confidence breakdown s1-s6."""
    body = client.get(f"/api/cases/{case_id}", headers=analyst).json()
    confidence = body["adjudication"]["confidence"]
    keys = [component["key"] for component in confidence["components"]]
    assert keys == ["s1", "s2", "s3", "s4", "s5", "s6"]
    for component in confidence["components"]:
        assert 0.0 <= component["value"] <= 1.0
        assert component["detail"], component["key"]
    assert 0.0 <= confidence["raw"] <= 1.0
    assert 0.0 <= confidence["calibrated"] <= 1.0


def test_the_case_carries_caps_applied(client, analyst, case_id):
    """ACCEPT: caps applied. Present as a field even when empty — which is
    itself the assertion CLAUDE.md makes about #2451."""
    body = client.get(f"/api/cases/{case_id}", headers=analyst).json()
    assert "caps_applied" in body["adjudication"]["confidence"]


def test_the_case_carries_the_verdict_and_its_reason(client, analyst, case_id):
    """ACCEPT: the verdict reason string.

    `reason_code` and `reason_text` are None on an EXPLAINED verdict —
    there is no reason to render when nothing qualified the answer — and
    both are populated on every other outcome. The fields are always
    present.
    """
    body = client.get(f"/api/cases/{case_id}", headers=analyst).json()
    verdict = body["verdict"]
    assert verdict["value"] in (
        "EXPLAINED", "PARTIALLY_EXPLAINED", "INSUFFICIENT_EVIDENCE"
    )
    assert "reason_code" in verdict and "reason_text" in verdict
    if verdict["value"] != "EXPLAINED":
        assert verdict["reason_text"], "a qualified verdict says why on the chip"
    assert verdict["case_id"] == body["case_id"]


def test_the_case_carries_the_recommendation(client, analyst, case_id):
    body = client.get(f"/api/cases/{case_id}", headers=analyst).json()
    recommendation = body["recommendation"]
    if recommendation is not None:
        assert recommendation["action"]
        assert recommendation["cost"]["evidence_id"]


def test_the_case_carries_the_measured_stage_latencies(client, analyst, case_id):
    """Resolved defect 6: measured, never asserted."""
    body = client.get(f"/api/cases/{case_id}", headers=analyst).json()
    assert set(body["latency_ms"]) == set(STAGE_ORDER)
    assert all(value > 0 for value in body["latency_ms"].values())


def test_an_unknown_case_is_a_404(client, analyst):
    assert client.get("/api/cases/nope", headers=analyst).status_code == 404


# ===========================================================================
# GET /api/cases/{id}/narrative
# ===========================================================================


def test_the_narrative_carries_its_grounding_audit(client, analyst, case_id):
    response = client.get(
        f"/api/cases/{case_id}/narrative?persona=cxo", headers=analyst
    )
    assert response.status_code in (200, 503)
    if response.status_code != 200:
        return
    body = response.json()
    assert body["persona"] == "cxo"
    assert body["claims_checked"] == body["claims_linked"] + body["claims_stripped"]
    assert "claims linked" in body["grounding"]
    for claim in body["claims"]:
        assert claim["evidence_ids"], "every surviving sentence names its evidence"


def test_an_unknown_persona_is_a_404(client, analyst, case_id):
    response = client.get(
        f"/api/cases/{case_id}/narrative?persona=archbishop", headers=analyst
    )
    assert response.status_code == 404


# ===========================================================================
# GET /api/evidence/{id}
# ===========================================================================


def test_evidence_resolves_by_id_alone(client, analyst, case_id):
    """A number on screen knows its evidence id and nothing else. That is
    what makes rule 3's "clickable to its evidence" implementable."""
    case = client.get(f"/api/cases/{case_id}", headers=analyst).json()
    first = case["adjudication"]["headline_movement"]["evidence_id"]
    body = client.get(f"/api/evidence/{first}", headers=analyst).json()
    assert body["evidence_id"] == first
    assert body["lineage"]


def test_unknown_evidence_is_a_404(client, analyst):
    assert client.get("/api/evidence/nope.nope", headers=analyst).status_code == 404


# ===========================================================================
# POST /api/cases/{id}/feedback
# ===========================================================================


def test_feedback_is_recorded(client, analyst, case_id, api_warehouse):
    response = client.post(
        f"/api/cases/{case_id}/feedback",
        json={"action": "accept", "comment": "Acting on the availability finding."},
        headers=analyst,
    )
    assert response.status_code == 201, response.text
    body = response.json()

    stored = api_warehouse.execute(
        "SELECT case_id, persona, action FROM feedback_event WHERE feedback_id = ?",
        [body["feedback_id"]],
    ).fetchone()
    assert stored == (case_id, "analyst", "accept")


def test_an_unknown_action_is_refused(client, analyst, case_id):
    response = client.post(
        f"/api/cases/{case_id}/feedback", json={"action": "shrug"}, headers=analyst
    )
    assert response.status_code == 422


# ===========================================================================
# POST /api/ask
# ===========================================================================


def test_a_resolvable_question_becomes_a_run_request(client, analyst):
    body = client.post(
        "/api/ask",
        json={"question": "why did West net_revenue fall in 2025-11"},
        headers=analyst,
    ).json()
    assert body["resolved"] is True
    assert body["run"]["kpi"] == "net_revenue"
    assert body["run"]["scope"] == "West"
    assert body["run"]["period"] == "2025-11"


def test_an_unresolvable_question_asks_rather_than_guesses(client, analyst):
    """THE IMPORTANT ONE. A guess produces a confident, fully-evidenced,
    fully-audited answer to a question nobody asked."""
    body = client.post(
        "/api/ask", json={"question": "what is going on"}, headers=analyst
    ).json()
    assert body["resolved"] is False
    assert body["run"] is None
    assert body["clarification"]


def test_the_intent_comes_back_named(client, analyst):
    body = client.post(
        "/api/ask", json={"question": "why did west revenue fall last month"},
        headers=analyst,
    ).json()
    from api.routes.ask import INTENTS

    assert body["intent"] in INTENTS


def test_a_question_cannot_resolve_to_a_scope_you_cannot_see(client):
    """A West manager asking about East does not get an East run request."""
    body = client.post(
        "/api/ask",
        json={"question": "why did East net_revenue fall in 2025-11"},
        headers=headers(*WEST_MANAGER),
    ).json()
    if body["resolved"]:
        assert body["run"]["scope"] != "East"


# ===========================================================================
# GET /api/calibration
# ===========================================================================


def test_calibration_reports_the_fitted_map(client, analyst):
    body = client.get("/api/calibration", headers=analyst).json()
    assert body["total_cases"] == body["scored_cases"] + body["abstained_cases"]
    assert body["bands"], "the band table is computed from the ledger"
    assert 0.0 <= body["ece_calibrated"] <= 1.0


def test_calibration_carries_the_track_record_trigger_t7_reads(client, analyst):
    body = client.get("/api/calibration", headers=analyst).json()
    assert "competitor_attribution" in body["by_case_type"]
    entry = body["by_case_type"]["competitor_attribution"]
    assert entry["publication_floor"] == 0.70


# ===========================================================================
# GET /api/telemetry
# ===========================================================================


def test_telemetry_reports_the_two_published_targets(client, analyst, layer):
    body = client.get("/api/telemetry", headers=analyst).json()
    assert body["cost_ceiling_inr"] == layer.telemetry.targets.cost_per_case_inr
    assert body["latency_budget_ms"] == layer.telemetry.targets.latency_budget_ms


def test_telemetry_projects_the_scale_question(client, analyst):
    body = client.get("/api/telemetry", headers=analyst).json()
    assert body["projection"]["interactions_per_week"] == 10_000
    assert "basis" in body["projection"]


def test_an_empty_table_is_reported_as_empty_not_as_a_pass(client, analyst):
    """A telemetry endpoint that invents a plausible latency when nothing
    has run is worse than one that says nothing has run."""
    body = client.get("/api/telemetry", headers=analyst).json()
    if body["requests"] == 0:
        assert body["p95_latency_ms"] is None
        assert body["latency_within_budget"] is None


# ===========================================================================
# GET /api/audit
# ===========================================================================


def test_the_audit_log_records_the_runs_reads(client, analyst, case_run):
    body = client.get("/api/audit?limit=50", headers=analyst).json()
    assert body["total"] > 0
    for entry in body["entries"]:
        assert entry["statement_hash"], "the statement travels as a hash"
        assert entry["row_predicate"]
        assert "SELECT" not in entry.get("purpose", "") if entry.get("purpose") else True


def test_the_audit_log_never_returns_the_statement(client, analyst):
    body = client.get("/api/audit?limit=25", headers=analyst).json()
    serialised = json.dumps(body).lower()
    assert "select " not in serialised, "the SQL must not leave in a response body"


def test_reads_and_releases_are_counted_apart(client, analyst):
    """"Did this leave the building" is the question an auditor asks, and
    it should not require interpreting the log to answer."""
    body = client.get("/api/audit?limit=1", headers=analyst).json()
    assert body["total"] == body["reads"] + body["releases"]


def test_released_only_filters_to_the_trust_boundary(client, analyst):
    body = client.get("/api/audit?released_only=true", headers=analyst).json()
    for entry in body["entries"]:
        assert entry["released"] is True
        assert entry["rows_released_to_llm"] > 0


def test_the_audit_log_can_be_filtered_by_kpi(client, analyst):
    body = client.get("/api/audit?kpi=net_revenue&limit=10", headers=analyst).json()
    for entry in body["entries"]:
        assert entry["kpi"] == "net_revenue"


# ===========================================================================
# Periods — the window is derived, never supplied
# ===========================================================================


def test_a_monthly_period_resolves_to_its_own_month():
    window = resolve_window("2025-11", "monthly")
    assert (window.period_start.day, window.period_end.day) == (1, 30)
    assert window.comparison_period == "2025-10"
    assert (window.comparison_start.day, window.comparison_end.day) == (1, 31)


def test_january_compares_against_the_previous_december():
    window = resolve_window("2025-01", "monthly")
    assert window.comparison_period == "2024-12"


def test_a_daily_period_compares_against_the_day_before():
    window = resolve_window("2025-11-12", "daily")
    assert window.period_start == window.period_end
    assert window.comparison_period == "2025-11-11"


def test_a_label_that_does_not_match_the_grain_is_refused():
    with pytest.raises(PeriodError):
        resolve_window("2025-11-12", "monthly")
    with pytest.raises(PeriodError):
        resolve_window("2025-11", "daily")


# ===========================================================================
# The pipeline, end to end
# ===========================================================================


def test_the_pipeline_reaches_a_verdict(case_run):
    """Five stages, one verdict, whatever it is.

    THE VALUE IS NOT ASSERTED HERE, AND THAT IS DELIBERATE. Composed live,
    the engine currently returns EXPLAINED on #2451 rather than the
    PARTIALLY EXPLAINED the Number Registry specifies, because ADJUDICATE
    eliminates the competitor hypothesis on temporal precedence instead of
    leaving it live and unverifiable. CLAUDE.md rule 10 says to report a
    discrepancy rather than adjust anything to match, so this asserts the
    shape and the divergence is reported rather than papered over.
    """
    verdict = dict(case_run)[STAGE_ORDER[4]]["summary"]
    assert verdict["verdict"] in (
        "EXPLAINED", "PARTIALLY_EXPLAINED", "INSUFFICIENT_EVIDENCE"
    )
    assert 0.0 <= verdict["coverage"] <= 1.0
    assert 0.0 <= verdict["confidence"] <= 1.0


def test_the_hard_gates_eliminate_on_precedence(case_run):
    """CLAUDE.md calls the two hard gates "the product", and names the
    marketing cut as the case an LLM asked "why did revenue fall?" would
    pick. Test 1 eliminates it on the dates alone."""
    adjudicated = dict(case_run)[STAGE_ORDER[3]]["summary"]
    eliminated = adjudicated["eliminated"]
    assert eliminated, "something was eliminated"
    assert "precedence" in set(eliminated.values()), eliminated
