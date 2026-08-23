"""Shared builders and fixtures.

The contract builders here are structurally complete but carry no Number
Registry values — reproducing those is `tests/test_number_registry.py`.

The `warehouse` fixture builds the whole DuckDB warehouse once per test
session, in memory. It takes about twenty seconds, which is why it is
session-scoped; tests that mutate it (the audit and gap-register tests)
clear their own table first rather than asking for a fresh build.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

# CLAUDE.md rule 8: the full demo runs offline from recorded fixtures, and
# so does the suite. Set before anything imports `llm.provider`, so a test
# that forgets to pass a provider gets the mock rather than a socket.
os.environ.setdefault("MOCK_LLM", "true")

from engine.contracts import (
    Adjudication,
    CapApplied,
    ConfidenceBreakdown,
    ConfidenceComponent,
    Evidence,
    GateResult,
    Hypothesis,
    LineageStep,
    Recommendation,
    TelemetryEvent,
    TestResult,
    Verdict,
)
from llm.provider import LLMMessage, LLMRequest

FIXED_TS = datetime(2025, 11, 30, 12, 0, 0, tzinfo=UTC)


def make_lineage_step(step: int = 0) -> LineageStep:
    return LineageStep(
        step=step,
        operation="sql",
        description="Sum net revenue by region and month.",
        inputs=("src.sales_daily",),
        ref="semantic_layer/kpis/net_revenue.yaml",
        statement="SELECT region, SUM(net_revenue_inr) FROM fact_sales_daily GROUP BY 1",
    )


def make_evidence(evidence_id: str = "ev.001", **overrides) -> Evidence:
    payload = dict(
        evidence_id=evidence_id,
        kind="structured_query",
        produced_by="code",
        label="West net revenue, month M",
        value=0.0,
        unit="INR_CR",
        reliability=0.95,
        source_system="pos",
        method="sql",
        source_ref="q.net_revenue_by_region_month",
        source_as_of=FIXED_TS,
        retrieved_at=FIXED_TS,
        freshness_hours=1.5,
        completeness=1.0,
        lineage=(make_lineage_step(),),
        notes=None,
    )
    payload.update(overrides)
    return Evidence(**payload)


def make_lineage() -> LineageStep:
    return make_lineage_step()


def make_test_result() -> TestResult:
    return TestResult(
        test_id=5,
        name="Matched-control DiD",
        type="weighted",
        passed=True,
        statistic=0.0,
        p_value=0.5,
        weight=0.45,
        detail="Treated vs matched control, parallel-trends pre-test passed.",
        evidence_ids=("ev.001",),
    )


def make_gate_result() -> GateResult:
    return GateResult(
        gate_id=1,
        name="Data completeness",
        passed=False,
        outcome_code="DATA_INCIDENT",
        detail="Store feeds failed to load for the period.",
        evidence_ids=("ev.001",),
    )


def make_hypothesis() -> Hypothesis:
    return Hypothesis(
        hypothesis_id="H1",
        label="Availability collapse on top SKUs",
        description="Treated stores lost availability on the top-20 SKUs.",
        status="supported",
        elimination_reason=None,
        tests=(make_test_result(),),
        required_sources=("inventory_snapshots", "store_notes"),
        missing_sources=(),
        verifiable=True,
        attributed_share=0.0,
        attributed=make_evidence("ev.attr", unit="pt"),
        residual_held=None,
        evidence_ids=("ev.001",),
    )


def make_confidence() -> ConfidenceBreakdown:
    return ConfidenceBreakdown(
        components=(
            ConfidenceComponent(
                key="s1",
                name="Evidence strength",
                value=0.0,
                weight=0.28,
                detail="sigmoid(abs(did_t_stat))",
                evidence_ids=("ev.001",),
            ),
        ),
        raw=0.0,
        caps_applied=(
            CapApplied(
                name="missing_required_source",
                condition="A source required by this hypothesis is missing",
                ceiling=0.45,
                forced_trigger="T3",
            ),
        ),
        after_caps=0.0,
        calibrated=0.0,
        calibration_method="isotonic",
        calibration_sample_size=0,
    )


def make_adjudication() -> Adjudication:
    return Adjudication(
        case_id="CASE-TEST",
        kpi="net_revenue",
        scope="West",
        grain="monthly",
        period="2025-11",
        opened_at=FIXED_TS,
        status="in_progress",
        headline_movement=make_evidence("ev.headline", unit="pct"),
        attributed=(make_evidence("ev.calendar", unit="pt"),),
        qualified_residual=make_evidence("ev.residual", unit="pt"),
        materiality=make_evidence("ev.materiality", unit="INR_CR"),
        empirical_band=make_evidence("ev.band", unit="pt"),
        gates=(make_gate_result(),),
        hypotheses=(make_hypothesis(),),
        coverage=0.0,
        confidence=make_confidence(),
        triggers_fired=("T3", "T7"),
        evidence=(make_evidence(),),
        elapsed_ms=0.0,
    )


def make_verdict() -> Verdict:
    return Verdict(
        case_id="CASE-TEST",
        value="PARTIALLY_EXPLAINED",
        reason_code="live_unverifiable_above_materiality",
        reason_text="A live, unverifiable hypothesis holds residual above materiality.",
        triggers_fired=(),
        coverage=0.0,
        confidence_calibrated=0.0,
        decided_at=FIXED_TS,
    )


def make_recommendation() -> Recommendation:
    return Recommendation(
        case_id="CASE-TEST",
        action="Expedite logistics to the treated stores.",
        playbook_ref="playbooks/availability_recovery.yaml",
        cost=make_evidence("ev.cost", unit="INR_L"),
        expected_recovery_low=make_evidence("ev.rec.low", unit="INR_CR"),
        expected_recovery_high=make_evidence("ev.rec.high", unit="INR_CR"),
        roi_low=0.0,
        roi_high=0.0,
        recovery_confidence="Medium",
        sample_size=3,
        effort="9 managers, 2 hours",
        not_recommended=("Price response — high margin exposure, coin-flip odds.",),
        linked_case_id="CASE-TEST-LINKED",
        evidence_ids=("ev.001",),
    )


def make_telemetry_event() -> TelemetryEvent:
    return TelemetryEvent(
        event_id="tel.001",
        event="llm_call",
        stage="ADJUDICATE",
        case_id="CASE-TEST",
        started_at=FIXED_TS,
        duration_ms=0.0,
        model="claude-haiku-4-5",
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        cost_inr=0.0,
        mock=True,
        detail=None,
    )


def smoke_request() -> LLMRequest:
    """The canonical request backing the committed fixture in llm/fixtures/.

    Changing this function changes its fingerprint and invalidates the
    committed fixture. That is intentional: the test will fail loudly and
    print the new fingerprint.
    """
    return LLMRequest.for_task(
        "classify",
        system="You classify analyst questions into intents. Reply with one token.",
        messages=(LLMMessage(role="user", content="why did west revenue fall last month"),),
        max_tokens=16,
    )


@pytest.fixture
def smoke_llm_request() -> LLMRequest:
    return smoke_request()


# ---------------------------------------------------------------------------
# Warehouse fixtures (P4)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def layer():
    """The loaded, cross-checked semantic layer."""
    from semantic_layer.schema import load_semantic_layer

    return load_semantic_layer()


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory):
    """A fully loaded warehouse, opened once for the session.

    Parsing 4.5 million rows of CSV takes minutes, so the build is cached:
    `data/casefile.duckdb` is created on first use (or by `make seed`) and
    every later run copies it, which takes about a second.

    The COPY is what the tests get. Two of them write to `audit_log` and
    `data_gap_register`, and a test run must not leave rows behind in the
    warehouse a developer is working against.
    """
    import shutil

    from engine.db import DEFAULT_DB_PATH, connect
    from engine.warehouse.load import build_warehouse

    if not DEFAULT_DB_PATH.exists():
        build_warehouse(DEFAULT_DB_PATH)

    scratch = tmp_path_factory.mktemp("warehouse") / DEFAULT_DB_PATH.name
    shutil.copyfile(DEFAULT_DB_PATH, scratch)

    connection = connect(scratch)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture(scope="session")
def reconciliation(warehouse, layer):
    """The four disagreements, measured against the loaded warehouse."""
    from engine.warehouse.reconcile import reconcile

    return reconcile(warehouse, layer=layer)
