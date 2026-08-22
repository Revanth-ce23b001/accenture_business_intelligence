"""Every contract round-trips through JSON, and the structural rules hold."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engine.contracts import (
    Adjudication,
    ConfidenceBreakdown,
    Evidence,
    GateResult,
    Hypothesis,
    LineageStep,
    Recommendation,
    TelemetryEvent,
    TestResult,
    Verdict,
)
from tests.conftest import (
    make_adjudication,
    make_confidence,
    make_evidence,
    make_gate_result,
    make_hypothesis,
    make_lineage,
    make_recommendation,
    make_telemetry_event,
    make_test_result,
    make_verdict,
)

# The ten contracts named in the P1 brief, each with a builder.
CONTRACTS = [
    (Evidence, make_evidence),
    (LineageStep, make_lineage),
    (TestResult, make_test_result),
    (Hypothesis, make_hypothesis),
    (GateResult, make_gate_result),
    (Adjudication, make_adjudication),
    (ConfidenceBreakdown, make_confidence),
    (Verdict, make_verdict),
    (Recommendation, make_recommendation),
    (TelemetryEvent, make_telemetry_event),
]


@pytest.mark.parametrize("model_cls,builder", CONTRACTS, ids=lambda v: getattr(v, "__name__", ""))
def test_round_trips_through_json(model_cls, builder):
    original = builder()
    restored = model_cls.model_validate_json(original.model_dump_json())
    assert restored == original


@pytest.mark.parametrize("model_cls,builder", CONTRACTS, ids=lambda v: getattr(v, "__name__", ""))
def test_round_trips_through_dict(model_cls, builder):
    original = builder()
    restored = model_cls.model_validate(original.model_dump(mode="json"))
    assert restored == original


@pytest.mark.parametrize("model_cls,builder", CONTRACTS, ids=lambda v: getattr(v, "__name__", ""))
def test_contracts_are_frozen(model_cls, builder):
    instance = builder()
    field = next(iter(model_cls.model_fields))
    with pytest.raises(ValidationError):
        setattr(instance, field, None)


@pytest.mark.parametrize("model_cls,builder", CONTRACTS, ids=lambda v: getattr(v, "__name__", ""))
def test_unknown_fields_are_rejected(model_cls, builder):
    payload = builder().model_dump(mode="json")
    payload["definitely_not_a_field"] = 1
    with pytest.raises(ValidationError):
        model_cls.model_validate(payload)


# --- rule 1 -----------------------------------------------------------------


def test_model_produced_evidence_may_not_carry_a_number():
    """CLAUDE.md rule 1: the LLM is never the source of a number."""
    with pytest.raises(ValidationError, match="rule 1"):
        make_evidence(produced_by="model", value=4.10)


def test_model_produced_evidence_may_carry_text():
    evidence = make_evidence(
        evidence_id="ev.note",
        kind="store_note",
        produced_by="model",
        value="Customers asked for sizes we did not have.",
        unit=None,
        reliability=0.55,
    )
    assert evidence.produced_by == "model"


def test_code_produced_evidence_may_carry_a_number():
    assert make_evidence(produced_by="code", value=4.10).value == pytest.approx(4.10)


# --- structural invariants --------------------------------------------------


def test_eliminated_hypothesis_requires_a_reason():
    payload = make_hypothesis().model_dump(mode="json")
    payload["status"] = "eliminated"
    payload["elimination_reason"] = None
    with pytest.raises(ValidationError, match="elimination_reason"):
        Hypothesis.model_validate(payload)


def test_eliminated_hypothesis_with_a_reason_is_accepted():
    payload = make_hypothesis().model_dump(mode="json")
    payload.update({"status": "eliminated", "elimination_reason": "precedence"})
    assert Hypothesis.model_validate(payload).elimination_reason == "precedence"


def test_reliability_is_bounded():
    with pytest.raises(ValidationError):
        make_evidence(reliability=1.4)


def test_trigger_ids_are_a_closed_set():
    payload = make_adjudication().model_dump(mode="json")
    payload["triggers_fired"] = ["T9"]
    with pytest.raises(ValidationError):
        Adjudication.model_validate(payload)


def test_verdict_values_are_a_closed_set():
    payload = make_verdict().model_dump(mode="json")
    payload["value"] = "MOSTLY_EXPLAINED"
    with pytest.raises(ValidationError):
        Verdict.model_validate(payload)


def test_gate_ids_are_bounded_to_five():
    payload = make_gate_result().model_dump(mode="json")
    payload["gate_id"] = 6
    with pytest.raises(ValidationError):
        GateResult.model_validate(payload)


def test_case_level_quantities_are_evidence_not_floats():
    """Rule 3 — no bare floats leave the engine."""
    adjudication = make_adjudication()
    assert isinstance(adjudication.qualified_residual, Evidence)
    assert isinstance(adjudication.headline_movement, Evidence)
    assert isinstance(adjudication.materiality, Evidence)
