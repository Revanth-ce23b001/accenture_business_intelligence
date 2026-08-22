"""Typed contracts for CaseFile.ai.

These Pydantic v2 models *are* the evidence and adjudication contracts
(CLAUDE.md §"Stack"). Everything the engine emits and everything the API
serialises is one of these types.

Design constraints taken from CLAUDE.md:

  rule 1  The LLM is never the source of a number. Enforced on `Evidence`:
          a `produced_by="model"` record may not carry a numeric `value`.
  rule 2  No threshold lives in Python. Nothing here carries a default drawn
          from the semantic layer — `reliability`, `weight`, `materiality`
          and friends are *required* inputs supplied from
          `semantic_layer/*.yaml`. There are deliberately no magic numbers
          in this module.
  rule 3  No bare floats leave the engine. Case-level quantities on
          `Adjudication` and `Recommendation` are typed `Evidence`, not
          `float`, so the UI can always link a displayed number to its
          derivation.
  rule 6  Abstention is architectural. `TriggerId` is a closed set evaluated
          outside the model.
  rule 7  Contribution (WHERE) and causation (WHY) stay separate. There is no
          field on `Hypothesis` that holds a contribution decomposition, and
          no field on `Adjudication.attributed` that holds a hypothesis.

Sequence containers are tuples: these are value objects, and the models are
frozen so an adjudication cannot be mutated after it is handed to the
narrative layer.

No business logic lives here. Validators enforce structural invariants only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------

#: The five pipeline stages, locked from the Round 1 deck (CLAUDE.md
#: §"Architecture"). These are the module names, the API event names and the
#: UI progress rail, in this order and with this spelling.
Stage = Literal["VALIDATE", "QUALIFY", "GATHER", "ADJUDICATE", "VERDICT"]

#: Which side of the code/model line produced a value. Drives the monospace
#: vs. rounded badge in the UI (CLAUDE.md §"Stack" → Palette).
ProducedBy = Literal["code", "model"]

#: The seven evidence tiers in CLAUDE.md §"Reliability weights for evidence",
#: in descending weight order. The *weights themselves* live in the semantic
#: layer, not here (rule 2) — this enumerates the kinds only.
EvidenceKind = Literal[
    "structured_query",           # warehouse query
    "derived_estimate",           # derived statistical estimate
    "corroborated_unstructured",  # n>=20 independent notes
    "ticket_aggregate",           # support ticket aggregate
    "store_note",                 # single store note
    "news_item",                  # external news item
    "social_mention",             # social / review mention
]

#: CLAUDE.md §"The six adjudication tests": tests 1-2 are hard gates, tests
#: 3-5 carry weight, test 6 applies a confidence cap.
TestType = Literal["hard_gate", "weighted", "cap"]

HypothesisStatus = Literal["live", "supported", "eliminated"]

#: CLAUDE.md §"Abstention triggers". Each is a named boolean surfaced by name
#: in the UI when it fires.
TriggerId = Literal["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]

#: CLAUDE.md §"Verdict decision table". Three outcomes, no others.
VerdictValue = Literal["EXPLAINED", "PARTIALLY_EXPLAINED", "INSUFFICIENT_EVIDENCE"]

Grain = Literal["daily", "weekly", "monthly"]

#: CLAUDE.md §"Number Registry" → #2472 carries adjudication `in_progress`.
AdjudicationStatus = Literal["in_progress", "closed"]

#: CLAUDE.md §"Number Registry" → #2451 "Recovery confidence Medium (n=3)".
RecoveryConfidence = Literal["Low", "Medium", "High"]


class _Contract(BaseModel):
    """Base for every contract: frozen, and unknown fields are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Evidence and lineage
# ---------------------------------------------------------------------------


class LineageStep(_Contract):
    """One traceable step from raw source to a computed value.

    Rule 3 requires every number in the UI to be clickable through to its
    evidence; the lineage is what that click reveals.
    """

    step: int = Field(ge=0, description="0-based position in the derivation chain")
    operation: str = Field(description="e.g. 'sql', 'stl_decompose', 'match', 'did', 'ols'")
    description: str = Field(description="Human-readable statement of what this step did")
    inputs: tuple[str, ...] = Field(
        default=(), description="evidence_ids or named artefacts consumed by this step"
    )
    ref: str | None = Field(
        default=None,
        description="Pointer to the producing artefact: query name, module:function, or YAML key",
    )


class Evidence(_Contract):
    """A single addressable fact.

    Rule 3: modules emit `Evidence`, never bare floats. Rule 1: if this record
    came from the model, it carries no number.
    """

    evidence_id: str = Field(description="Stable id; referenced by narrative sentences")
    kind: EvidenceKind
    produced_by: ProducedBy
    label: str = Field(description="What this evidence asserts, in one line")

    value: float | int | str | None = Field(
        default=None,
        description="The quantity. None for qualitative evidence and for all model output.",
    )
    unit: str | None = Field(
        default=None,
        description="e.g. 'INR_CR', 'INR_L', 'pt', 'pct', 'count', 'ratio', 'hours'",
    )

    reliability: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Reliability weight for this evidence tier. Supplied from the semantic layer — "
            "never defaulted in Python (rule 2)."
        ),
    )

    source_ref: str = Field(description="Query name, document id, ticket filter, or URL")
    as_of: datetime = Field(description="Timestamp the underlying source was current as of")
    freshness_hours: float = Field(
        ge=0.0, description="Age at retrieval; feeds s4 data quality and trigger T8"
    )
    completeness: float = Field(
        ge=0.0, le=1.0, description="Share of expected records present; feeds s4 with freshness"
    )

    lineage: tuple[LineageStep, ...] = Field(default=())
    notes: str | None = Field(default=None)

    @model_validator(mode="after")
    def _model_output_carries_no_number(self) -> Evidence:
        """Rule 1 — the LLM is never the source of a number.

        A model-produced evidence record may carry text, never a quantity.
        If you need a number the model mentioned, compute it in Python and
        emit it as separate `produced_by="code"` evidence.
        """
        if self.produced_by == "model" and isinstance(self.value, (int, float)):
            raise ValueError(
                f"Evidence {self.evidence_id!r}: produced_by='model' cannot carry a numeric "
                "value (CLAUDE.md rule 1). Compute the number in Python and emit it as "
                "produced_by='code'."
            )
        return self


# ---------------------------------------------------------------------------
# Gates and tests
# ---------------------------------------------------------------------------


class GateResult(_Contract):
    """Outcome of one pipeline gate.

    NOTE: CLAUDE.md refers to "Gates 1-5" in §"Architecture" and names the
    effect of Gate 1 (`DATA_INCIDENT`, case #2470) and Gate 3
    (`INSUFFICIENT_HISTORY`, case #2471), but never enumerates all five.
    `gate_id` is therefore a bounded int and `outcome_code` a free string
    rather than Literals. Tighten both once the gates are specified.
    """

    gate_id: int = Field(ge=1, le=5)
    name: str
    passed: bool
    outcome_code: str | None = Field(
        default=None, description="Set when the gate stops the case, e.g. 'DATA_INCIDENT'"
    )
    detail: str
    evidence_ids: tuple[str, ...] = Field(default=())


class TestResult(_Contract):
    """Outcome of one of the six adjudication tests, for one hypothesis."""

    # Name begins with "Test"; tell pytest this is a contract, not a test class.
    __test__ = False

    test_id: int = Field(ge=1, le=6, description="1..6 per CLAUDE.md §'The six adjudication tests'")
    name: str
    type: TestType
    passed: bool
    statistic: float | None = Field(default=None, description="t, r, chi-square, etc.")
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Only meaningful for type='weighted'; supplied from the semantic layer",
    )
    detail: str
    evidence_ids: tuple[str, ...] = Field(default=())


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------


class Hypothesis(_Contract):
    """One candidate explanation for the qualified residual."""

    hypothesis_id: str = Field(description="e.g. 'H1'")
    label: str
    description: str
    status: HypothesisStatus
    elimination_reason: str | None = Field(
        default=None,
        description="Required when status='eliminated', e.g. 'precedence', 'sufficiency'",
    )

    tests: tuple[TestResult, ...] = Field(default=())

    required_sources: tuple[str, ...] = Field(
        default=(), description="Sources this hypothesis needs to be verifiable"
    )
    missing_sources: tuple[str, ...] = Field(
        default=(), description="Subset of required_sources the organisation does not hold"
    )
    verifiable: bool = Field(
        description="False when a required source is missing; drives 'live-unverifiable'"
    )

    attributed_share: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Share of the qualified residual this hypothesis accounts for",
    )
    attributed: Evidence | None = Field(
        default=None, description="The attributed quantity itself, as evidence (rule 3)"
    )
    residual_held: Evidence | None = Field(
        default=None,
        description=(
            "Residual this hypothesis holds but cannot verify. Compared against materiality "
            "by the verdict decision table."
        ),
    )

    evidence_ids: tuple[str, ...] = Field(default=())

    @model_validator(mode="after")
    def _eliminated_needs_reason(self) -> Hypothesis:
        if self.status == "eliminated" and not self.elimination_reason:
            raise ValueError(
                f"Hypothesis {self.hypothesis_id!r}: status='eliminated' requires "
                "elimination_reason (it is surfaced in the UI)."
            )
        return self


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class ConfidenceComponent(_Contract):
    """One of s1..s6 in CLAUDE.md §"Confidence specification"."""

    key: Literal["s1", "s2", "s3", "s4", "s5", "s6"]
    name: str
    value: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0, description="Supplied from the semantic layer (rule 2)")
    detail: str | None = Field(default=None)
    evidence_ids: tuple[str, ...] = Field(default=())


class CapApplied(_Contract):
    """A confidence cap that fired.

    Caps are applied after the weighted sum and cannot be outvoted
    (CLAUDE.md §"Confidence specification").
    """

    name: str
    condition: str = Field(description="The condition that fired, in words")
    ceiling: float | None = Field(default=None, ge=0.0, le=1.0)
    forced_trigger: TriggerId | None = Field(
        default=None, description="Some caps force an abstention trigger, e.g. missing source -> T3"
    )


class ConfidenceBreakdown(_Contract):
    """The full confidence derivation, raw through calibrated.

    `caps_applied` is empty when no cap fires — which is the case for #2451.
    See the critical clarification in CLAUDE.md §"Confidence specification".
    """

    components: tuple[ConfidenceComponent, ...]
    raw: float = Field(ge=0.0, le=1.0, description="The weighted sum, before caps")
    caps_applied: tuple[CapApplied, ...] = Field(default=())
    after_caps: float = Field(ge=0.0, le=1.0)
    calibrated: float = Field(ge=0.0, le=1.0, description="Published value, after the isotonic map")
    calibration_method: str = Field(default="isotonic")
    calibration_sample_size: int | None = Field(
        default=None, ge=0, description="Closed cases backing the calibration for this case type"
    )


# ---------------------------------------------------------------------------
# Adjudication, verdict, recommendation
# ---------------------------------------------------------------------------


class Adjudication(_Contract):
    """The frozen object handed to the narrative layer.

    Rule 1: the model receives this *after* adjudication and may render it,
    never alter it. The model is not given a writable path to any field here.
    """

    case_id: str
    kpi: str = Field(description="e.g. 'net_revenue', 'conversion_rate'")
    scope: str = Field(description="e.g. 'West', 'All-India'")
    grain: Grain
    period: str = Field(description="e.g. '2025-11' or '2025-11-12'")
    opened_at: datetime
    status: AdjudicationStatus

    headline_movement: Evidence
    attributed: tuple[Evidence, ...] = Field(
        default=(), description="Calendar / mix components. WHERE, not WHY (rule 7)."
    )
    qualified_residual: Evidence
    materiality: Evidence = Field(description="The KPI's materiality threshold, from its contract")
    empirical_band: Evidence | None = Field(
        default=None, description="Residual quantile band for this scope"
    )

    gates: tuple[GateResult, ...] = Field(default=())
    hypotheses: tuple[Hypothesis, ...] = Field(default=())

    coverage: float = Field(
        ge=0.0, le=1.0, description="Share of qualified residual attributed; feeds the verdict table"
    )
    confidence: ConfidenceBreakdown
    triggers_fired: tuple[TriggerId, ...] = Field(default=())

    evidence: tuple[Evidence, ...] = Field(
        default=(), description="Full evidence index for this case; ids referenced throughout"
    )
    elapsed_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Real wall-clock time to reach adjudication (resolved defect 6 — measured, never asserted)",
    )


class Verdict(_Contract):
    """One of three outcomes, plus the reason string rendered on the chip."""

    case_id: str
    value: VerdictValue
    reason_code: str | None = Field(
        default=None, description="e.g. 'live_unverifiable_above_materiality', 'coverage_below_0_70'"
    )
    reason_text: str | None = Field(default=None, description="Rendered on the verdict chip")
    triggers_fired: tuple[TriggerId, ...] = Field(
        default=(), description="Every trigger that fired, named — not just the first"
    )
    coverage: float = Field(ge=0.0, le=1.0)
    confidence_calibrated: float = Field(ge=0.0, le=1.0)
    decided_at: datetime


class Recommendation(_Contract):
    """Action, cost, expected recovery and what is explicitly not recommended."""

    case_id: str
    action: str = Field(description="Phrased by the model from playbook fields; never invented")
    playbook_ref: str = Field(description="Key into semantic_layer/playbooks/*.yaml")

    cost: Evidence
    expected_recovery_low: Evidence
    expected_recovery_high: Evidence
    roi_low: float | None = Field(default=None, ge=0.0)
    roi_high: float | None = Field(default=None, ge=0.0)

    recovery_confidence: RecoveryConfidence
    sample_size: int = Field(ge=0, description="Prior comparable interventions backing the estimate")

    effort: str | None = Field(default=None, description="e.g. '9 managers, 2 hours'")
    not_recommended: tuple[str, ...] = Field(
        default=(),
        description="Actions deliberately advised against, with the reason (see case #2467)",
    )
    linked_case_id: str | None = Field(
        default=None, description="The cause of the cause, via the playbook chain"
    )
    evidence_ids: tuple[str, ...] = Field(default=())


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TelemetryEvent(_Contract):
    """One recorded step. Backs the measured elapsed time and cost per case.

    Performance targets in CLAUDE.md §"Definition of done": P95 latency < 9 s
    warm, cost per case < INR 6.
    """

    event_id: str
    event: str = Field(description="e.g. 'stage_start', 'stage_end', 'llm_call', 'sql_query'")
    stage: Stage | None = Field(default=None)
    case_id: str | None = Field(default=None)

    started_at: datetime
    duration_ms: float | None = Field(default=None, ge=0.0)

    model: str | None = Field(default=None)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    cost_inr: float | None = Field(default=None, ge=0.0)

    mock: bool = Field(default=False, description="True when served by MockProvider")
    detail: str | None = Field(default=None)


__all__ = [
    "Adjudication",
    "AdjudicationStatus",
    "CapApplied",
    "ConfidenceBreakdown",
    "ConfidenceComponent",
    "Evidence",
    "EvidenceKind",
    "GateResult",
    "Grain",
    "Hypothesis",
    "HypothesisStatus",
    "LineageStep",
    "ProducedBy",
    "RecoveryConfidence",
    "Recommendation",
    "Stage",
    "TelemetryEvent",
    "TestResult",
    "TestType",
    "TriggerId",
    "Verdict",
    "VerdictValue",
]
