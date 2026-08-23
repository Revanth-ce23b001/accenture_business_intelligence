"""Typed schema and loader for the semantic layer.

The YAML files in this package are the only place a threshold, materiality
limit, access rule or prior may live (CLAUDE.md rule 2). This module gives
them a schema, validates them individually and against each other, and
hands the engine a single frozen object.

Two kinds of validation happen here:

  per-file   Pydantic models. A missing mandatory field, an out-of-range
             number or an unknown key is rejected at parse time.
  cross-file `SemanticLayer` checks that every reference resolves — a
             playbook's driver must be a real hypothesis, a hypothesis's
             affects must be real KPIs, a recovery_curve_ref must exist.
             A dangling reference is the failure mode YAML makes easiest
             and is worth catching at load rather than at adjudication.

Usage:

    from semantic_layer.schema import load_semantic_layer
    layer = load_semantic_layer()
    layer.kpis["net_revenue"].thresholds.materiality.value
    layer.causal_graph.hypotheses["competitor_action"].missing_sources()
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PACKAGE_ROOT = Path(__file__).parent

#: PyYAML's pure-Python parser accounts for almost all of the load time
#: (~84 ms of ~89 ms across these files). libyaml does the same parse in
#: ~9 ms. Use it when the wheel was built with it, fall back when not.
_YAML_LOADER: type = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
USING_LIBYAML = _YAML_LOADER is not yaml.SafeLoader

Grain = Literal["daily", "weekly", "monthly"]
ReliabilityTier = Literal[
    "structured_query",
    "derived_estimate",
    "corroborated_unstructured",
    "ticket_aggregate",
    "store_note",
    "news_item",
    "social_mention",
]
TestType = Literal["hard_gate", "weighted", "cap"]


class SemanticLayerError(RuntimeError):
    """A semantic-layer file is missing, unparseable or inconsistent."""


class _Node(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# KPI contracts
# ---------------------------------------------------------------------------


class Materiality(_Node):
    value: float = Field(gt=0.0, description="Below this, a movement is not worth a case")
    unit: str = Field(description="INR_CR, pt, pct, count")
    display: str = Field(description="How it renders in the UI, e.g. '₹50 L'")
    note: str | None = None


class KpiThresholds(_Node):
    #: None is legitimate — a KPI with no agreed materiality limit cannot
    #: open a case. See qcomm_fulfilment_rate.
    materiality: Materiality | None = None
    statistical_method: str
    band_lookback_weeks: int | None = Field(default=None, gt=0)
    band_quantile: float | None = Field(default=None, gt=0.0, lt=1.0)
    persistence_periods: int | None = Field(default=None, gt=0)
    min_relative_movement_pct: float | None = Field(default=None, ge=0.0)


class Baseline(_Node):
    #: None is legitimate — a KPI without enough history to fit a seasonal
    #: profile declares no method rather than a bad one.
    method: str | None = None
    min_history_weeks: int = Field(gt=0)
    seasonal_period_days: int | None = Field(default=None, gt=0)
    calendar_regressors: list[str] = Field(default_factory=list)


class SourceSystem(_Node):
    name: str
    description: str
    tables: list[str] = Field(min_length=1)
    reliability_tier: ReliabilityTier


class Refresh(_Node):
    cadence: str
    time_ist: str
    sla_hours: float = Field(gt=0.0)
    restatement_window_days: int = Field(ge=0)


class LineageStage(_Node):
    stage: str
    description: str
    produces: str


class PersonaPolicy(_Node):
    row_predicate: str = Field(min_length=1, description="SQL predicate; 'TRUE' means unrestricted")
    masked_columns: list[str] = Field(default_factory=list)


class AccessPolicy(_Node):
    default_persona: str
    personas: dict[str, PersonaPolicy] = Field(min_length=1)

    @model_validator(mode="after")
    def _default_persona_exists(self) -> AccessPolicy:
        if self.default_persona not in self.personas:
            raise ValueError(
                f"default_persona {self.default_persona!r} is not defined in personas "
                f"({sorted(self.personas)})"
            )
        return self


class ConfidenceRules(_Node):
    required_sources: list[str] = Field(min_length=1)
    min_source_reliability: float = Field(ge=0.0, le=1.0)
    freshness_full_credit_hours: float = Field(gt=0.0)
    freshness_zero_credit_hours: float = Field(gt=0.0)
    regime_change_penalty: float = Field(ge=0.0, le=1.0)
    publication_floor: float = Field(ge=0.0, le=1.0)
    coverage_note: str | None = None

    @model_validator(mode="after")
    def _freshness_window_is_ordered(self) -> ConfidenceRules:
        if self.freshness_zero_credit_hours <= self.freshness_full_credit_hours:
            raise ValueError(
                "freshness_zero_credit_hours must exceed freshness_full_credit_hours"
            )
        return self


class GrainSpec(_Node):
    supported: list[Grain] = Field(min_length=1)
    default: Grain

    @model_validator(mode="after")
    def _default_is_supported(self) -> GrainSpec:
        if self.default not in self.supported:
            raise ValueError(f"default grain {self.default!r} is not in supported {self.supported}")
        return self


class KpiContract(_Node):
    kpi: str
    version: int = Field(ge=1)
    display_name: str
    unit: str
    owner_role: str
    lifecycle: Literal["active", "monitoring_only", "deprecated"] = "active"

    definition: str = Field(min_length=1)
    formula_sql: str = Field(min_length=1)

    grain: GrainSpec
    dimensions: list[str] = Field(min_length=1)
    source_systems: list[SourceSystem] = Field(min_length=1)
    refresh: Refresh

    thresholds: KpiThresholds
    baseline: Baseline
    history_weeks: int = Field(ge=0)

    drivers: list[str] = Field(default_factory=list)
    related_kpis: list[str] = Field(default_factory=list)
    lineage_stages: list[LineageStage] = Field(min_length=1)
    access_policy: AccessPolicy
    confidence_rules: ConfidenceRules

    def has_sufficient_history(self) -> bool:
        """False means this KPI stops at the history gate."""
        return self.history_weeks >= self.baseline.min_history_weeks

    def can_open_a_case(self) -> bool:
        """A KPI with no materiality limit and no baseline is monitoring only."""
        return (
            self.thresholds.materiality is not None
            and self.baseline.method is not None
            and self.has_sufficient_history()
        )


# ---------------------------------------------------------------------------
# Causal graph
# ---------------------------------------------------------------------------


class SourceSpec(_Node):
    held: bool
    reliability_tier: ReliabilityTier
    description: str


class HypothesisTemplate(_Node):
    label: str
    description: str
    affects: list[str] = Field(min_length=1)
    required_sources: list[str] = Field(min_length=1)
    available_sources: list[str] = Field(default_factory=list)
    applicable_tests: list[str] = Field(min_length=1)
    confounders: list[str] = Field(default_factory=list)
    prior: float = Field(gt=0.0, lt=1.0)
    calibration_note: str | None = None

    def missing_sources(self) -> list[str]:
        """Required minus available. Non-empty on the leading hypothesis fires T3."""
        available = set(self.available_sources)
        return [s for s in self.required_sources if s not in available]

    def is_structurally_unverifiable(self) -> bool:
        return bool(self.missing_sources())


class CausalGraph(_Node):
    version: int = Field(ge=1)
    sources: dict[str, SourceSpec] = Field(min_length=1)
    hypotheses: dict[str, HypothesisTemplate] = Field(min_length=1)

    @model_validator(mode="after")
    def _references_resolve(self) -> CausalGraph:
        known = set(self.sources)
        for name, hypothesis in self.hypotheses.items():
            for source in (*hypothesis.required_sources, *hypothesis.available_sources):
                if source not in known:
                    raise ValueError(
                        f"hypothesis {name!r} references unknown source {source!r}; "
                        f"add it to the `sources` registry"
                    )
            for source in hypothesis.available_sources:
                if not self.sources[source].held:
                    raise ValueError(
                        f"hypothesis {name!r} lists {source!r} as available, but the source "
                        "registry says the organisation does not hold it"
                    )
            for confounder in hypothesis.confounders:
                if confounder not in self.hypotheses:
                    raise ValueError(
                        f"hypothesis {name!r} names unknown confounder {confounder!r}"
                    )
                if confounder == name:
                    raise ValueError(f"hypothesis {name!r} lists itself as its own confounder")
        return self


# ---------------------------------------------------------------------------
# Playbooks
# ---------------------------------------------------------------------------


class CostModel(_Node):
    type: str
    unit: str
    scales_with: str
    unit_cost_inr: float | None = Field(default=None, ge=0.0)
    fixed_cost_inr: float = Field(default=0.0, ge=0.0)
    currency: str = "INR"
    depth_source: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _cost_is_computable(self) -> CostModel:
        if self.unit_cost_inr is None and self.fixed_cost_inr == 0.0 and self.depth_source is None:
            raise ValueError(
                "cost model yields nothing: set unit_cost_inr, fixed_cost_inr, or depth_source"
            )
        return self


class LeadTime(_Node):
    value: float = Field(ge=0.0)
    unit: Literal["hours", "days", "weeks", "months"]
    note: str | None = None


class MonitoringCheckpoint(_Node):
    at_week: int = Field(ge=0)
    expect: str = Field(min_length=1)


class MonitoringPlan(_Node):
    primary_kpi: str
    secondary_kpis: list[str] = Field(default_factory=list)
    cadence: Literal["daily", "weekly", "monthly"]
    horizon_weeks: int = Field(gt=0)
    checkpoints: list[MonitoringCheckpoint] = Field(min_length=1)
    kill_criterion: str = Field(min_length=1)
    escalation_role: str

    @model_validator(mode="after")
    def _checkpoints_fit_the_horizon(self) -> MonitoringPlan:
        for checkpoint in self.checkpoints:
            if checkpoint.at_week > self.horizon_weeks:
                raise ValueError(
                    f"checkpoint at week {checkpoint.at_week} falls outside the "
                    f"{self.horizon_weeks}-week horizon"
                )
        return self


class LinkedCaseTemplate(_Node):
    driver: str
    question: str = Field(min_length=1)
    scope_hint: str
    rationale: str


class Preconditions(_Node):
    rules: list[str] = Field(min_length=1)
    note: str | None = None


class Playbook(_Node):
    """Mandatory fields are mandatory. Omitting any one fails validation."""

    playbook: str
    version: int = Field(ge=1)

    driver: str
    lever: str
    action: str = Field(min_length=1)
    cost_model: CostModel
    lead_time: LeadTime
    owner_role: str
    recovery_curve_ref: str
    monitoring_plan: MonitoringPlan

    # Optional
    linked_case_template: LinkedCaseTemplate | None = None
    preconditions: Preconditions | None = None


#: The field names the P2 brief calls mandatory. Kept as data so the test
#: suite can drive each one out in turn.
MANDATORY_PLAYBOOK_FIELDS = (
    "driver",
    "lever",
    "action",
    "cost_model",
    "lead_time",
    "owner_role",
    "recovery_curve_ref",
    "monitoring_plan",
)


# ---------------------------------------------------------------------------
# Recovery curves
# ---------------------------------------------------------------------------


class RecoveryCurve(_Node):
    description: str
    p25: float = Field(ge=0.0, le=1.0)
    p75: float = Field(ge=0.0, le=1.0)
    horizon_weeks: int = Field(ge=0)
    sample_size: int = Field(ge=0)
    confidence_label: Literal["Low", "Medium", "High"]
    basis: str

    @model_validator(mode="after")
    def _quartiles_are_ordered(self) -> RecoveryCurve:
        if self.p25 > self.p75:
            raise ValueError(f"p25 ({self.p25}) exceeds p75 ({self.p75})")
        return self


class RecoveryCurves(_Node):
    version: int = Field(ge=1)
    curves: dict[str, RecoveryCurve] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Adjudication constants
# ---------------------------------------------------------------------------


class ConfidenceComponentSpec(_Node):
    name: str
    weight: float = Field(ge=0.0, le=1.0)
    method: str
    penalise_when: str | None = None
    reference_weeks: int | None = Field(default=None, gt=0)
    regime_change_penalty: float | None = Field(default=None, ge=0.0, le=1.0)


class CapSpec(_Node):
    condition: str
    ceiling: float = Field(ge=0.0, le=1.0)
    forced_trigger: str | None = None
    eliminates_hypothesis: bool = False


class CalibrationSpec(_Node):
    method: str
    publication_floor: float = Field(ge=0.0, le=1.0)
    max_expected_calibration_error: float = Field(ge=0.0, le=1.0)
    min_cases_for_band: int = Field(ge=1)


class ConfidenceSpec(_Node):
    components: dict[str, ConfidenceComponentSpec] = Field(min_length=1)
    caps: dict[str, CapSpec] = Field(min_length=1)
    calibration: CalibrationSpec

    @model_validator(mode="after")
    def _weights_form_a_weighted_mean(self) -> ConfidenceSpec:
        total = sum(c.weight for c in self.components.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"confidence component weights sum to {total!r}, not 1.0 — a weighted mean "
                "whose weights do not sum to one is not a mean"
            )
        return self


class ExplainedRule(_Node):
    min_coverage: float = Field(ge=0.0, le=1.0)
    min_confidence: float = Field(ge=0.0, le=1.0)
    forbid_live_unverifiable_above_materiality: bool


class PartiallyExplainedRule(_Node):
    min_coverage: float = Field(ge=0.0, le=1.0)
    min_confidence: float = Field(ge=0.0, le=1.0)


class VerdictSpec(_Node):
    explained: ExplainedRule
    partially_explained: PartiallyExplainedRule
    reason_codes: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def _explained_is_the_stricter_branch(self) -> VerdictSpec:
        if self.explained.min_coverage < self.partially_explained.min_coverage:
            raise ValueError("EXPLAINED requires less coverage than PARTIALLY EXPLAINED")
        if self.explained.min_confidence < self.partially_explained.min_confidence:
            raise ValueError("EXPLAINED requires less confidence than PARTIALLY EXPLAINED")
        return self


class TriggerSpec(_Node):
    name: str
    evaluated_against: str
    min_residual_share: float | None = Field(default=None, ge=0.0, le=1.0)
    max_confidence_separation: float | None = Field(default=None, ge=0.0, le=1.0)
    min_independent_sources: int | None = Field(default=None, ge=1)
    reliability_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    max_low_reliability_share: float | None = Field(default=None, ge=0.0, le=1.0)
    publication_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    min_closed_cases: int | None = Field(default=None, ge=1)
    max_staleness_hours: float | None = Field(default=None, gt=0.0)


class DidPretest(_Node):
    name: str
    lookback_weeks: int = Field(gt=0)
    min_p_value: float = Field(ge=0.0, le=1.0)


class TestSpec(_Node):
    test_id: int = Field(ge=1, le=6)
    name: str
    type: TestType
    method: str
    weight: float | None = Field(default=None, ge=0.0, le=1.0)
    requirement: str | None = None
    min_residual_share: float | None = Field(default=None, ge=0.0, le=1.0)
    min_r: float | None = Field(default=None, ge=0.0, le=1.0)
    max_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    matching_covariates: list[str] = Field(default_factory=list)
    pretest: DidPretest | None = None

    @model_validator(mode="after")
    def _weight_matches_type(self) -> TestSpec:
        if self.type == "weighted" and self.weight is None:
            raise ValueError(f"test {self.name!r} is weighted but carries no weight")
        if self.type != "weighted" and self.weight is not None:
            raise ValueError(
                f"test {self.name!r} is {self.type} but carries weight {self.weight!r}; "
                "hard gates and caps are not voted on"
            )
        return self


class Corroboration(_Node):
    min_independent_notes: int = Field(ge=1)
    promotes_from: ReliabilityTier
    promotes_to: ReliabilityTier


class ReliabilitySpec(_Node):
    floor: float = Field(ge=0.0, le=1.0)
    weights: dict[ReliabilityTier, float] = Field(min_length=1)
    corroboration: Corroboration

    @model_validator(mode="after")
    def _every_tier_has_a_weight(self) -> ReliabilitySpec:
        expected = set(ReliabilityTier.__args__)  # type: ignore[attr-defined]
        missing = expected - set(self.weights)
        if missing:
            raise ValueError(f"reliability weights missing for tiers: {sorted(missing)}")
        for tier, weight in self.weights.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"reliability weight for {tier!r} is out of range: {weight}")
        return self


class GateSpec(_Node):
    name: str
    outcome_code: str
    proposed: bool = False
    max_missing_feed_share: float | None = Field(default=None, ge=0.0, le=1.0)


class AdjudicationConfig(_Node):
    version: int = Field(ge=1)
    confidence: ConfidenceSpec
    verdict: VerdictSpec
    triggers: dict[str, TriggerSpec] = Field(min_length=1)
    tests: dict[str, TestSpec] = Field(min_length=1)
    reliability: ReliabilitySpec
    gates: dict[int, GateSpec] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Warehouse policy
# ---------------------------------------------------------------------------


class Units(_Node):
    """Unit conversions. Rule 2 admits no numbers in engine/ at all."""

    inr_per_crore: float = Field(gt=0.0)
    inr_per_lakh: float = Field(gt=0.0)
    percent_scale: float = Field(gt=0.0)
    crore_places: int = Field(ge=0)
    pct_places: int = Field(ge=0)
    days_per_week: int = Field(gt=0)
    days_per_year: float = Field(gt=0.0)


class Governance(_Node):
    """How `engine/db.py::execute_governed` binds a persona to a query."""

    connection_module: str
    audit_table: str
    reserved_binding_prefix: str = Field(min_length=1)
    reserved_bindings: dict[str, str] = Field(min_length=1)
    metadata_tables: list[str] = Field(min_length=1)
    unrestricted_predicate: str = Field(min_length=1)
    visibility_column: str = Field(min_length=1)

    @model_validator(mode="after")
    def _metadata_tables_hold_no_business_data(self) -> Governance:
        """The allow-list must not become a way to read the facts.

        `execute_metadata` applies no row predicate, so anything on this
        list is readable by every persona. A fact, document or external
        table on it would be a silent bypass of the whole access policy.
        """
        forbidden = tuple(
            name
            for name in self.metadata_tables
            if name.startswith(("fact_", "doc_", "ext_"))
        )
        if forbidden:
            raise ValueError(
                f"metadata_tables contains business tables {list(forbidden)}; "
                "execute_metadata applies no row predicate, so these would be readable "
                "by every persona"
            )
        return self

    @model_validator(mode="after")
    def _reserved_bindings_carry_the_prefix(self) -> Governance:
        """A binding the engine fills must be recognisable as one.

        `execute_governed` refuses any caller-supplied parameter starting
        with the prefix. If a reserved name did not carry it, a caller
        could supply that name and overwrite the row filter.
        """
        for name in self.reserved_bindings:
            if not name.startswith(self.reserved_binding_prefix):
                raise ValueError(
                    f"reserved binding {name!r} does not start with "
                    f"{self.reserved_binding_prefix!r}; a caller could then supply it"
                )
        return self


class RevenueDefinition(_Node):
    """One team's answer to 'what was revenue in period P'."""

    label: str
    owner: str
    expression: str = Field(min_length=1, description="SQL aggregate over the fact table")
    rationale: str = Field(min_length=1)


class DefinitionGap(_Node):
    numerator: str
    denominator: str
    unit: str


#: How loudly a gap is reported. Declared per problem in warehouse.yaml
#: rather than derived from a size, because "how bad is this" is a
#: business judgement and not a quantity the warehouse can measure.
Severity = Literal["ASSUMPTION", "WARNING", "ERROR"]


class SharedExclusion(_Node):
    column: str
    note: str


class DefinitionConflict(_Node):
    kpi: str
    fact_table: str
    arbiter: str
    definitions: dict[str, RevenueDefinition] = Field(min_length=2)
    gap: DefinitionGap
    gap_code: str
    severity: Severity
    shared_exclusion: SharedExclusion

    @model_validator(mode="after")
    def _references_resolve(self) -> DefinitionConflict:
        for role, name in (
            ("arbiter", self.arbiter),
            ("gap.numerator", self.gap.numerator),
            ("gap.denominator", self.gap.denominator),
        ):
            if name not in self.definitions:
                raise ValueError(
                    f"{role} names {name!r}, which is not one of the definitions "
                    f"({sorted(self.definitions)})"
                )
        if self.gap.numerator == self.gap.denominator:
            raise ValueError("the definition gap compares a definition with itself")
        return self


class EntityKeyMismatch(_Node):
    pos_key: str
    ops_key: str
    bridge_table: str
    fact_table: str
    measure: str
    policy: Literal["quarantine"]
    gap_code: str
    severity: Severity


class GrainMismatch(_Node):
    source_table: str
    source_grain: str
    target_grain: str
    method: str
    assumption_tag: str = Field(min_length=1)
    days_per_week: int = Field(gt=0)
    gap_code: str
    severity: Severity
    note: str


class CalendarPeriod(_Node):
    label: str
    column: str


class CalendarMismatch(_Node):
    calendar_table: str
    date_column: str
    festival_column: str
    festival_flag_column: str
    gap_code: str
    severity: Severity
    periods: dict[str, CalendarPeriod] = Field(min_length=2)
    compare: list[str] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def _compared_periods_exist(self) -> CalendarMismatch:
        for name in self.compare:
            if name not in self.periods:
                raise ValueError(
                    f"calendar_mismatch compares {name!r}, which is not a declared period "
                    f"({sorted(self.periods)})"
                )
        if self.compare[0] == self.compare[1]:
            raise ValueError("calendar_mismatch compares a calendar with itself")
        return self


class Reconciliation(_Node):
    """The four disagreements the warehouse exposes rather than repairs."""

    version: int = Field(ge=1)
    definition_conflict: DefinitionConflict
    entity_key_mismatch: EntityKeyMismatch
    grain_mismatch: GrainMismatch
    calendar_mismatch: CalendarMismatch

    def gap_codes(self) -> list[str]:
        """Every code this warehouse can raise. One per problem."""
        return sorted(
            {
                self.definition_conflict.gap_code,
                self.entity_key_mismatch.gap_code,
                self.grain_mismatch.gap_code,
                self.calendar_mismatch.gap_code,
            }
        )


class SourceLocation(_Node):
    """Where a KPI contract's named source system lands in the warehouse.

    `date_column` is None for a static master. A master has no arrival
    time, so it cannot be stale — Gate 1 reports it as static rather than
    inventing a freshness of zero and calling that fresh.
    """

    table: str = Field(min_length=1)
    date_column: str | None = None
    grain: Literal["daily", "weekly", "static"]
    scope_join: Literal["region", "store_id", "none"]
    #: Set when the table names its own scope instead of carrying a region,
    #: as `fact_qcomm_weekly` does with All-India.
    scope_column: str | None = None
    description: str = Field(min_length=1)

    @property
    def is_static(self) -> bool:
        return self.date_column is None


class EvidenceVocabulary(_Node):
    """What an `Evidence` record is allowed to say about itself.

    Closed sets, so a typo fails at load rather than becoming an
    unqueryable string in the `evidence` table.
    """

    non_source_origins: dict[str, str] = Field(min_length=1)
    methods: dict[str, str] = Field(min_length=1)


class WarehouseConfig(_Node):
    version: int = Field(ge=1)
    units: Units
    governance: Governance
    sources: dict[str, SourceLocation] = Field(min_length=1)
    evidence: EvidenceVocabulary
    reconciliation: Reconciliation

    def known_origins(self) -> frozenset[str]:
        """Every value `Evidence.source_system` may take."""
        return frozenset(self.sources) | frozenset(self.evidence.non_source_origins)

    @model_validator(mode="after")
    def _origins_do_not_collide_with_sources(self) -> WarehouseConfig:
        clash = set(self.sources) & set(self.evidence.non_source_origins)
        if clash:
            raise ValueError(
                f"{sorted(clash)} is both a source system and a non-source origin; "
                "an evidence record could not say which it meant"
            )
        return self


# ---------------------------------------------------------------------------
# VALIDATE — Gate 1
# ---------------------------------------------------------------------------

#: The named exits Gate 1 can take. Not a boolean: "validation failed"
#: tells an analyst nothing, and "25 store feeds did not load" tells them
#: everything. Closed set — a new exit is a change to the UI and to
#: `case_registry`, so it should not be possible to add one by typo.
ExitCode = Literal[
    "DATA_INCIDENT",
    "DEFINITION_CHANGE",
    "ONE_OFF",
    "PENDING_RESTATEMENT",
    "INSUFFICIENT_BASELINE",
]

#: Which of a KPI contract's sources the freshness check covers.
FreshnessScope = Literal["required_sources", "all_sources"]


class _CheckSpec(_Node):
    """Common to all five checks."""

    order: int = Field(ge=1, description="Evaluation and reporting priority; 1 is first")
    name: str = Field(min_length=1)
    exit_code: ExitCode


class SourceFreshnessCheck(_CheckSpec):
    scope: FreshnessScope
    sla_from: str = Field(description="Documents which contract field supplies the limit")
    sla_multiplier: float = Field(gt=0.0)
    static_sources_pass: bool


class RowCountDeltaCheck(_CheckSpec):
    source_table: str
    measure: str
    loaded_status: str = Field(min_length=1)
    lookback_weeks: int = Field(gt=0)
    statistic: Literal["median", "mean"]
    max_store_shortfall: float = Field(gt=0.0, le=1.0)
    max_affected_store_share: float = Field(ge=0.0, le=1.0)
    min_lookback_days: int = Field(gt=0)


class DefinitionDriftCheck(_CheckSpec):
    algorithm: Literal["sha256"]
    hashed_fields: list[str] = Field(min_length=1)
    log_table: str
    display_hash_chars: int = Field(gt=0)
    first_run_passes: bool


class SingleTransactionCheck(_CheckSpec):
    source_table: str
    transaction_key: str
    amount_expression: str = Field(min_length=1)
    max_top_transaction_share: float = Field(gt=0.0, le=1.0)
    min_movement_inr: float = Field(ge=0.0)


class RestatementCheck(_CheckSpec):
    register_table: str
    open_status: str
    window_from: str = Field(description="Documents which contract field supplies the window")
    scope_wildcard: str


class ValidateChecks(_Node):
    """All five, named. Gate 1 runs exactly these."""

    source_freshness: SourceFreshnessCheck
    row_count_delta: RowCountDeltaCheck
    definition_drift: DefinitionDriftCheck
    single_transaction_dominance: SingleTransactionCheck
    restatement_pending: RestatementCheck

    def in_order(self) -> list[_CheckSpec]:
        """The five specs, by declared priority."""
        specs = [getattr(self, name) for name in type(self).model_fields]
        return sorted(specs, key=lambda spec: spec.order)


class ValidateConfig(_Node):
    version: int = Field(ge=1)
    gate_id: int = Field(ge=1, le=5)
    name: str
    outcome_code: ExitCode
    checks: ValidateChecks

    @model_validator(mode="after")
    def _orders_are_distinct(self) -> ValidateConfig:
        orders = [spec.order for spec in self.checks.in_order()]
        if len(set(orders)) != len(orders):
            raise ValueError(
                f"two Gate 1 checks share an evaluation order ({orders}); the exit reported "
                "first would then depend on dict iteration"
            )
        return self


# ---------------------------------------------------------------------------
# QUALIFY — Gates 2 to 5, and restraint
# ---------------------------------------------------------------------------


class CalendarRegressors(_Node):
    """What the calendar baseline is allowed to know about.

    Marketing SPEND is deliberately absent and must stay absent: a spend
    cut absorbed into "calendar-expected" could never be eliminated on
    precedence, which is exactly what case #2451 turns on.
    """

    day_of_week: bool
    trading_days: Literal["by_construction"]
    trend: bool
    annual_harmonics: int = Field(ge=0)
    pay_cycle: bool
    month_end: bool
    promo_window: bool
    festival_window_segments: int = Field(gt=0)
    festival_hangover_days: int = Field(ge=0)
    festival_hangover_segments: int = Field(gt=0)


class AnomalyExclusion(_Node):
    method: Literal["monthly_mean_residual"]
    sigma: float = Field(gt=0.0)


class CalendarModelSpec(_Node):
    estimator: Literal["ols"]
    target_transform: Literal["log", "identity"]
    requested_history_years: int = Field(gt=0)
    min_fit_days: int = Field(gt=0)
    exclude_period_under_test: bool
    regressors: CalendarRegressors
    anomaly_exclusion: AnomalyExclusion


class DecompositionSpec(_Node):
    denominator: Literal["actual_previous_period", "predicted_previous_period"]
    emit_previous_period_fit: bool
    #: headline = calendar + residual holds by construction. This is the
    #: arithmetic tolerance it is checked to, not a business one.
    reconciliation_tolerance_pt: float = Field(gt=0.0)


class CalendarGate(_Node):
    gate_id: int = Field(ge=1, le=5)
    name: str
    outcome_code: str
    model: CalendarModelSpec
    decomposition: DecompositionSpec


class HistorySpec(_Node):
    source: Literal["observed_periods"]
    min_from: str


class StlSpec(_Node):
    period: int = Field(gt=1)
    robust: bool
    input: Literal["calendar_residual", "raw"]


class BandGate(_Node):
    gate_id: int = Field(ge=1, le=5)
    name: str
    history_outcome_code: str
    within_band_outcome_code: str
    history: HistorySpec
    method: Literal["stl_residual_empirical_quantile"]
    stl: StlSpec
    min_band_observations: int = Field(gt=0)


class SpecificityGate(_Node):
    gate_id: int = Field(ge=1, le=5)
    name: str
    outcome_code: str
    peer_dimension: str
    min_peers_breaching: int = Field(gt=0)
    same_direction_required: bool
    escalate_to_role: str
    unknown_outcome_code: str


class MaterialityGate(_Node):
    gate_id: int = Field(ge=1, le=5)
    name: str
    outcome_code: str
    threshold_from: str
    sub_threshold_route: str


class DeduplicationSpec(_Node):
    enabled: bool
    link_via: list[str] = Field(min_length=1)
    require_same_scope: bool
    require_same_period: bool
    outcome_code: str
    keep: Literal["largest_relative_to_materiality"]


class SuppressionSpec(_Node):
    enabled: bool
    window_days: int = Field(gt=0)
    outcome_code: str
    escalation_multiple: float = Field(gt=1.0)


class OwnerLoadSpec(_Node):
    enabled: bool
    max_open_cases_per_owner_per_week: int = Field(gt=0)
    owner_from: str
    window: Literal["iso_week"]
    outcome_code: str
    over_cap_route: str


class RestraintSpec(_Node):
    deduplication: DeduplicationSpec
    suppression: SuppressionSpec
    owner_load: OwnerLoadSpec


class QualifyConfig(_Node):
    version: int = Field(ge=1)
    calendar: CalendarGate
    band: BandGate
    specificity: SpecificityGate
    materiality: MaterialityGate
    restraint: RestraintSpec

    def gate_ids(self) -> list[int]:
        return sorted(
            {
                self.calendar.gate_id,
                self.band.gate_id,
                self.specificity.gate_id,
                self.materiality.gate_id,
            }
        )


# ---------------------------------------------------------------------------
# The whole layer
# ---------------------------------------------------------------------------


class SemanticLayer(_Node):
    kpis: dict[str, KpiContract]
    causal_graph: CausalGraph
    playbooks: dict[str, Playbook]
    recovery_curves: RecoveryCurves
    adjudication: AdjudicationConfig
    warehouse: WarehouseConfig
    #: `validate.yaml`, named `validation` here because a field called
    #: `validate` shadows a BaseModel method and pydantic warns about it.
    validation: ValidateConfig
    qualify: QualifyConfig

    @model_validator(mode="after")
    def _cross_references_resolve(self) -> SemanticLayer:
        kpi_names = set(self.kpis)
        hypothesis_names = set(self.causal_graph.hypotheses)
        test_names = set(self.adjudication.tests)
        curve_names = set(self.recovery_curves.curves)
        tier_weights = self.adjudication.reliability.weights

        for name, kpi in self.kpis.items():
            if kpi.kpi != name:
                raise ValueError(f"file for {name!r} declares kpi: {kpi.kpi!r}")
            for referenced in (*kpi.drivers, *kpi.related_kpis):
                if referenced not in kpi_names:
                    raise ValueError(f"kpi {name!r} references unknown kpi {referenced!r}")
            if name in kpi.related_kpis or name in kpi.drivers:
                raise ValueError(f"kpi {name!r} references itself")
            declared = {s.name for s in kpi.source_systems}
            for source in kpi.confidence_rules.required_sources:
                if source not in declared:
                    raise ValueError(
                        f"kpi {name!r} requires source {source!r} which is not in its "
                        f"source_systems ({sorted(declared)})"
                    )
            for system in kpi.source_systems:
                if system.reliability_tier not in tier_weights:
                    raise ValueError(
                        f"kpi {name!r} source {system.name!r} uses tier "
                        f"{system.reliability_tier!r} with no reliability weight"
                    )

        for name, hypothesis in self.causal_graph.hypotheses.items():
            for kpi in hypothesis.affects:
                if kpi not in kpi_names:
                    raise ValueError(f"hypothesis {name!r} affects unknown kpi {kpi!r}")
            for test in hypothesis.applicable_tests:
                if test not in test_names:
                    raise ValueError(f"hypothesis {name!r} names unknown test {test!r}")

        for name, playbook in self.playbooks.items():
            if playbook.playbook != name:
                raise ValueError(f"file for {name!r} declares playbook: {playbook.playbook!r}")
            if playbook.driver not in hypothesis_names:
                raise ValueError(
                    f"playbook {name!r} has driver {playbook.driver!r} which is not a "
                    f"hypothesis in the causal graph"
                )
            if playbook.recovery_curve_ref not in curve_names:
                raise ValueError(
                    f"playbook {name!r} references unknown recovery curve "
                    f"{playbook.recovery_curve_ref!r}"
                )
            plan = playbook.monitoring_plan
            for kpi in (plan.primary_kpi, *plan.secondary_kpis):
                if kpi not in kpi_names:
                    raise ValueError(f"playbook {name!r} monitors unknown kpi {kpi!r}")
            if playbook.linked_case_template is not None:
                linked = playbook.linked_case_template.driver
                if linked not in hypothesis_names:
                    raise ValueError(
                        f"playbook {name!r} links to unknown driver {linked!r}"
                    )

        for key, cap in self.adjudication.confidence.caps.items():
            if cap.forced_trigger and cap.forced_trigger not in self.adjudication.triggers:
                raise ValueError(
                    f"confidence cap {key!r} forces unknown trigger {cap.forced_trigger!r}"
                )

        registry = self.warehouse.sources
        for name, kpi in self.kpis.items():
            for system in kpi.source_systems:
                if system.name not in registry:
                    raise ValueError(
                        f"kpi {name!r} depends on source {system.name!r}, which has no entry "
                        f"in warehouse.yaml -> sources ({sorted(registry)}). Gate 1's "
                        "freshness check would have nowhere to look."
                    )

        gate_id = self.validation.gate_id
        if gate_id not in self.adjudication.gates:
            raise ValueError(
                f"validate.yaml is gate {gate_id}, which adjudication.yaml does not declare"
            )
        declared = self.adjudication.gates[gate_id]
        if declared.outcome_code != self.validation.outcome_code:
            raise ValueError(
                f"gate {gate_id} is {declared.outcome_code!r} in adjudication.yaml and "
                f"{self.validation.outcome_code!r} in validate.yaml"
            )

        # The gate map in adjudication.yaml is the single home for the
        # numbering; the stages carry the configuration. They must agree on
        # both the number and the headline outcome code, or a kill chip
        # would name a gate the map calls something else.
        staged = {
            self.qualify.calendar.gate_id: self.qualify.calendar.outcome_code,
            self.qualify.band.gate_id: self.qualify.band.history_outcome_code,
            self.qualify.specificity.gate_id: self.qualify.specificity.outcome_code,
            self.qualify.materiality.gate_id: self.qualify.materiality.outcome_code,
            self.validation.gate_id: self.validation.outcome_code,
        }
        if len(staged) != len(self.qualify.gate_ids()) + 1:
            raise ValueError(
                "two stages claim the same gate number across validate.yaml and "
                "qualify.yaml"
            )
        for gate_id, code in sorted(staged.items()):
            if gate_id not in self.adjudication.gates:
                raise ValueError(
                    f"a stage declares gate {gate_id}, which adjudication.yaml does not"
                )
            declared = self.adjudication.gates[gate_id].outcome_code
            if declared != code:
                raise ValueError(
                    f"gate {gate_id} is {declared!r} in adjudication.yaml and {code!r} "
                    "in the stage that runs it"
                )

        conflict = self.warehouse.reconciliation.definition_conflict
        if conflict.kpi not in kpi_names:
            raise ValueError(
                f"the definition conflict arbitrates {conflict.kpi!r}, which is not a KPI"
            )
        # The arbiter's expression is the contract's, so a change to one that
        # is not made to the other would let the warehouse publish a figure
        # the KPI contract does not recognise.
        for name, definition in conflict.definitions.items():
            if not definition.expression.strip():
                raise ValueError(f"revenue definition {name!r} has no expression")

        return self

    # -- convenience -------------------------------------------------------

    def materiality(self, kpi: str) -> Materiality | None:
        return self.kpis[kpi].thresholds.materiality

    def reliability_weight(self, tier: str) -> float:
        return self.adjudication.reliability.weights[tier]  # type: ignore[index]

    def playbooks_for(self, driver: str) -> list[Playbook]:
        return [p for p in self.playbooks.values() if p.driver == driver]

    def unverifiable_hypotheses(self) -> list[str]:
        return sorted(
            name
            for name, h in self.causal_graph.hypotheses.items()
            if h.is_structurally_unverifiable()
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SemanticLayerError(f"semantic layer file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise SemanticLayerError(f"{path.name} is not UTF-8: {exc}") from exc
    try:
        return yaml.load(text, Loader=_YAML_LOADER)
    except yaml.YAMLError as exc:
        raise SemanticLayerError(f"{path.name} is not valid YAML: {exc}") from exc


def _build(model: type[BaseModel], payload: Any, path: Path) -> Any:
    if payload is None:
        raise SemanticLayerError(f"{path.name} is empty")
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise SemanticLayerError(f"{path.name} failed validation:\n{exc}") from exc


def load_semantic_layer(root: Path | str | None = None) -> SemanticLayer:
    """Load, validate and cross-check every semantic-layer file.

    Files whose name starts with '_' are ignored, so helper YAML can live
    alongside contracts without being mistaken for one.
    """
    base = Path(root) if root else PACKAGE_ROOT

    kpi_dir = base / "kpis"
    playbook_dir = base / "playbooks"
    for directory in (kpi_dir, playbook_dir):
        if not directory.is_dir():
            raise SemanticLayerError(f"missing semantic-layer directory: {directory}")

    kpis: dict[str, KpiContract] = {}
    for path in sorted(kpi_dir.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        contract = _build(KpiContract, _read_yaml(path), path)
        kpis[contract.kpi] = contract
    if not kpis:
        raise SemanticLayerError(f"no KPI contracts found in {kpi_dir}")

    playbooks: dict[str, Playbook] = {}
    for path in sorted(playbook_dir.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        playbook = _build(Playbook, _read_yaml(path), path)
        playbooks[playbook.playbook] = playbook
    if not playbooks:
        raise SemanticLayerError(f"no playbooks found in {playbook_dir}")

    graph_path = base / "causal_graph.yaml"
    curves_path = base / "recovery_curves.yaml"
    adjudication_path = base / "adjudication.yaml"
    warehouse_path = base / "warehouse.yaml"
    validate_path = base / "validate.yaml"
    qualify_path = base / "qualify.yaml"

    layer_payload = {
        "kpis": kpis,
        "causal_graph": _build(CausalGraph, _read_yaml(graph_path), graph_path),
        "playbooks": playbooks,
        "recovery_curves": _build(RecoveryCurves, _read_yaml(curves_path), curves_path),
        "adjudication": _build(
            AdjudicationConfig, _read_yaml(adjudication_path), adjudication_path
        ),
        "warehouse": _build(WarehouseConfig, _read_yaml(warehouse_path), warehouse_path),
        "validation": _build(ValidateConfig, _read_yaml(validate_path), validate_path),
        "qualify": _build(QualifyConfig, _read_yaml(qualify_path), qualify_path),
    }
    try:
        return SemanticLayer.model_validate(layer_payload)
    except ValidationError as exc:
        raise SemanticLayerError(f"semantic layer is internally inconsistent:\n{exc}") from exc


@lru_cache(maxsize=4)
def get_semantic_layer(root: str | None = None) -> SemanticLayer:
    """Cached loader. The engine should use this rather than re-reading YAML."""
    return load_semantic_layer(root)


def load_timed(root: Path | str | None = None) -> tuple[SemanticLayer, float]:
    """Load and report elapsed milliseconds. Used by the performance test."""
    started = time.perf_counter()
    layer = load_semantic_layer(root)
    return layer, (time.perf_counter() - started) * 1000.0


__all__ = [
    "MANDATORY_PLAYBOOK_FIELDS",
    "AccessPolicy",
    "AdjudicationConfig",
    "Baseline",
    "CalendarMismatch",
    "CausalGraph",
    "CostModel",
    "DefinitionConflict",
    "EntityKeyMismatch",
    "EvidenceVocabulary",
    "Governance",
    "GrainMismatch",
    "HypothesisTemplate",
    "KpiContract",
    "Materiality",
    "BandGate",
    "CalendarGate",
    "MaterialityGate",
    "Playbook",
    "QualifyConfig",
    "Reconciliation",
    "RestraintSpec",
    "SpecificityGate",
    "Severity",
    "RecoveryCurve",
    "RecoveryCurves",
    "RevenueDefinition",
    "DefinitionDriftCheck",
    "ExitCode",
    "RestatementCheck",
    "RowCountDeltaCheck",
    "SemanticLayer",
    "SemanticLayerError",
    "SingleTransactionCheck",
    "SourceFreshnessCheck",
    "SourceLocation",
    "Units",
    "ValidateChecks",
    "ValidateConfig",
    "WarehouseConfig",
    "get_semantic_layer",
    "load_semantic_layer",
    "load_timed",
]
