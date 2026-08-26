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

import re
import time
from datetime import date
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


class DriverOwner(_Node):
    """Who answers for a movement this driver caused."""

    role: str = Field(min_length=1, description="Stable key; what the audit trail stores")
    title: str = Field(min_length=1, description="How the role renders in the UI")


class DriverOwnership(_Node):
    """Ownership by CAUSAL driver, which is not ownership by KPI.

    `KpiContract.drivers` decomposes the KPI — WHERE the movement sits.
    This maps the causal graph's drivers — WHY it moved — onto the person
    who can act. CLAUDE.md rule 7 keeps those two apart everywhere else;
    it has to keep them apart here too, or the recommendation is addressed
    to whoever owns the metric rather than whoever owns the cause.
    """

    default: DriverOwner
    by_driver: dict[str, DriverOwner] = Field(default_factory=dict)

    def owner(self, driver: str | None) -> DriverOwner:
        """The named owner for this driver, or the KPI's own owner."""
        if driver is None:
            return self.default
        return self.by_driver.get(driver, self.default)


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
    #: None means the KPI has not been asked the question yet, and every
    #: driver falls back to `owner_role`. Absent rather than guessed.
    driver_ownership: DriverOwnership | None = None
    related_kpis: list[str] = Field(default_factory=list)
    lineage_stages: list[LineageStage] = Field(min_length=1)
    access_policy: AccessPolicy
    confidence_rules: ConfidenceRules

    def has_sufficient_history(self) -> bool:
        """False means this KPI stops at the history gate."""
        return self.history_weeks >= self.baseline.min_history_weeks

    def owner_for(self, driver: str | None) -> tuple[str, str]:
        """(role, title) for an adjudicated driver. Never hardcoded in Python.

        Falls back through the contract's own default and then to
        `owner_role`, whose title is the role key when nobody has written
        one down — visibly ugly, which is the point. A missing title is a
        contract that has not been finished, not something to invent.
        """
        if self.driver_ownership is None:
            return self.owner_role, self.owner_role
        owner = self.driver_ownership.owner(driver)
        return owner.role, owner.title

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
    #: Key into `calibration_ledger.case_type`. What trigger T7 asks the
    #: organisation's track record about. Several hypotheses share one.
    case_type: str = Field(min_length=1)
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
    #: s1 — the logistic's zero and unit, in standard errors.
    midpoint_t: float | None = Field(default=None, ge=0.0)
    scale_t: float | None = Field(default=None, gt=0.0)
    pretest_failure_penalty: float | None = Field(default=None, ge=0.0, le=1.0)
    #: s2 — whether a test that could not run counts against the score.
    not_testable_counts: bool | None = None
    #: s3 — which agreement statistic, and whether it floors at zero.
    statistic: str | None = None
    floor_at_zero: bool | None = None
    #: s4 — the multiple of the SLA at which freshness scores zero.
    zero_at_sla_multiple: float | None = Field(default=None, gt=1.0)
    #: s5 — which sources the depth is measured over.
    scope: str | None = None


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
    min_cases_for_fit: int = Field(ge=1)
    band_edges: list[float] = Field(min_length=2)

    @model_validator(mode="after")
    def _band_edges_ascend_and_span_the_unit_interval(self) -> CalibrationSpec:
        edges = self.band_edges
        if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
            raise ValueError(f"band edges must ascend: {edges}")
        if edges[0] != 0.0 or edges[-1] != 1.0:
            raise ValueError(
                f"band edges must span [0, 1]; a confidence outside the table "
                f"has nowhere to be counted. Got {edges[0]} to {edges[-1]}"
            )
        return self

    @model_validator(mode="after")
    def _a_map_needs_more_cases_than_a_band(self) -> CalibrationSpec:
        if self.min_cases_for_fit < self.min_cases_for_band:
            raise ValueError(
                f"min_cases_for_fit ({self.min_cases_for_fit}) is below "
                f"min_cases_for_band ({self.min_cases_for_band}): the whole map "
                "would be fitted on fewer cases than one band of it needs"
            )
        return self


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
    seconds_per_hour: float = Field(gt=0.0)
    milliseconds_per_second: float = Field(gt=0.0)
    minutes_per_hour: float = Field(gt=0.0)
    hours_per_day: float = Field(gt=0.0)
    months_per_year: int = Field(gt=0)


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
# GATHER — the three lanes
# ---------------------------------------------------------------------------


class CausalGraphSource(_Node):
    enabled: bool


class CaseHistorySource(_Node):
    enabled: bool
    lookback_days: int = Field(gt=0)
    bonus_per_case: float = Field(ge=0.0)
    max_bonus: float = Field(ge=0.0)
    closed_only: bool


class LongTailSource(_Node):
    enabled: bool
    max_candidates: int = Field(ge=0)
    prior: float = Field(gt=0.0, lt=1.0)
    tag_prefix: str = Field(min_length=1)
    #: Token overlap above which a proposed candidate is treated as a
    #: restatement of a graph hypothesis rather than a new one.
    novelty_overlap_threshold: float = Field(gt=0.0, le=1.0)
    novelty_stopwords: list[str] = Field(default_factory=list)


class HypothesisSources(_Node):
    causal_graph: CausalGraphSource
    case_history: CaseHistorySource
    long_tail: LongTailSource


class Applicability(_Node):
    kpi_not_affected: float = Field(ge=0.0, le=1.0)
    grain_supported: float = Field(ge=0.0)
    base: float = Field(gt=0.0)


class HypothesisScreening(_Node):
    top_n: int = Field(gt=0)
    sources: HypothesisSources
    applicability: Applicability
    #: Hypothesis -> why an earlier stage already removed it.
    already_removed: dict[str, str] = Field(default_factory=dict)


class StructuredTemplate(_Node):
    """One parameterised query, and what to make of the rows it returns."""

    label: str
    source_system: str
    unit: str
    measure: str = Field(min_length=1)
    aggregate: Literal["mean", "sum", "count", "min", "max"]
    sql: str = Field(min_length=1)


class StructuredLane(_Node):
    measure_places: int = Field(ge=0)
    templates: dict[str, StructuredTemplate] = Field(min_length=1)
    #: Hypothesis -> why there is no template. Declared, because "we have
    #: no query" and "we forgot to write one" look identical when absent.
    unavailable: dict[str, str] = Field(default_factory=dict)


class Corpus(_Node):
    table: str
    id_column: str
    text_column: str
    date_column: str
    scope_join: Literal["store_id", "region", "none"]
    reliability_kind: ReliabilityTier
    source_system: str


class Bm25Spec(_Node):
    k1: float = Field(gt=0.0)
    b: float = Field(ge=0.0, le=1.0)
    idf_smoothing: float = Field(gt=0.0)


class EmbeddingSpec(_Node):
    method: Literal["tfidf_svd"]
    components: int = Field(gt=0)
    random_state: int
    min_df: int = Field(ge=1)


class RetrievalSpec(_Node):
    method: Literal["hybrid", "bm25", "embeddings"]
    bm25: Bm25Spec
    embeddings: EmbeddingSpec
    bm25_weight: float = Field(ge=0.0, le=1.0)
    embedding_weight: float = Field(ge=0.0, le=1.0)
    top_k_per_corpus: int = Field(gt=0)
    min_score_range: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _weights_form_a_blend(self) -> RetrievalSpec:
        total = self.bm25_weight + self.embedding_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"retrieval weights sum to {total!r}, not 1.0 — a blend whose weights "
                "do not sum to one is not a blend"
            )
        return self


class ClassificationSpec(_Node):
    task: str
    batch_size: int = Field(gt=0)
    cache_table: str
    prompt_version: int = Field(ge=1)
    min_confidence: float = Field(ge=0.0, le=1.0)
    null_tag: str = Field(min_length=1)


class StatisticsSpec(_Node):
    test: Literal["chi_square"]
    yates_correction: bool
    min_expected_cell: int = Field(ge=0)


class UnstructuredLane(_Node):
    corpora: dict[str, Corpus] = Field(min_length=1)
    retrieval: RetrievalSpec
    classification: ClassificationSpec
    statistics: StatisticsSpec


class ExternalFeed(_Node):
    table: str
    source_system: str
    description: str
    scope_join: Literal["store_id", "region", "none"] | None = None
    measure: str | None = None
    aggregate: Literal["mean", "sum", "count"] | None = None
    unit: str | None = None


class ExternalLane(_Node):
    feeds: dict[str, ExternalFeed] = Field(min_length=1)


class GatherExecution(_Node):
    parallel_lanes: bool
    max_workers: int = Field(gt=0)
    fail_open: bool


class GatherConfig(_Node):
    version: int = Field(ge=1)
    hypotheses: HypothesisScreening
    structured: StructuredLane
    unstructured: UnstructuredLane
    external: ExternalLane
    execution: GatherExecution


# ---------------------------------------------------------------------------
# ADJUDICATE — what the six tests read
# ---------------------------------------------------------------------------


class Series(_Node):
    """A cause or effect series one test reads."""

    label: str
    source_system: str
    direction: Literal["up", "down"] | None = None
    sql: str = Field(min_length=1)
    grain_note: str | None = None


class ChangepointSpec(_Node):
    method: Literal["pelt"]
    model: Literal["rbf", "l1", "l2", "normal"]
    penalty: float = Field(gt=0.0)
    min_size: int = Field(gt=0)
    jump: int = Field(gt=0)
    min_segments: int = Field(gt=0)


class PrecedenceSpec(_Node):
    test_id: int = Field(ge=1, le=6)
    changepoint: ChangepointSpec
    #: A cause seen on the same DAY as its effect has not failed
    #: precedence; it has been observed at daily resolution.
    tolerance_days: int = Field(ge=0)
    #: Whether a store-grain cause series is measured on the exposed
    #: stores or on the whole scope.
    measure_cause_on: Literal["exposed", "scope"]
    effect: Series
    causes: dict[str, Series] = Field(min_length=1)
    window_weeks_before: int = Field(gt=0)
    window_weeks_after: int = Field(ge=0)


class ElasticityFallback(_Node):
    """What Test 2 does when the history will not yield an elasticity.

    Not silence. A hypothesis nobody can bound is a hypothesis nobody can
    eliminate, and a hard gate that never fires is not a gate. The bound
    falls back to a declared figure, the evidence records that it is
    declared, and `elimination_margin` requires the conclusion to survive
    the declared figure being wrong by that factor.
    """

    method: Literal["declared_upper_bound"]
    elimination_margin: float = Field(gt=1.0)
    source: str = Field(min_length=1)


class ElasticitySpec(_Node):
    method: Literal["log_log_ols"]
    lookback_weeks: int = Field(gt=0)
    min_observations: int = Field(gt=0)
    min_r_squared: float = Field(ge=0.0, le=1.0)
    fallback: ElasticityFallback | None = None


class SufficiencySeries(_Node):
    label: str
    driver_unit: str
    sql: str = Field(min_length=1)
    declared_elasticity: float | None = Field(default=None, gt=0.0)
    declared_source: str | None = None

    @model_validator(mode="after")
    def _a_declared_elasticity_names_its_source(self) -> SufficiencySeries:
        if (self.declared_elasticity is None) != (self.declared_source is None):
            raise ValueError(
                f"{self.label!r}: a declared elasticity and the source it came "
                "from are declared together or not at all. A number with no "
                "provenance is the thing this project exists to refuse."
            )
        return self


class SufficiencySpec(_Node):
    test_id: int = Field(ge=1, le=6)
    elasticity: ElasticitySpec
    hypotheses: dict[str, SufficiencySeries] = Field(min_length=1)
    not_testable_passes: bool


class DoseResponseSpec(_Node):
    test_id: int = Field(ge=1, le=6)
    method: Literal["pearson_then_ols"]
    min_stores: int = Field(gt=0)
    requires_store_grain: bool


class SpecificitySpec(_Node):
    test_id: int = Field(ge=1, le=6)
    method: Literal["welch_t"]
    min_group_size: int = Field(gt=0)


class ExposureSpec(_Node):
    """Which stores a hypothesis's cause actually reached.

    Derived per hypothesis from that hypothesis's own cause series, and
    shared by Tests 4 and 5 so the two are answering the same question.
    Never read off a column: which stores were affected is a CONCLUSION.
    """

    method: Literal["robust_deviation"]
    sigma: float = Field(gt=0.0)
    mad_scale: float = Field(gt=0.0)
    min_group_size: int = Field(gt=0)


class Covariate(_Node):
    type: Literal["exact", "numeric"]
    weight: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _numeric_covariates_carry_a_weight(self) -> Covariate:
        if self.type == "numeric" and self.weight is None:
            raise ValueError("a numeric matching covariate needs a weight")
        return self


class MatchingSpec(_Node):
    method: Literal["nearest_neighbour_mahalanobis"]
    covariates: dict[str, Covariate] = Field(min_length=1)
    ratio: int = Field(gt=0)
    replacement: bool
    max_distance: float = Field(gt=0.0)
    forbidden_distance: float = Field(gt=0.0)


class ParallelTrendsSpec(_Node):
    name: str
    method: Literal["ols_slope_of_log_ratio"]
    lookback_weeks: int = Field(gt=0)
    min_p_value: float = Field(ge=0.0, le=1.0)
    min_points: int = Field(gt=0)
    interpretation: str = Field(min_length=1)


class AttributionSpec(_Node):
    """How much of a DiD point estimate a hypothesis may claim."""

    method: Literal["interval_lower_bound", "point_estimate"]
    confidence: float = Field(gt=0.0, lt=1.0)


class DidSpec(_Node):
    test_id: int = Field(ge=1, le=6)
    matching: MatchingSpec
    pretest: ParallelTrendsSpec
    attribution: AttributionSpec


class ConfounderMeasure(_Node):
    """How one confounder is screened across the treated and control groups.

    Three ways, and they are different claims:

      store    measured per store and balance-tested. The strongest.
      region   identical for every store in the region, so it cannot
               differ between two groups inside one. Resolved by
               construction, and the reason is written down.
      matched  removed by the matching design, naming the covariate that
               does it. The screen checks that covariate is balanced, so
               the claim is verified rather than asserted.
    """

    label: str
    source_system: str
    scope_level: Literal["store", "region", "matched"]
    sql: str | None = None
    note: str | None = None
    #: For `matched`: which matching covariate removes it.
    resolved_by: str | None = None

    @model_validator(mode="after")
    def _each_scope_level_carries_its_argument(self) -> ConfounderMeasure:
        if self.scope_level == "store" and not self.sql:
            raise ValueError(
                "a store-level confounder measure needs a query; without one it cannot "
                "be balanced between the two groups"
            )
        if self.scope_level == "region" and not self.note:
            raise ValueError(
                "a region-level confounder resolves by construction, and the reason has "
                "to be written down"
            )
        if self.scope_level == "matched" and not (self.resolved_by and self.note):
            raise ValueError(
                "a matched confounder must name the covariate that removes it and say "
                "why; 'the matching handles it' is not an argument"
            )
        return self


class ConfounderScreenSpec(_Node):
    test_id: int = Field(ge=1, le=6)
    method: Literal["balance_test"]
    min_group_size: int = Field(gt=0)
    min_balance_p: float = Field(ge=0.0, le=1.0)
    measures: dict[str, ConfounderMeasure] = Field(min_length=1)
    region_level_resolves: bool


class EliminationSpec(_Node):
    reasons: dict[str, str] = Field(min_length=1)
    not_testable_eliminates: bool


class AdjudicateConfig(_Node):
    version: int = Field(ge=1)
    precedence: PrecedenceSpec
    sufficiency: SufficiencySpec
    dose_response: DoseResponseSpec
    specificity: SpecificitySpec
    exposure: ExposureSpec
    did: DidSpec
    confounder_screen: ConfounderScreenSpec
    elimination: EliminationSpec

    @model_validator(mode="after")
    def _matched_confounders_name_a_real_covariate(self) -> AdjudicateConfig:
        covariates = set(self.did.matching.covariates)
        for name, measure in self.confounder_screen.measures.items():
            if measure.scope_level != "matched":
                continue
            if measure.resolved_by not in covariates:
                raise ValueError(
                    f"confounder {name!r} claims to be resolved by "
                    f"{measure.resolved_by!r}, which is not a matching covariate "
                    f"({sorted(covariates)})"
                )
        return self

    def test_ids(self) -> dict[str, int]:
        return {
            "precedence": self.precedence.test_id,
            "sufficiency": self.sufficiency.test_id,
            "dose_response": self.dose_response.test_id,
            "specificity": self.specificity.test_id,
            "did": self.did.test_id,
            "confounder_screen": self.confounder_screen.test_id,
        }


# ---------------------------------------------------------------------------
# RECOMMEND
# ---------------------------------------------------------------------------


class MatchingSpec(_Node):
    match_on: Literal["adjudicated_driver"]
    information_levers: list[str] = Field(min_length=1)
    prefer_recovering_lever: bool
    tie_break: Literal["shortest_lead_time"]
    no_playbook_outcome: str = Field(min_length=1)


class CostTypeSpec(_Node):
    arithmetic: Literal["rate_times_basis", "depth_times_basis"]
    description: str = Field(min_length=1)


class UnobservedDepthSpec(_Node):
    """What a depth-priced lever risks when its depth cannot be observed."""

    basis: str = Field(min_length=1)
    multiplier: Literal["gross_margin_rate"]
    note: str | None = None


class CostSpec(_Node):
    bases: dict[str, str] = Field(min_length=1)
    types: dict[str, CostTypeSpec] = Field(min_length=1)
    depth_sources: dict[str, str] = Field(min_length=1)
    unobserved_depth: UnobservedDepthSpec
    display_unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def _unobserved_depth_names_a_basis(self) -> CostSpec:
        if self.unobserved_depth.basis not in self.bases:
            raise ValueError(
                f"unobserved_depth.basis {self.unobserved_depth.basis!r} is not one of "
                f"the declared cost bases ({sorted(self.bases)})"
            )
        return self


class EconomicsSpec(_Node):
    gross_margin_rate: float = Field(gt=0.0, lt=1.0)
    gross_margin_basis: str = Field(min_length=1)
    manager_hour_inr: float = Field(gt=0.0)
    manager_hour_basis: str | None = None


class CallDownSpec(_Node):
    playbook: str = Field(min_length=1)
    span_of_control_stores: int = Field(gt=0)
    minutes_per_store: float = Field(gt=0.0)
    rounding: Literal["ceiling"]
    note: str | None = None


class RecoveryConfidenceBand(_Node):
    min_cases: int = Field(ge=0)
    label: Literal["Low", "Medium", "High"]


class RecoverySpec(_Node):
    curve_from: Literal["playbook.recovery_curve_ref"]
    attributable_from: Literal["leading_hypothesis_attribution"]
    confidence_by_sample_size: list[RecoveryConfidenceBand] = Field(min_length=1)
    suppress_roi_for_information_levers: bool

    @model_validator(mode="after")
    def _bands_descend_and_reach_zero(self) -> RecoverySpec:
        """Top down, and the last one must catch everything.

        A table whose lowest band starts above zero has a sample size it
        cannot label, and the engine would have to invent one.
        """
        thresholds = [band.min_cases for band in self.confidence_by_sample_size]
        if thresholds != sorted(thresholds, reverse=True):
            raise ValueError(
                f"confidence_by_sample_size must descend by min_cases, got {thresholds}"
            )
        if thresholds[-1] != 0:
            raise ValueError(
                "the last confidence band must start at min_cases: 0, or a curve fitted "
                "on no cases has no label"
            )
        return self

    def label_for(self, sample_size: int) -> str:
        for band in self.confidence_by_sample_size:
            if sample_size >= band.min_cases:
                return band.label
        raise ValueError(f"no confidence band covers sample size {sample_size}")


class NotRecommendedSpec(_Node):
    required: bool
    error_rate_sources: dict[str, str] = Field(min_length=1)
    error_rate_rule: Literal["worst_of"]
    refuse_when: Literal["preconditions_unmet"]


class ResolutionSpec(_Node):
    rank_by: Literal["value_per_rupee"]
    value_model: Literal["residual_times_prior"]
    min_entries: int = Field(ge=1)
    not_recommended: NotRecommendedSpec


class LinkedCaseSpec(_Node):
    enabled: bool
    id_strategy: Literal["max_numeric_plus_one"]
    status: str = Field(min_length=1)
    kpi_from: Literal["linked_driver_affects_first"]
    scope_hints: dict[str, str] = Field(min_length=1)

    def scope_for(self, hint: str, scope: str) -> str:
        template = self.scope_hints.get(hint)
        if template is None:
            raise ValueError(
                f"scope hint {hint!r} has no resolution in recommend.yaml -> "
                f"linked_case.scope_hints ({sorted(self.scope_hints)})"
            )
        return template.format(scope=scope)


class DataGapSpec(_Node):
    trigger: str = Field(min_length=1)
    gap_code: str = Field(min_length=1)
    severity: str = Field(min_length=1)
    measure: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    value_when_missing: float
    detail_template: str = Field(min_length=1)
    resolution_hint_from: Literal["acquisition_playbook"]
    acquisition_lever: str = Field(min_length=1)
    no_playbook_hint: str = Field(min_length=1)


class RecommendConfig(_Node):
    version: int = Field(ge=1)
    matching: MatchingSpec
    cost: CostSpec
    economics: EconomicsSpec
    call_down: CallDownSpec
    recovery: RecoverySpec
    resolution: ResolutionSpec
    linked_case: LinkedCaseSpec
    data_gap: DataGapSpec


# ---------------------------------------------------------------------------
# NARRATE
# ---------------------------------------------------------------------------


class PersonaNarrative(_Node):
    """One reader, and what they are told first."""

    display_name: str = Field(min_length=1)
    audience: str = Field(min_length=1)
    leads_with: str = Field(min_length=1)
    max_sentences: int = Field(gt=0)
    #: `narrate.yaml` says `voice`, not `register`: a field named
    #: `register` shadows a BaseModel method and pydantic warns. Same
    #: reason `validate.yaml` loads as `validation`.
    voice: str = Field(min_length=1)
    must_cover: list[str] = Field(min_length=1)
    omit: list[str] = Field(default_factory=list)


class NumericGrounding(_Node):
    """How a number written in prose is matched to the frozen object."""

    match: Literal["round_to_token_precision"]
    tolerance_relative: float = Field(ge=0.0, lt=1.0)
    percent_expansion: bool
    ignore_patterns: dict[str, str] = Field(min_length=1)
    allow_small_integers_up_to: int = Field(ge=0)

    @model_validator(mode="after")
    def _ignore_patterns_compile(self) -> NumericGrounding:
        for name, pattern in self.ignore_patterns.items():
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"ignore pattern {name!r} does not compile: {exc}") from exc
        return self


class GroundingSpec(_Node):
    min_evidence_ids_per_sentence: int = Field(ge=1)
    max_regenerations: int = Field(ge=0)
    hard_gate_tests: list[str] = Field(min_length=1)
    numeric: NumericGrounding


class BareCausalClaim(_Node):
    pattern: str = Field(min_length=1)
    why: str = Field(min_length=1)

    @model_validator(mode="after")
    def _pattern_compiles(self) -> BareCausalClaim:
        try:
            re.compile(self.pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"bare causal pattern does not compile: {exc}") from exc
        return self


class ForbiddenPhrase(_Node):
    """A phrase banned outright, with what to write instead and why.

    `instead` is not decoration. It is quoted back to the model on the one
    regeneration it gets, and a rule with no replacement produces a second
    attempt that is merely shorter.
    """

    phrase: str = Field(min_length=1)
    instead: str = Field(min_length=1)
    why: str = Field(min_length=1)


class LanguageSpec(_Node):
    causal_connectives: list[str] = Field(min_length=1)
    required_attribution_form: str = Field(min_length=1)
    bare_causal_claims: list[BareCausalClaim] = Field(min_length=1)
    forbidden_phrases: list[ForbiddenPhrase] = Field(min_length=1)
    house_style: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _phrases_are_lowercase(self) -> LanguageSpec:
        """Matching is case-insensitive; the config must not pretend otherwise.

        A rule written "Root Cause" would read as though case mattered,
        and the next person to add one would copy the capitalisation and
        assume it does.
        """
        for entry in self.forbidden_phrases:
            if entry.phrase != entry.phrase.lower():
                raise ValueError(
                    f"forbidden phrase {entry.phrase!r} is not lowercase; matching is "
                    "case-insensitive and the config should say so"
                )
        for connective in self.causal_connectives:
            if connective != connective.lower():
                raise ValueError(f"causal connective {connective!r} is not lowercase")
        return self


class NarrateConfig(_Node):
    version: int = Field(ge=1)
    prompt_version: int = Field(ge=1)
    personas: dict[str, PersonaNarrative] = Field(min_length=1)
    grounding: GroundingSpec
    language: LanguageSpec


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class Currency(_Node):
    """The one place dollars become rupees, with the date it was true."""

    usd_to_inr: float = Field(gt=0.0)
    as_of: date
    source: str

    def to_inr(self, usd: float) -> float:
        return usd * self.usd_to_inr


class ModelPrice(_Node):
    """List price for one model, US dollars per million tokens."""

    description: str
    input_usd_per_mtok: float = Field(ge=0.0)
    output_usd_per_mtok: float = Field(ge=0.0)
    cache_read_usd_per_mtok: float = Field(ge=0.0)
    cache_write_usd_per_mtok: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _cache_rates_bracket_the_input_rate(self) -> ModelPrice:
        """A cache read is cheaper than fresh input; a cache write is dearer.

        Enforced because the whole argument for the content-hash cache in
        `llm/classify.py` is that a hit costs less than a miss. A price
        list that inverted this would make the cache read as a cost
        increase and nobody would notice for a quarter.
        """
        if self.cache_read_usd_per_mtok > self.input_usd_per_mtok:
            raise ValueError(
                f"cache read ({self.cache_read_usd_per_mtok}) costs more than fresh "
                f"input ({self.input_usd_per_mtok}); caching would be a loss"
            )
        if self.cache_write_usd_per_mtok < self.input_usd_per_mtok:
            raise ValueError(
                f"cache write ({self.cache_write_usd_per_mtok}) costs less than fresh "
                f"input ({self.input_usd_per_mtok}); a write is input plus storage"
            )
        return self


class EstimationSpec(_Node):
    """How to guess a token count when the provider reported none.

    Only ever used on a replayed fixture, and the row it produces is
    flagged. See the note in telemetry.yaml.
    """

    chars_per_token: float = Field(gt=0.0)

    def tokens_in(self, text: str | None) -> int:
        if not text:
            return 0
        return max(1, round(len(text) / self.chars_per_token))


class TelemetryTargets(_Node):
    """CLAUDE.md §"Definition of done", as loadable values."""

    cost_per_case_inr: float = Field(gt=0.0)
    latency_budget_ms: float = Field(gt=0.0)
    latency_percentile: float = Field(gt=0.0, lt=1.0)
    #: How many requests at the start of a process count as cold. The
    #: percentile is computed over the rest.
    warmup_requests: int = Field(ge=0)


class ProjectionSpec(_Node):
    """Volume assumptions. The cost per interaction is measured, not here."""

    interactions_per_week: int = Field(ge=1)
    weeks_per_year: float = Field(gt=0.0)
    months_per_year: int = Field(ge=1)


class TelemetryConfig(_Node):
    version: int = Field(ge=1)
    currency: Currency
    pricing: dict[str, ModelPrice] = Field(min_length=1)
    estimation: EstimationSpec
    targets: TelemetryTargets
    projection: ProjectionSpec

    def price(self, model: str) -> ModelPrice:
        """The price list for `model`, or a refusal naming what is priced.

        Deliberately not a `.get` returning None. An unpriced call costed
        at zero is a budget that is wrong and silent about it.
        """
        try:
            return self.pricing[model]
        except KeyError:
            raise SemanticLayerError(
                f"no price for model {model!r}; telemetry.yaml prices "
                f"{sorted(self.pricing)}. Add it there rather than defaulting to zero."
            ) from None


# ---------------------------------------------------------------------------
# Learning — the feedback loop
# ---------------------------------------------------------------------------


class VerdictAction(_Node):
    """One of the four verdict-level options."""

    label: str
    description: str
    #: True is a hit, False a miss, None no judgement — and None writes no
    #: calibration entry at all rather than an unscored row.
    scores_as_correct: bool | None = None
    requires_reason: bool = False


class DriverAction(_Node):
    label: str
    description: str
    confirms: bool


class ActionAction(_Node):
    label: str
    description: str
    accepted: bool


class FeedbackSpec(_Node):
    verdict_actions: dict[str, VerdictAction] = Field(min_length=1)
    driver_actions: dict[str, DriverAction] = Field(min_length=1)
    action_actions: dict[str, ActionAction] = Field(min_length=1)
    reason_codes: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def _a_reason_is_demanded_of_at_least_one_action(self) -> FeedbackSpec:
        if not any(action.requires_reason for action in self.verdict_actions.values()):
            raise ValueError(
                "no verdict action requires a reason; 'reject with reason' is one of "
                "the four options CLAUDE.md's loop depends on"
            )
        return self


class OutcomeHorizon(_Node):
    days: int = Field(gt=0)
    name: str
    measures: str
    description: str


class OutcomeSpec(_Node):
    horizons: list[OutcomeHorizon] = Field(min_length=1)
    unresolved_after_days: int = Field(gt=0)
    recovered_at_share_of_low: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _horizons_ascend_and_end_before_unresolved(self) -> OutcomeSpec:
        days = [horizon.days for horizon in self.horizons]
        if any(b <= a for a, b in zip(days, days[1:], strict=False)):
            raise ValueError(f"outcome horizons must ascend: {days}")
        if self.unresolved_after_days <= days[-1]:
            raise ValueError(
                f"a case is called unresolved at {self.unresolved_after_days} days, "
                f"before its last horizon at {days[-1]}"
            )
        return self

    def horizon(self, days: int) -> OutcomeHorizon:
        for horizon in self.horizons:
            if horizon.days == days:
                return horizon
        raise SemanticLayerError(
            f"no outcome horizon at D+{days}; learning.yaml declares "
            f"{[h.days for h in self.horizons]}"
        )


class PriorLearningSpec(_Node):
    enabled: bool
    strength: float = Field(gt=0.0)
    min_observations: int = Field(ge=1)
    max_absolute_shift: float = Field(gt=0.0, le=1.0)
    floor: float = Field(gt=0.0, lt=1.0)
    ceiling: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _the_band_is_a_band(self) -> PriorLearningSpec:
        if self.floor >= self.ceiling:
            raise ValueError(
                f"prior floor {self.floor} is not below ceiling {self.ceiling}"
            )
        return self

    def posterior(self, declared: float, confirmed: int, rejected: int) -> float:
        """Beta posterior mean, bounded by the shift cap and the band.

        The whole update rule, in one place, so the API, the tests, the
        screening and the overlay writer cannot each have their own idea
        of what a learned prior is.
        """
        observations = confirmed + rejected
        if not self.enabled or observations < self.min_observations:
            return declared
        alpha = declared * self.strength + confirmed
        total = self.strength + observations
        moved = alpha / total
        low = max(self.floor, declared - self.max_absolute_shift)
        high = min(self.ceiling, declared + self.max_absolute_shift)
        return min(max(moved, low), high)


class Quantiles(_Node):
    """What p25 and p75 mean. A definition, not a tunable."""

    low: float = Field(gt=0.0, lt=1.0)
    high: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _low_is_below_high(self) -> Quantiles:
        if self.low >= self.high:
            raise ValueError(f"low quantile {self.low} is not below high {self.high}")
        return self


class RecoveryLearningSpec(_Node):
    enabled: bool
    quantiles: Quantiles
    min_realisations: int = Field(ge=1)
    max_absolute_shift: float = Field(gt=0.0, le=1.0)
    implausible_share_above: float = Field(gt=0.0)

    def blend(
        self, declared: float, declared_n: int, observed: float, observed_n: int
    ) -> float:
        """Weighted move towards what was observed.

        Never a refit — see the note in learning.yaml on why the raw
        realisations behind a seeded curve do not exist to refit against.
        """
        if not self.enabled or observed_n < self.min_realisations:
            return declared
        if declared_n + observed_n == 0:
            return declared
        moved = (declared_n * declared + observed_n * observed) / (
            declared_n + observed_n
        )
        low = declared - self.max_absolute_shift
        high = declared + self.max_absolute_shift
        return min(max(moved, low), high)


class CalibrationLearningSpec(_Node):
    write_on_feedback: bool
    outcome_supersedes_feedback: bool


class LearningConfig(_Node):
    version: int = Field(ge=1)
    feedback: FeedbackSpec
    outcomes: OutcomeSpec
    priors: PriorLearningSpec
    recovery: RecoveryLearningSpec
    calibration: CalibrationLearningSpec


# ---------------------------------------------------------------------------
# Series — the sparkline behind a KPI card
# ---------------------------------------------------------------------------


class KpiSeries(_Node):
    """One KPI's executable series, or the reason it has none.

    Deliberately separate from the contract's `formula_sql`, which is a
    DEFINITION against the conceptual model rather than a query against
    the tables this warehouse loaded. See the note at the top of
    series.yaml.
    """

    available: bool
    grain: Grain
    unit: str
    description: str | None = None
    source_table: str | None = None
    sql: str | None = None
    #: Required when `available` is false. A KPI with no series says why.
    reason: str | None = None

    @model_validator(mode="after")
    def _an_unavailable_series_explains_itself(self) -> KpiSeries:
        if self.available and not (self.sql and self.source_table):
            raise ValueError(
                "a series declared available needs both a source_table and sql"
            )
        if not self.available and not self.reason:
            raise ValueError(
                "a KPI with no series must say why; an absence with no reason "
                "reads as an oversight and gets 'fixed' with something wrong"
            )
        return self


class SeriesConfig(_Node):
    version: int = Field(ge=1)
    points: int = Field(ge=2)
    series: dict[str, KpiSeries] = Field(min_length=1)


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
    gather: GatherConfig
    adjudicate: AdjudicateConfig
    recommend: RecommendConfig
    narrate: NarrateConfig
    telemetry: TelemetryConfig
    learning: LearningConfig
    series: SeriesConfig

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

        # Every KPI carries a series entry, even if only to say it has
        # none. A KPI the watchlist cannot draw and cannot explain is a
        # blank card nobody can account for.
        missing_series = sorted(set(self.kpis) - set(self.series.series))
        if missing_series:
            raise ValueError(
                f"series.yaml declares nothing for {missing_series}; a KPI with no "
                "series must say so explicitly rather than by omission"
            )
        unknown_series = sorted(set(self.series.series) - set(self.kpis))
        if unknown_series:
            raise ValueError(f"series.yaml declares series for non-KPIs {unknown_series}")

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

        # The six tests are named in two files: adjudication.yaml says what
        # each is worth, adjudicate.yaml says what it reads. A test in one
        # and not the other is a test nobody can run or nobody can score.
        scored = {name: spec.test_id for name, spec in self.adjudication.tests.items()}
        operational = self.adjudicate.test_ids()
        if set(scored) != set(operational):
            raise ValueError(
                f"adjudication.yaml scores {sorted(scored)} and adjudicate.yaml reads "
                f"{sorted(operational)}; every test needs both"
            )
        for name, test_id in operational.items():
            if scored[name] != test_id:
                raise ValueError(
                    f"test {name!r} is {scored[name]} in adjudication.yaml and "
                    f"{test_id} in adjudicate.yaml"
                )

        graph = self.causal_graph.hypotheses
        for name in self.gather.structured.templates:
            if name not in graph:
                raise ValueError(
                    f"structured template {name!r} is not a hypothesis in the causal graph"
                )
        for name in self.gather.structured.unavailable:
            if name not in graph:
                raise ValueError(
                    f"gather declares {name!r} as having no template, but it is not a "
                    "hypothesis in the causal graph"
                )
        for name in self.gather.hypotheses.already_removed:
            if name not in graph:
                raise ValueError(
                    f"gather removes {name!r} from screening, but it is not a hypothesis "
                    "in the causal graph"
                )
        for name in self.adjudicate.precedence.causes:
            if name not in graph:
                raise ValueError(
                    f"adjudicate declares a cause series for {name!r}, which is not a "
                    "hypothesis in the causal graph"
                )
        declared_confounders = {
            confounder
            for template in graph.values()
            for confounder in template.confounders
        }
        missing = declared_confounders - set(self.adjudicate.confounder_screen.measures)
        if missing:
            raise ValueError(
                f"the causal graph declares confounders {sorted(missing)} that Test 6 "
                "has no way to measure; add a measure or stop declaring them"
            )

        for name, corpus in self.gather.unstructured.corpora.items():
            if corpus.source_system not in self.causal_graph.sources:
                raise ValueError(
                    f"corpus {name!r} names source {corpus.source_system!r}, which the "
                    "causal graph's source registry does not declare"
                )

        # --- RECOMMEND ----------------------------------------------------
        recommend = self.recommend
        for name, kpi in self.kpis.items():
            ownership = kpi.driver_ownership
            if ownership is None:
                continue
            if ownership.default.role != kpi.owner_role:
                raise ValueError(
                    f"kpi {name!r} names owner_role {kpi.owner_role!r} but its "
                    f"driver_ownership default is {ownership.default.role!r}; the KPI's "
                    "own owner is who a driver with no named owner falls back to"
                )
            for driver in ownership.by_driver:
                if driver not in hypothesis_names:
                    raise ValueError(
                        f"kpi {name!r} assigns ownership of {driver!r}, which is not a "
                        "hypothesis in the causal graph"
                    )
                # A driver that cannot move this KPI has no owner OF this
                # KPI. Naming one is dead config that reads as a decision
                # somebody made: no recommendation can ever reach the row,
                # because a case is only opened on a movement the driver
                # could have caused.
                if name not in self.causal_graph.hypotheses[driver].affects:
                    raise ValueError(
                        f"kpi {name!r} assigns an owner for {driver!r}, which does not "
                        f"affect it (it affects "
                        f"{self.causal_graph.hypotheses[driver].affects}); no case on "
                        "this KPI could ever reach that row"
                    )

        # One role, one title, everywhere. Two contracts calling the same
        # role by different names would put two different people on the
        # screen for one job.
        titles: dict[str, tuple[str, str]] = {}
        for name, kpi in self.kpis.items():
            if kpi.driver_ownership is None:
                continue
            owners = [kpi.driver_ownership.default, *kpi.driver_ownership.by_driver.values()]
            for owner in owners:
                seen = titles.setdefault(owner.role, (owner.title, name))
                if seen[0] != owner.title:
                    raise ValueError(
                        f"role {owner.role!r} is titled {seen[0]!r} in {seen[1]} and "
                        f"{owner.title!r} in {name}"
                    )

        for name, playbook in self.playbooks.items():
            model = playbook.cost_model
            if model.type not in recommend.cost.types:
                raise ValueError(
                    f"playbook {name!r} uses cost model type {model.type!r}, which "
                    f"recommend.yaml does not declare ({sorted(recommend.cost.types)})"
                )
            if model.scales_with not in recommend.cost.bases:
                raise ValueError(
                    f"playbook {name!r} scales with {model.scales_with!r}, which is not "
                    f"a cost basis in recommend.yaml ({sorted(recommend.cost.bases)})"
                )
            arithmetic = recommend.cost.types[model.type].arithmetic
            if arithmetic == "depth_times_basis" and model.depth_source is None:
                raise ValueError(
                    f"playbook {name!r} is priced on a depth but names no depth_source"
                )
            if model.depth_source is not None and (
                model.depth_source not in recommend.cost.depth_sources
            ):
                raise ValueError(
                    f"playbook {name!r} takes its depth from {model.depth_source!r}, "
                    f"which recommend.yaml does not declare "
                    f"({sorted(recommend.cost.depth_sources)})"
                )
            if playbook.linked_case_template is not None:
                hint = playbook.linked_case_template.scope_hint
                if hint not in recommend.linked_case.scope_hints:
                    raise ValueError(
                        f"playbook {name!r} links a case with scope hint {hint!r}, which "
                        f"recommend.yaml cannot resolve "
                        f"({sorted(recommend.linked_case.scope_hints)})"
                    )

        if recommend.call_down.playbook not in self.playbooks:
            raise ValueError(
                f"recommend.yaml names call-down playbook "
                f"{recommend.call_down.playbook!r}, which does not exist"
            )
        acquiring = {
            playbook.lever
            for playbook in self.playbooks.values()
        }
        if recommend.data_gap.acquisition_lever not in acquiring:
            raise ValueError(
                f"recommend.yaml resolves data gaps with lever "
                f"{recommend.data_gap.acquisition_lever!r}, which no playbook carries "
                f"({sorted(acquiring)}); every gap would come back unresolvable"
            )
        if recommend.data_gap.acquisition_lever not in recommend.matching.information_levers:
            raise ValueError(
                f"lever {recommend.data_gap.acquisition_lever!r} resolves data gaps but "
                "is not an information lever; a lever that recovers revenue is an "
                "action, not a way to close a source gap"
            )
        if recommend.data_gap.trigger not in self.adjudication.triggers:
            raise ValueError(
                f"recommend.yaml writes a data gap on trigger "
                f"{recommend.data_gap.trigger!r}, which is not a declared trigger"
            )

        # A curve carries a label AND a sample size, and the sample size is
        # what the label is supposed to mean. Checked here so the two can
        # never drift: an edit to one without the other fails to load.
        for curve_name, curve in self.recovery_curves.curves.items():
            expected = recommend.recovery.label_for(curve.sample_size)
            if curve.confidence_label != expected:
                raise ValueError(
                    f"recovery curve {curve_name!r} is fitted on {curve.sample_size} "
                    f"cases, which recommend.yaml labels {expected!r}, but the curve "
                    f"declares {curve.confidence_label!r}"
                )

        # --- NARRATE ------------------------------------------------------
        narrate = self.narrate
        for persona in narrate.personas:
            missing_from = sorted(
                name
                for name, kpi in self.kpis.items()
                if persona not in kpi.access_policy.personas
            )
            if missing_from:
                raise ValueError(
                    f"narrate.yaml writes for persona {persona!r}, which "
                    f"{missing_from} do not grant access to; the narrative would be "
                    "written for a reader the row filter refuses to serve"
                )

        declared_gates = sorted(
            name for name, spec in self.adjudication.tests.items() if spec.type == "hard_gate"
        )
        if sorted(narrate.grounding.hard_gate_tests) != declared_gates:
            raise ValueError(
                f"narrate.yaml gates causal language on "
                f"{sorted(narrate.grounding.hard_gate_tests)}, but adjudication.yaml "
                f"declares the hard gates as {declared_gates}. A connective would be "
                "allowed on a hypothesis that never passed one."
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
    gather_path = base / "gather.yaml"
    adjudicate_path = base / "adjudicate.yaml"
    recommend_path = base / "recommend.yaml"
    narrate_path = base / "narrate.yaml"
    telemetry_path = base / "telemetry.yaml"
    learning_path = base / "learning.yaml"
    series_path = base / "series.yaml"

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
        "gather": _build(GatherConfig, _read_yaml(gather_path), gather_path),
        "adjudicate": _build(
            AdjudicateConfig, _read_yaml(adjudicate_path), adjudicate_path
        ),
        "recommend": _build(RecommendConfig, _read_yaml(recommend_path), recommend_path),
        "narrate": _build(NarrateConfig, _read_yaml(narrate_path), narrate_path),
        "telemetry": _build(
            TelemetryConfig, _read_yaml(telemetry_path), telemetry_path
        ),
        "learning": _build(LearningConfig, _read_yaml(learning_path), learning_path),
        "series": _build(SeriesConfig, _read_yaml(series_path), series_path),
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
    "AccessPolicy",
    "AdjudicateConfig",
    "AdjudicationConfig",
    "ActionAction",
    "AttributionSpec",
    "BandGate",
    "BareCausalClaim",
    "Baseline",
    "CalibrationLearningSpec",
    "CalendarGate",
    "CalendarMismatch",
    "CallDownSpec",
    "CausalGraph",
    "ClassificationSpec",
    "ConfounderMeasure",
    "ConfounderScreenSpec",
    "Corpus",
    "CostModel",
    "CostSpec",
    "CostTypeSpec",
    "Currency",
    "DataGapSpec",
    "DefinitionConflict",
    "DefinitionDriftCheck",
    "DidSpec",
    "DriverAction",
    "DoseResponseSpec",
    "DriverOwner",
    "DriverOwnership",
    "EconomicsSpec",
    "EliminationSpec",
    "EntityKeyMismatch",
    "EstimationSpec",
    "EvidenceVocabulary",
    "ExitCode",
    "FeedbackSpec",
    "ForbiddenPhrase",
    "GatherConfig",
    "get_semantic_layer",
    "Governance",
    "GrainMismatch",
    "GroundingSpec",
    "HypothesisTemplate",
    "KpiContract",
    "KpiSeries",
    "LanguageSpec",
    "LearningConfig",
    "LinkedCaseSpec",
    "load_semantic_layer",
    "load_timed",
    "MANDATORY_PLAYBOOK_FIELDS",
    "MatchingSpec",
    "Materiality",
    "MaterialityGate",
    "ModelPrice",
    "NarrateConfig",
    "NotRecommendedSpec",
    "NumericGrounding",
    "OutcomeHorizon",
    "OutcomeSpec",
    "PersonaNarrative",
    "Playbook",
    "PrecedenceSpec",
    "PriorLearningSpec",
    "ProjectionSpec",
    "QualifyConfig",
    "Quantiles",
    "RecommendConfig",
    "Reconciliation",
    "RecoveryConfidenceBand",
    "RecoveryCurve",
    "RecoveryCurves",
    "RecoveryLearningSpec",
    "RecoverySpec",
    "ResolutionSpec",
    "RestatementCheck",
    "RestraintSpec",
    "RetrievalSpec",
    "RevenueDefinition",
    "RowCountDeltaCheck",
    "SemanticLayer",
    "SeriesConfig",
    "SemanticLayerError",
    "Series",
    "Severity",
    "SingleTransactionCheck",
    "SourceFreshnessCheck",
    "SourceLocation",
    "SpecificityGate",
    "SpecificitySpec",
    "StructuredTemplate",
    "SufficiencySpec",
    "TelemetryConfig",
    "TelemetryTargets",
    "Units",
    "UnobservedDepthSpec",
    "ValidateChecks",
    "VerdictAction",
    "ValidateConfig",
    "WarehouseConfig",
]
