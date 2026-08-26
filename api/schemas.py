"""Response bodies.

MOST OF THEM ARE NOT HERE, AND THAT IS THE POINT. CLAUDE.md §"Stack"
says the Pydantic models in `engine/contracts.py` ARE the evidence and
adjudication contracts. So `GET /api/cases/{id}` returns an
`Adjudication`, `GET /api/evidence/{id}` returns an `Evidence`, and this
module defines only the shapes the ENGINE has no opinion about: a
watchlist row, a KPI summary, a telemetry roll-up. Re-declaring the
engine's types here as "API models" would create a second contract that
drifts from the first.

Every model forbids extra fields for the same reason the engine's do: a
response with a field nobody declared is a field nobody tested.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from engine.contracts import (
    Adjudication,
    Evidence,
    GateResult,
    Recommendation,
    Stage,
    Verdict,
)


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class Whoami(_Body):
    """Who the signed header resolved to. What the UI puts in the corner."""

    user_id: str
    persona: str
    display_name: str | None = None
    region: str | None = None
    store_id: str | None = None
    #: KPIs this persona's access policy admits. Not every KPI exists for
    #: every reader, and a UI that offers one it cannot open is a UI that
    #: generates 403s for a living.
    kpis: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


class KpiSummary(_Body):
    """One KPI, as the catalogue lists it."""

    kpi: str
    display_name: str
    unit: str
    grain: str
    owner_role: str
    definition: str
    materiality_value: float | None = None
    materiality_unit: str | None = None
    materiality_display: str | None = None
    #: False when the contract declares no materiality limit or too little
    #: history — `qcomm_fulfilment_rate` is both, and cannot open a case.
    can_open_case: bool = True
    drivers: tuple[str, ...] = ()
    related_kpis: tuple[str, ...] = ()
    source_systems: tuple[str, ...] = ()
    refresh_sla_hours: float | None = None


class SeriesPoint(_Body):
    period: str
    value: float


class KpiCard(_Body):
    """One watchlist card: the contract, the sparkline, and the status.

    `series_available` is false for two of the six KPIs, and `series_reason`
    says why. A card that silently drew nothing would look like a KPI that
    had not moved.
    """

    kpi: str
    display_name: str
    unit: str
    grain: str
    owner_role: str
    materiality_display: str | None = None
    can_open_case: bool = True

    series: tuple[SeriesPoint, ...] = ()
    series_available: bool = True
    series_reason: str | None = None
    series_unit: str | None = None
    source_table: str | None = None
    latest: float | None = None
    previous: float | None = None
    change_pct: float | None = None
    rows_filtered: int = 0

    #: What the watchlist has to say about this KPI right now: an open
    #: case, a movement a gate stopped, or nothing at all.
    status: Literal["open_case", "suppressed", "monitoring_only", "quiet"] = "quiet"
    status_detail: str | None = None
    case_id: str | None = None


class SemanticContract(_Body):
    """One KPI's whole contract, as the semantic layer holds it.

    `GET /api/semantic/{kpi_id}` exists so a number on screen can be
    traced past the evidence record to the DEFINITION it was computed
    under — the SQL, the exclusions, the thresholds, the access policy.
    Returned as the loaded contract's own dump rather than a re-modelled
    subset, because a subset is a second definition.
    """

    kpi: str
    version: int
    contract: dict[str, Any]
    source_file: str
    #: What this persona is allowed to see of it, resolved for the caller.
    row_predicate: str
    masked_columns: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

WatchlistState = Literal["open", "suppressed"]


class WatchlistItem(_Body):
    """A movement the system looked at, and what happened to it.

    THE SUPPRESSED ROWS ARE THE INTERESTING ONES. A watchlist of open
    cases is a queue. A watchlist that also shows what was stopped, by
    which gate, and why, is a record of judgement — and it is the only
    way a reader can tell "nothing is wrong" from "nothing was checked".
    """

    state: WatchlistState
    kpi: str
    scope: str
    grain: str
    period: str

    case_id: str | None = None
    #: The movement itself, formatted for the row: "-18.0%".
    headline: str | None = None
    status: str | None = None
    verdict: str | None = None
    confidence_calibrated: float | None = None
    materiality_multiple: float | None = None
    opened_at: datetime | None = None
    elapsed_ms: float | None = None

    #: Suppressed rows only: the gate that stopped it, by number and name,
    #: and the outcome code it stopped with.
    stopped_by_gate: int | None = None
    stopped_by_name: str | None = None
    outcome_code: str | None = None
    detail: str | None = None


class Watchlist(_Body):
    cards: tuple[KpiCard, ...] = ()
    open_cases: tuple[WatchlistItem, ...] = ()
    suppressed: tuple[WatchlistItem, ...] = ()
    generated_at: datetime
    #: The window the suppressed panel covers, for its heading.
    period: str
    scanned: bool = False


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


class CaseRunRequest(_Body):
    """What to investigate. The body of `POST /api/cases/run`."""

    kpi: str
    scope: str
    grain: Literal["daily", "weekly", "monthly"] = "monthly"
    period: str
    comparison_period: str | None = None
    case_id: str | None = None
    direction: Literal["up", "down"] = "down"
    #: Persona the narrative is written for once the verdict lands.
    persona: str | None = None
    #: Whether RECOMMEND may open the chained "cause of the cause" case.
    #: Off by default over HTTP: a GET-shaped action that writes a case is
    #: a surprise, and the demo turns it on explicitly.
    open_linked_case: bool = False


class StageMessage(_Body):
    """One SSE event. The progress rail, one entry at a time."""

    stage: Stage | Literal["NARRATE"]
    ordinal: int
    status: Literal["ok", "stopped", "failed"]
    duration_ms: float
    case_id: str
    detail: str | None = None
    outcome_code: str | None = None
    #: Small, stage-shaped summary. The full objects come from
    #: `GET /api/cases/{id}` — an SSE frame is a progress update, not a
    #: delivery mechanism for a 70-record evidence ledger.
    summary: dict[str, Any] = Field(default_factory=dict)


class CaseDetail(_Body):
    """The whole case. What `GET /api/cases/{id}` returns.

    Every element the accept criterion names is here as an engine
    contract: `adjudication.gates`, `adjudication.hypotheses[].tests` (all
    six per hypothesis), `adjudication.evidence`,
    `adjudication.confidence.components` (s1-s6),
    `adjudication.confidence.caps_applied`, and `verdict.reason_text`.
    """

    case_id: str
    adjudication: Adjudication
    #: None on a movement a gate killed and on a case still in progress.
    verdict: Verdict | None = None
    recommendation: Recommendation | None = None
    decision_trace: tuple[str, ...] = ()
    #: "canonical" for a Number Registry scenario assembled from the
    #: generator's config, "live" for one this process investigated. The
    #: two diverge today and a screen that showed both without saying
    #: which was which would be the worst of both.
    source: str = "live"
    config_ref: str | None = None
    #: Personas `narrate.yaml` declares. The UI's switcher is built from
    #: this rather than from a list in the frontend, so it cannot offer a
    #: reader the narrator has no voice for.
    narrative_personas: tuple[str, ...] = ()
    latency_ms: dict[str, float] = Field(default_factory=dict)
    run_as_persona: str
    stored_at: datetime


class Claim(_Body):
    """One narrative sentence and the evidence it rests on."""

    sentence: str
    evidence_ids: tuple[str, ...] = ()


class NarrativeResponse(_Body):
    """A persona's narrative, and the grounding audit behind it."""

    case_id: str
    persona: str
    text: str
    claims: tuple[Claim, ...] = ()
    claims_checked: int = 0
    claims_linked: int = 0
    claims_stripped: int = 0
    regenerated: bool = False
    model: str
    from_fixture: bool = False
    #: Renders under the narrative: "grounding: 14/14 claims linked - 0 stripped".
    grounding: str


class FeedbackRequest(_Body):
    """What a reader did about a case."""

    action: Literal["accept", "reject", "escalate", "act", "dismiss", "request_more"]
    comment: str | None = None


class FeedbackResponse(_Body):
    feedback_id: str
    case_id: str
    action: str
    recorded_at: datetime


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------


class AskRequest(_Body):
    question: str = Field(min_length=1)
    persona: str | None = None


class AskResponse(_Body):
    """What a question resolved to, and what is still missing.

    A question that does not resolve comes back as a CLARIFICATION rather
    than a guess. Running an investigation against a scope the asker did
    not mean produces a confident answer to the wrong question, which is
    worse than asking.
    """

    question: str
    intent: str
    resolved: bool
    run: CaseRunRequest | None = None
    clarification: str | None = None
    #: What the resolver matched, so the UI can show its working.
    matched: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    from_fixture: bool = False


# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------


class CalibrationBand(_Body):
    label: str
    low: float
    high: float
    cases: int
    mean_confidence: float
    accuracy: float | None = None
    gap: float | None = None
    thin: bool = False


class CalibrationResponse(_Body):
    """The organisation's own track record. Computed from the ledger."""

    scored_cases: int
    abstained_cases: int
    total_cases: int
    abstention_rate: float
    calibrating: bool
    method: str
    bands: tuple[CalibrationBand, ...] = ()
    ece_raw: float
    ece_calibrated: float
    by_case_type: dict[str, dict[str, Any]] = Field(default_factory=dict)


class TelemetryResponse(_Body):
    """Cost and latency, measured. Never asserted."""

    requests: int
    warm_requests: int
    p95_latency_ms: float | None = None
    latency_budget_ms: float
    latency_within_budget: bool | None = None
    mean_cost_inr: float
    cost_ceiling_inr: float
    cost_within_ceiling: bool
    #: True when any row priced a call whose token counts were estimated
    #: from text rather than billed — always the case offline.
    cost_estimated: bool = False
    llm_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    rows_released_to_llm: int = 0
    grounding_claims_checked: int = 0
    grounding_claims_stripped: int = 0
    projection: dict[str, Any] = Field(default_factory=dict)


class AuditEntry(_Body):
    """One row of `audit_log`. The statement is a hash, never the text."""

    audit_id: str
    occurred_at: datetime
    user_id: str
    persona: str
    kpi: str
    statement_hash: str
    row_predicate: str
    rows_returned: int
    rows_filtered: int
    columns_masked: tuple[str, ...] = ()
    rows_released_to_llm: int = 0
    purpose: str | None = None
    #: True when this row records rows crossing the trust boundary into a
    #: model prompt, rather than a read that stayed inside the process.
    released: bool = False


class AuditResponse(_Body):
    entries: tuple[AuditEntry, ...] = ()
    total: int
    reads: int
    releases: int
    rows_filtered_by_policy: int
    rows_released_to_llm: int


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------


class FeedbackOptions(_Body):
    """The feedback vocabulary, served from the contract.

    The UI renders its buttons from this rather than from a list in the
    frontend. Two copies of a closed set is one too many — and the second
    copy is the one that ships a button the engine refuses.
    """

    verdict_actions: dict[str, dict[str, Any]]
    driver_actions: dict[str, dict[str, Any]]
    action_actions: dict[str, dict[str, Any]]
    reason_codes: dict[str, str]


class VerdictFeedbackRequest(_Body):
    """One of the four verdict-level options."""

    action: Literal["accept", "modify", "reject", "request_investigation"]
    reason_code: str | None = None
    comment: str | None = None


class DriverFeedbackRequest(_Body):
    """Confirm or reject one driver, or accept or reject one action."""

    driver: str = Field(description="The hypothesis tag, or the playbook ref")
    action: str
    reason_code: str | None = None
    comment: str | None = None


class FeedbackResult(_Body):
    """What the feedback MOVED, not that it was received.

    A loop whose response is `{"ok": true}` is a loop nobody can tell is
    running. `notes` carries the reason when nothing moved, which is a
    real and frequent answer — "Request investigation" writes no
    calibration entry by design.
    """

    feedback_id: str
    case_id: str
    level: str
    action: str
    target_id: str | None = None
    recorded_at: datetime

    moved_calibration: bool = False
    ledger_entry_id: str | None = None

    moved_prior: bool = False
    prior_kpi: str | None = None
    prior_hypothesis: str | None = None
    prior_before: float | None = None
    prior_after: float | None = None
    prior_shift: float | None = None
    overlay_path: str | None = None

    notes: tuple[str, ...] = ()


class OutcomeRequest(_Body):
    """What actually happened, at one of the declared horizons."""

    horizon_days: int
    cause_confirmed: bool | None = None
    realised_recovery_inr: float | None = None
    attributable_inr: float | None = None
    expected_low_inr: float | None = None
    expected_high_inr: float | None = None
    horizon_weeks: int | None = None
    action_taken: str | None = None
    note: str | None = None
    recorded_at: datetime | None = None


class OutcomeResult(_Body):
    case_id: str
    horizon_days: int
    horizon_name: str
    outcome: str
    cause_confirmed: bool | None = None
    recovered: bool | None = None
    realised_recovery_inr: float | None = None
    expected_low_inr: float | None = None
    expected_high_inr: float | None = None
    was_correct: bool | None = None
    recorded_at: datetime
    ledger_entry_id: str | None = None
    #: True when this outcome replaced a calibration entry the reader's
    #: feedback had written. What happened outranks what we were told.
    superseded_feedback: bool = False


class PriorStanding(_Body):
    """One hypothesis's prior for one KPI, declared beside effective."""

    kpi: str
    hypothesis: str
    declared: float
    effective: float
    shift: float
    confirmed: int
    rejected: int
    #: True when the shift cap is binding: more feedback will not move it,
    #: and the causal graph itself needs a person to look at it.
    capped: bool = False


class PriorsResponse(_Body):
    priors: tuple[PriorStanding, ...] = ()
    strength: float
    max_absolute_shift: float
    overlay_path: str
    overlay_written: bool = False


class CurveStanding(_Body):
    """One recovery curve, declared beside where realisations moved it."""

    curve_ref: str
    declared_p25: float
    declared_p75: float
    declared_sample_size: int
    observed_p25: float | None = None
    observed_p75: float | None = None
    observations: int = 0
    excluded_implausible: int = 0
    effective_p25: float
    effective_p75: float
    sample_size: int
    moved: bool = False


class ErrorBody(_Body):
    """Every failure, one shape."""

    error: str
    detail: str
    hint: str | None = None


__all__ = [
    "AskRequest",
    "AskResponse",
    "AuditEntry",
    "AuditResponse",
    "CalibrationBand",
    "CalibrationResponse",
    "CaseDetail",
    "CurveStanding",
    "CaseRunRequest",
    "Claim",
    "DriverFeedbackRequest",
    "ErrorBody",
    "FeedbackOptions",
    "FeedbackRequest",
    "FeedbackResult",
    "FeedbackResponse",
    "GateResult",
    "KpiCard",
    "KpiSummary",
    "NarrativeResponse",
    "OutcomeRequest",
    "OutcomeResult",
    "PriorStanding",
    "PriorsResponse",
    "SemanticContract",
    "SeriesPoint",
    "StageMessage",
    "TelemetryResponse",
    "VerdictFeedbackRequest",
    "Watchlist",
    "WatchlistItem",
    "Whoami",
]
