/**
 * The API's shapes, as TypeScript.
 *
 * Hand-written rather than generated, and deliberately partial: the UI
 * declares the fields it renders and nothing else, so a field added to an
 * engine contract does not require a frontend change to keep compiling.
 * Every one of these mirrors a Pydantic model in `engine/contracts.py` or
 * `api/schemas.py` — those are the contracts; this is a reader of them.
 */

export type Stage = "VALIDATE" | "QUALIFY" | "GATHER" | "ADJUDICATE" | "VERDICT";
export type StageOrNarrate = Stage | "NARRATE";

export const STAGES: Stage[] = [
  "VALIDATE",
  "QUALIFY",
  "GATHER",
  "ADJUDICATE",
  "VERDICT",
];

export type VerdictValue =
  | "EXPLAINED"
  | "PARTIALLY_EXPLAINED"
  | "INSUFFICIENT_EVIDENCE";

export interface LineageStep {
  step: number;
  operation: string;
  description: string;
  inputs: string[];
  ref: string;
  statement: string | null;
}

export interface Evidence {
  evidence_id: string;
  kind: string;
  produced_by: "code" | "model";
  label: string;
  value: number | string | null;
  unit: string | null;
  reliability: number;
  source_system: string;
  method: string;
  source_ref: string;
  source_as_of: string;
  retrieved_at: string | null;
  freshness_hours: number;
  completeness: number;
  lineage: LineageStep[];
  assumptions: string[];
  notes: string | null;
}

export interface TestResult {
  test_id: number;
  name: string;
  type: "hard_gate" | "weighted" | "cap";
  passed: boolean;
  testable: boolean;
  statistic: number | null;
  p_value: number | null;
  detail: string;
  weight: number | null;
  evidence_ids: string[];
}

export interface Hypothesis {
  hypothesis_id: string;
  label: string;
  description: string;
  status: "live" | "supported" | "eliminated";
  elimination_reason: string | null;
  tests: TestResult[];
  required_sources: string[];
  missing_sources: string[];
  verifiable: boolean;
  attributed_share: number | null;
  attributed: Evidence | null;
  residual_held: Evidence | null;
  evidence_ids: string[];
}

export interface GateResult {
  gate_id: number;
  name: string;
  passed: boolean;
  outcome_code: string | null;
  detail: string;
  evidence_ids: string[];
}

export interface ConfidenceComponent {
  key: string;
  name: string;
  value: number;
  weight: number;
  detail: string;
  evidence_ids: string[];
}

export interface CapApplied {
  name: string;
  condition: string;
  ceiling: number;
  forced_trigger: string | null;
}

export interface ConfidenceBreakdown {
  components: ConfidenceComponent[];
  raw: number;
  caps_applied: CapApplied[];
  after_caps: number;
  calibrated: number;
  calibration_method: string;
  calibration_sample_size: number | null;
}

export interface Adjudication {
  case_id: string;
  kpi: string;
  scope: string;
  grain: string;
  period: string;
  opened_at: string;
  status: string;
  headline_movement: Evidence;
  attributed: Evidence[];
  qualified_residual: Evidence;
  materiality: Evidence;
  empirical_band: Evidence | null;
  gates: GateResult[];
  hypotheses: Hypothesis[];
  coverage: number;
  confidence: ConfidenceBreakdown;
  triggers_fired: string[];
  evidence: Evidence[];
  elapsed_ms: number | null;
}

export interface Verdict {
  case_id: string;
  value: VerdictValue;
  reason_code: string | null;
  reason_text: string | null;
  triggers_fired: string[];
  coverage: number;
  confidence_calibrated: number;
  decided_at: string;
}

export interface Recommendation {
  case_id: string;
  action: string;
  playbook_ref: string;
  cost: Evidence;
  expected_recovery_low: Evidence;
  expected_recovery_high: Evidence;
  roi_low: number | null;
  roi_high: number | null;
  recovery_confidence: string;
  sample_size: number;
  effort: string | null;
  not_recommended: string[];
  linked_case_id: string | null;
  evidence_ids: string[];
}

export interface CaseDetail {
  case_id: string;
  adjudication: Adjudication;
  verdict: Verdict | null;
  recommendation: Recommendation | null;
  decision_trace: string[];
  latency_ms: Record<string, number>;
  run_as_persona: string;
  stored_at: string;
  source: string;
  config_ref: string | null;
  narrative_personas: string[];
}

export interface SeriesPoint {
  period: string;
  value: number;
}

export interface KpiCard {
  kpi: string;
  display_name: string;
  unit: string;
  grain: string;
  owner_role: string;
  materiality_display: string | null;
  can_open_case: boolean;
  series: SeriesPoint[];
  series_available: boolean;
  series_reason: string | null;
  series_unit: string | null;
  source_table: string | null;
  latest: number | null;
  previous: number | null;
  change_pct: number | null;
  rows_filtered: number;
  status: "open_case" | "suppressed" | "monitoring_only" | "quiet";
  status_detail: string | null;
  case_id: string | null;
}

export interface WatchlistItem {
  state: "open" | "suppressed";
  kpi: string;
  scope: string;
  grain: string;
  period: string;
  case_id: string | null;
  headline: string | null;
  status: string | null;
  verdict: string | null;
  confidence_calibrated: number | null;
  materiality_multiple: number | null;
  opened_at: string | null;
  elapsed_ms: number | null;
  stopped_by_gate: number | null;
  stopped_by_name: string | null;
  outcome_code: string | null;
  detail: string | null;
}

export interface Watchlist {
  cards: KpiCard[];
  open_cases: WatchlistItem[];
  suppressed: WatchlistItem[];
  generated_at: string;
  period: string;
  scanned: boolean;
}

export interface Claim {
  sentence: string;
  evidence_ids: string[];
}

export interface Narrative {
  case_id: string;
  persona: string;
  text: string;
  claims: Claim[];
  claims_checked: number;
  claims_linked: number;
  claims_stripped: number;
  regenerated: boolean;
  model: string;
  from_fixture: boolean;
  grounding: string;
}

export interface StageMessage {
  stage: StageOrNarrate;
  ordinal: number;
  status: "ok" | "stopped" | "failed";
  duration_ms: number;
  case_id: string;
  detail: string | null;
  outcome_code: string | null;
  summary: Record<string, unknown>;
}
