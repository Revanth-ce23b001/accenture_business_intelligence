"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { ConfidenceBreakdown, Narrative } from "@/lib/types";
import { EvidenceChip } from "@/components/evidence-drawer";
import { PersonaSwitcher } from "@/components/persona-switcher";
import { Chip, Meter, Panel, ProvenanceBadge, cn } from "@/components/ui/primitives";

/**
 * The persistent right rail: how sure, how grounded, who is reading.
 *
 * IT IS PERSISTENT ON PURPOSE. The confidence and the grounding are not a
 * tab a reader might not open — they are the qualifications on everything
 * in the other three panes, and a qualification that can be navigated
 * away from is a qualification nobody reads.
 */

export function RightRail({
  caseId,
  confidence,
  persona,
  personas,
}: {
  caseId: string;
  confidence: ConfidenceBreakdown;
  persona: string;
  personas: readonly string[];
}) {
  return (
    <aside className="space-y-4">
      <Panel title="Reading as">
        <div className="p-4">
          <PersonaSwitcher persona={persona} label="" options={personas} />
          <p className="mt-2 text-xs text-ink-muted">
            Switching re-renders the narrative for a different reader. It does
            not re-run the pipeline: the case, the statistics and the verdict
            are the same object.
          </p>
        </div>
      </Panel>

      <NarrativePanel caseId={caseId} persona={persona} />
      <ConfidencePanel confidence={confidence} />
      <ProvenanceLegend />
    </aside>
  );
}

/* ------------------------------------------------------------------ */
/* Narrative + grounding                                               */
/* ------------------------------------------------------------------ */

function NarrativePanel({ caseId, persona }: { caseId: string; persona: string }) {
  const [narrative, setNarrative] = useState<Narrative | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    api
      .narrative(caseId, persona)
      .then((value) => live && setNarrative(value))
      .catch((exc: ApiError) => live && setError(exc.message))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [caseId, persona]);

  return (
    <Panel title="Narrative">
      <div className="space-y-3 p-4">
        {loading && <p className="text-sm text-ink-muted">Writing…</p>}

        {error && (
          <div className="rounded border border-unknown/40 bg-unknown-soft p-3">
            <p className="text-sm text-unknown-ink">
              The narrator could not write this case.
            </p>
            <p className="mt-1 text-xs text-unknown-ink/85">{error}</p>
            <p className="mt-2 text-xs text-unknown-ink/85">
              The case is unaffected — everything to the left is the engine&rsquo;s,
              and none of it came from a model.
            </p>
          </div>
        )}

        {narrative && (
          <>
            <div className="flex items-center gap-2">
              <ProvenanceBadge producedBy="model" />
              <span className="font-mono text-[10px] text-ink-faint">
                {narrative.model}
                {narrative.from_fixture && " · replayed offline"}
              </span>
            </div>

            <div className="space-y-2">
              {narrative.claims.map((claim, index) => (
                <p key={index} className="text-sm leading-relaxed text-ink">
                  {claim.sentence}{" "}
                  <span className="whitespace-nowrap">
                    {claim.evidence_ids.slice(0, 3).map((id) => (
                      <EvidenceChip
                        key={id}
                        evidenceId={id}
                        tone="neutral"
                        className="ml-1 align-middle"
                        title={`Evidence behind this sentence: ${id}`}
                      >
                        ●
                      </EvidenceChip>
                    ))}
                  </span>
                </p>
              ))}
            </div>

            {/* The grounding chip. `14/14 claims linked · 0 stripped` is
                not decoration: a sentence that failed the check was
                removed before it reached this panel. */}
            <div
              className={cn(
                "rounded border px-3 py-2 font-mono text-[11px]",
                narrative.claims_stripped === 0
                  ? "border-verified/40 bg-verified-soft text-verified-ink"
                  : "border-unknown/40 bg-unknown-soft text-unknown-ink",
              )}
              title="Every sentence carries at least one evidence id; every numeric token in it appears in the frozen adjudication object; no causal connective survives on a hypothesis that failed a hard gate."
            >
              {narrative.grounding}
            </div>
          </>
        )}
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ */
/* Confidence                                                          */
/* ------------------------------------------------------------------ */

function ConfidencePanel({ confidence }: { confidence: ConfidenceBreakdown }) {
  const capped = confidence.after_caps !== confidence.raw;
  const shifted = confidence.calibrated !== confidence.after_caps;

  return (
    <Panel
      title="Confidence"
      subtitle="Six components, weighted. Caps cannot be outvoted."
    >
      <div className="space-y-3 p-4">
        <ul className="space-y-2">
          {confidence.components.map((component) => (
            <li key={component.key}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-mono text-[11px] text-ink">
                  <span className="text-ink-faint">{component.key}</span>{" "}
                  {component.name}
                </span>
                <span className="font-mono text-[11px] tabular-nums text-ink">
                  {component.value.toFixed(2)}
                  <span className="ml-1 text-ink-faint">
                    ×{component.weight.toFixed(2)}
                  </span>
                </span>
              </div>
              <Meter
                value={component.value * 100}
                tone={component.value >= 0.7 ? "verified" : "unknown"}
                className="mt-1"
              />
              <p
                className="mt-0.5 line-clamp-1 text-[11px] text-ink-faint"
                title={component.detail}
              >
                {component.detail}
              </p>
            </li>
          ))}
        </ul>

        <div className="space-y-1.5 border-t border-paper-edge pt-3">
          <Row label="Weighted sum (raw)" value={confidence.raw.toFixed(4)} />
          {capped ? (
            <Row label="After caps" value={confidence.after_caps.toFixed(4)} />
          ) : (
            <p className="font-mono text-[11px] text-ink-faint">
              No cap fired.
            </p>
          )}
          <Row
            label="Published"
            value={confidence.calibrated.toFixed(2)}
            strong
          />
        </div>

        {shifted && (
          <p className="rounded border border-accent/25 bg-accent-soft p-2.5 text-xs text-accent-ink">
            The model scored this{" "}
            <strong>{(confidence.after_caps * 100).toFixed(0)}%</strong>. Our
            record on cases like it says we run about{" "}
            {Math.abs((confidence.after_caps - confidence.calibrated) * 100).toFixed(
              0,
            )}{" "}
            points hot, so we publish{" "}
            <strong>{(confidence.calibrated * 100).toFixed(0)}%</strong>.
            {confidence.calibration_sample_size !== null && (
              <> Fitted on {confidence.calibration_sample_size} closed cases.</>
            )}
          </p>
        )}

        {confidence.caps_applied.length > 0 && (
          <ul className="space-y-1">
            {confidence.caps_applied.map((cap) => (
              <li key={cap.name}>
                <Chip tone="unknown" title={cap.condition}>
                  cap {cap.name} → {cap.ceiling.toFixed(2)}
                  {cap.forced_trigger && ` · forces ${cap.forced_trigger}`}
                </Chip>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Panel>
  );
}

function Row({
  label,
  value,
  strong,
}: {
  label: string;
  value: string;
  strong?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span
        className={cn(
          "font-mono text-[11px]",
          strong ? "font-semibold text-ink" : "text-ink-muted",
        )}
      >
        {label}
      </span>
      <span
        className={cn(
          "font-mono tabular-nums",
          strong ? "text-base font-semibold text-ink" : "text-[11px] text-ink",
        )}
      >
        {value}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Provenance legend                                                   */
/* ------------------------------------------------------------------ */

function ProvenanceLegend() {
  return (
    <Panel title="Reading the page">
      <ul className="space-y-2 p-4 text-xs text-ink-muted">
        <li className="flex items-center gap-2">
          <ProvenanceBadge producedBy="code" />
          <span>Computed by Python or SQL. The model never makes a number.</span>
        </li>
        <li className="flex items-center gap-2">
          <ProvenanceBadge producedBy="model" />
          <span>Written by the model from a frozen object it cannot alter.</span>
        </li>
        <li className="flex items-center gap-2">
          <span className="h-3 w-3 shrink-0 rounded-sm bg-expected/60" />
          <span>Calendar-expected. Not a business problem.</span>
        </li>
        <li className="flex items-center gap-2">
          <span className="h-3 w-3 shrink-0 rounded-sm bg-real" />
          <span>The real residual, and anything requiring action.</span>
        </li>
        <li className="flex items-center gap-2">
          <span className="h-3 w-3 shrink-0 rounded-sm bg-verified" />
          <span>Verified, structured evidence.</span>
        </li>
        <li className="flex items-center gap-2">
          <span className="h-3 w-3 shrink-0 rounded-sm bg-unknown" />
          <span>Live, and nobody can check it.</span>
        </li>
        <li className="border-t border-paper-edge pt-2">
          Every dotted number opens the record behind it — source, as-of
          timestamp, freshness, method, the SQL, the lineage and the
          reliability weight.
        </li>
      </ul>
    </Panel>
  );
}
