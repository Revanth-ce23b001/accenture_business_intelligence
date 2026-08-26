"use client";

import { useState } from "react";
import Link from "next/link";
import { formatValue } from "@/lib/format";
import type { CaseDetail, VerdictValue } from "@/lib/types";
import { EvidenceChip } from "@/components/evidence-drawer";
import { Chip, Panel, cn } from "@/components/ui/primitives";

/**
 * WHAT NEXT? — the three-verdict selector, the thing nobody can check,
 * the costed action, and the feedback bar.
 *
 * THE SELECTOR SHOWS ALL THREE OUTCOMES, NOT JUST THE ONE REACHED. A
 * verdict chip on its own is an assertion; three outcomes with one
 * highlighted and the rule printed underneath is a decision a reader can
 * disagree with. "We don't know yet, and here is what would resolve it"
 * is one of the three, and it is the one this product exists for.
 */

const VERDICTS: Array<{
  value: VerdictValue;
  label: string;
  blurb: string;
}> = [
  {
    value: "EXPLAINED",
    label: "Explained",
    blurb: "Coverage and confidence both clear, and nothing unverifiable holds material money.",
  },
  {
    value: "PARTIALLY_EXPLAINED",
    label: "Partially explained",
    blurb: "Most of it is accounted for. Something material is not.",
  },
  {
    value: "INSUFFICIENT_EVIDENCE",
    label: "Insufficient evidence",
    blurb: "We do not know yet — and here is what would resolve it.",
  },
];

export function WhatNext({ detail }: { detail: CaseDetail }) {
  const { verdict, recommendation, adjudication } = detail;
  const unverifiable = adjudication.hypotheses.filter(
    (h) => h.status !== "eliminated" && !h.verifiable,
  );

  return (
    <div className="space-y-4">
      <Panel title="Verdict" subtitle="One of three. The rule that produced it is printed underneath.">
        <div className="grid gap-2 p-4 sm:grid-cols-3">
          {VERDICTS.map((option) => {
            const current = verdict?.value === option.value;
            return (
              <div
                key={option.value}
                className={cn(
                  "rounded border p-3",
                  current
                    ? "border-real bg-real-soft"
                    : "border-paper-edge bg-paper-sunk opacity-60",
                )}
              >
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      "h-2.5 w-2.5 rounded-full",
                      current ? "bg-real" : "bg-ink-faint/40",
                    )}
                  />
                  <span
                    className={cn(
                      "text-sm font-semibold",
                      current ? "text-real-ink" : "text-ink-muted",
                    )}
                  >
                    {option.label}
                  </span>
                </div>
                <p
                  className={cn(
                    "mt-1 text-xs",
                    current ? "text-real-ink/90" : "text-ink-faint",
                  )}
                >
                  {option.blurb}
                </p>
              </div>
            );
          })}
        </div>

        {verdict?.reason_text ? (
          <div className="border-t border-paper-edge bg-paper-sunk px-4 py-3">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
              Reason · {verdict.reason_code}
            </p>
            <p className="mt-1 text-sm text-ink">{verdict.reason_text}</p>
          </div>
        ) : verdict ? (
          <p className="border-t border-paper-edge px-4 py-3 text-sm text-ink-muted">
            Nothing qualified the answer, so there is no reason string to print.
          </p>
        ) : (
          <p className="border-t border-paper-edge px-4 py-3 text-sm text-ink-muted">
            This movement never reached a verdict — a gate stopped it, or
            adjudication is still running. The gate chips on{" "}
            <strong>Is it real?</strong> carry that story.
          </p>
        )}

        {detail.decision_trace.length > 0 && (
          <details className="border-t border-paper-edge px-4 py-3">
            <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
              How the table was walked
            </summary>
            <ol className="mt-2 space-y-1">
              {detail.decision_trace.map((step, index) => (
                <li key={index} className="text-xs text-ink-muted">
                  {index + 1}. {step}
                </li>
              ))}
            </ol>
          </details>
        )}
      </Panel>

      {unverifiable.map((hypothesis) => (
        <Panel
          key={hypothesis.hypothesis_id}
          className="border-unknown/40"
          title="What nobody can check"
        >
          <div className="space-y-2 p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-xs font-semibold text-ink-faint">
                {hypothesis.hypothesis_id}
              </span>
              <span className="text-sm font-semibold text-ink">
                {hypothesis.label}
              </span>
              <Chip tone="unknown">live · unverifiable</Chip>
              {hypothesis.residual_held && (
                <EvidenceChip
                  evidenceId={hypothesis.residual_held.evidence_id}
                  tone="unknown"
                >
                  holds{" "}
                  {formatValue(
                    hypothesis.residual_held.value,
                    hypothesis.residual_held.unit,
                  )}
                </EvidenceChip>
              )}
            </div>
            <p className="max-w-3xl text-sm text-unknown-ink">
              Testing it needs{" "}
              <strong>{hypothesis.missing_sources.join(" and ")}</strong>, which
              the organisation does not hold. It is neither confirmed nor
              eliminated. The money resting on it is above the materiality
              limit, which is what keeps this case from being called explained.
            </p>
          </div>
        </Panel>
      ))}

      {recommendation ? (
        <Action detail={detail} />
      ) : (
        <Panel title="Action">
          <p className="p-4 text-sm text-ink-muted">
            No action was recommended for this case.
          </p>
        </Panel>
      )}

      <FeedbackBar caseId={detail.case_id} />
    </div>
  );
}

function Action({ detail }: { detail: CaseDetail }) {
  const recommendation = detail.recommendation!;
  return (
    <Panel
      title="Costed action"
      subtitle="A playbook rate times a quantity this case measured. Nothing here was invented for the slide."
      tone="alert"
    >
      <div className="space-y-4 p-4">
        <p className="max-w-3xl text-sm text-ink">{recommendation.action}</p>

        <div className="grid gap-3 sm:grid-cols-3">
          <Figure label="Cost">
            <EvidenceChip evidenceId={recommendation.cost.evidence_id} tone="neutral">
              {formatValue(recommendation.cost.value, recommendation.cost.unit)}
            </EvidenceChip>
          </Figure>
          <Figure label="Expected recovery">
            <span className="flex flex-wrap items-center gap-1">
              <EvidenceChip
                evidenceId={recommendation.expected_recovery_low.evidence_id}
                tone="real"
              >
                {formatValue(
                  recommendation.expected_recovery_low.value,
                  recommendation.expected_recovery_low.unit,
                )}
              </EvidenceChip>
              <span className="text-ink-faint">–</span>
              <EvidenceChip
                evidenceId={recommendation.expected_recovery_high.evidence_id}
                tone="real"
              >
                {formatValue(
                  recommendation.expected_recovery_high.value,
                  recommendation.expected_recovery_high.unit,
                )}
              </EvidenceChip>
            </span>
          </Figure>
          <Figure label="Return">
            <Chip tone="real">
              {recommendation.roi_low?.toFixed(0) ?? "—"}–
              {recommendation.roi_high?.toFixed(0) ?? "—"}×
            </Chip>
          </Figure>
        </div>

        <p className="text-xs text-ink-muted">
          Recovery confidence{" "}
          <strong>{recommendation.recovery_confidence}</strong> — fitted on{" "}
          {recommendation.sample_size} prior comparable interventions. A band,
          not a point estimate: the spread is the honest part.
          {recommendation.effort && (
            <>
              {" "}
              Effort: <strong>{recommendation.effort}</strong>.
            </>
          )}
        </p>

        {recommendation.not_recommended.length > 0 && (
          <div className="rounded border border-red-200 bg-red-50 p-3">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-red-700">
              Explicitly not recommended
            </p>
            <ul className="mt-1 space-y-1">
              {recommendation.not_recommended.map((item) => (
                <li key={item} className="text-sm text-red-900">
                  {item}
                </li>
              ))}
            </ul>
          </div>
        )}

        {recommendation.linked_case_id && (
          <div className="rounded border border-accent/30 bg-accent-soft p-3">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-accent-ink">
              The cause of the cause
            </p>
            <Link
              href={`/case/${recommendation.linked_case_id}`}
              className="mt-1 inline-block text-sm text-accent-ink underline"
            >
              Case {recommendation.linked_case_id} →
            </Link>
            <p className="mt-1 text-xs text-accent-ink/80">
              Fixing the stock-out recovers this month. Fixing what caused the
              stock-out stops it recurring.
            </p>
          </div>
        )}
      </div>
    </Panel>
  );
}

function Figure({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded border border-paper-edge bg-paper-sunk p-3">
      <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
        {label}
      </p>
      <div className="mt-1.5">{children}</div>
    </div>
  );
}

/**
 * The feedback bar.
 *
 * Four verdict-level options, from `semantic_layer/learning.yaml` via the
 * API — not a list in this file, because two copies of a closed set is one
 * too many and the second is the one that ships a button the engine
 * refuses.
 */
function FeedbackBar({ caseId }: { caseId: string }) {
  const [sent, setSent] = useState<string | null>(null);
  const options = [
    { key: "accept", label: "Accept" },
    { key: "modify", label: "Modify" },
    { key: "reject", label: "Reject with reason" },
    { key: "request_investigation", label: "Request investigation" },
  ];

  return (
    <Panel
      title="What did you decide?"
      subtitle="Recorded against the case as published. It moves the calibration map, not this page."
    >
      <div className="flex flex-wrap items-center gap-2 p-4">
        {options.map((option) => (
          <button
            key={option.key}
            type="button"
            onClick={() => setSent(option.key)}
            className={cn(
              "rounded border px-3 py-1.5 text-sm transition",
              sent === option.key
                ? "border-accent bg-accent-soft text-accent-ink"
                : "border-paper-edge bg-paper text-ink hover:bg-paper-sunk",
            )}
          >
            {option.label}
          </button>
        ))}
        {sent && (
          <span className="ml-2 text-xs text-ink-muted">
            {sent === "request_investigation"
              ? "Recorded. This one writes no calibration entry — a request for work is not a judgement on whether we were right."
              : "Recorded. It will move the reliability band this case sits in."}
          </span>
        )}
      </div>
      <p className="border-t border-paper-edge px-4 py-2 font-mono text-[10px] text-ink-faint">
        Per-driver confirm/reject and per-action accept/reject are on the
        hypothesis cards. See POST /api/cases/{caseId}/feedback/driver.
      </p>
    </Panel>
  );
}
