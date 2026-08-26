"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { formatElapsed, formatValue, titleCase } from "@/lib/format";
import type { CaseDetail } from "@/lib/types";
import { EvidenceProvider } from "@/components/evidence-drawer";
import { IsItReal } from "@/components/case/is-it-real";
import { RightRail } from "@/components/case/right-rail";
import { StageRail, railFromLatency, useLiveRun } from "@/components/case/stage-rail";
import { WhatNext } from "@/components/case/what-next";
import { Why } from "@/components/case/why";
import { Chip, cn } from "@/components/ui/primitives";

/**
 * Screen 2 — the Case File.
 *
 * WHAT A READER SHOULD GET IN THIRTY SECONDS, in this order and by
 * construction:
 *
 *   the header      how big it was, how sure we are, and what we concluded
 *   Is it real?     part of the drop was expected; this part was not
 *   Why?            one cause tested and confirmed, three tested and rejected,
 *                   one that cannot be tested at all
 *   What next?      what it costs to act, and what we refuse to do while
 *                   we do not know
 *
 * The tabs are that sentence, in the order the argument is made.
 */

type Tab = "real" | "why" | "next";

const TABS: Array<{ key: Tab; label: string; hint: string }> = [
  { key: "real", label: "Is it real?", hint: "Expected against unexplained" },
  { key: "why", label: "Why?", hint: "Tested, confirmed, rejected" },
  { key: "next", label: "What next?", hint: "The verdict, and what it costs" },
];

export function CaseView({
  detail,
  persona,
}: {
  detail: CaseDetail;
  persona: string;
}) {
  const [tab, setTab] = useState<Tab>("real");
  const { adjudication, verdict } = detail;

  const runBody = useMemo(
    () => ({
      kpi: adjudication.kpi,
      scope: adjudication.scope,
      grain: adjudication.grain,
      period: adjudication.period,
      persona,
    }),
    [adjudication, persona],
  );
  const live = useLiveRun(runBody, persona);

  const rail = live.running || live.events.length > 0
    ? live.state
    : railFromLatency(detail.latency_ms, verdict !== null);

  return (
    <EvidenceProvider persona={persona}>
      <div className="space-y-4">
        <Header
          detail={detail}
          rail={rail}
          live={live.running || live.events.length > 0}
          onRerun={live.start}
          running={live.running}
          liveError={live.error}
          liveCaseId={live.caseId}
        />

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
          <div>
            <nav className="flex gap-1 border-b border-paper-edge" role="tablist">
              {TABS.map((entry) => (
                <button
                  key={entry.key}
                  role="tab"
                  aria-selected={tab === entry.key}
                  onClick={() => setTab(entry.key)}
                  className={cn(
                    "-mb-px border-b-2 px-4 py-2 text-left transition",
                    tab === entry.key
                      ? "border-accent text-ink"
                      : "border-transparent text-ink-muted hover:text-ink",
                  )}
                >
                  <span className="block text-sm font-medium">{entry.label}</span>
                  <span className="block text-[10px] text-ink-faint">
                    {entry.hint}
                  </span>
                </button>
              ))}
            </nav>

            <div className="pt-4">
              {tab === "real" && <IsItReal adjudication={adjudication} />}
              {tab === "why" && <Why adjudication={adjudication} />}
              {tab === "next" && <WhatNext detail={detail} />}
            </div>
          </div>

          <RightRail
            caseId={detail.case_id}
            confidence={adjudication.confidence}
            persona={persona}
            personas={detail.narrative_personas}
          />
        </div>
      </div>
    </EvidenceProvider>
  );
}

/* ------------------------------------------------------------------ */
/* Header                                                              */
/* ------------------------------------------------------------------ */

const VERDICT_TONE = {
  EXPLAINED: "verified",
  PARTIALLY_EXPLAINED: "real",
  INSUFFICIENT_EVIDENCE: "unknown",
} as const;

function Header({
  detail,
  rail,
  live,
  onRerun,
  running,
  liveError,
  liveCaseId,
}: {
  detail: CaseDetail;
  rail: ReturnType<typeof railFromLatency>;
  live: boolean;
  onRerun: () => void;
  running: boolean;
  liveError: string | null;
  liveCaseId: string | null;
}) {
  const { adjudication, verdict } = detail;
  // A canonical scenario was assembled, not investigated, so there is no
  // wall-clock to report. Zero would read as "instant", which is the
  // opposite of what an unmeasured value means — and this is the very
  // figure that replaced an asserted "11 minutes".
  const measured =
    adjudication.elapsed_ms ??
    Object.values(detail.latency_ms).reduce((sum, ms) => sum + ms, 0);
  const elapsed = measured > 0 ? measured : null;

  return (
    <header className="rounded-lg border border-paper-edge bg-paper p-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <Link href="/" className="text-xs text-ink-muted hover:text-accent">
              ← Watchlist
            </Link>
            <h1 className="text-lg font-semibold tracking-tight text-ink">
              Case #{detail.case_id}
            </h1>
            <Chip tone={detail.source === "canonical" ? "expected" : "accent"}>
              {detail.source === "canonical"
                ? "canonical scenario"
                : "live run"}
            </Chip>
          </div>
          <p className="mt-1 text-sm text-ink-muted">
            {adjudication.scope} · {titleCase(adjudication.kpi)} ·{" "}
            {adjudication.period} · {adjudication.grain}
          </p>
          {detail.config_ref && (
            <p className="mt-0.5 font-mono text-[10px] text-ink-faint">
              every figure derived from {detail.config_ref}
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <Metric label="Owner" value={titleCase(detail.run_as_persona)} />
          <Metric
            label="Elapsed to verdict"
            value={elapsed === null ? "not run here" : formatElapsed(elapsed)}
            title={
              elapsed === null
                ? "This scenario was assembled from the generator's config, not investigated in this process. Re-investigate to measure it."
                : "Measured wall-clock, not an estimate. This is what replaced an asserted '11 minutes'."
            }
          />
          <Metric
            label="Headline"
            value={formatValue(
              adjudication.headline_movement.value,
              adjudication.headline_movement.unit,
            )}
          />
          <button
            type="button"
            onClick={onRerun}
            disabled={running}
            className="rounded border border-accent/40 bg-accent-soft px-3 py-1.5 text-xs font-medium text-accent-ink hover:brightness-95 disabled:opacity-60"
          >
            {running ? "Investigating…" : "Re-investigate live"}
          </button>
        </div>
      </div>

      {verdict ? (
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <Chip tone={VERDICT_TONE[verdict.value]} className="px-3 py-1 text-sm">
            {verdict.value.replace(/_/g, " ")}
          </Chip>
          {verdict.reason_text && (
            <p className="max-w-3xl text-sm text-ink-muted">
              {verdict.reason_text}
            </p>
          )}
          {verdict.triggers_fired.length > 0 && (
            <span className="flex gap-1">
              {verdict.triggers_fired.map((trigger) => (
                <Chip key={trigger} tone="unknown">
                  {trigger}
                </Chip>
              ))}
            </span>
          )}
        </div>
      ) : (
        <p className="mt-4 text-sm text-ink-muted">
          No verdict — this movement was stopped before one could be reached.
        </p>
      )}

      <div className="mt-4 border-t border-paper-edge pt-3">
        <StageRail state={rail} live={live} />
        {liveError && (
          <p className="mt-2 text-xs text-red-700">{liveError}</p>
        )}
        {liveCaseId && liveCaseId !== detail.case_id && (
          <p className="mt-2 text-xs text-ink-muted">
            The live run opened{" "}
            <Link
              href={`/case/${liveCaseId}`}
              className="text-accent underline"
            >
              case {liveCaseId}
            </Link>
            . It is a separate case from this canonical scenario — compare
            them rather than assuming they agree.
          </p>
        )}
      </div>
    </header>
  );
}

function Metric({
  label,
  value,
  title,
}: {
  label: string;
  value: string;
  title?: string;
}) {
  return (
    <div title={title}>
      <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
        {label}
      </p>
      <p className="mt-0.5 font-mono text-sm tabular-nums text-ink">{value}</p>
    </div>
  );
}
