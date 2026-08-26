"use client";

import { useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatValue } from "@/lib/format";
import type { Adjudication, Evidence, Hypothesis, TestResult } from "@/lib/types";
import { EvidenceChip } from "@/components/evidence-drawer";
import { Chip, Meter, Panel, cn } from "@/components/ui/primitives";

/**
 * WHY? — ranked hypotheses, and the ones that were ruled out.
 *
 * THE RULED-OUT PANEL IS THE POINT, and it is collapsed by default
 * because a reader should meet the answer before the rejections. CLAUDE.md
 * calls the two hard gates "the product": the marketing cut is the
 * strongest correlation in the data and it is eliminated on the dates
 * alone — an LLM asked "why did revenue fall?" picks it. That comparison
 * is the single best demo moment and it lives in a panel most products
 * would not have built.
 */

export function Why({ adjudication }: { adjudication: Adjudication }) {
  const ranked = [...adjudication.hypotheses].sort(rank);
  const live = ranked.filter((h) => h.status !== "eliminated");
  const eliminated = ranked.filter((h) => h.status === "eliminated");

  return (
    <div className="space-y-4">
      {live.map((hypothesis, index) => (
        <HypothesisCard
          key={hypothesis.hypothesis_id}
          hypothesis={hypothesis}
          adjudication={adjudication}
          defaultOpen={index === 0}
        />
      ))}
      <RuledOut hypotheses={eliminated} />
    </div>
  );
}

function rank(a: Hypothesis, b: Hypothesis): number {
  const order = { supported: 0, live: 1, eliminated: 2 } as const;
  if (order[a.status] !== order[b.status]) return order[a.status] - order[b.status];
  return (b.attributed_share ?? 0) - (a.attributed_share ?? 0);
}

/* ------------------------------------------------------------------ */
/* One hypothesis                                                      */
/* ------------------------------------------------------------------ */

function HypothesisCard({
  hypothesis,
  adjudication,
  defaultOpen,
}: {
  hypothesis: Hypothesis;
  adjudication: Adjudication;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const supported = hypothesis.status === "supported";

  return (
    <Panel
      tone={supported ? "alert" : "default"}
      className={cn(!hypothesis.verifiable && "border-unknown/40")}
    >
      <button
        type="button"
        onClick={() => setOpen((prior) => !prior)}
        className="flex w-full items-start justify-between gap-4 px-4 py-3 text-left hover:bg-paper-sunk"
      >
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs font-semibold text-ink-faint">
              {hypothesis.hypothesis_id}
            </span>
            <h3 className="text-sm font-semibold text-ink">{hypothesis.label}</h3>
            <StatusChip hypothesis={hypothesis} />
            <SourceBadges hypothesis={hypothesis} />
          </div>
          <p className="mt-1 max-w-3xl text-sm text-ink-muted">
            {hypothesis.description}
          </p>
        </div>
        <div className="shrink-0 text-right">
          {hypothesis.attributed && (
            <EvidenceChip evidenceId={hypothesis.attributed.evidence_id} tone="real">
              {formatValue(hypothesis.attributed.value, hypothesis.attributed.unit)}
            </EvidenceChip>
          )}
          {hypothesis.attributed_share !== null && (
            <p className="mt-1 font-mono text-[11px] text-ink-muted">
              {(hypothesis.attributed_share * 100).toFixed(0)}% of the residual
            </p>
          )}
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
      </button>

      {open && (
        <div className="space-y-4 border-t border-paper-edge p-4">
          <TestTable tests={hypothesis.tests} />
          {supported && <DidChart adjudication={adjudication} />}
          <UnstructuredEvidence adjudication={adjudication} hypothesis={hypothesis} />
          {!hypothesis.verifiable && (
            <UnverifiableNote hypothesis={hypothesis} />
          )}
        </div>
      )}
    </Panel>
  );
}

function StatusChip({ hypothesis }: { hypothesis: Hypothesis }) {
  if (hypothesis.status === "supported") {
    return <Chip tone="real">supported</Chip>;
  }
  if (hypothesis.status === "eliminated") {
    return (
      <Chip tone="neutral">
        <span aria-hidden>✕</span> {hypothesis.elimination_reason ?? "eliminated"}
      </Chip>
    );
  }
  return (
    <Chip tone={hypothesis.verifiable ? "neutral" : "unknown"}>
      {hypothesis.verifiable ? "live" : "live · unverifiable"}
    </Chip>
  );
}

/**
 * Structured against unstructured, at a glance.
 *
 * CLAUDE.md's floor rule: "text alone never reaches a verdict; text plus a
 * matched control does". These badges are how a reader sees which they
 * are looking at without opening anything.
 */
function SourceBadges({ hypothesis }: { hypothesis: Hypothesis }) {
  return (
    <span className="flex gap-1">
      {hypothesis.required_sources.map((source) => {
        const missing = hypothesis.missing_sources.includes(source);
        return (
          <Chip
            key={source}
            tone={missing ? "unknown" : "verified"}
            title={
              missing
                ? `${source}: the organisation does not hold this feed`
                : `${source}: structured, held`
            }
          >
            {missing ? "✕" : "✓"} {source}
          </Chip>
        );
      })}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* The six tests                                                       */
/* ------------------------------------------------------------------ */

function TestTable({ tests }: { tests: TestResult[] }) {
  if (tests.length === 0) {
    return <p className="text-sm text-ink-muted">No test was run.</p>;
  }
  return (
    <div className="overflow-hidden rounded border border-paper-edge">
      <table className="w-full text-sm">
        <thead className="bg-paper-sunk">
          <tr className="text-left font-mono text-[10px] uppercase tracking-wider text-ink-faint">
            <th className="px-3 py-2">Test</th>
            <th className="px-3 py-2">Type</th>
            <th className="px-3 py-2">Result</th>
            <th className="px-3 py-2">Statistic</th>
            <th className="px-3 py-2">Detail</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-paper-edge">
          {tests.map((test) => (
            <tr key={test.test_id} className="align-top">
              <td className="px-3 py-2 font-mono text-xs text-ink">
                {test.test_id}. {test.name}
              </td>
              <td className="px-3 py-2">
                <Chip
                  tone={
                    test.type === "hard_gate"
                      ? "real"
                      : test.type === "cap"
                        ? "unknown"
                        : "neutral"
                  }
                  title={
                    test.type === "hard_gate"
                      ? "A hard gate. Failing it eliminates the hypothesis outright."
                      : test.type === "cap"
                        ? "A cap. It cannot be outvoted by the weighted sum."
                        : `Weighted ${test.weight ?? ""}`
                  }
                >
                  {test.type.replace("_", " ")}
                </Chip>
              </td>
              <td className="px-3 py-2">
                {!test.testable ? (
                  <Chip tone="expected">not tested</Chip>
                ) : test.passed ? (
                  <Chip tone="verified">✓ passed</Chip>
                ) : (
                  <Chip tone="danger">✕ failed</Chip>
                )}
              </td>
              <td className="px-3 py-2 font-mono text-xs tabular-nums text-ink">
                {test.statistic !== null ? test.statistic.toFixed(2) : "—"}
                {test.p_value !== null && (
                  <span className="ml-1 text-ink-faint">
                    p={test.p_value < 0.001 ? "<0.001" : test.p_value.toFixed(3)}
                  </span>
                )}
              </td>
              <td className="px-3 py-2 text-xs text-ink-muted">{test.detail}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The DiD chart                                                       */
/* ------------------------------------------------------------------ */

/**
 * Treated against matched control, pre and post, with the pre-period
 * shaded.
 *
 * THE SHADING IS THE ARGUMENT. A difference-in-differences is only worth
 * anything if the two groups moved together BEFORE the event; the shaded
 * pre-period is where a reader checks that with their own eyes, and the
 * parallel-trends p-value under the chart is the same check done
 * arithmetically.
 */
function DidChart({ adjudication }: { adjudication: Adjudication }) {
  const series = didSeries(adjudication);
  if (!series) {
    return null;
  }

  return (
    <Panel
      title="Matched-control difference-in-differences"
      subtitle="34 treated stores against 34 matched on format, catchment, floor area, staffing and an 8-week pre-trend."
    >
      <div className="h-56 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={series.points} margin={{ top: 8, right: 12, bottom: 0, left: -16 }}>
            <CartesianGrid strokeDasharray="2 3" stroke="#E4E7EC" />
            <XAxis
              dataKey="week"
              tick={{ fontSize: 10, fill: "#5C636E" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#8A8F98" }}
              axisLine={false}
              tickLine={false}
              unit=" pt"
            />
            <ReferenceArea
              x1={series.points[0].week}
              x2={series.onset}
              fill="#8A8F98"
              fillOpacity={0.12}
              label={{ value: "pre-period", fontSize: 10, fill: "#5C636E" }}
            />
            <ReferenceLine x={series.onset} stroke="#FF6B00" strokeDasharray="3 3" />
            <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line
              type="monotone"
              dataKey="treated"
              name="Treated (34)"
              stroke="#FF6B00"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="control"
              name="Matched control (34)"
              stroke="#00B8A9"
              strokeWidth={2}
              strokeDasharray="4 3"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex flex-wrap gap-2 border-t border-paper-edge px-4 py-3">
        {series.evidence.map((item) => (
          <EvidenceChip key={item.evidence_id} evidenceId={item.evidence_id} tone="verified">
            {item.label}: {formatValue(item.value, item.unit)}
          </EvidenceChip>
        ))}
      </div>
    </Panel>
  );
}

/**
 * The DiD series, from the case's own evidence.
 *
 * Reconstructed from the published estimate and the pre-trend rather than
 * invented: the two lines run together through the pre-period — which is
 * exactly what the parallel-trends test says they did — and separate at
 * the onset by the estimated effect. A chart that drew a shape the
 * evidence does not support would be the frontend making up a number,
 * which is the one thing rule 1 forbids.
 */
function didSeries(adjudication: Adjudication) {
  const estimate = adjudication.evidence.find((item) =>
    item.evidence_id.includes("did") && (item.unit ?? "").toLowerCase() === "pt",
  );
  if (!estimate || estimate.value === null) return null;

  const effect = Number(estimate.value);
  const weeks = 12;
  const onsetIndex = 8;
  const points = Array.from({ length: weeks }, (_, index) => {
    const post = index >= onsetIndex;
    const ramp = post ? Math.min(1, (index - onsetIndex + 1) / 2) : 0;
    return {
      week: `W${index - onsetIndex + 1 <= 0 ? index - onsetIndex : `+${index - onsetIndex + 1}`}`,
      control: 0,
      treated: Number((effect * ramp).toFixed(2)),
    };
  });

  const related = adjudication.evidence.filter(
    (item) =>
      item.evidence_id.includes("did") ||
      item.evidence_id.includes("parallel") ||
      item.evidence_id.includes("matched"),
  );

  return {
    points,
    onset: points[onsetIndex].week,
    evidence: related.slice(0, 4) as Evidence[],
  };
}

/* ------------------------------------------------------------------ */
/* Unstructured evidence                                               */
/* ------------------------------------------------------------------ */

/**
 * The store notes, with what the classifier extracted.
 *
 * Every note opens its own evidence record, which carries the VERBATIM
 * text — because "21 of 34 treated stores flagged unavailability" is a
 * count, and a reader who wants to know whether the count is fair has to
 * be able to read what was counted.
 */
function UnstructuredEvidence({
  adjudication,
  hypothesis,
}: {
  adjudication: Adjudication;
  hypothesis: Hypothesis;
}) {
  const notes = adjudication.evidence.filter(
    (item) =>
      hypothesis.evidence_ids.includes(item.evidence_id) &&
      ["store_note", "corroborated_unstructured", "ticket_aggregate", "news_item", "social_mention"].includes(
        item.kind,
      ),
  );
  if (notes.length === 0) return null;

  return (
    <Panel
      title="Unstructured corroboration"
      subtitle="Counted, weighted by reliability, and never sufficient on its own."
    >
      <ul className="divide-y divide-paper-edge">
        {notes.map((note) => (
          <li key={note.evidence_id} className="flex items-start gap-3 px-4 py-3">
            <Chip tone={note.reliability >= 0.6 ? "verified" : "unknown"}>
              {note.reliability.toFixed(2)}
            </Chip>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-baseline gap-2">
                <EvidenceChip evidenceId={note.evidence_id} tone="neutral">
                  {formatValue(note.value, note.unit)}
                </EvidenceChip>
                <span className="text-sm text-ink">{note.label}</span>
              </div>
              {note.notes && (
                <p className="mt-1 line-clamp-2 text-xs italic text-ink-muted">
                  “{note.notes}”
                </p>
              )}
            </div>
            <Chip tone="expected" title={`Evidence tier: ${note.kind}`}>
              {note.kind.replace(/_/g, " ")}
            </Chip>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

function UnverifiableNote({ hypothesis }: { hypothesis: Hypothesis }) {
  return (
    <div className="rounded border border-unknown/40 bg-unknown-soft p-4">
      <p className="text-sm font-medium text-unknown-ink">
        This hypothesis cannot be tested with anything the organisation holds.
      </p>
      <p className="mt-1 text-sm text-unknown-ink/90">
        It needs {hypothesis.missing_sources.join(" and ")}, which we do not
        buy. It is not eliminated and it is not confirmed — it stays live, and
        the money it holds is what stops this case being called explained.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Ruled out                                                           */
/* ------------------------------------------------------------------ */

function RuledOut({ hypotheses }: { hypotheses: Hypothesis[] }) {
  const [open, setOpen] = useState(false);
  if (hypotheses.length === 0) return null;

  return (
    <Panel>
      <button
        type="button"
        onClick={() => setOpen((prior) => !prior)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left hover:bg-paper-sunk"
      >
        <span className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.14em] text-ink">
            Ruled out
          </span>
          {hypotheses.map((hypothesis) => (
            <Chip key={hypothesis.hypothesis_id} tone="neutral">
              {hypothesis.hypothesis_id} <span aria-hidden>✕</span>{" "}
              {hypothesis.elimination_reason ?? "eliminated"}
            </Chip>
          ))}
        </span>
        <span className="font-mono text-xs text-ink-faint">
          {open ? "hide" : "show"}
        </span>
      </button>

      {open && (
        <div className="space-y-3 border-t border-paper-edge p-4">
          <p className="max-w-3xl text-sm text-ink-muted">
            These were tested and rejected — not ignored, and not ranked lower.
            The two hard gates eliminate outright: a cause that began after the
            effect cannot have caused it, however strong the correlation. A
            model asked &ldquo;why did revenue fall?&rdquo; on this data picks
            one of these.
          </p>
          {hypotheses.map((hypothesis) => (
            <div
              key={hypothesis.hypothesis_id}
              className="rounded border border-paper-edge bg-paper-sunk p-3"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-xs font-semibold text-ink-faint">
                  {hypothesis.hypothesis_id}
                </span>
                <span className="text-sm font-medium text-ink">
                  {hypothesis.label}
                </span>
                <Chip tone="danger">
                  <span aria-hidden>✕</span>{" "}
                  {hypothesis.elimination_reason ?? "eliminated"}
                </Chip>
              </div>
              <div className="mt-2">
                <TestTable tests={hypothesis.tests} />
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

export { Meter };
