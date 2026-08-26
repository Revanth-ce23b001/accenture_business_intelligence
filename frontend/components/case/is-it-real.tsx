"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatValue, share } from "@/lib/format";
import type { Adjudication, Evidence } from "@/lib/types";
import { EvidenceChip } from "@/components/evidence-drawer";
import { GateChip, Panel } from "@/components/ui/primitives";

/**
 * IS IT REAL? — the tab that answers the question before "why".
 *
 * THE DECOMPOSITION BAR IS THE WHOLE ARGUMENT IN ONE OBJECT. Grey is what
 * the calendar accounts for; orange is what it does not. A reader who sees
 * only this has already learned the thing most dashboards never tell them:
 * a −8.1% month was −3.2 points of festival timing and −4.9 points of
 * something real.
 *
 * The colours are load-bearing, not decorative (CLAUDE.md §Stack →
 * Palette): grey means expected, orange means real and requiring action.
 */

export function IsItReal({ adjudication }: { adjudication: Adjudication }) {
  const headline = adjudication.headline_movement;
  const calendar = adjudication.attributed[0] ?? null;
  const residual = adjudication.qualified_residual;
  const band = adjudication.empirical_band;

  return (
    <div className="space-y-4">
      <Decomposition
        headline={headline}
        calendar={calendar}
        residual={residual}
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <RegionStrip adjudication={adjudication} />
        <BandChart residual={residual} band={band} scope={adjudication.scope} />
      </div>

      <Panel
        title="Gates"
        subtitle="Every screen the movement passed before anybody started explaining it."
      >
        <div className="flex flex-wrap gap-2 p-4">
          {adjudication.gates.length === 0 ? (
            <p className="text-sm text-ink-muted">No gate result recorded.</p>
          ) : (
            adjudication.gates.map((gate) => (
              <GateChip
                key={gate.gate_id}
                gateId={gate.gate_id}
                name={gate.name}
                passed={gate.passed}
                detail={gate.detail}
              />
            ))
          )}
        </div>
        {adjudication.gates.some((gate) => !gate.passed) && (
          <p className="border-t border-paper-edge px-4 py-3 text-sm text-red-800">
            {adjudication.gates.find((gate) => !gate.passed)?.detail}
          </p>
        )}
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The 8.1 / 3.2 / 4.9 bar                                             */
/* ------------------------------------------------------------------ */

function Decomposition({
  headline,
  calendar,
  residual,
}: {
  headline: Evidence;
  calendar: Evidence | null;
  residual: Evidence;
}) {
  const total = Math.abs(Number(headline.value ?? 0));
  const expected = Math.abs(Number(calendar?.value ?? 0));
  const real = Math.abs(Number(residual.value ?? 0));

  return (
    <Panel
      title="Is it real?"
      subtitle="What the calendar accounts for, and what it does not."
    >
      <div className="space-y-4 p-4">
        <div className="flex flex-wrap items-baseline gap-3">
          <span className="text-sm text-ink-muted">Headline movement</span>
          <EvidenceChip evidenceId={headline.evidence_id} tone="neutral">
            {formatValue(headline.value, headline.unit)}
          </EvidenceChip>
        </div>

        <div className="flex h-9 w-full overflow-hidden rounded border border-paper-edge">
          {calendar && (
            <div
              className="flex items-center justify-center bg-expected/60"
              style={{ width: `${share(expected, total)}%` }}
              title="Calendar-expected — the festival timing shift"
            >
              <span className="font-mono text-[11px] font-semibold text-expected-ink">
                {formatValue(calendar.value, calendar.unit)}
              </span>
            </div>
          )}
          <div
            className="flex items-center justify-center bg-real"
            style={{ width: `${share(real, total)}%` }}
            title="The qualified residual — what the calendar does not explain"
          >
            <span className="font-mono text-[11px] font-semibold text-white">
              {formatValue(residual.value, residual.unit)}
            </span>
          </div>
        </div>

        <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
          {calendar && (
            <Legend
              swatch="bg-expected/60"
              label="Calendar-expected"
              evidence={calendar}
              note="Ganesh Chaturthi fell in a different month. Not a business problem."
            />
          )}
          <Legend
            swatch="bg-real"
            label="Qualified residual — the real part"
            evidence={residual}
            note="What no calendar effect accounts for. This is what gets investigated."
          />
        </div>
      </div>
    </Panel>
  );
}

function Legend({
  swatch,
  label,
  evidence,
  note,
}: {
  swatch: string;
  label: string;
  evidence: Evidence;
  note: string;
}) {
  return (
    <div className="flex max-w-sm items-start gap-2">
      <span className={`mt-1 h-3 w-3 shrink-0 rounded-sm ${swatch}`} />
      <div>
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-medium text-ink">{label}</span>
          <EvidenceChip
            evidenceId={evidence.evidence_id}
            tone={swatch.includes("real") ? "real" : "expected"}
          >
            {formatValue(evidence.value, evidence.unit)}
          </EvidenceChip>
        </div>
        <p className="mt-0.5 text-xs text-ink-muted">{note}</p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The four-region residual strip                                      */
/* ------------------------------------------------------------------ */

function RegionStrip({ adjudication }: { adjudication: Adjudication }) {
  const strip = adjudication.evidence
    .filter((item) => item.evidence_id.includes("region_residual"))
    .map((item) => ({
      region: item.label.split(" ")[0] ?? item.evidence_id,
      value: Number(item.value ?? 0),
      evidenceId: item.evidence_id,
    }));

  if (strip.length === 0) {
    return (
      <Panel title="Residual by region">
        <p className="p-4 text-sm text-ink-muted">
          No per-region strip was published for this case. Gate 4 compares the
          subject against its peers; a case that never reached it has none.
        </p>
      </Panel>
    );
  }

  return (
    <Panel
      title="Residual by region"
      subtitle="Same method, every region. This is what makes it specific rather than the market."
    >
      <div className="h-44 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={strip} margin={{ top: 4, right: 8, bottom: 0, left: -18 }}>
            <CartesianGrid strokeDasharray="2 3" stroke="#E4E7EC" vertical={false} />
            <XAxis
              dataKey="region"
              tick={{ fontSize: 11, fill: "#5C636E" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#8A8F98" }}
              axisLine={false}
              tickLine={false}
              unit=" pt"
            />
            <ReferenceLine y={0} stroke="#8A8F98" />
            <Tooltip
              contentStyle={{ fontSize: 12, borderRadius: 6 }}
              formatter={(value) => [`${Number(value).toFixed(1)} pt`, "residual"]}
            />
            <Bar dataKey="value" radius={[2, 2, 0, 0]} isAnimationActive={false}>
              {strip.map((entry) => (
                <Cell
                  key={entry.region}
                  fill={entry.region === adjudication.scope ? "#FF6B00" : "#8A8F98"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex flex-wrap gap-2 border-t border-paper-edge px-4 py-3">
        {strip.map((entry) => (
          <EvidenceChip
            key={entry.region}
            evidenceId={entry.evidenceId}
            tone={entry.region === adjudication.scope ? "real" : "neutral"}
          >
            {entry.region} {entry.value > 0 ? "+" : ""}
            {entry.value.toFixed(1)}
          </EvidenceChip>
        ))}
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ */
/* The empirical band                                                  */
/* ------------------------------------------------------------------ */

function BandChart({
  residual,
  band,
  scope,
}: {
  residual: Evidence;
  band: Evidence | null;
  scope: string;
}) {
  if (!band) {
    return (
      <Panel title="Against the empirical band">
        <p className="p-4 text-sm text-ink-muted">
          No band was fitted for this case.
        </p>
      </Panel>
    );
  }

  const width = Math.abs(Number(band.value ?? 0));
  const value = Number(residual.value ?? 0);
  const outside = Math.abs(value) > width;
  const data = [{ name: scope, residual: value }];

  return (
    <Panel
      title="Against the empirical band"
      subtitle="An empirical quantile of this scope's own residuals — not two standard deviations of a Gaussian nobody has seen."
    >
      <div className="h-44 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 8, right: 12, bottom: 0, left: 8 }}
          >
            <CartesianGrid strokeDasharray="2 3" stroke="#E4E7EC" horizontal={false} />
            <XAxis
              type="number"
              domain={[
                Math.min(-width * 1.6, value * 1.25),
                Math.max(width * 1.6, Math.abs(value) * 1.25),
              ]}
              tick={{ fontSize: 10, fill: "#8A8F98" }}
              unit=" pt"
            />
            <YAxis type="category" dataKey="name" hide />
            <ReferenceArea
              x1={-width}
              x2={width}
              fill="#8A8F98"
              fillOpacity={0.16}
              label={{ value: `±${width.toFixed(1)} pt band`, fontSize: 10, fill: "#5C636E" }}
            />
            <ReferenceLine x={0} stroke="#8A8F98" />
            <Tooltip
              contentStyle={{ fontSize: 12, borderRadius: 6 }}
              formatter={(value) => [`${Number(value).toFixed(1)} pt`, "residual"]}
            />
            <Bar
              dataKey="residual"
              barSize={26}
              radius={2}
              fill={outside ? "#FF6B00" : "#8A8F98"}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="border-t border-paper-edge px-4 py-3 text-sm">
        <EvidenceChip evidenceId={residual.evidence_id} tone={outside ? "real" : "neutral"}>
          {formatValue(residual.value, residual.unit)}
        </EvidenceChip>{" "}
        <span className="text-ink-muted">against a</span>{" "}
        <EvidenceChip evidenceId={band.evidence_id} tone="expected">
          {/* A band is a WIDTH, so it takes no sign of its own. */}
          ±{width.toFixed(1)} pt
        </EvidenceChip>{" "}
        <span className={outside ? "font-medium text-real-ink" : "text-ink-muted"}>
          {outside
            ? "— outside the band. This is not ordinary variation."
            : "— inside the band. Not a case."}
        </span>
      </p>
    </Panel>
  );
}
