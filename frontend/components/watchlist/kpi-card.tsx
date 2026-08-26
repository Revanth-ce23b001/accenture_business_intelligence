"use client";

import Link from "next/link";
import { Line, LineChart, ResponsiveContainer, YAxis } from "recharts";
import { formatCompact, formatPercentChange } from "@/lib/format";
import type { KpiCard as Card } from "@/lib/types";
import { Chip, cn } from "@/components/ui/primitives";

/**
 * One KPI, its recent history and what the watchlist has to say about it.
 *
 * TWO OF THE SIX HAVE NO SPARKLINE, AND THE CARD SAYS SO. The warehouse
 * holds transaction detail as a sample, so a transactions line — and the
 * conversion rate that divides by it — would be quietly wrong. A card that
 * drew nothing would read as a KPI that had not moved, which is the same
 * failure this product exists to prevent one level up.
 */

const STATUS: Record<
  Card["status"],
  { label: string; tone: "real" | "unknown" | "expected" | "verified" }
> = {
  open_case: { label: "case open", tone: "real" },
  suppressed: { label: "not opened", tone: "unknown" },
  monitoring_only: { label: "monitoring only", tone: "expected" },
  quiet: { label: "within band", tone: "verified" },
};

export function KpiCard({ card }: { card: Card }) {
  const status = STATUS[card.status];
  const falling = (card.change_pct ?? 0) < 0;
  const material = Math.abs(card.change_pct ?? 0) >= 5;

  return (
    <article className="flex flex-col rounded-lg border border-paper-edge bg-paper p-4">
      <header className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold text-ink">
            {card.display_name}
          </h3>
          <p className="mt-0.5 font-mono text-[10px] uppercase tracking-wider text-ink-faint">
            {card.grain} · {card.owner_role.replace(/_/g, " ")}
          </p>
        </div>
        <Chip tone={status.tone}>{status.label}</Chip>
      </header>

      <div className="mt-3 flex items-baseline gap-2">
        <span className="text-xl font-semibold tabular-nums text-ink">
          {card.series_available
            ? formatCompact(card.latest, card.series_unit)
            : "—"}
        </span>
        {card.series_available && card.change_pct !== null && (
          <span
            className={cn(
              "font-mono text-sm tabular-nums",
              material && falling
                ? "text-real"
                : falling
                  ? "text-ink-muted"
                  : "text-verified-ink",
            )}
          >
            {formatPercentChange(card.change_pct)}
          </span>
        )}
      </div>

      <div className="mt-2 h-12">
        {card.series_available && card.series.length > 1 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={card.series}>
              <YAxis hide domain={["dataMin", "dataMax"]} />
              <Line
                type="monotone"
                dataKey="value"
                stroke={material && falling ? "#FF6B00" : "#8A8F98"}
                strokeWidth={1.75}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center rounded border border-dashed border-paper-edge bg-paper-sunk px-2">
            <p className="line-clamp-2 text-[10px] leading-tight text-ink-muted">
              {card.series_reason ?? "No series."}
            </p>
          </div>
        )}
      </div>

      <footer className="mt-3 space-y-1 border-t border-paper-edge pt-2">
        {card.materiality_display && (
          <p className="font-mono text-[10px] text-ink-faint">
            materiality {card.materiality_display}
          </p>
        )}
        {card.status_detail && (
          <p className="line-clamp-2 text-[11px] leading-tight text-ink-muted">
            {card.status_detail}
          </p>
        )}
        {card.rows_filtered > 0 && (
          <p
            className="font-mono text-[10px] text-ink-faint"
            title="Rows your access policy withheld. The line is your scope, not the estate."
          >
            {card.rows_filtered.toLocaleString("en-IN")} rows withheld by policy
          </p>
        )}
        {card.case_id && (
          <Link
            href={`/case/${card.case_id}`}
            className="inline-block font-mono text-[11px] text-accent hover:underline"
          >
            open case {card.case_id} →
          </Link>
        )}
      </footer>
    </article>
  );
}
