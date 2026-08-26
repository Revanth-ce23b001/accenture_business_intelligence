"use client";

import Link from "next/link";
import type { WatchlistItem } from "@/lib/types";
import { Panel, cn } from "@/components/ui/primitives";

/**
 * NOT OPENED AS CASES — the panel that matters most.
 *
 * A watchlist of open cases is a queue: any dashboard has one. A list of
 * what the system LOOKED AT and declined to open, naming the gate that
 * stopped each and why, is a record of judgement — and it is the only way
 * a reader can tell "nothing is wrong this week" from "nothing was
 * checked this week".
 *
 * EVERY ROW NAMES ITS GATE. A row that said only "suppressed" would be a
 * status; the gate number and the outcome code are what make it an
 * argument somebody can disagree with.
 */

export function SuppressedPanel({
  items,
  period,
  scanned,
}: {
  items: WatchlistItem[];
  period: string;
  scanned: boolean;
}) {
  return (
    <Panel
      title="Not opened as cases"
      count={items.length}
      tone="alert"
      subtitle={
        <>
          Movements checked in {period} and declined, with the gate that
          stopped each.{" "}
          {!scanned && (
            <span className="text-ink-faint">
              Live scan off — showing recorded outcomes only.
            </span>
          )}
        </>
      }
    >
      {items.length === 0 ? (
        <p className="px-4 py-6 text-sm text-ink-muted">
          Nothing was declined in this window. That is different from nothing
          being wrong — turn the scan on to check the visible scopes.
        </p>
      ) : (
        <ul className="divide-y divide-paper-edge">
          {items.map((item, index) => (
            <Row key={`${item.kpi}-${item.scope}-${index}`} item={item} />
          ))}
        </ul>
      )}
    </Panel>
  );
}

function Row({ item }: { item: WatchlistItem }) {
  const subject =
    item.scope && item.scope !== item.kpi
      ? `${item.scope} ${item.kpi.replace(/_/g, " ")}`
      : item.kpi.replace(/_/g, " ");

  return (
    <li className="grid grid-cols-[minmax(0,15rem)_5.5rem_1fr] items-baseline gap-3 px-4 py-3 hover:bg-paper-sunk">
      <span className="truncate text-sm font-medium text-ink">{subject}</span>

      <span
        className={cn(
          "font-mono text-sm tabular-nums",
          item.headline?.startsWith("-") ? "text-real" : "text-ink-muted",
        )}
      >
        {item.headline ?? "n/a"}
      </span>

      <span className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-sm">
        <span className="font-mono text-xs font-semibold text-red-700">
          <span aria-hidden>⛔ </span>
          {item.stopped_by_gate ? `GATE ${item.stopped_by_gate}` : "SUPPRESSED"}
        </span>
        <span className="text-ink-faint">·</span>
        <span className="text-ink-muted">{item.detail}</span>
        {item.case_id && (
          <Link
            href={`/case/${item.case_id}`}
            className="font-mono text-xs text-accent hover:underline"
          >
            #{item.case_id}
          </Link>
        )}
      </span>
    </li>
  );
}

export function OpenCasesPanel({ items }: { items: WatchlistItem[] }) {
  return (
    <Panel
      title="Open cases"
      count={items.length}
      subtitle="Ranked by how many times its own materiality limit each residual is."
    >
      {items.length === 0 ? (
        <p className="px-4 py-6 text-sm text-ink-muted">
          No case is open. Every movement this window either passed inside its
          band or was stopped by a gate — the panel below says which.
        </p>
      ) : (
        <ul className="divide-y divide-paper-edge">
          {items.map((item) => (
            <li
              key={item.case_id}
              className="flex items-baseline justify-between gap-3 px-4 py-3 hover:bg-paper-sunk"
            >
              <Link
                href={`/case/${item.case_id}`}
                className="text-sm font-medium text-ink hover:text-accent"
              >
                {item.scope} {item.kpi.replace(/_/g, " ")} · {item.period}
              </Link>
              <span className="flex items-baseline gap-3 font-mono text-xs text-ink-muted">
                {item.materiality_multiple !== null && (
                  <span title="How many times its own materiality limit the residual is">
                    {item.materiality_multiple.toFixed(1)}× limit
                  </span>
                )}
                {item.verdict && <span>{item.verdict}</span>}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
