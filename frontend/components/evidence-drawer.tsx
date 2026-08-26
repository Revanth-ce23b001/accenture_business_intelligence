"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { api, ApiError } from "@/lib/api";
import { formatHours, formatTimestamp, formatValue, titleCase } from "@/lib/format";
import type { Evidence } from "@/lib/types";
import { Chip, Meter, ProvenanceBadge, cn } from "@/components/ui/primitives";

/**
 * RULE 3, MADE OPERATIONAL. "Every number in the UI must be clickable to
 * its evidence." Every figure on every screen renders through
 * `<EvidenceChip>`, which knows only an evidence id — the same id the
 * engine minted when it published the number — and opens this drawer.
 *
 * A chip with no evidence id is a compile-time impossibility: the prop is
 * required. A chip whose id the API cannot resolve shows the failure
 * rather than an empty drawer, because a number that cannot be chased is
 * exactly what this mechanism exists to surface.
 */

interface DrawerState {
  open: (evidenceId: string) => void;
  persona: string;
}

const Context = createContext<DrawerState | null>(null);

export function EvidenceProvider({
  persona,
  children,
}: {
  persona: string;
  children: ReactNode;
}) {
  const [evidenceId, setEvidenceId] = useState<string | null>(null);

  const open = useCallback((id: string) => setEvidenceId(id), []);

  return (
    <Context.Provider value={{ open, persona }}>
      {children}
      {evidenceId && (
        <Drawer
          evidenceId={evidenceId}
          persona={persona}
          onClose={() => setEvidenceId(null)}
        />
      )}
    </Context.Provider>
  );
}

function useDrawer() {
  const context = useContext(Context);
  if (!context) {
    throw new Error("EvidenceChip must be rendered inside an EvidenceProvider");
  }
  return context;
}

/* ------------------------------------------------------------------ */
/* The chip                                                            */
/* ------------------------------------------------------------------ */

export function EvidenceChip({
  evidenceId,
  children,
  tone = "neutral",
  className,
  title,
}: {
  /** Required. A number without one cannot be rendered — see rule 3. */
  evidenceId: string;
  children: ReactNode;
  tone?: "neutral" | "real" | "expected" | "verified" | "unknown" | "accent";
  className?: string;
  title?: string;
}) {
  const { open } = useDrawer();
  return (
    <button
      type="button"
      onClick={() => open(evidenceId)}
      title={title ?? "Open the evidence behind this number"}
      className={cn(
        "cursor-pointer rounded transition hover:brightness-95 focus:outline-none focus:ring-2 focus:ring-accent/40",
        className,
      )}
    >
      <Chip tone={tone} className="underline decoration-dotted underline-offset-4">
        {children}
      </Chip>
    </button>
  );
}

/* ------------------------------------------------------------------ */
/* The drawer                                                          */
/* ------------------------------------------------------------------ */

function Drawer({
  evidenceId,
  persona,
  onClose,
}: {
  evidenceId: string;
  persona: string;
  onClose: () => void;
}) {
  const [record, setRecord] = useState<Evidence | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setRecord(null);
    setError(null);
    api
      .evidence(evidenceId, persona)
      .then((value) => live && setRecord(value))
      .catch((exc: ApiError) =>
        live ? setError(exc.message ?? String(exc)) : undefined,
      );
    return () => {
      live = false;
    };
  }, [evidenceId, persona]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) =>
      event.key === "Escape" ? onClose() : undefined;
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div
        className="absolute inset-0 bg-ink/20"
        onClick={onClose}
        aria-hidden
      />
      <aside
        role="dialog"
        aria-label="Evidence"
        className="relative flex h-full w-full max-w-xl flex-col overflow-y-auto border-l border-paper-edge bg-paper shadow-2xl"
      >
        <header className="sticky top-0 flex items-start justify-between gap-4 border-b border-paper-edge bg-paper px-5 py-4">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
              Evidence
            </p>
            <p className="mt-1 font-mono text-xs text-ink-muted">{evidenceId}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded border border-paper-edge px-2 py-1 text-xs text-ink-muted hover:bg-paper-sunk"
          >
            Close
          </button>
        </header>

        {error && (
          <div className="m-5 rounded border border-red-300 bg-red-50 p-4 text-sm text-red-800">
            <p className="font-semibold">This number could not be chased.</p>
            <p className="mt-1">{error}</p>
            <p className="mt-2 text-xs">
              An evidence id that does not resolve means a number was published
              without a record behind it. That is a defect, not a display
              problem.
            </p>
          </div>
        )}

        {!record && !error && (
          <p className="p-5 text-sm text-ink-muted">Loading the record…</p>
        )}

        {record && <Record record={record} />}
      </aside>
    </div>
  );
}

function Record({ record }: { record: Evidence }) {
  return (
    <div className="space-y-6 p-5">
      <div>
        <div className="flex items-center gap-2">
          <p className="text-2xl font-semibold tabular-nums text-ink">
            {formatValue(record.value, record.unit)}
          </p>
          <ProvenanceBadge producedBy={record.produced_by} />
        </div>
        <p className="mt-1 text-sm text-ink-muted">{record.label}</p>
      </div>

      <Facts record={record} />

      {/* Reliability. The weight that decides whether text alone can
          reach a verdict — CLAUDE.md §Reliability weights. */}
      <section>
        <Heading>Reliability</Heading>
        <div className="mt-2 flex items-center gap-3">
          <Meter
            value={record.reliability * 100}
            tone={record.reliability >= 0.6 ? "verified" : "unknown"}
            className="max-w-[200px]"
          />
          <span className="font-mono text-xs tabular-nums text-ink">
            {record.reliability.toFixed(2)}
          </span>
          <span className="text-xs text-ink-muted">
            {titleCase(record.kind)}
          </span>
        </div>
        {record.reliability < 0.6 && (
          <p className="mt-2 text-xs text-unknown-ink">
            Below the 0.6 floor. A hypothesis whose evidence is dominated by
            weights under this cannot be promoted to EXPLAINED — text alone
            never reaches a verdict.
          </p>
        )}
      </section>

      {record.assumptions.length > 0 && (
        <section>
          <Heading>Assumptions this value rests on</Heading>
          <ul className="mt-2 space-y-1">
            {record.assumptions.map((assumption) => (
              <li key={assumption} className="text-sm text-unknown-ink">
                • {assumption}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <Heading>Lineage</Heading>
        <ol className="mt-2 space-y-3">
          {record.lineage.map((step) => (
            <li
              key={step.step}
              className="rounded border border-paper-edge bg-paper-sunk p-3"
            >
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                  step {step.step}
                </span>
                <Chip tone="accent">{step.operation}</Chip>
              </div>
              <p className="mt-2 text-sm text-ink">{step.description}</p>
              {step.inputs.length > 0 && (
                <p className="mt-1 font-mono text-[11px] text-ink-muted">
                  from: {step.inputs.join(" · ")}
                </p>
              )}
              <p className="mt-1 font-mono text-[11px] text-ink-faint">
                {step.ref}
              </p>
              {step.statement && (
                <pre className="mt-2 overflow-x-auto whitespace-pre-wrap rounded bg-ink/95 p-3 font-mono text-[11px] leading-relaxed text-paper">
                  {step.statement}
                </pre>
              )}
            </li>
          ))}
        </ol>
      </section>

      {record.notes && (
        <section>
          <Heading>
            {record.kind.includes("unstructured") ||
            record.kind === "store_note" ||
            record.kind === "ticket_aggregate"
              ? "The note itself"
              : "Notes"}
          </Heading>
          <p className="mt-2 whitespace-pre-wrap rounded border border-paper-edge bg-paper-sunk p-3 text-sm leading-relaxed text-ink">
            {record.notes}
          </p>
        </section>
      )}
    </div>
  );
}

function Facts({ record }: { record: Evidence }) {
  const rows: Array<[string, string]> = [
    ["Source system", record.source_system],
    ["Method", record.method],
    ["Source ref", record.source_ref],
    ["As of (data)", formatTimestamp(record.source_as_of)],
    ["Retrieved at", formatTimestamp(record.retrieved_at)],
    ["Freshness", formatHours(record.freshness_hours)],
    ["Completeness", `${(record.completeness * 100).toFixed(0)}%`],
  ];
  return (
    <section>
      <Heading>Provenance</Heading>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5">
        {rows.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="font-mono text-[11px] uppercase tracking-wider text-ink-faint">
              {label}
            </dt>
            <dd className="font-mono text-xs text-ink">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-xs text-ink-muted">
        <strong>As of</strong> is how current the data is;{" "}
        <strong>retrieved at</strong> is when it was read. A figure computed
        today from a feed that stopped on Tuesday is as of Tuesday.
      </p>
    </section>
  );
}

function Heading({ children }: { children: ReactNode }) {
  return (
    <h3 className="font-mono text-[10px] font-semibold uppercase tracking-[0.16em] text-ink-faint">
      {children}
    </h3>
  );
}
