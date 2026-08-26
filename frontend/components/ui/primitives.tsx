"use client";

import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { ReactNode } from "react";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/* ------------------------------------------------------------------ */
/* Surfaces                                                            */
/* ------------------------------------------------------------------ */

export function Panel({
  title,
  count,
  subtitle,
  children,
  className,
  tone = "default",
}: {
  title?: string;
  count?: number | string;
  subtitle?: ReactNode;
  children: ReactNode;
  className?: string;
  tone?: "default" | "alert";
}) {
  return (
    <section
      className={cn(
        "rounded-lg border bg-paper",
        tone === "alert" ? "border-real/30" : "border-paper-edge",
        className,
      )}
    >
      {title && (
        <header className="flex items-baseline justify-between gap-3 border-b border-paper-edge px-4 py-3">
          <div>
            <h2 className="font-mono text-[11px] font-semibold uppercase tracking-[0.14em] text-ink">
              {title}
            </h2>
            {subtitle && (
              <p className="mt-1 text-xs text-ink-muted">{subtitle}</p>
            )}
          </div>
          {count !== undefined && (
            <span className="font-mono text-sm tabular-nums text-ink-muted">
              ({count})
            </span>
          )}
        </header>
      )}
      {children}
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Provenance badges — CLAUDE.md §Stack                                */
/* ------------------------------------------------------------------ */

/**
 * "Code-produced values carry a monospace badge; model-produced text
 * carries a rounded badge." The distinction is rule 1 made visible: a
 * reader can see, without being told, that no number on the page came
 * from the model.
 */
export function ProvenanceBadge({
  producedBy,
  className,
}: {
  producedBy: "code" | "model";
  className?: string;
}) {
  if (producedBy === "code") {
    return (
      <span
        title="Computed by Python or SQL. The model never produces a number (CLAUDE.md rule 1)."
        className={cn(
          "inline-flex items-center rounded-sm border border-verified/40 bg-verified-soft px-1.5 py-px font-mono text-[10px] font-semibold uppercase tracking-wider text-verified-ink",
          className,
        )}
      >
        computed
      </span>
    );
  }
  return (
    <span
      title="Written by the model from a frozen adjudication object it cannot alter."
      className={cn(
        "inline-flex items-center rounded-full border border-accent/30 bg-accent-soft px-2 py-px text-[10px] font-medium tracking-wide text-accent-ink",
        className,
      )}
    >
      narrated
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Chips                                                               */
/* ------------------------------------------------------------------ */

const TONES = {
  neutral: "border-paper-edge bg-paper-sunk text-ink",
  real: "border-real/35 bg-real-soft text-real-ink",
  expected: "border-expected/35 bg-expected-soft text-expected-ink",
  verified: "border-verified/35 bg-verified-soft text-verified-ink",
  unknown: "border-unknown/40 bg-unknown-soft text-unknown-ink",
  accent: "border-accent/30 bg-accent-soft text-accent-ink",
  danger: "border-red-300 bg-red-50 text-red-800",
} as const;

export type Tone = keyof typeof TONES;

export function Chip({
  children,
  tone = "neutral",
  className,
  title,
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1.5 rounded border px-2 py-0.5 font-mono text-xs tabular-nums",
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function GateChip({
  gateId,
  name,
  passed,
  detail,
}: {
  gateId: number;
  name: string;
  passed: boolean;
  detail: string;
}) {
  return (
    <span
      title={detail}
      className={cn(
        "inline-flex items-center gap-1.5 rounded border px-2 py-1 font-mono text-[11px]",
        passed
          ? "border-verified/40 bg-verified-soft text-verified-ink"
          : "border-red-300 bg-red-50 text-red-800",
      )}
    >
      <span aria-hidden>{passed ? "✓" : "⛔"}</span>
      <span className="font-semibold">GATE {gateId}</span>
      <span className="opacity-80">{name}</span>
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Bars                                                                */
/* ------------------------------------------------------------------ */

export function Meter({
  value,
  tone = "accent",
  className,
}: {
  value: number;
  tone?: Tone;
  className?: string;
}) {
  const fill: Record<string, string> = {
    accent: "bg-accent",
    real: "bg-real",
    expected: "bg-expected",
    verified: "bg-verified",
    unknown: "bg-unknown",
    neutral: "bg-ink-faint",
    danger: "bg-red-500",
  };
  return (
    <div
      className={cn(
        "h-1.5 w-full overflow-hidden rounded-full bg-paper-sunk",
        className,
      )}
    >
      <div
        className={cn("h-full rounded-full transition-all", fill[tone])}
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <p className="px-4 py-6 text-center text-sm text-ink-muted">{children}</p>
  );
}
