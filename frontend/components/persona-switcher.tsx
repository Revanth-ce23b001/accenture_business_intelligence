"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useTransition } from "react";

/**
 * Who is reading.
 *
 * TWO EFFECTS, AND THEY ARE DIFFERENT. Switching persona changes the row
 * predicate every warehouse read runs under — a West manager's watchlist
 * is West's — and it changes who the narrative is written for. Neither
 * re-runs the pipeline: the case object is the same object, and the
 * narrative is a rendering of it. That is the point.
 */

export const PERSONAS = [
  { key: "cxo", label: "Chief Executive" },
  { key: "cco", label: "Chief Commercial" },
  { key: "regional_manager", label: "Regional Manager · West" },
  { key: "analyst", label: "Analyst" },
  { key: "store_manager", label: "Store Manager" },
] as const;

export function PersonaSwitcher({
  persona,
  label = "Reading as",
  options,
}: {
  persona: string;
  label?: string;
  /** Restrict the list. The case rail passes the personas `narrate.yaml`
   *  declares, so the switcher cannot offer a reader the narrator has no
   *  voice for. The watchlist passes nothing and gets all of them, because
   *  there the persona is about the row filter rather than the prose. */
  options?: readonly string[];
}) {
  const router = useRouter();
  const params = useSearchParams();
  const [pending, startTransition] = useTransition();

  function change(next: string) {
    const query = new URLSearchParams(params.toString());
    query.set("persona", next);
    startTransition(() => router.push(`?${query.toString()}`));
  }

  return (
    <label className="flex items-center gap-2">
      <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
        {label}
      </span>
      <select
        value={persona}
        onChange={(event) => change(event.target.value)}
        disabled={pending}
        className="rounded border border-paper-edge bg-paper px-2 py-1 text-xs text-ink focus:outline-none focus:ring-2 focus:ring-accent/40 disabled:opacity-60"
      >
        {PERSONAS.filter(
          (option) => !options || options.includes(option.key),
        ).map((option) => (
          <option key={option.key} value={option.key}>
            {option.label}
          </option>
        ))}
      </select>
      {pending && (
        <span className="font-mono text-[10px] text-ink-faint">…</span>
      )}
    </label>
  );
}
