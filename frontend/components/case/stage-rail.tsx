"use client";

import { useCallback, useRef, useState } from "react";
import { formatElapsed } from "@/lib/format";
import { STAGES, type StageMessage, type StageOrNarrate } from "@/lib/types";
import { cn } from "@/components/ui/primitives";

/**
 * The five stages as a live progress bar, driven by SSE.
 *
 * THE SPELLING AND THE ORDER ARE LOCKED. CLAUDE.md §Architecture calls
 * VALIDATE, QUALIFY, GATHER, ADJUDICATE, VERDICT "the module names, the
 * API event names and the UI progress rail, in that order and with that
 * spelling". NARRATE is a sixth segment set apart, because the narrative
 * is the model rendering a decision the engine already made.
 *
 * A RAIL THAT IS NOT RUNNING SAYS SO. On a stored case the segments show
 * the measured time each stage actually took — read off the case, not
 * replayed as an animation. Animating a recording would be a progress bar
 * that lies about being live, which is the one thing a progress bar must
 * not do.
 */

const RAIL: StageOrNarrate[] = [...STAGES, "NARRATE"];

export type RailState = Record<
  string,
  { status: "waiting" | "running" | "ok" | "stopped" | "failed"; ms?: number }
>;

export function StageRail({
  state,
  live,
}: {
  state: RailState;
  live: boolean;
}) {
  return (
    <ol className="flex flex-wrap items-stretch gap-1.5" aria-label="Pipeline">
      {RAIL.map((stage) => {
        const entry = state[stage] ?? { status: "waiting" as const };
        const narrate = stage === "NARRATE";
        return (
          <li key={stage} className="min-w-[7.5rem] flex-1">
            <div
              className={cn(
                "h-1.5 w-full rounded-full",
                entry.status === "running" && "rail-running",
                entry.status === "ok" && (narrate ? "bg-accent/40" : "bg-accent"),
                entry.status === "stopped" && "bg-red-500",
                entry.status === "failed" && "bg-unknown",
                entry.status === "waiting" && "bg-paper-edge",
              )}
            />
            <div className="mt-1.5 flex items-baseline justify-between gap-1">
              <span
                className={cn(
                  "font-mono text-[10px] font-semibold uppercase tracking-[0.12em]",
                  entry.status === "waiting" ? "text-ink-faint" : "text-ink",
                  narrate && "italic",
                )}
              >
                {stage}
              </span>
              <span className="font-mono text-[10px] tabular-nums text-ink-faint">
                {entry.status === "stopped"
                  ? "stopped"
                  : entry.ms !== undefined
                    ? formatElapsed(entry.ms)
                    : entry.status === "running"
                      ? "…"
                      : ""}
              </span>
            </div>
          </li>
        );
      })}
      {!live && (
        <li className="w-full pt-1">
          <p className="font-mono text-[10px] text-ink-faint">
            Measured times from the stored run — not animated. Re-investigate to
            watch it live.
          </p>
        </li>
      )}
    </ol>
  );
}

/** Build the rail from a case's recorded per-stage milliseconds. */
export function railFromLatency(
  latency: Record<string, number>,
  reachedVerdict: boolean,
): RailState {
  const state: RailState = {};
  for (const stage of RAIL) {
    const ms = latency[stage];
    if (ms === undefined || ms === 0) {
      state[stage] = { status: "waiting" };
    } else {
      state[stage] = { status: "ok", ms };
    }
  }
  if (!reachedVerdict) {
    // A movement a gate stopped never reached the later stages, and the
    // rail must not imply that it did.
    const lastRun = RAIL.filter((s) => state[s]?.status === "ok").pop();
    if (lastRun) state[lastRun] = { ...state[lastRun], status: "stopped" };
  }
  return state;
}

/* ------------------------------------------------------------------ */
/* Live runs                                                           */
/* ------------------------------------------------------------------ */

export interface LiveRun {
  state: RailState;
  running: boolean;
  events: StageMessage[];
  error: string | null;
  caseId: string | null;
  start: () => void;
}

/**
 * Drive the rail from `POST /api/cases/run`.
 *
 * Reads the SSE body with a streaming fetch rather than `EventSource`,
 * because the run is a POST with a JSON body and `EventSource` can only
 * GET. Frames are parsed as they arrive; nothing waits for the stream to
 * finish, which is the whole reason the endpoint streams.
 */
export function useLiveRun(
  body: Record<string, unknown>,
  persona: string,
): LiveRun {
  const [state, setState] = useState<RailState>({});
  const [events, setEvents] = useState<StageMessage[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [caseId, setCaseId] = useState<string | null>(null);
  const started = useRef(false);

  const start = useCallback(() => {
    if (started.current) return;
    started.current = true;
    setRunning(true);
    setError(null);
    setEvents([]);
    setState({ VALIDATE: { status: "running" } });

    void (async () => {
      try {
        const response = await fetch(
          `/api/casefile/cases/run?persona=${encodeURIComponent(persona)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          },
        );
        if (!response.ok || !response.body) {
          throw new Error(`the run could not be started (${response.status})`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let boundary = buffer.indexOf("\n\n");
          while (boundary !== -1) {
            handleFrame(buffer.slice(0, boundary));
            buffer = buffer.slice(boundary + 2);
            boundary = buffer.indexOf("\n\n");
          }
        }
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc));
      } finally {
        setRunning(false);
      }
    })();

    function handleFrame(raw: string) {
      const lines = raw.split("\n");
      const name = lines
        .find((line) => line.startsWith("event: "))
        ?.slice(7)
        .trim();
      const dataLine = lines.find((line) => line.startsWith("data: "));
      if (!name || !dataLine) return;

      let data: Record<string, unknown>;
      try {
        data = JSON.parse(dataLine.slice(6));
      } catch {
        return;
      }

      if (name === "error") {
        setError(String(data.detail ?? "the run failed"));
        return;
      }
      if (name === "done") {
        if (typeof data.case_id === "string") setCaseId(data.case_id);
        setState((prior) => {
          const next = { ...prior };
          for (const stage of RAIL) {
            if (next[stage]?.status === "running") next[stage] = { status: "waiting" };
          }
          return next;
        });
        return;
      }

      const message = data as unknown as StageMessage;
      setEvents((prior) => [...prior, message]);
      if (typeof message.case_id === "string") setCaseId(message.case_id);

      setState((prior) => {
        const next: RailState = {
          ...prior,
          [name]: {
            status:
              message.status === "stopped"
                ? "stopped"
                : message.status === "failed"
                  ? "failed"
                  : "ok",
            ms: message.duration_ms,
          },
        };
        // Light the next segment, so the rail shows work in flight rather
        // than a row of finished ticks with a gap after it.
        const index = RAIL.indexOf(name as StageOrNarrate);
        const following = RAIL[index + 1];
        if (following && message.status === "ok") {
          next[following] = { status: "running" };
        }
        return next;
      });
    }
  }, [body, persona]);

  return { state, running, events, error, caseId, start };
}
