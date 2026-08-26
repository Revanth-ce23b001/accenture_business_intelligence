import type {
  CaseDetail,
  Evidence,
  Narrative,
  Watchlist,
} from "@/lib/types";

/**
 * The client. Every call goes to the local proxy, never to the Python API
 * directly, so the signed persona header is minted server-side and the
 * browser never holds a token.
 *
 * FAILURES COME BACK AS FAILURES. No `?? []`, no silent empty state. A
 * watchlist that renders as "nothing is wrong" because the API was
 * unreachable is the single worst thing this UI could do, given that
 * distinguishing "nothing is wrong" from "nothing was checked" is the
 * product.
 */

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly hint?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const base = () =>
  typeof window === "undefined"
    ? (process.env.CASEFILE_SELF ?? "http://127.0.0.1:3000")
    : "";

async function get<T>(path: string, persona: string): Promise<T> {
  const separator = path.includes("?") ? "&" : "?";
  const response = await fetch(
    `${base()}/api/casefile/${path}${separator}persona=${encodeURIComponent(persona)}`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    let detail = response.statusText;
    let hint: string | undefined;
    try {
      const body = await response.json();
      detail = body.detail ?? body.error ?? detail;
      hint = body.hint;
    } catch {
      /* a non-JSON error body is still an error */
    }
    throw new ApiError(detail, response.status, hint);
  }
  return (await response.json()) as T;
}

export const api = {
  watchlist: (persona: string, scan = false) =>
    get<Watchlist>(`watchlist?scan=${scan}`, persona),

  case: (caseId: string, persona: string) =>
    get<CaseDetail>(`cases/${encodeURIComponent(caseId)}`, persona),

  /**
   * The narrative for one persona.
   *
   * THIS IS THE PERSONA SWITCH. It re-renders the prose and nothing else:
   * no stage runs, no statistic is recomputed, the verdict does not move.
   * The API caches per case and persona, so switching back is free.
   */
  narrative: (caseId: string, persona: string) =>
    get<Narrative>(
      `cases/${encodeURIComponent(caseId)}/narrative?persona=${encodeURIComponent(persona)}`,
      persona,
    ),

  evidence: (evidenceId: string, persona: string) =>
    get<Evidence>(`evidence/${encodeURIComponent(evidenceId)}`, persona),
};
