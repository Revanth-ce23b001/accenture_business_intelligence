import { createHmac } from "node:crypto";

/**
 * The signed persona header, minted in TypeScript against the same secret
 * the Python side verifies with.
 *
 * WHY MINT IT HERE RATHER THAN ASK THE API FOR ONE. There is no endpoint
 * that hands out tokens, and there should not be: an unauthenticated
 * "give me a token for cxo" is the whole auth scheme defeated by one URL.
 * The prototype's answer is a shared secret between two server processes,
 * which is exactly what `api/auth.py` documents itself as a stand-in for —
 * in a deployment both sides validate an OIDC token from the customer's
 * IdP and this file is deleted.
 *
 * THE BROWSER NEVER SEES A TOKEN. This runs server-side only, inside the
 * proxy route. A token in client JavaScript is a token in the devtools.
 */

const VERSION = "v1";
const TTL_HOURS = 12;

/** The demo directory. Every id here exists in `dim_user`, and the API
 *  refuses a token whose persona the directory disagrees with — so this
 *  map cannot grant an authority the warehouse has not granted. */
export const PERSONA_USERS: Record<string, string> = {
  cxo: "U001",
  cco: "U002",
  regional_manager: "U003",
  analyst: "U008",
  store_manager: "U007",
};

export const DEFAULT_PERSONA = "regional_manager";

function b64url(input: Buffer | string): string {
  return Buffer.from(input).toString("base64url");
}

export function signingSecret(): string {
  const secret = process.env.CASEFILE_SIGNING_SECRET?.trim();
  if (!secret) {
    throw new Error(
      "CASEFILE_SIGNING_SECRET is not set. Both the API and this app must " +
        "share it, or every request is refused. Set it in the environment " +
        "of both processes — see frontend/README.md.",
    );
  }
  return secret;
}

export function mint(persona: string, userId?: string): string {
  const resolved = userId ?? PERSONA_USERS[persona];
  if (!resolved) {
    throw new Error(
      `no directory entry for persona ${persona}; known: ${Object.keys(PERSONA_USERS).join(", ")}`,
    );
  }
  const issued = new Date();
  const expires = new Date(issued.getTime() + TTL_HOURS * 3600 * 1000);
  // Sorted keys and an ISO timestamp: the Python side hashes the exact
  // bytes, so the JSON must serialise the same way on both sides.
  const claims = {
    expires_at: expires.toISOString().replace("Z", "+00:00"),
    issued_at: issued.toISOString().replace("Z", "+00:00"),
    persona,
    user_id: resolved,
  };
  const payload = b64url(JSON.stringify(claims));
  const signature = b64url(
    createHmac("sha256", signingSecret()).update(payload, "ascii").digest(),
  );
  return `${VERSION}.${payload}.${signature}`;
}
