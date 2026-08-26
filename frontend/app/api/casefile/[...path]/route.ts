import { NextRequest } from "next/server";
import { DEFAULT_PERSONA, mint } from "@/lib/token";

/**
 * The one door between the browser and the Python API.
 *
 * Everything the UI fetches goes through here so that the signed persona
 * header is minted server-side and never reaches client JavaScript. The
 * persona comes from a query parameter the UI controls; the API still
 * checks it against `dim_user` and still applies the KPI's row predicate,
 * so this proxy grants nothing — it only carries an assertion the
 * warehouse is free to refuse.
 *
 * SSE PASSES THROUGH UNBUFFERED. `POST /api/cases/run` streams six stage
 * events over about thirteen seconds, and a proxy that collected them into
 * one response would turn a progress rail into a very slow spinner. The
 * upstream body is piped straight to the client with the caching headers
 * that stop intermediaries doing the same thing.
 */

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const API = process.env.CASEFILE_API ?? "http://127.0.0.1:8000";
const PERSONA_HEADER = "X-CaseFile-Persona";

/** Query parameters this proxy consumes rather than forwards. */
const PROXY_ONLY = new Set(["persona"]);

async function forward(request: NextRequest, path: string[]) {
  const incoming = new URL(request.url);
  const persona = incoming.searchParams.get("persona") ?? DEFAULT_PERSONA;

  const upstream = new URL(`${API}/api/${path.join("/")}`);
  incoming.searchParams.forEach((value, key) => {
    if (!PROXY_ONLY.has(key)) upstream.searchParams.append(key, value);
  });

  let token: string;
  try {
    token = mint(persona);
  } catch (error) {
    return Response.json(
      {
        error: "unconfigured",
        detail: error instanceof Error ? error.message : String(error),
        hint: "The API and this app must share CASEFILE_SIGNING_SECRET.",
      },
      { status: 500 },
    );
  }

  const headers = new Headers();
  headers.set(PERSONA_HEADER, token);
  headers.set("Accept", request.headers.get("accept") ?? "application/json");
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("Content-Type", contentType);

  let response: Response;
  try {
    response = await fetch(upstream, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.text(),
      // Streaming needs the body handed over rather than buffered.
      cache: "no-store",
    });
  } catch (error) {
    return Response.json(
      {
        error: "api_unreachable",
        detail: error instanceof Error ? error.message : String(error),
        hint: `Is the API running? Expected it at ${API}. Try: make demo`,
      },
      { status: 502 },
    );
  }

  const upstreamType = response.headers.get("content-type") ?? "";
  if (upstreamType.includes("text/event-stream")) {
    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  }

  return new Response(await response.text(), {
    status: response.status,
    headers: { "Content-Type": upstreamType || "application/json" },
  });
}

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, ctx: Ctx) {
  return forward(request, (await ctx.params).path);
}
export async function POST(request: NextRequest, ctx: Ctx) {
  return forward(request, (await ctx.params).path);
}
