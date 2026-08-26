# Frontend — Watchlist and Case File

Next.js 15 App Router, TypeScript, Tailwind, Recharts. Two screens.

## Running it

Both processes must share a signing secret. The API verifies the header
this app mints; without a shared secret every request is refused.

```bash
export CASEFILE_SIGNING_SECRET="pick-something"

# terminal 1 — the API, offline
MOCK_LLM=true CASEFILE_SIGNING_SECRET=$CASEFILE_SIGNING_SECRET make demo

# terminal 2 — this app
cd frontend
npm install
CASEFILE_SIGNING_SECRET=$CASEFILE_SIGNING_SECRET npm run dev
```

Then <http://localhost:3000>. `/case/2451` is the worked case.

| Variable | Default | What it does |
|---|---|---|
| `CASEFILE_SIGNING_SECRET` | *(required)* | Shared with the API. Signs the persona header. |
| `CASEFILE_API` | `http://127.0.0.1:8000` | Where the API is. |

## How it talks to the API

The browser never calls the Python API and never holds a token.
`app/api/casefile/[...path]` proxies every request server-side and mints
the signed `X-CaseFile-Persona` header there. Switching persona changes a
query parameter; the proxy mints a token for that persona; the API checks
it against `dim_user` and applies the KPI's row predicate. **This app
grants nothing** — it carries an assertion the warehouse is free to
refuse, and it frequently does.

## The palette means something

From `CLAUDE.md` §Stack. Using a colour outside its meaning would make all
of them mean nothing, and the colours are how a reader sees the argument
before reading a word.

| Colour | Meaning |
|---|---|
| purple `#A100FF` | stage chrome — the progress rail, and nothing else |
| orange `#FF6B00` | the **real** residual, and anything requiring action |
| grey | calendar-**expected**. Not a business problem |
| teal `#00B8A9` | verified, structured evidence |
| amber | live, and nobody can check it |

A monospace `computed` badge marks a value Python or SQL produced; a
rounded `narrated` badge marks model-written text. That is rule 1 made
visible: no number on any screen came from the model.

## Every number opens its evidence

Rule 3. Figures render through `<EvidenceChip evidenceId=…>`, where the
prop is **required** — a number with no evidence id will not compile. The
chip opens a drawer showing the source system, both timestamps, freshness,
completeness, the method, the full lineage including verbatim SQL, the
reliability weight, and for unstructured evidence the note itself.

`tests/test_frontend_contract.py` fetches every id every case carries and
fails if one does not resolve.

## Canonical against live

`/case/2451` opens a **canonical scenario**: assembled from
`data/generator/config/scenario_*.yaml`, so it carries the Number Registry
values and nothing is typed in (rule 4). The header labels it, and names
the config file.

**Re-investigate live** runs the real pipeline and animates the rail from
the SSE stream — six events, `VALIDATE → QUALIFY → GATHER → ADJUDICATE →
VERDICT → NARRATE`. It opens a **separate** case, because the live engine
currently reaches a different verdict on this movement (see
`docs/ARCHITECTURE.md` → known divergence). The UI shows both and says
which is which rather than blending them.

On a stored case the rail shows measured per-stage times and says it is
not animating. A progress bar that pretends to be live is the one thing a
progress bar must not do.

## Layout

```
app/
  page.tsx                     Watchlist
  case/[id]/page.tsx           Case File
  api/casefile/[...path]/      the server-side proxy
components/
  ui/primitives.tsx            surfaces, chips, badges, meters
  evidence-drawer.tsx          the chip and the drawer behind every number
  persona-switcher.tsx         who is reading
  watchlist/                   KPI cards, the suppressed panel
  case/                        the rail, the three tabs, the right rail
lib/
  api.ts  types.ts  format.ts  token.ts
```

`npm run typecheck` · `npm run build`
