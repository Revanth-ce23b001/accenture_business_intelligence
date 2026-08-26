# CaseFile.ai

An AI business investigator, built for the Accenture Innovation Challenge 2026,
Round 2 (BusinessIntelligence.ai track) by team **Case Solvers**.

Dashboards describe. Copilots narrate. CaseFile investigates.

## Source of truth

[`CLAUDE.md`](./CLAUDE.md) is the source of truth for this repository — the ten
non-negotiable rules, the canonical Number Registry, the confidence and verdict
specifications, and the architecture. Read it before changing anything.

- [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — where to find each part of the design
- [`docs/NUMBER_REGISTRY.md`](./docs/NUMBER_REGISTRY.md) — where to find the canonical values

## Getting started

```bash
make install    # or: python -m pip install -e ".[dev]"
make generate   # build data/raw from the synthetic data generator
make seed       # build data/casefile.duckdb from data/raw
make test       # or: python -m pytest
```

`data/raw` and the warehouse are both generated and neither is committed, so a
fresh clone needs `make generate` once before `make seed`. Re-run `generate`
only when the generator or its config changes.

`make seed` takes a few minutes and is worth running before `make test`: the
suite builds the warehouse itself if it is missing, and every later run copies
the built file instead of re-parsing 4.6 million rows of CSV.

The full demo runs offline from recorded fixtures, with the network
disabled:

```bash
export CASEFILE_SIGNING_SECRET="pick-something"   # both processes share it
make demo                                          # the API, on :8000
cd frontend && npm install && npm run dev          # the UI, on :3000
```

Then <http://localhost:3000>, and `/case/2451` for the worked case. See
[`frontend/README.md`](./frontend/README.md).

Every endpoint needs a signed `X-CaseFile-Persona` header. Mint one:

```bash
python -c "from api.auth import mint; print(mint('U008','analyst'))"
```

```bash
curl -N -X POST localhost:8000/api/cases/run   -H "X-CaseFile-Persona: $TOKEN" -H 'Content-Type: application/json'   -d '{"kpi":"net_revenue","scope":"West","period":"2025-11"}'
```

With no `CASEFILE_SIGNING_SECRET` set, the signing key is generated per
process — tokens work until the server restarts, which is correct for a
prototype and makes it impossible to ship a default secret by accident.
Interactive docs are at `/docs`; `/api/health` needs no header.

## Status

P17 complete — typed contracts, semantic layer, synthetic data generator, the
DuckDB warehouse with its governed query path and reconciliation report,
VALIDATE (Gate 1), QUALIFY (Gates 2–5, plus the restraint mechanics), the
evidence engine every stage mints through, GATHER's three lanes, and
ADJUDICATE: the six tests, per-hypothesis exposure, optimal matched-control
matching, and the contribution decomposition that sits beside it and may not
import it, and the confidence, calibration and abstention machinery:
`conf_raw` over six components, four caps that cannot be outvoted, an
isotonic map fitted to 213 seeded closed cases, eight named abstention
triggers and the three-condition verdict table; RECOMMEND, which turns a
verdict into an action and an abstention into a priced investigation list;
the grounding validator that strips a sentence rather than let it carry an
unlinked number; the trust boundary and the audit log that records what left;
telemetry — one row per request carrying all five stage latencies, the
token counts, the rupee cost and the methods that ran; the API, with the
five stages finally composed into one run behind it; the feedback loop
that closes it; and the frontend — a watchlist and a case file.
`tests/test_number_registry.py` is green.

The watchlist's headline panel is the one most products would not build:
**what was checked and NOT opened as a case**, with the gate that stopped
each and why. The case file makes the argument in three tabs — is it
real, why, what next — over a persistent rail carrying the confidence
breakdown, the grounding chip and the persona switcher. Every number on
screen is a chip that opens its evidence: source, both timestamps,
freshness, method, the verbatim SQL, the lineage chain, the reliability
weight, and for unstructured evidence the note itself.

Feedback comes at three levels, because a reader can accept a verdict and
still reject one driver inside it. The verdict's four options move the
isotonic map, driver feedback moves the causal graph's priors per
hypothesis per KPI, and an accepted action opens a recovery realisation
that moves the playbook's curve at D+56 — not on the click, because a
loop that learned from intentions would learn the wrong thing. Every
update is bounded: one rejection moves a 0.18 prior to 0.171, and no
volume of clicks moves it past ±0.10 without a person editing the graph.

`POST /api/cases/run` streams six Server-Sent Events — VALIDATE, QUALIFY,
GATHER, ADJUDICATE, VERDICT, then NARRATE — and a run a gate stops emits
the stages that ran and no others. Authentication is a signed
`X-CaseFile-Persona` header: a prototype stand-in for enterprise SSO,
with the production-shaped policy layer already behind it, so the
signature proves who minted the token and the directory and the KPI's
`access_policy` decide everything else.

**Known divergence.** Composed end to end for the first time at P15, the
live engine returns EXPLAINED on #2451 where the Number Registry
specifies PARTIALLY EXPLAINED — ADJUDICATE eliminates the competitor
hypothesis on temporal precedence rather than leaving it live and
unverifiable. Test 2 also does not run on a live case, because GATHER
emits no affected volume share. Both are reported rather than tuned away
(CLAUDE.md rule 10); see `docs/ARCHITECTURE.md`.

The elapsed time on a case header is a measurement. `telemetry/recorder.py`
times from the moment the case was opened to the moment the verdict was
reached and writes it to `case_registry.elapsed_ms` — which is what replaced
the "11 minutes" the Round 1 material asserted (resolved defect 6). An
offline run's cost is estimated from text rather than billed, because a
replayed fixture buys no tokens; every such row carries `cost_estimated` so
it can never be quoted as a measurement.

On case #2451 the engine eliminates the marketing cut and the complaint spike
on temporal precedence, eliminates the price rise on effect-size sufficiency,
and leaves one hypothesis standing holding 79% of the qualified residual — with
₹0.84 Cr above the materiality limit that nobody can attribute, which is what
makes the verdict *partially* explained rather than explained. No confidence
cap fires on that case — the unverifiable competitor hypothesis is a competing
explanation for the residual, not a confounder of the leading one, and it is
the verdict table that acts on it. Nothing under `engine/adjudicate/`,
`engine/confidence/` or `engine/abstain/` imports `llm/`, and tests assert it.

The suite runs offline: `tests/conftest.py` sets `MOCK_LLM=true`, and every
model call is replayed from `llm/fixtures/`. Read
[`llm/fixtures/README.md`](./llm/fixtures/README.md) before quoting any
number about the classifier — the committed fixtures are synthesised, not
recorded, and they say so.
