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

The full demo runs offline from recorded fixtures:

```bash
MOCK_LLM=true make demo
```

`make demo` is a placeholder until the demo path lands at P8.

## Status

P7 complete — typed contracts, semantic layer, synthetic data generator, the
DuckDB warehouse with its governed query path and reconciliation report,
VALIDATE (Gate 1), QUALIFY (Gates 2–5, plus the restraint mechanics), and the
evidence engine every stage now mints through.
`tests/test_number_registry.py` is green.
