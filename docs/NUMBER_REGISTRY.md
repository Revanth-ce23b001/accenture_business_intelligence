# Number Registry — pointer document

**This file deliberately contains no numbers.** The canonical values for CaseFile.ai live in the
**Number Registry — canonical values** section of [`/CLAUDE.md`](../CLAUDE.md), and in the generator
configuration under `data/generator/config/*.yaml`.

`CLAUDE.md` states that those two must never diverge. Adding a third copy here would create a third
thing to keep in sync, so this page only tells you where to look.

---

## Where to look

| If you need | Read this section of `CLAUDE.md` |
|---|---|
| The entity: store counts by region, revenue scale, data window, current period | **Number Registry** → *Entity* |
| The primary worked case — West / net revenue / monthly, ending in PARTIALLY EXPLAINED | **Number Registry** → *Case #2451* |
| The reconciliation identity the attributed and residual figures must satisfy | **Number Registry** → *Case #2451*, the *Reconciliation check* row |
| Why that case is partially explained rather than explained | **Number Registry** → *Case #2451*, the *Materiality check* row, plus **Verdict decision table** |
| The abstaining case — East / conversion, ending in INSUFFICIENT EVIDENCE | **Number Registry** → *Case #2467* |
| The three supporting scenarios and the gates that stop each one | **Number Registry** → *Cases #2470–#2472* |
| Why the scenarios use non-overlapping periods and scopes | **Number Registry** → the note under *Cases #2470–#2472* |
| The seeded calibration ledger, and the only two things asserted about it | **Number Registry** → *Calibration ledger* |
| The per-component confidence inputs for the worked case | **Confidence specification**, the `#2451` column |
| Which Round 1 figures changed, and what they were before | **Resolved defects — changelog from the Round 1 material** |

## Rules that govern this registry

Stated in full in **The ten non-negotiable rules** in `CLAUDE.md`:

- **Rule 1** — every quantity is computed by Python or SQL; the model receives a frozen JSON object
  after adjudication and may render it, never alter it.
- **Rule 2** — no threshold, materiality limit, access rule or prior lives in Python; they come from
  `semantic_layer/*.yaml`.
- **Rule 4** — nothing on screen is a typed-in constant. Every value in the registry is produced by
  the generator, computed by the engine, and asserted by `tests/test_number_registry.py`.
- **Rule 10** — if you cannot reproduce a registry value, **stop and report the discrepancy**. Never
  adjust the target to match your implementation.

## The test

`tests/test_number_registry.py` is written at **P3, before any engine code**, and is the definition of
done for the project as a whole. See **Definition of done** in `CLAUDE.md`.

## Related

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — pointer to the pipeline, stack and adjudication machinery
