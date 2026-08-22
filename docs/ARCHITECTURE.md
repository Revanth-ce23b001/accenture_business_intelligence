# Architecture — pointer document

**This file contains no architecture content.** The architecture of CaseFile.ai is defined in
[`/CLAUDE.md`](../CLAUDE.md) and nowhere else. This page exists only to tell you which section to read.

Do not copy specifications, tables, weights, stage names or thresholds into this file. A second copy
drifts from the first, and then neither is trustworthy. If you find yourself wanting to restate
something here, edit `CLAUDE.md` instead.

---

## Where to look

| If you need | Read this section of `CLAUDE.md` |
|---|---|
| The pipeline, its five locked stage names, and the full loop through recalibration | **Architecture** |
| What is deterministic code vs. what the LLM is allowed to do | **Architecture** → the two-column split table |
| Model routing (which model does classification/extraction vs. hypothesis/narrative) | **Architecture** → *Model routing* |
| How narrative is grounded to evidence and what happens on failure | **Architecture** → *Grounding validator* |
| Languages, libraries, store, backend, frontend, deploy | **Stack** |
| What is deliberately **not** in the stack, and why | **Stack** → *Explicitly not used* |
| Colour palette and the code-vs-model badge convention | **Stack** → *Palette* |
| Directory tree and the `make` targets | **Repository layout** |
| The six adjudication tests, which are hard gates, which carry weight, which caps | **The six adjudication tests** |
| The confidence formula, its six components, and the caps applied after the sum | **Confidence specification** |
| How a verdict is chosen between the three outcomes | **Verdict decision table** |
| The T1–T8 abstention triggers | **Abstention triggers** |
| Evidence source reliability weights and the floor rule | **Reliability weights for evidence** |
| What "done" means for a step and for the project | **Definition of done** |
| Performance, cost and offline-demo targets | **Definition of done** |

## Constraints that shape the code

These are stated as rules in **The ten non-negotiable rules** in `CLAUDE.md`. Read them there in full;
listed here only so you know they are architectural and not stylistic:

- the LLM never originates a number (rule 1)
- no threshold in Python — thresholds live in `semantic_layer/*.yaml` (rule 2)
- modules emit `Evidence` objects, never bare floats (rule 3)
- `engine/db.py::execute_governed()` is the single path to the warehouse (rule 5)
- abstention is evaluated outside the model as deterministic booleans (rule 6)
- contribution (WHERE) and causation (WHY) stay in separate modules and separate panels (rule 7)
- `MOCK_LLM=true` must work from day one (rule 8)
- no frontend work before P9 is green (rule 9)

## Related

- [`NUMBER_REGISTRY.md`](./NUMBER_REGISTRY.md) — pointer to the canonical values
- `DEMO_SCRIPT.md`, `REQUIREMENT_MATRIX.md` — listed in the repository layout, not yet written
