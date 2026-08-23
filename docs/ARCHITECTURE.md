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

## Where the code puts things `CLAUDE.md` does not name

`CLAUDE.md` specifies the pipeline but not every file inside it. These are
decisions the implementation made, recorded here so they are findable — not
restated, just located.

| Decision | Where it is written down |
|---|---|
| The warehouse schema, and which tables exist | `engine/warehouse/schema.sql` |
| Which raw CSV loads into which table | `engine/warehouse/load.sql` |
| The four reconciliation problems, and how each is expressed in SQL | `engine/warehouse/views.sql` |
| What each team means by "revenue", and which definition arbitrates | `semantic_layer/warehouse.yaml` |
| Row predicates and masked columns, per persona, per KPI | `semantic_layer/kpis/*.yaml` → `access_policy` |
| The single door to the warehouse | `engine/db.py::execute_governed` |
| The single way to publish a number | `engine/evidence.py::EvidenceFactory` |
| What an evidence record may say about where it came from and how it was made | `semantic_layer/warehouse.yaml` → `evidence` |
| The floor rule, and what it blocks | `engine/evidence.py::can_promote_to_explained` |
| Where hypotheses come from, and the screen to five | `semantic_layer/gather.yaml` → `hypotheses` |
| One query template per hypothesis type | `semantic_layer/gather.yaml` → `structured.templates` |
| BM25 and the in-memory embeddings | `engine/gather/retrieval.py` |
| The classifier, and the line it does not cross | `llm/classify.py` |
| Why a model call did not happen | table `llm_cache` |
| What a committed fixture is and is not | `llm/fixtures/README.md` |
| What each of the six tests READS — every series, every SQL statement | `semantic_layer/adjudicate.yaml` |
| What each of the six tests is WORTH — weights, gates, caps | `semantic_layer/adjudication.yaml` → `tests` |
| Which stores a hypothesis's cause reached, and how that is derived rather than read | `semantic_layer/adjudicate.yaml` → `exposure` |
| What a hypothesis is allowed to claim from a DiD | `semantic_layer/adjudicate.yaml` → `did.attribution` |
| What happens when an elasticity cannot be fitted from history | `semantic_layer/adjudicate.yaml` → `sufficiency.elasticity.fallback` |
| How each confounder in the causal graph is measured | `semantic_layer/adjudicate.yaml` → `confounder_screen.measures` |
| Why a hypothesis was eliminated, in one machine-readable word | `engine/adjudicate/gate.py` → `HypothesisVerdict.elimination_reason` |
| The price/volume/mix split, and why it is LMDI | `engine/contribution/decomposition.py` |
| What each of s1..s6 measures, and the parameters each one needs | `semantic_layer/adjudication.yaml` → `confidence.components` |
| Why a cap is a cap and not a seventh component | `engine/confidence/caps.py` |
| Why the isotonic map is fitted over the whole ledger and never per case type | `engine/confidence/gate.py::_calibration_for` |
| What is declared about the seeded ledger, and what is measured off it | `data/generator/config/calibration.yaml` |
| Why the ledger's outcomes are laid out rather than drawn from the RNG | `data/generator/sources_calibration.py` |
| The eight abstention triggers, as named booleans | `engine/abstain/triggers.py` |
| Which hypothesis each trigger is evaluated against | `semantic_layer/adjudication.yaml` → `triggers.*.evaluated_against` |
| The verdict decision table, and the third condition | `engine/abstain/verdict.py::decide` |
| The organisation's own track record | table `calibration_ledger` |
| The narrow companion for pipeline metadata, and its allow-list | `engine/db.py::execute_metadata`, `semantic_layer/warehouse.yaml` → `governance.metadata_tables` |
| Gate 1's five checks and every threshold they use | `semantic_layer/validate.yaml` |
| Gates 2–5, and the restraint that limits how many cases open | `semantic_layer/qualify.yaml` |
| Which gate number belongs to which stage | `semantic_layer/adjudication.yaml` → `gates` |
| Festival membership per day, with overlaps intact | table `dim_festival_window` |
| Meridian's own scheduled promo windows | `dim_calendar.promo_window` |
| Where each KPI source system lands in the warehouse | `semantic_layer/warehouse.yaml` → `sources` |
| Which definition a KPI was computed under, run by run | table `kpi_definition_log` |
| Periods finance has reopened | table `restatement_register` |
| Every disagreement the warehouse found and did not fix | table `data_gap_register` |
| Every query that reached the warehouse | table `audit_log` |

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
- **no model call anywhere under `engine/adjudicate/`** — asserted statically by
  `tests/test_adjudicate.py::test_nothing_under_adjudicate_imports_the_model_layer`
- **`engine/contribution/` cannot import `engine/adjudicate/`** — asserted statically by
  `tests/test_contribution.py::test_contribution_cannot_import_adjudicate`
- **the engine reads no scenario flag.** Which stores an event reached is a conclusion of
  Tests 3, 4 and 5, derived from the cause series; asserted by
  `tests/test_adjudicate.py::test_the_engine_reads_no_scenario_flag`
- **no model call anywhere under `engine/confidence/` or `engine/abstain/`** — a confidence
  score is arithmetic and an abstention is a boolean; asserted by
  `tests/test_confidence.py::test_nothing_under_confidence_or_abstain_imports_the_model_layer`
- **no cap fires on #2451.** CLAUDE.md warns that reading H2's unverifiability as a confounder
  of H1 produces 0.85 and is wrong; asserted by
  `tests/test_confidence.py::test_no_cap_fires_on_2451`

## Related

- [`NUMBER_REGISTRY.md`](./NUMBER_REGISTRY.md) — pointer to the canonical values
- `DEMO_SCRIPT.md`, `REQUIREMENT_MATRIX.md` — listed in the repository layout, not yet written
