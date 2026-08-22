# CLAUDE.md — CaseFile.ai

> This file is read at the start of every Claude Code session. It is the source of truth for this repository. If any instruction in a task prompt conflicts with this file, **stop and ask** rather than choosing.

---

## What this is

**CaseFile.ai** — an AI business investigator built for the Accenture Innovation Challenge 2026, Round 2 (BusinessIntelligence.ai track), by team **Case Solvers**.

It opens a case on every material KPI movement, adjudicates competing explanations with statistics rather than assertion, and closes with one of three verdicts — including *"we don't know yet, and here is what would resolve it."*

**Positioning:** Dashboards describe. Copilots narrate. CaseFile investigates.

Round 1 sold this as a concept. **Round 2's only job is to prove the machinery is ordinary engineering.** Every technique here is off-the-shelf — STL, propensity matching, difference-in-differences, retrieval, isotonic regression. The innovation is composition and discipline, not new mathematics.

---

## The ten non-negotiable rules

1. **The LLM is never the source of a number.** Every quantity is computed by Python or SQL. The model receives a frozen JSON object *after* adjudication and may render it, never alter it.
2. **No threshold lives in Python.** All thresholds, materiality limits, access rules and priors come from `semantic_layer/*.yaml`. A `grep` for hardcoded thresholds in `engine/` must return nothing.
3. **Every number in the UI must be clickable to its evidence.** No bare floats leave the engine; modules emit `Evidence` objects.
4. **Nothing on screen is a typed-in constant.** Every value in §"Number Registry" is *produced by the generator and computed by the engine*, and asserted by `tests/test_number_registry.py`.
5. **Authorization happens before retrieval.** `engine/db.py::execute_governed()` is the only path to the warehouse. A static test asserts no other module opens a connection.
6. **Abstention is architectural, not a prompt instruction.** Triggers T1–T8 are deterministic booleans evaluated outside the model.
7. **Contribution is labelled WHERE, causation is labelled WHY.** They are separate modules, separate panels, never merged.
8. **`MOCK_LLM=true` must always work.** The full demo runs offline from recorded fixtures. Build this from day one, not at the end.
9. **Do not start the frontend until P9 is green.** A complete engine with a plain UI scores far better than the reverse.
10. **If you cannot reproduce a Number Registry value, stop and report the discrepancy.** Never adjust the target to match your implementation.

---

## Number Registry — canonical values

These are the corrected, internally consistent values. Seven inconsistencies from the Round 1 material have been resolved; see §"Resolved defects" at the end.

All targets live in `data/generator/config/*.yaml`. **This section and those files must never diverge.**

### Entity

*Meridian Footwear India* — 412 stores across four regions: North 118, South 96, East 58, **West 140**. Annual net revenue ≈ ₹3,400 Cr, of which West ≈ ₹1,000 Cr.

Timeline: 18 months of daily data ending **30 Nov 2025**. Current period **M = Nov 2025**.

### Case #2451 — West, net revenue, Nov 2025 (monthly) → PARTIALLY EXPLAINED

| Quantity | Value | Derivation |
|---|---|---|
| West net revenue, Oct 2025 (M−1) | **₹83.70 Cr** | generator base |
| West net revenue, Nov 2025 (M) | **₹76.92 Cr** | 83.70 × 0.919 |
| Headline movement | **−8.1%** | computed |
| Absolute decline | **₹6.78 Cr** | 83.70 − 76.92 |
| Calendar-attributed | **−3.2 pt** = **₹2.68 Cr** | 0.032 × 83.70 |
| **Qualified residual** | **−4.9 pt** = **₹4.10 Cr** | 0.049 × 83.70 |
| *Reconciliation check* | 2.68 + 4.10 = 6.78 ✅ | must hold to ±₹0.02 Cr |
| Regional residual strip | **N −0.4 · S +1.1 · E −0.6 · W −4.9** | same method per region |
| West 8-week empirical band | **±1.8 pt** | residual quantile band |
| Materiality threshold (net revenue) | **₹50 L** = ₹0.50 Cr | KPI contract |
| Stores | **140 West: 34 treated, 106 untreated** | of which 34 selected as matched controls |
| Top-20 SKU availability, treated | **94% → 71%** | inventory snapshots, 12:00 daily mean |
| Store notes flagging unavailability | **21 of 34 treated · 2 of 34 control** | classifier output, χ² p<0.001 |
| Support tickets "size not available" | **340 · +186%** | ticket aggregate |
| Affected SKU share of West volume | **61%** | computed |
| Dose–response | **r = 0.71** | OLS, coverage gap vs store decline |
| **Matched-control DiD** | **−4.3 pt, p < 0.01** | 34 treated vs 34 matched |
| Parallel-trends pre-test | **p = 0.41** (passes) | 8-week pre-period |
| H1 attribution | **3.87 pt = ₹3.24 Cr = 79%** | 0.79 × 4.9 pt |
| Unattributed residual | **1.03 pt = ₹0.86 Cr** | 4.10 − 3.24 |
| *Materiality check* | ₹0.86 Cr > ₹0.50 Cr ✅ | **this is what forces PARTIALLY EXPLAINED** |
| H3 price rise | ASP **+4.1%** on sub-cats = **8%** of volume → max **−0.4 pt** | eliminated on sufficiency |
| H4 marketing cut | **−22%, onset D+11** (after decline began) | eliminated on precedence |
| H5 complaints | **+31%, onset D+6** | eliminated on precedence + confounder |
| Confidence raw | **0.89** | weighted s1–s6, no caps fire |
| **Confidence calibrated** | **0.84** | isotonic map |
| **Verdict** | **PARTIALLY EXPLAINED** | reason: *H2 unverifiable, holds ₹0.86 Cr above ₹50 L materiality* |
| Action cost | **₹18 L** expedited logistics | playbook: unit cost × 34 stores |
| **Expected recovery** | **₹2.3–3.1 Cr / 8 weeks** | 3.24 × recovery_curve[p25=0.71, p75=0.96] |
| ROI on action | **13–17×** | 2.3/0.18 … 3.1/0.18 |
| Recovery confidence | **Medium (n=3)** | sample-size rule |
| Call-down | **9 managers, 2 hours** | playbook |
| Linked case | DC allocation logic — *the cause of the cause* | playbook chain |

### Case #2467 — East, conversion rate, Oct 2025 (monthly) → INSUFFICIENT EVIDENCE

| Quantity | Value |
|---|---|
| East net revenue, monthly | **₹34.5 Cr** (58 stores) |
| Conversion rate | **22.0% → 19.6%** = **−2.4 pt** headline |
| Calendar + mix attributed | **−1.2 pt** |
| **Qualified residual** | **−1.2 pt** = **₹1.88 Cr** revenue-equivalent |
| Materiality (conversion) | **0.5 pt** → passes ✅ |
| Marketing feed freshness | **74 hrs stale** ⚠️ |
| H2 competitor promotion | 3 news items, 14 review mentions; **no competitor pricing or footfall held** |
| H6 catchment footfall | 2 road closures in notes; footfall counters in **31 of 140** stores only |
| Calibration, competitor-attribution cases | **58% (n=12)** — below the 70% publication floor |
| **Triggers fired** | **T3 · T4 · T7** |
| **Verdict** | **INSUFFICIENT EVIDENCE** |
| Resolution ranked by value/₹ | call-down 9 managers ~2 hrs · competitor feed ₹4 L/yr, 2 wks · footfall counters ₹12 L, 6 wks |
| Explicitly not recommended | price response — **₹1.2 Cr margin, ~50% chance of being wrong** |

### Cases #2470–#2472 — supporting scenarios

| Case | Period / grain | Scope | Setup | Outcome |
|---|---|---|---|---|
| **#2470** | **12 Nov 2025, daily** | West, net revenue | **25 of 140 store feeds failed to load** → headline **−18.0%** | **Gate 1 kill** — `DATA_INCIDENT`, no case opened |
| **#2471** | Nov 2025, weekly | All-India, Q-comm fulfilment | launched 7 weeks ago; **n=7 weekly points, 26 required** | **Gate 3** — `INSUFFICIENT_HISTORY`, monitoring only |
| **#2472** | **Sep 2025, monthly** | **South**, net revenue | **flat (0.0%)** against calendar-expected **+12%** (Ganesh Chaturthi) | **CASE OPENED**, residual **−12 pt**, adjudication `in_progress` |

> **Periods and scopes are deliberately non-overlapping** so no two scenarios contradict each other's numbers. #2470 is a *daily* check inside #2451's month — a data incident caught before it becomes a case. Do not move a scenario to a different period or region.

### Calibration ledger

**213 closed cases seeded** (145 published + 68 abstained), spanning the trailing 12 months.

Only two things are asserted. **Everything else in the ledger is computed and printed from the seeded data — do not hardcode the band table.**

```
ASSERT isotonic_map(0.8876) == 0.84 ± 0.01      # the #2451 raw → published step
ASSERT expected_calibration_error <= 0.05
```

Approximate shape to aim for when tuning the seed (report actuals, do not force them): 90–100% band ≈ 93% accurate · 80–89% ≈ 83% · 70–79% ≈ 74% · 60–69% ≈ 66% · abstention rate ≈ 32%.

> The 0.89 → 0.84 step is the demo's proof of Accenture requirement 7: *"the model scored this 89%; our track record on similar cases says we run about five points hot, so we publish 84%."*

---

## Confidence specification

```
conf_raw = 0.28·s1 + 0.20·s2 + 0.16·s3 + 0.14·s4 + 0.10·s5 + 0.12·s6
```

| | Component | Definition | #2451 |
|---|---|---|---|
| s1 | Evidence strength | `sigmoid(abs(did_t_stat))`, penalised if parallel-trends pre-test fails | 0.93 |
| s2 | Test coverage | applicable tests passed ÷ applicable tests | 1.00 |
| s3 | Source agreement | concordance between structured and unstructured lanes | 0.84 |
| s4 | Data quality | `min` over required sources of (freshness × completeness) | 0.90 |
| s5 | Historical depth | `min(1, history_weeks/52)`, penalty for regime change | 0.72 |
| s6 | Residual coverage | share of qualified residual attributed | 0.79 |

Sum = **0.8876** → publish **0.84** after the isotonic map.

**Caps, applied after the weighted sum (a cap cannot be outvoted):**

| Condition | Effect |
|---|---|
| Test 6 confounder screen **fails** (a confounder of *this hypothesis* is unresolved) | `min(conf, 0.85)` |
| A source required by this hypothesis is missing | `min(conf, 0.45)` **and** force T3 |
| Temporal precedence failed | `conf = 0`, hypothesis eliminated |
| Evidence dominated by reliability weights < 0.6 | `min(conf, 0.65)` |

> ⚠️ **Critical clarification — do not get this wrong.** On #2451 **no cap fires.** H2 (competitor promotion) is a *competing hypothesis for the residual*, not a confounder of H1. Test 6 screened promo, weather, competitor openings and staffing across the treated and control groups and found no difference — it **passes**. H2 is handled by the verdict decision table, not by a confidence cap. If your implementation applies the 0.85 cap here and produces 0.85, it is wrong.

---

## Verdict decision table

```
IF any trigger T1–T8 fires:
    → INSUFFICIENT EVIDENCE   (name every trigger that fired)

ELIF coverage >= 0.70
     AND conf_final >= 0.70
     AND no live-unverifiable hypothesis holds residual > materiality:
    → EXPLAINED

ELIF coverage >= 0.30 AND conf_final >= 0.60:
    → PARTIALLY EXPLAINED
       reason = "coverage below 0.70"  OR  "live-unverifiable hypothesis above materiality"

ELSE:
    → INSUFFICIENT EVIDENCE
```

**#2451 traced:** coverage 0.79 ✅ · conf 0.84 ✅ · but H2 is live, unverifiable, and holds ₹0.86 Cr > ₹0.50 Cr materiality → **PARTIALLY EXPLAINED**, reason `live-unverifiable hypothesis above materiality`. The reason string renders on the verdict chip.

## Abstention triggers

| | Trigger |
|---|---|
| T1 | No hypothesis passes both hard gates |
| T2 | Best hypothesis explains < 30% of qualified residual |
| T3 | Leading hypothesis requires a source the organisation does not hold |
| T4 | Two or more hypotheses statistically indistinguishable on available data |
| T5 | Evidence internally contradictory across independent sources |
| T6 | Evidence dominated by low-reliability unstructured sources |
| T7 | Historical calibration for this case type below the 70% publication floor |
| T8 | A known-relevant source is stale or unavailable |

Each is a named boolean, surfaced by name in the UI when it fires.

---

## The six adjudication tests

| # | Test | Method | Type |
|---|---|---|---|
| 1 | Temporal precedence | PELT changepoint on cause and effect series; require `cause_onset < effect_onset` | **HARD GATE** |
| 2 | Effect-size sufficiency | affected volume share × observed elasticity → modelled max impact vs residual | **HARD GATE** |
| 3 | Dose–response | OLS of store-level effect on store-level cause magnitude | weight 0.15 |
| 4 | Specificity | present-vs-absent group means, Welch t-test | weight 0.15 |
| 5 | Matched-control DiD | covariate matching (format, catchment, sqft, staffing, 8-wk pre-trend) + parallel-trends pre-test | weight **0.45** |
| 6 | Confounder screen | test known confounders from the causal graph | **CAP** |

**The two hard gates are the product.** H4 (marketing cut, strongest correlation in the data) and H5 (complaints) are both eliminated by Test 1 alone. That comparison is the single best demo moment — an LLM asked "why did revenue fall?" on this data picks one of them.

---

## Reliability weights for evidence

structured warehouse query **0.95** · derived statistical estimate **0.85** · corroborated unstructured, n≥20 independent notes **0.80** · support ticket aggregate **0.75** · single store note **0.55** · external news item **0.45** · social/review mention **0.35**

**Floor rule:** a hypothesis whose evidence is dominated by weights < 0.6 cannot be promoted to EXPLAINED (trigger T6). Text alone never reaches a verdict; text plus a matched control does.

---

## Architecture

```
Data → VALIDATE → QUALIFY → GATHER → ADJUDICATE → VERDICT → human decides → outcome logged → recalibrate
```

The five stage names are locked from the Round 1 deck. They are the module names, the API event names, and the UI progress rail, in that order and with that spelling.

| Deterministic — code and statistics | LLM |
|---|---|
| KPI computation, all SQL | Intent classification |
| Gates 1–5 and all thresholds | Hypothesis generation (long tail only) |
| STL, quantile bands, changepoints | Document → typed event extraction |
| Contribution decomposition | Narrative generation, per persona |
| Matching, DiD, all six tests | Clarification-question wording |
| Confidence score and caps | Action phrasing (from playbook fields) |
| Abstention triggers | — |
| Calibration and isotonic fit | — |
| RBAC, row filtering, masking | — |
| Telemetry and cost accounting | — |

**Model routing:** Haiku 4.5 for batched classification and extraction (content-hash cached). Sonnet for hypothesis generation and narrative, 2–4 calls per case.

**Grounding validator** (`llm/grounding.py`) — narrative is emitted as `[{sentence, evidence_ids[]}]`, then deterministically checked: every sentence carries ≥1 valid evidence_id; every numeric token appears in the adjudication object; no causal connective ("because", "caused by", "due to") in a sentence whose hypothesis did not pass both hard gates. Failures are stripped, regenerated once, then dropped. The UI prints `grounding: 14/14 claims linked · 0 stripped`.

---

## Stack

| Layer | Choice |
|---|---|
| Analytical store | **DuckDB**, single file, seeded from CSV |
| Analytics | Python 3.12 · pandas · numpy · scipy · statsmodels (STL, OLS, DiD) · ruptures (changepoints) · scikit-learn (isotonic, matching) |
| Backend | **FastAPI** + Pydantic (the Pydantic models *are* the evidence/adjudication contracts) |
| Frontend | **Next.js 15 App Router** · TypeScript · Tailwind · shadcn/ui · Recharts |
| LLM | Anthropic API behind `llm/provider.py`; `MockProvider` replays `llm/fixtures/` |
| Deploy | `docker compose up` locally as primary |

**Explicitly not used:** Kafka, Airflow, a vector database, microservices, Kubernetes, a graph database, Prophet, any deep-learning forecaster. ~2,000 documents means BM25 plus in-memory embeddings is sufficient and honest.

**Palette:** Accenture purple `#A100FF` for stage chrome · orange `#FF6B00` for the *real* residual and anything requiring action · grey for calendar-expected · teal `#00B8A9` for verified/structured evidence · amber for unverifiable. Code-produced values carry a monospace badge; model-produced text carries a rounded badge.

---

## Repository layout

```
casefile-ai/
├── CLAUDE.md · README.md · Makefile · docker-compose.yml
├── semantic_layer/   kpis/*.yaml · causal_graph.yaml · playbooks/*.yaml · schema.py
├── data/             generator/{base_series.py, scenarios/, config/*.yaml} · raw/ · casefile.duckdb
├── engine/           db.py · validate/ · qualify/ · gather/ · adjudicate/ · contribution/
│                     confidence/ · abstain/ · recommend/ · evidence.py
├── llm/              provider.py · classify.py · hypothesise.py · narrate.py · grounding.py · fixtures/
├── security/         policy.py · audit.py
├── telemetry/        recorder.py · cost.py
├── api/ · frontend/ · tests/
└── docs/             ARCHITECTURE.md · NUMBER_REGISTRY.md · DEMO_SCRIPT.md · REQUIREMENT_MATRIX.md
```

`make seed` · `make demo` · `make test` · `make reset`

---

## Definition of done

A step is complete only when its acceptance criteria pass **and** `make test` is green. `tests/test_number_registry.py` is written at P3, before any engine code, and is the definition of done for the project as a whole.

Performance targets: **P95 latency < 9 s warm** · **cost per case < ₹6** · **0 grounding violations** · full demo runs with the network disabled.

---

## Resolved defects — changelog from the Round 1 material

Seven inconsistencies were found and corrected. All corrected values above are canonical; the Round 1 originals are recorded here so the change is traceable.

| # | Defect | Was | Now |
|---|---|---|---|
| 1 | Portfolio, residual pt and residual ₹ mutually inconsistent | ₹840 Cr annual + 4.9 pt + ₹4.1 Cr (implies ₹1,004 Cr) | West monthly **₹83.70 Cr**, portfolio restated **~₹1,000 Cr**; 4.9 pt and ₹4.10 Cr preserved |
| 2 | Verdict rule contradicted the worked case | EXPLAINED = "≥70% coverage"; case at 79% labelled PARTIALLY | Three-condition decision table; 79% → PARTIALLY **by rule**, reason string rendered |
| 3 | One case ID, two different verdicts | #2451 was both PARTIALLY EXPLAINED and INSUFFICIENT EVIDENCE | Split: **#2451** partially explained (West revenue) · **#2467** insufficient evidence (East conversion) |
| 4 | Digit collision on the deck | DiD −4.1 pt beside residual ₹4.1 Cr | DiD retuned to **−4.3 pt**, still p<0.01 |
| 5 | Recovery exceeded attributable loss | ₹2.6–3.4 Cr against ₹3.24 Cr attributable (105%) | **₹2.3–3.1 Cr** = 71–96% of attributable, computed from the recovery curve |
| 6 | "11 minutes" was asserted, not measured | illustrative figure | telemetry records **real wall-clock** elapsed time; case header displays it |
| 7 | Data-incident arithmetic impossible | "2 of 34 feeds failed" → −18% (2/140 ≈ 1.4%) | **25 of 140 feeds failed** → −18.0%, and scoped to a **daily** check on 12 Nov 2025 |

Also corrected during this pass: the seeded calibration ledger's band table is now **computed from seeded cases** rather than specified, because the Round 1 table (80–89% band at 87%) could not produce the 0.89 → 0.84 step it was supposed to justify. Only the isotonic step and the ECE bound are asserted.

Outstanding, not a code issue: the Round 1 team slide reads *"Civil Engineeriing"*. Fix before any Round 2 submission.
