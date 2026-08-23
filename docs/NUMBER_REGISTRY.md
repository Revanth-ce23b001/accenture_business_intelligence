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

## Targets that are NOT in the registry

Two figures were introduced by the P4 warehouse brief and are not in `CLAUDE.md`'s Number Registry,
because that section predates them. They are targets in exactly the same sense — declared in the
generator config, solved for by the generator, and asserted against the loaded warehouse — so they
are listed here by location, not by value.

| Target | Declared in | Asserted by |
|---|---|---|
| The gap between the POS ledger's revenue definition and marketing's | `data/generator/config/entity.yaml` → `reconciliation.definition_conflict` | `tests/test_reconciliation.py` |
| The share of revenue quarantined by the unresolvable store keys | `data/generator/config/entity.yaml` → `reconciliation.entity_key_mismatch` | `tests/test_reconciliation.py` |

Both are also measured back into `data/raw/manifest.json` under `measured`, alongside every other
emergent figure, so a reader can see what the generator achieved rather than what it aimed at.

## Registry values the engine does not reproduce exactly

Rule 10 says: if you cannot reproduce a registry value, stop and report the discrepancy — never adjust
the target. Each gap below is asserted by a test named after it, in a `REGISTRY GAPS` section, so the
divergence fails loudly the moment the method or the generator moves. No target has been edited.

| Registry value | Engine measures | Pinned by |
|---|---|---|
| Peer residual strip: N −0.4 · S +1.1 · E −0.6 | within about a third of a point | `tests/test_qualify.py::test_the_refit_does_not_reproduce_the_strip_exactly` |
| West empirical band ±1.8 pt | wider; the residual is outside it either way | `tests/test_qualify.py::test_the_empirical_band_is_not_the_registrys_1_8_pt` |
| Matched-control DiD −4.3 pt | **−4.77 pt**, about one standard error away | `tests/test_adjudicate.py::test_the_did_does_not_reproduce_minus_4_3_exactly` |
| Parallel-trends pre-test p = 0.41 | **p ≈ 0.74**; both pass, which is the claim | `tests/test_adjudicate.py::test_the_parallel_trends_p_value_is_not_the_registrys_0_41` |
| H3's price elasticity, fitted from history | not fittable here; **declared**, and labelled as such | `tests/test_adjudicate.py::test_no_price_elasticity_can_be_fitted_from_this_warehouse` |

The DiD gap is the only substantive one, and it is entirely a question of which 34 of the 106 untreated
West stores become controls. The engine's optimally matched set grew −2.66% in November, the
generator's own control set grew −5.13%, and the whole untreated pool grew −3.32%: two draws either
side of the pool, one standard error apart on an estimate whose standard error is 0.45 pt.

### One gap closed at P9

`tests/test_number_registry.py` pins an open gap at P3: the registry gives a DiD of −4.3 pt **and** an
H1 attribution of 3.87 pt, and 3.87 / 4.3 = 0.90 implies a shrinkage factor `CLAUDE.md` never defines.

ADJUDICATE closes it without one. The attributed figure is the near end of the 95% interval on the
engine's own −4.77 pt estimate — 3.89 pt, which is the registry's 3.87 to two places. The undefined
shrinkage was a confidence interval. The rest of the chain then follows from it and is computed, not
entered:

| | Engine | Registry |
|---|---|---|
| H1 attribution | 3.89 pt | 3.87 pt |
| Coverage | 0.795 | 0.79 |
| Unattributed residual | 1.01 pt | 1.03 pt |
| Unattributed residual | ₹0.84 Cr | ₹0.86 Cr |

₹0.84 Cr clears the ₹0.50 Cr materiality limit, which is what forces **PARTIALLY EXPLAINED**.

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
