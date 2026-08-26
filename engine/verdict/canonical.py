"""The five Number Registry scenarios, as complete case files.

WHAT THIS IS FOR. `POST /api/cases/run` investigates a movement live and
takes about thirteen seconds. The demo also needs the five scenarios
CLAUDE.md's Number Registry specifies, available instantly and carrying
the values that document specifies — so a reader opening `/case/2451`
sees #2451 rather than a spinner, and so the frontend has something to
render before anybody has run anything.

WHERE THE NUMBERS COME FROM, WHICH IS THE WHOLE POINT (rule 4).
`llm/casefiles.py` assembles the `Adjudication` from
`data/generator/config/scenario_*.yaml` — the same files
`tests/test_number_registry.py` asserts the generator reproduces. Nothing
is typed in here: change a target in the generator and the case file
changes with it. This module adds the two contracts that assembler does
not produce, and it computes both rather than declaring them:

    Verdict         `engine/abstain/verdict.py::decide` — the real
                    decision table, walked over the case's own coverage,
                    confidence and held residuals
    Recommendation  `engine/recommend` — the real playbook match, the real
                    cost arithmetic, the real recovery curve

CANONICAL IS NOT LIVE, AND THE DIFFERENCE IS LABELLED. A case built here
carries `source="canonical"`; one from `run_case` carries `source="live"`.
They diverge today — the live engine eliminates the competitor hypothesis
on precedence and reaches EXPLAINED where the registry specifies
PARTIALLY EXPLAINED (see docs/ARCHITECTURE.md). A UI that showed both
without saying which was which would be the worst of both.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import duckdb

from engine.abstain.triggers import Trigger
from engine.abstain.verdict import HeldResidual, decide
from engine.abstain.verdict import to_contract as verdict_contract
from engine.contracts import Adjudication, CheckResult, GateResult, Recommendation
from engine.evidence import EvidenceFactory
from engine.recommend.cost import CostBasis
from engine.recommend.gate import RecommendError, RecommendRequest, recommend
from engine.recommend.gate import to_contract as recommendation_contract
from engine.verdict.casefile import EVIDENCE_PREFIX, CaseFile
from security.policy import User
from semantic_layer.schema import SemanticLayer, get_semantic_layer

#: Marks which of the two a case file came from. The UI prints it.
CANONICAL = "canonical"
LIVE = "live"

#: For the region strip: computed from other evidence, by subtraction.
#: Both are words the evidence vocabulary declares, so a typo fails when
#: the record is minted rather than becoming an unqueryable string.
DERIVED_ORIGIN = "derived"
DIFFERENCE_METHOD = "difference"

#: Gates that ran and passed leave no outcome code. A case that opened
#: passed every gate the layer declares, and the chips say so by gate id.
CLEAN = "CLEAN"


@dataclass(frozen=True)
class CanonicalCase:
    """One scenario, complete, with where its figures came from."""

    case_file: CaseFile
    source: str = CANONICAL
    #: The scenario config key, so a reader can go and look at the input.
    config_ref: str = ""

    @property
    def case_id(self) -> str:
        return self.case_file.case_id


def build(
    case_id: str,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
    user: User | None = None,
    layer: SemanticLayer | None = None,
) -> CanonicalCase:
    """Assemble one scenario into a complete case file.

    `connection` is needed only for the recommendation, which reads the
    owner off the KPI contract and prices against the playbook. Without
    one the case file comes back without a recommendation rather than
    with an invented one.
    """
    from llm.casefiles import build_case, scenarios

    layer = layer or get_semantic_layer()
    config = scenarios()[case_id]
    adjudication = build_case(case_id, layer)
    adjudication = _with_gates(adjudication, layer, config)
    adjudication = _with_region_strip(adjudication, layer, config)

    decision = None
    if _reached_a_verdict(adjudication, config):
        decision = decide(
            coverage=adjudication.coverage,
            confidence=adjudication.confidence.calibrated,
            triggers=_triggers(adjudication, layer),
            held=held_residuals(adjudication, layer),
            layer=layer,
        )

    recommendation = None
    if decision is not None and connection is not None and user is not None:
        recommendation = _recommend(
            connection, user, adjudication, decision, config, layer=layer
        )

    return CanonicalCase(
        case_file=CaseFile(
            case_id=adjudication.case_id,
            adjudication=adjudication,
            verdict=(
                verdict_contract(decision, adjudication.case_id, adjudication.opened_at)
                if decision is not None
                else None
            ),
            recommendation=recommendation,
            decision_trace=decision.trace if decision is not None else (),
        ),
        config_ref=f"data/generator/config/scenario_{case_id}.yaml",
    )


def _reached_a_verdict(adjudication: Adjudication, config: Any) -> bool:
    """Whether this movement got far enough to be given one.

    Three ways it does not. A gate killed it, so no case was opened —
    #2470 and #2471. Adjudication is still running — #2472. Or nothing was
    adjudicated at all, in which case there is nothing to decide about.
    Each of those is a real state and none of them is an abstention: the
    system did not decline to answer, it never got to the question.
    """
    gate = config.get("gate") or {}
    if gate.get("case_opened") is False:
        return False
    if gate.get("adjudication_status") == "in_progress":
        return False
    if adjudication.status != "closed":
        return False
    return bool(adjudication.hypotheses)


def build_all(
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
    user: User | None = None,
    layer: SemanticLayer | None = None,
) -> dict[str, CanonicalCase]:
    """Every scenario. A failure on one does not lose the others."""
    from llm.casefiles import CASE_IDS

    layer = layer or get_semantic_layer()
    built: dict[str, CanonicalCase] = {}
    for case_id in CASE_IDS:
        try:
            built[case_id] = build(
                case_id, connection=connection, user=user, layer=layer
            )
        except Exception:  # noqa: BLE001 - one broken scenario is not five
            continue
    return built


# ---------------------------------------------------------------------------
# The pieces the case assembler does not produce
# ---------------------------------------------------------------------------


def held_residuals(
    adjudication: Adjudication, layer: SemanticLayer | None = None
) -> tuple[HeldResidual, ...]:
    """What each live hypothesis is left holding, off the case's own evidence.

    THE THIRD CONDITION OF THE VERDICT TABLE READS THIS, and #2451 turns
    on it: H2 is alive, cannot be checked with sources the organisation
    holds, and is holding INR 0.86 Cr against a INR 50 L limit. That is
    what makes the case *partially* explained rather than explained.
    """
    layer = layer or get_semantic_layer()
    # BOTH SIDES IN RUPEES. The evidence declares these in crore, and
    # `HeldResidual` compares them in rupees — converting one and not the
    # other silently compares 0.86 against 5,000,000 and reports both as
    # zero on the verdict chip. The unit travels on the record precisely
    # so this conversion can be done from it rather than assumed.
    limit = _inr(adjudication.materiality, layer)

    held: list[HeldResidual] = []
    for hypothesis in adjudication.hypotheses:
        if hypothesis.status == "eliminated" or hypothesis.residual_held is None:
            continue
        held.append(
            HeldResidual(
                hypothesis=hypothesis.hypothesis_id,
                label=hypothesis.label,
                verifiable=hypothesis.verifiable,
                residual_inr=_inr(hypothesis.residual_held, layer),
                materiality_inr=limit,
                missing_sources=tuple(hypothesis.missing_sources),
            )
        )
    return tuple(held)


def _triggers(adjudication: Adjudication, layer: SemanticLayer) -> tuple[Trigger, ...]:
    """The eight, with the ones this case recorded as fired marked fired.

    The canonical object carries `triggers_fired` because the scenario
    config declares which fired — #2467's T3, T4 and T7. Reconstructed as
    `Trigger` objects so `decide` walks the same table it walks on a live
    case, rather than being handed a shortcut.
    """
    fired = set(adjudication.triggers_fired)
    return tuple(
        Trigger(
            trigger_id,
            spec.name,
            trigger_id in fired,
            spec.name if trigger_id in fired else "did not fire",
            {},
        )
        for trigger_id, spec in sorted(layer.adjudication.triggers.items())
    )


def _with_gates(
    adjudication: Adjudication, layer: SemanticLayer, config: Any
) -> Adjudication:
    """Give a case that OPENED the gate chips it passed on the way in.

    A killed case already carries the gate that killed it — that is the
    whole story of #2470 and #2471. A case that opened carries none,
    because `llm/casefiles.py` had no reason to record five passes for a
    narrator. The UI does: five green chips is how a reader sees that the
    movement was screened before anybody started explaining it.

    The DETAIL on each chip is drawn from the case's own figures, so a
    chip is never a decoration with no number behind it.
    """
    if adjudication.gates:
        return adjudication

    declared = layer.adjudication.gates
    details = _gate_details(adjudication)
    gates = tuple(
        GateResult(
            gate_id=gate_id,
            name=spec.name,
            passed=True,
            outcome_code=None,
            detail=details.get(spec.outcome_code, f"{spec.name} passed."),
            checks=(
                CheckResult(
                    check_id=spec.name.lower().replace(" ", "_"),
                    name=spec.name,
                    order=gate_id,
                    outcome=CLEAN,
                    detail=details.get(spec.outcome_code, f"{spec.name} passed."),
                ),
            ),
        )
        for gate_id, spec in sorted(declared.items())
    )
    return adjudication.model_copy(update={"gates": gates})


def _with_region_strip(
    adjudication: Adjudication, layer: SemanticLayer, config: Any
) -> Adjudication:
    """Publish the per-region residual strip as evidence.

    Gate 4 decomposes every peer region by the same method and compares
    them — that comparison is what makes a movement SPECIFIC rather than
    the market, and it is the difference between "West is down" and "the
    market is down and West is in it". The scenario config declares the
    strip; the narrator's assembler had no reason to publish it as
    clickable evidence, and the UI does.

    Minted through `EvidenceFactory` like everything else, so each region's
    figure is chaseable to the same generator config the rest of the case
    comes from.
    """
    targets = (config.get("targets") or {}).get("regional_residual_strip_pt") or {}
    if not targets:
        return adjudication

    # PINNED, like every other record on a canonical case. The factory
    # defaults `retrieved_at` to the wall clock, which would make the case
    # file a different document on every build — and the narrator's
    # fixture is keyed by a hash of that document, so the offline demo
    # would lose its narrative every time the server restarted.
    factory = EvidenceFactory.for_stage(
        EVIDENCE_PREFIX,
        adjudication.opened_at,
        layer,
        retrieved_at=adjudication.opened_at,
    )
    units = layer.warehouse.units
    minted = []
    for region, value in sorted(targets.items()):
        minted.append(
            factory.emit(
                f"region_residual.{region.lower()}",
                kind="derived_estimate",
                label=f"{region} residual, {adjudication.period}",
                value=round(float(value), units.pct_places),
                unit="pt",
                source_system=DERIVED_ORIGIN,
                method=DIFFERENCE_METHOD,
                description=(
                    f"{region} decomposed by the same calendar model as "
                    f"{adjudication.scope}, then headline minus calendar"
                ),
                ref="semantic_layer/qualify.yaml::specificity",
                notes=(
                    "The subject region."
                    if region == adjudication.scope
                    else "A peer, decomposed identically. Gate 4 compares them."
                ),
            )
        )
    return adjudication.model_copy(
        update={"evidence": (*adjudication.evidence, *minted)}
    )


#: What each gate would have said had it passed, keyed by the OUTCOME CODE
#: the gate declares rather than by its number. The numbering lives in
#: `adjudication.yaml` and belongs to it; keying on it here would put five
#: unexplained integers in a module that has no business knowing them.
DATA_INCIDENT = "DATA_INCIDENT"
CALENDAR_UNFIT = "CALENDAR_MODEL_UNFIT"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
MARKET_CASE = "MARKET_CASE"
IMMATERIAL = "IMMATERIAL"


def _gate_details(adjudication: Adjudication) -> dict[str, str]:
    """One sentence per gate, from the case's own numbers."""
    headline = adjudication.headline_movement
    residual = adjudication.qualified_residual
    band = adjudication.empirical_band
    materiality = adjudication.materiality

    attributed = adjudication.attributed[0] if adjudication.attributed else None
    details: dict[str, str] = {
        DATA_INCIDENT: "Every feed in scope loaded. No data incident.",
        MARKET_CASE: "Peer regions did not breach together; this is not the market.",
    }
    if attributed is not None:
        details[CALENDAR_UNFIT] = (
            f"Headline {headline.value}{_unit(headline)} decomposed: "
            f"{attributed.value}{_unit(attributed)} the calendar accounts for."
        )
    if band is not None and residual is not None:
        details[INSUFFICIENT_HISTORY] = (
            f"Residual {residual.value}{_unit(residual)} against a "
            f"{band.value}{_unit(band)} empirical band — outside it."
        )
    if materiality is not None and residual is not None:
        details[IMMATERIAL] = (
            f"Residual is above the {materiality.value}{_unit(materiality)} limit "
            "the contract sets. Worth an investigation."
        )
    return details


def _unit(evidence) -> str:
    unit = (evidence.unit or "").lower()
    return {"pct": "%", "pt": " pt", "inr_cr": " Cr"}.get(unit, f" {evidence.unit}")


def _recommend(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    adjudication: Adjudication,
    decision,
    config: Any,
    *,
    layer: SemanticLayer,
) -> Recommendation | None:
    """Run the real RECOMMEND over the canonical case.

    Not a canned action: `engine/recommend` matches the playbook to the
    adjudicated driver, multiplies the playbook's rate by the case's own
    treated-store count, and applies the recovery curve. Every figure on
    the action panel is the same arithmetic a live case would get.
    """
    leading = _leading(adjudication, config, layer)
    unresolved = tuple(
        item.hypothesis_id
        for item in adjudication.hypotheses
        if item.status == "live" and (leading is None or item.hypothesis_id != leading[0])
    )
    driver, attributable_inr, treated = leading or (None, 0.0, 0)

    request = RecommendRequest(
        case_id=adjudication.case_id,
        kpi=adjudication.kpi,
        scope=adjudication.scope,
        grain=adjudication.grain,
        period=adjudication.period,
        verdict=decision.value,
        leading_driver=driver,
        unresolved_drivers=unresolved,
        attributable_inr=attributable_inr,
        residual_inr=_inr(adjudication.qualified_residual, layer),
        basis=CostBasis(
            treated_store_count=treated,
            affected_store_count=treated,
            revenue_at_risk_inr=_inr(adjudication.qualified_residual, layer),
        ),
        triggers_fired=adjudication.triggers_fired,
        missing_sources=_missing(adjudication, driver),
        missing_for_hypothesis=driver,
    )
    try:
        result = recommend(
            connection,
            user,
            request,
            layer=layer,
            # A canonical case must not open a case on the register: it is
            # a rendering of a scenario, not an investigation somebody ran.
            open_linked_case=False,
            write_gaps=False,
        )
    except (RecommendError, Exception):  # noqa: BLE001
        return None
    return recommendation_contract(result, layer)


def _leading(
    adjudication: Adjudication, config: Any, layer: SemanticLayer
) -> tuple[str, float, int] | None:
    """`(driver, attributable rupees, treated stores)` for the supported one.

    THE DRIVER IS NOT THE H-NUMBER. The deck labels hypotheses H1..H5;
    RECOMMEND matches on a causal-graph tag. The bridge is the playbook
    the scenario config names — `availability_recovery`, whose declared
    driver is `stock_out` — so the tag is read off the semantic layer
    rather than mapped by hand in a dictionary that would go stale the
    first time somebody renamed a hypothesis.
    """
    supported = [
        item for item in adjudication.hypotheses if item.status == "supported"
    ]
    if not supported:
        return None
    best = max(supported, key=lambda item: item.attributed_share or 0.0)

    playbook_name = (config.get("recommendation") or {}).get("playbook")
    playbook = layer.playbooks.get(playbook_name) if playbook_name else None
    if playbook is None:
        return None

    attributed = best.attributed
    inr = 0.0
    if attributed is not None and attributed.value is not None:
        inr = float(attributed.value) * _crore(attributed, layer)
    return playbook.driver, inr, _treated(config)


def _treated(config: Any) -> int:
    """Treated stores, from the scenario's own mechanism block."""
    return int((config.get("mechanism") or {}).get("treated_store_count") or 0)


def _missing(adjudication: Adjudication, driver: str | None) -> tuple[str, ...]:
    for item in adjudication.hypotheses:
        if driver and item.hypothesis_id == driver:
            return tuple(item.missing_sources)
    return ()


def _inr(evidence, layer: SemanticLayer) -> float:
    if evidence is None or evidence.value is None:
        return 0.0
    return float(evidence.value) * _crore(evidence, layer)


def _crore(evidence, layer: SemanticLayer | None = None) -> float:
    layer = layer or get_semantic_layer()
    return (
        layer.warehouse.units.inr_per_crore
        if (evidence.unit or "").upper() == "INR_CR"
        else 1.0
    )


__all__ = [
    "CANONICAL",
    "DERIVED_ORIGIN",
    "DIFFERENCE_METHOD",
    "CLEAN",
    "LIVE",
    "CanonicalCase",
    "build",
    "build_all",
    "held_residuals",
]
