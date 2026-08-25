"""RECOMMEND — the stage that turns a verdict into something a person does.

    verdict reached   ->  an ACTION: playbook, owner, cost, recovery band,
                          ROI, monitoring plan, and the linked case that
                          fixes the cause of the cause
    verdict abstained ->  an INVESTIGATION LIST ranked by value per rupee,
                          plus at least one lever refused out loud, priced,
                          with its error rate

THE SECOND SHAPE IS THE PRODUCT. Any system can recommend an action once
it has decided something. Publishing a ranked, priced list of ways to
find out — and naming the thing you must not do while you do not know —
is what makes an abstention useful instead of an apology.

NOTHING HERE INVENTS A NUMBER. The cost is a playbook rate times a
measured quantity. The recovery band is the attributed money times a
curve fitted on prior closed cases. The owner is read off the KPI
contract's driver ownership. The error rate on a refusal is (k-1)/k from
trigger T4 or one minus our own accuracy, whichever is worse. Every one
of them leaves here as `Evidence`, so every figure on the recommendation
panel is clickable to the multiplication that produced it (rule 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import duckdb

from engine.abstain.verdict import INSUFFICIENT_EVIDENCE
from engine.contracts import Evidence, Recommendation
from engine.db import warehouse_clock
from engine.evidence import EvidenceFactory, EvidenceLedger
from engine.recommend.cost import CallDown, Cost, CostBasis, compute_cost, plan_call_down
from engine.recommend.investigate import (
    Investigation,
    MatchError,
    NotRecommended,
    match_playbook,
    rank_investigations,
    refuse,
)
from engine.recommend.linked import (
    LinkedCase,
    SourceGap,
    create_linked_case,
    source_gaps,
    write_source_gaps,
)
from engine.recommend.ownership import Owner, escalation_owner, resolve_owner
from engine.recommend.recovery import Recovery, compute_recovery
from security.policy import User
from semantic_layer.schema import Playbook, SemanticLayer, get_semantic_layer

EVIDENCE_PREFIX = "recommend"
DERIVED_KIND = "derived_estimate"
DERIVED = "derived"
CONTRACT = "semantic_layer"

INR = "INR"
RATIO = "ratio"
COUNT = "count"
HOURS = "hours"


class RecommendError(RuntimeError):
    """The case cannot be turned into a recommendation as asked."""


@dataclass(frozen=True)
class RecommendRequest:
    """The frozen case, as VERDICT left it.

    Everything on it was measured somewhere upstream. This stage adds
    rates and arithmetic; it does not re-open a finding.
    """

    case_id: str
    kpi: str
    scope: str
    grain: str
    period: str
    verdict: str

    #: The driver ADJUDICATE settled on. None on an abstained case, which
    #: is exactly why an abstained case gets investigations instead.
    leading_driver: str | None
    #: Hypotheses the case could not settle. What the investigations are
    #: ranked against.
    unresolved_drivers: tuple[str, ...] = ()

    #: Money the leading hypothesis accounted for. Recovery is quoted
    #: against this and never against the headline.
    attributable_inr: float = 0.0
    #: The qualified residual, in rupees. What an investigation could
    #: unlock and what a refused lever would be chasing.
    residual_inr: float = 0.0

    basis: CostBasis = field(default_factory=CostBasis)

    triggers_fired: tuple[str, ...] = ()
    #: Sources the leading hypothesis needs and the organisation lacks.
    #: Written to the data gap register when T3 is among the triggers.
    missing_sources: tuple[str, ...] = ()
    missing_for_hypothesis: str | None = None

    #: T4's tie count, and our own accuracy on this case type. Both feed
    #: the error rate quoted on a refusal.
    indistinguishable_count: int = 0
    case_type_accuracy: float | None = None

    @property
    def abstained(self) -> bool:
        return self.verdict == INSUFFICIENT_EVIDENCE


@dataclass(frozen=True)
class RecommendationResult:
    """Everything the recommendation panel renders, with its derivation."""

    request: RecommendRequest
    playbook: Playbook | None
    owner: Owner | None
    escalation: Owner | None
    cost: Cost | None
    recovery: Recovery | None
    call_down: CallDown
    investigations: tuple[Investigation, ...] = ()
    not_recommended: tuple[NotRecommended, ...] = ()
    linked_case: LinkedCase | None = None
    gaps: tuple[SourceGap, ...] = ()
    gaps_written: int = 0
    evidence: tuple[Evidence, ...] = ()
    as_of: datetime | None = None
    no_playbook_reason: str | None = None

    @property
    def action(self) -> str | None:
        """The playbook's own words. The model may re-phrase, never invent."""
        if self.playbook is not None:
            return self.playbook.action.strip()
        if self.investigations:
            return self.investigations[0].action
        return None

    @property
    def playbook_ref(self) -> str | None:
        if self.playbook is not None:
            return self.playbook.playbook
        return self.investigations[0].playbook if self.investigations else None

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.evidence}

    def render(self, layer: SemanticLayer) -> str:
        lines = [f"recommendation for case #{self.request.case_id} ({self.request.verdict})"]
        if self.playbook is not None:
            lines.append(f"  playbook: {self.playbook.playbook} ({self.playbook.lever})")
        if self.owner is not None:
            lines.append(f"  {self.owner.render()}")
        if self.cost is not None:
            lines.append(f"  cost: {self.cost.render(layer)}")
        if self.recovery is not None:
            lines.append(f"  {self.recovery.render(layer)}")
        lines.append(f"  call-down: {self.call_down.render()}")
        if self.investigations:
            lines.append("  resolution, by value per rupee:")
            lines.extend(item.render(layer) for item in self.investigations)
        for refused in self.not_recommended:
            lines.append(f"  {refused.render(layer)}")
        if self.linked_case is not None:
            lines.append(f"  {self.linked_case.render()}")
        if self.gaps:
            lines.append(f"  data gaps registered ({self.gaps_written}):")
            lines.extend(gap.render(layer) for gap in self.gaps)
        if self.no_playbook_reason:
            lines.append(f"  {self.no_playbook_reason}")
        return "\n".join(lines)


def recommend(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    request: RecommendRequest,
    *,
    layer: SemanticLayer | None = None,
    clock: datetime | None = None,
    open_linked_case: bool = True,
    write_gaps: bool = True,
) -> RecommendationResult:
    """Recommend on one adjudicated case."""
    layer = layer or get_semantic_layer()
    clock = clock or warehouse_clock(connection, layer)
    contract = layer.kpis.get(request.kpi)
    if contract is None:
        raise RecommendError(
            f"no KPI contract named {request.kpi!r}; known: {sorted(layer.kpis)}"
        )

    factory = EvidenceFactory.for_stage(EVIDENCE_PREFIX, clock, layer)
    ledger = EvidenceLedger()
    call_down = plan_call_down(request.basis.call_down_stores, layer)
    _emit_call_down(factory, ledger, layer, call_down)

    playbook: Playbook | None = None
    owner: Owner | None = None
    escalation: Owner | None = None
    cost: Cost | None = None
    recovery: Recovery | None = None
    linked: LinkedCase | None = None
    investigations: tuple[Investigation, ...] = ()
    not_recommended: tuple[NotRecommended, ...] = ()
    no_playbook_reason: str | None = None

    if not request.abstained:
        if request.leading_driver is None:
            raise RecommendError(
                f"case {request.case_id} reached {request.verdict} with no leading "
                "driver; a recommendation attaches to the driver adjudication settled "
                "on, and there is nothing to attach to"
            )
        try:
            playbook = match_playbook(request.leading_driver, layer)
        except MatchError as exc:
            no_playbook_reason = str(exc)
        if playbook is not None:
            cost = compute_cost(playbook, request.basis, layer)
            recovery = compute_recovery(
                playbook, request.attributable_inr, layer, cost=cost
            )
            owner = resolve_owner(contract, request.leading_driver, playbook=playbook)
            escalation = escalation_owner(contract, playbook)
            _emit_action(factory, ledger, layer, playbook, cost, recovery, owner)
            if open_linked_case:
                linked = create_linked_case(
                    connection,
                    user,
                    playbook,
                    layer,
                    parent_case_id=request.case_id,
                    scope=request.scope,
                    grain=request.grain,
                    period=request.period,
                    opened_at=clock,
                )
                if linked is not None:
                    _emit_linked(factory, ledger, linked)
    else:
        investigations = rank_investigations(
            request.unresolved_drivers,
            request.basis,
            layer,
            residual_at_risk_inr=abs(request.residual_inr),
        )
        minimum = layer.recommend.resolution.min_entries
        if len(investigations) < minimum:
            raise RecommendError(
                f"case {request.case_id} abstained and produced {len(investigations)} "
                f"investigations, below the {minimum} an abstention must offer. An "
                "abstention with no priced way out is an apology, not a finding."
            )
        not_recommended = refuse(
            request.unresolved_drivers,
            request.basis,
            layer,
            indistinguishable_count=request.indistinguishable_count,
            case_type_accuracy=request.case_type_accuracy,
        )
        if layer.recommend.resolution.not_recommended.required and not not_recommended:
            raise RecommendError(
                f"case {request.case_id} abstained and refused nothing. recommend.yaml "
                "requires an explicit refusal: a resolution panel that only lists things "
                "to buy reads as a shopping list."
            )
        owner = resolve_owner(contract, request.leading_driver)
        _emit_resolution(factory, ledger, layer, investigations, not_recommended)

    gaps: tuple[SourceGap, ...] = ()
    written = 0
    gap_spec = layer.recommend.data_gap
    if gap_spec.trigger in request.triggers_fired and request.missing_sources:
        hypothesis = request.missing_for_hypothesis or (
            request.leading_driver or request.unresolved_drivers[0]
        )
        gaps = source_gaps(
            hypothesis,
            request.missing_sources,
            request.basis,
            layer,
            case_id=request.case_id,
        )
        if write_gaps:
            written = write_source_gaps(
                connection, gaps, layer, scope=request.scope, detected_at=clock
            )
        _emit_gaps(factory, ledger, layer, gaps)

    return RecommendationResult(
        request=request,
        playbook=playbook,
        owner=owner,
        escalation=escalation,
        cost=cost,
        recovery=recovery,
        call_down=call_down,
        investigations=investigations,
        not_recommended=not_recommended,
        linked_case=linked,
        gaps=gaps,
        gaps_written=written,
        evidence=tuple(ledger),
        as_of=clock,
        no_playbook_reason=no_playbook_reason,
    )


# ---------------------------------------------------------------------------
# Evidence. Every number on the panel, with its multiplication attached.
# ---------------------------------------------------------------------------


def _emit_call_down(factory, ledger, layer, call_down: CallDown) -> None:
    ledger.add(
        factory.emit(
            "call_down_managers",
            kind=DERIVED_KIND,
            label="Managers on the call-down",
            value=call_down.managers,
            unit=COUNT,
            source_system=DERIVED,
            method=COUNT,
            description=(
                f"{call_down.stores} stores at {call_down.span_of_control} to a cluster "
                "manager, rounded up"
            ),
            ref="semantic_layer/recommend.yaml::call_down.span_of_control_stores",
            notes="The store count is the case's; the span of control is the business's.",
        )
    )
    ledger.add(
        factory.emit(
            "call_down_hours",
            kind=DERIVED_KIND,
            label="Hours each manager spends",
            value=round(call_down.hours, layer.warehouse.units.crore_places),
            unit=HOURS,
            source_system=DERIVED,
            method="scale",
            description=(
                f"{call_down.span_of_control} calls at "
                f"{call_down.minutes_per_store:g} minutes each"
            ),
            ref="semantic_layer/recommend.yaml::call_down.minutes_per_store",
            inputs=(f"{EVIDENCE_PREFIX}.call_down_managers",),
        )
    )


def _emit_action(factory, ledger, layer, playbook, cost: Cost, recovery: Recovery, owner) -> None:
    units = layer.warehouse.units
    ledger.add(
        factory.emit(
            "cost",
            kind=DERIVED_KIND,
            label=f"Cost of {playbook.playbook}",
            value=round(cost.lakh(layer), units.crore_places),
            unit="INR_L",
            source_system=DERIVED,
            method="scale",
            description=cost.detail,
            ref=f"semantic_layer/playbooks/{playbook.playbook}.yaml::cost_model",
            notes=(
                "The playbook supplies the rate and the case supplies the quantity. "
                "Neither file can produce this number on its own."
            ),
        )
    )
    ledger.add(
        factory.emit(
            "owner",
            kind=DERIVED_KIND,
            label="Owner",
            value=owner.title,
            unit="role",
            source_system=CONTRACT,
            method="lookup",
            description=(
                f"driver {owner.driver!r} is owned by {owner.role} under "
                f"{owner.resolved_from}"
            ),
            ref=f"semantic_layer/kpis/{playbook.monitoring_plan.primary_kpi}.yaml",
            notes=(
                f"playbook {playbook.playbook} names {owner.playbook_role}; the KPI "
                "contract wins, and the override is recorded rather than resolved "
                "quietly."
                if owner.overrides_playbook
                else None
            ),
        )
    )
    for key, label, amount, quartile in (
        ("recovery_low", "Expected recovery, p25", recovery.low_inr, recovery.p25),
        ("recovery_high", "Expected recovery, p75", recovery.high_inr, recovery.p75),
    ):
        ledger.add(
            factory.emit(
                key,
                kind=DERIVED_KIND,
                label=label,
                value=round(amount / units.inr_per_crore, units.crore_places),
                unit="INR_CR",
                source_system=DERIVED,
                method="scale",
                description=(
                    f"{quartile:.0%} of INR "
                    f"{recovery.attributable_inr / units.inr_per_crore:.2f} Cr "
                    f"attributed, over {recovery.horizon_weeks} weeks"
                ),
                ref=f"semantic_layer/recovery_curves.yaml::curves.{recovery.curve_ref}",
                inputs=("adjudicate.coverage",),
                notes=(
                    f"fitted on {recovery.sample_size} prior comparable cases, which "
                    f"recommend.yaml labels {recovery.confidence}. {recovery.basis}"
                ),
            )
        )
    if recovery.roi_low is not None and recovery.roi_high is not None:
        for key, label, value in (
            ("roi_low", "ROI at p25", recovery.roi_low),
            ("roi_high", "ROI at p75", recovery.roi_high),
        ):
            ledger.add(
                factory.emit(
                    key,
                    kind=DERIVED_KIND,
                    label=label,
                    value=round(value, units.crore_places),
                    unit=RATIO,
                    source_system=DERIVED,
                    method="ratio",
                    description="expected recovery over the cost of the action",
                    ref="semantic_layer/recommend.yaml::recovery",
                    inputs=(
                        f"{EVIDENCE_PREFIX}.recovery_"
                        + ("low" if key == "roi_low" else "high"),
                        f"{EVIDENCE_PREFIX}.cost",
                    ),
                )
            )


def _emit_resolution(factory, ledger, layer, investigations, refused) -> None:
    units = layer.warehouse.units
    for rank, item in enumerate(investigations):
        ledger.add(
            factory.emit(
                f"investigation_{rank}_cost",
                kind=DERIVED_KIND,
                label=f"Cost of {item.playbook}",
                value=round(item.cost.lakh(layer), units.crore_places),
                unit="INR_L",
                source_system=DERIVED,
                method="scale",
                description=item.cost.detail,
                ref=f"semantic_layer/playbooks/{item.playbook}.yaml::cost_model",
            )
        )
        ledger.add(
            factory.emit(
                f"investigation_{rank}_value",
                kind=DERIVED_KIND,
                label=f"Value per rupee, {item.playbook}",
                value=round(item.value_per_rupee, units.crore_places),
                unit=RATIO,
                source_system=DERIVED,
                method="ratio",
                description=(
                    f"INR {item.residual_at_risk_inr / units.inr_per_crore:.2f} Cr at "
                    f"risk x {item.prior:.0%} prior on {item.driver}, over the cost"
                ),
                ref=f"semantic_layer/causal_graph.yaml::hypotheses.{item.driver}.prior",
                inputs=(f"{EVIDENCE_PREFIX}.investigation_{rank}_cost",),
                notes=(
                    "The prior is the causal graph's. An investigation into a driver we "
                    "rarely find guilty is worth less than the same money on a common "
                    "one, and the graph already says which is which."
                ),
            )
        )
    for rank, item in enumerate(refused):
        ledger.add(
            factory.emit(
                f"not_recommended_{rank}_exposure",
                kind=DERIVED_KIND,
                label=f"Margin at risk, {item.playbook}",
                value=round(item.cost.crore(layer), units.crore_places),
                unit="INR_CR",
                source_system=DERIVED,
                method="scale",
                description=item.cost.detail,
                ref=f"semantic_layer/playbooks/{item.playbook}.yaml::cost_model",
                notes=(
                    "No price event was observed, so there is no rollback depth to "
                    "price. Quoted at the gross margin carried by the revenue the lever "
                    "would be chasing."
                    if item.cost.depth_source != "observed_price_increase"
                    else None
                ),
            )
        )
        ledger.add(
            factory.emit(
                f"not_recommended_{rank}_error_rate",
                kind=DERIVED_KIND,
                label=f"Chance of being wrong, {item.playbook}",
                value=round(item.error_rate, units.crore_places),
                unit=RATIO,
                source_system=DERIVED,
                method="compare",
                description=item.error_rate_detail,
                ref="semantic_layer/recommend.yaml::resolution.not_recommended",
                notes=(
                    "The worse of the two computed sources is quoted. Reaching for the "
                    "flattering one on a case that just abstained would be exactly the "
                    "wrong instinct."
                ),
            )
        )


def _emit_linked(factory, ledger, linked: LinkedCase) -> None:
    ledger.add(
        factory.emit(
            "linked_case",
            kind=DERIVED_KIND,
            label="Linked case — the cause of the cause",
            value=linked.case_id,
            unit="case_id",
            source_system="warehouse",
            method="lookup",
            description=" ".join(linked.question.split()),
            ref=f"semantic_layer/playbooks/{linked.playbook}.yaml::linked_case_template",
            notes=(
                f"opened against {linked.kpi} for {linked.scope}, driver "
                f"{linked.driver}, from case #{linked.parent_case_id}"
            ),
        )
    )


def _emit_gaps(factory, ledger, layer, gaps: tuple[SourceGap, ...]) -> None:
    spec = layer.recommend.data_gap
    for gap in gaps:
        ledger.add(
            factory.emit(
                f"data_gap_{gap.source}",
                kind=DERIVED_KIND,
                label=f"Source not held: {gap.source}",
                value=spec.value_when_missing,
                unit=spec.unit,
                source_system="warehouse",
                method="lookup",
                description=(
                    f"{gap.hypothesis} needs {gap.source}, which the organisation does "
                    f"not hold. {gap.resolution_hint}"
                ),
                ref=f"semantic_layer/causal_graph.yaml::sources.{gap.source}",
                notes=(
                    f"written to data_gap_register as {spec.gap_code} — a standing fact "
                    "about the business, not an incident in this case."
                ),
            )
        )


# ---------------------------------------------------------------------------
# The frozen contract
# ---------------------------------------------------------------------------


def to_contract(result: RecommendationResult, layer: SemanticLayer) -> Recommendation:
    """The recommendation as the object the narrative layer receives.

    Rule 1: the model may render this and never alter it. Which is why
    `action` is the playbook's own text — the model re-phrases it for a
    persona, but the words it starts from came from a file somebody in the
    business signed off.
    """
    ledger = result.evidence_by_id()
    action = result.action
    playbook_ref = result.playbook_ref
    if action is None or playbook_ref is None:
        raise RecommendError(
            f"case {result.request.case_id} produced neither an action nor an "
            f"investigation to lead with. {result.no_playbook_reason or ''}".strip()
        )

    def eid(key: str) -> str:
        return f"{EVIDENCE_PREFIX}.{key}"

    if result.playbook is not None:
        cost_id, low_id, high_id = eid("cost"), eid("recovery_low"), eid("recovery_high")
        recovery = result.recovery
        sample_size = recovery.sample_size if recovery else 0
        confidence = recovery.confidence if recovery else "Low"
        roi_low = recovery.roi_low if recovery else None
        roi_high = recovery.roi_high if recovery else None
    else:
        # An abstained case leads with its top investigation. It recovers
        # nothing by construction, and the contract says so with a zero
        # band rather than by omitting the field.
        cost_id = eid("investigation_0_cost")
        low_id = high_id = None
        sample_size = 0
        confidence = "Low"
        roi_low = roi_high = None

    zero = None
    if low_id is None:
        factory = EvidenceFactory.for_stage(EVIDENCE_PREFIX, result.as_of, layer)
        zero = factory.emit(
            "recovery_none",
            kind=DERIVED_KIND,
            label="Expected recovery",
            value=0.0,
            unit="INR_CR",
            source_system=DERIVED,
            method="scale",
            description=(
                "An investigation buys an answer, not revenue. Crediting it with "
                "recovery would double-count the action it unblocks."
            ),
            ref="semantic_layer/recovery_curves.yaml::curves.instrumentation_none",
        )

    return Recommendation(
        case_id=result.request.case_id,
        action=action,
        playbook_ref=playbook_ref,
        cost=ledger[cost_id],
        expected_recovery_low=ledger[low_id] if low_id else zero,
        expected_recovery_high=ledger[high_id] if high_id else zero,
        roi_low=roi_low,
        roi_high=roi_high,
        recovery_confidence=confidence,
        sample_size=sample_size,
        effort=result.call_down.render(),
        not_recommended=tuple(item.render(layer) for item in result.not_recommended),
        linked_case_id=result.linked_case.case_id if result.linked_case else None,
        evidence_ids=tuple(ledger),
    )


__all__ = [
    "EVIDENCE_PREFIX",
    "RecommendError",
    "RecommendRequest",
    "RecommendationResult",
    "recommend",
    "to_contract",
]
