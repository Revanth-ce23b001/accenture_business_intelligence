"""Playbook matching, and what an abstained case outputs instead of an action.

TWO JOBS, and the second is the one worth reading.

MATCHING is short. A recommendation attaches to the driver ADJUDICATION
settled on — `stock_out` on #2451 — and the playbook is whichever one
carries that driver. Where a driver carries several, a lever that
recovers beats a lever that only buys information, and a shorter lead
time breaks the tie.

RESOLUTION is what happens when there is no driver to attach to, because
the case abstained. The output is then a list of INVESTIGATIONS ranked by
value per rupee:

    value = the residual at stake x the causal graph's prior for the
            driver this investigation would make testable
    cost  = the playbook's cost model against this case's own quantities

Both halves are computed. The prior is the graph's, so an investigation
into a driver the organisation rarely finds guilty is worth less than the
same money spent on a common one, and nobody has to argue about that in
the room.

AND SOMETHING MUST BE REFUSED. A resolution panel that only lists things
to buy reads as a shopping list. The refusal is the part that costs
credibility to write and earns it to publish: on #2467 the lever a
reasonable person reaches for first is a price response, and it is
refused out loud, with what it would give away and how likely it is to be
aimed at the wrong hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_layer.schema import LeadTime, Playbook, SemanticLayer

from engine.recommend.cost import Cost, CostBasis, CostError, compute_cost


class MatchError(RuntimeError):
    """No playbook can be attached to this driver."""


def lead_time_days(lead_time: LeadTime, layer: SemanticLayer) -> float:
    """One scale for lead times, so two playbooks can be compared."""
    units = layer.warehouse.units
    if lead_time.unit == "hours":
        return lead_time.value / units.hours_per_day
    if lead_time.unit == "days":
        return lead_time.value
    if lead_time.unit == "weeks":
        return lead_time.value * units.days_per_week
    return lead_time.value * units.days_per_year / units.months_per_year


def is_informational(playbook: Playbook, layer: SemanticLayer) -> bool:
    """True when the lever buys an answer rather than revenue."""
    return playbook.lever in layer.recommend.matching.information_levers


def candidates(driver: str, layer: SemanticLayer) -> list[Playbook]:
    """Every playbook carrying this driver, best first.

    Recovering levers ahead of informational ones, then shortest lead
    time. Both rules are `recommend.yaml -> matching`, not this module's
    opinion.
    """
    spec = layer.recommend.matching
    found = layer.playbooks_for(driver)

    def key(playbook: Playbook) -> tuple[int, float, str]:
        recovering = 0 if not is_informational(playbook, layer) else 1
        if not spec.prefer_recovering_lever:
            recovering = 0
        return (recovering, lead_time_days(playbook.lead_time, layer), playbook.playbook)

    return sorted(found, key=key)


def match_playbook(driver: str, layer: SemanticLayer) -> Playbook:
    """The action for an adjudicated driver. Raises rather than improvises."""
    found = candidates(driver, layer)
    if not found:
        raise MatchError(
            f"{layer.recommend.matching.no_playbook_outcome}: no playbook carries driver "
            f"{driver!r}. It has to be written before this case can be actioned."
        )
    return found[0]


# ---------------------------------------------------------------------------
# Investigations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Investigation:
    """One thing we could buy, and what it would be worth."""

    playbook: str
    driver: str
    label: str
    lever: str
    action: str
    cost: Cost
    prior: float
    residual_at_risk_inr: float
    value_inr: float
    lead_time_days: float
    lead_time: str
    unblocks: tuple[str, ...]
    owner_role: str

    @property
    def value_per_rupee(self) -> float:
        """What a rupee spent here could unlock. The ranking key."""
        if self.cost.amount_inr <= 0.0:
            raise ValueError(
                f"investigation {self.playbook!r} has no cost; it cannot be ranked by "
                "value per rupee"
            )
        return self.value_inr / self.cost.amount_inr

    def render(self, layer: SemanticLayer) -> str:
        return (
            f"  {self.playbook} ({self.label}): "
            f"INR {self.cost.lakh(layer):.2f} L, {self.lead_time} "
            f"-> unlocks INR {self.value_inr / layer.warehouse.units.inr_per_crore:.2f} Cr "
            f"({self.prior:.0%} prior on {self.driver}); "
            f"{self.value_per_rupee:.1f}x per rupee"
        )


def rank_investigations(
    drivers: tuple[str, ...],
    basis: CostBasis,
    layer: SemanticLayer,
    *,
    residual_at_risk_inr: float,
) -> tuple[Investigation, ...]:
    """Every investigation that could resolve one of `drivers`, best first.

    `drivers` are the hypotheses the case could not settle. Each one is
    looked up in the causal graph for its prior and in the playbook set
    for the levers that would make it testable. A driver with no playbook
    is skipped here and reported by the caller — silently dropping it
    would turn "we have no way to investigate this" into "there is
    nothing to investigate".

    ONLY INFORMATION LEVERS ARE LISTED. A recovering lever is an action,
    and an action on a case that abstained is the thing this whole system
    exists to refuse. `price_correction` is not a cheap investigation into
    a price hypothesis; it is a price cut. It belongs in `refuse` below,
    priced at what it would give away, and it must not appear on a
    resolution panel wearing the word "investigate".
    """
    graph = layer.causal_graph.hypotheses
    found: dict[str, Investigation] = {}

    for driver in drivers:
        template = graph.get(driver)
        if template is None:
            continue
        for playbook in candidates(driver, layer):
            if playbook.playbook in found or not is_informational(playbook, layer):
                continue
            try:
                cost = compute_cost(playbook, basis, layer)
            except CostError:
                # Unpriceable against this case's quantities. Left out of
                # a ranking that divides by cost rather than ranked at
                # zero, which would put it top of every list.
                continue
            if cost.amount_inr <= 0.0:
                continue
            found[playbook.playbook] = Investigation(
                playbook=playbook.playbook,
                driver=driver,
                label=template.label,
                lever=playbook.lever,
                action=playbook.action.strip(),
                cost=cost,
                prior=template.prior,
                residual_at_risk_inr=residual_at_risk_inr,
                value_inr=residual_at_risk_inr * template.prior,
                lead_time_days=lead_time_days(playbook.lead_time, layer),
                lead_time=f"{playbook.lead_time.value:g} {playbook.lead_time.unit}",
                unblocks=tuple(template.missing_sources()),
                owner_role=playbook.owner_role,
            )

    return tuple(
        sorted(
            found.values(),
            key=lambda item: (-item.value_per_rupee, item.lead_time_days, item.playbook),
        )
    )


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------

TIE = "indistinguishable_tie"
CALIBRATION = "case_type_calibration"


@dataclass(frozen=True)
class NotRecommended:
    """A lever we are advising against, priced, with its error rate."""

    playbook: str
    driver: str
    label: str
    action: str
    cost: Cost
    error_rate: float
    error_rate_source: str
    error_rate_detail: str
    preconditions: tuple[str, ...]
    reason: str

    def render(self, layer: SemanticLayer) -> str:
        return (
            f"NOT recommended — {self.label}: "
            f"INR {self.cost.crore(layer):.2f} Cr at risk, "
            f"{self.error_rate:.0%} chance of being wrong. {self.reason}"
        )


def error_rate(
    layer: SemanticLayer,
    *,
    indistinguishable_count: int = 0,
    case_type_accuracy: float | None = None,
) -> tuple[float, str, str]:
    """How likely acting on this would be acting on the wrong thing.

    Two computed sources and the WORSE of them is quoted. Reaching for
    the flattering number on a case that has just abstained would be
    exactly the wrong instinct, and `recommend.yaml` says `worst_of` so
    that instinct has nowhere to express itself.

    The tie source is the honest reading of trigger T4: if k hypotheses
    cannot be told apart on the data held, picking one is a (k-1)/k chance
    of picking wrong. At k=2 that is a coin flip, and the recommendation
    says so in those words.
    """
    spec = layer.recommend.resolution.not_recommended
    options: list[tuple[float, str, str]] = []

    if indistinguishable_count > 1:
        rate = (indistinguishable_count - 1) / indistinguishable_count
        options.append(
            (
                rate,
                TIE,
                f"{indistinguishable_count} hypotheses the held data cannot separate "
                f"(T4): acting on one is a {rate:.0%} chance of acting on the wrong one",
            )
        )
    if case_type_accuracy is not None:
        rate = 1.0 - case_type_accuracy
        options.append(
            (
                rate,
                CALIBRATION,
                f"our own record on this case type is {case_type_accuracy:.0%} accurate, "
                f"so {rate:.0%} of the time we were wrong",
            )
        )
    if not options:
        raise ValueError(
            "an explicit refusal must carry an error rate, and neither source was "
            f"supplied ({sorted(spec.error_rate_sources)})"
        )
    return max(options, key=lambda item: item[0])


def refuse(
    drivers: tuple[str, ...],
    basis: CostBasis,
    layer: SemanticLayer,
    *,
    indistinguishable_count: int = 0,
    case_type_accuracy: float | None = None,
) -> tuple[NotRecommended, ...]:
    """Price the levers this case cannot justify, and say so.

    A lever is refused when it carries preconditions the case cannot meet.
    On an abstained case NO hypothesis was established, so a playbook
    whose preconditions begin "the price event is confirmed in the pricing
    log" is refused by construction — and the whole point is to publish
    that refusal with its price attached rather than to leave the lever
    off the page and hope nobody reaches for it.
    """
    graph = layer.causal_graph.hypotheses
    rate, source, detail = error_rate(
        layer,
        indistinguishable_count=indistinguishable_count,
        case_type_accuracy=case_type_accuracy,
    )
    refused: list[NotRecommended] = []

    for driver in drivers:
        template = graph.get(driver)
        if template is None:
            continue
        for playbook in candidates(driver, layer):
            if playbook.preconditions is None:
                continue
            try:
                cost = compute_cost(playbook, basis, layer)
            except CostError:
                continue
            refused.append(
                NotRecommended(
                    playbook=playbook.playbook,
                    driver=driver,
                    label=template.label,
                    action=playbook.action.strip(),
                    cost=cost,
                    error_rate=rate,
                    error_rate_source=source,
                    error_rate_detail=detail,
                    preconditions=tuple(playbook.preconditions.rules),
                    reason=(
                        f"{len(playbook.preconditions.rules)} preconditions on "
                        f"{playbook.playbook} are unmet — the case never established "
                        f"{template.label.lower()} — and {detail}."
                    ),
                )
            )

    return tuple(sorted(refused, key=lambda item: -item.cost.amount_inr))


__all__ = [
    "CALIBRATION",
    "TIE",
    "Investigation",
    "MatchError",
    "NotRecommended",
    "candidates",
    "error_rate",
    "is_informational",
    "lead_time_days",
    "match_playbook",
    "rank_investigations",
    "refuse",
]
