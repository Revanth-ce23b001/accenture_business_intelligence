"""What it costs, and where each half of that number came from.

    cost = a RATE the playbook carries  x  a QUANTITY the case measured

Neither file can produce the number alone, and that is the point. INR
53,000 per store is a procurement fact and belongs to the playbook.
Thirty-four stores is a finding and belongs to the case. ₹18 L is what
you get when you multiply them, and there is nowhere in this repository
it could have been typed in instead.

`CostBasis` is the case's half. It carries only measured quantities —
counts of stores, a revenue exposure, an observed price movement — and
every one of them arrives from a governed query or from ADJUDICATE. The
playbook names which of them it scales with; `recommend.yaml` says what
the names mean; this module does the arithmetic and shows its working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from semantic_layer.schema import Playbook, RecommendConfig, SemanticLayer

#: Basis names, matching `recommend.yaml -> cost.bases`. Held as constants
#: so a typo is an import error rather than a silent zero.
NONE = "none"
TREATED_STORES = "treated_store_count"
AFFECTED_STORES = "affected_store_count"
UNINSTRUMENTED_STORES = "uninstrumented_store_count"
AFFECTED_STORE_WEEKS = "affected_store_weeks"
MANAGER_HOURS = "manager_hours"
REVENUE_AT_RISK = "revenue_at_risk"

RATE_TIMES_BASIS = "rate_times_basis"
DEPTH_TIMES_BASIS = "depth_times_basis"

#: What a depth-priced lever is quoted at when its depth was never
#: observed. Named rather than inlined so the evidence can cite it.
UNOBSERVED_DEPTH = "unobserved_depth"


class CostError(RuntimeError):
    """The playbook cannot be priced against this case."""


@dataclass(frozen=True)
class CallDown:
    """Managers, hours, and the arithmetic that produced both.

    An incident call-down goes to the cluster managers who own the
    affected stores, not to the stores. Retail operations puts four stores
    under a manager and half an hour on a store conversation, so 34
    treated stores is nine managers spending two hours each — and moving
    the case to a different number of stores moves both figures.
    """

    stores: int
    span_of_control: int
    managers: int
    minutes_per_store: float
    hours: float
    manager_hours: float

    def render(self) -> str:
        hours = int(self.hours) if float(self.hours).is_integer() else round(self.hours, 1)
        managers = f"{self.managers} manager" + ("s" if self.managers != 1 else "")
        return f"{managers}, {hours} hours"


def plan_call_down(stores: int, layer: SemanticLayer) -> CallDown:
    """Size the call-down for a case covering `stores` stores."""
    spec = layer.recommend.call_down
    managers = math.ceil(stores / spec.span_of_control_stores) if stores else 0
    per_manager = min(stores, spec.span_of_control_stores) if stores else 0
    hours = per_manager * spec.minutes_per_store / layer.warehouse.units.minutes_per_hour
    return CallDown(
        stores=stores,
        span_of_control=spec.span_of_control_stores,
        managers=managers,
        minutes_per_store=spec.minutes_per_store,
        hours=hours,
        manager_hours=managers * hours,
    )


@dataclass(frozen=True)
class CostBasis:
    """The case's half of every cost on the page.

    Everything here is MEASURED. A field left at its default is a quantity
    this case did not measure, and a playbook that scales with it is
    refused rather than priced at zero — a lever quoted at nothing is
    worse than a lever quoted as unpriceable, because it wins every
    ranking.
    """

    #: ADJUDICATE's treated group. Zero when no mechanism was identified.
    treated_store_count: int = 0
    #: Stores in the case scope.
    affected_store_count: int = 0
    #: Stores in scope with no counter for the sensor a hypothesis needs.
    uninstrumented_store_count: int = 0
    #: The money actually in play — the qualified residual, in rupees.
    revenue_at_risk_inr: float = 0.0
    #: The price movement the pricing log shows in scope, as a fraction.
    #: None means no price event was observed, which is not the same as
    #: an observed movement of zero and is not treated as one.
    observed_price_increase: float | None = None

    @property
    def call_down_stores(self) -> int:
        """Who gets rung.

        The treated stores when adjudication found a mechanism and knows
        which stores it reached; every store in scope when it did not,
        because on an abstained case the point of ringing round is to find
        out where the thing happened.
        """
        return self.treated_store_count or self.affected_store_count

    def quantity(self, basis: str, playbook: Playbook, layer: SemanticLayer) -> float:
        """The measured quantity a playbook scales with."""
        spec: RecommendConfig = layer.recommend
        if basis not in spec.cost.bases:
            raise CostError(
                f"playbook {playbook.playbook!r} scales with {basis!r}, which is not a "
                f"declared cost basis ({sorted(spec.cost.bases)})"
            )
        if basis == NONE:
            return 1.0
        if basis == TREATED_STORES:
            return float(self.treated_store_count)
        if basis == AFFECTED_STORES:
            return float(self.affected_store_count)
        if basis == UNINSTRUMENTED_STORES:
            return float(self.uninstrumented_store_count)
        if basis == AFFECTED_STORE_WEEKS:
            return float(self.affected_store_count * playbook.monitoring_plan.horizon_weeks)
        if basis == MANAGER_HOURS:
            return plan_call_down(self.call_down_stores, layer).manager_hours
        if basis == REVENUE_AT_RISK:
            return float(self.revenue_at_risk_inr)
        raise CostError(  # pragma: no cover - unreachable while the two lists agree
            f"cost basis {basis!r} is declared in recommend.yaml but this module has no "
            "way to measure it"
        )


@dataclass(frozen=True)
class Cost:
    """A priced playbook, with both halves of the multiplication kept."""

    playbook: str
    model_type: str
    basis: str
    quantity: float
    unit: str
    rate_inr: float | None
    depth: float | None
    depth_source: str | None
    amount_inr: float
    detail: str

    def lakh(self, layer: SemanticLayer) -> float:
        return self.amount_inr / layer.warehouse.units.inr_per_lakh

    def crore(self, layer: SemanticLayer) -> float:
        return self.amount_inr / layer.warehouse.units.inr_per_crore

    @property
    def priced(self) -> bool:
        return self.amount_inr > 0.0

    def render(self, layer: SemanticLayer) -> str:
        return f"{self.playbook}: INR {self.lakh(layer):.2f} L  ({self.detail})"


def compute_cost(
    playbook: Playbook, basis: CostBasis, layer: SemanticLayer
) -> Cost:
    """Price one playbook against one case.

    Refuses rather than guesses. A playbook whose basis this case did not
    measure, or whose depth cannot be established, comes back as an error
    the caller has to handle — never as a cheap number.
    """
    spec = layer.recommend.cost
    model = playbook.cost_model
    if model.type not in spec.types:
        raise CostError(
            f"playbook {playbook.playbook!r} uses cost model {model.type!r}, which "
            f"recommend.yaml does not declare ({sorted(spec.types)})"
        )
    arithmetic = spec.types[model.type].arithmetic
    quantity = basis.quantity(model.scales_with, playbook, layer)

    if arithmetic == RATE_TIMES_BASIS:
        rate = model.unit_cost_inr
        if rate is None:
            raise CostError(
                f"playbook {playbook.playbook!r} is priced per {model.unit!r} but "
                "carries no unit_cost_inr"
            )
        if quantity <= 0.0 and model.scales_with != NONE:
            raise CostError(
                f"playbook {playbook.playbook!r} scales with {model.scales_with!r}, "
                "which this case measured as zero; it cannot be priced here"
            )
        amount = rate * quantity + model.fixed_cost_inr
        detail = (
            f"INR {rate:,.0f} per {model.unit} x {quantity:g} {model.scales_with}"
            + (f" + INR {model.fixed_cost_inr:,.0f} fixed" if model.fixed_cost_inr else "")
        )
        return Cost(
            playbook=playbook.playbook,
            model_type=model.type,
            basis=model.scales_with,
            quantity=quantity,
            unit=model.unit,
            rate_inr=rate,
            depth=None,
            depth_source=None,
            amount_inr=amount,
            detail=detail,
        )

    # depth_times_basis — margin given away rather than money spent.
    depth, source = _depth(playbook, basis, layer)
    amount = depth * quantity + model.fixed_cost_inr
    detail = (
        f"{depth:.2%} of INR {quantity / layer.warehouse.units.inr_per_crore:.2f} Cr "
        f"{model.scales_with}, depth from {source}"
    )
    return Cost(
        playbook=playbook.playbook,
        model_type=model.type,
        basis=model.scales_with,
        quantity=quantity,
        unit=model.unit,
        rate_inr=None,
        depth=depth,
        depth_source=source,
        amount_inr=amount,
        detail=detail,
    )


def _depth(
    playbook: Playbook, basis: CostBasis, layer: SemanticLayer
) -> tuple[float, str]:
    """The rollback depth, and where it came from.

    THE INTERESTING BRANCH IS THE SECOND ONE. A price rollback can only
    give back a price rise that happened, so when the pricing log shows no
    event there is no depth to apply. The lever is not therefore free: it
    still gives away the gross margin carried by the revenue it would be
    chasing, and that is what it is quoted at. Pricing an unevidenced
    lever at its optimistic depth is how a coin-flip action gets onto a
    page looking affordable.
    """
    spec = layer.recommend.cost
    model = playbook.cost_model
    source = model.depth_source
    if source is None:
        raise CostError(
            f"playbook {playbook.playbook!r} is priced on a depth but names no "
            "depth_source"
        )
    if source not in spec.depth_sources:
        raise CostError(
            f"playbook {playbook.playbook!r} takes its depth from {source!r}, which "
            f"recommend.yaml does not declare ({sorted(spec.depth_sources)})"
        )
    if basis.observed_price_increase is not None:
        return basis.observed_price_increase, source

    fallback = spec.unobserved_depth
    if fallback.basis != model.scales_with:
        raise CostError(
            f"playbook {playbook.playbook!r} has no observed depth and scales with "
            f"{model.scales_with!r}, but recommend.yaml prices an unobserved depth "
            f"against {fallback.basis!r}"
        )
    return layer.recommend.economics.gross_margin_rate, UNOBSERVED_DEPTH


__all__ = [
    "AFFECTED_STORES",
    "AFFECTED_STORE_WEEKS",
    "DEPTH_TIMES_BASIS",
    "MANAGER_HOURS",
    "NONE",
    "RATE_TIMES_BASIS",
    "REVENUE_AT_RISK",
    "TREATED_STORES",
    "UNINSTRUMENTED_STORES",
    "UNOBSERVED_DEPTH",
    "CallDown",
    "Cost",
    "CostBasis",
    "CostError",
    "compute_cost",
    "plan_call_down",
]
