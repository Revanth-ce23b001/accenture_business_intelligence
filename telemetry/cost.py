"""What a call cost, what a case cost, and what a year of it would cost.

Three rules, and they are the same three rules the rest of the engine
runs on.

RATES COME FROM YAML. `semantic_layer/telemetry.yaml` holds the price
list, the dollar-rupee rate and the volume assumption. Nothing in this
module contains a price, and `tests/test_telemetry.py` proves it by
taking every value out of the loaded config and searching this package
for it as a literal — so the guard can never go stale against an edit to
the YAML.

AN UNPRICED MODEL IS A REFUSAL, NOT A ZERO. `SemanticLayer.price` raises
when a model has no entry. A call silently costed at zero is a budget
that is wrong and says nothing about it, which is the failure mode this
whole file exists to prevent.

A COST IS ITEMISED. `CallCost` keeps the four components apart — fresh
input, output, cache reads, cache writes — because "the case cost INR
4.10" is only useful if you can see that INR 3.60 of it was one narrative
call and the two thousand classified documents cost eleven paise between
them. That is the argument for the content-hash cache, and it should be
readable off the number rather than asserted next to it.

Tokens are counted in millions because that is the unit the price list is
published in; converting the price to per-token instead would introduce
rounding at the tenth decimal place for no reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from semantic_layer.schema import ModelPrice, SemanticLayer, get_semantic_layer

#: Tokens per unit of the published price. Not a threshold — it is what
#: "per MTok" means.
TOKENS_PER_PRICED_UNIT = 1_000_000


@dataclass(frozen=True)
class Usage:
    """Token counts for one model call, as the API reported them.

    Built from an `llm.provider.LLMResponse` by `Usage.from_response`, so
    the counts are the provider's and never an estimate of our own.
    """

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_response(cls, response) -> Usage:
        """Read the usage off an `LLMResponse`.

        Untyped on purpose: `telemetry` does not import `llm`, so that the
        cost accounting can be exercised without a provider and so the
        dependency runs one way only.
        """
        return cls(
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_input_tokens=response.cache_read_input_tokens,
            cache_creation_input_tokens=response.cache_creation_input_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


def estimate_usage(
    response: Any,
    *,
    request: Any = None,
    layer: SemanticLayer | None = None,
) -> Usage:
    """Token counts guessed from text, for a response that reported none.

    A replayed fixture bought no tokens, so `MockProvider` returns the
    counts that were recorded — zero, for a fixture written by hand. An
    offline demo priced off those zeros would report that a case cost
    nothing, which is true about the replay and a lie about the system.

    So this reads the text and divides by `estimation.chars_per_token`.
    It is coarse, it is meant to be, and every row it contributes to is
    flagged `cost_estimated`. `reported_usage()` decides which of the two
    a caller gets, and never mixes them within one call.
    """
    layer = layer or get_semantic_layer()
    spec = layer.telemetry.estimation

    prompt = ""
    if request is not None:
        prompt = (request.system or "") + "".join(
            message.content for message in request.messages
        )

    return Usage(
        model=response.model,
        input_tokens=spec.tokens_in(prompt),
        output_tokens=spec.tokens_in(getattr(response, "text", "")),
    )


def reported_usage(
    response: Any,
    *,
    request: Any = None,
    layer: SemanticLayer | None = None,
) -> tuple[Usage, bool]:
    """`(usage, estimated)` — the provider's counts, or a guess at them.

    The provider's counts win whenever it reported any. Only a response
    that reports nothing at all falls through to the estimate, and the
    boolean says which happened so the row can record it.
    """
    usage = Usage.from_response(response)
    if usage.total_tokens:
        return usage, False
    return estimate_usage(response, request=request, layer=layer), True


@dataclass(frozen=True)
class CallCost:
    """One call, priced, with the four components kept apart."""

    model: str
    input_usd: float
    output_usd: float
    cache_read_usd: float
    cache_write_usd: float
    usd_to_inr: float

    @property
    def usd(self) -> float:
        return self.input_usd + self.output_usd + self.cache_read_usd + self.cache_write_usd

    @property
    def inr(self) -> float:
        return self.usd * self.usd_to_inr

    def render(self) -> str:
        return f"{self.model}: INR {self.inr:.4f} (USD {self.usd:.6f})"


def price_of(model: str, layer: SemanticLayer | None = None) -> ModelPrice:
    """The published rates for `model`. Raises if it is not priced."""
    layer = layer or get_semantic_layer()
    return layer.telemetry.price(model)


def cost_of_call(usage: Usage, layer: SemanticLayer | None = None) -> CallCost:
    """Price one call against the loaded price list."""
    layer = layer or get_semantic_layer()
    price = layer.telemetry.price(usage.model)
    rate = layer.telemetry.currency.usd_to_inr

    def usd(tokens: int, per_mtok: float) -> float:
        return tokens / TOKENS_PER_PRICED_UNIT * per_mtok

    return CallCost(
        model=usage.model,
        input_usd=usd(usage.input_tokens, price.input_usd_per_mtok),
        output_usd=usd(usage.output_tokens, price.output_usd_per_mtok),
        cache_read_usd=usd(usage.cache_read_input_tokens, price.cache_read_usd_per_mtok),
        cache_write_usd=usd(
            usage.cache_creation_input_tokens, price.cache_write_usd_per_mtok
        ),
        usd_to_inr=rate,
    )


def cost_of_case(
    usages: Iterable[Usage], layer: SemanticLayer | None = None
) -> tuple[float, tuple[CallCost, ...]]:
    """Total rupees for a case, and the per-call breakdown behind it."""
    calls = tuple(cost_of_call(usage, layer) for usage in usages)
    return sum(call.inr for call in calls), calls


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """A measured cost against the published ceiling."""

    measured_inr: float
    ceiling_inr: float

    @property
    def within(self) -> bool:
        return self.measured_inr <= self.ceiling_inr

    @property
    def headroom_inr(self) -> float:
        return self.ceiling_inr - self.measured_inr

    @property
    def utilisation(self) -> float:
        """Share of the ceiling used. Above 1.0 is a breach."""
        return self.measured_inr / self.ceiling_inr if self.ceiling_inr else 0.0

    def render(self) -> str:
        verb = "within" if self.within else "OVER"
        return (
            f"cost per case: INR {self.measured_inr:.2f} — {verb} the INR "
            f"{self.ceiling_inr:.2f} ceiling ({self.utilisation:.0%})"
        )


def check_cost(measured_inr: float, layer: SemanticLayer | None = None) -> Budget:
    """Measured cost per case against `targets.cost_per_case_inr`."""
    layer = layer or get_semantic_layer()
    return Budget(
        measured_inr=measured_inr, ceiling_inr=layer.telemetry.targets.cost_per_case_inr
    )


def mean_cost_per_request(rows: Sequence) -> float:
    """Mean `estimated_cost_inr` over telemetry rows. 0.0 over none.

    Takes anything with an `estimated_cost_inr` attribute, so it reads a
    `RequestTelemetry` and a row fetched back out of the warehouse alike.
    """
    costs = [float(row.estimated_cost_inr) for row in rows]
    return sum(costs) / len(costs) if costs else 0.0


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Projection:
    """Measured cost per interaction, multiplied by an assumed volume.

    Both halves are named on the object because the answer is only as
    good as the assumption, and the assumption should travel with it. The
    volume is `semantic_layer/telemetry.yaml -> projection`; the cost is
    whatever the telemetry rows actually recorded.
    """

    cost_per_interaction_inr: float
    interactions_per_week: int
    weeks_per_year: float
    months_per_year: int
    basis: str

    @property
    def weekly_inr(self) -> float:
        return self.cost_per_interaction_inr * self.interactions_per_week

    @property
    def annual_inr(self) -> float:
        return self.weekly_inr * self.weeks_per_year

    @property
    def monthly_inr(self) -> float:
        """A twelfth of the year, not four weeks — see the note in the YAML."""
        return self.annual_inr / self.months_per_year

    @property
    def annual_inr_lakh(self) -> float:
        return self.annual_inr / 100_000

    @property
    def interactions_per_year(self) -> float:
        return self.interactions_per_week * self.weeks_per_year

    def render(self) -> str:
        return (
            f"{self.interactions_per_week:,}/week at INR "
            f"{self.cost_per_interaction_inr:.2f} per interaction "
            f"({self.basis}) -> INR {self.weekly_inr:,.0f}/week · "
            f"INR {self.monthly_inr:,.0f}/month · "
            f"INR {self.annual_inr_lakh:,.1f} L/year"
        )


def project(
    cost_per_interaction_inr: float,
    *,
    layer: SemanticLayer | None = None,
    interactions_per_week: int | None = None,
    basis: str = "measured",
) -> Projection:
    """What the model spend comes to at the assumed weekly volume.

    `interactions_per_week` defaults to the 10,000 in the semantic layer.
    Override it to answer "and at three times that?" without editing the
    contract every business-development conversation.
    """
    layer = layer or get_semantic_layer()
    spec = layer.telemetry.projection
    return Projection(
        cost_per_interaction_inr=cost_per_interaction_inr,
        interactions_per_week=(
            spec.interactions_per_week
            if interactions_per_week is None
            else interactions_per_week
        ),
        weeks_per_year=spec.weeks_per_year,
        months_per_year=spec.months_per_year,
        basis=basis,
    )


def project_measured(
    rows: Sequence,
    *,
    layer: SemanticLayer | None = None,
    interactions_per_week: int | None = None,
) -> Projection:
    """Project from recorded rows rather than from a figure typed in.

    This is the form to quote. `project()` will happily scale a number
    somebody made up; this one can only scale what was measured, and says
    how many requests it was measured over.
    """
    return project(
        mean_cost_per_request(rows),
        layer=layer,
        interactions_per_week=interactions_per_week,
        basis=f"mean of {len(rows)} recorded requests",
    )


def project_at_ceiling(
    *,
    layer: SemanticLayer | None = None,
    interactions_per_week: int | None = None,
) -> Projection:
    """The worst case the published target permits.

    Every case running right at the INR 6 ceiling. Useful as the upper
    bound in a procurement conversation: whatever the measured figure is,
    the contract cannot cost more than this without breaching a target we
    have already published.
    """
    layer = layer or get_semantic_layer()
    return project(
        layer.telemetry.targets.cost_per_case_inr,
        layer=layer,
        interactions_per_week=interactions_per_week,
        basis="every case at the published ceiling",
    )


__all__ = [
    "TOKENS_PER_PRICED_UNIT",
    "Budget",
    "CallCost",
    "Projection",
    "Usage",
    "check_cost",
    "cost_of_call",
    "cost_of_case",
    "estimate_usage",
    "mean_cost_per_request",
    "price_of",
    "project",
    "project_at_ceiling",
    "project_measured",
    "reported_usage",
]
