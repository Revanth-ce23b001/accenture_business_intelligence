"""WHERE the change concentrated — not WHY it happened.

CLAUDE.md rule 7: contribution is labelled WHERE, causation is labelled
WHY. They are separate modules, separate panels, never merged. This module
does not import from `engine/adjudicate/` and a static test enforces it,
because the moment a decomposition can see a hypothesis somebody will read
"price contributed -2.1 pt" as "the price rise caused it", and those are
different claims.

A contribution says: of the movement that happened, this much of it sits
in price, this much in volume, this much in mix. It says nothing about
what moved any of them. A price component of -2.1 pt is equally consistent
with a deliberate price cut, a discount war, and a change in what people
bought — the decomposition cannot tell them apart and does not try.

THE METHOD is the log-mean Divisia index, LMDI-I. It is chosen for one
property: the components sum EXACTLY to the total change, with no residual
term to explain away. A naive price-times-volume split leaves an
interaction term that has to be allocated by a rule nobody agrees on, and
whichever rule you pick becomes an argument. LMDI has no such term.

For revenue R = sum over SKUs of price_i x quantity_i, the change from
period 0 to period 1 decomposes as

    dR = dR_price + dR_volume + dR_mix

where each component sums the log-mean weight of that SKU times the log
change in its price, total volume, and share of volume respectively. The
log-mean weight

    L(a, b) = (a - b) / (ln a - ln b),   L(a, a) = a

is what makes the identity exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from semantic_layer.schema import get_semantic_layer

#: The three components, in the order they render.
PRICE = "price"
VOLUME = "volume"
MIX = "mix"
COMPONENTS = (PRICE, VOLUME, MIX)

#: The header this decomposition renders under, verbatim. It is not a
#: caption — it is the guard against a reader taking WHERE for WHY.
HEADER = "WHERE the change concentrated — not WHY it happened"


class ContributionError(ValueError):
    """The decomposition cannot be computed from what was given."""


def log_mean(a: float, b: float) -> float:
    """L(a, b) = (a - b) / (ln a - ln b), and L(a, a) = a.

    The weight that makes the LMDI identity exact. Zero when either side is
    zero: a SKU that did not exist in one period contributes nothing to a
    ratio it has no denominator for.
    """
    if a <= 0.0 or b <= 0.0:
        return 0.0
    if math.isclose(a, b):
        return a
    return (a - b) / (math.log(a) - math.log(b))


@dataclass(frozen=True)
class Component:
    """One of the three, in rupees and in points of the base period."""

    name: str
    value_inr: float
    share_pt: float

    @property
    def direction(self) -> str:
        return "up" if self.value_inr > 0 else "down"


@dataclass(frozen=True)
class Decomposition:
    """The whole movement, split three ways and summing to itself."""

    total_inr: float
    base_inr: float
    components: tuple[Component, ...]
    skus: int
    header: str = HEADER

    def component(self, name: str) -> Component:
        for item in self.components:
            if item.name == name:
                return item
        raise KeyError(f"no {name!r} component")

    @property
    def sum_inr(self) -> float:
        return sum(item.value_inr for item in self.components)

    @property
    def residual_inr(self) -> float:
        """What the three components fail to account for. LMDI leaves none."""
        return self.total_inr - self.sum_inr

    def reconciles(self, tolerance_inr: float) -> bool:
        return abs(self.residual_inr) <= tolerance_inr

    @property
    def total_pt(self) -> float:
        if not self.base_inr:
            return 0.0
        return self.total_inr / self.base_inr * get_semantic_layer().warehouse.units.percent_scale


def decompose(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    key: str = "sku_id",
    revenue: str = "revenue",
    units: str = "units",
) -> Decomposition:
    """Split a revenue movement into price, volume and mix.

    `before` and `after` are one row per SKU with a revenue and a unit
    count. A SKU present in one period and not the other carries no log
    ratio, so it contributes to the total and to no component — which is
    why the reconciliation tolerance is a tolerance rather than zero.
    """
    for frame, label in ((before, "before"), (after, "after")):
        missing = {key, revenue, units} - set(frame.columns)
        if missing:
            raise ContributionError(f"the {label} frame is missing {sorted(missing)}")

    joined = before.merge(after, on=key, how="outer", suffixes=("_0", "_1")).fillna(0.0)
    revenue_0 = joined[f"{revenue}_0"].to_numpy(float)
    revenue_1 = joined[f"{revenue}_1"].to_numpy(float)
    units_0 = joined[f"{units}_0"].to_numpy(float)
    units_1 = joined[f"{units}_1"].to_numpy(float)

    total_units_0 = float(units_0.sum())
    total_units_1 = float(units_1.sum())
    if total_units_0 <= 0 or total_units_1 <= 0:
        raise ContributionError("one of the periods sold nothing; there is no ratio to take")

    base = float(revenue_0.sum())
    total = float(revenue_1.sum()) - base

    with np.errstate(divide="ignore", invalid="ignore"):
        price_0 = np.where(units_0 > 0, revenue_0 / units_0, 0.0)
        price_1 = np.where(units_1 > 0, revenue_1 / units_1, 0.0)
        share_0 = units_0 / total_units_0
        share_1 = units_1 / total_units_1

    price_effect = 0.0
    volume_effect = 0.0
    mix_effect = 0.0
    volume_ratio = math.log(total_units_1 / total_units_0)

    for index in range(len(joined)):
        weight = log_mean(revenue_1[index], revenue_0[index])
        if weight <= 0.0:
            continue
        if price_0[index] > 0 and price_1[index] > 0:
            price_effect += weight * math.log(price_1[index] / price_0[index])
        volume_effect += weight * volume_ratio
        if share_0[index] > 0 and share_1[index] > 0:
            mix_effect += weight * math.log(share_1[index] / share_0[index])

    percent = get_semantic_layer().warehouse.units.percent_scale
    components = tuple(
        Component(
            name=name,
            value_inr=value,
            share_pt=(value / base * percent) if base else 0.0,
        )
        for name, value in (
            (PRICE, price_effect), (VOLUME, volume_effect), (MIX, mix_effect)
        )
    )
    return Decomposition(
        total_inr=total, base_inr=base, components=components, skus=len(joined)
    )


__all__ = [
    "COMPONENTS",
    "HEADER",
    "MIX",
    "PRICE",
    "VOLUME",
    "Component",
    "ContributionError",
    "Decomposition",
    "decompose",
    "log_mean",
]
