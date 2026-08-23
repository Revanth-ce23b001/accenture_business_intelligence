"""Fetch the two periods and decompose the movement between them.

Reads through `engine/db.py::execute_governed` like everything else, and
emits `Evidence` like everything else. It knows nothing about hypotheses,
tests or verdicts, and it must stay that way (CLAUDE.md rule 7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import duckdb
import pandas as pd

from engine.contracts import Evidence
from engine.contribution.decomposition import (
    COMPONENTS,
    HEADER,
    ContributionError,
    Decomposition,
    decompose,
)
from engine.db import execute_governed, warehouse_clock
from engine.evidence import EvidenceFactory, EvidenceLedger
from security.policy import User
from semantic_layer.schema import SemanticLayer, get_semantic_layer

EVIDENCE_PREFIX = "contribution"
DERIVED_KIND = "derived_estimate"
QUERY_KIND = "structured_query"
DERIVED = "derived"
POS = "pos"

#: The components must sum to the total within this. CLAUDE.md's own
#: figure: ₹1 lakh on a movement of several crore.
TOLERANCE_INR_LAKH = 1.0

SKU_PERIOD_SQL = """
SELECT
    d.region,
    k.sku_id,
    SUM(k.net_revenue_inr) AS revenue,
    SUM(k.units)           AS units
FROM fact_sales_daily_sku AS k
JOIN dim_store AS d USING (store_id)
WHERE k.txn_date BETWEEN $start AND $end
  AND d.region = $scope
GROUP BY 1, 2
"""


@dataclass(frozen=True)
class ContributionResult:
    """The decomposition, its evidence, and the header it renders under."""

    scope: str
    period: str
    comparison_period: str
    decomposition: Decomposition
    evidence: tuple[Evidence, ...]
    as_of: datetime
    header: str = HEADER

    #: Rupees per lakh, from the layer, so the tolerance means what it says.
    inr_per_lakh: float = 0.0

    @property
    def reconciles(self) -> bool:
        return self.decomposition.reconciles(TOLERANCE_INR_LAKH * self.inr_per_lakh)


def contribution(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    *,
    kpi: str,
    scope: str,
    period: str,
    comparison_period: str,
    period_start: date,
    period_end: date,
    comparison_start: date,
    comparison_end: date,
    layer: SemanticLayer | None = None,
    clock: datetime | None = None,
) -> ContributionResult:
    """Where the movement sits, split into price, volume and mix."""
    layer = layer or get_semantic_layer()
    clock = clock or warehouse_clock(connection, layer)
    units = layer.warehouse.units
    factory = EvidenceFactory.for_stage(EVIDENCE_PREFIX, clock, layer)
    ledger = EvidenceLedger()

    before = _fetch(connection, user, layer, kpi, scope, comparison_start, comparison_end)
    after = _fetch(connection, user, layer, kpi, scope, period_start, period_end)
    if before.empty or after.empty:
        raise ContributionError(
            f"no SKU-level revenue for {scope} in "
            f"{comparison_period if before.empty else period}"
        )

    split = decompose(before, after)

    ledger.add(
        factory.emit(
            "total",
            kind=QUERY_KIND,
            label=f"Movement being decomposed — {scope} {period} against {comparison_period}",
            value=round(split.total_inr / units.inr_per_crore, units.crore_places),
            unit="INR_CR",
            source_system=POS,
            method="sql",
            description=f"SKU-level revenue, {period} less {comparison_period}",
            ref="engine/contribution/decomposition.py",
            inputs=("fact_sales_daily_sku",),
            statement=SKU_PERIOD_SQL.strip(),
            notes=(
                f"{HEADER}. This is a decomposition of the movement, not an explanation "
                "of it: a price component is equally consistent with a deliberate cut, a "
                "discount war and a change in what people bought."
            ),
        )
    )
    for name in COMPONENTS:
        component = split.component(name)
        ledger.add(
            factory.emit(
                name,
                kind=DERIVED_KIND,
                label=f"{name.capitalize()} component — {scope}",
                value=round(component.value_inr / units.inr_per_crore, units.crore_places),
                unit="INR_CR",
                source_system=DERIVED,
                method="difference",
                description=(
                    f"log-mean Divisia {name} effect over {split.skus} SKUs, "
                    f"{component.share_pt:+.2f} pt of {comparison_period}"
                ),
                ref="engine/contribution/decomposition.py::decompose",
                inputs=(f"{EVIDENCE_PREFIX}.total",),
                notes=(
                    "LMDI, so the three components sum to the total exactly. There is no "
                    "interaction term to allocate and therefore nothing to argue about."
                ),
            )
        )

    return ContributionResult(
        scope=scope,
        period=period,
        comparison_period=comparison_period,
        decomposition=split,
        evidence=tuple(ledger),
        as_of=clock,
        inr_per_lakh=units.inr_per_lakh,
    )


def _fetch(connection, user, layer, kpi, scope, start, end) -> pd.DataFrame:
    rows, _filtered, _masked = execute_governed(
        user, kpi, SKU_PERIOD_SQL,
        {"start": start, "end": end, "scope": scope},
        connection=connection, layer=layer, purpose="contribution.sku_period",
    )
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    return frame.groupby("sku_id", as_index=False)[["revenue", "units"]].sum()


__all__ = ["TOLERANCE_INR_LAKH", "ContributionResult", "contribution"]
