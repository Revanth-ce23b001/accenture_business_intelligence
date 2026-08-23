"""Contribution — WHERE the change concentrated, and nothing about WHY.

CLAUDE.md rule 7: contribution is labelled WHERE, causation is labelled
WHY, they are separate modules and separate panels, and they are never
merged. The separation is enforced statically here, not by convention:
`engine/contribution/` may not import `engine/adjudicate/`.

The arithmetic requirement is one line: the three components sum to the
total movement within INR 1 lakh. LMDI gives it exactly, which is why it
was chosen over a price-times-volume split — that leaves an interaction
term somebody has to allocate, and whichever rule they pick becomes an
argument.
"""

from __future__ import annotations

import ast
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.contribution import (
    COMPONENTS,
    HEADER,
    MIX,
    PRICE,
    TOLERANCE_INR_LAKH,
    VOLUME,
    ContributionError,
    contribution,
    decompose,
    log_mean,
)
from engine.db import GovernanceError
from security.policy import User

ANALYST = User(user_id="U008", persona="analyst", display_name="Meera Joshi")
LAKH = 1e5
CRORE = 1e7


@pytest.fixture(scope="module")
def west(warehouse):
    """November against October in West — case #2451's movement."""
    return contribution(
        warehouse, ANALYST,
        kpi="net_revenue", scope="West",
        period="2025-11", comparison_period="2025-10",
        period_start=date(2025, 11, 1), period_end=date(2025, 11, 30),
        comparison_start=date(2025, 10, 1), comparison_end=date(2025, 10, 31),
    )


# ===========================================================================
# The header is the guard, so it is checked like a number
# ===========================================================================


def test_the_header_is_verbatim(west):
    assert HEADER == "WHERE the change concentrated — not WHY it happened"
    assert west.header == HEADER
    assert west.decomposition.header == HEADER


def test_contribution_cannot_import_adjudicate():
    """Rule 7, statically.

    The moment a decomposition can see a hypothesis, somebody reads
    "price contributed -2.1 pt" as "the price rise caused it", and those
    are different claims about the world.
    """
    offenders: list[str] = []
    for path in sorted((Path("engine") / "contribution").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if "adjudicate" in name:
                    offenders.append(f"{path}:{node.lineno} imports {name}")
    assert offenders == [], "\n".join(offenders)


def test_no_component_is_described_as_a_cause(west):
    """No causal connective may appear beside a contribution figure."""
    forbidden = ("because", "caused by", "due to", "explains", "the reason")
    for evidence in west.evidence:
        text = f"{evidence.label} {evidence.method} {evidence.notes or ''}".lower()
        for word in forbidden:
            assert word not in text, f"{evidence.evidence_id}: {word!r}"


# ===========================================================================
# The arithmetic: the components sum to the movement
# ===========================================================================


def test_the_components_reconcile_to_within_one_lakh(west):
    decomposition = west.decomposition
    assert west.reconciles
    assert abs(decomposition.residual_inr) <= TOLERANCE_INR_LAKH * LAKH
    assert decomposition.sum_inr == pytest.approx(decomposition.total_inr, abs=LAKH)


def test_all_three_components_are_present_and_named(west):
    names = tuple(item.name for item in west.decomposition.components)
    assert names == COMPONENTS == (PRICE, VOLUME, MIX)


def test_the_movement_is_the_registry_movement(west):
    """INR 83.70 Cr to INR 76.92 Cr: a INR 6.78 Cr fall, -8.1%."""
    decomposition = west.decomposition
    assert decomposition.base_inr / CRORE == pytest.approx(83.70, abs=0.01)
    assert decomposition.total_inr / CRORE == pytest.approx(-6.78, abs=0.01)
    assert decomposition.total_pt == pytest.approx(-8.1, abs=0.05)


def test_the_movement_sits_in_volume(west):
    """Which is WHERE it sits, and says nothing about what moved it.

    An availability fault suppresses units and leaves price alone, so this
    is consistent with the stock-out — and equally consistent with a
    demand collapse, a competitor opening and a bus strike. The
    decomposition cannot tell them apart and does not try.
    """
    decomposition = west.decomposition
    volume = decomposition.component(VOLUME)
    assert volume.value_inr < 0
    assert abs(volume.value_inr) > 0.9 * abs(decomposition.total_inr)
    assert volume.direction == "down"


# ===========================================================================
# Evidence — rules 1, 3 and 4 hold here too
# ===========================================================================


def test_every_component_emits_evidence(west):
    ids = {evidence.evidence_id for evidence in west.evidence}
    assert ids == {
        "contribution.total", "contribution.price",
        "contribution.volume", "contribution.mix",
    }


def test_no_bare_float_leaves_the_module(west):
    for evidence in west.evidence:
        assert evidence.produced_by == "code"
        assert evidence.unit
        assert evidence.source_as_of is not None
        assert evidence.lineage


def test_a_persona_outside_the_contract_is_refused(warehouse):
    with pytest.raises(GovernanceError):
        contribution(
            warehouse, User(user_id="U999", persona="intern"),
            kpi="net_revenue", scope="West",
            period="2025-11", comparison_period="2025-10",
            period_start=date(2025, 11, 1), period_end=date(2025, 11, 30),
            comparison_start=date(2025, 10, 1), comparison_end=date(2025, 10, 31),
        )


# ===========================================================================
# The maths, on its own
# ===========================================================================


def test_log_mean_of_equal_values_is_the_value():
    assert log_mean(5.0, 5.0) == pytest.approx(5.0)


def test_log_mean_sits_between_its_arguments():
    value = log_mean(10.0, 20.0)
    assert 10.0 < value < 20.0
    assert value == pytest.approx((20.0 - 10.0) / (math.log(20.0) - math.log(10.0)))


def test_log_mean_is_symmetric():
    assert log_mean(3.0, 9.0) == pytest.approx(log_mean(9.0, 3.0))


def test_log_mean_of_a_missing_period_is_zero():
    """A SKU with no revenue in one period has no ratio to take."""
    assert log_mean(0.0, 5.0) == 0.0
    assert log_mean(5.0, 0.0) == 0.0


def _frame(units, prices):
    units = np.asarray(units, float)
    prices = np.asarray(prices, float)
    return pd.DataFrame({
        "sku_id": [f"K{i:02d}" for i in range(len(units))],
        "units": units,
        "revenue": units * prices,
    })


def test_a_pure_price_move_lands_entirely_in_price():
    before = _frame([100, 200, 300], [10.0, 20.0, 30.0])
    after = _frame([100, 200, 300], [11.0, 22.0, 33.0])
    split = decompose(before, after)
    assert split.component(PRICE).value_inr == pytest.approx(split.total_inr, rel=1e-9)
    assert split.component(VOLUME).value_inr == pytest.approx(0.0, abs=1e-6)
    assert split.component(MIX).value_inr == pytest.approx(0.0, abs=1e-6)


def test_a_proportional_volume_move_lands_entirely_in_volume():
    before = _frame([100, 200, 300], [10.0, 20.0, 30.0])
    after = _frame([90, 180, 270], [10.0, 20.0, 30.0])
    split = decompose(before, after)
    assert split.component(VOLUME).value_inr == pytest.approx(split.total_inr, rel=1e-9)
    assert split.component(PRICE).value_inr == pytest.approx(0.0, abs=1e-6)
    assert split.component(MIX).value_inr == pytest.approx(0.0, abs=1e-6)


def test_a_shift_between_skus_at_constant_total_volume_lands_in_mix():
    """Same units sold, same prices, a different basket."""
    before = _frame([300, 100], [10.0, 30.0])
    after = _frame([100, 300], [10.0, 30.0])
    split = decompose(before, after)
    assert split.component(MIX).value_inr != pytest.approx(0.0, abs=1.0)
    assert split.component(VOLUME).value_inr == pytest.approx(0.0, abs=1e-6)
    assert split.component(PRICE).value_inr == pytest.approx(0.0, abs=1e-6)


def test_the_identity_holds_on_random_baskets():
    """LMDI's whole point: no interaction term to allocate."""
    rng = np.random.default_rng(20260822)
    for _ in range(25):
        size = int(rng.integers(5, 40))
        units_0 = rng.integers(50, 5000, size)
        units_1 = rng.integers(50, 5000, size)
        prices_0 = rng.uniform(100.0, 5000.0, size)
        prices_1 = prices_0 * rng.uniform(0.8, 1.2, size)
        split = decompose(_frame(units_0, prices_0), _frame(units_1, prices_1))
        assert split.reconciles(TOLERANCE_INR_LAKH * LAKH)


def test_a_sku_present_in_one_period_only_still_reconciles():
    """It carries no log ratio, so it lands in the total and in no
    component. That is why the tolerance is a tolerance and not zero."""
    before = _frame([100, 200], [10.0, 20.0])
    after = _frame([100, 200, 50], [10.0, 20.0, 40.0])
    split = decompose(before, after)
    assert split.skus == 3
    assert abs(split.residual_inr) <= TOLERANCE_INR_LAKH * LAKH


def test_a_frame_missing_a_column_is_refused():
    before = _frame([100], [10.0]).drop(columns=["units"])
    with pytest.raises(ContributionError, match="units"):
        decompose(before, _frame([100], [10.0]))


def test_a_period_that_sold_nothing_is_refused():
    with pytest.raises(ContributionError, match="no ratio"):
        decompose(_frame([0], [10.0]), _frame([100], [10.0]))
