"""What an action recovered, against what the playbook promised.

A recovery curve says an availability intervention gets back between 71%
and 96% of the attributed loss in eight weeks, fitted on three prior
cases. This module records the fourth, the fifth and the sixth, and moves
the curve towards them.

A BLEND, NOT A REFIT, AND THE REASON IS WORTH BEING PLAIN ABOUT.
`recovery_curves.yaml` carries p25, p75 and a sample size — the summary of
a fit, not the realisations it was fitted on. Those were never stored, so
there is no data to recompute a quantile against. What CAN be done
honestly is a weighted move towards what we now observe, using the
declared sample size as the weight of the past:

    updated = (n_declared x declared + n_observed x observed) / (n_declared + n_observed)

When the raw realisations of the seeded cases exist, this becomes a real
quantile refit and this note comes out. Until then, calling it a refit
would be claiming a rigour the inputs do not support.

WHAT IS EXCLUDED, AND WHY IT IS STILL RECORDED. A case that recovered
three times the loss it was attributed did not have a miraculous
playbook; something else moved. `implausible_share_above` flags those,
they stay in `recovery_realisation` where anyone can see them, and they
are kept out of the update. Silently dropping them would hide a signal
that usually means the attribution was wrong.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb

from engine.learn.rows import fetch_dicts
from semantic_layer.schema import SemanticLayer, get_semantic_layer

REALISATION_INSERT = (
    "INSERT INTO recovery_realisation ("
    "realisation_id, case_id, playbook, curve_ref, attributable_inr, "
    "expected_low_inr, expected_high_inr, realised_inr, horizon_weeks, "
    "realised_share, implausible, recorded_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

REALISATIONS_FOR_CURVE = (
    "SELECT realised_share, implausible FROM recovery_realisation WHERE curve_ref = ?"
)

REALISATIONS_SQL = (
    "SELECT realisation_id, case_id, playbook, curve_ref, attributable_inr, "
    "expected_low_inr, expected_high_inr, realised_inr, horizon_weeks, "
    "realised_share, implausible, recorded_at FROM recovery_realisation "
    "ORDER BY recorded_at"
)


class CurveError(ValueError):
    """The realisation cannot be recorded as given."""


@dataclass(frozen=True)
class Realisation:
    """One action, and what it actually got back."""

    realisation_id: str
    case_id: str
    playbook: str
    curve_ref: str
    attributable_inr: float
    expected_low_inr: float
    expected_high_inr: float
    realised_inr: float
    horizon_weeks: int
    realised_share: float
    implausible: bool
    recorded_at: datetime

    def as_row(self) -> tuple:
        from engine.db import as_stored_timestamp

        return (
            self.realisation_id,
            self.case_id,
            self.playbook,
            self.curve_ref,
            self.attributable_inr,
            self.expected_low_inr,
            self.expected_high_inr,
            self.realised_inr,
            self.horizon_weeks,
            self.realised_share,
            self.implausible,
            as_stored_timestamp(self.recorded_at),
        )


@dataclass(frozen=True)
class CurveUpdate:
    """A curve, before and after, with what moved it."""

    curve_ref: str
    declared_p25: float
    declared_p75: float
    declared_sample_size: int
    observed_p25: float | None
    observed_p75: float | None
    observations: int
    excluded: int
    updated_p25: float
    updated_p75: float

    @property
    def moved(self) -> bool:
        return (
            self.updated_p25 != self.declared_p25
            or self.updated_p75 != self.declared_p75
        )

    @property
    def sample_size(self) -> int:
        """What the curve would now claim to be fitted on."""
        return self.declared_sample_size + self.observations

    def render(self) -> str:
        if not self.moved:
            return (
                f"{self.curve_ref}: unchanged "
                f"({self.observations} realisations, minimum not met)"
            )
        return (
            f"{self.curve_ref}: p25 {self.declared_p25:.2f} -> {self.updated_p25:.2f}, "
            f"p75 {self.declared_p75:.2f} -> {self.updated_p75:.2f} "
            f"over {self.observations} realisations "
            f"({self.excluded} excluded as implausible)"
        )


def record_realisation(
    connection: duckdb.DuckDBPyConnection,
    *,
    case_id: str,
    playbook: str,
    curve_ref: str,
    attributable_inr: float,
    expected_low_inr: float,
    expected_high_inr: float,
    realised_inr: float,
    horizon_weeks: int,
    at: datetime | None = None,
    layer: SemanticLayer | None = None,
) -> Realisation:
    """Record what one action recovered.

    `realised_share` is realised over ATTRIBUTABLE, not over the expected
    band — that is the unit the curve is quoted in, so an update compares
    like with like.
    """
    layer = layer or get_semantic_layer()
    if attributable_inr <= 0.0:
        raise CurveError(
            "attributable loss must be positive; a recovery share against zero "
            "attributed money is not a number about the playbook"
        )
    share = realised_inr / attributable_inr
    realisation = Realisation(
        realisation_id=uuid.uuid4().hex,
        case_id=case_id,
        playbook=playbook,
        curve_ref=curve_ref,
        attributable_inr=attributable_inr,
        expected_low_inr=expected_low_inr,
        expected_high_inr=expected_high_inr,
        realised_inr=realised_inr,
        horizon_weeks=horizon_weeks,
        realised_share=share,
        implausible=share > layer.learning.recovery.implausible_share_above,
        recorded_at=at or datetime.now(UTC),
    )
    connection.execute(REALISATION_INSERT, list(realisation.as_row()))
    return realisation


def shares_for(
    connection: duckdb.DuckDBPyConnection, curve_ref: str
) -> tuple[tuple[float, ...], int]:
    """`(usable shares, how many were excluded as implausible)`."""
    rows = fetch_dicts(connection, REALISATIONS_FOR_CURVE, [curve_ref])
    usable = tuple(
        float(row["realised_share"]) for row in rows if not row["implausible"]
    )
    return usable, sum(1 for row in rows if row["implausible"])


def update_for(
    connection: duckdb.DuckDBPyConnection,
    curve_ref: str,
    *,
    layer: SemanticLayer | None = None,
) -> CurveUpdate:
    """Where `curve_ref` now stands, given everything realised against it.

    Computed, never stored. A stored curve is a number that goes stale the
    moment either input changes, and both inputs — the declared curve and
    the realisations — are things people edit.
    """
    layer = layer or get_semantic_layer()
    curve = layer.recovery_curves.curves.get(curve_ref)
    if curve is None:
        raise CurveError(
            f"no recovery curve {curve_ref!r}; recovery_curves.yaml declares "
            f"{sorted(layer.recovery_curves.curves)}"
        )
    spec = layer.learning.recovery
    usable, excluded = shares_for(connection, curve_ref)

    observed_p25 = observed_p75 = None
    if usable:
        ordered = sorted(usable)
        observed_p25 = _quantile(ordered, spec.quantiles.low)
        observed_p75 = _quantile(ordered, spec.quantiles.high)

    updated_p25 = curve.p25
    updated_p75 = curve.p75
    if observed_p25 is not None and observed_p75 is not None:
        updated_p25 = spec.blend(curve.p25, curve.sample_size, observed_p25, len(usable))
        updated_p75 = spec.blend(curve.p75, curve.sample_size, observed_p75, len(usable))
        # A blend can invert the quartiles when the observations are
        # tight and the caps bind asymmetrically. Ordered rather than
        # published inverted: a curve whose p25 exceeds its p75 would
        # quote a band with a negative width.
        updated_p25, updated_p75 = min(updated_p25, updated_p75), max(
            updated_p25, updated_p75
        )

    return CurveUpdate(
        curve_ref=curve_ref,
        declared_p25=curve.p25,
        declared_p75=curve.p75,
        declared_sample_size=curve.sample_size,
        observed_p25=observed_p25,
        observed_p75=observed_p75,
        observations=len(usable),
        excluded=excluded,
        updated_p25=updated_p25,
        updated_p75=updated_p75,
    )


def all_updates(
    connection: duckdb.DuckDBPyConnection, layer: SemanticLayer | None = None
) -> tuple[CurveUpdate, ...]:
    """Every declared curve, with where it now stands."""
    layer = layer or get_semantic_layer()
    return tuple(
        update_for(connection, name, layer=layer)
        for name in sorted(layer.recovery_curves.curves)
    )


def _quantile(ordered: list[float], q: float) -> float:
    """Linear-interpolated quantile over an already-sorted list.

    Written out rather than reached for, so the interpolation behind a
    published band is readable at the point the band is computed.
    """
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


__all__ = [
    "REALISATION_INSERT",
    "CurveError",
    "CurveUpdate",
    "Realisation",
    "all_updates",
    "record_realisation",
    "shares_for",
    "update_for",
]
