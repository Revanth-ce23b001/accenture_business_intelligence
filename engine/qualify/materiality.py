"""Gate 5 — is it worth the owner's afternoon?

The residual, in the KPI contract's own unit, against the limit the
contract's OWNER set. Rupees for a value KPI, percentage points for a rate
KPI; this module does not convert between them and does not second-guess
the number.

Sub-threshold does not mean ignore. It means "not a case": the movement
goes to the weekly digest, where a person can still see it. The difference
matters — a system that silently drops small movements is a system nobody
can audit for what it chose not to tell them.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_layer.schema import KpiContract, MaterialityGate, Units

#: Units a materiality limit can be expressed in, and what the residual has
#: to be converted to in order to be compared with it.
VALUE_UNITS = ("INR_CR", "INR_L", "INR")
RATE_UNITS = ("pt", "pct")


class MaterialityError(RuntimeError):
    """The residual and the limit are not in comparable units."""


@dataclass(frozen=True)
class MaterialityVerdict:
    """The residual against the owner's limit, in the owner's unit."""

    kpi: str
    residual: float
    threshold: float
    unit: str
    display: str
    owner_role: str
    route: str | None

    @property
    def material(self) -> bool:
        return abs(self.residual) >= self.threshold

    @property
    def multiple(self) -> float:
        """How many times the limit the residual is. Zero limit means None."""
        return abs(self.residual) / self.threshold if self.threshold else 0.0


def assess_materiality(
    kpi: KpiContract,
    residual_pt: float,
    residual_inr: float,
    spec: MaterialityGate,
    units: Units,
) -> MaterialityVerdict:
    """Compare the residual with the contract's materiality limit.

    A KPI with no agreed limit cannot open a case at all — that is what
    `can_open_a_case` means — and this raises rather than inventing one.
    """
    limit = kpi.thresholds.materiality
    if limit is None:
        raise MaterialityError(
            f"{kpi.kpi} declares no materiality limit; it is monitoring only and "
            "cannot open a case"
        )

    if limit.unit in VALUE_UNITS:
        residual = abs(residual_inr) / _scale(limit.unit, units)
    elif limit.unit in RATE_UNITS:
        residual = abs(residual_pt)
    else:
        raise MaterialityError(
            f"{kpi.kpi} materiality is in {limit.unit!r}, which is neither a value "
            f"({VALUE_UNITS}) nor a rate ({RATE_UNITS})"
        )

    verdict = MaterialityVerdict(
        kpi=kpi.kpi,
        residual=residual,
        threshold=limit.value,
        unit=limit.unit,
        display=limit.display,
        owner_role=kpi.owner_role,
        route=None,
    )
    if verdict.material:
        return verdict
    return MaterialityVerdict(
        kpi=verdict.kpi,
        residual=verdict.residual,
        threshold=verdict.threshold,
        unit=verdict.unit,
        display=verdict.display,
        owner_role=verdict.owner_role,
        route=spec.sub_threshold_route,
    )


def _scale(unit: str, units: Units) -> float:
    if unit == "INR_CR":
        return units.inr_per_crore
    if unit == "INR_L":
        return units.inr_per_lakh
    return 1.0


__all__ = [
    "RATE_UNITS",
    "VALUE_UNITS",
    "MaterialityError",
    "MaterialityVerdict",
    "assess_materiality",
]
