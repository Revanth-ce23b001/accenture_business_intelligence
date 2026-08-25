"""Expected recovery — a band, from a curve, over prior comparable cases.

    expected recovery = attributable money  x  the curve's p25 and p75

Three things about that line are deliberate.

IT IS A RANGE. A point estimate of recovery is a claim nobody can keep.
The spread between p25 and p75 is the honest part of the answer, and it
is what the p25/p75 pair exists to carry.

IT IS BOUNDED BY WHAT WE ATTRIBUTED, not by the headline. #2451 loses
INR 6.78 Cr; INR 3.24 Cr of it is attributed to the hypothesis this
action addresses; the action is credited against that and nothing more.
Recovery that exceeded the attributable loss was Round 1's defect 5, and
the recovery curve's own loader refuses a p75 above 1.0 so it cannot come
back.

THE LABEL COMES FROM THE SAMPLE SIZE, not from the width of the band.
Three closed cases is Medium however tight the quartiles look. A narrow
interval fitted on three observations is not precision, it is three
observations, and the label is the only place that says so.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_layer.schema import Playbook, RecoveryCurve, SemanticLayer

from engine.recommend.cost import Cost


class RecoveryError(RuntimeError):
    """The playbook's recovery cannot be estimated for this case."""


@dataclass(frozen=True)
class Recovery:
    """The band, the curve behind it, and the ROI it implies."""

    curve_ref: str
    p25: float
    p75: float
    horizon_weeks: int
    sample_size: int
    confidence: str
    attributable_inr: float
    low_inr: float
    high_inr: float
    roi_low: float | None
    roi_high: float | None
    informational: bool
    basis: str

    @property
    def recovers(self) -> bool:
        """False for an information lever: it buys an answer, not revenue."""
        return not self.informational and self.high_inr > 0.0

    def low_crore(self, layer: SemanticLayer) -> float:
        return self.low_inr / layer.warehouse.units.inr_per_crore

    def high_crore(self, layer: SemanticLayer) -> float:
        return self.high_inr / layer.warehouse.units.inr_per_crore

    def render(self, layer: SemanticLayer) -> str:
        if not self.recovers:
            return (
                f"recovery: none by construction ({self.curve_ref}) — this lever buys "
                "information, and crediting it with revenue would double-count the "
                "action it unblocks."
            )
        line = (
            f"recovery: INR {self.low_crore(layer):.2f}-{self.high_crore(layer):.2f} Cr "
            f"over {self.horizon_weeks} weeks "
            f"({self.p25:.0%}-{self.p75:.0%} of INR "
            f"{self.attributable_inr / layer.warehouse.units.inr_per_crore:.2f} Cr "
            f"attributable), confidence {self.confidence} on n={self.sample_size}"
        )
        if self.roi_low is not None and self.roi_high is not None:
            line += f"; ROI {self.roi_low:.0f}-{self.roi_high:.0f}x"
        return line


def curve_for(playbook: Playbook, layer: SemanticLayer) -> RecoveryCurve:
    curve = layer.recovery_curves.curves.get(playbook.recovery_curve_ref)
    if curve is None:  # pragma: no cover - the loader cross-checks this
        raise RecoveryError(
            f"playbook {playbook.playbook!r} references recovery curve "
            f"{playbook.recovery_curve_ref!r}, which does not exist"
        )
    return curve


def confidence_label(sample_size: int, layer: SemanticLayer) -> str:
    """Low / Medium / High, from how many prior cases fitted the curve."""
    return layer.recommend.recovery.label_for(sample_size)


def compute_recovery(
    playbook: Playbook,
    attributable_inr: float,
    layer: SemanticLayer,
    *,
    cost: Cost | None = None,
) -> Recovery:
    """Apply the playbook's curve to the money this case attributed."""
    spec = layer.recommend
    curve = curve_for(playbook, layer)
    informational = playbook.lever in spec.matching.information_levers

    if attributable_inr < 0.0:
        raise RecoveryError(
            "attributable loss is negative; recovery is quoted against the money the "
            "leading hypothesis accounted for, which is a magnitude"
        )

    low = attributable_inr * curve.p25
    high = attributable_inr * curve.p75

    roi_low = roi_high = None
    suppress = informational and spec.recovery.suppress_roi_for_information_levers
    if cost is not None and cost.priced and not suppress:
        roi_low = low / cost.amount_inr
        roi_high = high / cost.amount_inr

    return Recovery(
        curve_ref=playbook.recovery_curve_ref,
        p25=curve.p25,
        p75=curve.p75,
        horizon_weeks=curve.horizon_weeks,
        sample_size=curve.sample_size,
        confidence=confidence_label(curve.sample_size, layer),
        attributable_inr=attributable_inr,
        low_inr=low,
        high_inr=high,
        roi_low=roi_low,
        roi_high=roi_high,
        informational=informational,
        basis=curve.basis,
    )


__all__ = [
    "Recovery",
    "RecoveryError",
    "compute_recovery",
    "confidence_label",
    "curve_for",
]
