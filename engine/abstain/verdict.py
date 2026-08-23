"""The verdict decision table. Three outcomes, evaluated strictly in order.

CLAUDE.md §"Verdict decision table":

    IF any trigger T1-T8 fires:
        -> INSUFFICIENT EVIDENCE   (name every trigger that fired)

    ELIF coverage >= 0.70
         AND conf_final >= 0.70
         AND no live-unverifiable hypothesis holds residual > materiality:
        -> EXPLAINED

    ELIF coverage >= 0.30 AND conf_final >= 0.60:
        -> PARTIALLY EXPLAINED
           reason = "coverage below 0.70" OR
                    "live-unverifiable hypothesis above materiality"

    ELSE:
        -> INSUFFICIENT EVIDENCE

THE THIRD CONDITION IS THE ONE THAT MATTERS, and it is what most systems
do not have. Coverage and confidence are both about the hypothesis being
proposed. The third condition is about the hypothesis that is NOT being
proposed: is there a competing explanation that nobody can check, holding
more money than the business has said it cares about?

On #2451 there is. H2, a competitor promotion, cannot be verified because
the organisation holds no competitor pricing feed, and it is left holding
INR 0.86 Cr against a INR 50 L materiality limit. Coverage is 0.79 and
confidence clears its bar, so on the first two conditions the case is
EXPLAINED. The third condition is why it is not, and the reason string
says so on the chip rather than in a footnote.

INSUFFICIENT EVIDENCE ARRIVES TWICE, deliberately, and they are different
statements. The first is "we could not run this properly" — a trigger
fired, and the resolution panel prices what would fix it. The last is "we
ran it properly and it did not reach the bar" — nothing is broken, the
answer is just not good enough to publish. The reason code distinguishes
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from engine.abstain.triggers import Trigger, fired
from engine.contracts import Verdict as VerdictContract
from semantic_layer.schema import SemanticLayer

EXPLAINED = "EXPLAINED"
PARTIALLY_EXPLAINED = "PARTIALLY_EXPLAINED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

#: Reason codes, matching `adjudication.yaml -> verdict.reason_codes`
#: where one exists. `triggers_fired` and `below_all_thresholds` are the
#: two INSUFFICIENT EVIDENCE paths and are named here because the config
#: names reasons for the PARTIALLY branch only.
TRIGGERS_FIRED = "triggers_fired"
BELOW_ALL_THRESHOLDS = "below_all_thresholds"
COVERAGE_BELOW = "coverage_below_threshold"
LIVE_UNVERIFIABLE = "live_unverifiable_above_materiality"


@dataclass(frozen=True)
class HeldResidual:
    """A hypothesis nobody can check, and what it is left holding."""

    hypothesis: str
    label: str
    verifiable: bool
    residual_inr: float
    materiality_inr: float
    missing_sources: tuple[str, ...] = ()

    @property
    def above_materiality(self) -> bool:
        return self.residual_inr > self.materiality_inr

    @property
    def blocks_explained(self) -> bool:
        return (not self.verifiable) and self.above_materiality


@dataclass(frozen=True)
class VerdictDecision:
    """The outcome, the rule that produced it, and the trace."""

    value: str
    reason_code: str | None
    reason_text: str | None
    triggers_fired: tuple[str, ...]
    coverage: float
    confidence: float
    held: tuple[HeldResidual, ...] = ()
    trace: tuple[str, ...] = field(default_factory=tuple)

    @property
    def abstained(self) -> bool:
        return self.value == INSUFFICIENT_EVIDENCE

    def render(self) -> str:
        lines = [f"verdict: {self.value}"]
        if self.reason_text:
            lines.append(f"  reason: {self.reason_text}")
        if self.triggers_fired:
            lines.append(f"  triggers: {', '.join(self.triggers_fired)}")
        lines.extend(f"  {step}" for step in self.trace)
        return "\n".join(lines)


def decide(
    *,
    coverage: float,
    confidence: float,
    triggers: tuple[Trigger, ...],
    held: tuple[HeldResidual, ...],
    layer: SemanticLayer,
) -> VerdictDecision:
    """Walk the table, in order, recording every step that was taken."""
    spec = layer.adjudication.verdict
    names = fired(triggers)
    trace: list[str] = []

    # --- 1. any trigger at all ------------------------------------------
    if names:
        detail = "; ".join(
            f"{item.id} {item.name}" for item in triggers if item.fired
        )
        return VerdictDecision(
            value=INSUFFICIENT_EVIDENCE,
            reason_code=TRIGGERS_FIRED,
            reason_text=detail,
            triggers_fired=names,
            coverage=coverage,
            confidence=confidence,
            held=held,
            trace=(
                f"{len(names)} of {len(triggers)} abstention triggers fired, so the "
                "coverage and confidence thresholds were never reached.",
            ),
        )
    trace.append(f"no abstention trigger fired ({len(triggers)} evaluated).")

    blockers = tuple(item for item in held if item.blocks_explained)

    # --- 2. EXPLAINED, all three conditions ------------------------------
    rule = spec.explained
    coverage_ok = coverage >= rule.min_coverage
    confidence_ok = confidence >= rule.min_confidence
    unblocked = not (blockers and rule.forbid_live_unverifiable_above_materiality)
    trace.append(
        f"coverage {coverage:.2f} {'>=' if coverage_ok else '<'} "
        f"{rule.min_coverage:.2f}; confidence {confidence:.2f} "
        f"{'>=' if confidence_ok else '<'} {rule.min_confidence:.2f}; "
        + (
            "no live unverifiable hypothesis holds residual above materiality."
            if unblocked
            else f"{blockers[0].label} is live, unverifiable and holds residual above "
                 "materiality."
        )
    )
    if coverage_ok and confidence_ok and unblocked:
        return VerdictDecision(
            EXPLAINED, None, None, (), coverage, confidence, held, tuple(trace)
        )

    # --- 3. PARTIALLY EXPLAINED ------------------------------------------
    partial = spec.partially_explained
    if coverage >= partial.min_coverage and confidence >= partial.min_confidence:
        # Which of the two things stopped it being EXPLAINED. Both can be
        # true; the unverifiable holding is named first because it is the
        # substantive one — coverage below the bar is a quantity, and a
        # competing explanation nobody can check is a gap in the argument.
        if blockers:
            code = LIVE_UNVERIFIABLE
            blocker = blockers[0]
            crore = layer.warehouse.units.inr_per_crore
            text = (
                f"{spec.reason_codes[code]}: {blocker.label} cannot be verified "
                f"— {', '.join(blocker.missing_sources) or 'the source is not held'} "
                f"— and holds INR {blocker.residual_inr / crore:.2f} Cr against a "
                f"INR {blocker.materiality_inr / crore:.2f} Cr materiality limit."
            )
        else:
            code = COVERAGE_BELOW
            text = (
                f"{spec.reason_codes[code]}: {coverage:.0%} of the qualified residual "
                "is accounted for."
            )
        return VerdictDecision(
            PARTIALLY_EXPLAINED, code, text, (), coverage, confidence, held, tuple(trace)
        )

    # --- 4. nothing reached the bar --------------------------------------
    shortfalls = []
    if coverage < partial.min_coverage:
        shortfalls.append(f"coverage {coverage:.2f} below {partial.min_coverage:.2f}")
    if confidence < partial.min_confidence:
        shortfalls.append(
            f"confidence {confidence:.2f} below {partial.min_confidence:.2f}"
        )
    return VerdictDecision(
        value=INSUFFICIENT_EVIDENCE,
        reason_code=BELOW_ALL_THRESHOLDS,
        reason_text=(
            "nothing was broken and the answer did not reach the bar: "
            + ", ".join(shortfalls)
            + "."
        ),
        triggers_fired=(),
        coverage=coverage,
        confidence=confidence,
        held=held,
        trace=tuple(trace),
    )


def to_contract(
    decision: VerdictDecision, case_id: str, decided_at: datetime
) -> VerdictContract:
    """The decision as the frozen contract the case file carries."""
    return VerdictContract(
        case_id=case_id,
        value=decision.value,
        reason_code=decision.reason_code,
        reason_text=decision.reason_text,
        triggers_fired=decision.triggers_fired,
        coverage=min(max(decision.coverage, 0.0), 1.0),
        confidence_calibrated=min(max(decision.confidence, 0.0), 1.0),
        decided_at=decided_at,
    )


__all__ = [
    "BELOW_ALL_THRESHOLDS",
    "COVERAGE_BELOW",
    "EXPLAINED",
    "INSUFFICIENT_EVIDENCE",
    "LIVE_UNVERIFIABLE",
    "PARTIALLY_EXPLAINED",
    "TRIGGERS_FIRED",
    "HeldResidual",
    "VerdictDecision",
    "decide",
    "to_contract",
]
