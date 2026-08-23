"""The four caps. Applied after the weighted sum, and not outvotable.

CLAUDE.md §"Confidence specification":

  Test 6 confounder screen fails       min(conf, 0.85)
  A required source is missing         min(conf, 0.45) and force T3
  Temporal precedence failed           conf = 0, hypothesis eliminated
  Evidence dominated by weights < 0.6  min(conf, 0.65)

WHY A CAP AND NOT A COMPONENT. A component is a vote: a low s4 can be
outweighed by a high s1 and a high s6, and the answer comes out
confident anyway. Some conditions must not be outvotable. If the leading
hypothesis needs a feed the organisation does not have, no amount of
statistical strength on the feeds it does have should let the number past
0.45 — the strength is real and it is strength about the wrong thing.
Caps are how "this cannot be argued around" is expressed in arithmetic.

THE CLARIFICATION THAT DECIDES #2451, and it is easy to get wrong. Every
cap is evaluated PER HYPOTHESIS, against the hypothesis being scored.

  The confounder cap asks whether a confounder OF THIS HYPOTHESIS is
  unresolved. On #2451 Test 6 screened promo, weather, competitor action
  and staffing across the treated and control groups and found no
  difference, so it PASSES and the cap does not fire.

  H2 — a competitor promotion — is a COMPETING HYPOTHESIS for the same
  residual. It is unverifiable, it holds INR 0.86 Cr, and it is what makes
  the verdict PARTIALLY EXPLAINED rather than EXPLAINED. That is the
  verdict decision table's job, not this module's. An implementation that
  reads H2's unverifiability as a confounder of H1 produces 0.85 and is
  wrong.

The two questions look similar and are not:

  a confounder      could something else have produced THIS hypothesis's
                    signature in the treated group?
  a competitor      could something else account for the part of the
                    residual this hypothesis does not?
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.adjudicate import HypothesisVerdict
from engine.evidence import can_promote_to_explained
from semantic_layer.schema import SemanticLayer

#: Names, matching `adjudication.yaml -> confidence.caps`. A cap that is
#: not in the layer cannot fire, and a name here that is not there is a
#: load-time error.
CONFOUNDER = "confounder_screen_failed"
MISSING_SOURCE = "required_source_missing"
PRECEDENCE = "temporal_precedence_failed"
LOW_RELIABILITY = "low_reliability_dominated"

ORDER = (PRECEDENCE, MISSING_SOURCE, CONFOUNDER, LOW_RELIABILITY)


@dataclass(frozen=True)
class Cap:
    """A cap that fired, what fired it, and what it cost."""

    name: str
    condition: str
    ceiling: float
    forced_trigger: str | None
    eliminates: bool
    detail: str


def evaluate_caps(
    verdict: HypothesisVerdict,
    evidence,
    layer: SemanticLayer,
) -> tuple[Cap, ...]:
    """Every cap that fires, in the order they are declared.

    Every one that fires is returned, not just the binding one. A case
    file that reported only the lowest ceiling would hide the fact that
    two separate things went wrong.
    """
    specs = layer.adjudication.confidence.caps
    caps: list[Cap] = []

    # --- precedence: not a cap so much as a death certificate -------------
    precedence = verdict.finding(layer.adjudicate.precedence.test_id)
    if precedence is not None and precedence.testable and not precedence.passed:
        spec = specs[PRECEDENCE]
        caps.append(
            Cap(
                PRECEDENCE, spec.condition, spec.ceiling, spec.forced_trigger,
                spec.eliminates_hypothesis,
                f"{precedence.detail} There is no confidence to have in a cause "
                "that arrived after its effect.",
            )
        )

    # --- a source the organisation does not hold --------------------------
    if verdict.missing_sources:
        spec = specs[MISSING_SOURCE]
        names = ", ".join(verdict.missing_sources)
        caps.append(
            Cap(
                MISSING_SOURCE, spec.condition, spec.ceiling, spec.forced_trigger,
                spec.eliminates_hypothesis,
                f"this hypothesis requires {names}, which the organisation does not "
                f"hold. Whatever the held sources say, they are not evidence about "
                f"{names}.",
            )
        )

    # --- a confounder OF THIS HYPOTHESIS, unresolved ----------------------
    unresolved = verdict.unresolved_confounders
    if unresolved:
        spec = specs[CONFOUNDER]
        caps.append(
            Cap(
                CONFOUNDER, spec.condition, spec.ceiling, spec.forced_trigger,
                spec.eliminates_hypothesis,
                f"Test 6 could not rule out {', '.join(unresolved)} as an alternative "
                "producer of this hypothesis's own signature in the treated group.",
            )
        )

    # --- evidence that is mostly text -------------------------------------
    check = can_promote_to_explained(evidence, layer)
    if not check.allowed:
        spec = specs[LOW_RELIABILITY]
        caps.append(
            Cap(
                LOW_RELIABILITY, spec.condition, spec.ceiling, spec.forced_trigger,
                spec.eliminates_hypothesis,
                check.detail,
            )
        )

    order = {name: index for index, name in enumerate(ORDER)}
    return tuple(sorted(caps, key=lambda cap: order.get(cap.name, len(order))))


def apply_caps(raw: float, caps) -> float:
    """The lowest ceiling wins. A cap cannot be outvoted by another cap."""
    value = float(raw)
    for cap in caps:
        value = min(value, cap.ceiling)
    return value


def forced_triggers(caps) -> tuple[str, ...]:
    """Triggers a cap forces regardless of what the trigger itself found.

    A missing required source forces T3. The trigger would normally reach
    the same conclusion on its own; forcing it means the two can never
    disagree, which matters because the UI names the trigger and the cap
    separately and a reader would notice.
    """
    seen: list[str] = []
    for cap in caps:
        if cap.forced_trigger and cap.forced_trigger not in seen:
            seen.append(cap.forced_trigger)
    return tuple(seen)


__all__ = [
    "CONFOUNDER",
    "LOW_RELIABILITY",
    "MISSING_SOURCE",
    "ORDER",
    "PRECEDENCE",
    "Cap",
    "apply_caps",
    "evaluate_caps",
    "forced_triggers",
]
