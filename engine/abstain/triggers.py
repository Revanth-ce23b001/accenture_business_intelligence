"""T1 through T8 — eight named booleans, evaluated outside the model.

CLAUDE.md rule 6: abstention is architectural, not a prompt instruction.
No model is asked whether it is sure. Eight deterministic conditions are
evaluated over the adjudication, the evidence and the organisation's own
track record, and any one of them firing sends the case to INSUFFICIENT
EVIDENCE before the coverage and confidence thresholds are looked at.

    T1  no hypothesis passes both hard gates
    T2  the best hypothesis explains too little of the residual
    T3  the leading hypothesis needs a source we do not hold
    T4  two hypotheses are indistinguishable on what we do hold
    T5  the evidence contradicts itself across independent lanes
    T6  the evidence is mostly text
    T7  our record on this kind of case is below the publication floor
    T8  a source this case depends on is stale

EVERY TRIGGER IS EVALUATED AND REPORTED, including the ones that did not
fire. A UI that showed only the fired ones would leave a reader unable to
tell "we checked and it was fine" from "we did not check".

SCOPE, and it decides #2451. T1, T2, T3 and T6 are evaluated against the
LEADING hypothesis — the one the case is proposing. Evaluated against
every hypothesis on the slate, #2451 would abstain, because its competing
hypothesis H2 is unverifiable by construction and always will be. That H2
is unverifiable and holds real money is not an abstention; it is what
makes the verdict PARTIALLY explained, and the verdict decision table is
where it belongs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_layer.schema import SemanticLayer

#: The closed set, in order. `engine/contracts.py::TriggerId` is the same
#: eight and a load-time check keeps them in step.
IDS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")


@dataclass(frozen=True)
class Trigger:
    """One named boolean, its answer, and why."""

    id: str
    name: str
    fired: bool
    detail: str
    facts: dict = field(default_factory=dict)
    #: True when a confidence cap forced it rather than the condition
    #: being evaluated on its own.
    forced: bool = False

    def render(self) -> str:
        mark = "FIRED" if self.fired else "  ok "
        return f"  {mark} {self.id}  {self.name}: {self.detail}"


@dataclass(frozen=True)
class TriggerInput:
    """Everything the eight conditions read.

    A single typed bundle rather than eight signatures, so a trigger
    cannot quietly start depending on something the others cannot see.
    """

    #: Every hypothesis that survived both hard gates.
    live_hypotheses: tuple[str, ...]
    #: Every hypothesis that was adjudicated at all.
    all_hypotheses: tuple[str, ...]
    leading: str | None
    #: Share of the qualified residual the leading hypothesis attributes.
    leading_share: float
    #: Sources the leading hypothesis requires and the organisation lacks.
    missing_sources: tuple[str, ...]
    #: Per live hypothesis, the confidence it would carry on its own.
    hypothesis_confidence: dict[str, float]
    #: Cohen's kappa between the structured and unstructured lanes. None
    #: when only one lane reported, which is not a contradiction.
    lane_kappa: float | None
    #: How many independent lanes reported on the leading hypothesis.
    independent_lanes: int
    #: Share of the leading hypothesis's evidence weight below the floor.
    low_reliability_share: float
    #: Historical accuracy for this case type, and how many cases back it.
    case_type_accuracy: float | None
    case_type_cases: int
    #: Staleness in hours of each source the leading hypothesis requires.
    source_staleness_hours: dict[str, float]
    #: Sources the leading hypothesis requires that could not be read.
    unavailable_sources: tuple[str, ...] = ()
    #: Triggers a confidence cap forces regardless of its own condition.
    forced: tuple[str, ...] = ()


def evaluate_triggers(inputs: TriggerInput, layer: SemanticLayer) -> tuple[Trigger, ...]:
    """All eight, fired or not, in order."""
    specs = layer.adjudication.triggers
    results = [
        _t1(inputs, specs["T1"]),
        _t2(inputs, specs["T2"]),
        _t3(inputs, specs["T3"]),
        _t4(inputs, specs["T4"]),
        _t5(inputs, specs["T5"]),
        _t6(inputs, specs["T6"]),
        _t7(inputs, specs["T7"]),
        _t8(inputs, specs["T8"]),
    ]
    # A cap can force a trigger. The condition would normally reach the
    # same answer on its own; forcing it means the cap and the trigger can
    # never disagree, which matters because the UI names them separately.
    return tuple(
        item
        if item.fired or item.id not in inputs.forced
        else Trigger(
            item.id, item.name, True,
            f"forced by a confidence cap. {item.detail}",
            item.facts, forced=True,
        )
        for item in results
    )


def fired(triggers) -> tuple[str, ...]:
    return tuple(item.id for item in triggers if item.fired)


# ===========================================================================
# The eight
# ===========================================================================


def _t1(inputs: TriggerInput, spec) -> Trigger:
    """Nothing survived both hard gates.

    Not "nothing looked promising" — nothing passed temporal precedence
    and effect-size sufficiency. Every candidate either arrived after the
    thing it was meant to have caused, or could not have moved the number
    this far.
    """
    count = len(inputs.live_hypotheses)
    return Trigger(
        "T1", spec.name, count == 0,
        (
            f"none of {len(inputs.all_hypotheses)} candidates passed both hard gates."
            if count == 0
            else f"{count} of {len(inputs.all_hypotheses)} candidates passed both hard "
                 f"gates: {', '.join(inputs.live_hypotheses)}."
        ),
        {"live": count, "considered": len(inputs.all_hypotheses)},
    )


def _t2(inputs: TriggerInput, spec) -> Trigger:
    """The best surviving hypothesis accounts for too little to be useful.

    A hypothesis that explains a fifth of a movement is not an answer to
    "why did this happen"; it is a fifth of an answer, and publishing it
    as the answer is how a dashboard becomes misleading.
    """
    floor = spec.min_residual_share
    share = inputs.leading_share
    fires = bool(inputs.leading) and share < floor
    return Trigger(
        "T2", spec.name, fires,
        (
            f"the leading hypothesis accounts for {share:.0%} of the qualified "
            f"residual, below the {floor:.0%} floor."
            if fires
            else f"the leading hypothesis accounts for {share:.0%} of the qualified "
                 f"residual, at or above the {floor:.0%} floor."
        ),
        {"share": share, "floor": floor},
    )


def _t3(inputs: TriggerInput, spec) -> Trigger:
    """The leading hypothesis needs something the organisation does not have.

    The most honest trigger of the eight, and the one that turns an
    abstention into a purchase order: it names the feed, and the
    resolution panel prices it.
    """
    missing = inputs.missing_sources
    return Trigger(
        "T3", spec.name, bool(missing),
        (
            f"the leading hypothesis requires {', '.join(missing)}, which the "
            "organisation does not hold. No amount of analysis of what is held is "
            "evidence about what is not."
            if missing
            else "every source the leading hypothesis requires is held."
        ),
        {"missing_sources": list(missing)},
    )


def _t4(inputs: TriggerInput, spec) -> Trigger:
    """Two hypotheses that the available data cannot tell apart.

    Not "two hypotheses are both plausible" — two hypotheses score within
    `max_confidence_separation` of each other, which means picking one is
    picking, not concluding. The right answer is to say so and name what
    would separate them.
    """
    separation = spec.max_confidence_separation
    ranked = sorted(
        ((name, value) for name, value in inputs.hypothesis_confidence.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    # T4 asks whether two hypotheses are indistinguishable, so it needs a
    # SECOND one. Written as "is there anything after the leader" rather
    # than a count, so no number appears.
    leader = next(iter(ranked), None)
    runner_up = next(iter(ranked[1:]), None)
    if leader is None or runner_up is None:
        return Trigger(
            "T4", spec.name, False,
            f"{len(ranked)} hypothesis carries a confidence, so there is no pair to "
            "be indistinguishable.",
            {"pairs": []},
        )
    (top_name, top), (next_name, runner) = leader, runner_up
    gap = top - runner
    fires = gap < separation
    return Trigger(
        "T4", spec.name, fires,
        (
            f"{top_name} ({top:.2f}) and {next_name} ({runner:.2f}) are {gap:.2f} apart, "
            f"inside the {separation:.2f} the data can resolve. Choosing between them "
            "would be choosing, not concluding."
            if fires
            else f"{top_name} ({top:.2f}) leads {next_name} ({runner:.2f}) by {gap:.2f}, "
                 f"outside the {separation:.2f} resolution floor."
        ),
        {"leader": top_name, "runner_up": next_name, "separation": gap},
    )


def _t5(inputs: TriggerInput, spec) -> Trigger:
    """The lanes contradict each other.

    Two independent readings of the same estate that agree WORSE than
    chance is not weak evidence, it is conflicting evidence, and averaging
    conflicting evidence produces a number about neither reading. Measured
    as a negative Cohen's kappa, which is what "worse than chance" means.
    """
    minimum = spec.min_independent_sources
    kappa = inputs.lane_kappa
    if inputs.independent_lanes < minimum or kappa is None:
        return Trigger(
            "T5", spec.name, False,
            f"{inputs.independent_lanes} independent lane(s) reported; {minimum} are "
            "needed before they can contradict each other.",
            {"kappa": kappa, "lanes": inputs.independent_lanes},
        )
    fires = kappa < 0.0
    return Trigger(
        "T5", spec.name, fires,
        (
            f"the structured and unstructured lanes agree worse than chance "
            f"(kappa = {kappa:.2f}). They are pointing at different stores."
            if fires
            else f"the two lanes agree better than chance (kappa = {kappa:.2f})."
        ),
        {"kappa": kappa, "lanes": inputs.independent_lanes},
    )


def _t6(inputs: TriggerInput, spec) -> Trigger:
    """The case rests on text.

    CLAUDE.md's floor rule: text alone never reaches a verdict; text plus
    a matched control does. A hundred store notes saying the shelf was
    empty is a hundred people's impressions, and it is not a measurement.
    """
    limit = spec.max_low_reliability_share
    share = inputs.low_reliability_share
    fires = share > limit
    return Trigger(
        "T6", spec.name, fires,
        (
            f"{share:.0%} of the leading hypothesis's evidence weight sits below the "
            f"{spec.reliability_floor:.2f} reliability floor, above the {limit:.0%} "
            "limit. Text alone does not reach a verdict."
            if fires
            else f"{share:.0%} of the evidence weight is below the "
                 f"{spec.reliability_floor:.2f} floor, within the {limit:.0%} limit."
        ),
        {"low_reliability_share": share, "limit": limit},
    )


def _t7(inputs: TriggerInput, spec) -> Trigger:
    """Our own record on this kind of case is not good enough to publish.

    The trigger that makes the calibration ledger load-bearing rather than
    decorative. A case type the organisation gets right 58% of the time
    does not get a published attribution, however confident this
    particular case looks — the confidence is a claim about this case, and
    the track record is a claim about claims like it.

    Too FEW closed cases is not the same as a poor record, and does not
    fire. It is reported as no track record, which the resolution panel
    can act on differently.
    """
    floor = spec.publication_floor
    minimum = spec.min_closed_cases
    accuracy, cases = inputs.case_type_accuracy, inputs.case_type_cases
    if accuracy is None or cases < minimum:
        return Trigger(
            "T7", spec.name, False,
            f"{cases} closed cases of this type; {minimum} are needed before a track "
            "record means anything. No record is not a bad record.",
            {"accuracy": accuracy, "cases": cases, "floor": floor},
        )
    fires = accuracy < floor
    return Trigger(
        "T7", spec.name, fires,
        (
            f"cases of this type have been right {accuracy:.0%} of the time over "
            f"{cases} closed cases, below the {floor:.0%} publication floor."
            if fires
            else f"cases of this type have been right {accuracy:.0%} of the time over "
                 f"{cases} closed cases, at or above the {floor:.0%} floor."
        ),
        {"accuracy": accuracy, "cases": cases, "floor": floor},
    )


def _t8(inputs: TriggerInput, spec) -> Trigger:
    """A source this case depends on has not arrived.

    Only the sources the LEADING hypothesis requires. A stale feed nobody
    is relying on is a data-quality ticket, not a reason to withhold an
    answer that does not use it.
    """
    limit = spec.max_staleness_hours
    stale = {
        name: hours
        for name, hours in inputs.source_staleness_hours.items()
        if hours > limit
    }
    unavailable = inputs.unavailable_sources
    fires = bool(stale or unavailable)
    parts = []
    if stale:
        parts.append(
            ", ".join(f"{name} {hours:.0f}h" for name, hours in sorted(stale.items()))
            + f" (limit {limit:.0f}h)"
        )
    if unavailable:
        parts.append(f"unavailable: {', '.join(unavailable)}")
    return Trigger(
        "T8", spec.name, fires,
        (
            "a source the leading hypothesis requires is not current: " + "; ".join(parts)
            if fires
            else f"all {len(inputs.source_staleness_hours)} required sources are inside "
                 f"the {limit:.0f}h staleness limit."
        ),
        {"stale": stale, "unavailable": list(unavailable), "limit": limit},
    )


__all__ = ["IDS", "Trigger", "TriggerInput", "evaluate_triggers", "fired"]
