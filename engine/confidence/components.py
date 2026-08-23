"""s1 through s6 — the six inputs to the confidence score.

    conf_raw = 0.28*s1 + 0.20*s2 + 0.16*s3 + 0.14*s4 + 0.10*s5 + 0.12*s6

Every weight comes from `semantic_layer/adjudication.yaml` and so does
every parameter each component needs (CLAUDE.md rule 2). Nothing in this
module chooses a number.

WHAT EACH ONE IS ASKING, because the six are easy to confuse:

  s1  how strong is the evidence         a statistic
  s2  how much of it did we manage       a count of tests
      to test
  s3  do the lanes agree                 two independent readings
  s4  is the data any good               freshness and completeness
  s5  how far back can we see            weeks of history
  s6  how much of the movement           the attributed share
      is accounted for

They are not redundant. A case can have overwhelming evidence for a small
part of the movement (high s1, low s6), or complete test coverage over a
warehouse that only started collecting in September (high s2, low s5).
Averaging them into one number is a compression, and the breakdown is
published alongside so a reader can see which of the six is carrying it.

NO MODEL CALL HAPPENS HERE EITHER. Confidence is arithmetic over
ADJUDICATE's findings and the warehouse's own metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.adjudicate import AdjudicationResult, HypothesisVerdict
from semantic_layer.schema import ConfidenceComponentSpec, SemanticLayer

#: The keys, in the order they render. Not a set: the breakdown is read
#: top to bottom by a human and s1 comes first.
KEYS = ("s1", "s2", "s3", "s4", "s5", "s6")


class ComponentError(RuntimeError):
    """A component cannot be computed from what was supplied."""


@dataclass(frozen=True)
class Component:
    """One of the six, with the arithmetic that produced it."""

    key: str
    name: str
    weight: float
    value: float
    detail: str
    method: str
    facts: dict = field(default_factory=dict)

    @property
    def contribution(self) -> float:
        return self.weight * self.value


@dataclass(frozen=True)
class SourceHealth:
    """What the warehouse knows about one source system.

    `freshness` and `completeness` are scores in [0, 1]; `history_weeks`
    is how far back the source actually goes, which is not the same as how
    far back the KPI goes.
    """

    name: str
    freshness: float
    completeness: float
    history_weeks: float
    held: bool = True
    #: The age itself, alongside the score. T8 asks a question about
    #: hours ("is this feed stale?") and s4 asks one about quality, and
    #: converting the first into the second and back loses the answer.
    staleness_hours: float = 0.0

    @property
    def quality(self) -> float:
        return self.freshness * self.completeness


@dataclass(frozen=True)
class ScoringContext:
    """Everything the six components read, gathered once.

    Assembled by `engine/confidence/gate.py` so no component reaches for
    the warehouse on its own and no two of them can disagree about what
    the data was.
    """

    adjudication: AdjudicationResult
    leading: HypothesisVerdict
    layer: SemanticLayer
    #: Per required source, as named in the causal graph.
    sources: dict[str, SourceHealth]
    #: Per store, whether the structured lane placed the cause there.
    structured_present: frozenset[str] = frozenset()
    #: Per store, whether the unstructured lane's classifier tagged a
    #: document for this hypothesis.
    unstructured_present: frozenset[str] = frozenset()
    #: The universe both lanes are judged over.
    stores_in_scope: frozenset[str] = frozenset()
    #: A structural break inside the history the baseline was fitted on.
    regime_change: bool = False

    def spec(self, key: str) -> ConfidenceComponentSpec:
        return self.layer.adjudication.confidence.components[key]


# ===========================================================================
# s1 — evidence strength
# ===========================================================================


def s1_evidence_strength(ctx: ScoringContext) -> Component:
    """How many standard errors past significance the estimate sits.

    A logistic of the difference-in-differences t-statistic, with the
    midpoint at the conventional 5% critical value: a just-significant
    result scores 0.5, because a just-significant result really could go
    either way. Both the midpoint and the scale come from the semantic
    layer and neither was chosen by looking at a case.

    The pre-test penalty is the honest part. A failed parallel-trends
    pre-test does not say the effect is absent — it says the design cannot
    separate the effect from a trend that was already running, and the
    strength of the evidence has to fall whatever the t-statistic says.
    """
    spec = ctx.spec("s1")
    estimate = ctx.leading.did
    if estimate is None:
        return Component(
            "s1", spec.name, spec.weight, 0.0,
            "no difference-in-differences was estimated for this hypothesis, so "
            "there is no effect size to score. Not a weak result; an absent one.",
            spec.method, {"t_statistic": None},
        )

    t_statistic = abs(float(estimate.t_statistic))
    value = _logistic((t_statistic - spec.midpoint_t) / spec.scale_t)

    pretest = ctx.leading.finding(ctx.layer.adjudicate.did.test_id)
    pretest_passed = bool(pretest.facts.get("pretest_passes", True)) if pretest else True
    penalty = spec.pretest_failure_penalty if not pretest_passed else 0.0
    value = max(0.0, value - penalty)

    detail = (
        f"|t| = {t_statistic:.2f} against a {spec.midpoint_t:g} midpoint "
        f"and a {spec.scale_t:g}-standard-error scale"
    )
    if penalty:
        detail += (
            f", less {penalty:.2f} because the parallel-trends pre-test failed: the "
            "design cannot separate this effect from a trend already running"
        )
    return Component(
        "s1", spec.name, spec.weight, value, detail, spec.method,
        {
            "t_statistic": t_statistic,
            "pretest_passed": pretest_passed,
            "penalty": penalty,
        },
    )


def _logistic(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


# ===========================================================================
# s2 — test coverage
# ===========================================================================


def s2_test_coverage(ctx: ScoringContext) -> Component:
    """Of the tests that COULD run on this hypothesis, how many passed.

    A test that could not run is excluded from both sides of the ratio.
    Counting it as a failure would let a missing feed look like a refuted
    hypothesis, and counting it as a pass would let a missing feed look
    like a confirmed one. It is reported separately instead.
    """
    spec = ctx.spec("s2")
    findings = ctx.leading.findings
    testable = [item for item in findings if item.testable]
    untestable = [item for item in findings if not item.testable]

    if not testable:
        return Component(
            "s2", spec.name, spec.weight, 0.0,
            f"none of the {len(findings)} applicable tests could be run.",
            spec.method, {"passed": 0, "testable": 0, "not_testable": len(untestable)},
        )

    passed = sum(1 for item in testable if item.passed)
    value = passed / len(testable)
    detail = f"{passed} of {len(testable)} runnable tests passed"
    if untestable:
        names = ", ".join(item.name.lower() for item in untestable)
        detail += (
            f"; {len(untestable)} could not be run and are counted on neither "
            f"side ({names})"
        )
    return Component(
        "s2", spec.name, spec.weight, value, detail + ".", spec.method,
        {"passed": passed, "testable": len(testable), "not_testable": len(untestable)},
    )


# ===========================================================================
# s3 — source agreement
# ===========================================================================


def s3_source_agreement(ctx: ScoringContext) -> Component:
    """Do the structured and unstructured lanes point at the same stores?

    Cohen's kappa rather than raw agreement. Two lanes that both find
    nothing in 106 of 140 stores agree about 89% of the estate without
    either of them having found anything, and a statistic that rewards
    that is measuring the size of the estate. Kappa subtracts the
    agreement two independent raters would reach by chance.

    Kappa can go negative — worse than chance, which is a real finding and
    a bad one. It floors at zero here because confidence is a [0, 1]
    quantity, and the raw kappa is reported beside it.
    """
    spec = ctx.spec("s3")
    universe = ctx.stores_in_scope
    if not universe:
        return Component(
            "s3", spec.name, spec.weight, 0.0,
            "no store universe was supplied, so the two lanes cannot be compared.",
            spec.method, {"kappa": None},
        )

    structured = ctx.structured_present & universe
    unstructured = ctx.unstructured_present & universe
    total = len(universe)
    both = len(structured & unstructured)
    neither = len(universe - structured - unstructured)

    observed = (both + neither) / total
    expected = (
        (len(structured) / total) * (len(unstructured) / total)
        + (1 - len(structured) / total) * (1 - len(unstructured) / total)
    )
    kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)
    value = max(0.0, kappa) if spec.floor_at_zero else kappa

    return Component(
        "s3", spec.name, spec.weight, value,
        (
            f"Cohen's kappa = {kappa:.2f} over {total} stores: the two lanes agree on "
            f"{observed:.0%} where chance alone would give {expected:.0%}. "
            f"{both} stores carry both a structured signal and a tagged document, "
            f"{neither} carry neither."
        ),
        spec.method,
        {
            "kappa": kappa,
            "observed_agreement": observed,
            "chance_agreement": expected,
            "both": both,
            "neither": neither,
            "structured_only": len(structured - unstructured),
            "unstructured_only": len(unstructured - structured),
            "stores": total,
        },
    )


# ===========================================================================
# s4 — data quality
# ===========================================================================


def s4_data_quality(ctx: ScoringContext) -> Component:
    """The worst of the sources this hypothesis rests on.

    A minimum, not a mean. The argument is only as good as the feed it
    most depends on, and averaging lets three clean sources hide one that
    has not arrived since Tuesday.
    """
    spec = ctx.spec("s4")
    sources = ctx.sources
    if not sources:
        return Component(
            "s4", spec.name, spec.weight, 0.0,
            "no source health was supplied for this hypothesis.",
            spec.method, {"sources": {}},
        )

    worst_name = min(sources, key=lambda name: sources[name].quality)
    worst = sources[worst_name]
    return Component(
        "s4", spec.name, spec.weight, worst.quality,
        (
            f"the weakest of {len(sources)} required sources is {worst_name}: "
            f"freshness {worst.freshness:.2f} x completeness {worst.completeness:.2f} "
            f"= {worst.quality:.2f}. "
            + ", ".join(
                f"{name} {health.quality:.2f}" for name, health in sorted(sources.items())
            )
        ),
        spec.method,
        {
            "worst_source": worst_name,
            "sources": {
                name: {"freshness": h.freshness, "completeness": h.completeness}
                for name, h in sources.items()
            },
        },
    )


# ===========================================================================
# s5 — historical depth
# ===========================================================================


def s5_historical_depth(ctx: ScoringContext) -> Component:
    """How far back the SHORTEST required source goes.

    Not the longest series in the warehouse. Eighteen months of sales is
    no comfort when the argument rests on inventory snapshots that started
    in September — the pre-period the design compares against is bounded
    by whichever feed started last.
    """
    spec = ctx.spec("s5")
    sources = ctx.sources
    if not sources:
        return Component(
            "s5", spec.name, spec.weight, 0.0,
            "no source history was supplied for this hypothesis.",
            spec.method, {"weeks": None},
        )

    shortest_name = min(sources, key=lambda name: sources[name].history_weeks)
    weeks = sources[shortest_name].history_weeks
    depth = min(1.0, weeks / spec.reference_weeks)
    penalty = spec.regime_change_penalty if ctx.regime_change else 0.0
    value = max(0.0, depth - penalty)

    detail = (
        f"{weeks:.0f} weeks of {shortest_name}, the shortest of "
        f"{len(sources)} required sources, against a {spec.reference_weeks}-week "
        f"reference"
    )
    if penalty:
        detail += (
            f", less {penalty:.2f} for a regime change inside that window: history "
            "either side of a break is not one history"
        )
    return Component(
        "s5", spec.name, spec.weight, value, detail + ".", spec.method,
        {
            "shortest_source": shortest_name,
            "history_weeks": weeks,
            "reference_weeks": spec.reference_weeks,
            "regime_change": ctx.regime_change,
            "penalty": penalty,
        },
    )


# ===========================================================================
# s6 — residual coverage
# ===========================================================================


def s6_residual_coverage(ctx: ScoringContext) -> Component:
    """How much of the qualified residual anything at all accounts for.

    ADJUDICATE's coverage, unchanged. It is here because a hypothesis can
    be beyond doubt and still explain a fifth of the movement, and a
    confidence score that did not know the difference would publish
    certainty about a fifth of an answer.
    """
    spec = ctx.spec("s6")
    coverage = ctx.adjudication.coverage
    return Component(
        "s6", spec.name, spec.weight, coverage,
        (
            f"{coverage:.0%} of the qualified residual is attributed; "
            f"{ctx.adjudication.unattributed_pt:.2f} pt is not."
        ),
        spec.method,
        {
            "coverage": coverage,
            "unattributed_pt": ctx.adjudication.unattributed_pt,
        },
    )


COMPONENTS = {
    "s1": s1_evidence_strength,
    "s2": s2_test_coverage,
    "s3": s3_source_agreement,
    "s4": s4_data_quality,
    "s5": s5_historical_depth,
    "s6": s6_residual_coverage,
}


def compute_components(ctx: ScoringContext) -> tuple[Component, ...]:
    """All six, in order."""
    return tuple(COMPONENTS[key](ctx) for key in KEYS)


def weighted_sum(components) -> float:
    """conf_raw. The weights come from the layer and sum to one."""
    return float(sum(item.contribution for item in components))


__all__ = [
    "COMPONENTS",
    "KEYS",
    "Component",
    "ComponentError",
    "ScoringContext",
    "SourceHealth",
    "compute_components",
    "s1_evidence_strength",
    "s2_test_coverage",
    "s3_source_agreement",
    "s4_data_quality",
    "s5_historical_depth",
    "s6_residual_coverage",
    "weighted_sum",
]
