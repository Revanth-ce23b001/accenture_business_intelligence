"""Where the candidate hypotheses come from, and which five survive.

Three sources, and they are deliberately unequal:

  the causal graph   what the business has already written down, at the
                     prior it wrote down
  case history       what has actually moved this KPI in this region
                     before. A driver that explained a movement here twice
                     is a better bet than one that never has, and the case
                     registry already knows.
  the long tail      what the model suggests. Enters at a low prior and has
                     to earn its place.

Then a screen to five. More than five and the six adjudication tests stop
being affordable; fewer and the answer is whatever the first plausible
story was.

TWO THINGS THE SCREEN DELIBERATELY DOES NOT DO.

It does not penalise a hypothesis for needing a source the organisation
lacks. Unverifiable is not the same as unlikely, and the difference
decides case #2451: H2 (competitor promotion) cannot be tested, holds
INR 0.86 Cr of the residual, and is exactly why the verdict is PARTIALLY
EXPLAINED rather than EXPLAINED. Screening it out here would delete the
case's conclusion.

It does not offer hypotheses an earlier stage has already removed. The
residual is calendar-free because Gate 2 subtracted the calendar, and
incident-free because Gate 1 would have killed the case. Offering
`calendar_shift` as an explanation for what is left is offering to explain
the same thing twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb

from engine.db import execute_metadata
from llm.hypothesise import LongTailCandidate, generate
from llm.provider import LLMProvider
from security.policy import User
from semantic_layer.schema import HypothesisTemplate, SemanticLayer

#: Where a candidate came from. Named, because "the model suggested this"
#: and "the business declared this" must never render the same way.
ORIGIN_GRAPH = "causal_graph"
ORIGIN_LONG_TAIL = "long_tail"


@dataclass(frozen=True)
class Candidate:
    """One hypothesis in the running, and why it is in the running."""

    tag: str
    label: str
    description: str
    origin: str
    #: The prior the screen actually sorts on. The DECLARED prior from
    #: `causal_graph.yaml` until somebody has confirmed or rejected this
    #: driver on this KPI, and the learned posterior after that.
    prior: float
    applicability: float
    prior_cases: int
    history_bonus: float
    required_sources: tuple[str, ...]
    missing_sources: tuple[str, ...]
    template: HypothesisTemplate | None = None
    #: What the causal graph declares, kept beside the effective figure so
    #: the screening panel can show a prior that has moved AS moved. A
    #: learned prior that renders identically to a written-down one hides
    #: the only visible evidence that the feedback loop is running.
    declared_prior: float | None = None
    prior_confirmed: int = 0
    prior_rejected: int = 0

    @property
    def score(self) -> float:
        """prior x applicability. The screen sorts on this."""
        return self.prior * self.applicability

    @property
    def prior_moved(self) -> bool:
        """Whether feedback has moved this prior off the declared one."""
        return (
            self.declared_prior is not None and self.prior != self.declared_prior
        )

    @property
    def prior_shift(self) -> float:
        if self.declared_prior is None:
            return 0.0
        return self.prior - self.declared_prior

    @property
    def verifiable(self) -> bool:
        """False when a source it needs is not held. Not a reason to drop it."""
        return not self.missing_sources

    @property
    def generated(self) -> bool:
        return self.origin == ORIGIN_LONG_TAIL


@dataclass(frozen=True)
class Screening:
    """The whole slate, and the five that survived it."""

    selected: tuple[Candidate, ...]
    considered: tuple[Candidate, ...]
    excluded: dict[str, str]
    long_tail_offered: tuple[LongTailCandidate, ...]

    def by_tag(self) -> dict[str, Candidate]:
        return {candidate.tag: candidate for candidate in self.selected}

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(candidate.tag for candidate in self.selected)

    @property
    def unverifiable(self) -> tuple[str, ...]:
        return tuple(
            candidate.tag for candidate in self.selected if not candidate.verifiable
        )


def prior_case_counts(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
    *,
    kpi: str,
    scope: str,
    now: datetime,
) -> dict[str, int]:
    """How often each driver has explained this KPI in this scope before.

    Read from `case_hypothesis` joined to `case_registry`. An OPEN case is
    a question, not a precedent, so only closed ones count.
    """
    spec = layer.gather.hypotheses.sources.case_history
    if not spec.enabled:
        return {}

    since = now - timedelta(days=spec.lookback_days)
    rows = execute_metadata(
        user,
        "case_registry",
        """
        SELECT h.hypothesis_id AS tag, COUNT(*) AS cases
        FROM case_hypothesis AS h
        JOIN case_registry   AS r USING (case_id)
        WHERE r.kpi = $kpi
          AND r.scope = $scope
          AND r.opened_at >= $since
          AND (NOT $closed_only OR r.status = 'closed')
          AND h.status = 'supported'
        GROUP BY 1
        """,
        {
            "kpi": kpi,
            "scope": scope,
            "since": since.replace(tzinfo=None) if now.tzinfo else since,
            "closed_only": spec.closed_only,
        },
        connection=connection,
        layer=layer,
        purpose="gather.hypotheses.case_history",
    )
    return {str(row["tag"]): int(row["cases"]) for row in rows}


def _learned_priors(connection, layer, kpi: str) -> dict:
    """Effective priors for this KPI, or nothing if they cannot be read.

    FAIL OPEN, TO THE CONTRACT. A warehouse that predates
    `hypothesis_prior` — or a read that fails for any other reason —
    leaves screening on the DECLARED priors, which is exactly what it did
    before the loop existed. The alternative is a case that cannot be
    opened because nobody has given feedback yet.
    """
    from engine.learn.priors import effective_priors

    try:
        return effective_priors(connection, kpi=kpi, layer=layer)
    except Exception:  # noqa: BLE001 - the declared prior is a safe answer
        return {}


def screen(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
    *,
    kpi: str,
    scope: str,
    grain: str,
    period: str,
    direction: str,
    now: datetime,
    provider: LLMProvider | None = None,
) -> Screening:
    """Assemble the candidates from all three sources and take the top five."""
    spec = layer.gather.hypotheses
    graph = layer.causal_graph.hypotheses
    history = prior_case_counts(
        connection, user, layer, kpi=kpi, scope=scope, now=now
    )
    # The feedback loop reaches the engine here, and only here. A driver
    # readers keep rejecting on this KPI is screened lower next time; one
    # they keep confirming is screened higher. Bounded by
    # `learning.yaml -> priors`, so a hypothesis cannot be argued out of
    # the slate by volume of clicks.
    learned = _learned_priors(connection, layer, kpi)

    considered: list[Candidate] = []
    excluded: dict[str, str] = {}

    for tag, template in sorted(graph.items()):
        if tag in spec.already_removed:
            excluded[tag] = spec.already_removed[tag]
            continue
        if kpi not in template.affects:
            excluded[tag] = f"does not affect {kpi}"
            continue

        cases = history.get(tag, 0)
        bonus = min(cases * spec.sources.case_history.bonus_per_case,
                    spec.sources.case_history.max_bonus)
        effective = learned.get(tag)
        considered.append(
            Candidate(
                tag=tag,
                label=template.label,
                description=template.description.strip(),
                origin=ORIGIN_GRAPH,
                prior=effective.effective if effective else template.prior,
                applicability=spec.applicability.base + bonus,
                prior_cases=cases,
                history_bonus=bonus,
                required_sources=tuple(template.required_sources),
                missing_sources=tuple(template.missing_sources()),
                template=template,
                declared_prior=template.prior,
                prior_confirmed=effective.confirmed if effective else 0,
                prior_rejected=effective.rejected if effective else 0,
            )
        )

    offered: tuple[LongTailCandidate, ...] = ()
    if spec.sources.long_tail.enabled:
        offered = generate(
            kpi,
            scope,
            period,
            direction,
            [candidate.tag for candidate in considered],
            spec.sources.long_tail,
            provider=provider,
        )
        for candidate in offered:
            considered.append(
                Candidate(
                    tag=candidate.tag,
                    label=candidate.label,
                    description=candidate.description,
                    origin=ORIGIN_LONG_TAIL,
                    prior=candidate.prior,
                    applicability=spec.applicability.base,
                    prior_cases=0,
                    history_bonus=0.0,
                    required_sources=(),
                    missing_sources=(),
                )
            )

    # Descending score; ties broken by tag so the slate is reproducible.
    ranked = sorted(considered, key=lambda item: (-item.score, item.tag))
    selected = tuple(ranked[: spec.top_n])
    for candidate in ranked[spec.top_n :]:
        excluded[candidate.tag] = (
            f"ranked {ranked.index(candidate) + 1} of {len(ranked)} at "
            f"{candidate.score:.3f}, below the top {spec.top_n}"
        )
    return Screening(
        selected=selected,
        considered=tuple(ranked),
        excluded=excluded,
        long_tail_offered=offered,
    )


__all__ = [
    "ORIGIN_GRAPH",
    "ORIGIN_LONG_TAIL",
    "Candidate",
    "Screening",
    "prior_case_counts",
    "screen",
]
