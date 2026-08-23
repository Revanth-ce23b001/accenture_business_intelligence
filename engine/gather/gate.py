"""GATHER — screen the hypotheses, then run three lanes at once.

    Data -> VALIDATE -> QUALIFY -> GATHER -> ADJUDICATE -> VERDICT

QUALIFY produced a residual worth explaining. GATHER finds the evidence
for every candidate at once and then stops. It forms no opinion: nothing
here decides whether a hypothesis is supported, eliminated, or leading.
That is ADJUDICATE's job, and it gets the evidence cold.

The three lanes are independent, so they run concurrently. The warehouse
is not: one DuckDB connection is one transaction, and a cursor is no help
because it cannot see its parent's uncommitted writes. So warehouse reads
are serialised behind one lock (`engine/gather/guard.py`) and everything
else overlaps — BM25, the SVD, and above all the model calls, which are
network-bound and are most of the wall clock on a live run.

A lane that fails does not fail the stage. An adjudication that knows one
lane is missing is worth more than one that never ran, so a failure is
recorded by name and the other two continue.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime

import duckdb

from engine.contracts import Evidence
from engine.db import GovernanceError, warehouse_clock
from engine.evidence import EvidenceFactory, EvidenceLedger
from engine.gather.external import ExternalResult, gather_external
from engine.gather.guard import LockGuard
from engine.gather.hypotheses import Candidate, Screening, screen
from engine.gather.structured import StructuredResult, gather_structured
from engine.gather.unstructured import UnstructuredResult, gather_unstructured
from llm.provider import LLMProvider
from security.policy import PolicyError, User, resolve_policy
from semantic_layer.schema import SemanticLayer, get_semantic_layer

EVIDENCE_PREFIX = "gather"


class GatherError(RuntimeError):
    """GATHER could not be run as asked."""


@dataclass(frozen=True)
class GatherRequest:
    """What is being gathered for. Mirrors the QUALIFY request."""

    kpi: str
    scope: str
    grain: str
    period: str
    period_start: date
    period_end: date
    comparison_start: date
    comparison_end: date
    #: Which way the unexplained movement went. Given to the long-tail
    #: generator so it proposes causes of the right sign.
    direction: str = "down"


@dataclass
class GatherResult:
    """Everything the three lanes brought back, and what they could not."""

    request: GatherRequest
    screening: Screening
    structured: tuple[StructuredResult, ...] = ()
    unstructured: UnstructuredResult | None = None
    external: ExternalResult | None = None
    evidence: tuple[Evidence, ...] = ()
    lane_failures: dict[str, str] = field(default_factory=dict)
    as_of: datetime | None = None

    @property
    def hypotheses(self) -> tuple[Candidate, ...]:
        return self.screening.selected

    @property
    def lanes_run(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, ran in (
                ("structured", bool(self.structured)),
                ("unstructured", self.unstructured is not None),
                ("external", self.external is not None),
            )
            if ran
        )

    @property
    def cache_hit_rate(self) -> float:
        return self.unstructured.cache_hit_rate if self.unstructured else 0.0

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.evidence}

    def structured_for(self, tag: str) -> StructuredResult | None:
        for result in self.structured:
            if result.hypothesis == tag:
                return result
        return None


def gather(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    request: GatherRequest,
    *,
    layer: SemanticLayer | None = None,
    clock: datetime | None = None,
    provider: LLMProvider | None = None,
) -> GatherResult:
    """Screen to five candidates, then gather for all five at once."""
    layer = layer or get_semantic_layer()
    if request.kpi not in layer.kpis:
        raise GatherError(
            f"no KPI contract named {request.kpi!r}; known: {sorted(layer.kpis)}"
        )
    # Authorization before retrieval (CLAUDE.md rule 5). The lanes are
    # fail-open, so a persona the contract does not know would otherwise
    # surface as three lane failures rather than one refusal. Raised as the
    # same GovernanceError the governed door raises, so a caller has one
    # exception to catch across every stage.
    try:
        resolve_policy(layer.kpis[request.kpi], user, layer.warehouse.governance)
    except PolicyError as exc:
        raise GovernanceError(str(exc)) from exc

    clock = clock or warehouse_clock(connection, layer)
    factory = EvidenceFactory.for_stage(EVIDENCE_PREFIX, clock, layer)

    guard = LockGuard()
    with guard():
        screening = screen(
            connection, user, layer,
            kpi=request.kpi,
            scope=request.scope,
            grain=request.grain,
            period=request.period,
            direction=request.direction,
            now=clock,
            provider=provider,
        )
    tags = screening.tags

    result = GatherResult(request=request, screening=screening, as_of=clock)
    ledger = EvidenceLedger()

    lanes = {
        "structured": lambda: gather_structured(
            connection, user, layer, factory,
            kpi=request.kpi, scope=request.scope,
            period_start=request.period_start, period_end=request.period_end,
            comparison_start=request.comparison_start,
            comparison_end=request.comparison_end,
            hypotheses=tags, guard=guard,
        ),
        "unstructured": lambda: gather_unstructured(
            connection, user, layer, factory,
            kpi=request.kpi, scope=request.scope,
            period_start=request.period_start, period_end=request.period_end,
            hypotheses=tags, provider=provider, now=clock, guard=guard,
        ),
        "external": lambda: gather_external(
            connection, user, layer, factory,
            kpi=request.kpi, scope=request.scope,
            period_start=request.period_start, period_end=request.period_end,
            guard=guard,
        ),
    }

    execution = layer.gather.execution
    if execution.parallel_lanes:
        with ThreadPoolExecutor(max_workers=execution.max_workers) as pool:
            futures = {name: pool.submit(_guarded, run) for name, run in lanes.items()}
            outcomes = {name: future.result() for name, future in futures.items()}
    else:
        outcomes = {name: _guarded(run) for name, run in lanes.items()}

    for name, (value, failure) in outcomes.items():
        if failure is not None:
            if not execution.fail_open:
                raise GatherError(f"the {name} lane failed: {failure}")
            result.lane_failures[name] = failure
            continue
        if name == "structured":
            result.structured = value
            for item in value:
                ledger.extend(item.evidence)
        elif name == "unstructured":
            result.unstructured = value
            ledger.extend(value.evidence)
        else:
            result.external = value
            ledger.extend(value.evidence)

    result.evidence = tuple(ledger)
    return result


def _guarded(run):
    """Run a lane, and turn a failure into a name rather than a stack trace."""
    try:
        return run(), None
    except Exception as exc:  # noqa: BLE001 - fail-open is the contract
        return None, f"{type(exc).__name__}: {exc}"


__all__ = ["GatherError", "GatherRequest", "GatherResult", "gather"]
