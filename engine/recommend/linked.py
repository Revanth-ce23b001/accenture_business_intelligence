"""The cause of the cause, and the gaps that stopped us finding it.

TWO WRITES LEAVE THIS MODULE, and both exist because the alternative is a
finding that lives only in one screenshot.

A LINKED CASE. Expedited freight puts stock back on 34 shelves for eight
weeks. It does not touch the allocation rule that emptied them, and that
rule will do the same thing next cycle. `availability_recovery` says so
in its `linked_case_template`, and closing #2451 therefore OPENS a second
case against the distribution centre — with a parent pointer, so the
chain is walkable in the warehouse and not only in the narrative.

A DATA GAP. When T3 fires, the leading hypothesis needed a source the
organisation does not hold. That is a standing fact about the business,
not an incident: it will stop the next competitor case too, and the one
after that. It goes into `data_gap_register` beside the four
reconciliation gaps, carrying the playbook that would acquire the source
and what that playbook costs — so somebody reading the register next
quarter gets the fix and not just the complaint.

Both are WRITES, so neither goes through `execute_governed`, which is the
read path. Reads here — the highest case id, the register's own rows —
go through `execute_metadata`, which is the governed path for the
warehouse's own bookkeeping tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import duckdb

from engine.db import execute_metadata
from engine.qualify.restraint import register_case
from engine.recommend.cost import Cost, CostBasis, CostError, compute_cost
from engine.recommend.investigate import candidates
from engine.warehouse.reconcile import GAP_INSERT, DataGap
from security.policy import User
from semantic_layer.schema import Playbook, SemanticLayer

CASE_REGISTRY = "case_registry"

#: Case ids are numeric strings in this warehouse ("2451"). Anything that
#: is not is skipped when allocating the next one rather than parsed
#: optimistically — a single malformed id must not reset the sequence.
NUMERIC_ID = re.compile(r"^\d+$")


class LinkedCaseError(RuntimeError):
    """The linked case cannot be raised as the playbook describes it."""


@dataclass(frozen=True)
class LinkedCase:
    """A case opened by another case, and why."""

    case_id: str
    parent_case_id: str
    kpi: str
    scope: str
    grain: str
    period: str
    driver: str
    question: str
    rationale: str
    playbook: str
    opened_at: datetime

    def render(self) -> str:
        return (
            f"linked case #{self.case_id} ({self.rationale}): {self.kpi} / "
            f"{self.scope} / {self.period}, driver {self.driver} — "
            f"{' '.join(self.question.split())}"
        )


def next_case_id(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
) -> str:
    """One past the highest numeric case id the registry holds."""
    strategy = layer.recommend.linked_case.id_strategy
    if strategy != "max_numeric_plus_one":  # pragma: no cover - single strategy
        raise LinkedCaseError(f"unknown case id strategy {strategy!r}")
    rows = execute_metadata(
        user,
        CASE_REGISTRY,
        "SELECT case_id FROM case_registry",
        connection=connection,
        layer=layer,
        purpose="recommend.next_case_id",
    )
    numeric = [
        int(str(row["case_id"])) for row in rows if NUMERIC_ID.match(str(row["case_id"]))
    ]
    if not numeric:
        raise LinkedCaseError(
            "the case registry holds no numeric case id to count on from. A linked case "
            "is raised BY a registered case, so an empty registry means the parent was "
            "never written."
        )
    return str(max(numeric) + 1)


def linked_kpi(playbook: Playbook, layer: SemanticLayer) -> str:
    """The KPI the child case argues about.

    `linked_driver_affects_first`: the first KPI the linked driver affects
    that can actually carry a case. `transport_disruption` affects
    on_shelf_availability first, which is the right place to have an
    argument about a distribution centre — a DC does not move net revenue
    directly, it moves what is on the shelf.
    """
    template = playbook.linked_case_template
    if template is None:  # pragma: no cover - guarded by the caller
        raise LinkedCaseError(f"playbook {playbook.playbook!r} links no case")
    driver = layer.causal_graph.hypotheses.get(template.driver)
    if driver is None:  # pragma: no cover - the loader cross-checks this
        raise LinkedCaseError(f"linked driver {template.driver!r} is not in the graph")
    for kpi in driver.affects:
        contract = layer.kpis.get(kpi)
        if contract is not None and contract.can_open_a_case():
            return kpi
    raise LinkedCaseError(
        f"driver {template.driver!r} affects {driver.affects}, none of which can carry "
        "a case; the child would be opened against a KPI with no materiality limit"
    )


def create_linked_case(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    playbook: Playbook,
    layer: SemanticLayer,
    *,
    parent_case_id: str,
    scope: str,
    grain: str,
    period: str,
    opened_at: datetime,
) -> LinkedCase | None:
    """Raise the playbook's linked case. Returns None when it declares none.

    The child is opened, not proposed. A "consider also investigating"
    line in a narrative is forgotten by Friday; a row in `case_registry`
    with a parent pointer is still there next month, and the restraint
    rules in QUALIFY will see it when the same scope moves again.
    """
    spec = layer.recommend.linked_case
    template = playbook.linked_case_template
    if not spec.enabled or template is None:
        return None

    case_id = next_case_id(connection, user, layer)
    if case_id == parent_case_id:  # pragma: no cover - allocation is max+1
        raise LinkedCaseError(f"allocated case id {case_id!r} collides with its parent")

    linked = LinkedCase(
        case_id=case_id,
        parent_case_id=parent_case_id,
        kpi=linked_kpi(playbook, layer),
        scope=spec.scope_for(template.scope_hint, scope),
        grain=grain,
        period=period,
        driver=template.driver,
        question=template.question.strip(),
        rationale=template.rationale,
        playbook=playbook.playbook,
        opened_at=opened_at,
    )
    register_case(
        connection,
        case_id=linked.case_id,
        kpi=linked.kpi,
        scope=linked.scope,
        grain=linked.grain,
        period=linked.period,
        opened_at=linked.opened_at,
        status=spec.status,
        linked_from_case_id=parent_case_id,
    )
    return linked


# ---------------------------------------------------------------------------
# The data gap register
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceGap:
    """One source the organisation does not hold, priced if it can be."""

    source: str
    hypothesis: str
    case_id: str
    acquisition_playbook: str | None
    acquisition_cost: Cost | None
    resolution_hint: str

    def render(self, layer: SemanticLayer) -> str:
        return f"  {self.source} (for {self.hypothesis}): {self.resolution_hint}"


def acquisition_playbook(
    source: str, driver: str, layer: SemanticLayer
) -> Playbook | None:
    """The playbook that would buy this source, if one exists.

    Matched through the driver, not through the source name: a playbook
    says which hypothesis it makes testable, and the causal graph says
    which sources that hypothesis is missing. Nothing has to maintain a
    second source-to-playbook table that could disagree with the first.

    ONLY A PLAYBOOK THAT ACQUIRES THE SOURCE COUNTS. A manager call-down
    is a fine investigation and it is cheaper than anything else on the
    page, but it does not put a competitor price feed in the warehouse. A
    gap in the register whose resolution hint is a lever that cannot close
    it is worse than a gap with no hint at all, because it reads as
    solved.
    """
    template = layer.causal_graph.hypotheses.get(driver)
    if template is None or source not in template.missing_sources():
        return None
    lever = layer.recommend.data_gap.acquisition_lever
    for playbook in candidates(driver, layer):
        if playbook.lever == lever:
            return playbook
    return None


def source_gaps(
    hypothesis: str,
    missing_sources: tuple[str, ...],
    basis: CostBasis,
    layer: SemanticLayer,
    *,
    case_id: str,
) -> tuple[SourceGap, ...]:
    """Turn T3's set difference into priced, registrable gaps."""
    spec = layer.recommend.data_gap
    gaps: list[SourceGap] = []
    for source in missing_sources:
        playbook = acquisition_playbook(source, hypothesis, layer)
        cost: Cost | None = None
        if playbook is not None:
            try:
                cost = compute_cost(playbook, basis, layer)
            except CostError:
                cost = None
        if playbook is None:
            hint = " ".join(spec.no_playbook_hint.split())
        elif cost is None:
            hint = (
                f"{playbook.playbook} would acquire it; it cannot be priced against "
                "this case's quantities."
            )
        else:
            hint = (
                f"{playbook.playbook}: INR {cost.lakh(layer):.2f} L, "
                f"{playbook.lead_time.value:g} {playbook.lead_time.unit}."
            )
        gaps.append(
            SourceGap(
                source=source,
                hypothesis=hypothesis,
                case_id=case_id,
                acquisition_playbook=playbook.playbook if playbook else None,
                acquisition_cost=cost,
                resolution_hint=hint,
            )
        )
    return tuple(gaps)


def write_source_gaps(
    connection: duckdb.DuckDBPyConnection,
    gaps: tuple[SourceGap, ...],
    layer: SemanticLayer,
    *,
    scope: str,
    detected_at: datetime,
) -> int:
    """Record the gaps. Returns rows written.

    Keyed by gap code, source and scope rather than by case, so the second
    competitor case in the same region updates the entry instead of adding
    a duplicate. The register is a list of things that are true about the
    business, not a log of every time one of them bit.
    """
    spec = layer.recommend.data_gap
    rows = []
    for gap in gaps:
        detail = " ".join(
            spec.detail_template.format(
                hypothesis=gap.hypothesis,
                source=gap.source,
                case_id=gap.case_id,
                trigger=spec.trigger,
            ).split()
        )
        rows.append(
            DataGap(
                gap_id="-".join((spec.gap_code, gap.source, scope)),
                detected_at=detected_at,
                gap_code=spec.gap_code,
                severity=spec.severity,
                subject=gap.source,
                scope=scope,
                measure=spec.measure,
                value_numeric=spec.value_when_missing,
                unit=spec.unit,
                detail=detail,
                resolution_hint=gap.resolution_hint,
            ).as_row()
        )
    if not rows:
        return 0
    connection.executemany(
        "DELETE FROM data_gap_register WHERE gap_id = ?",
        [(row[0],) for row in rows],
    )
    connection.executemany(GAP_INSERT, rows)
    return len(rows)


__all__ = [
    "CASE_REGISTRY",
    "LinkedCase",
    "LinkedCaseError",
    "SourceGap",
    "acquisition_playbook",
    "create_linked_case",
    "linked_kpi",
    "next_case_id",
    "source_gaps",
    "write_source_gaps",
]
