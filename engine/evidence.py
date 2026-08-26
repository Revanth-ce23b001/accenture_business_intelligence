"""The evidence engine: how every number in CaseFile gets made.

CLAUDE.md rule 3 — "every number in the UI must be clickable to its
evidence. No bare floats leave the engine; modules emit `Evidence`
objects." This module is how they do it, and it is deliberately the only
way: a stage that wants to publish a number asks the factory, and the
factory refuses to mint a record that could not be chased.

What it refuses:

  * a `source_system` that is neither a source in the warehouse registry
    nor one of its declared non-source origins. A number whose system is
    unknown cannot be chased when it turns out to be wrong.
  * a `method` outside the declared vocabulary. "How was this made" with a
    free-text answer is not an answer.
  * a record with no lineage. A number with nothing behind it is exactly
    the thing rule 3 exists to prevent.
  * a `produced_by="model"` record carrying a number (rule 1, enforced on
    the contract itself).

What it insists on:

  * `source_as_of` is the DATA timestamp, taken from the warehouse clock —
    never `now()`. `retrieved_at` is the wall clock, carried separately so
    the two can never be confused. A stale feed shown with today's date
    beside it reads as fresh, and nobody looks again.
  * the SQL is kept VERBATIM, not hashed. `audit_log` stores a hash
    because it is a different table with different access rules; the
    statement that produced a number travels with the number.
  * the reliability weight comes from `semantic_layer/adjudication.yaml`
    and is never passed in. A caller that could choose its own weight
    could choose to be believed.

And the floor rule, which is the reason the weights exist at all:

    a hypothesis whose evidence is dominated by weights below the floor
    cannot be promoted to EXPLAINED.

Text alone never reaches a verdict. Text plus a matched control does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable, Iterator, Mapping, Sequence

import duckdb

from engine.contracts import Evidence, EvidenceKind, LineageStep, ProducedBy
from semantic_layer.schema import SemanticLayer, get_semantic_layer

#: Separator for list-valued columns in the `evidence` table. A method
#: name, an origin and an evidence id all exclude it.
LIST_SEPARATOR = ","

#: `evidence.value` is one column in the contract and two in the table:
#: a number belongs in `value_numeric` so it can be aggregated, and a
#: string in `value_text` so it is not silently coerced to NaN.
EVIDENCE_INSERT = (
    "INSERT INTO evidence (evidence_id, case_id, kind, produced_by, label, "
    "value_numeric, value_text, unit, reliability, source_system, method, source_ref, "
    "source_as_of, retrieved_at, freshness_hours, completeness, assumptions, "
    "lineage_json, notes) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


class EvidenceError(ValueError):
    """A value was offered that could not be turned into evidence."""


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def reliability_weight(kind: EvidenceKind | str, layer: SemanticLayer | None = None) -> float:
    """The weight for an evidence tier, from the semantic layer.

    CLAUDE.md §"Reliability weights for evidence":

        structured warehouse query 0.95 · derived statistical estimate 0.85 ·
        corroborated unstructured, n>=20 independent notes 0.80 · support
        ticket aggregate 0.75 · single store note 0.55 · external news item
        0.45 · social/review mention 0.35

    The numbers live in `semantic_layer/adjudication.yaml`. This function
    exists so nothing in `engine/` ever writes one down (rule 2).
    """
    layer = layer or get_semantic_layer()
    try:
        return layer.adjudication.reliability.weights[kind]  # type: ignore[index]
    except KeyError as exc:
        raise EvidenceError(
            f"no reliability weight for evidence kind {kind!r}; known kinds: "
            f"{sorted(layer.adjudication.reliability.weights)}"
        ) from exc


def reliability_floor(layer: SemanticLayer | None = None) -> float:
    """Below this, evidence cannot carry a hypothesis on its own."""
    layer = layer or get_semantic_layer()
    return layer.adjudication.reliability.floor


# ---------------------------------------------------------------------------
# The floor rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityProfile:
    """What a hypothesis's evidence is actually made of.

    The floor rule is judged on the WEIGHTED share, not the count. One
    matched-control estimate and three social mentions is not a
    three-to-one case for the mentions: the estimate carries more because
    it is worth more, and that is the whole point of having weights. The
    count share is reported too, because it is what a reader will notice
    first.
    """

    total_weight: float
    low_weight: float
    total_count: int
    low_count: int
    floor: float
    max_low_share: float
    kinds: tuple[str, ...]

    @property
    def low_weight_share(self) -> float:
        return self.low_weight / self.total_weight if self.total_weight else 1.0

    @property
    def low_count_share(self) -> float:
        return self.low_count / self.total_count if self.total_count else 1.0

    @property
    def dominated(self) -> bool:
        """True when weak evidence carries more than its allowed share.

        Evidence-free is dominated. A hypothesis nothing supports cannot be
        promoted to EXPLAINED on the grounds that nothing contradicts it
        either.
        """
        if not self.total_count:
            return True
        return self.low_weight_share > self.max_low_share

    def explain(self) -> str:
        if not self.total_count:
            return "no evidence at all, so nothing to be dominated by."
        return (
            f"{self.low_count} of {self.total_count} records sit below the "
            f"{self.floor:.2f} reliability floor and carry "
            f"{self.low_weight_share:.0%} of the total weight "
            f"(limit {self.max_low_share:.0%})."
        )


def reliability_profile(
    records: Iterable[Evidence], layer: SemanticLayer | None = None
) -> ReliabilityProfile:
    """Weigh a hypothesis's evidence against the floor."""
    layer = layer or get_semantic_layer()
    spec = layer.adjudication.reliability
    trigger = layer.adjudication.triggers.get("T6")
    max_low_share = (
        trigger.max_low_reliability_share
        if trigger is not None and trigger.max_low_reliability_share is not None
        else spec.floor
    )
    floor = (
        trigger.reliability_floor
        if trigger is not None and trigger.reliability_floor is not None
        else spec.floor
    )

    items = tuple(records)
    weights = [item.reliability for item in items]
    low = [weight for weight in weights if weight < floor]
    return ReliabilityProfile(
        total_weight=sum(weights),
        low_weight=sum(low),
        total_count=len(weights),
        low_count=len(low),
        floor=floor,
        max_low_share=max_low_share,
        kinds=tuple(sorted({item.kind for item in items})),
    )


@dataclass(frozen=True)
class PromotionCheck:
    """Whether a hypothesis may be promoted to EXPLAINED."""

    allowed: bool
    profile: ReliabilityProfile
    trigger: str | None
    confidence_ceiling: float | None
    detail: str


def can_promote_to_explained(
    records: Iterable[Evidence], layer: SemanticLayer | None = None
) -> PromotionCheck:
    """The floor rule.

    CLAUDE.md §"Reliability weights for evidence":

        Floor rule: a hypothesis whose evidence is dominated by weights
        < 0.6 cannot be promoted to EXPLAINED (trigger T6). Text alone
        never reaches a verdict; text plus a matched control does.

    Returns the decision, the profile it was made on, the trigger that
    fires, and the confidence ceiling that comes with it. Nothing here
    decides a verdict — it decides what the verdict table is allowed to
    consider.
    """
    layer = layer or get_semantic_layer()
    profile = reliability_profile(records, layer)
    if not profile.dominated:
        return PromotionCheck(
            allowed=True,
            profile=profile,
            trigger=None,
            confidence_ceiling=None,
            detail=(
                f"{profile.low_count} of {profile.total_count} records are below the "
                f"{profile.floor:.2f} floor, carrying {profile.low_weight_share:.0%} of "
                "the weight. Not dominated."
            ),
        )

    cap = layer.adjudication.confidence.caps.get("low_reliability_dominated")
    return PromotionCheck(
        allowed=False,
        profile=profile,
        trigger="T6",
        confidence_ceiling=cap.ceiling if cap is not None else None,
        detail=(
            "cannot be promoted to EXPLAINED: " + profile.explain() +
            " Text alone does not reach a verdict."
        ),
    )


# ---------------------------------------------------------------------------
# Minting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceFactory:
    """Mints `Evidence` for one stage of one run.

    Bound to a semantic layer and to a single `source_as_of` — the data
    timestamp for the whole run — so that every number a stage publishes
    agrees about how current the data was.
    """

    layer: SemanticLayer
    stage: str
    source_as_of: datetime
    retrieved_at: datetime | None = None

    @classmethod
    def for_stage(
        cls,
        stage: str,
        source_as_of: datetime,
        layer: SemanticLayer | None = None,
        *,
        retrieved_at: datetime | None = None,
    ) -> EvidenceFactory:
        return cls(
            layer=layer or get_semantic_layer(),
            stage=stage,
            source_as_of=source_as_of,
            retrieved_at=retrieved_at or datetime.now(UTC),
        )

    # -- the one way to make a number public ------------------------------

    def emit(
        self,
        key: str,
        *,
        kind: EvidenceKind,
        label: str,
        value: float | int | str | None,
        unit: str,
        source_system: str,
        method: str,
        description: str,
        ref: str,
        inputs: Sequence[str] = (),
        statement: str | None = None,
        steps: Sequence[LineageStep] = (),
        freshness_hours: float = 0.0,
        completeness: float = 1.0,
        assumptions: Sequence[str] = (),
        notes: str | None = None,
        produced_by: ProducedBy = "code",
        source_as_of: datetime | None = None,
    ) -> Evidence:
        """Build one record, or refuse to.

        `key` is the stage-local name; the evidence id is `stage.key`, so
        two stages cannot collide and a reader can tell at a glance which
        stage produced a number.

        `statement` is retained verbatim when the value came from SQL.
        """
        self._check_origin(source_system)
        self._check_method(method)

        lineage = tuple(steps) or (
            LineageStep(
                step=0,
                operation=method,
                description=description,
                inputs=tuple(inputs),
                ref=ref,
                statement=_normalise(statement),
            ),
        )
        return Evidence(
            evidence_id=f"{self.stage}.{key}",
            kind=kind,
            produced_by=produced_by,
            label=label,
            value=value,
            unit=unit,
            reliability=reliability_weight(kind, self.layer),
            source_system=source_system,
            method=method,
            source_ref=ref,
            source_as_of=source_as_of or self.source_as_of,
            retrieved_at=self.retrieved_at,
            freshness_hours=freshness_hours,
            completeness=completeness,
            lineage=lineage,
            assumptions=tuple(assumptions),
            notes=notes,
        )

    def step(
        self,
        step: int,
        *,
        operation: str,
        description: str,
        inputs: Sequence[str] = (),
        ref: str | None = None,
        statement: str | None = None,
    ) -> LineageStep:
        """One step of a multi-step derivation."""
        self._check_method(operation)
        return LineageStep(
            step=step,
            operation=operation,
            description=description,
            inputs=tuple(inputs),
            ref=ref,
            statement=_normalise(statement),
        )

    # -- refusals ---------------------------------------------------------

    def _check_origin(self, source_system: str) -> None:
        known = self.layer.warehouse.known_origins()
        if source_system not in known:
            raise EvidenceError(
                f"{source_system!r} is not a known evidence origin. It must be a source "
                f"in warehouse.yaml -> sources, or one of its non-source origins: "
                f"{sorted(known)}. A number whose system is unknown cannot be chased "
                "when it is wrong."
            )

    def _check_method(self, method: str) -> None:
        known = self.layer.warehouse.evidence.methods
        if method not in known:
            raise EvidenceError(
                f"{method!r} is not a declared method. Known: {sorted(known)}. "
                "'How was this made' with a free-text answer is not an answer."
            )


def _normalise(statement: str | None) -> str | None:
    """Keep the statement verbatim, minus the indentation of the source file."""
    if statement is None:
        return None
    lines = [line.rstrip() for line in statement.strip("\n").splitlines()]
    body = [line for line in lines if line.strip()]
    if not body:
        return None
    indent = min(len(line) - len(line.lstrip()) for line in body)
    return "\n".join(line[indent:] if line.strip() else "" for line in lines).strip()


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


@dataclass
class EvidenceLedger:
    """Every record a stage produced, in the order it produced them.

    Ids are unique by construction: the second attempt to register the same
    id is a programming error, not a last-write-wins.
    """

    records: list[Evidence] = field(default_factory=list)
    _index: dict[str, Evidence] = field(default_factory=dict, repr=False)

    def add(self, record: Evidence) -> Evidence:
        if record.evidence_id in self._index:
            raise EvidenceError(
                f"evidence id {record.evidence_id!r} was already registered; two numbers "
                "sharing an id means one of them cannot be clicked through to"
            )
        self._index[record.evidence_id] = record
        self.records.append(record)
        return record

    def extend(self, records: Iterable[Evidence]) -> None:
        for record in records:
            self.add(record)

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._index

    def __getitem__(self, evidence_id: str) -> Evidence:
        return self._index[evidence_id]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(record.evidence_id for record in self.records)

    def by_id(self) -> dict[str, Evidence]:
        return dict(self._index)

    def unresolved_inputs(self) -> tuple[str, ...]:
        """Lineage inputs that look like evidence ids but are not in the ledger.

        A dangling `inputs` entry is a broken link in the UI: the reader
        clicks and lands nowhere.
        """
        missing: set[str] = set()
        for record in self.records:
            for lineage in record.lineage:
                for name in lineage.inputs:
                    if "." in name and name not in self._index:
                        missing.add(name)
        return tuple(sorted(missing))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def as_row(record: Evidence, case_id: str | None) -> tuple:
    """Positional values for the `evidence` insert.

    The four columns the P7 brief requires to be non-null — `source_system`,
    `source_as_of`, `method` and the lineage — are non-null on the contract,
    so a row that reached here cannot be missing them. The table declares
    them NOT NULL as well, which is belt and braces on purpose: the
    contract protects what the engine writes, and the column protects what
    anything else does.
    """
    from engine.db import as_stored_timestamp

    numeric = record.value if isinstance(record.value, (int, float)) else None
    text = record.value if isinstance(record.value, str) else None
    return (
        record.evidence_id,
        case_id,
        record.kind,
        record.produced_by,
        record.label,
        None if isinstance(numeric, bool) else numeric,
        text,
        record.unit,
        record.reliability,
        record.source_system,
        record.method,
        record.source_ref,
        as_stored_timestamp(record.source_as_of),
        as_stored_timestamp(record.retrieved_at) if record.retrieved_at else None,
        record.freshness_hours,
        record.completeness,
        LIST_SEPARATOR.join(record.assumptions) or None,
        lineage_json(record),
        record.notes,
    )


def lineage_json(record: Evidence) -> str:
    """The lineage, including any verbatim statement, as JSON."""
    return json.dumps(
        [step.model_dump(mode="json") for step in record.lineage],
        sort_keys=True,
    )


def persist(
    connection: duckdb.DuckDBPyConnection,
    records: Iterable[Evidence],
    *,
    case_id: str | None = None,
) -> int:
    """Write evidence to the warehouse. Returns rows written."""
    rows = [as_row(record, case_id) for record in records]
    if rows:
        connection.executemany(EVIDENCE_INSERT, rows)
    return len(rows)


def from_row(row: Mapping[str, Any]) -> Evidence:
    """Rebuild the contract from a stored row, lineage and all.

    The inverse of `as_row`, and it lives here for the same reason
    `as_row` does: the evidence engine owns what an evidence record IS,
    and a second reconstruction elsewhere would be a second opinion about
    it. `engine/db.py` does not know this shape and the API should not
    have to.
    """
    lineage = json.loads(row["lineage_json"] or "[]")
    assumptions = row["assumptions"] or ""
    numeric, text = row["value_numeric"], row["value_text"]
    return Evidence(
        evidence_id=str(row["evidence_id"]),
        kind=str(row["kind"]),
        produced_by=str(row["produced_by"]),
        label=str(row["label"]),
        value=numeric if numeric is not None else text,
        unit=str(row["unit"]) if row["unit"] is not None else None,
        reliability=float(row["reliability"]),
        source_system=str(row["source_system"]),
        method=str(row["method"]),
        source_ref=str(row["source_ref"]),
        source_as_of=row["source_as_of"],
        retrieved_at=row["retrieved_at"],
        freshness_hours=float(row["freshness_hours"]),
        completeness=float(row["completeness"]),
        lineage=tuple(LineageStep.model_validate(step) for step in lineage),
        assumptions=tuple(
            part for part in assumptions.split(LIST_SEPARATOR) if part
        ),
        notes=str(row["notes"]) if row["notes"] is not None else None,
    )


def load(
    connection: duckdb.DuckDBPyConnection, case_id: str | None = None
) -> tuple[Mapping[str, Any], ...]:
    """Read evidence rows back, for a case or for everything."""
    where = " WHERE case_id = $case" if case_id else ""
    params = {"case": case_id} if case_id else {}
    cursor = connection.execute(
        "SELECT evidence_id, case_id, kind, produced_by, label, value_numeric, "
        "value_text, unit, reliability, source_system, method, source_ref, "
        "source_as_of, retrieved_at, freshness_hours, completeness, assumptions, "
        f"lineage_json, notes FROM evidence{where} ORDER BY evidence_id",
        params or None,
    )
    columns = [description[0] for description in cursor.description]
    return tuple(
        dict(zip(columns, row, strict=True)) for row in cursor.fetchall()
    )


__all__ = [
    "EVIDENCE_INSERT",
    "EvidenceError",
    "EvidenceFactory",
    "EvidenceLedger",
    "PromotionCheck",
    "ReliabilityProfile",
    "as_row",
    "from_row",
    "can_promote_to_explained",
    "lineage_json",
    "load",
    "persist",
    "reliability_floor",
    "reliability_profile",
    "reliability_weight",
]
