"""Lane 1 — the structured lane.

One parameterised template per hypothesis type, every one run through
`engine/db.py::execute_governed`. A persona who cannot see a scope cannot
gather evidence about it either, and that is not a limitation to work
around — it is the point.

The templates live in `semantic_layer/gather.yaml`. This module binds five
parameters, runs the statement, and aggregates one declared measure. It
writes no SQL of its own, and it does not decide what a result means:
GATHER brings evidence back, ADJUDICATE forms the opinion.

A hypothesis with no template is DECLARED as having none, with the reason.
`competitor_action` is the case that matters: the two sources that would
test it are not held, so a structured query would be a number manufactured
from absent data. "We have no query for this" and "we forgot to write one"
look identical in an empty result, so the difference is written down.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import fmean

import duckdb

from engine.contracts import Evidence
from engine.db import GovernanceError, execute_governed
from engine.evidence import EvidenceFactory
from engine.gather.guard import NO_GUARD, Guard
from security.policy import User
from semantic_layer.schema import SemanticLayer, StructuredTemplate

QUERY_KIND = "structured_query"

_AGGREGATES = {
    "mean": fmean,
    "sum": sum,
    "count": len,
    "min": min,
    "max": max,
}


@dataclass(frozen=True)
class StructuredResult:
    """What one template found, or why it found nothing."""

    hypothesis: str
    available: bool
    reason: str
    rows: int = 0
    value: float | None = None
    unit: str | None = None
    evidence: tuple[Evidence, ...] = ()
    statement: str | None = None


def gather_structured(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
    factory: EvidenceFactory,
    *,
    kpi: str,
    scope: str,
    period_start: date,
    period_end: date,
    comparison_start: date,
    comparison_end: date,
    hypotheses: tuple[str, ...],
    guard: Guard = NO_GUARD,
) -> tuple[StructuredResult, ...]:
    """Run the template for each candidate that has one."""
    spec = layer.gather.structured
    results: list[StructuredResult] = []

    for tag in hypotheses:
        template = spec.templates.get(tag)
        if template is None:
            results.append(
                StructuredResult(
                    hypothesis=tag,
                    available=False,
                    reason=spec.unavailable.get(
                        tag,
                        "no structured template is declared for this hypothesis type",
                    ),
                )
            )
            continue
        results.append(
            _run_template(
                connection, user, layer, factory, tag, template,
                kpi=kpi,
                scope=scope,
                period_start=period_start,
                period_end=period_end,
                comparison_start=comparison_start,
                comparison_end=comparison_end,
                guard=guard,
            )
        )
    return tuple(results)


def _run_template(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
    factory: EvidenceFactory,
    tag: str,
    template: StructuredTemplate,
    *,
    kpi: str,
    scope: str,
    period_start: date,
    period_end: date,
    comparison_start: date,
    comparison_end: date,
    guard: Guard = NO_GUARD,
) -> StructuredResult:
    params = {
        "scope": scope,
        "period_start": period_start,
        "period_end": period_end,
        "comparison_start": comparison_start,
        "comparison_end": comparison_end,
    }
    statement = template.sql.strip()
    bound = {name: value for name, value in params.items() if f"${name}" in statement}

    try:
        with guard():
            rows, _filtered, _masked = execute_governed(
                user,
                kpi,
                statement,
                bound,
                connection=connection,
                layer=layer,
                purpose=f"gather.structured.{tag}",
            )
    except GovernanceError as exc:
        # A scope the caller cannot see, or a table with no column for the
        # policy to bind to. Reported, never treated as an absence.
        return StructuredResult(
            hypothesis=tag,
            available=False,
            reason=f"not readable at this persona's scope: {exc}",
            statement=statement,
        )

    values = [
        float(row[template.measure])
        for row in rows
        if row.get(template.measure) is not None
    ]
    if not values:
        return StructuredResult(
            hypothesis=tag,
            available=True,
            reason=f"the template returned no {template.measure} for {scope}",
            rows=len(rows),
            statement=statement,
        )

    aggregate = _AGGREGATES[template.aggregate](values)
    value = round(float(aggregate), layer.gather.structured.measure_places)

    evidence = factory.emit(
        f"structured.{tag}",
        kind=QUERY_KIND,
        label=f"{template.label} — {scope}",
        value=value,
        unit=template.unit,
        source_system=template.source_system,
        method="sql",
        description=(
            f"{template.aggregate} of {template.measure} over {len(values)} rows, "
            f"{period_start} to {period_end}"
        ),
        ref=f"semantic_layer/gather.yaml::structured.templates.{tag}",
        inputs=(template.source_system,),
        statement=statement,
        notes=(
            "Gathered, not judged. Whether this supports the hypothesis is "
            "ADJUDICATE's question."
        ),
    )
    return StructuredResult(
        hypothesis=tag,
        available=True,
        reason=f"{len(values)} rows aggregated by {template.aggregate}",
        rows=len(rows),
        value=value,
        unit=template.unit,
        evidence=(evidence,),
        statement=statement,
    )


__all__ = ["StructuredResult", "gather_structured"]
