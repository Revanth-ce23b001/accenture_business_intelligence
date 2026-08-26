"""The causal graph's priors, and what feedback does to them.

    declared      causal_graph.yaml, written by a person, in review
    observed      hypothesis_prior, counted from driver-level feedback
    effective     the Beta posterior of the two, bounded

TWO FILES, ONE OF THEM GENERATED. `causal_graph.yaml` is the contract and
nothing here ever writes to it. `semantic_layer/runtime/priors.yaml` is
the RUNTIME OVERLAY: a materialised view of what the counts have done to
the declared priors, rewritten whenever they change, and safe to delete —
it can always be regenerated from the warehouse, which is the source of
truth.

WHY AN OVERLAY FILE AT ALL, WHEN THE WAREHOUSE IS AUTHORITATIVE. Because
"the system has quietly decided stock-outs are less likely than we wrote
down" is exactly the kind of change that should be visible in a diff and
reviewable in a pull request. A learned prior that lives only in a
database column is a change to the engine's judgement that no code review
will ever see.

PER HYPOTHESIS PER KPI. "Stock-outs are usually the answer" is a claim
about a KPI, not about the business — they explain availability movements
far more often than they explain average selling price. One pooled count
would blur the two into a number true of neither.

THE UPDATE IS BOUNDED, AND THE BOUND IS THE POINT. A loop wired to a
button can be moved by whoever clicks most. `learning.yaml -> priors`
gives the declared prior a strength in pseudo-observations and caps how
far the posterior may move from it in absolute probability. Past the cap
the graph has to be edited by a person, which is the correct escalation:
the system may notice, and a human decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from engine.learn.rows import fetch_dicts
from semantic_layer.overlay import OVERLAY_DIR, OVERLAY_PATH
from semantic_layer.schema import SemanticLayer, get_semantic_layer

COUNTS_SQL = "SELECT kpi, hypothesis, confirmed, rejected FROM hypothesis_prior"

COUNTS_FOR_KPI = (
    "SELECT kpi, hypothesis, confirmed, rejected FROM hypothesis_prior WHERE kpi = ?"
)

#: One statement, both directions. A hypothesis nobody has judged before
#: has no row; the first judgement creates it.
UPSERT = """
INSERT INTO hypothesis_prior
    (kpi, hypothesis, confirmed, rejected, first_seen_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT (kpi, hypothesis) DO UPDATE SET
    confirmed  = hypothesis_prior.confirmed + excluded.confirmed,
    rejected   = hypothesis_prior.rejected  + excluded.rejected,
    updated_at = excluded.updated_at
"""


class PriorError(RuntimeError):
    """The prior cannot be resolved as asked."""


@dataclass(frozen=True)
class LearnedPrior:
    """One hypothesis's prior for one KPI, and everything behind it."""

    kpi: str
    hypothesis: str
    declared: float
    confirmed: int
    rejected: int
    effective: float

    @property
    def observations(self) -> int:
        return self.confirmed + self.rejected

    @property
    def shift(self) -> float:
        """How far feedback moved it. Negative means rejections dominated."""
        return self.effective - self.declared

    @property
    def moved(self) -> bool:
        return self.effective != self.declared

    def render(self) -> str:
        arrow = "->" if self.moved else "=="
        return (
            f"{self.kpi}/{self.hypothesis}: {self.declared:.4f} {arrow} "
            f"{self.effective:.4f} ({self.confirmed}+/{self.rejected}-)"
        )


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def observe(
    connection: duckdb.DuckDBPyConnection,
    *,
    kpi: str,
    hypothesis: str,
    confirmed: bool,
    at: datetime | None = None,
) -> None:
    """Record one driver judgement. Additive and idempotent per event.

    Not a set: two people rejecting the same driver on the same case is
    two observations, and the arithmetic should say so. Deduplicating by
    case would silently weight a case with one reader the same as a case
    ten people looked at.
    """
    from engine.db import as_stored_timestamp

    moment = as_stored_timestamp(at or datetime.now(UTC))
    connection.execute(
        UPSERT,
        [kpi, hypothesis, int(confirmed), int(not confirmed), moment, moment],
    )


def counts(
    connection: duckdb.DuckDBPyConnection, *, kpi: str | None = None
) -> dict[tuple[str, str], tuple[int, int]]:
    """`{(kpi, hypothesis): (confirmed, rejected)}`."""
    rows = (
        fetch_dicts(connection, COUNTS_FOR_KPI, [kpi])
        if kpi
        else fetch_dicts(connection, COUNTS_SQL)
    )
    return {
        (str(row["kpi"]), str(row["hypothesis"])): (
            int(row["confirmed"]),
            int(row["rejected"]),
        )
        for row in rows
    }


# ---------------------------------------------------------------------------
# Resolving
# ---------------------------------------------------------------------------


def effective_priors(
    connection: duckdb.DuckDBPyConnection,
    *,
    kpi: str,
    layer: SemanticLayer | None = None,
) -> dict[str, LearnedPrior]:
    """Every hypothesis that affects `kpi`, with its prior as it now stands.

    This is what GATHER's screening reads. A hypothesis nobody has judged
    comes back with its declared prior and zero observations, which is the
    correct answer and not a missing one.
    """
    layer = layer or get_semantic_layer()
    spec = layer.learning.priors
    observed = counts(connection, kpi=kpi)

    resolved: dict[str, LearnedPrior] = {}
    for tag, template in layer.causal_graph.hypotheses.items():
        if kpi not in template.affects:
            continue
        confirmed, rejected = observed.get((kpi, tag), (0, 0))
        resolved[tag] = LearnedPrior(
            kpi=kpi,
            hypothesis=tag,
            declared=template.prior,
            confirmed=confirmed,
            rejected=rejected,
            effective=spec.posterior(template.prior, confirmed, rejected),
        )
    return resolved


def effective_prior(
    connection: duckdb.DuckDBPyConnection,
    *,
    kpi: str,
    hypothesis: str,
    layer: SemanticLayer | None = None,
) -> LearnedPrior:
    """One hypothesis's effective prior for one KPI."""
    layer = layer or get_semantic_layer()
    template = layer.causal_graph.hypotheses.get(hypothesis)
    if template is None:
        raise PriorError(
            f"{hypothesis!r} is not a hypothesis in the causal graph; known: "
            f"{sorted(layer.causal_graph.hypotheses)}"
        )
    confirmed, rejected = counts(connection, kpi=kpi).get((kpi, hypothesis), (0, 0))
    return LearnedPrior(
        kpi=kpi,
        hypothesis=hypothesis,
        declared=template.prior,
        confirmed=confirmed,
        rejected=rejected,
        effective=layer.learning.priors.posterior(template.prior, confirmed, rejected),
    )


def all_learned(
    connection: duckdb.DuckDBPyConnection, layer: SemanticLayer | None = None
) -> tuple[LearnedPrior, ...]:
    """Every (kpi, hypothesis) pair anyone has judged, ordered."""
    layer = layer or get_semantic_layer()
    spec = layer.learning.priors
    graph = layer.causal_graph.hypotheses

    learned: list[LearnedPrior] = []
    for (kpi, tag), (confirmed, rejected) in sorted(counts(connection).items()):
        template = graph.get(tag)
        if template is None:
            continue
        learned.append(
            LearnedPrior(
                kpi=kpi,
                hypothesis=tag,
                declared=template.prior,
                confirmed=confirmed,
                rejected=rejected,
                effective=spec.posterior(template.prior, confirmed, rejected),
            )
        )
    return tuple(learned)


# ---------------------------------------------------------------------------
# The overlay file
# ---------------------------------------------------------------------------

def write_overlay(
    connection: duckdb.DuckDBPyConnection,
    *,
    path: Path | None = None,
    layer: SemanticLayer | None = None,
    generated_at: datetime | None = None,
) -> Path | None:
    """Materialise the overlay. Returns the path, or None if it could not.

    BEST EFFORT, AND DELIBERATELY SO. The warehouse is the source of
    truth; this file is a view of it. A read-only filesystem, a container
    without a writable mount, a permissions problem — none of those should
    fail a reader's feedback, because the counts have already been
    recorded and the overlay can be regenerated at any time.
    """
    from semantic_layer.overlay import write_priors_overlay

    layer = layer or get_semantic_layer()
    spec = layer.learning.priors
    learned = all_learned(connection, layer)

    entries = {
        kpi: {
            entry.hypothesis: {
                "declared": entry.declared,
                "effective": entry.effective,
                "shift": entry.shift,
                "confirmed": entry.confirmed,
                "rejected": entry.rejected,
                "capped": abs(entry.shift) >= spec.max_absolute_shift,
            }
            for entry in learned
            if entry.kpi == kpi
        }
        for kpi in sorted({entry.kpi for entry in learned})
    }
    return write_priors_overlay(
        entries,
        path=path,
        generated_at=generated_at or datetime.now(UTC),
        rule={
            "strength": spec.strength,
            "min_observations": spec.min_observations,
            "max_absolute_shift": spec.max_absolute_shift,
        },
    )


def read_overlay(path: Path | None = None) -> dict:
    """The overlay as it stands on disk. `{}` when it has never been written."""
    from semantic_layer.overlay import read_priors_overlay

    return read_priors_overlay(path)


__all__ = [
    "COUNTS_SQL",
    "OVERLAY_DIR",
    "OVERLAY_PATH",
    "UPSERT",
    "LearnedPrior",
    "PriorError",
    "all_learned",
    "counts",
    "effective_prior",
    "effective_priors",
    "observe",
    "read_overlay",
    "write_overlay",
]
