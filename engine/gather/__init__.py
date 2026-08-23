"""GATHER — three lanes, run before anything is adjudicated.

CLAUDE.md §Architecture:

    Data -> VALIDATE -> QUALIFY -> GATHER -> ADJUDICATE -> VERDICT

QUALIFY produced a residual worth explaining. GATHER screens the
candidates to five and then finds the evidence for all five at once:

    structured    parameterised queries, one template per hypothesis type,
                  every one through engine/db.py::execute_governed
    unstructured  BM25 plus in-memory embeddings, then a batched Haiku pass
                  that assigns a TAG. Counting is Python.
    external      the calendar, the weather, the competitor news feed

It forms no opinion. Whether a hypothesis is supported, eliminated or
leading is ADJUDICATE's question, and it gets the evidence cold.

    from engine.gather import GatherRequest, gather

    result = gather(connection, user, GatherRequest(...))
    result.hypotheses          # the five candidates, scored
    result.cache_hit_rate      # classification served without a model call
    result.evidence            # every figure, as Evidence
"""

from engine.gather.external import ExternalResult, FeedResult, gather_external
from engine.gather.gate import GatherError, GatherRequest, GatherResult, gather
from engine.gather.guard import NO_GUARD, Guard, LockGuard
from engine.gather.hypotheses import Candidate, Screening, prior_case_counts, screen
from engine.gather.retrieval import Bm25, Embeddings, HybridIndex, Ranked, tokenise
from engine.gather.structured import StructuredResult, gather_structured
from engine.gather.unstructured import (
    CONTROL,
    TREATED,
    ChiSquare,
    CorpusResult,
    GroupCount,
    UnstructuredResult,
    chi_square,
    count_by_group,
    gather_unstructured,
    load_documents,
)

__all__ = [
    "CONTROL",
    "NO_GUARD",
    "TREATED",
    "Bm25",
    "Candidate",
    "ChiSquare",
    "CorpusResult",
    "Embeddings",
    "ExternalResult",
    "FeedResult",
    "GatherError",
    "GatherRequest",
    "GatherResult",
    "GroupCount",
    "Guard",
    "HybridIndex",
    "LockGuard",
    "Ranked",
    "Screening",
    "StructuredResult",
    "UnstructuredResult",
    "chi_square",
    "count_by_group",
    "gather",
    "gather_external",
    "gather_structured",
    "gather_unstructured",
    "load_documents",
    "prior_case_counts",
    "screen",
    "tokenise",
]
