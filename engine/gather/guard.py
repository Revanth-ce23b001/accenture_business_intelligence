"""The lock the three lanes share, and why they need one.

One DuckDB connection is one transaction. Two threads reading it at once
is not safe, and a cursor is not the answer either: a cursor does not see
its parent's uncommitted writes, so a lane running inside a transaction
would silently read the wrong world.

So the lanes overlap, and their warehouse reads do not. What actually runs
in parallel is the work that is not the warehouse — BM25 scoring, the SVD,
JSON parsing, and above all the model calls, which are network-bound and
are most of the wall clock on a live run.

`NO_GUARD` is the default so a lane can be called directly in a test
without inventing a lock it does not need.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager, nullcontext

# The type is DEFINED in llm/classify.py, which the classifier needs and
# which must not import engine/. Re-exported here so the engine has one
# name for it and there is still only one definition.
from llm.classify import Guard


def no_guard() -> AbstractContextManager:
    """No serialisation. For single-threaded callers and tests."""
    return nullcontext()


class LockGuard:
    """Serialises warehouse access across the lanes of one GATHER run."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def __call__(self) -> AbstractContextManager:
        return self._lock


NO_GUARD: Guard = no_guard


__all__ = ["NO_GUARD", "Guard", "LockGuard", "no_guard"]
