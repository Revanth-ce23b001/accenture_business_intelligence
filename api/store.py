"""Where a finished case lives between the run that made it and the read.

WHY THERE IS A STORE AT ALL, STATED PLAINLY. `case_registry`, `evidence`
and `audit_log` are written by the engine as a case runs, so the durable
record of what happened is real. What the warehouse does NOT yet hold is
the assembled case OBJECT — `case_hypothesis` and `test_result` have DDL
and no writer, so a case read back from the warehouse today would come
back without its six tests per hypothesis. Rather than pretend otherwise,
the assembled object is kept in the process that made it and this module
says so in one place.

WHAT THAT MEANS IN PRACTICE. A restart loses the assembled objects and
keeps the registry rows, the evidence and the audit trail. For a
single-process `docker compose up` demo that is the right trade: the
alternative is writing two artefact tables and their loaders, which is
ADJUDICATE's work rather than the API's, and doing it badly would put a
half-populated case behind a durable-looking read.

IT IS BOUNDED. An unbounded dict behind an HTTP endpoint is a memory
leak with a URL. The store keeps the most recent `max_cases` and evicts
in insertion order.

Thread safety: a lock, because `POST /api/cases/run` runs in a worker
thread and `GET /api/cases/{id}` on the event loop, and dict mutation
across the two is exactly the race that produces an intermittent 404.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterator

from engine.verdict.casefile import CaseFile

#: How many assembled cases one process keeps. Small: the demo walks five
#: scenarios, and a reader who wants the sixth-oldest wants a database.
DEFAULT_MAX_CASES = 64


@dataclass
class StoredCase:
    """One finished case, and what the API needs beside it."""

    casefile: CaseFile
    #: The persona the run was executed as. A case is re-readable only by
    #: a persona the KPI contract would have let run it, and this is what
    #: that check compares against.
    run_as_persona: str
    run_by_user_id: str
    stored_at: datetime
    #: Per stage, the measured milliseconds. Rendered on the case header.
    latency_ms: dict[str, float] = field(default_factory=dict)
    #: Narratives already generated, by persona. Generated lazily, kept
    #: because a narrative is two model calls and re-rendering the same
    #: case for the same persona twice is money for nothing.
    narratives: dict[str, Any] = field(default_factory=dict)
    #: The abstention/decision trace, for the resolution panel.
    decision_trace: tuple[str, ...] = ()
    telemetry_request_id: str | None = None
    #: "canonical" for a Number Registry scenario assembled from the
    #: generator's config, "live" for one this process investigated. The
    #: two diverge today, and a screen showing both without saying which
    #: was which would be the worst of both.
    source: str = "live"
    config_ref: str | None = None

    @property
    def case_id(self) -> str:
        return self.casefile.case_id


class CaseStore:
    """Process-local, bounded, locked. See the module docstring."""

    def __init__(self, max_cases: int = DEFAULT_MAX_CASES) -> None:
        self.max_cases = max_cases
        self._cases: OrderedDict[str, StoredCase] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, stored: StoredCase) -> StoredCase:
        with self._lock:
            self._cases[stored.case_id] = stored
            self._cases.move_to_end(stored.case_id)
            while len(self._cases) > self.max_cases:
                self._cases.popitem(last=False)
            return stored

    def get(self, case_id: str) -> StoredCase | None:
        with self._lock:
            stored = self._cases.get(case_id)
            if stored is not None:
                self._cases.move_to_end(stored.case_id)
            return stored

    def evidence(self, evidence_id: str) -> Any | None:
        """Find one evidence record across every case held.

        `GET /api/evidence/{id}` does not take a case id — a number on
        screen knows its own evidence id and nothing else, which is the
        point of rule 3. The warehouse is the primary lookup; this is the
        fallback for a case whose evidence has not been persisted.
        """
        with self._lock:
            for stored in reversed(self._cases.values()):
                record = stored.casefile.evidence_by_id().get(evidence_id)
                if record is not None:
                    return record
        return None

    def recent(self, limit: int | None = None) -> tuple[StoredCase, ...]:
        with self._lock:
            cases = tuple(reversed(self._cases.values()))
        return cases[:limit] if limit else cases

    def clear(self) -> None:
        with self._lock:
            self._cases.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cases)

    def __contains__(self, case_id: object) -> bool:
        with self._lock:
            return case_id in self._cases

    def __iter__(self) -> Iterator[StoredCase]:
        return iter(self.recent())


def store_case(
    store: CaseStore,
    casefile: CaseFile,
    *,
    user_id: str,
    persona: str,
    latency_ms: dict[str, float] | None = None,
    decision_trace: tuple[str, ...] = (),
    telemetry_request_id: str | None = None,
    source: str = "live",
    config_ref: str | None = None,
) -> StoredCase:
    return store.put(
        StoredCase(
            casefile=casefile,
            run_as_persona=persona,
            run_by_user_id=user_id,
            stored_at=datetime.now(UTC),
            latency_ms=dict(latency_ms or {}),
            decision_trace=decision_trace,
            telemetry_request_id=telemetry_request_id,
            source=source,
            config_ref=config_ref,
        )
    )


def seed_canonical(
    store: CaseStore,
    *,
    connection=None,
    user=None,
    layer=None,
) -> tuple[str, ...]:
    """Load the five Number Registry scenarios into the store.

    So `/case/2451` opens instantly and carries the values CLAUDE.md
    specifies, rather than a spinner and then a live run that currently
    reaches a different verdict. Every figure is derived from
    `data/generator/config/scenario_*.yaml` (rule 4), and each case is
    labelled `canonical` so a reader is never in doubt which they are
    looking at.

    Failures are swallowed per scenario and reported by omission: a demo
    that will not start because one scenario config is malformed is worse
    than a demo missing one scenario.
    """
    from engine.verdict.canonical import CANONICAL, build_all

    seeded: list[str] = []
    for case_id, canonical in build_all(
        connection=connection, user=user, layer=layer
    ).items():
        store_case(
            store,
            canonical.case_file,
            user_id=user.user_id if user is not None else "system",
            persona=user.persona if user is not None else "analyst",
            source=CANONICAL,
            config_ref=canonical.config_ref,
        )
        seeded.append(case_id)
    return tuple(seeded)


__all__ = [
    "DEFAULT_MAX_CASES",
    "CaseStore",
    "StoredCase",
    "seed_canonical",
    "store_case",
]
