"""The trust boundary. What is allowed to become prompt text, and proof it did.

Row filtering and column masking already happened in
`engine/db.py::execute_governed`. This module exists because that is not
the same as knowing they held.

THE THREAT IS NOT A BROKEN MASK. It is a masked column arriving at the
model by a route that never went through the mask: a query written
against a second table, a dict assembled from two results, a caption
built by string formatting, a debug field somebody added last week.
`execute_governed` cannot see any of those, because by then the rows are
ordinary Python objects and Python does not know one dict from another.

So the check here is on the SERIALISED REQUEST — the exact bytes that
would go over the wire — and it is a search for the masked column names
in the text. That is a blunt instrument and it is the right one: it does
not care how the value got there, which is precisely the property the
mask itself lacks.

    payload = build_payload(result, policy)          # asserts, then packs
    guard_serialised(request.model_dump_json(), policy)   # asserts again
    release_to_llm(connection, user, policy, ...)    # records what left

Two locks, one on the structure and one on the bytes, and an audit row
saying how many rows crossed. A release is a separate access event from
the read that produced it, so it gets its own row: "did this leave the
building" is the question an auditor asks, and it should not require
interpreting the log to answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import duckdb

from security.audit import AuditRecord
from security.policy import AppliedPolicy, User

#: Purpose prefix for a release row, so the two kinds of access can be
#: told apart by a `LIKE` without joining anything.
RELEASE_PURPOSE_PREFIX = "llm.release"


class RedactionError(PermissionError):
    """Something the policy masks was about to reach a model."""


@dataclass(frozen=True)
class LlmPayload:
    """Rows cleared to cross the boundary, and what was held back."""

    rows: tuple[dict[str, Any], ...]
    columns: tuple[str, ...]
    withheld_columns: tuple[str, ...]
    kpi: str
    persona: str

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def as_json(self) -> str:
        """The payload as it would be embedded in a prompt.

        Sorted and separator-tight so the same rows always serialise the
        same way — a prompt that varies between runs cannot be replayed
        from a fixture (rule 8).
        """
        return json.dumps(
            [dict(sorted(row.items())) for row in self.rows],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

    def render(self) -> str:
        held = ", ".join(self.withheld_columns) or "nothing"
        return (
            f"{self.row_count} rows, {len(self.columns)} columns released to the "
            f"model for {self.persona} on {self.kpi}; withheld: {held}"
        )


def build_payload(
    rows: Sequence[Mapping[str, Any]],
    policy: AppliedPolicy,
    *,
    visibility_column: str | None = None,
) -> LlmPayload:
    """Pack governed rows for a prompt, refusing if anything masked is present.

    This is the structural lock. It re-checks what `execute_governed`
    already did, which is deliberate: the rows arrive here as plain dicts
    and this module has no way to know which function built them. A
    payload assembled by hand from two results, or widened by a caller
    who added a field, fails here rather than at the model.
    """
    masked = set(policy.masked_columns)
    present: set[str] = set()
    columns: list[str] = []
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
            if name in masked:
                present.add(name)

    if present:
        raise RedactionError(
            f"{sorted(present)} are masked for persona {policy.persona!r} on "
            f"{policy.kpi!r} and are present in the rows about to be sent to a "
            "model. The mask was applied at the warehouse; these rows did not "
            "come through it, or something put them back."
        )
    if visibility_column and visibility_column in columns:
        raise RedactionError(
            f"the policy's visibility column {visibility_column!r} is in the payload. "
            "It is an implementation detail of the row filter and tells a reader "
            "which rows they were not shown."
        )

    return LlmPayload(
        rows=tuple(dict(row) for row in rows),
        columns=tuple(columns),
        withheld_columns=tuple(policy.masked_columns),
        kpi=policy.kpi,
        persona=policy.persona,
    )


def guard_serialised(serialised: str, policy: AppliedPolicy) -> None:
    """The byte lock. Refuse if a masked column NAME appears in the request.

    Deliberately checks the name, not the value. A value can be a plausible
    number that appears legitimately elsewhere; the name is what makes a
    field readable, and a payload that carries `staff_cost` as a key has
    leaked the column whatever the number beside it says.

    Word-bounded, so `staff_cost_band` in a prompt's own prose does not
    trip it and `"staff_cost":` does.
    """
    offenders = [
        name
        for name in policy.masked_columns
        if re.search(rf"\b{re.escape(name)}\b", serialised)
    ]
    if offenders:
        raise RedactionError(
            f"the serialised request names {sorted(offenders)}, which the policy "
            f"masks for persona {policy.persona!r} on {policy.kpi!r}. Checked on the "
            "bytes that would go over the wire, so it does not matter which code "
            "path put them there."
        )


def release_to_llm(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    policy: AppliedPolicy,
    payload: LlmPayload,
    *,
    purpose: str,
    sql: str = "",
) -> AuditRecord:
    """Record that rows crossed the boundary. Returns the row written.

    Called once per prompt that carries warehouse data, whatever the
    model does with it afterwards. The count is the rows in the payload,
    not the rows the query returned: a caller that reads 140 rows and
    sends 34 has released 34, and the log should say so.
    """
    from engine.db import AUDIT_INSERT

    record = AuditRecord.for_query(
        user_id=user.user_id,
        persona=policy.persona,
        kpi=policy.kpi,
        sql=sql or payload.as_json(),
        row_predicate=policy.row_predicate,
        rows_returned=payload.row_count,
        rows_filtered=0,
        columns_masked=tuple(policy.masked_columns),
        rows_released_to_llm=payload.row_count,
        purpose=f"{RELEASE_PURPOSE_PREFIX}.{purpose}",
    )
    try:
        connection.execute(AUDIT_INSERT, list(record.as_row()))
    except duckdb.Error:  # pragma: no cover - warehouse without the artefact tables
        pass
    return record


__all__ = [
    "RELEASE_PURPOSE_PREFIX",
    "LlmPayload",
    "RedactionError",
    "build_payload",
    "guard_serialised",
    "release_to_llm",
]
