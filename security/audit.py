"""The record of every query that reached the warehouse.

`engine/db.py::execute_governed` is the only path to the warehouse
(CLAUDE.md rule 5), and it writes one of these on every call — including
the calls that returned nothing and the calls that raised. So the audit
log is not a sample of warehouse activity; it is all of it.

The statement itself is not stored. A hash is, which is enough to prove
two calls ran the same query and to link a displayed number back to the
statement that produced it, without copying customer-scoped SQL into a
table with different access rules from the one it read.

TWO KINDS OF ACCESS, and the log distinguishes them by construction.

A READ is `execute_governed` returning rows to Python. It records what
the policy did — the predicate, the rows withheld, the columns dropped —
and `rows_released_to_llm` is 0, because at that moment nothing has left
the process.

A RELEASE is those rows being serialised into a model prompt, which is a
different act against a different party and gets its own row with its own
count. Folding the two together would mean reading the log to work out
whether a number ever left the building, and "did this leave" is the
question an auditor actually asks.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

#: Separator for the masked-column list. A single column name cannot
#: contain it, so the stored string round-trips.
COLUMN_SEPARATOR = ","


def statement_hash(sql: str) -> str:
    """Stable fingerprint of a statement, whitespace-insensitive."""
    normalised = " ".join(sql.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def new_audit_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class AuditRecord:
    """One row of `audit_log`."""

    audit_id: str
    occurred_at: datetime
    user_id: str
    persona: str
    kpi: str
    statement_hash: str
    row_predicate: str
    rows_returned: int
    rows_filtered: int
    columns_masked: tuple[str, ...]
    #: Rows that crossed the trust boundary into a model prompt. Zero on a
    #: read; set only by `security/redaction.py`, which is the one place
    #: warehouse rows are allowed to become prompt text.
    rows_released_to_llm: int = 0
    purpose: str | None = None

    @property
    def released(self) -> bool:
        return self.rows_released_to_llm > 0

    @classmethod
    def for_query(
        cls,
        *,
        user_id: str,
        persona: str,
        kpi: str,
        sql: str,
        row_predicate: str,
        rows_returned: int,
        rows_filtered: int,
        columns_masked: tuple[str, ...],
        rows_released_to_llm: int = 0,
        purpose: str | None = None,
    ) -> AuditRecord:
        return cls(
            audit_id=new_audit_id(),
            occurred_at=datetime.now(UTC),
            user_id=user_id,
            persona=persona,
            kpi=kpi,
            statement_hash=statement_hash(sql),
            row_predicate=row_predicate,
            rows_returned=rows_returned,
            rows_filtered=rows_filtered,
            columns_masked=columns_masked,
            rows_released_to_llm=rows_released_to_llm,
            purpose=purpose,
        )

    def as_row(self) -> tuple:
        """Positional values for the `audit_log` insert.

        The timestamp is normalised by the caller's storage helper rather
        than here: how a datetime lands in a column is the warehouse's
        business, and this module does not know what the column is.
        """
        from engine.db import as_stored_timestamp

        return (
            self.audit_id,
            as_stored_timestamp(self.occurred_at),
            self.user_id,
            self.persona,
            self.kpi,
            self.statement_hash,
            self.row_predicate,
            self.rows_returned,
            self.rows_filtered,
            COLUMN_SEPARATOR.join(self.columns_masked),
            self.rows_released_to_llm,
            self.purpose,
        )


__all__ = ["COLUMN_SEPARATOR", "AuditRecord", "new_audit_id", "statement_hash"]
