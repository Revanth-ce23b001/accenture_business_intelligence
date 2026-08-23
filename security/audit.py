"""The record of every query that reached the warehouse.

`engine/db.py::execute_governed` is the only path to the warehouse
(CLAUDE.md rule 5), and it writes one of these on every call — including
the calls that returned nothing and the calls that raised. So the audit
log is not a sample of warehouse activity; it is all of it.

The statement itself is not stored. A hash is, which is enough to prove
two calls ran the same query and to link a displayed number back to the
statement that produced it, without copying customer-scoped SQL into a
table with different access rules from the one it read.
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
    purpose: str | None = None

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
            purpose=purpose,
        )

    def as_row(self) -> tuple:
        """Positional values for the `audit_log` insert."""
        return (
            self.audit_id,
            self.occurred_at,
            self.user_id,
            self.persona,
            self.kpi,
            self.statement_hash,
            self.row_predicate,
            self.rows_returned,
            self.rows_filtered,
            COLUMN_SEPARATOR.join(self.columns_masked),
            self.purpose,
        )


__all__ = ["COLUMN_SEPARATOR", "AuditRecord", "new_audit_id", "statement_hash"]
