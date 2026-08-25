"""Who is asking, what they may see, and a record of every time they asked.

Three modules, one job each, and the split is the point.

    policy.py     resolves a user against a KPI contract's access policy
                  and produces the row predicate and the masked column
                  list. Holds no thresholds and touches no database.
    redaction.py  the trust boundary. Decides what may become prompt text
                  and proves it on the serialised bytes.
    audit.py      one row per access, reads and releases alike.

Authorization happens BEFORE retrieval (CLAUDE.md rule 5).
`engine/db.py::execute_governed` is the only path to the warehouse and it
resolves the policy before it builds a statement, so a persona who cannot
see a scope never gets a query that touches it.
"""

from security.audit import AuditRecord, new_audit_id, statement_hash
from security.policy import AppliedPolicy, PolicyError, User, bind_names, resolve_policy
from security.redaction import (
    LlmPayload,
    RedactionError,
    build_payload,
    guard_serialised,
    release_to_llm,
)

__all__ = [
    "AppliedPolicy",
    "AuditRecord",
    "LlmPayload",
    "PolicyError",
    "RedactionError",
    "User",
    "bind_names",
    "build_payload",
    "guard_serialised",
    "new_audit_id",
    "release_to_llm",
    "resolve_policy",
    "statement_hash",
]
