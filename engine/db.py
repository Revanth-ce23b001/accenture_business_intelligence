"""The only path to the warehouse.

CLAUDE.md rule 5: authorization happens BEFORE retrieval, and
`execute_governed` is the single door. `tests/test_governed_access.py`
asserts statically that no other module in the repository calls
`duckdb.connect`, so there is no second door to forget to lock.

What `execute_governed` does, in order:

  1. resolves the caller's persona against the KPI contract's
     `access_policy` (security/policy.py). An unknown persona, or a
     predicate whose binding the user cannot supply, is refused here —
     before anything is read.
  2. refuses any caller-supplied parameter whose name is reserved for the
     policy. Allowing one would let a caller overwrite its own row filter,
     which is the whole attack.
  3. wraps the caller's SQL so the row predicate is evaluated as an extra
     column, rather than appended to a WHERE clause it cannot see. The
     caller's query keeps its own semantics; the policy decides row by row.
  4. drops the persona's masked columns from the result.
  5. writes an `audit_log` row.

It returns `(rows, rows_filtered, columns_masked)`. `rows_filtered` is
what the policy removed, which is the number that makes a filtered result
honest: a regional manager sees 140 rows and is told 272 were withheld,
rather than seeing 140 rows and assuming that is the estate.

Rule 2 applies here too: this module contains no thresholds, no table
names and no SQL text of its own. The DDL is `warehouse/schema.sql`, the
loads are `warehouse/load.sql`, and every policy string comes from
`semantic_layer/warehouse.yaml`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import duckdb

from security.audit import AuditRecord
from security.policy import AppliedPolicy, PolicyError, User, resolve_policy
from semantic_layer.schema import SemanticLayer, get_semantic_layer

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The warehouse file. One file, no server (CLAUDE.md §Stack).
DEFAULT_DB_PATH = REPO_ROOT / "data" / "casefile.duckdb"

#: DuckDB's in-memory database. Used by the tests and by `make demo` when
#: the warehouse is rebuilt from scratch each run.
IN_MEMORY = ":memory:"

#: DuckDB binds named parameters as `$name`; the KPI contracts and the
#: access policies are written with `:name`, which is what a SQL reader
#: expects. Translate at the boundary rather than making the semantic
#: layer speak DuckDB.
_NAMED_PARAM = re.compile(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)")

#: Alias for the caller's query inside the governed wrapper.
_SOURCE_ALIAS = "governed_source"


class GovernanceError(PermissionError):
    """The query cannot be run as asked, and was not run."""


class GovernedResult(NamedTuple):
    """`(rows, rows_filtered, columns_masked)`.

    A NamedTuple so callers can unpack it positionally, as the P4 brief
    specifies, and still read the fields by name.
    """

    rows: tuple[dict[str, Any], ...]
    rows_filtered: int
    columns_masked: tuple[str, ...]


# ---------------------------------------------------------------------------
# Connections. The only `duckdb.connect` in the repository.
# ---------------------------------------------------------------------------

_CONNECTIONS: dict[tuple[str, bool], duckdb.DuckDBPyConnection] = {}


def connect(
    path: str | Path | None = None, *, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Open a NEW connection to the warehouse.

    Prefer `get_connection` unless you specifically want an independent
    handle: an audit row and the query it describes should land in the
    same transaction context.
    """
    target = _resolve(path)
    if target != IN_MEMORY:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(target, read_only=read_only)


def get_connection(
    path: str | Path | None = None, *, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """The process-wide connection for `path`, opened on first use."""
    key = (_resolve(path), read_only)
    connection = _CONNECTIONS.get(key)
    if connection is None:
        connection = connect(key[0], read_only=read_only)
        _CONNECTIONS[key] = connection
    return connection


def close_connections() -> None:
    """Close every memoised connection. Tests and `make reset` use this."""
    while _CONNECTIONS:
        _, connection = _CONNECTIONS.popitem()
        connection.close()


def _resolve(path: str | Path | None) -> str:
    if path is None:
        return str(DEFAULT_DB_PATH)
    if str(path) == IN_MEMORY:
        return IN_MEMORY
    return str(Path(path))


# ---------------------------------------------------------------------------
# The governed query
# ---------------------------------------------------------------------------


def execute_governed(
    user: User,
    kpi_id: str,
    sql: str,
    params: Mapping[str, Any] | None = None,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
    layer: SemanticLayer | None = None,
    purpose: str | None = None,
) -> GovernedResult:
    """Run `sql` for `user` under the access policy of `kpi_id`.

    Returns `(rows, rows_filtered, columns_masked)`.

    Raises `GovernanceError` before touching the warehouse when the KPI is
    unknown, the persona is not in its access policy, the persona's
    predicate needs a user attribute that is missing, or the caller tried
    to supply a reserved parameter.
    """
    layer = layer or get_semantic_layer()
    governance = layer.warehouse.governance

    kpi = layer.kpis.get(kpi_id)
    if kpi is None:
        raise GovernanceError(
            f"no KPI contract named {kpi_id!r}; known contracts: {sorted(layer.kpis)}"
        )

    try:
        policy = resolve_policy(kpi, user, governance)
    except PolicyError as exc:
        raise GovernanceError(str(exc)) from exc

    supplied = dict(params or {})
    trespassing = sorted(
        name for name in supplied if name.startswith(governance.reserved_binding_prefix)
    )
    if trespassing:
        raise GovernanceError(
            f"parameters {trespassing} are reserved for the access policy and may not be "
            "supplied by the caller — that is how a row filter gets bypassed"
        )

    statement = _wrap(sql, policy, governance.visibility_column)
    bindings = _bindings_used(statement, {**supplied, **policy.bindings})

    connection = connection if connection is not None else get_connection()

    try:
        cursor = connection.execute(statement, bindings) if bindings else connection.execute(
            statement
        )
        columns = [description[0] for description in cursor.description]
        fetched: Sequence[tuple] = cursor.fetchall()
    except Exception:
        _audit(connection, user, policy, sql, rows_returned=0, rows_filtered=0, purpose=purpose)
        raise

    rows, rows_filtered, columns_masked = _apply_masking(
        columns, fetched, policy, governance.visibility_column
    )
    _audit(
        connection,
        user,
        policy,
        sql,
        rows_returned=len(rows),
        rows_filtered=rows_filtered,
        purpose=purpose,
        columns_masked=columns_masked,
    )
    return GovernedResult(rows, rows_filtered, columns_masked)


def to_duckdb_params(sql: str) -> str:
    """Rewrite `:name` placeholders to DuckDB's `$name`, leaving `::` casts."""
    return _NAMED_PARAM.sub(r"$\1", sql)


def _wrap(sql: str, policy: AppliedPolicy, visibility_column: str) -> str:
    """Evaluate the row predicate as a column beside the caller's result.

    Appending `AND <predicate>` to the caller's SQL would be wrong twice
    over: it cannot be done safely for a query with its own GROUP BY or
    UNION, and it makes filtered rows indistinguishable from rows that
    were never there. Wrapping keeps the caller's semantics intact and
    keeps the count of withheld rows.
    """
    inner = to_duckdb_params(sql.strip().rstrip(";"))
    predicate = to_duckdb_params(policy.row_predicate)
    return (
        f'SELECT *, ({predicate}) AS "{visibility_column}" '
        f"FROM ({inner}) AS {_SOURCE_ALIAS}"
    )


def _bindings_used(statement: str, candidates: Mapping[str, Any]) -> dict[str, Any]:
    """Only the parameters the statement actually references.

    DuckDB rejects a binding it was not asked for, and the policy always
    offers every reserved binding it could resolve.
    """
    return {name: value for name, value in candidates.items() if f"${name}" in statement}


def _apply_masking(
    columns: Sequence[str],
    fetched: Sequence[tuple],
    policy: AppliedPolicy,
    visibility_column: str,
) -> tuple[tuple[dict[str, Any], ...], int, tuple[str, ...]]:
    """Split visible from withheld rows and drop the masked columns.

    A predicate that evaluates to NULL counts as NOT visible. That is the
    safe reading: a row whose region is unknown is not a row a regional
    manager has been granted.
    """
    visible_at = columns.index(visibility_column)
    masked = tuple(name for name in policy.masked_columns if name in columns)
    keep = [
        index
        for index, name in enumerate(columns)
        if index != visible_at and name not in masked
    ]

    rows: list[dict[str, Any]] = []
    rows_filtered = 0
    for record in fetched:
        if not record[visible_at]:
            rows_filtered += 1
            continue
        rows.append({columns[index]: record[index] for index in keep})
    return tuple(rows), rows_filtered, masked


def _audit(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    policy: AppliedPolicy,
    sql: str,
    *,
    rows_returned: int,
    rows_filtered: int,
    purpose: str | None,
    columns_masked: tuple[str, ...] = (),
) -> None:
    """Record the call. Never raises: an audit failure must not eat a result."""
    record = AuditRecord.for_query(
        user_id=user.user_id,
        persona=policy.persona,
        kpi=policy.kpi,
        sql=sql,
        row_predicate=policy.row_predicate,
        rows_returned=rows_returned,
        rows_filtered=rows_filtered,
        columns_masked=columns_masked,
        purpose=purpose,
    )
    try:
        connection.execute(AUDIT_INSERT, list(record.as_row()))
    except duckdb.Error:  # pragma: no cover - warehouse without the artefact tables
        pass


#: The one statement this module owns. Everything else it runs was handed
#: to it by a caller or read from a .sql file.
AUDIT_INSERT = (
    "INSERT INTO audit_log (audit_id, occurred_at, user_id, persona, kpi, "
    "statement_hash, row_predicate, rows_returned, rows_filtered, columns_masked, purpose) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


__all__ = [
    "AUDIT_INSERT",
    "DEFAULT_DB_PATH",
    "IN_MEMORY",
    "GovernanceError",
    "GovernedResult",
    "close_connections",
    "connect",
    "execute_governed",
    "get_connection",
    "to_duckdb_params",
]
