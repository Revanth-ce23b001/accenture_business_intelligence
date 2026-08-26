"""Changes `CREATE TABLE IF NOT EXISTS` cannot make.

Most schema evolution in this project is free: the DDL is idempotent, so
a new table or a new column arrives by re-running `create_schema` over an
existing file. Two things are not free — changing a primary key and
changing a column's type — because DuckDB has no ALTER for either.

THE RULE THIS MODULE FOLLOWS. A migration that would destroy data
REFUSES. Every rebuild here first proves the table is empty and raises if
it is not, naming what it found. A prototype's convenience is not worth a
silent `DROP TABLE` in somebody's warehouse, and the day one of these
tables does hold rows, the refusal is the correct behaviour and the
message tells whoever hit it what to do.

Idempotent: running it twice is running it once. `create_schema` calls it
after the DDL, so `make seed`, the test fixtures and a developer opening
a stale warehouse all converge on the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb


#: How a CREATE TABLE ends in schema.sql. Named so the slice that lifts a
#: statement out reads as "up to and including the terminator" rather than
#: as an unexplained arithmetic offset.
STATEMENT_END = ");"


class MigrationError(RuntimeError):
    """A migration would have destroyed data, and did not run."""


@dataclass(frozen=True)
class Migration:
    """One change, and how to tell whether it has already happened."""

    name: str
    table: str
    #: The column whose presence means this migration has already run.
    sentinel_column: str
    reason: str


#: `case_outcome` was keyed on `case_id` alone, which allows one outcome
#: per case. P16 records one per HORIZON — D+14 for the cause, D+56 for
#: the money — so the key became (case_id, horizon_days).
#:
#: Safe to rebuild: the table has had no writer since it was created at
#: P4, so on any warehouse built before P16 it is empty. The rebuild
#: checks rather than assuming.
CASE_OUTCOME = Migration(
    name="case_outcome_per_horizon",
    table="case_outcome",
    sentinel_column="horizon_days",
    reason=(
        "case_outcome now records one row per outcome horizon (D+14 for the cause, "
        "D+56 for the money) and its primary key changed from (case_id) to "
        "(case_id, horizon_days). DuckDB cannot ALTER a primary key."
    ),
)

MIGRATIONS = (CASE_OUTCOME,)


def columns_of(connection: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = connection.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
        [table],
    ).fetchall()
    return {str(row[0]) for row in rows}


def table_exists(connection: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        connection.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()
    )


def row_count(connection: duckdb.DuckDBPyConnection, table: str) -> int:
    return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def pending(connection: duckdb.DuckDBPyConnection) -> tuple[Migration, ...]:
    """Migrations this warehouse still needs."""
    return tuple(
        migration
        for migration in MIGRATIONS
        if table_exists(connection, migration.table)
        and migration.sentinel_column not in columns_of(connection, migration.table)
    )


def migrate(connection: duckdb.DuckDBPyConnection, ddl: str) -> tuple[str, ...]:
    """Apply what is pending. Returns the names of the migrations that ran.

    `ddl` is the full `schema.sql` text: a rebuilt table is recreated from
    the same statement everything else is created from, so there is one
    definition of the table and not a second one hiding in this module.
    """
    applied: list[str] = []
    for migration in pending(connection):
        held = row_count(connection, migration.table)
        if held:
            raise MigrationError(
                f"{migration.name}: {migration.table} holds {held} rows and this "
                f"migration would drop them.\n  {migration.reason}\n"
                "Export the table, drop it by hand, and re-run — the rows are not "
                "discarded automatically."
            )
        connection.execute(f'DROP TABLE "{migration.table}"')
        connection.execute(_statement_for(ddl, migration.table))
        applied.append(migration.name)
    return tuple(applied)


def _statement_for(ddl: str, table: str) -> str:
    """The `CREATE TABLE` for `table`, lifted out of the DDL text.

    Read from `schema.sql` rather than duplicated here so a rebuilt table
    is byte-for-byte the table the DDL declares — including the comments
    above it, which is where the reason for the shape is written down.
    """
    marker = f"CREATE TABLE IF NOT EXISTS {table} ("
    start = ddl.find(marker)
    if start < 0:
        raise MigrationError(
            f"schema.sql declares no {marker!r}; the migration cannot rebuild a "
            "table the DDL does not define"
        )
    end = ddl.find(STATEMENT_END, start)
    if end < 0:
        raise MigrationError(f"the {table} statement in schema.sql is unterminated")
    return ddl[start : end + len(STATEMENT_END)]


__all__ = [
    "MIGRATIONS",
    "STATEMENT_END",
    "Migration",
    "MigrationError",
    "columns_of",
    "migrate",
    "pending",
    "row_count",
    "table_exists",
]
