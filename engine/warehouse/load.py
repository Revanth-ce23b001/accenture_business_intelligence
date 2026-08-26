"""Build the DuckDB warehouse from `data/raw`.

Three SQL files do the work and this module only sequences them:

  schema.sql   DDL for the source tables and the case artefact tables
  load.sql     one INSERT per source CSV, columns named and dates cast
  views.sql    the conformance views, where the four reconciliation
               problems become queryable

Nothing here repairs anything. A row whose store_code does not resolve is
loaded; a revenue column the KPI contract excludes is loaded; weekly
marketing spend stays weekly. `engine/warehouse/reconcile.py` reports what
that left behind, and writes it to `data_gap_register`.

The connection comes from `engine/db.py`, which owns the only
`duckdb.connect` call in the repository (CLAUDE.md rule 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from engine.db import REPO_ROOT, connect

SQL_DIR = Path(__file__).parent
RAW_DIR = REPO_ROOT / "data" / "raw"

SCHEMA_SQL = SQL_DIR / "schema.sql"
LOAD_SQL = SQL_DIR / "load.sql"
VIEWS_SQL = SQL_DIR / "views.sql"

#: Substituted into load.sql. Only one, and it is a directory.
RAW_PLACEHOLDER = "{raw}"

#: Written by the generator last. Its presence means data/raw is complete.
MANIFEST = "manifest.json"


class WarehouseError(RuntimeError):
    """The warehouse could not be built from the raw files as they stand."""


@dataclass(frozen=True)
class LoadReport:
    """What the load produced. Row counts are read back, never assumed."""

    database: str
    tables: dict[str, int]
    views: tuple[str, ...]

    @property
    def total_rows(self) -> int:
        return sum(self.tables.values())

    def empty_tables(self) -> tuple[str, ...]:
        """Tables with no rows. Expected for the case artefacts at P4."""
        return tuple(sorted(name for name, count in self.tables.items() if not count))


def create_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Create every table, then apply what the DDL cannot express.

    Idempotent twice over: the DDL is CREATE TABLE IF NOT EXISTS, and
    `engine/warehouse/migrate.py` skips a migration whose sentinel column
    is already there. Running this over a warehouse built by any earlier
    revision converges it on the current shape, which is why the test
    fixtures call it on the copy they were handed.

    A migration that would destroy rows raises rather than running. See
    the note at the top of `migrate.py`.
    """
    from engine.warehouse.migrate import migrate

    ddl = _read(SCHEMA_SQL)
    connection.execute(ddl)
    migrate(connection, ddl)


def create_views(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(_read(VIEWS_SQL))


def load_sources(
    connection: duckdb.DuckDBPyConnection, raw_dir: Path | str = RAW_DIR
) -> None:
    """Run the INSERTs. Assumes the tables are empty."""
    raw = Path(raw_dir)
    if not raw.is_dir():
        raise WarehouseError(f"raw directory not found: {raw}. Run `make generate` first.")
    # data/raw is gitignored, so on a fresh clone the directory exists (it
    # holds .gitkeep) and is otherwise empty. Say so, rather than letting
    # DuckDB report a missing CSV three statements in.
    if not (raw / MANIFEST).is_file():
        raise WarehouseError(
            f"{raw} holds no generated data ({MANIFEST} is missing). "
            "Run `make generate` to build it — data/raw is not committed."
        )
    statement = _read(LOAD_SQL).replace(RAW_PLACEHOLDER, raw.as_posix())
    try:
        connection.execute(statement)
    except duckdb.Error as exc:
        raise WarehouseError(f"loading {raw} failed: {exc}") from exc


def table_names(connection: duckdb.DuckDBPyConnection) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' ORDER BY table_name"
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def view_names(connection: duckdb.DuckDBPyConnection) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_type = 'VIEW' ORDER BY table_name"
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def row_counts(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """One COUNT(*) per table, read back off the warehouse."""
    return {
        name: int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
        for name in table_names(connection)
    }


def build_warehouse(
    database: str | Path | None = None,
    raw_dir: Path | str = RAW_DIR,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> LoadReport:
    """Create the schema, load the sources, create the views, report back.

    Pass `connection` to build into a handle you already hold — the tests
    build into an in-memory database that way.
    """
    owned = connection is None
    connection = connection if connection is not None else connect(database)
    try:
        create_schema(connection)
        load_sources(connection, raw_dir)
        create_views(connection)
        return LoadReport(
            database=str(database) if database is not None else ":in-place:",
            tables=row_counts(connection),
            views=view_names(connection),
        )
    finally:
        if owned:
            connection.close()


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - packaging error
        raise WarehouseError(f"missing SQL file: {path}") from exc


def main() -> None:  # pragma: no cover - entry point
    """`make seed`. Rebuilds the warehouse and prints what it loaded."""
    from engine.db import DEFAULT_DB_PATH

    if DEFAULT_DB_PATH.exists():
        DEFAULT_DB_PATH.unlink()
    report = build_warehouse(DEFAULT_DB_PATH)

    width = max(len(name) for name in report.tables)
    print(DEFAULT_DB_PATH)
    for name, count in sorted(report.tables.items()):
        print(f"  {name:<{width}}  {count:>9,}")
    print(f"  {'TOTAL':<{width}}  {report.total_rows:>9,}")
    print("views: " + ", ".join(report.views))
    empty = report.empty_tables()
    if empty:
        print("empty (no case has produced rows yet): " + ", ".join(empty))


if __name__ == "__main__":  # pragma: no cover
    main()
