"""Reading a result set by column name.

DuckDB hands back tuples. Unpacking them by position works right up until
somebody adds a column in the middle of a `SELECT`, at which point every
field after it is silently reading its neighbour's value — a bug that
produces plausible wrong answers rather than an exception.

`fetch_dicts` uses the cursor's own `description`, so the mapping comes
from the statement that ran rather than from a positional convention
maintained by hand in two places. `engine/db.py` does the same thing for
governed reads; this is the artefact-table equivalent, and it exists as
its own module so the learning package has one way of doing it.

A pleasant side effect: rule 2's scan for magic numbers in `engine/` no
longer has to distinguish a column index from a threshold, because there
are no column indices left to mistake for one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

import duckdb


def fetch_dicts(
    connection: duckdb.DuckDBPyConnection,
    sql: str,
    params: Sequence[Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Run `sql` and return rows keyed by column name."""
    cursor = connection.execute(sql, list(params)) if params else connection.execute(sql)
    columns = [description[0] for description in cursor.description]
    return tuple(
        dict(zip(columns, record, strict=True)) for record in cursor.fetchall()
    )


def text(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    return None if value is None else str(value)


def number(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value is None else float(value)


def integer(row: Mapping[str, Any], key: str) -> int | None:
    value = row.get(key)
    return None if value is None else int(value)


def flag(row: Mapping[str, Any], key: str) -> bool | None:
    """Tri-state on purpose. `None` is not `False`.

    `was_correct` and `cause_confirmed` both use NULL to mean "nobody has
    said", which is a different statement from "no" and calibrates
    differently.
    """
    value = row.get(key)
    return None if value is None else bool(value)


def moment(row: Mapping[str, Any], key: str) -> datetime | None:
    """A stored timestamp, read back as UTC.

    DuckDB's TIMESTAMP is naive and `engine/db.py::as_stored_timestamp`
    writes UTC into it. Reading it back without re-attaching the zone
    would produce a datetime that compares wrongly against an aware one.
    """
    value = row.get(key)
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


__all__ = ["fetch_dicts", "flag", "integer", "moment", "number", "text"]
