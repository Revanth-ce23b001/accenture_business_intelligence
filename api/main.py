"""`python -m api.main` — the demo server.

`make demo` runs this with `MOCK_LLM=true`, which is CLAUDE.md rule 8: the
full demo runs offline from recorded fixtures, with the network disabled.

One worker, on purpose. The case store is process-local (see
`api/store.py`), so a second worker would serve 404s for cases the first
one ran. When the artefact tables have writers and a case can be read back
from the warehouse, this becomes a normal multi-worker deployment and
nothing else about the API changes.
"""

from __future__ import annotations

import os

import logging

from api.app import create_app

HOST = os.environ.get("CASEFILE_HOST", "127.0.0.1")
PORT = int(os.environ.get("CASEFILE_PORT", "8000"))

logger = logging.getLogger("casefile.api")


def _ensure_schema() -> None:
    """Bring the warehouse up to the current shape before serving.

    A developer's `data/casefile.duckdb` may have been built by an earlier
    revision and be missing a table a later step added — P14's
    `telemetry_request`, P16's `hypothesis_prior`. The DDL is idempotent
    and `engine/warehouse/migrate.py` refuses any migration that would
    destroy rows, so this is additive and safe.

    Done HERE and not in `create_app`, on purpose. This module is the demo
    server; the factory is a library, and a library that migrates whatever
    database it is handed is a library nobody can safely import.
    """
    from engine.db import get_connection
    from engine.warehouse.load import create_schema

    create_schema(get_connection())


_ensure_schema()
app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
