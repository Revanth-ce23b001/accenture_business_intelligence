"""What every request needs, resolved once.

Four dependencies, and the order they resolve in is the order CLAUDE.md
requires: the semantic layer and the warehouse are process-wide, the USER
is resolved per request from the signed header, and no governed read
happens until it has been (rule 5).

The connection is a single process-wide DuckDB handle. That is not a
pooling oversight: DuckDB is an embedded, single-file store, and the
audit row for a query has to land in the same transaction context as the
query it describes. `engine/db.py::get_connection` owns it — this module
asks for it and never opens one.
"""

from __future__ import annotations

from typing import Annotated

import duckdb
from fastapi import Depends, Header, HTTPException, Request, status

from api.auth import HEADER, AuthError, authenticate
from api.store import CaseStore
from security.policy import User
from semantic_layer.schema import SemanticLayer, get_semantic_layer

#: FastAPI turns the header name into this parameter alias.
PERSONA_HEADER = HEADER


def get_layer(request: Request) -> SemanticLayer:
    """The loaded, cross-checked semantic layer.

    Taken from app state when the app was given one — the tests build an
    app around a scratch warehouse — and from the process-wide cache
    otherwise.
    """
    layer = getattr(request.app.state, "layer", None)
    return layer if layer is not None else get_semantic_layer()


def get_connection(request: Request) -> duckdb.DuckDBPyConnection:
    connection = getattr(request.app.state, "connection", None)
    if connection is None:  # pragma: no cover - the factory always sets one
        from engine.db import get_connection as open_connection

        connection = open_connection()
        request.app.state.connection = connection
    return connection


def get_store(request: Request) -> CaseStore:
    return request.app.state.store


def get_provider(request: Request):
    """The LLM provider. `MockProvider` under MOCK_LLM (rule 8)."""
    provider = getattr(request.app.state, "provider", None)
    if provider is None:  # pragma: no cover - the factory always sets one
        from llm.provider import get_provider as select

        provider = select()
        request.app.state.provider = provider
    return provider


def get_user(
    connection: Annotated[duckdb.DuckDBPyConnection, Depends(get_connection)],
    x_casefile_persona: Annotated[str | None, Header(alias=PERSONA_HEADER)] = None,
) -> User:
    """The caller, verified and resolved. Every endpoint depends on this.

    A missing header is 401 and a bad one is 401. They are not
    distinguished in the body for the same reason `verify` does not
    distinguish its failures: an unauthenticated caller learns whether a
    token was accepted, and nothing more.
    """
    if not x_casefile_persona:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"{PERSONA_HEADER} is required.",
            headers={"WWW-Authenticate": PERSONA_HEADER},
        )
    try:
        return authenticate(connection, x_casefile_persona)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": PERSONA_HEADER},
        ) from exc


LayerDep = Annotated[SemanticLayer, Depends(get_layer)]
ConnectionDep = Annotated[duckdb.DuckDBPyConnection, Depends(get_connection)]
UserDep = Annotated[User, Depends(get_user)]
StoreDep = Annotated[CaseStore, Depends(get_store)]
ProviderDep = Annotated[object, Depends(get_provider)]


__all__ = [
    "PERSONA_HEADER",
    "ConnectionDep",
    "LayerDep",
    "ProviderDep",
    "StoreDep",
    "UserDep",
    "get_connection",
    "get_layer",
    "get_provider",
    "get_store",
    "get_user",
]
