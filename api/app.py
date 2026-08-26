"""The application. One factory, one set of error shapes, one place to look.

    from api.app import create_app
    app = create_app()

WHY A FACTORY AND NOT A MODULE-LEVEL `app`. The tests build an app around
a scratch warehouse, and a module-level app would open the real one at
import time — which means importing the API to test it would touch the
developer's own data file. The factory takes the connection, the layer and
the provider, and defaults each to the process-wide one.

EVERY FAILURE HAS THE SAME SHAPE. `{"error": ..., "detail": ..., "hint":
...}`, whatever raised it. A governance refusal is a 403 with the
predicate that refused; a missing case is a 404 saying where cases live;
an unparseable period is a 422 saying what a period looks like. A client
that has to guess which of four error shapes it got is a client that
handles none of them.

WHAT IS NOT HERE. No CORS-allow-all, no debug endpoints, no silent
fallback that serves a case from somewhere else when the store misses.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.auth import HEADER, AuthError, SECRET_ENV, signing_secret_is_ephemeral
from api.periods import PeriodError
from api.routes import ask, cases, catalog, learning, platform
from api.store import CaseStore, seed_canonical
from engine.db import GovernanceError
from engine.learn.feedback import FeedbackError
from engine.learn.loop import LoopError
from engine.verdict.casefile import AssemblyError
from engine.verdict.pipeline import PipelineError
from security.policy import PolicyError
from security.redaction import RedactionError

TITLE = "CaseFile.ai"
VERSION = "0.1.0"
SUMMARY = "An AI business investigator. Dashboards describe; CaseFile investigates."

logger = logging.getLogger("casefile.api")


def create_app(
    *,
    connection: Any = None,
    layer: Any = None,
    provider: Any = None,
    store: CaseStore | None = None,
    overlay_path: Any = None,
    canonical: bool = True,
) -> FastAPI:
    """Build the app. Every dependency injectable, every default the real one."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if signing_secret_is_ephemeral():
            logger.warning(
                "%s is not set: tokens are signed with a per-process secret and stop "
                "working when this process restarts. That is correct for a prototype "
                "and wrong for a deployment.",
                SECRET_ENV,
            )
        yield
        # The connection is not closed here. It may be the process-wide
        # handle shared with a test or a script, and closing somebody
        # else's connection on shutdown is how a test suite acquires a
        # mysterious intermittent failure.

    app = FastAPI(
        title=TITLE,
        version=VERSION,
        summary=SUMMARY,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "catalogue", "description": "What exists, and what it is defined as."},
            {"name": "cases", "description": "Running an investigation and reading it back."},
            {"name": "ask", "description": "Natural language in, a run request or a question out."},
            {"name": "learning", "description": "Feedback in, and what it moved."},
            {"name": "platform", "description": "Calibration, telemetry, and the audit log."},
        ],
    )

    if connection is None:
        from engine.db import get_connection

        connection = get_connection()
    if layer is None:
        from semantic_layer.schema import get_semantic_layer

        layer = get_semantic_layer()
    if provider is None:
        from llm.provider import get_provider

        provider = get_provider()

    app.state.connection = connection
    app.state.layer = layer
    app.state.provider = provider
    app.state.store = store if store is not None else CaseStore()
    # Where driver feedback materialises the priors overlay. None means
    # the real `semantic_layer/runtime/priors.yaml`; the tests redirect it
    # so a suite run does not leave a generated file in the working tree.
    app.state.overlay_path = overlay_path

    # The five Number Registry scenarios, so a reader opening /case/2451
    # sees the case rather than an empty store. Each is labelled
    # `canonical`; a live run labels itself `live`.
    app.state.canonical_seeded = ()
    if canonical:
        from security.policy import User

        try:
            app.state.canonical_seeded = seed_canonical(
                app.state.store,
                connection=connection,
                user=User(user_id="system", persona="analyst"),
                layer=layer,
            )
        except Exception:  # noqa: BLE001 - a demo must still start
            logger.warning("could not seed the canonical scenarios", exc_info=True)

    app.include_router(catalog.router)
    app.include_router(cases.router)
    app.include_router(ask.router)
    app.include_router(learning.router)
    app.include_router(platform.router)

    _install_error_handlers(app)

    @app.get("/api/health", tags=["platform"])
    def health() -> dict[str, Any]:
        """Liveness, and what this process is configured as.

        Deliberately unauthenticated and deliberately says nothing about
        the business. `mock_llm` is here because "is this the offline
        demo" is the first question anyone debugging a narrative asks.
        """
        from llm.provider import mock_enabled

        return {
            "status": "ok",
            "version": VERSION,
            "mock_llm": mock_enabled(),
            "cases_held": len(app.state.store),
            "canonical_cases": list(app.state.canonical_seeded),
            "auth_header": HEADER,
            "signing_secret": "ephemeral" if signing_secret_is_ephemeral() else "configured",
        }

    return app


def _install_error_handlers(app: FastAPI) -> None:
    """One body shape for every failure, and the right status for each."""

    def body(error: str, detail: str, hint: str | None = None) -> dict[str, Any]:
        return {"error": error, "detail": detail, "hint": hint}

    @app.exception_handler(AuthError)
    async def _auth(request: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=body("unauthenticated", str(exc), f"Send a signed {HEADER} header."),
            headers={"WWW-Authenticate": HEADER},
        )

    @app.exception_handler(GovernanceError)
    async def _governance(request: Request, exc: GovernanceError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=body(
                "forbidden",
                str(exc),
                "The KPI's access_policy decides this, not the endpoint. See "
                "GET /api/semantic/{kpi_id} for the predicate applied to you.",
            ),
        )

    @app.exception_handler(PolicyError)
    async def _policy(request: Request, exc: PolicyError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=body("forbidden", str(exc)),
        )

    @app.exception_handler(RedactionError)
    async def _redaction(request: Request, exc: RedactionError) -> JSONResponse:
        # A masked column was about to reach a model. The request is
        # refused and the reason is not elaborated: the client does not
        # need to know which column, and telling it would be a hint.
        logger.error("redaction refused a release: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=body(
                "forbidden",
                "The response could not be produced without releasing masked data.",
            ),
        )

    @app.exception_handler(FeedbackError)
    async def _feedback(request: Request, exc: FeedbackError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=body(
                "invalid_feedback",
                str(exc),
                "The vocabulary is closed and comes from "
                "semantic_layer/learning.yaml. GET /api/feedback/options serves it.",
            ),
        )

    @app.exception_handler(LoopError)
    async def _loop(request: Request, exc: LoopError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=body("feedback_not_applied", str(exc)),
        )

    @app.exception_handler(PeriodError)
    async def _period(request: Request, exc: PeriodError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=body("invalid_period", str(exc)),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=body("invalid_request", str(exc.errors())),
        )

    @app.exception_handler(PipelineError)
    async def _pipeline(request: Request, exc: PipelineError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=body("pipeline_refused", str(exc)),
        )

    @app.exception_handler(AssemblyError)
    async def _assembly(request: Request, exc: AssemblyError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body(
                "case_incomplete",
                str(exc),
                "A stage did not produce something the case file requires. The run "
                "is not served partially — a half-assembled case would be read as a "
                "whole one.",
            ),
        )


__all__ = ["SUMMARY", "TITLE", "VERSION", "create_app"]
