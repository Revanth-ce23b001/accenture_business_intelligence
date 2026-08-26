"""Turning a question into an investigation, or into a clarification.

`POST /api/ask` is the natural-language front door. It does exactly two
things: classify the INTENT, and resolve the SCOPE. It does not answer
anything — an answer comes from running the pipeline, and the pipeline is
`POST /api/cases/run`.

THE MODEL IS USED FOR THE ONE JOB CLAUDE.md GIVES IT HERE. "Intent
classification" is in the LLM column of the architecture table; the KPI,
the region and the period are not. Those are resolved deterministically
against the semantic layer and the warehouse's own vocabulary, because a
model that guesses "West" from a question about the west is a model that
will eventually guess "West" from a question about Western Railway.

AN UNRESOLVED QUESTION COMES BACK AS A QUESTION. If the KPI or the scope
cannot be resolved, the response carries a clarification and no run.
Guessing would produce a confident, well-evidenced, fully-audited answer
to a question nobody asked, which is the most expensive kind of wrong
this system can be.

OFFLINE, THE CLASSIFIER MAY HAVE NO FIXTURE. `MockProvider` raises for a
request it has never recorded. That is reported as an unclassified intent
with the deterministic resolution still attached — the scope resolution
does not depend on the model, so a missing fixture costs the intent label
and nothing else.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter

from api.deps import ConnectionDep, LayerDep, ProviderDep, UserDep
from api.periods import MONTH_PATTERN, PeriodError, resolve_window
from api.schemas import AskRequest, AskResponse, CaseRunRequest
from engine.db import warehouse_clock
from engine.qualify.series import load_series
from llm.provider import LLMMessage, LLMRequest
from security.policy import PolicyError, resolve_policy

router = APIRouter(tags=["ask"])

#: The intent vocabulary. A closed set: an intent the API cannot act on is
#: not an intent, and letting the model invent one would mean the router
#: dispatching on a string nobody has implemented.
INTENTS = (
    "diagnose_kpi_movement",
    "explain_case",
    "list_watchlist",
    "compare_scopes",
    "unclassified",
)

DEFAULT_INTENT = "diagnose_kpi_movement"
UNCLASSIFIED = "unclassified"

SYSTEM_PROMPT = (
    "You classify analyst questions into intents. Reply with one token."
)

#: Month names, for a question that says "last November" rather than
#: "2025-11". Resolved against the warehouse clock, never against today.
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

WORD = re.compile(r"[a-z0-9_]+")


@router.post("/api/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    user: UserDep,
    layer: LayerDep,
    connection: ConnectionDep,
    provider: ProviderDep,
) -> AskResponse:
    """Resolve a question into a run request, or ask for what is missing."""
    question = body.question.strip()
    tokens = set(WORD.findall(question.lower()))

    intent, model, from_fixture = _classify(question, provider)
    clock = warehouse_clock(connection, layer)

    kpi = _resolve_kpi(tokens, question.lower(), layer, user)
    scope = _resolve_scope(tokens, connection, user, layer, kpi)
    period, grain = _resolve_period(question.lower(), tokens, clock, layer, kpi)

    matched: dict[str, Any] = {"kpi": kpi, "scope": scope, "period": period, "grain": grain}

    missing = [name for name, value in (("KPI", kpi), ("scope", scope)) if value is None]
    if missing:
        return AskResponse(
            question=question,
            intent=intent,
            resolved=False,
            clarification=_clarification(missing, layer, user, connection, kpi),
            matched=matched,
            model=model,
            from_fixture=from_fixture,
        )

    try:
        resolve_window(period, grain)
    except PeriodError as exc:
        return AskResponse(
            question=question, intent=intent, resolved=False,
            clarification=f"{exc}. Which period did you mean?",
            matched=matched, model=model, from_fixture=from_fixture,
        )

    return AskResponse(
        question=question,
        intent=intent,
        resolved=True,
        run=CaseRunRequest(
            kpi=kpi,
            scope=scope,
            grain=grain,
            period=period,
            persona=body.persona or user.persona,
        ),
        matched=matched,
        model=model,
        from_fixture=from_fixture,
    )


# ---------------------------------------------------------------------------
# Intent — the model's one job here
# ---------------------------------------------------------------------------


def _classify(question: str, provider) -> tuple[str, str | None, bool]:
    """Ask the model for one token. A failure is `unclassified`, not a guess."""
    request = LLMRequest.for_task(
        "classify",
        system=SYSTEM_PROMPT,
        messages=(LLMMessage(role="user", content=question.lower()),),
        max_tokens=16,
    )
    try:
        response = provider.complete(request)
    except Exception:  # noqa: BLE001 - a missing fixture is not a failed request
        return UNCLASSIFIED, None, False

    token = response.text.strip().split()[0] if response.text.strip() else ""
    intent = token if token in INTENTS else UNCLASSIFIED
    return intent, response.model, response.from_fixture


# ---------------------------------------------------------------------------
# Scope — resolved, never guessed
# ---------------------------------------------------------------------------


def _resolve_kpi(tokens: set[str], lowered: str, layer, user) -> str | None:
    """Match the question against KPI names and display names.

    Only KPIs the caller's policy admits are candidates, so a question
    cannot resolve to a KPI whose first governed read would refuse it.
    """
    candidates = []
    for name, contract in layer.kpis.items():
        try:
            resolve_policy(contract, user, layer.warehouse.governance)
        except PolicyError:
            continue
        candidates.append((name, contract))

    for name, contract in candidates:
        if name in tokens or name.replace("_", " ") in lowered:
            return name
        if contract.display_name.lower() in lowered:
            return name
    # A bare "revenue" or "sales" resolves only when exactly one admitted
    # KPI's display name contains it. Two matches is an ambiguity, and an
    # ambiguity is a clarification rather than a coin toss.
    for word in ("revenue", "conversion", "availability", "price", "fulfilment"):
        if word in lowered:
            hits = [n for n, c in candidates if word in c.display_name.lower() or word in n]
            if len(hits) == 1:
                return hits[0]
    return None


def _resolve_scope(tokens: set[str], connection, user, layer, kpi: str | None) -> str | None:
    """Match against the regions the caller can actually see.

    Read from the warehouse through the governed door rather than from a
    list in code: the set of regions is data, and a persona scoped to one
    of them should not be able to resolve a question to another.
    """
    if kpi is None:
        return None
    try:
        series = load_series(connection, user, kpi, layer)
    except Exception:  # noqa: BLE001 - an unreadable series is not a crash here
        return None
    regions = series.regions
    for region in regions:
        if region.lower() in tokens:
            return region
    # A caller scoped to exactly one region did not need to name it.
    if len(regions) == 1:
        return regions[0]
    if user.region and user.region in regions:
        return user.region
    return None


def _resolve_period(lowered: str, tokens: set[str], clock, layer, kpi) -> tuple[str, str]:
    """A period label and a grain. Defaults to the warehouse's latest month.

    THE WAREHOUSE CLOCK, NOT TODAY. The extract ends on 30 Nov 2025;
    "last month" resolved against the wall clock would ask for a period
    the warehouse has never held.
    """
    grain = layer.kpis[kpi].grain.default if kpi in layer.kpis else "monthly"

    explicit = MONTH_PATTERN.search(lowered)
    if explicit:
        return explicit.group(0), "monthly"

    for name, number in MONTHS.items():
        if name in tokens:
            year = clock.year if number <= clock.month else clock.year - 1
            return f"{year:04d}-{number:02d}", "monthly"

    if "last month" in lowered or "previous month" in lowered:
        year, month = (
            (clock.year - 1, 12) if clock.month == 1 else (clock.year, clock.month - 1)
        )
        return f"{year:04d}-{month:02d}", "monthly"

    return f"{clock.year:04d}-{clock.month:02d}", grain if grain == "monthly" else "monthly"


def _clarification(missing, layer, user, connection, kpi) -> str:
    """Name what is missing and what the acceptable answers are."""
    parts = [f"I could not work out the {' or the '.join(missing)} from that."]
    if "KPI" in missing:
        readable = []
        for name, contract in sorted(layer.kpis.items()):
            try:
                resolve_policy(contract, user, layer.warehouse.governance)
            except PolicyError:
                continue
            readable.append(f"{contract.display_name} ({name})")
        parts.append("Which KPI did you mean? " + "; ".join(readable) + ".")
    if "scope" in missing:
        try:
            regions = load_series(connection, user, kpi, layer).regions if kpi else ()
        except Exception:  # noqa: BLE001
            regions = ()
        parts.append(
            "Which scope? " + ", ".join(regions) + "."
            if regions
            else "Which scope did you mean?"
        )
    return " ".join(parts)


__all__ = ["DEFAULT_INTENT", "INTENTS", "UNCLASSIFIED", "router"]
