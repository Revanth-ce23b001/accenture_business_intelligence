"""What the system knows about: KPIs, the watchlist, contracts, evidence.

The read-only half of the API. Nothing here runs an investigation; these
are the endpoints a reader uses to decide which investigation to look at,
and to chase a number back to what produced it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status

from api.deps import ConnectionDep, LayerDep, StoreDep, UserDep
from api.periods import PeriodError, resolve_window
from api.schemas import (
    KpiCard,
    KpiSummary,
    SeriesPoint,
    SemanticContract,
    Watchlist,
    WatchlistItem,
    Whoami,
)
from engine.contracts import Evidence
from engine.db import GovernanceError, execute_metadata, warehouse_clock
from engine.evidence import from_row, load as load_evidence
from engine.qualify.gate import QualifyRequest, qualify
from engine.qualify.restraint import OPEN_STATUS
from engine.qualify.series import load_series
from engine.validate.checks import ValidationRequest
from engine.validate.gate import validate
from security.policy import PolicyError, resolve_policy

router = APIRouter(tags=["catalogue"])

#: `case_registry` is on the metadata allow-list, so the watchlist reads
#: it through `execute_metadata` rather than the governed door — it holds
#: no measure, and a row predicate has nothing to filter on it.
CASE_REGISTRY = "case_registry"

OPEN_CASES_SQL = f"""
SELECT case_id, kpi, scope, grain, period, opened_at, status, verdict,
       confidence_calibrated, materiality_multiple, elapsed_ms, reason_text
FROM {CASE_REGISTRY}
WHERE status = :status
ORDER BY materiality_multiple DESC NULLS LAST, opened_at DESC
"""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@router.get("/api/whoami", response_model=Whoami)
def whoami(user: UserDep, layer: LayerDep) -> Whoami:
    """Who the signed header resolved to, and what they may open.

    The KPI list is resolved through the same `resolve_policy` every read
    uses, so it cannot say yes to something a later read says no to.
    """
    readable = []
    for name, contract in sorted(layer.kpis.items()):
        try:
            resolve_policy(contract, user, layer.warehouse.governance)
        except PolicyError:
            continue
        readable.append(name)
    return Whoami(
        user_id=user.user_id,
        persona=user.persona,
        display_name=user.display_name,
        region=user.region,
        store_id=user.store_id,
        kpis=tuple(readable),
    )


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------


@router.get("/api/kpis", response_model=list[KpiSummary])
def list_kpis(user: UserDep, layer: LayerDep) -> list[KpiSummary]:
    """Every KPI contract this persona's access policy admits.

    Filtered by policy rather than listed in full and enforced later. A
    catalogue that advertises a KPI the caller cannot read is a catalogue
    that teaches its user to expect 403s.
    """
    summaries: list[KpiSummary] = []
    for name, contract in sorted(layer.kpis.items()):
        try:
            resolve_policy(contract, user, layer.warehouse.governance)
        except PolicyError:
            continue
        materiality = contract.thresholds.materiality
        summaries.append(
            KpiSummary(
                kpi=name,
                display_name=contract.display_name,
                unit=contract.unit,
                grain=contract.grain.default,
                owner_role=contract.owner_role,
                definition=contract.definition.strip(),
                materiality_value=materiality.value if materiality else None,
                materiality_unit=materiality.unit if materiality else None,
                materiality_display=materiality.display if materiality else None,
                can_open_case=contract.can_open_a_case(),
                drivers=tuple(contract.drivers),
                related_kpis=tuple(contract.related_kpis),
                source_systems=tuple(s.name for s in contract.source_systems),
                refresh_sla_hours=contract.refresh.sla_hours,
            )
        )
    return summaries


@router.get("/api/semantic/{kpi_id}", response_model=SemanticContract)
def semantic_contract(
    kpi_id: str, user: UserDep, layer: LayerDep
) -> SemanticContract:
    """One KPI's whole contract — the definition a number was computed under.

    The last link in the chain rule 3 asks for. A displayed figure links
    to its `Evidence`; the evidence names a `source_ref` into this file;
    this endpoint returns the file. The caller's own row predicate and
    masked columns are resolved and returned alongside, so a reader can
    see not only the definition but which slice of it they were served.
    """
    contract = layer.kpis.get(kpi_id)
    if contract is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no KPI contract named {kpi_id!r}; known: {sorted(layer.kpis)}",
        )
    try:
        policy = resolve_policy(contract, user, layer.warehouse.governance)
    except PolicyError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    return SemanticContract(
        kpi=kpi_id,
        version=contract.version,
        contract=contract.model_dump(mode="json"),
        source_file=f"semantic_layer/kpis/{kpi_id}.yaml",
        row_predicate=policy.row_predicate,
        masked_columns=tuple(policy.masked_columns),
    )


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------


@router.get("/api/watchlist", response_model=Watchlist)
def watchlist(
    request: Request,
    user: UserDep,
    layer: LayerDep,
    connection: ConnectionDep,
    kpi: str = Query(default="net_revenue"),
    period: str | None = Query(default=None),
    grain: str = Query(default="monthly"),
    scan: bool = Query(
        default=True,
        description="Check the visible scopes for movements that were suppressed.",
    ),
) -> Watchlist:
    """Open cases, and the movements that were stopped before becoming one.

    THE SUPPRESSED HALF IS THE POINT. Any system can list what it opened.
    Listing what it looked at and DECLINED to open, naming the gate and
    the reason, is what separates a queue from a record of judgement — and
    it is the only way a reader can tell "nothing is wrong this month"
    from "nothing was checked this month".

    The scan is real work: VALIDATE and QUALIFY run per visible scope, so
    each suppressed row carries the gate that actually stopped it rather
    than a status somebody wrote down. It costs a few seconds per scope,
    which is why `scan=false` exists for a caller that only wants the
    open cases. Nothing below QUALIFY runs — a suppressed movement is not
    investigated, which is the whole reason it was suppressed.
    """
    contract = layer.kpis.get(kpi)
    if contract is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no KPI contract named {kpi!r}"
        )
    try:
        resolve_policy(contract, user, layer.warehouse.governance)
    except PolicyError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    clock = warehouse_clock(connection, layer)
    resolved_period = period or _default_period(clock, grain)
    try:
        resolve_window(resolved_period, grain)
    except PeriodError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    suppressed = _canonical_suppressed(request, layer)
    if scan:
        suppressed += _scan(connection, user, layer, kpi, resolved_period, grain, clock)

    return Watchlist(
        cards=_cards(connection, user, layer, suppressed),
        open_cases=_open_cases(connection, user, layer),
        suppressed=suppressed,
        generated_at=datetime.now(UTC),
        period=resolved_period,
        scanned=scan,
    )


def _cards(connection, user, layer, suppressed) -> tuple[KpiCard, ...]:
    """One card per KPI the caller may read, with its sparkline.

    The series is governed like any other read, so a regional manager's
    line is their region's. Two of the six KPIs have no series and the
    card says why rather than drawing nothing.
    """
    from engine.series import load_series as load_kpi_series

    stopped = {item.kpi: item for item in suppressed}
    cards: list[KpiCard] = []
    for name, contract in sorted(layer.kpis.items()):
        try:
            resolve_policy(contract, user, layer.warehouse.governance)
        except PolicyError:
            continue

        series = load_kpi_series(connection, user, name, layer=layer)
        materiality = contract.thresholds.materiality
        status, detail = _status_for(name, contract, stopped.get(name))
        cards.append(
            KpiCard(
                kpi=name,
                display_name=contract.display_name,
                unit=contract.unit,
                grain=contract.grain.default,
                owner_role=contract.owner_role,
                materiality_display=materiality.display if materiality else None,
                can_open_case=contract.can_open_a_case(),
                series=tuple(
                    SeriesPoint(period=point.period, value=point.value)
                    for point in series.points
                ),
                series_available=series.available,
                series_reason=series.reason,
                series_unit=series.unit,
                source_table=series.source_table,
                latest=series.latest,
                previous=series.previous,
                change_pct=series.change_pct,
                rows_filtered=series.rows_filtered,
                status=status,
                status_detail=detail,
            )
        )
    return tuple(cards)


def _status_for(name, contract, stopped) -> tuple[str, str | None]:
    """What the watchlist has to say about this KPI right now."""
    if not contract.can_open_a_case():
        return "monitoring_only", (
            f"{contract.history_weeks} weeks of history against the "
            f"{contract.baseline.min_history_weeks} a baseline needs — monitoring only."
        )
    if stopped is not None:
        return "suppressed", stopped.detail
    return "quiet", None


def _canonical_suppressed(request, layer) -> tuple[WatchlistItem, ...]:
    """Movements the Number Registry scenarios record as stopped.

    #2470 is a Gate 1 data incident and #2471 a Gate 3 history kill. Both
    are cases the system looked at and declined to open, which is exactly
    what this panel is for — and both carry the gate, the outcome code and
    the reason from their own scenario config rather than from a literal.
    """
    store = getattr(request.app.state, "store", None)
    if store is None:
        return ()

    rows: list[WatchlistItem] = []
    for stored in store.recent():
        if stored.source != "canonical":
            continue
        adjudication = stored.casefile.adjudication
        failed = [gate for gate in adjudication.gates if not gate.passed]
        if not failed:
            continue
        gate = failed[0]
        rows.append(
            WatchlistItem(
                state="suppressed",
                kpi=adjudication.kpi,
                scope=adjudication.scope,
                grain=adjudication.grain,
                period=adjudication.period,
                case_id=adjudication.case_id,
                headline=_headline_of(adjudication),
                stopped_by_gate=gate.gate_id,
                stopped_by_name=gate.name,
                outcome_code=gate.outcome_code,
                detail=gate.detail,
            )
        )
    return tuple(rows)


def _headline_of(adjudication) -> str | None:
    movement = adjudication.headline_movement
    if movement is None or movement.value is None:
        return None
    unit = (movement.unit or "").lower()
    suffix = {"pct": "%", "pt": " pt", "inr_cr": " Cr"}.get(unit, "")
    return f"{float(movement.value):+.1f}{suffix}"


def _open_cases(connection, user, layer) -> tuple[WatchlistItem, ...]:
    """The register, ranked by how many times its own limit each case is."""
    rows = execute_metadata(
        user,
        CASE_REGISTRY,
        OPEN_CASES_SQL,
        {"status": OPEN_STATUS},
        connection=connection,
        layer=layer,
        purpose="api.watchlist.open",
    )
    return tuple(
        WatchlistItem(
            state="open",
            kpi=str(row["kpi"]),
            scope=str(row["scope"]),
            grain=str(row["grain"]),
            period=str(row["period"]),
            case_id=str(row["case_id"]),
            status=str(row["status"]),
            verdict=_optional(row["verdict"]),
            confidence_calibrated=_number(row["confidence_calibrated"]),
            materiality_multiple=_number(row["materiality_multiple"]),
            opened_at=row["opened_at"],
            elapsed_ms=_number(row["elapsed_ms"]),
            detail=_optional(row["reason_text"]),
        )
        for row in rows
    )


def _scan(
    connection, user, layer, kpi: str, period: str, grain: str, clock
) -> tuple[WatchlistItem, ...]:
    """VALIDATE and QUALIFY every visible scope. Report what stopped each."""
    try:
        series = load_series(connection, user, kpi, layer)
    except GovernanceError:
        return ()

    suppressed: list[WatchlistItem] = []
    for scope in series.regions:
        item = _check(connection, user, layer, kpi, scope, period, grain, clock)
        if item is not None:
            suppressed.append(item)
    return tuple(suppressed)


def _check(
    connection, user, layer, kpi, scope, period, grain, clock
) -> WatchlistItem | None:
    """One scope. `None` when the movement qualifies — that is a case, not
    a suppression, and it belongs in the other half of the list."""
    base = {"kpi": kpi, "scope": scope, "grain": grain, "period": period}

    try:
        checked = validate(
            connection,
            user,
            ValidationRequest(kpi=kpi, scope=scope, grain=grain, period=period),
            layer=layer,
            clock=clock,
        )
    except (GovernanceError, Exception) as exc:  # noqa: BLE001
        return WatchlistItem(
            state="suppressed", **base,
            outcome_code="SCAN_FAILED",
            detail=f"{type(exc).__name__}: {exc}",
        )
    if not checked.case_opened:
        failure = checked.failures[0] if checked.failures else None
        return WatchlistItem(
            state="suppressed", **base,
            stopped_by_gate=checked.gate.gate_id,
            stopped_by_name=checked.gate.name,
            outcome_code=checked.outcome_code,
            detail=failure.detail if failure else checked.gate.detail,
        )

    try:
        qualified = qualify(
            connection,
            user,
            QualifyRequest(kpi=kpi, scope=scope, grain=grain, period=period),
            layer=layer,
            clock=clock,
        )
    except Exception as exc:  # noqa: BLE001
        return WatchlistItem(
            state="suppressed", **base,
            outcome_code="SCAN_FAILED",
            detail=f"{type(exc).__name__}: {exc}",
        )
    if qualified.case_opened:
        return None

    stopping = next((gate for gate in qualified.gates if not gate.passed), None)
    if stopping is not None:
        return WatchlistItem(
            state="suppressed", **base,
            stopped_by_gate=stopping.gate_id,
            stopped_by_name=stopping.name,
            outcome_code=stopping.outcome_code,
            detail=stopping.detail,
        )

    # Every gate passed and the case still did not open: restraint. Not a
    # gate — it asks whether one more case helps the person who has to
    # read it — so it is reported without a gate number.
    restraint = qualified.restraint
    return WatchlistItem(
        state="suppressed", **base,
        outcome_code=restraint.outcome if restraint else "SUPPRESSED",
        detail=restraint.detail if restraint else "suppressed after every gate passed",
    )


def _default_period(clock: datetime, grain: str) -> str:
    """The latest period the warehouse holds, at this grain.

    The warehouse clock, not today. The extract ends on 30 Nov 2025 and a
    watchlist for the current wall-clock month would be empty for a reason
    that says nothing about the business.
    """
    if grain == "daily":
        return clock.date().isoformat()
    if grain == "weekly":
        iso = clock.date().isocalendar()
        return f"{iso.year:04d}-W{iso.week:02d}"
    return f"{clock.year:04d}-{clock.month:02d}"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@router.get("/api/evidence/{evidence_id}", response_model=Evidence)
def evidence(
    evidence_id: str, user: UserDep, connection: ConnectionDep, store: StoreDep
) -> Evidence:
    """One evidence record. What every number on screen links to (rule 3).

    The warehouse first, because that is the durable copy, then the
    in-process store for a case whose evidence has not been persisted.
    Both return the same contract, so a caller cannot tell — and should
    not need to — which one answered.
    """
    rows = load_evidence(connection)
    for row in rows:
        if str(row["evidence_id"]) == evidence_id:
            return from_row(row)

    record = store.evidence(evidence_id)
    if record is not None:
        return record

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=(
            f"no evidence {evidence_id!r}. Evidence ids are minted by the stage that "
            "publishes the number; an id that is not here was never published."
        ),
    )


def _optional(value: Any) -> str | None:
    return None if value is None else str(value)


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = ["router"]
