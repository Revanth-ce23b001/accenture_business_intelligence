"""The measurement. One recorder per request, one row per request.

    recorder = Recorder(user_id="u.reg.west", persona="regional_manager",
                        question="why did west revenue fall last month")

    with recorder.stage("VALIDATE"):
        ...
    with recorder.stage("QUALIFY") as t:
        t.record_method("stl_decomposition")
        recorder.mark_case_opened("2451")
    ...
    row = recorder.close(verdict="PARTIALLY_EXPLAINED", confidence=0.84)
    persist(connection, row, events=recorder.events())

WHY A CONTEXT MANAGER AND NOT A DECORATOR. A stage that raises still took
time, and the row for a request that died in GATHER is the most
interesting row in the table. `finally` records the elapsed time and
re-raises; there is no path through `stage()` that loses a measurement.

THE ROW IS ALWAYS COMPLETE. All five stage latencies are present on every
request. A request killed at Gate 1 records 0.0 ms for the four stages it
never reached and names the ones it did in `stages_entered`, so a zero is
never ambiguous between "instant" and "never happened". The NOT NULL
constraints in `telemetry_request` mean the database enforces this too,
rather than trusting every future caller to remember.

WARM IS MEASURED, NOT CLAIMED. The first request in a process pays for
loading the semantic layer, opening DuckDB and reading the fixture store.
`targets.warmup_requests` says how many requests that covers; each row
records which side of it the request fell on, and the P95 is computed
over the warm ones. A latency claim whose "warm" is a footnote rather
than a column is a claim nobody can check.

RESOLVED DEFECT 6 LIVES HERE. `case_registry.elapsed_ms` is written from
the monotonic clock, from the moment the case was opened to the moment
the VERDICT stage ended. Nothing about it is illustrative. It is the
number the case header shows, and it replaced an asserted "11 minutes".

Rule 1 is not at stake in this module and it is worth being explicit
about why: these are facts about the SYSTEM, not the business. No value
produced here may be wrapped as `Evidence`, and none of it is ever shown
to the model.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, Sequence, get_args

import duckdb

from engine.contracts import RequestTelemetry, Stage, TelemetryEvent, VerdictValue
from semantic_layer.schema import SemanticLayer, get_semantic_layer
from telemetry.cost import Usage, cost_of_call, reported_usage

#: The five stages in their locked order, taken from the contract rather
#: than retyped. CLAUDE.md §"Architecture" calls this order the module
#: names, the API event names and the UI rail; there is one copy of it.
STAGES: tuple[Stage, ...] = get_args(Stage)

#: Separator for the list-valued columns. Matches `security/audit.py`, and
#: for the same reason: none of the values can contain it, so the stored
#: string round-trips.
LIST_SEPARATOR = ","


class TelemetryError(RuntimeError):
    """The recorder was used in a way that would produce a false measurement."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

#: Requests completed by this process. Module-level because "warm" is a
#: property of the PROCESS — the semantic layer cache, the DuckDB handle
#: and the fixture reads are all process-wide — not of any one recorder.
_COMPLETED_REQUESTS = 0


def completed_requests() -> int:
    return _COMPLETED_REQUESTS


def reset_warmup() -> None:
    """Forget the warm-up count. For tests, and for a re-exec'd worker."""
    global _COMPLETED_REQUESTS
    _COMPLETED_REQUESTS = 0


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


@dataclass
class Recorder:
    """Everything one request did, accumulated as it does it.

    Mutable by design — it is a ledger being written. `close()` freezes it
    into a `RequestTelemetry`, which is not.
    """

    user_id: str
    persona: str
    question: str | None = None
    case_id: str | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    layer: SemanticLayer | None = None
    mock: bool | None = None

    #: Injectable so a test can measure without waiting. `monotonic` must
    #: be monotonic (durations); `wall_clock` is for timestamps only.
    monotonic: Callable[[], float] = perf_counter
    wall_clock: Callable[[], datetime] = _utc_now

    def __post_init__(self) -> None:
        self.layer = self.layer or get_semantic_layer()
        if self.mock is None:
            self.mock = _mock_enabled()

        self.started_at: datetime = self.wall_clock()
        self._request_start: float = self.monotonic()
        self._warm: bool = (
            completed_requests() >= self.layer.telemetry.targets.warmup_requests
        )
        self._closed = False

        self._latency: dict[Stage, float] = {stage: 0.0 for stage in STAGES}
        self._entered: list[Stage] = []
        self._open_stage: Stage | None = None

        self._methods: list[str] = []
        self._usages: list[Usage] = []
        self._cost_inr: float = 0.0
        self._cost_estimated = False

        self._cache_hits = 0
        self._cache_misses = 0
        self._rows_scanned = 0
        self._rows_filtered = 0
        self._rows_released = 0
        self._claims_checked = 0
        self._claims_stripped = 0

        self._case_opened_at: float | None = None
        self._case_elapsed_ms: float | None = None
        self._events: list[TelemetryEvent] = []

    # -- stages ------------------------------------------------------------

    @contextmanager
    def stage(self, name: Stage) -> Iterator[Recorder]:
        """Time one stage. Records the elapsed time even when it raises.

        Re-entering a stage accumulates. Entering an EARLIER stage than
        the last one is refused: the five names are a rail with an order,
        and a request that appears to run GATHER before QUALIFY is a bug
        in the caller, not a measurement.
        """
        if self._closed:
            raise TelemetryError(f"request {self.request_id} is already closed")
        if name not in self._latency:
            raise TelemetryError(f"{name!r} is not one of the five stages {list(STAGES)}")
        if self._open_stage is not None:
            raise TelemetryError(
                f"cannot open {name!r} while {self._open_stage!r} is still open; "
                "stages run in sequence, and nesting them would double-count the time"
            )
        if self._entered and STAGES.index(name) < STAGES.index(self._entered[-1]):
            raise TelemetryError(
                f"{name!r} runs before {self._entered[-1]!r} on the rail; the stage "
                "order is locked (CLAUDE.md §\"Architecture\")"
            )

        self._open_stage = name
        started_at = self.wall_clock()
        started = self.monotonic()
        try:
            yield self
        finally:
            elapsed_ms = (self.monotonic() - started) * 1000.0
            self._latency[name] += elapsed_ms
            if name not in self._entered:
                self._entered.append(name)
            self._open_stage = None
            self._events.append(
                self._event("stage", stage=name, started_at=started_at, duration_ms=elapsed_ms)
            )
            # The case's own clock stops when the verdict is reached, not
            # when the request finishes. Anything after VERDICT — writing
            # the narrative, persisting artefacts — is our housekeeping,
            # not the analyst's wait for an answer.
            if name == STAGES[-1] and self._case_opened_at is not None:
                self._case_elapsed_ms = (self.monotonic() - self._case_opened_at) * 1000.0

    def mark_case_opened(self, case_id: str, *, at: float | None = None) -> None:
        """Start the case clock. QUALIFY calls this when it registers a case.

        `at` is a `monotonic()` reading, for the caller that opened the
        case slightly before it got round to saying so. Everything else
        should leave it alone.
        """
        self.case_id = case_id
        self._case_opened_at = self.monotonic() if at is None else at

    # -- what happened -----------------------------------------------------

    def record_method(self, name: str) -> None:
        """Note that an analytical method ran. First-seen order, no repeats.

        The list is of names, not counts, because the question it answers
        is "which tests actually ran on this case" — a case that skipped
        the DiD should be visible as such at a glance.
        """
        if name not in self._methods:
            self._methods.append(name)

    def record_methods(self, names: Iterable[str]) -> None:
        for name in names:
            self.record_method(name)

    def record_llm_call(
        self, response: Any, *, request: Any = None, detail: str | None = None
    ) -> float:
        """Price one model call and add it to the request. Returns the rupees.

        Takes an `llm.provider.LLMResponse`. A response for a model with
        no entry in `telemetry.yaml -> pricing` raises rather than costing
        zero — see the note at the top of `telemetry/cost.py`.

        Pass `request` too when you have it. A replayed fixture reports no
        token counts, and the request is the only place the input text
        still exists to estimate them from; without it an offline run
        under-reports its own input. Either way the row records that the
        figure was estimated.
        """
        usage, estimated = reported_usage(response, request=request, layer=self.layer)
        call = cost_of_call(usage, self.layer)
        self._usages.append(usage)
        self._cost_estimated = self._cost_estimated or estimated
        self._cost_inr += call.inr
        self._events.append(
            self._event(
                "llm_call",
                stage=self._open_stage,
                started_at=self.wall_clock(),
                duration_ms=getattr(response, "latency_ms", None),
                usage=usage,
                cost_inr=call.inr,
                detail=detail,
            )
        )
        return call.inr

    def record_cache(self, *, hits: int = 0, misses: int = 0) -> None:
        """Content-hash cache outcomes (`llm_cache`), not the prompt cache."""
        self._cache_hits += hits
        self._cache_misses += misses

    def record_classification(self, run: Any) -> None:
        """Fold in an `llm.classify.ClassificationRun`'s cache outcomes."""
        hits = int(run.cache_hits)
        self.record_cache(hits=hits, misses=int(run.lookups) - hits)

    def record_query(self, result: Any) -> None:
        """Fold in one `engine.db.GovernedResult`.

        `rows_scanned` is what the policy EVALUATED — the rows the
        caller's query produced, before the row predicate removed any. It
        is not a storage-layer scan count; DuckDB's is not exposed here,
        and the number that matters for governance is how many rows the
        policy had an opinion about.
        """
        self._rows_scanned += len(result.rows) + int(result.rows_filtered)
        self._rows_filtered += int(result.rows_filtered)

    def record_rows(
        self, *, scanned: int = 0, filtered: int = 0, released: int = 0
    ) -> None:
        self._rows_scanned += scanned
        self._rows_filtered += filtered
        self._rows_released += released

    def record_release(self, payload: Any) -> None:
        """Note rows crossing the trust boundary.

        Takes anything with a `row_count` (a `security.redaction.LlmPayload`)
        or `rows_released_to_llm` (a `security.audit.AuditRecord`), so the
        release is counted from the same object the audit row was written
        from rather than from a second count that could disagree with it.
        """
        count = getattr(payload, "row_count", None)
        if count is None:
            count = getattr(payload, "rows_released_to_llm", None)
        if count is None:
            raise TelemetryError(
                f"{type(payload).__name__} carries neither row_count nor "
                "rows_released_to_llm; pass the payload or the audit record"
            )
        self._rows_released += int(count)

    def record_grounding(self, report: Any) -> None:
        """Fold in an `llm.grounding.GroundingReport`."""
        self._claims_checked += int(report.claims_checked)
        self._claims_stripped += int(report.claims_stripped)

    # -- closing -----------------------------------------------------------

    def close(
        self,
        *,
        verdict: VerdictValue | None = None,
        confidence: float | None = None,
    ) -> RequestTelemetry:
        """Freeze the ledger into the row. Callable once."""
        if self._closed:
            raise TelemetryError(f"request {self.request_id} is already closed")
        if self._open_stage is not None:
            raise TelemetryError(
                f"stage {self._open_stage!r} is still open; close it before the request"
            )

        total_ms = (self.monotonic() - self._request_start) * 1000.0
        elapsed_ms = self._case_elapsed_ms
        if elapsed_ms is None and self._case_opened_at is not None:
            # A case was opened but VERDICT never ran — a gate kill, or a
            # failure mid-adjudication. The case clock still stops here,
            # and the row says which stages ran, so the two read together.
            elapsed_ms = (self.monotonic() - self._case_opened_at) * 1000.0

        row = RequestTelemetry(
            request_id=self.request_id,
            case_id=self.case_id,
            user_id=self.user_id,
            persona=self.persona,
            question=self.question,
            verdict=verdict,
            confidence=confidence,
            started_at=self.started_at,
            total_latency_ms=total_ms,
            latency_by_stage=dict(self._latency),
            stages_entered=tuple(self._entered),
            warm=self._warm,
            analytical_methods_executed=tuple(self._methods),
            llm_calls=len(self._usages),
            model_per_call=tuple(usage.model for usage in self._usages),
            input_tokens=sum(u.input_tokens for u in self._usages),
            output_tokens=sum(u.output_tokens for u in self._usages),
            cache_read_input_tokens=sum(u.cache_read_input_tokens for u in self._usages),
            cache_creation_input_tokens=sum(
                u.cache_creation_input_tokens for u in self._usages
            ),
            estimated_cost_inr=self._cost_inr,
            cost_estimated=self._cost_estimated,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            rows_scanned=self._rows_scanned,
            rows_filtered_by_policy=self._rows_filtered,
            rows_released_to_llm=self._rows_released,
            grounding_claims_checked=self._claims_checked,
            grounding_claims_stripped=self._claims_stripped,
            case_elapsed_ms=elapsed_ms,
            mock=bool(self.mock),
        )

        self._closed = True
        global _COMPLETED_REQUESTS
        _COMPLETED_REQUESTS += 1
        return row

    def events(self) -> tuple[TelemetryEvent, ...]:
        """The per-step detail behind the row, for `telemetry_event`."""
        return tuple(self._events)

    # -- internals ---------------------------------------------------------

    def _event(
        self,
        event: str,
        *,
        stage: Stage | None,
        started_at: datetime,
        duration_ms: float | None = None,
        usage: Usage | None = None,
        cost_inr: float | None = None,
        detail: str | None = None,
    ) -> TelemetryEvent:
        return TelemetryEvent(
            event_id=uuid.uuid4().hex,
            event=event,
            stage=stage,
            case_id=self.case_id,
            started_at=started_at,
            duration_ms=duration_ms,
            model=usage.model if usage else None,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            cache_read_input_tokens=usage.cache_read_input_tokens if usage else None,
            cache_creation_input_tokens=(
                usage.cache_creation_input_tokens if usage else None
            ),
            cost_inr=cost_inr,
            mock=bool(self.mock),
            detail=detail,
        )


@dataclass
class MeteredProvider:
    """An `LLMProvider` that reports every call to a `Recorder`.

        provider = MeteredProvider(get_provider(), recorder)
        story = narrate(adjudication, "cxo", layer, provider=provider)

    WHY A WRAPPER RATHER THAN A CALL AT EVERY CALL SITE. `narrate()` makes
    one call, or two when a sentence fails grounding and is regenerated.
    `hypothesise` makes one. A call site that has to remember to meter
    itself will eventually forget, and the failure is silent — the case
    reports a cost that is missing a call. Wrapping the provider means
    the only way to make a model call is to be counted.

    It satisfies the `LLMProvider` protocol structurally (a `name` and a
    `complete`), so nothing downstream knows it is there.
    """

    inner: Any
    recorder: Recorder

    @property
    def name(self) -> str:
        return f"metered:{self.inner.name}"

    def complete(self, request: Any) -> Any:
        response = self.inner.complete(request)
        self.recorder.record_llm_call(response, request=request, detail=request.task)
        return response


def _mock_enabled() -> bool:
    """Whether this process is running offline (CLAUDE.md rule 8).

    Imported here rather than at module scope so `telemetry` does not
    depend on `llm` being importable, and so the truthy-string set lives
    in exactly one place.
    """
    from llm.provider import mock_enabled

    return mock_enabled()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

REQUEST_INSERT = (
    "INSERT INTO telemetry_request ("
    "request_id, case_id, user_id, persona, question, verdict, confidence, "
    "started_at, total_latency_ms, latency_validate_ms, latency_qualify_ms, "
    "latency_gather_ms, latency_adjudicate_ms, latency_verdict_ms, stages_entered, "
    "warm, analytical_methods_executed, llm_calls, model_per_call, input_tokens, "
    "output_tokens, cache_read_input_tokens, cache_creation_input_tokens, "
    "estimated_cost_inr, cost_estimated, cache_hits, cache_misses, rows_scanned, "
    "rows_filtered_by_policy, rows_released_to_llm, grounding_claims_checked, "
    "grounding_claims_stripped, mock) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
    "?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

EVENT_INSERT = (
    "INSERT INTO telemetry_event ("
    "event_id, event, stage, case_id, started_at, duration_ms, model, input_tokens, "
    "output_tokens, cache_read_input_tokens, cache_creation_input_tokens, cost_inr, "
    "mock, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

#: Written by telemetry and nothing else. The elapsed time is a
#: measurement, and the module that took it is the module that stores it.
CASE_ELAPSED_UPDATE = "UPDATE case_registry SET elapsed_ms = ? WHERE case_id = ?"


def as_row(row: RequestTelemetry) -> tuple:
    """Positional values for `REQUEST_INSERT`."""
    from engine.db import as_stored_timestamp

    return (
        row.request_id,
        row.case_id,
        row.user_id,
        row.persona,
        row.question,
        row.verdict,
        row.confidence,
        as_stored_timestamp(row.started_at),
        row.total_latency_ms,
        *(row.latency_by_stage[stage] for stage in STAGES),
        LIST_SEPARATOR.join(row.stages_entered),
        row.warm,
        LIST_SEPARATOR.join(row.analytical_methods_executed),
        row.llm_calls,
        LIST_SEPARATOR.join(row.model_per_call),
        row.input_tokens,
        row.output_tokens,
        row.cache_read_input_tokens,
        row.cache_creation_input_tokens,
        row.estimated_cost_inr,
        row.cost_estimated,
        row.cache_hits,
        row.cache_misses,
        row.rows_scanned,
        row.rows_filtered_by_policy,
        row.rows_released_to_llm,
        row.grounding_claims_checked,
        row.grounding_claims_stripped,
        row.mock,
    )


def persist(
    connection: duckdb.DuckDBPyConnection,
    row: RequestTelemetry,
    *,
    events: Sequence[TelemetryEvent] = (),
) -> None:
    """Write the request row, its events, and the case's elapsed time.

    Unlike the audit write in `engine/db.py`, this one RAISES on failure.
    An audit row that cannot be written must not eat a query result; a
    telemetry row that cannot be written means the accept criterion —
    every request writes a complete row — is not being met, and silence
    would hide exactly that.
    """
    from engine.db import as_stored_timestamp

    try:
        connection.execute(REQUEST_INSERT, list(as_row(row)))
    except duckdb.Error as exc:
        raise TelemetryError(
            f"could not write telemetry for request {row.request_id}: {exc}\n"
            "If the table is missing, the warehouse predates it — run "
            "`engine.warehouse.load.create_schema(connection)` or `make seed`."
        ) from exc

    for event in events:
        connection.execute(
            EVENT_INSERT,
            [
                event.event_id,
                event.event,
                event.stage,
                event.case_id,
                as_stored_timestamp(event.started_at),
                event.duration_ms,
                event.model,
                event.input_tokens,
                event.output_tokens,
                event.cache_read_input_tokens,
                event.cache_creation_input_tokens,
                event.cost_inr,
                event.mock,
                event.detail,
            ],
        )

    if row.case_id is not None and row.case_elapsed_ms is not None:
        write_case_elapsed(connection, row.case_id, row.case_elapsed_ms)


def write_case_elapsed(
    connection: duckdb.DuckDBPyConnection, case_id: str, elapsed_ms: float
) -> None:
    """Stamp `case_registry.elapsed_ms` — resolved defect 6, measured.

    Refuses to write against a case that was never registered. A stamp on
    a case nobody opened is a number with nothing behind it, which is the
    exact failure this measurement was introduced to correct.
    """
    known = connection.execute(
        "SELECT 1 FROM case_registry WHERE case_id = ?", [case_id]
    ).fetchone()
    if known is None:
        raise TelemetryError(
            f"case {case_id!r} is not in case_registry; QUALIFY registers a case "
            "before telemetry can stamp how long it took"
        )
    connection.execute(CASE_ELAPSED_UPDATE, [elapsed_ms, case_id])


def load_requests(
    connection: duckdb.DuckDBPyConnection, *, case_id: str | None = None
) -> tuple[RequestTelemetry, ...]:
    """Read rows back out of `telemetry_request`, oldest first."""
    sql = "SELECT * FROM telemetry_request"
    params: list[Any] = []
    if case_id is not None:
        sql += " WHERE case_id = ?"
        params.append(case_id)
    sql += " ORDER BY started_at"

    cursor = connection.execute(sql, params) if params else connection.execute(sql)
    columns = [description[0] for description in cursor.description]
    return tuple(
        _from_stored(dict(zip(columns, record, strict=True)))
        for record in cursor.fetchall()
    )


def _split(value: str | None) -> tuple[str, ...]:
    return tuple(part for part in (value or "").split(LIST_SEPARATOR) if part)


def _from_stored(stored: dict[str, Any]) -> RequestTelemetry:
    latency = {
        stage: float(stored[f"latency_{stage.lower()}_ms"]) for stage in STAGES
    }
    return RequestTelemetry(
        request_id=stored["request_id"],
        case_id=stored["case_id"],
        user_id=stored["user_id"],
        persona=stored["persona"],
        question=stored["question"],
        verdict=stored["verdict"],
        confidence=stored["confidence"],
        started_at=stored["started_at"].replace(tzinfo=UTC),
        total_latency_ms=float(stored["total_latency_ms"]),
        latency_by_stage=latency,
        stages_entered=_split(stored["stages_entered"]),
        warm=bool(stored["warm"]),
        analytical_methods_executed=_split(stored["analytical_methods_executed"]),
        llm_calls=int(stored["llm_calls"]),
        model_per_call=_split(stored["model_per_call"]),
        input_tokens=int(stored["input_tokens"]),
        output_tokens=int(stored["output_tokens"]),
        cache_read_input_tokens=int(stored["cache_read_input_tokens"]),
        cache_creation_input_tokens=int(stored["cache_creation_input_tokens"]),
        estimated_cost_inr=float(stored["estimated_cost_inr"]),
        cost_estimated=bool(stored["cost_estimated"]),
        cache_hits=int(stored["cache_hits"]),
        cache_misses=int(stored["cache_misses"]),
        rows_scanned=int(stored["rows_scanned"]),
        rows_filtered_by_policy=int(stored["rows_filtered_by_policy"]),
        rows_released_to_llm=int(stored["rows_released_to_llm"]),
        grounding_claims_checked=int(stored["grounding_claims_checked"]),
        grounding_claims_stripped=int(stored["grounding_claims_stripped"]),
        # Not a column: the case's elapsed time is stored once, on
        # case_registry, so there is one copy of it and no way for the
        # two to disagree.
        case_elapsed_ms=None,
        mock=bool(stored["mock"]),
    )


# ---------------------------------------------------------------------------
# Latency, read back
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. `q` in [0, 1]. Raises on no values.

    Written out rather than reached for, because the alternative is
    importing numpy into the telemetry package for six lines, and because
    which interpolation is in use should be readable at the point the
    published claim is computed.
    """
    if not values:
        raise TelemetryError("no measurements to take a percentile of")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True)
class LatencyReport:
    """A measured percentile against the published budget."""

    measured_ms: float
    budget_ms: float
    quantile: float
    sample_size: int
    warm_only: bool
    excluded_cold: int

    @property
    def within(self) -> bool:
        return self.measured_ms <= self.budget_ms

    @property
    def headroom_ms(self) -> float:
        return self.budget_ms - self.measured_ms

    def render(self) -> str:
        verb = "within" if self.within else "OVER"
        warmth = "warm" if self.warm_only else "all"
        return (
            f"P{self.quantile:.0%} latency: {self.measured_ms:,.0f} ms over "
            f"{self.sample_size} {warmth} requests "
            f"({self.excluded_cold} cold excluded) — {verb} the "
            f"{self.budget_ms:,.0f} ms budget"
        )


def check_latency(
    rows: Sequence[RequestTelemetry],
    *,
    layer: SemanticLayer | None = None,
    warm_only: bool = True,
) -> LatencyReport:
    """The published latency claim, computed from rows.

    Warm rows only by default: the target in CLAUDE.md says "warm", and
    the honest way to honour that qualifier is to exclude the requests
    the row itself marks cold, not to run the numbers twice and pick.
    """
    layer = layer or get_semantic_layer()
    targets = layer.telemetry.targets

    considered = [row for row in rows if row.warm] if warm_only else list(rows)
    excluded = len(rows) - len(considered)
    if not considered:
        why = f"all {len(rows)} were cold" if rows else "none were supplied"
        raise TelemetryError(
            f"no {'warm ' if warm_only else ''}requests to measure; {why}"
        )

    return LatencyReport(
        measured_ms=percentile(
            [row.total_latency_ms for row in considered], targets.latency_percentile
        ),
        budget_ms=targets.latency_budget_ms,
        quantile=targets.latency_percentile,
        sample_size=len(considered),
        warm_only=warm_only,
        excluded_cold=excluded,
    )


__all__ = [
    "CASE_ELAPSED_UPDATE",
    "EVENT_INSERT",
    "LIST_SEPARATOR",
    "REQUEST_INSERT",
    "STAGES",
    "LatencyReport",
    "MeteredProvider",
    "Recorder",
    "TelemetryError",
    "as_row",
    "check_latency",
    "completed_requests",
    "load_requests",
    "percentile",
    "persist",
    "reset_warmup",
    "write_case_elapsed",
]
