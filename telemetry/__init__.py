"""Telemetry — what the system did, what it cost and how long it took.

Two modules, as CLAUDE.md §"Repository layout" specifies:

    recorder.py   the context manager around each of the five stages, the
                  request row, and the wall-clock measurement that lands
                  on `case_registry.elapsed_ms`
    cost.py       the price list applied to token counts, and the scale
                  projection

Everything here describes the SYSTEM. A case's evidence describes the
BUSINESS. The two never mix: no value produced in this package is wrapped
as `Evidence`, shown on an evidence panel, or handed to the model.
"""

from telemetry.cost import (
    Budget,
    CallCost,
    Projection,
    Usage,
    check_cost,
    cost_of_call,
    cost_of_case,
    estimate_usage,
    mean_cost_per_request,
    price_of,
    project,
    project_at_ceiling,
    project_measured,
    reported_usage,
)
from telemetry.recorder import (
    STAGES,
    LatencyReport,
    MeteredProvider,
    Recorder,
    TelemetryError,
    check_latency,
    load_requests,
    percentile,
    persist,
    reset_warmup,
    write_case_elapsed,
)

__all__ = [
    "STAGES",
    "Budget",
    "CallCost",
    "LatencyReport",
    "MeteredProvider",
    "Projection",
    "Recorder",
    "TelemetryError",
    "Usage",
    "check_cost",
    "check_latency",
    "cost_of_call",
    "cost_of_case",
    "estimate_usage",
    "load_requests",
    "mean_cost_per_request",
    "percentile",
    "persist",
    "price_of",
    "project",
    "project_at_ceiling",
    "project_measured",
    "reported_usage",
    "reset_warmup",
    "write_case_elapsed",
]
