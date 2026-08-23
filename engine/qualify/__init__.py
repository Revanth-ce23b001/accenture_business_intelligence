"""QUALIFY — Gates 2 to 5, and the restraint that keeps the system readable.

CLAUDE.md §Architecture:

    Data -> VALIDATE -> QUALIFY -> GATHER -> ADJUDICATE -> VERDICT

VALIDATE asked whether the movement is real. QUALIFY asks whether it is
worth explaining:

    Gate 2  calendar      decompose, do not adjust
    Gate 3  band          enough history? residual outside the band?
    Gate 4  specificity   this scope's problem, or the market's?
    Gate 5  materiality   worth the owner's afternoon?

then restraint: one cause one case, no re-opening what is open, and three
cases a week per owner.

    from engine.qualify import QualifyRequest, qualify

    result = qualify(connection, user, QualifyRequest(
        kpi="net_revenue", scope="West", grain="monthly",
        period="2025-11", comparison_period="2025-10",
    ))
    result.decomposition.headline_pt   # -8.1
    result.decomposition.calendar_pt   # -3.2
    result.decomposition.residual_pt   # -4.9
    result.region_strip                # N -0.4 . S +1.1 . E -0.6 . W -4.9
"""

from engine.qualify.band import HistoryCheck, ResidualBand, check_history, residual_band
from engine.qualify.calendar import (
    CalendarDecomposition,
    CalendarError,
    CalendarFit,
    decompose,
    fit_calendar,
    previous_month,
)
from engine.qualify.gate import (
    QualifyError,
    QualifyRequest,
    QualifyResult,
    qualify,
)
from engine.qualify.history import observed_period_count, primary_source
from engine.qualify.materiality import MaterialityVerdict, assess_materiality
from engine.qualify.restraint import (
    RestraintVerdict,
    assess_restraint,
    correlated_kpis,
    open_cases,
    register_case,
)
from engine.qualify.series import RegionalSeries, SeriesError, load_series
from engine.qualify.specificity import SpecificityVerdict, assess_specificity

__all__ = [
    "CalendarDecomposition",
    "CalendarError",
    "CalendarFit",
    "HistoryCheck",
    "MaterialityVerdict",
    "QualifyError",
    "QualifyRequest",
    "QualifyResult",
    "RegionalSeries",
    "ResidualBand",
    "RestraintVerdict",
    "SeriesError",
    "SpecificityVerdict",
    "assess_materiality",
    "assess_restraint",
    "assess_specificity",
    "check_history",
    "correlated_kpis",
    "decompose",
    "fit_calendar",
    "load_series",
    "observed_period_count",
    "open_cases",
    "previous_month",
    "primary_source",
    "qualify",
    "register_case",
    "residual_band",
]
