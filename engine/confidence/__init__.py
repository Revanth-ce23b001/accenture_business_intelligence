"""Confidence — six components, four caps, one isotonic map.

    conf_raw = 0.28*s1 + 0.20*s2 + 0.16*s3 + 0.14*s4 + 0.10*s5 + 0.12*s6

then the caps, which cannot be outvoted, then the organisation's own
track record:

    the model scored this 89%
    our record on cases like it says we run about five points hot
    so we publish 84%

    from engine.confidence import score_case
    score = score_case(connection, user, adjudication,
                       leading_tag="stock_out", case_type="availability_attribution")
    score.raw          # the weighted sum
    score.after_caps   # what it is allowed to be
    score.calibrated   # what gets published
    print(score.render())
"""

from engine.confidence.calibration import (
    Band,
    Calibration,
    CalibrationError,
    fit_calibration,
)
from engine.confidence.caps import (
    CONFOUNDER,
    LOW_RELIABILITY,
    MISSING_SOURCE,
    PRECEDENCE,
    Cap,
    apply_caps,
    evaluate_caps,
    forced_triggers,
)
from engine.confidence.components import (
    KEYS,
    Component,
    ComponentError,
    ScoringContext,
    SourceHealth,
    compute_components,
    weighted_sum,
)
from engine.confidence.gate import (
    ConfidenceError,
    Score,
    score_case,
    source_health,
    to_contract,
)

__all__ = [
    "CONFOUNDER",
    "KEYS",
    "LOW_RELIABILITY",
    "MISSING_SOURCE",
    "PRECEDENCE",
    "Band",
    "Calibration",
    "CalibrationError",
    "Cap",
    "Component",
    "ComponentError",
    "ConfidenceError",
    "Score",
    "ScoringContext",
    "SourceHealth",
    "apply_caps",
    "compute_components",
    "evaluate_caps",
    "fit_calibration",
    "forced_triggers",
    "score_case",
    "source_health",
    "to_contract",
    "weighted_sum",
]
