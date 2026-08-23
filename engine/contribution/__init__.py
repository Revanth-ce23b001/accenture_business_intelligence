"""WHERE the change concentrated — not WHY it happened.

CLAUDE.md rule 7. This package is deliberately isolated from
`engine/adjudicate/`: it may not import it, and `tests/test_contribution.py`
asserts so statically. A decomposition and a verdict are different claims and
they render in different panels.
"""

from engine.contribution.decomposition import (
    COMPONENTS,
    HEADER,
    MIX,
    PRICE,
    VOLUME,
    Component,
    ContributionError,
    Decomposition,
    decompose,
    log_mean,
)
from engine.contribution.gate import (
    TOLERANCE_INR_LAKH,
    ContributionResult,
    contribution,
)

__all__ = [
    "COMPONENTS",
    "HEADER",
    "MIX",
    "PRICE",
    "TOLERANCE_INR_LAKH",
    "VOLUME",
    "Component",
    "ContributionError",
    "ContributionResult",
    "Decomposition",
    "contribution",
    "decompose",
    "log_mean",
]
