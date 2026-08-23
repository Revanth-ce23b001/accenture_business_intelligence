"""ADJUDICATE — the six tests, and no model call anywhere.

CLAUDE.md §Architecture:

    Data -> VALIDATE -> QUALIFY -> GATHER -> ADJUDICATE -> VERDICT

    1  temporal precedence      PELT changepoints. HARD GATE.
    2  effect-size sufficiency  volume share x elasticity. HARD GATE.
    3  dose-response            OLS of store effect on store cause. 0.15
    4  specificity              present against absent, Welch's t. 0.15
    5  matched-control DiD      covariate matching, plus a pre-test. 0.45
    6  confounder screen        balance across the two groups. CAP.

The two hard gates are the product. On case #2451 the marketing cut has
the strongest correlation with the decline in the whole data set and Test
1 eliminates it, because the cut lands after the decline started.

    from engine.adjudicate import AdjudicateRequest, adjudicate

    result = adjudicate(connection, user, AdjudicateRequest(...))
    result.verdict("stock_out").did.point_pt   # the DiD, in scope points
    result.eliminated                           # with machine-readable reasons
    result.coverage                             # what the survivors account for
"""

from engine.adjudicate.gate import (
    AdjudicateRequest,
    AdjudicationError,
    AdjudicationResult,
    HypothesisVerdict,
    adjudicate,
    to_contracts,
)
from engine.adjudicate.series import (
    DailySeries,
    SeriesError,
    StorePanel,
    daily_points,
    load_panel,
    load_series,
)
from engine.adjudicate.tests import (
    NOT_TESTABLE,
    ConfounderResult,
    DidEstimate,
    Exposure,
    Finding,
    Matching,
    Pair,
    balance,
    changepoint,
    estimate_did,
    exposed_stores,
    fit_elasticity,
    match_controls,
    parallel_trends,
    signed,
    test_confounders,
    test_did,
    test_dose_response,
    test_precedence,
    test_specificity,
    test_sufficiency,
)

__all__ = [
    "NOT_TESTABLE",
    "AdjudicateRequest",
    "AdjudicationError",
    "AdjudicationResult",
    "ConfounderResult",
    "DailySeries",
    "DidEstimate",
    "Exposure",
    "Finding",
    "HypothesisVerdict",
    "Matching",
    "Pair",
    "SeriesError",
    "StorePanel",
    "adjudicate",
    "balance",
    "changepoint",
    "daily_points",
    "estimate_did",
    "exposed_stores",
    "fit_elasticity",
    "load_panel",
    "load_series",
    "match_controls",
    "parallel_trends",
    "signed",
    "test_confounders",
    "test_did",
    "test_dose_response",
    "test_precedence",
    "test_specificity",
    "test_sufficiency",
    "to_contracts",
]
