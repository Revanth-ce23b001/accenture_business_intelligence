"""The six tests. Every number here is computed, none is asked for.

    1  temporal precedence      PELT changepoints. HARD GATE.
    2  effect-size sufficiency  volume share x elasticity. HARD GATE.
    3  dose-response            OLS of store effect on store cause. 0.15
    4  specificity              present against absent, Welch's t. 0.15
    5  matched-control DiD      covariate matching, plus a pre-test. 0.45
    6  confounder screen        balance across the two groups. CAP.

THE TWO HARD GATES ARE THE PRODUCT. On case #2451 the marketing cut has
the strongest correlation with the decline in the whole data set, and Test
1 eliminates it in one line because the cut lands after the decline
started. The price rise is real, and Test 2 eliminates it because 8% of
volume cannot move the number by 4.9 points however real it is. An LLM
asked "why did revenue fall?" on this data picks one of them.

A hard-gate FAILURE eliminates outright, with a machine-readable reason. A
hard gate that could not be RUN eliminates nothing: "we could not test
this" and "this is false" are different findings and must not render the
same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple
from datetime import date, timedelta

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats
from scipy.optimize import linear_sum_assignment

from engine.adjudicate.series import STORE, DailySeries, StorePanel, daily_points, units
from semantic_layer.schema import (
    ConfounderScreenSpec,
    DidSpec,
    ExposureSpec,
    DoseResponseSpec,
    PrecedenceSpec,
    SpecificitySpec,
    SufficiencySpec,
)

#: A test that could not be run. Not a pass and not a failure.
NOT_TESTABLE = "NOT_TESTABLE"


@dataclass(frozen=True)
class Finding:
    """One test's result, before it becomes a `TestResult` contract."""

    test_id: int
    name: str
    passed: bool
    testable: bool
    statistic: float | None
    p_value: float | None
    detail: str
    facts: dict[str, object] = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        if not self.testable:
            return NOT_TESTABLE
        return "PASS" if self.passed else "FAIL"


# ===========================================================================
# Test 1 — temporal precedence. HARD GATE.
# ===========================================================================


def changepoint(points: pd.Series, spec: PrecedenceSpec) -> date | None:
    """First changepoint PELT finds, as a date.

    The series is standardised first so one penalty works for a
    percentage, a rupee total and a count of complaints alike. `rbf`
    rather than `l2` because a ramp is a change in distribution before it
    is a change in mean, and #2451's availability falls over five days.
    """
    values = points.to_numpy(dtype=float)
    if len(values) <= spec.changepoint.min_size * spec.changepoint.min_segments:
        return None
    spread = values.std()
    standardised = (values - values.mean()) / (spread if spread else 1.0)

    algorithm = rpt.Pelt(
        model=spec.changepoint.model,
        min_size=spec.changepoint.min_size,
        jump=spec.changepoint.jump,
    ).fit(standardised.reshape(-1, 1))
    breaks = algorithm.predict(pen=spec.changepoint.penalty)[:-1]
    if not breaks:
        return None
    return points.index[breaks[0]].date()


def test_precedence(
    cause: DailySeries | None,
    effect: DailySeries,
    spec: PrecedenceSpec,
    *,
    reason: str,
    on_stores: tuple[str, ...] = (),
) -> Finding:
    """Did the cause begin before the effect did?

    Same-DAY is not a failure. #2451's mechanism is instantaneous — the
    shelf is empty and the basket is lost on the same trip — so a strict
    `<` at daily resolution would eliminate the true hypothesis and keep
    the late ones, which is precisely the wrong way round.
    """
    # A store-grain cause is measured where the cause is. Averaging a
    # fault across an estate it reached a quarter of dilutes the step by
    # a factor of four and reports the onset days late.
    restrict = on_stores if spec.measure_cause_on == "exposed" else ()
    effect_onset = changepoint(daily_points(effect), spec)
    if cause is None:
        return Finding(
            spec.test_id, "Temporal precedence", passed=True, testable=False,
            statistic=None, p_value=None,
            detail=f"no cause series is configured for this hypothesis: {reason}",
            facts={"effect_onset": effect_onset},
        )

    cause_onset = changepoint(daily_points(cause, restrict), spec)
    if cause_onset is None or effect_onset is None:
        return Finding(
            spec.test_id, "Temporal precedence", passed=True, testable=False,
            statistic=None, p_value=None,
            detail=(
                "no changepoint found in "
                + ("the cause" if cause_onset is None else "the effect")
                + " series; precedence cannot be established either way"
            ),
            facts={"cause_onset": cause_onset, "effect_onset": effect_onset},
        )

    lag = (cause_onset - effect_onset).days
    passed = lag <= spec.tolerance_days
    return Finding(
        spec.test_id, "Temporal precedence", passed=passed, testable=True,
        statistic=float(lag), p_value=None,
        detail=(
            f"cause onset {cause_onset}, effect onset {effect_onset}: the cause began "
            f"{abs(lag)} day(s) {'after' if lag > 0 else 'before or with'} the effect."
            + ("" if passed else " A cause cannot follow its effect.")
        ),
        facts={
            "cause_onset": cause_onset,
            "effect_onset": effect_onset,
            "lag_days": lag,
            "cause_label": cause.label,
            "measured_on": len(restrict) if restrict and cause.store_grain else None,
        },
    )


# ===========================================================================
# Test 2 — effect-size sufficiency. HARD GATE.
# ===========================================================================


def fit_elasticity(frame: pd.DataFrame, spec: SufficiencySpec) -> tuple[float, float, int]:
    """Log-log OLS of volume on the driver. Returns (elasticity, r2, n).

    Observed, never assumed. A sufficiency test run on a textbook
    elasticity is a test of the textbook.
    """
    usable = frame[(frame["volume"] > 0) & (frame["price"] > 0)]
    if len(usable) < spec.elasticity.min_observations:
        return float("nan"), 0.0, len(usable)
    fit = stats.linregress(
        np.log(usable["price"].to_numpy(float)), np.log(usable["volume"].to_numpy(float))
    )
    return float(fit.slope), float(fit.rvalue * fit.rvalue), len(usable)


def test_sufficiency(
    elasticity: float,
    r_squared: float,
    observations: int,
    *,
    volume_share: float | None,
    cause_magnitude_pct: float | None,
    residual_pt: float,
    spec: SufficiencySpec,
    min_residual_share: float,
    label: str,
    declared: float | None = None,
    declared_source: str | None = None,
) -> Finding:
    """Could this hypothesis produce a movement this size, at its very best?

        modelled_max = volume_share x cause_magnitude x |elasticity|

    This is the test that eliminates a hypothesis which is TRUE and too
    small. H3's price rise happened; on 8% of volume it cannot move the
    number by 4.9 points, and no amount of correlation changes that.
    """
    test_id = spec.test_id
    if volume_share is None or cause_magnitude_pct is None:
        return Finding(
            test_id, "Effect-size sufficiency", passed=True, testable=False,
            statistic=None, p_value=None,
            detail=(
                f"{label}: no affected volume share or cause magnitude is available, so "
                "no upper bound can be computed. Not tested, not eliminated."
            ),
        )
    fitted = np.isfinite(elasticity) and r_squared >= spec.elasticity.min_r_squared
    fallback = spec.elasticity.fallback
    provenance = "fitted"
    margin = 1.0

    if not fitted:
        # The history will not yield an elasticity. Fall back to the
        # declared one if there is one, and carry the fact that it is
        # declared all the way to the rendered evidence.
        if fallback is None or declared is None:
            return Finding(
                test_id, "Effect-size sufficiency", passed=True, testable=False,
                statistic=None, p_value=None,
                detail=(
                    f"{label}: the elasticity fit explains {r_squared:.1%} of variance "
                    f"over {observations} observations, below the "
                    f"{spec.elasticity.min_r_squared:.0%} floor, and no elasticity is "
                    "declared for this hypothesis. Nothing can be bounded, so nothing "
                    "is eliminated."
                ),
                facts={"elasticity": elasticity, "r_squared": r_squared},
            )
        elasticity = declared
        provenance = "declared"
        margin = fallback.elimination_margin

    modelled_max = abs(volume_share * cause_magnitude_pct * elasticity)
    required = abs(residual_pt) * min_residual_share
    # On a declared elasticity the bound has to miss by the margin: the
    # conclusion must survive the declared figure being wrong by that
    # factor, or it is not drawn.
    passed = modelled_max * margin >= required

    if provenance == "fitted":
        preamble = f"elasticity {elasticity:.2f} (fitted, r2 {r_squared:.2f})"
        caveat = ""
    else:
        preamble = f"elasticity {elasticity:.2f} (DECLARED, not fitted)"
        caveat = (
            f" The elasticity could not be fitted from this warehouse's history "
            f"({r_squared:.1%} of variance explained over {observations} observations), "
            f"so the bound uses {declared_source}. It would have to be wrong by more "
            f"than {margin:.0f}x to change this."
        )

    return Finding(
        test_id, "Effect-size sufficiency", passed=passed, testable=True,
        statistic=float(modelled_max), p_value=None,
        detail=(
            f"{volume_share:.0%} of volume x {cause_magnitude_pct:.1f}% x {preamble} "
            f"= at most {modelled_max:.2f} pt, against a "
            f"{abs(residual_pt):.1f} pt residual "
            f"(needs {required:.2f} pt to be worth pursuing)."
            + ("" if passed else " It cannot have produced a movement this size.")
            + caveat
        ),
        facts={
            "modelled_max_pt": modelled_max,
            "required_pt": required,
            "elasticity": float(elasticity),
            "elasticity_provenance": provenance,
            "elimination_margin": margin,
            "r_squared": r_squared,
            "observations": observations,
            "volume_share": volume_share,
            "cause_magnitude_pct": cause_magnitude_pct,
        },
    )


# ===========================================================================
# Test 3 — dose-response. Weighted.
# ===========================================================================


def test_dose_response(
    dose: pd.Series,
    panel: StorePanel,
    spec: DoseResponseSpec,
    *,
    min_r: float,
    direction: str | None,
) -> Finding:
    """Does a bigger cause go with a bigger effect, store by store?

    Across EVERY store in scope, treated and untreated. Restricting it to
    the treated group would measure the spread inside a group that all got
    the same treatment, which is a different and much weaker question.
    """
    if dose.empty:
        return Finding(
            spec.test_id, "Dose-response", passed=True, testable=False,
            statistic=None, p_value=None,
            detail=(
                "the cause has no store dimension, so it has no dose. A regional feed "
                "cannot be regressed on store-level effect."
            ),
        )

    merged = panel.frame.merge(
        dose.rename("dose"), left_on=STORE, right_index=True, how="inner"
    )
    if len(merged) < spec.min_stores:
        return Finding(
            spec.test_id, "Dose-response", passed=True, testable=False,
            statistic=None, p_value=None,
            detail=f"{len(merged)} stores carry both a dose and an effect; "
                   f"{spec.min_stores} are needed.",
        )

    percent = units().percent_scale
    decline = ((merged["pre"] - merged["post"]) / merged["pre"] * percent).to_numpy()
    magnitude = signed(merged["dose"].to_numpy(float), direction)

    r, p_value = stats.pearsonr(magnitude, decline)
    fit = stats.linregress(magnitude, decline)
    passed = abs(r) >= min_r
    return Finding(
        spec.test_id, "Dose-response", passed=passed, testable=True,
        statistic=float(r), p_value=float(p_value),
        detail=(
            f"r = {r:.2f} over {len(merged)} stores (p = {p_value:.2e}); a one-unit "
            f"larger cause goes with a {fit.slope:.2f} pt larger decline."
        ),
        facts={
            "r": float(r),
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
            "stores": len(merged),
        },
    )


# ===========================================================================
# Exposure — which stores the cause reached. Shared by Tests 4 and 5.
# ===========================================================================


@dataclass(frozen=True)
class Exposure:
    """The present-and-absent split, and how it was drawn."""

    exposed: tuple[str, ...]
    unexposed: tuple[str, ...]
    threshold: float | None
    median: float | None
    testable: bool
    reason: str

    @property
    def share(self) -> float:
        total = len(self.exposed) + len(self.unexposed)
        return len(self.exposed) / total if total else 0.0


def signed(magnitude: np.ndarray, direction: str | None) -> np.ndarray:
    """Point the dose the same way as the decline.

    A cause that FALLS — availability — is a bigger dose when it falls
    further, so its sign is flipped. Without this a stock-out looks like a
    negative dose and every correlation comes out backwards.
    """
    return -magnitude if direction == "down" else magnitude


def exposed_stores(
    dose: pd.Series, spec: ExposureSpec, *, direction: str | None
) -> Exposure:
    """Stores whose own cause value stands apart from the scope's.

    A robust deviation, not a quantile. A quantile always selects a fixed
    share of the estate including when the cause reached nobody; the
    median plus a multiple of the median absolute deviation selects the
    stores that actually stand apart, and selects none when none do.
    """
    if dose.empty:
        return Exposure((), (), None, None, False, "the cause has no store dimension.")

    ids = np.asarray(dose.index, dtype=object)
    magnitude = signed(dose.to_numpy(float), direction)
    median = float(np.median(magnitude))
    mad = float(np.median(np.abs(magnitude - median)))
    if mad <= 0.0:
        return Exposure(
            (), tuple(str(i) for i in ids), None, median, False,
            "every store in scope carries the same cause value; there is no split.",
        )

    threshold = median + spec.sigma * spec.mad_scale * mad
    mask = magnitude >= threshold
    exposed = tuple(str(i) for i in ids[mask])
    unexposed = tuple(str(i) for i in ids[~mask])

    if len(exposed) < spec.min_group_size or len(unexposed) < spec.min_group_size:
        return Exposure(
            exposed, unexposed, threshold, median, False,
            f"{len(exposed)} stores stand apart and {len(unexposed)} do not; "
            f"{spec.min_group_size} are needed on each side.",
        )
    return Exposure(
        exposed, unexposed, threshold, median, True,
        f"{len(exposed)} of {len(ids)} stores sit more than {spec.sigma:g} robust "
        f"deviations adverse of the scope median.",
    )


# ===========================================================================
# Test 4 — specificity. Weighted.
# ===========================================================================


def test_specificity(
    exposure: Exposure,
    panel: StorePanel,
    spec: SpecificitySpec,
    *,
    max_p: float,
) -> Finding:
    """Did it happen where the cause was, and not where it was not?

    Welch's t, because two groups defined by a cause have no reason to
    share a variance. The split is the same `Exposure` that Test 5 uses,
    so the two tests are answering the same question about the same
    stores.
    """
    if not exposure.testable:
        return Finding(
            spec.test_id, "Specificity", passed=True, testable=False,
            statistic=None, p_value=None,
            detail=f"no present-and-absent split: {exposure.reason}",
        )

    present = panel.growth_pct(exposure.exposed) * -1.0
    absent = panel.growth_pct(exposure.unexposed) * -1.0
    if min(len(present), len(absent)) < spec.min_group_size:
        return Finding(
            spec.test_id, "Specificity", passed=True, testable=False,
            statistic=None, p_value=None,
            detail=(
                f"the present group has {len(present)} stores and the absent group "
                f"{len(absent)}; {spec.min_group_size} are needed on each side."
            ),
        )

    statistic, p_value = stats.ttest_ind(present, absent, equal_var=False)
    passed = p_value <= max_p and present.mean() > absent.mean()
    return Finding(
        spec.test_id, "Specificity", passed=passed, testable=True,
        statistic=float(statistic), p_value=float(p_value),
        detail=(
            f"stores where the cause was present declined {present.mean():.1f} pt against "
            f"{absent.mean():.1f} pt where it was absent "
            f"(n = {len(present)} and {len(absent)}, Welch t = {statistic:.1f}, "
            f"p = {p_value:.2e})."
        ),
        facts={
            "present_mean": float(present.mean()),
            "absent_mean": float(absent.mean()),
            "present_n": len(present),
            "absent_n": len(absent),
        },
    )


# ===========================================================================
# Test 5 — matched-control DiD. Weight 0.45.
# ===========================================================================


class Pair(NamedTuple):
    """One treated store, its control, and the distance between them."""

    treated: str
    control: str
    distance: float


@dataclass(frozen=True)
class Matching:
    """Which control each treated store was paired with, and how well."""

    pairs: tuple[Pair, ...]
    dropped: tuple[str, ...]
    covariates: tuple[str, ...]

    @property
    def treated(self) -> tuple[str, ...]:
        return tuple(pair.treated for pair in self.pairs)

    @property
    def controls(self) -> tuple[str, ...]:
        return tuple(pair.control for pair in self.pairs)

    @property
    def worst_distance(self) -> float:
        return max((pair.distance for pair in self.pairs), default=0.0)


def match_controls(panel: StorePanel, spec: DidSpec, treated_ids: list[str]) -> Matching:
    """Weighted-Mahalanobis matching, without replacement, solved OPTIMALLY.

    Greedy nearest-neighbour was the first implementation and it is
    order-dependent: the first treated store to be considered takes the
    best control, and whichever store happened to sort first therefore
    changes the answer. That is a defect in a number a decision hangs on.
    The pairing here minimises TOTAL distance across the whole set at once
    (`scipy.optimize.linear_sum_assignment`, the Hungarian algorithm), so
    the result depends on the data and not on the sort.

    A treated store with no acceptable match is DROPPED and counted, not
    paired with whatever was nearest. A bad match is worse than a smaller
    sample: it puts back the difference the design exists to remove.
    """
    exact = [name for name, cov in spec.matching.covariates.items() if cov.type == "exact"]
    numeric = [name for name, cov in spec.matching.covariates.items() if cov.type == "numeric"]
    weights = np.array([spec.matching.covariates[name].weight for name in numeric])
    scale = panel.frame[numeric].std().to_numpy()
    scale = np.where(scale == 0.0, 1.0, scale)

    wanted = {str(item) for item in treated_ids}
    is_treated = panel.frame[STORE].astype(str).isin(wanted)
    treated = panel.frame[is_treated].reset_index(drop=True)
    pool = panel.frame[~is_treated].reset_index(drop=True)
    if treated.empty or pool.empty:
        return Matching(
            pairs=(),
            dropped=tuple(str(item) for item in treated[STORE]),
            covariates=tuple(exact + numeric),
        )

    pool_numeric = pool[numeric].to_numpy(float)
    distances = np.full((len(treated), len(pool)), spec.matching.forbidden_distance)
    for index, row in treated.iterrows():
        allowed = np.ones(len(pool), dtype=bool)
        for name in exact:
            allowed &= (pool[name] == row[name]).to_numpy()
        delta = (pool_numeric - row[numeric].to_numpy(float)) / scale
        distances[index, allowed] = np.sqrt(
            (weights * delta * delta).sum(axis=1)
        )[allowed]

    rows, columns = linear_sum_assignment(distances)
    pairs: list[Pair] = []
    dropped: list[str] = []
    assigned = set()
    for left, right in zip(rows, columns, strict=True):
        distance = float(distances[left, right])
        store = str(treated.iloc[left][STORE])
        assigned.add(left)
        if distance > spec.matching.max_distance:
            dropped.append(store)
            continue
        pairs.append(Pair(store, str(pool.iloc[right][STORE]), distance))
    # More treated stores than controls: the assignment leaves the surplus
    # unpaired, and they are dropped and counted like any other.
    dropped.extend(
        str(treated.iloc[index][STORE])
        for index in range(len(treated))
        if index not in assigned
    )

    return Matching(
        pairs=tuple(sorted(pairs)),
        dropped=tuple(sorted(dropped)),
        covariates=tuple(exact + numeric),
    )


@dataclass(frozen=True)
class DidEstimate:
    """The DiD, in the same points as the residual it has to explain."""

    point_pt: float
    standard_error_pt: float
    lower_pt: float
    upper_pt: float
    confidence: float
    t_statistic: float
    p_value: float
    treated_growth_pct: float
    control_growth_pct: float
    matched_pairs: int

    @property
    def attributable_pt(self) -> float:
        """The conservative end of the interval, signed like the estimate.

        Attributing the point estimate claims a precision the data has not
        got. The hypothesis gets what it can defend and the rest stays
        unattributed, where the verdict table can see it.
        """
        return self.upper_pt if self.point_pt < 0 else self.lower_pt


def estimate_did(panel: StorePanel, matching: Matching, spec: DidSpec) -> DidEstimate:
    """did = (T_post - T_pre) - (C_post - C_pre), against the scope's level.

    Expressed against the scope's own pre-period revenue, so the answer is
    in the same percentage points as the residual it is competing to
    explain — not in points of the treated group, which would be a
    different and much larger number.
    """
    treated = panel.frame[panel.frame[STORE].isin(matching.treated)]
    control = panel.frame[panel.frame[STORE].isin(matching.controls)]

    percent = units().percent_scale
    treated_growth = ((treated["post"] / treated["pre"] - 1.0) * percent).to_numpy()
    control_growth = ((control["post"] / control["pre"] - 1.0) * percent).to_numpy()

    counterfactual = treated["pre"].to_numpy() * (1.0 + control_growth.mean() / percent)
    loss = float((treated["post"].to_numpy() - counterfactual).sum())
    scope_pre = panel.scope_pre_inr
    point = loss / scope_pre * percent

    treated_share = float(treated["pre"].sum()) / scope_pre
    standard_error = float(
        np.sqrt(
            treated_growth.var(ddof=1) / len(treated_growth)
            + control_growth.var(ddof=1) / len(control_growth)
        )
        * treated_share
    )
    statistic, p_value = stats.ttest_ind(treated_growth, control_growth, equal_var=False)
    # The two-sided interval at the declared confidence. `norm.interval`
    # rather than an arithmetic half-and-double, so the tail convention is
    # scipy's and not this module's.
    z = float(stats.norm.interval(spec.attribution.confidence)[1])

    return DidEstimate(
        point_pt=point,
        standard_error_pt=standard_error,
        lower_pt=point - z * standard_error,
        upper_pt=point + z * standard_error,
        confidence=spec.attribution.confidence,
        t_statistic=float(statistic),
        p_value=float(p_value),
        treated_growth_pct=float(treated_growth.mean()),
        control_growth_pct=float(control_growth.mean()),
        matched_pairs=len(matching.pairs),
    )


def parallel_trends(
    history: pd.DataFrame, matching: Matching, spec: DidSpec, onset: date
) -> tuple[float, float, int]:
    """OLS slope of log(treated / control) over the pre-period weeks.

    Returns (slope, p_value, weeks). A HIGH p-value is the pass, and it
    means no pre-trend was detected — not that the trends were parallel.
    Absence of evidence, and the evidence says so.
    """
    frame = history.copy()
    frame["txn_date"] = pd.to_datetime(frame["txn_date"])
    frame["week"] = (
        (frame["txn_date"] - pd.Timestamp(onset)).dt.days // units().days_per_week
    )

    treated = set(matching.treated)
    control = set(matching.controls)
    rows = []
    for week, block in frame[frame["week"] < 0].groupby("week"):
        t = block[block[STORE].isin(treated)]["net_revenue_inr"].sum()
        c = block[block[STORE].isin(control)]["net_revenue_inr"].sum()
        if t > 0 and c > 0:
            rows.append((int(week), float(np.log(t / c))))
    if len(rows) < spec.pretest.min_points:
        return float("nan"), float("nan"), len(rows)

    weeks = np.array([row[0] for row in rows], dtype=float)
    ratio = np.array([row[1] for row in rows], dtype=float)
    fit = stats.linregress(weeks, ratio)
    return float(fit.slope), float(fit.pvalue), len(rows)


def test_did(
    estimate: DidEstimate,
    matching: Matching,
    pretest: tuple[float, float, int],
    spec: DidSpec,
    *,
    max_p: float,
) -> Finding:
    slope, pretest_p, weeks = pretest
    pretest_passed = bool(np.isfinite(pretest_p) and pretest_p > spec.pretest.min_p_value)
    passed = estimate.p_value <= max_p and pretest_passed

    return Finding(
        spec.test_id, "Matched-control difference-in-differences",
        passed=passed, testable=True,
        statistic=estimate.point_pt, p_value=estimate.p_value,
        detail=(
            f"{estimate.matched_pairs} treated stores against {estimate.matched_pairs} "
            f"matched controls: {estimate.point_pt:+.2f} pt of scope revenue "
            f"(p = {estimate.p_value:.2e}). Parallel-trends pre-test over {weeks} weeks: "
            f"p = {pretest_p:.2f}, {'passes' if pretest_passed else 'FAILS'}."
            + ("" if not matching.dropped else
               f" {len(matching.dropped)} treated store(s) had no acceptable match and "
               "were dropped rather than paired badly.")
        ),
        facts={
            "did_pt": estimate.point_pt,
            "standard_error_pt": estimate.standard_error_pt,
            "attributable_pt": estimate.attributable_pt,
            "confidence": estimate.confidence,
            "pretest_p": pretest_p,
            "pretest_slope": slope,
            "pretest_weeks": weeks,
            "pretest_passed": pretest_passed,
            "matched_pairs": estimate.matched_pairs,
            "dropped": matching.dropped,
        },
    )


# ===========================================================================
# Test 6 — confounder screen. CAP.
# ===========================================================================


@dataclass(frozen=True)
class ConfounderResult:
    """One confounder, and how it was resolved or why it was not."""

    name: str
    resolution: str
    p_value: float | None
    detail: str

    @property
    def resolved(self) -> bool:
        return self.resolution in {"balanced", "by_construction", "by_matching"}


def test_confounders(
    results: list[ConfounderResult], spec: ConfounderScreenSpec
) -> Finding:
    """Does anything else in the graph produce the same signature?

    A confounder balanced across the treated and control groups is
    RESOLVED: the design removed it. One that differs is unresolved and
    caps confidence — it does not eliminate, because a confounded
    hypothesis may still be true.
    """
    unresolved = [item for item in results if not item.resolved]
    if not results:
        return Finding(
            spec.test_id, "Confounder screen", passed=True, testable=False,
            statistic=None, p_value=None,
            detail="the causal graph declares no confounders for this hypothesis.",
        )
    return Finding(
        spec.test_id, "Confounder screen", passed=not unresolved, testable=True,
        statistic=float(len(unresolved)), p_value=None,
        detail=(
            f"{len(results) - len(unresolved)} of {len(results)} confounders resolved"
            + (
                ". None left unresolved."
                if not unresolved
                else ": " + "; ".join(f"{item.name} — {item.detail}" for item in unresolved)
            )
        ),
        facts={
            "resolved": tuple(item.name for item in results if item.resolved),
            "unresolved": tuple(item.name for item in unresolved),
            "results": tuple(
                {"name": item.name, "resolution": item.resolution, "p_value": item.p_value}
                for item in results
            ),
        },
    )


def balance(
    measure: pd.Series, matching: Matching, spec: ConfounderScreenSpec, name: str, label: str
) -> ConfounderResult:
    """Welch's t on one confounder across the treated and control groups."""
    treated = measure.reindex(list(matching.treated)).dropna().to_numpy(float)
    control = measure.reindex(list(matching.controls)).dropna().to_numpy(float)
    if min(len(treated), len(control)) < spec.min_group_size:
        return ConfounderResult(
            name, "unmeasured", None,
            f"{label}: too few stores carry a value to compare the two groups.",
        )
    statistic, p_value = stats.ttest_ind(treated, control, equal_var=False)
    balanced = bool(p_value > spec.min_balance_p)
    return ConfounderResult(
        name,
        "balanced" if balanced else "differs",
        float(p_value),
        (
            f"{label}: treated {treated.mean():.2f} against control {control.mean():.2f}, "
            f"Welch p = {p_value:.2f} — "
            + ("no difference the design has not removed." if balanced
               else "the two groups differ on this, so it is not controlled for.")
        ),
    )


__all__ = [
    "NOT_TESTABLE",
    "ConfounderResult",
    "DidEstimate",
    "Exposure",
    "Finding",
    "Matching",
    "Pair",
    "balance",
    "changepoint",
    "estimate_did",
    "exposed_stores",
    "fit_elasticity",
    "match_controls",
    "parallel_trends",
    "signed",
    "test_confounders",
    "test_dose_response",
    "test_did",
    "test_precedence",
    "test_specificity",
    "test_sufficiency",
]
