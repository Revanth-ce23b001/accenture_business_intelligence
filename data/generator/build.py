"""Assemble the synthetic world and write it to CSV.

Order matters and is deliberate:

  1. calendar          real, dated festivals; Diwali moves 11 days
  2. stores / SKUs     entity master
  3. festival fit      solve each region's festival amplitude so the
                       EXPECTED month-on-month change matches the registry
  4. base series       demand before anything went wrong
  5. level calibration anchor West on Oct 2025 = ₹83.70 Cr
  6. beta fit          solve the availability elasticity so the mechanism
                       delivers the registry's treatment effect
  7. mechanism         availability -> units -> revenue (the causal core)
  8. region shocks     ordinary variation elsewhere; #2472's South miss
  9. emit              pos_erp, store_ops, context

Steps 3 and 6 are parameter fits, not value edits. They solve for how
hard a region trades on a festival, and for how much a shopper's basket
walks when the shelf is empty. Revenue is never assigned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from data.generator.model import (
    REGIONS,
    BaseSeries,
    Mechanism,
    Streams,
    apply_availability_mechanism,
    apply_region_shocks,
    apply_store_idiosyncratic_noise,
    assign_footfall_counters,
    build_base_series,
    build_calendar,
    build_skus,
    build_stores,
    calibrate_levels,
    festival_factor,
    load_all_configs,
)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CRORE = 1e7


@dataclass
class World:
    entity: dict[str, Any]
    scenarios: dict[str, Any]
    #: The seeded calibration ledger's layout. Declared counts only; the
    #: map fitted to them is measured, never written down.
    calibration: dict[str, Any]
    streams: Streams
    calendar: pd.DataFrame
    stores: pd.DataFrame
    skus: pd.DataFrame
    expected: np.ndarray      # (n_stores, n_days) calendar-expected revenue
    actual: np.ndarray        # (n_stores, n_days) after mechanism and shocks
    mechanism: Mechanism
    festival_amplitudes: dict[str, float]
    fitted_beta: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    amplitude_achieved: dict[str, float] = field(default_factory=dict)
    fitted_dose_sigma: float = 0.0

    #: RECONCILIATION 2 — store_id -> the NEW POS store_code issued at the
    #: re-fascia. These codes exist in the POS facts and in no xref row,
    #: which is what makes them unjoinable and therefore quarantined.
    #: Deliberately not written to dim_store: the warehouse has to
    #: DISCOVER the gap by failing to join, not be told about it.
    refascia_codes: dict[str, str] = field(default_factory=dict)
    refascia_cutover: str = ""

    #: RECONCILIATION 1 — B2B net revenue as a share of retail net revenue,
    #: solved so the two rival definitions differ by the configured gap.
    fitted_b2b_share: float = 0.0

    # -- helpers ----------------------------------------------------------

    def region_month(self, matrix: np.ndarray, region: str, period: str) -> float:
        rows = self.stores["region"].to_numpy() == region
        cols = self.calendar["period_month"].to_numpy() == period
        return float(matrix[np.ix_(rows, cols)].sum())

    def region_month_cr(self, matrix: np.ndarray, region: str, period: str) -> float:
        return self.region_month(matrix, region, period) / CRORE


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def _mechanism_reduction(
    beta: float,
    expected: np.ndarray,
    world_stores: pd.DataFrame,
    calendar: pd.DataFrame,
    scenario: dict[str, Any],
    streams: Streams,
    region: str,
    period: str,
) -> float:
    """Fractional reduction the mechanism causes in `region` during `period`."""
    scenario = json.loads(json.dumps(scenario))  # cheap deep copy
    scenario["mechanism"]["availability_elasticity_beta"] = float(beta)
    adjusted, _ = apply_availability_mechanism(
        expected, world_stores, calendar, scenario, streams
    )
    rows = world_stores["region"].to_numpy() == region
    cols = calendar["period_month"].to_numpy() == period
    before = expected[np.ix_(rows, cols)].sum()
    after = adjusted[np.ix_(rows, cols)].sum()
    return 1.0 - after / before


def _month_on_month(
    entity: dict[str, Any],
    stores: pd.DataFrame,
    calendar: pd.DataFrame,
    streams: Streams,
    amplitudes: dict[str, float],
    region: str,
    base_month: str,
    month: str,
) -> float:
    """Expected month-on-month change, measured off the REAL pipeline."""
    base = build_base_series(entity, stores, calendar, streams, amplitudes)
    expected = calibrate_levels(base, entity)
    prev = _region_month_cr(expected, stores, calendar, region, base_month)
    curr = _region_month_cr(expected, stores, calendar, region, month)
    return (curr / prev - 1.0) * 100.0


def fit_amplitudes(
    entity: dict[str, Any],
    stores: pd.DataFrame,
    calendar: pd.DataFrame,
    streams: Streams,
    targets: dict[str, tuple[str, str, float]],
    default: float = 0.55,
    grid: int = 28,
    lo: float = 0.02,
    hi: float = 3.5,
) -> tuple[dict[str, float], dict[str, float]]:
    """Solve each region's festival amplitude against the real base series.

    An earlier version solved against a simplified proxy that omitted the
    noise terms, so the fitted amplitude did not reproduce the target on
    the series actually built. This measures what it is fitting.

    Returns (amplitudes, achieved) so the caller can report any target the
    real festival calendar cannot reach, rather than silently clamping.
    """
    amplitudes = {region: default for region in REGIONS}
    achieved: dict[str, float] = {}

    for region, (base_month, month, want) in targets.items():
        def measure(amp: float) -> float:
            trial = dict(amplitudes)
            trial[region] = amp
            return _month_on_month(
                entity, stores, calendar, streams, trial, region, base_month, month
            )

        candidates = np.linspace(lo, hi, grid)
        values = [measure(float(a)) for a in candidates]

        bracket = None
        for i in range(len(candidates) - 1):
            if (values[i] - want) * (values[i + 1] - want) <= 0:
                bracket = (float(candidates[i]), float(candidates[i + 1]))
                break

        if bracket is None:
            # Unreachable from the real festival dates. Take the closest
            # and let the caller report the gap.
            best = int(np.argmin([abs(v - want) for v in values]))
            amplitudes[region] = float(candidates[best])
            achieved[region] = float(values[best])
            continue

        low, high = bracket
        root = brentq(lambda a: measure(a) - want, low, high, xtol=1e-10, rtol=1e-12)
        amplitudes[region] = float(root)
        achieved[region] = measure(float(root))

    return amplitudes, achieved



def _dose_response_r(
    actual: np.ndarray,
    expected: np.ndarray,
    stores: pd.DataFrame,
    calendar: pd.DataFrame,
    mechanism: Mechanism,
) -> float:
    """OLS correlation of store-level decline on store-level availability gap.

    Positive by construction: a bigger coverage gap goes with a bigger
    decline. Measured across all West stores, treated and untreated.
    """
    from scipy import stats as _stats

    period = calendar["period_month"].to_numpy()
    oct_m, nov_m = period == "2025-10", period == "2025-11"
    west = stores["region"].to_numpy() == "West"

    pre = actual[:, oct_m].sum(axis=1)
    post = actual[:, nov_m].sum(axis=1)
    decline_pct = (pre - post) / pre * 100.0

    baseline = mechanism.true_availability[:, oct_m].mean(axis=1)
    gap = baseline - mechanism.true_availability[:, nov_m].mean(axis=1)

    r, _ = _stats.pearsonr(gap[west], decline_pct[west])
    return float(r)


def fit_beta(
    expected: np.ndarray,
    stores: pd.DataFrame,
    calendar: pd.DataFrame,
    scenario: dict[str, Any],
    streams: Streams,
    target_reduction: float,
    lo: float = 0.05,
    hi: float = 6.0,
    tolerance: float = 1e-6,
    max_iter: int = 80,
) -> float:
    """Solve the availability elasticity for the target treatment effect.

    beta is a MECHANISM parameter — how much of a basket walks when the
    sought SKU is missing. Fitting it is not the same as assigning
    revenue: every rupee still moves through availability and units.
    """
    low, high = lo, hi
    mid = (low + high) / 2.0
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        got = _mechanism_reduction(
            mid, expected, stores, calendar, scenario, streams, "West", "2025-11"
        )
        if abs(got - target_reduction) < tolerance:
            break
        if got < target_reduction:
            low = mid
        else:
            high = mid
    return mid


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_world() -> World:
    configs = load_all_configs()
    entity = configs["entity"]
    scenarios = configs["scenarios"]
    calibration = configs["calibration"]
    streams = Streams(entity["seed"])

    end = date.fromisoformat(entity["timeline"]["end_date"])
    # 18 months ENDING 30 Nov 2025 means 1 Jun 2024 .. 30 Nov 2025 = 548 days.
    # Subtracting 18 months from the 30th and adding a day lands on 31 May and
    # yields 549 -- anchor on the first of the month instead.
    months = int(entity["timeline"]["months"])
    start = (pd.Timestamp(end).replace(day=1) - pd.DateOffset(months=months - 1)).date()
    fiscal = entity["reconciliation"]["calendar_mismatch"]
    calendar = build_calendar(
        start,
        end,
        fiscal_year_start=(
            int(fiscal["fiscal_year_start_month"]),
            int(fiscal["fiscal_year_start_day"]),
        ),
        promotions=entity["promotions"]["windows"],
    )

    stores = build_stores(entity, streams)
    stores = assign_footfall_counters(stores, entity, streams)
    skus = build_skus(entity, streams)

    s2451 = scenarios["2451"]
    s2472 = scenarios["2472"]

    # --- 3. festival amplitudes ---------------------------------------
    # West: October 2025 -> November 2025 must be EXPECTED at -3.2 pt.
    # South: August 2025 -> September 2025 must be EXPECTED at +12.0 pt.
    amplitudes, amplitude_achieved = fit_amplitudes(
        entity,
        stores,
        calendar,
        streams,
        targets={
            "West": ("2025-10", "2025-11", s2451["targets"]["calendar_attributed_pt"]),
            "South": (
                "2025-08",
                "2025-09",
                s2472["targets"]["calendar_expected_movement_pct"],
            ),
        },
    )

    # --- 4/5. base series and levels ----------------------------------
    base = build_base_series(entity, stores, calendar, streams, amplitudes)
    expected = calibrate_levels(base, entity)

    # --- 6/7. mechanism ------------------------------------------------
    targets = s2451["targets"]
    measured = s2451["measured_targets"]

    oct_cr = _region_month_cr(expected, stores, calendar, "West", "2025-10")
    nov_expected_cr = _region_month_cr(expected, stores, calendar, "West", "2025-11")

    # The mechanism must remove `did_pt` percentage points OF THE OCTOBER
    # LEVEL from November. Expressed as a fraction of November's expected
    # revenue, that is:
    did_pt = abs(float(measured["matched_control_did_pt"]))
    target_reduction = (did_pt / 100.0) * oct_cr / nov_expected_cr

    fitted_beta = fit_beta(
        expected, stores, calendar, s2451, streams, target_reduction
    )
    s2451_fitted = json.loads(json.dumps(s2451))
    s2451_fitted["mechanism"]["availability_elasticity_beta"] = fitted_beta

    actual_raw, mechanism = apply_availability_mechanism(
        expected, stores, calendar, s2451_fitted, streams
    )

    # --- 7b. idiosyncratic store variation -----------------------------
    # Fitted so the dose-response correlation matches the registry.
    # Applied to treated and control alike; it dilutes the measured r,
    # it cannot create the effect.
    dose_target = float(measured["dose_response_r"])

    def dose_r(sigma: float) -> float:
        trial = apply_store_idiosyncratic_noise(
            actual_raw, stores, calendar, "2025-11", sigma, streams
        )
        return _dose_response_r(trial, expected, stores, calendar, mechanism)

    lo_s, hi_s = 0.0, 0.60
    for _ in range(50):
        mid_s = (lo_s + hi_s) / 2.0
        if dose_r(mid_s) > dose_target:
            lo_s = mid_s
        else:
            hi_s = mid_s
    fitted_sigma = (lo_s + hi_s) / 2.0
    actual = apply_store_idiosyncratic_noise(
        actual_raw, stores, calendar, "2025-11", fitted_sigma, streams
    )

    # --- 8. region shocks ----------------------------------------------
    # Measure what the mechanism actually delivered, then close each
    # region's November residual onto its target with the ordinary
    # variation term. West's correction is small precisely BECAUSE the
    # mechanism already produced most of its residual causally; the
    # remainder is the portion #2451 cannot attribute.
    strip = targets["regional_residual_strip_pt"]
    shocks: dict[tuple[str, str], float] = {}

    def residual_pt(matrix: np.ndarray, region: str, period: str, prev: str) -> float:
        a = _region_month_cr(matrix, stores, calendar, region, period)
        e = _region_month_cr(expected, stores, calendar, region, period)
        p = _region_month_cr(matrix, stores, calendar, region, prev)
        return (a - e) / p * 100.0

    for region in REGIONS:
        current = residual_pt(actual, region, "2025-11", "2025-10")
        delta_pt = float(strip[region]) - current
        oct_cr_r = _region_month_cr(actual, stores, calendar, region, "2025-10")
        # Divide by ACTUAL November, not expected: the shock is applied to
        # the post-mechanism series, so that is what the correction scales.
        # Using expected here under-delivered on West, the one region where
        # the two differ materially.
        nov_act_r = _region_month_cr(actual, stores, calendar, region, "2025-11")
        shocks[(region, "2025-11")] = (delta_pt / 100.0) * oct_cr_r / nov_act_r

    # #2472: South is flat in September against a calendar expecting +12.
    sep_expected = float(s2472["targets"]["calendar_expected_movement_pct"])
    sep_actual = float(s2472["targets"]["headline_movement_pct"])
    shocks[("South", "2025-09")] = (1.0 + sep_actual / 100.0) / (
        1.0 + sep_expected / 100.0
    ) - 1.0

    actual = apply_region_shocks(actual, stores, calendar, shocks)

    # --- 8b. anchor levels on the registry ------------------------------
    # calibrate_levels anchored `expected`; the mechanism and the shocks
    # then moved `actual` off that anchor by a fraction of a percent.
    # Rescale each region by a CONSTANT so West October lands exactly on
    # the registry. A constant scale changes no ratio, no residual and no
    # percentage point -- only the level.
    region_arr = stores["region"].to_numpy()
    period_arr = calendar["period_month"].to_numpy()
    west_oct = actual[np.ix_(region_arr == "West", period_arr == "2025-10")].sum()
    west_scale = entity["anchors"]["west_net_revenue_oct_2025_inr_cr"] * CRORE / west_oct
    actual[region_arr == "West", :] *= west_scale
    expected[region_arr == "West", :] *= west_scale

    months = pd.unique(period_arr)
    east_rows = region_arr == "East"
    east_monthly = np.array([actual[np.ix_(east_rows, period_arr == m)].sum() for m in months])
    east_scale = (
        entity["anchors"]["east_net_revenue_monthly_inr_cr"] * CRORE / east_monthly.mean()
    )
    actual[east_rows, :] *= east_scale
    expected[east_rows, :] *= east_scale

    world = World(
        entity=entity,
        scenarios=scenarios,
        calibration=calibration,
        streams=streams,
        calendar=calendar,
        stores=stores,
        skus=skus,
        expected=expected,
        actual=actual,
        mechanism=mechanism,
        festival_amplitudes=amplitudes,
        fitted_beta=fitted_beta,
    )
    world.fitted_dose_sigma = fitted_sigma
    world.diagnostics = {}
    world.amplitude_achieved = amplitude_achieved

    # --- 9. reconciliation parameters ----------------------------------
    beta, _rho = solve_b2b_share(entity)
    world.fitted_b2b_share = beta

    refascia_ids, cutover, _achieved = select_refascia_stores(actual, stores, calendar, entity)
    world.refascia_cutover = cutover
    world.refascia_codes = {
        sid: f"RC-{i + 1:03d}-{sid[1:]}" for i, sid in enumerate(refascia_ids)
    }

    world.diagnostics = _diagnose(world)
    return world


# ---------------------------------------------------------------------------
# Reconciliation solves
#
# Neither of these assigns a value. Each picks the one free parameter that
# lands a declared target, and the achieved figure is measured back out in
# `_diagnose` so a reader can see the difference between the two.
# ---------------------------------------------------------------------------


def solve_b2b_share(entity: dict[str, Any]) -> tuple[float, float]:
    """Solve the B2B share of net revenue that lands the definition gap.

    The two rival figures for the same period are

        pos_ledger = net + b2b        (POS does not apply the contract's
                                       channel exclusion)
        marketing  = net + returns    (marketing never deducts a return)

    so, writing beta = b2b/net and rho = returns/net,

        gap = (beta - rho) / (1 + rho)   ==>   beta = gap * (1 + rho) + rho

    `rho` is not free: it falls out of the netting already configured.
    A bill's return credits the discounted, pre-tax value, so

        net     = gross * (1 - discount) * (1 - tax - return_rate)
        returns = gross * (1 - discount) * return_rate
        rho     = return_rate / (1 - tax - return_rate)

    Returns (beta, rho).
    """
    netting = entity["netting"]
    tax = float(netting["tax_rate"])
    return_rate = float(netting["return_rate"])
    denominator = 1.0 - tax - return_rate
    if denominator <= 0.0:
        raise ValueError(
            f"netting leaves no net revenue: tax {tax} + return_rate {return_rate} >= 1"
        )
    rho = return_rate / denominator

    gap = float(entity["reconciliation"]["definition_conflict"]["target_gap_pct"]) / 100.0
    beta = gap * (1.0 + rho) + rho
    return beta, rho


def select_refascia_stores(
    actual: np.ndarray,
    stores: pd.DataFrame,
    calendar: pd.DataFrame,
    entity: dict[str, Any],
) -> tuple[list[str], str, float]:
    """Pick WHICH stores are re-fasciad so the quarantine lands on target.

    The COUNT is given by the business (three stores were re-fasciad); the
    revenue share that ends up unjoinable is a consequence of which three,
    and that is what is solved for here. Nothing is scaled and no revenue
    is moved — the search only chooses stores.

    Returns (store_ids, cutover_date, achieved_share).
    """
    defects = entity["defects"]
    n = int(defects["store_code_changes"])
    cutover = str(defects["store_code_change_window"][0])
    target = float(
        entity["reconciliation"]["entity_key_mismatch"]["target_quarantine_revenue_share"]
    )

    after = calendar["date"].to_numpy() >= np.datetime64(cutover)
    per_store = actual[:, after].sum(axis=1)
    share = per_store / actual.sum()

    order = np.argsort(share, kind="stable")          # ascending, ties by index
    sorted_share = share[order]
    cumulative_best: tuple[float, tuple[int, ...]] | None = None

    # Every unordered pair, then the best third by binary search. O(n^2 log n)
    # over 412 stores is a few hundred thousand operations.
    for a in range(len(order) - n + 1):
        for b in range(a + 1, len(order)):
            wanted = target - sorted_share[a] - sorted_share[b]
            c = int(np.searchsorted(sorted_share, wanted))
            for candidate in (c - 1, c, c + 1):
                if candidate <= b or candidate >= len(order):
                    continue
                total = sorted_share[a] + sorted_share[b] + sorted_share[candidate]
                key = (abs(total - target), (a, b, candidate))
                if cumulative_best is None or key < cumulative_best:
                    cumulative_best = key
    if cumulative_best is None:  # pragma: no cover - needs fewer than 3 stores
        raise ValueError("cannot choose re-fascia stores: fewer stores than the target count")

    picked = [int(order[i]) for i in cumulative_best[1]]
    achieved = float(share[picked].sum())
    store_ids = sorted(stores["store_id"].to_numpy()[picked].tolist())
    return store_ids, cutover, achieved


def _region_month_cr(
    matrix: np.ndarray, stores: pd.DataFrame, calendar: pd.DataFrame, region: str, period: str
) -> float:
    rows = stores["region"].to_numpy() == region
    cols = calendar["period_month"].to_numpy() == period
    return float(matrix[np.ix_(rows, cols)].sum()) / CRORE


def _diagnose(world: World) -> dict[str, Any]:
    """Measure what actually emerged. Nothing here is set; all is read back."""
    out: dict[str, Any] = {}

    oct_cr = world.region_month_cr(world.actual, "West", "2025-10")
    nov_cr = world.region_month_cr(world.actual, "West", "2025-11")
    nov_exp_cr = world.region_month_cr(world.expected, "West", "2025-11")

    out["west_oct_cr"] = oct_cr
    out["west_nov_cr"] = nov_cr
    out["west_nov_expected_cr"] = nov_exp_cr
    out["headline_pct"] = (nov_cr / oct_cr - 1.0) * 100.0
    out["absolute_decline_cr"] = oct_cr - nov_cr
    out["calendar_pt"] = (nov_exp_cr / oct_cr - 1.0) * 100.0
    out["residual_pt"] = out["headline_pct"] - out["calendar_pt"]
    out["calendar_cr"] = -out["calendar_pt"] / 100.0 * oct_cr
    out["residual_cr"] = -out["residual_pt"] / 100.0 * oct_cr
    out["reconciliation_cr"] = out["calendar_cr"] + out["residual_cr"]

    for region in REGIONS:
        a = world.region_month_cr(world.actual, region, "2025-11")
        e = world.region_month_cr(world.expected, region, "2025-11")
        prev = world.region_month_cr(world.actual, region, "2025-10")
        out[f"residual_pt_{region}"] = ((a - e) / prev) * 100.0

    out["fitted_beta"] = world.fitted_beta
    out["festival_amplitudes"] = world.festival_amplitudes
    out["treated_count"] = int(world.mechanism.treated_mask.sum())
    out["control_count"] = int(world.mechanism.control_mask.sum())

    rows = world.stores["region"].to_numpy() == "West"
    treated_rev = world.expected[world.mechanism.treated_mask, :][
        :, world.calendar["period_month"].to_numpy() == "2025-10"
    ].sum()
    west_rev = world.expected[rows, :][
        :, world.calendar["period_month"].to_numpy() == "2025-10"
    ].sum()
    out["treated_revenue_share"] = float(treated_rev / west_rev)

    # Availability on the affected SKUs, treated stores, 12:00 mean.
    nov = world.calendar["period_month"].to_numpy() == "2025-11"
    oct_ = world.calendar["period_month"].to_numpy() == "2025-10"
    avail = world.mechanism.availability
    out["avail_treated_before_pct"] = float(
        avail[world.mechanism.treated_mask, :][:, oct_].mean() * 100.0
    )
    settled = nov & (
        world.calendar["date"].to_numpy() >= np.datetime64("2025-11-12")
    )
    out["avail_treated_after_pct"] = float(
        avail[world.mechanism.treated_mask, :][:, settled].mean() * 100.0
    )

    # --- reconciliation, measured -------------------------------------
    _, rho = solve_b2b_share(world.entity)
    beta = world.fitted_b2b_share
    out["returns_share_of_net"] = float(rho)
    out["b2b_share_of_net"] = float(beta)
    out["definition_gap_pct"] = float((beta - rho) / (1.0 + rho) * 100.0)

    after = world.calendar["date"].to_numpy() >= np.datetime64(world.refascia_cutover)
    refascia_rows = world.stores["store_id"].isin(world.refascia_codes).to_numpy()
    out["refascia_stores"] = sorted(world.refascia_codes)
    out["refascia_cutover"] = world.refascia_cutover
    out["quarantine_revenue_share"] = float(
        world.actual[np.ix_(refascia_rows, after)].sum() / world.actual.sum()
    )
    return out




# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def write_outputs(world: World, out: Path | None = None) -> dict[str, Any]:
    """Write every table and a manifest describing how the world was built.

    The manifest records the fitted parameters and the measured emergent
    statistics, so a reader can see which numbers were solved for and
    which were read back off the data.
    """
    from data.generator import sources, sources_calibration, sources_ops

    destination = Path(out) if out else RAW_DIR
    destination.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for emit in (
        sources.emit_dimensions,
        sources.emit_store_xref,
        sources.emit_users,
        sources.emit_sales_daily,
        sources.emit_sales_daily_sku,
        sources.emit_bill_lines,
        sources.emit_feed_status,
        sources.emit_restatements,
        sources_ops.emit_inventory_snapshot,
        sources_ops.emit_store_notes,
        sources_ops.emit_tickets,
        sources_ops.emit_reviews,
        sources_ops.emit_footfall,
        sources_ops.emit_calendar,
        sources_ops.emit_festival_windows,
        sources_ops.emit_marketing_spend,
        sources_ops.emit_competitor_news,
        sources_ops.emit_weather,
        sources_ops.emit_qcomm_weekly,
        sources_calibration.emit_calibration_ledger,
    ):
        counts.update(emit(world, destination))

    manifest = {
        "seed": world.entity["seed"],
        "fitted": {
            "festival_amplitudes": world.festival_amplitudes,
            "availability_elasticity_beta": world.fitted_beta,
            "idiosyncratic_sigma": world.fitted_dose_sigma,
            "b2b_share_of_net_revenue": world.fitted_b2b_share,
            "refascia_store_ids": sorted(world.refascia_codes),
        },
        "measured": world.diagnostics,
        "row_counts": counts,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    world = build_world()
    manifest = write_outputs(world)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":  # pragma: no cover
    main()
