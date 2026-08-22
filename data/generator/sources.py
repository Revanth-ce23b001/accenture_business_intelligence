"""Write the synthetic world out as three source systems.

  pos_erp     bill lines, daily sales, SKU-level sales, feed status
  store_ops   inventory snapshots, store notes, support tickets
  context     marketing spend, calendar, competitor news, weather

DELIBERATE DEFECTS live here, and each one exists so a downstream stage
has something real to catch:

  * returns are restated up to 3 days after the original bill, so a
    same-day join understates them
  * B2B bulk orders sit in the same table as retail sales
  * three stores change store_code mid-period, so a naive group-by
    splits them in two
  * the ticket `category` field is unreliable — the body is the truth
  * store notes carry no tags, and are written in English and Hinglish
  * footfall is null for 109 of 140 West stores

DETERMINISM: every frame is sorted on a stable key and every float is
written at fixed precision, so re-running produces byte-identical files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CRORE = 1e7


def write_csv(frame: pd.DataFrame, path: Path) -> int:
    """Deterministic CSV write: LF endings, no index, fixed float format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.4f")
    return len(frame)


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


def emit_dimensions(world, out: Path) -> dict[str, int]:
    stores = world.stores.copy()
    stores["is_treated_2451"] = world.mechanism.treated_mask
    stores["is_matched_control_2451"] = world.mechanism.control_mask
    stores = stores.sort_values("store_id").reset_index(drop=True)

    skus = world.skus.copy().sort_values("sku_id").reset_index(drop=True)

    return {
        "dim/dim_store.csv": write_csv(stores, out / "dim" / "dim_store.csv"),
        "dim/dim_sku.csv": write_csv(skus, out / "dim" / "dim_sku.csv"),
    }


# ---------------------------------------------------------------------------
# pos_erp
# ---------------------------------------------------------------------------


def _net_asp(world) -> float:
    """Average net price per unit across the catalogue."""
    skus = world.skus
    netting = world.entity["netting"]
    gross = float((skus["list_price_inr"] * skus["demand_share"]).sum())
    return gross * (1.0 - netting["discount_rate_mean"]) * (1.0 - netting["tax_rate"])


def emit_sales_daily(world, out: Path) -> dict[str, int]:
    """Authoritative store x day net revenue. Every statistic reads this."""
    stores = world.stores
    calendar = world.calendar
    n_stores, n_days = world.actual.shape

    store_ids = np.repeat(stores["store_id"].to_numpy(), n_days)
    regions = np.repeat(stores["region"].to_numpy(), n_days)
    dates = np.tile(calendar["date"].dt.strftime("%Y-%m-%d").to_numpy(), n_stores)
    net = world.actual.reshape(-1)
    expected = world.expected.reshape(-1)

    asp = _net_asp(world)
    units = np.round(net / asp).astype(np.int64)

    frame = pd.DataFrame(
        {
            "store_id": store_ids,
            "region": regions,
            "txn_date": dates,
            "net_revenue_inr": np.round(net, 2),
            "units": units,
            "calendar_expected_inr": np.round(expected, 2),
        }
    ).sort_values(["store_id", "txn_date"], kind="stable").reset_index(drop=True)

    return {"pos_erp/sales_daily.csv": write_csv(frame, out / "pos_erp" / "sales_daily.csv")}


def emit_sales_daily_sku(world, out: Path) -> dict[str, int]:
    """Store x day x SKU for the top-20, over a bounded recent window.

    The split follows availability: in a treated store during the fault,
    the affected SKUs lose share because they were not on the shelf. That
    is what makes the SKU table agree with the inventory snapshots.
    """
    entity = world.entity
    window = int(entity["emission"]["sku_daily_window_days"])
    top_n = int(entity["catalogue"]["top_n"])

    calendar = world.calendar
    day_mask = np.zeros(len(calendar), dtype=bool)
    day_mask[-window:] = True
    day_idx = np.where(day_mask)[0]

    skus = world.skus.iloc[:top_n]
    share = skus["demand_share"].to_numpy()
    share = share / share.sum()

    avail = world.mechanism.true_availability[:, day_idx]      # (stores, days)
    n_stores = len(world.stores)

    rows = []
    store_ids = world.stores["store_id"].to_numpy()
    dates = calendar["date"].dt.strftime("%Y-%m-%d").to_numpy()[day_idx]
    net = world.actual[:, day_idx]
    prices = skus["list_price_inr"].to_numpy()

    # Availability suppresses the affected SKUs' share. Renormalise so the
    # store-day total still reconciles to sales_daily -- the SKU table must
    # sum back to the authoritative daily table, or the two disagree.
    for si in range(n_stores):
        weights = np.outer(np.ones(len(day_idx)), share)          # (days, skus)
        suppress = avail[si, :][:, None] / avail[si, :].max()
        weights = weights * (0.35 + 0.65 * suppress)
        weights = weights / weights.sum(axis=1, keepdims=True)
        revenue = weights * net[si, :][:, None]
        rows.append(revenue)

    revenue_stack = np.stack(rows)  # (stores, days, skus)
    n_days, n_sku = len(day_idx), top_n

    frame = pd.DataFrame(
        {
            "store_id": np.repeat(store_ids, n_days * n_sku),
            "txn_date": np.tile(np.repeat(dates, n_sku), n_stores),
            "sku_id": np.tile(skus["sku_id"].to_numpy(), n_stores * n_days),
            "net_revenue_inr": np.round(revenue_stack.reshape(-1), 2),
            "units": np.round(
                revenue_stack.reshape(-1) / np.tile(prices, n_stores * n_days) * 1.0
            ).astype(np.int64),
        }
    )
    frame = frame.sort_values(["store_id", "txn_date", "sku_id"], kind="stable").reset_index(
        drop=True
    )
    return {
        "pos_erp/sales_daily_sku.csv": write_csv(
            frame, out / "pos_erp" / "sales_daily_sku.csv"
        )
    }


def emit_bill_lines(world, out: Path) -> dict[str, int]:
    """Bill-line grain, deterministically sampled, carrying the defects.

    Full grain for 412 stores over 18 months is ~25M lines and ~2 GB. The
    authoritative revenue tables are sales_daily and sales_daily_sku; this
    table exists so the line-level defects are real and inspectable, and
    it records its own sample rate so nothing sums it by mistake.
    """
    entity = world.entity
    emission = entity["emission"]
    netting = entity["netting"]
    defects = entity["defects"]
    rng = world.streams.fresh("bill_lines")

    window = int(emission["bill_lines_window_days"])
    rate = float(emission["bill_lines_sample_rate"])

    calendar = world.calendar
    day_idx = np.arange(len(calendar))[-window:]
    dates = calendar["date"].to_numpy()[day_idx]

    stores = world.stores
    skus = world.skus
    asp = _net_asp(world)
    units_per_bill = float(entity["catalogue"]["units_per_bill_mean"])
    bill_value = asp * units_per_bill

    net = world.actual[:, day_idx]
    expected_bills = net / bill_value
    sampled_bills = rng.poisson(np.clip(expected_bills * rate, 0, None))

    sku_ids = skus["sku_id"].to_numpy()
    sku_p = (skus["demand_share"] / skus["demand_share"].sum()).to_numpy()
    prices = dict(zip(skus["sku_id"], skus["list_price_inr"], strict=True))

    # Three stores change store_code mid-period.
    changed = stores["store_id"].to_numpy()[
        rng.choice(len(stores), size=int(defects["store_code_changes"]), replace=False)
    ]
    change_from = np.datetime64(defects["store_code_change_window"][0])
    new_codes = {
        sid: f"RC-{i + 1:03d}-{sid[1:]}" for i, sid in enumerate(sorted(changed))
    }

    records: list[tuple] = []
    open_h = int(entity["trade"]["open_hour_ist"])
    close_h = int(entity["trade"]["close_hour_ist"])
    store_codes = dict(zip(stores["store_id"], stores["store_code"], strict=True))
    store_ids = stores["store_id"].to_numpy()

    bill_seq = 0
    for si, sid in enumerate(store_ids):
        counts = sampled_bills[si]
        for di, count in enumerate(counts):
            if count == 0:
                continue
            day = dates[di]
            n_lines = rng.poisson(units_per_bill, size=int(count)) + 1
            for b in range(int(count)):
                bill_seq += 1
                bill_id = f"B{bill_seq:09d}"
                hour = int(rng.integers(open_h, close_h))
                minute = int(rng.integers(0, 60))
                ts = f"{np.datetime_as_string(day, unit='D')} {hour:02d}:{minute:02d}:00"

                code = store_codes[sid]
                if sid in new_codes and day >= change_from:
                    code = new_codes[sid]

                is_b2b = rng.random() < defects["b2b_share_of_gross"]
                is_transfer = (not is_b2b) and rng.random() < defects["transfer_share_of_rows"]
                channel = "B2B" if is_b2b else "RETAIL"
                txn_type = "TRANSFER" if is_transfer else "SALE"

                for line_no in range(int(n_lines[b])):
                    sku = str(rng.choice(sku_ids, p=sku_p))
                    qty = int(rng.integers(1, 4)) * (14 if is_b2b else 1)
                    gross = prices[sku] * qty
                    discount = gross * float(
                        np.clip(
                            rng.normal(
                                netting["discount_rate_mean"], netting["discount_rate_sigma"]
                            ),
                            0.0,
                            0.5,
                        )
                    )
                    tax = (gross - discount) * netting["tax_rate"]

                    # DEFECT: returns are restated up to 3 days later, so
                    # the return row carries a LATER posting date than the
                    # bill it belongs to.
                    returned = rng.random() < netting["return_rate"]
                    if returned:
                        lag = int(rng.integers(0, netting["return_lag_days_max"] + 1))
                        posted = np.datetime_as_string(
                            day + np.timedelta64(lag, "D"), unit="D"
                        )
                        return_amount = round(float(gross - discount), 2)
                    else:
                        posted = np.datetime_as_string(day, unit="D")
                        return_amount = 0.0

                    records.append(
                        (
                            bill_id,
                            line_no + 1,
                            sid,
                            code,
                            sku,
                            ts,
                            posted,
                            channel,
                            txn_type,
                            qty,
                            round(float(gross), 2),
                            round(float(discount), 2),
                            round(float(tax), 2),
                            return_amount,
                        )
                    )

    frame = pd.DataFrame(
        records,
        columns=[
            "bill_id",
            "line_no",
            "store_id",
            "store_code",
            "sku_id",
            "txn_ts",
            "posted_date",
            "channel",
            "txn_type",
            "units",
            "gross_amount",
            "discount_amount",
            "tax_amount",
            "returns_amount",
        ],
    )
    frame["sample_rate"] = rate
    frame = frame.sort_values(["bill_id", "line_no"], kind="stable").reset_index(drop=True)
    return {"pos_erp/bill_lines.csv": write_csv(frame, out / "pos_erp" / "bill_lines.csv")}


def emit_feed_status(world, out: Path) -> dict[str, int]:
    """#2470: 25 West store feeds fail to load on 12 Nov 2025.

    The FEED is broken; the sales are not. sales_daily carries the true
    figure, so #2451's monthly reconciliation is unaffected and the day is
    recoverable on reload. That is the only reading under which
    CLAUDE.md's "non-overlapping scenarios" and "a daily check inside
    #2451's month" can both hold.
    """
    scenario = world.scenarios["2470"]
    rng = world.streams.fresh("feed_status_2470")
    stores = world.stores
    calendar = world.calendar

    incident_date = scenario["period"]
    target_share = float(scenario["incident"]["target_revenue_share"])
    n_failed = int(scenario["targets"]["failed_store_feeds"])

    west = stores["region"].to_numpy() == "West"
    day_col = calendar["date"].dt.strftime("%Y-%m-%d").to_numpy() == incident_date
    day_revenue = world.actual[:, day_col].reshape(-1)
    west_total = day_revenue[west].sum()

    # Choose the 25 stores whose combined share is closest to 18.0%.
    west_idx = np.where(west)[0]
    order = west_idx[np.argsort(-day_revenue[west_idx])]
    best, best_err = None, np.inf
    for start in range(0, len(order) - n_failed + 1):
        pick = order[start : start + n_failed]
        share = day_revenue[pick].sum() / west_total
        err = abs(share - target_share)
        if err < best_err:
            best, best_err = pick, err
    failed = set(world.stores["store_id"].to_numpy()[best])

    rows = []
    for sid, region in zip(
        stores["store_id"].to_numpy(), stores["region"].to_numpy(), strict=True
    ):
        loaded = sid not in failed
        rows.append(
            {
                "feed_date": incident_date,
                "store_id": sid,
                "region": region,
                "source_system": "pos_erp",
                "status": "LOADED" if loaded else "FAILED",
                "rows_loaded": int(rng.integers(400, 900)) if loaded else 0,
                "recoverable": True,
            }
        )
    frame = pd.DataFrame(rows).sort_values("store_id", kind="stable").reset_index(drop=True)
    achieved = day_revenue[best].sum() / west_total
    world.diagnostics["incident_revenue_share"] = float(achieved)
    world.diagnostics["incident_failed_feeds"] = int(n_failed)
    return {"pos_erp/feed_status.csv": write_csv(frame, out / "pos_erp" / "feed_status.csv")}
