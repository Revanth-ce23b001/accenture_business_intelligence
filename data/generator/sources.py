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
    splits them in two -- and dim_store_xref never learned the new
    codes, so a join to the operations key drops them entirely
  * the POS ledger books B2B in the same table, and marketing's revenue
    figure never deducts a return, so three revenue figures exist for
    the same period and none of them is silently reconciled
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
    """Entity masters.

    `dim_store` is the STORE OPERATIONS master and carries `outlet_id`,
    the key that system issues. It deliberately does NOT carry the POS
    `store_code`: the only mapping the organisation maintains between the
    two systems is `dim_store_xref`, and duplicating it here would give
    the warehouse a private second copy that could never go stale — which
    is precisely the failure this data set exists to make visible.
    """
    stores = world.stores.copy()
    stores["is_treated_2451"] = world.mechanism.treated_mask
    stores["is_matched_control_2451"] = world.mechanism.control_mask
    stores = stores.drop(columns=["store_code"])
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


def store_code_on(world) -> np.ndarray:
    """POS store_code per (store, day), honouring the re-fascia.

    Three stores were issued a new store_code when their fascia changed,
    and the POS system started stamping the new code that day. Nothing
    told store operations, so `dim_store_xref` has no row for the new
    code and those rows cannot be joined -- see `emit_store_xref`.
    """
    n_stores, n_days = world.actual.shape
    codes = np.repeat(world.stores["store_code"].to_numpy()[:, None], n_days, axis=1)
    if not world.refascia_codes:
        return codes
    after = world.calendar["date"].to_numpy() >= np.datetime64(world.refascia_cutover)
    positions = {sid: i for i, sid in enumerate(world.stores["store_id"].to_numpy())}
    for store_id, new_code in world.refascia_codes.items():
        codes[positions[store_id], after] = new_code
    return codes


def emit_sales_daily(world, out: Path) -> dict[str, int]:
    """Authoritative store x day net revenue. Every statistic reads this.

    `net_revenue_inr` is the KPI contract's figure: retail only, no
    transfers, net of returns, discount and tax. The columns beside it are
    what the contract had to arbitrate between, and they are carried so
    the arbitration is reproducible rather than asserted:

      returns_amount_inr    marketing's figure never deducts this
      b2b_net_revenue_inr   the POS ledger's figure includes this
      transfer_amount_inr   both definitions exclude this; carried so the
                            second scope exclusion is visible at this grain

    The identity `gross - returns - discount - tax = net` holds EXACTLY in
    the emitted CSV, not to within rounding: gross is computed back from
    the already-rounded components.
    """
    entity = world.entity
    netting = entity["netting"]
    stores = world.stores
    calendar = world.calendar
    n_stores, n_days = world.actual.shape

    store_ids = np.repeat(stores["store_id"].to_numpy(), n_days)
    regions = np.repeat(stores["region"].to_numpy(), n_days)
    dates = np.tile(calendar["date"].dt.strftime("%Y-%m-%d").to_numpy(), n_stores)
    net = np.round(world.actual.reshape(-1), 2)
    expected = world.expected.reshape(-1)

    asp = _net_asp(world)
    units = np.round(net / asp).astype(np.int64)

    # Unwind the netting. net = taxable * (1 - tax - return_rate), where
    # taxable = gross * (1 - discount), so every component follows from net.
    tax_rate = float(netting["tax_rate"])
    return_rate = float(netting["return_rate"])
    discount_rate = float(netting["discount_rate_mean"])
    taxable = net / (1.0 - tax_rate - return_rate)
    tax = np.round(taxable * tax_rate, 2)
    returns = np.round(taxable * return_rate, 2)
    discount = np.round(taxable * discount_rate / (1.0 - discount_rate), 2)
    gross = np.round(net + returns + discount + tax, 2)

    # B2B wholesale, booked to the same store in the same ledger. Lumpy,
    # then rescaled so the national total lands on the solved share -- the
    # gap between the two rival definitions is a target, not a coincidence.
    reconciliation = entity["reconciliation"]
    rng = world.streams.fresh("b2b_daily")
    sigma = float(reconciliation["definition_conflict"]["b2b_lumpiness_sigma"])
    jitter = np.exp(rng.normal(-0.5 * sigma * sigma, sigma, size=net.shape))
    b2b = net * jitter
    b2b = np.round(b2b * (world.fitted_b2b_share * net.sum() / b2b.sum()), 2)

    transfer_jitter = np.exp(rng.normal(-0.5 * sigma * sigma, sigma, size=net.shape))
    transfer = np.round(
        gross * float(entity["defects"]["transfer_share_of_rows"]) * transfer_jitter, 2
    )

    frame = pd.DataFrame(
        {
            "store_id": store_ids,
            "store_code": store_code_on(world).reshape(-1),
            "region": regions,
            "txn_date": dates,
            "gross_amount_inr": gross,
            "returns_amount_inr": returns,
            "discount_amount_inr": discount,
            "tax_amount_inr": tax,
            "net_revenue_inr": net,
            "b2b_net_revenue_inr": b2b,
            "transfer_amount_inr": transfer,
            "units": units,
            "calendar_expected_inr": np.round(expected, 2),
        }
    ).sort_values(["store_id", "txn_date"], kind="stable").reset_index(drop=True)

    return {"pos_erp/sales_daily.csv": write_csv(frame, out / "pos_erp" / "sales_daily.csv")}


def emit_store_xref(world, out: Path) -> dict[str, int]:
    """The only bridge between the POS key and the operations key.

    POS facts key on `store_code`; store operations, footfall and the
    inventory system key on `outlet_id`. This table maps one to the other.

    It is deliberately INCOMPLETE. The three re-fasciad stores were issued
    new POS store_codes and nobody told operations, so no row here covers
    them. A join through this table drops those rows, and the warehouse
    quarantines and counts them rather than silently losing them. The new
    codes appear nowhere in this file: the gap has to be found by the join
    failing, not by being told about it.
    """
    stores = world.stores
    frame = pd.DataFrame(
        {
            "outlet_id": stores["outlet_id"].to_numpy(),
            "store_code": stores["store_code"].to_numpy(),
            "valid_from": world.calendar["date"].min().strftime("%Y-%m-%d"),
            "valid_to": "",
            "mapping_source": "ops_master",
        }
    ).sort_values("outlet_id", kind="stable").reset_index(drop=True)
    return {"dim/dim_store_xref.csv": write_csv(frame, out / "dim" / "dim_store_xref.csv")}


def emit_users(world, out: Path) -> dict[str, int]:
    """The personas `execute_governed` resolves an access policy against.

    `persona` must be a key in every KPI contract's `access_policy`, and
    `region` / `store_id` are what the contract's row predicate binds to.
    A regional manager with no region would silently see everything.
    """
    cfg = world.entity["users"]
    stores = world.stores
    domain = cfg["email_domain"]

    rows = []
    for entry in cfg["roster"]:
        region = entry.get("region")
        store_id = ""
        if "store_rank_in_region" in entry:
            in_region = stores[stores["region"] == region].sort_values(
                "store_id", kind="stable"
            )
            store_id = str(in_region["store_id"].iloc[int(entry["store_rank_in_region"]) - 1])
        handle = entry["display_name"].lower().replace(" ", ".")
        rows.append(
            {
                "user_id": entry["user_id"],
                "display_name": entry["display_name"],
                "email": f"{handle}@{domain}",
                "persona": entry["persona"],
                "region": region if region else "",
                "store_id": store_id,
            }
        )
    frame = pd.DataFrame(rows).sort_values("user_id", kind="stable").reset_index(drop=True)
    return {"dim/dim_user.csv": write_csv(frame, out / "dim" / "dim_user.csv")}


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

    # Three stores change store_code mid-period. WHICH three is decided in
    # build.py, where the resulting quarantine can be measured against its
    # target; this table only stamps the codes the POS system issued.
    new_codes = dict(world.refascia_codes)
    change_from = np.datetime64(world.refascia_cutover)

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

                is_b2b = rng.random() < defects["b2b_bill_share"]
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
