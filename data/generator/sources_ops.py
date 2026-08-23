"""store_ops and context source systems.

Continues `sources.py`. Everything unstructured lives here: inventory
snapshots, free-text store notes, support tickets, and the context feeds.

The unstructured tables carry no labels. A note about an empty shelf has
no `is_stockout` column and no tag — a classifier has to read the body.
The ticket `category` field exists but is populated by whoever took the
call and is wrong about a quarter of the time, so the body is the truth.
Both are deliberate: they are what the extraction lane has to earn.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.generator.sources import write_csv


# ---------------------------------------------------------------------------
# store_ops
# ---------------------------------------------------------------------------


def emit_inventory_snapshot(world, out: Path) -> dict[str, int]:
    """Store x SKU x four snapshots a day, over a bounded window.

    Snapshot hours are 06:00 / 12:00 / 18:00 / 23:00. The 12:00 read is
    what the availability KPI uses; the intraday shape makes 06:00 the
    fullest and 23:00 the most depleted, so the choice of snapshot hour
    genuinely changes the answer.
    """
    entity = world.entity
    rng = world.streams.fresh("inventory_snapshot")
    window = int(entity["emission"]["inventory_snapshot_window_days"])
    hours = list(entity["emission"]["snapshot_hours_ist"])
    top_n = int(entity["catalogue"]["top_n"])

    calendar = world.calendar
    day_idx = np.arange(len(calendar))[-window:]
    dates = calendar["date"].dt.strftime("%Y-%m-%d").to_numpy()[day_idx]

    stores = world.stores
    skus = world.skus.iloc[:top_n]
    n_stores, n_days, n_sku, n_hour = len(stores), len(day_idx), top_n, len(hours)

    observed = world.mechanism.availability[:, day_idx]
    hour_shape = np.array([1.03, 1.00, 0.96, 0.92])[:n_hour]

    base = observed[:, :, None, None] * hour_shape[None, None, None, :]
    sku_jitter = rng.normal(0.0, 0.02, size=(n_stores, 1, n_sku, 1))
    avail = np.clip(base + sku_jitter, 0.02, 1.0)

    in_stock = rng.random(avail.shape) < avail
    on_hand = np.where(in_stock, rng.integers(1, 40, size=avail.shape), 0)

    frame = pd.DataFrame(
        {
            "store_id": np.repeat(stores["store_id"].to_numpy(), n_days * n_sku * n_hour),
            "snapshot_date": np.tile(np.repeat(dates, n_sku * n_hour), n_stores),
            "sku_id": np.tile(
                np.repeat(skus["sku_id"].to_numpy(), n_hour), n_stores * n_days
            ),
            "snapshot_hour_ist": np.tile(hours, n_stores * n_days * n_sku),
            "on_hand_units": on_hand.reshape(-1).astype(np.int32),
            "is_ranged": True,
        }
    )
    frame = frame.sort_values(
        ["store_id", "snapshot_date", "sku_id", "snapshot_hour_ist"], kind="stable"
    ).reset_index(drop=True)
    return {
        "store_ops/inventory_snapshot.csv": write_csv(
            frame, out / "store_ops" / "inventory_snapshot.csv"
        )
    }


#: Free text. No tags, no category column. English and Hinglish mixed,
#: because that is what shift notes in an Indian retail chain look like.
NOTE_UNAVAILABLE = [
    "Customer asked for size 8 in the running range, not available since morning.",
    "Bahut customers aa rahe hain but popular sizes stock mein nahi hai.",
    "Third day running we have no stock in the fast moving articles.",
    "Size 7 aur 8 dono khatam. Customer wapas chala gaya without buying.",
    "Shelf for top sellers empty again, replenishment truck did not come.",
    "Lost at least 10 sales today, article not available in required size.",
    "Stock out on hero articles, customers upset. Please escalate to DC.",
    "Popular models ka stock nil hai. Kal se same problem chal rahi hai.",
]
NOTE_ORDINARY = [
    "Routine day, footfall normal, no issues to report.",
    "AC repair happened in morning, otherwise normal trading.",
    "Weekend rush was good, billing counter queue managed well.",
    "Staff training conducted for new POS screen.",
    "Aaj ka din theek tha, koi badi problem nahi.",
    "Cleaning contractor came late, store opened on time.",
    "Two staff on leave, managed with existing team.",
    "Local road work outside, slight drop in walk-ins.",
    "New window display installed as per visual merchandising guide.",
    "Customer feedback positive on the new casual range.",
]


def emit_store_notes(world, out: Path) -> dict[str, int]:
    """Unstructured shift notes, roughly 180 a week nationally.

    During the fault, a targeted number of treated and control stores file
    notes about unavailability. The COUNTS come from the scenario config
    because they are evidence the classifier has to find; the note text
    carries no label saying so.
    """
    entity = world.entity
    scenario = world.scenarios["2451"]
    rng = world.streams.fresh("store_notes")
    per_week = int(entity["emission"]["store_notes_per_week_national"])

    calendar = world.calendar
    stores = world.stores
    store_ids = stores["store_id"].to_numpy()
    dates = calendar["date"].to_numpy()

    iso_year = calendar["iso_year"].to_numpy()
    iso_week = calendar["iso_week"].to_numpy()
    week_keys = sorted({(int(y), int(w)) for y, w in zip(iso_year, iso_week, strict=True)})

    treated = sorted(store_ids[world.mechanism.treated_mask])
    control = sorted(store_ids[world.mechanism.control_mask])
    n_treated_flag = int(scenario["evidence"]["store_notes_flagging_treated"])
    n_control_flag = int(scenario["evidence"]["store_notes_flagging_control"])

    fault_start = np.datetime64(scenario["mechanism"]["onset_date"])

    rows: list[dict] = []
    seq = 0
    for year, week in week_keys:
        mask = (iso_year == year) & (iso_week == week)
        week_dates = dates[mask]
        if len(week_dates) == 0:
            continue
        for _ in range(per_week):
            seq += 1
            rows.append(
                {
                    "note_id": f"N{seq:06d}",
                    "store_id": str(rng.choice(store_ids)),
                    "note_date": np.datetime_as_string(
                        week_dates[int(rng.integers(0, len(week_dates)))], unit="D"
                    ),
                    "author_role": str(rng.choice(["Store Manager", "Shift Lead", "ASM"])),
                    "body": str(rng.choice(NOTE_ORDINARY)),
                }
            )

    fault_days = dates[dates >= fault_start]
    for group, count in ((treated, n_treated_flag), (control, n_control_flag)):
        chosen = rng.choice(np.array(group), size=count, replace=False)
        for sid in chosen:
            seq += 1
            rows.append(
                {
                    "note_id": f"N{seq:06d}",
                    "store_id": str(sid),
                    "note_date": np.datetime_as_string(
                        fault_days[int(rng.integers(0, len(fault_days)))], unit="D"
                    ),
                    "author_role": "Store Manager",
                    "body": str(rng.choice(NOTE_UNAVAILABLE)),
                }
            )

    frame = pd.DataFrame(rows).sort_values(
        ["note_date", "note_id"], kind="stable"
    ).reset_index(drop=True)
    return {"store_ops/store_notes.csv": write_csv(frame, out / "store_ops" / "store_notes.csv")}


TICKET_SIZE_BODIES = [
    "Wanted size 8 but shop did not have it, went to another brand.",
    "Size not available in the model I saw online. Very disappointing.",
    "Mera size store mein nahi mila, teesri baar aaya hoon.",
    "Asked for 7 and 8, both out of stock. Staff said stock not received.",
]
TICKET_OTHER_BODIES = [
    "Billing took too long at the counter.",
    "Product quality issue, sole came off within a month.",
    "Exchange request for wrong colour delivered.",
    "Staff was helpful, just wanted to appreciate.",
    "Refund not credited yet, please check.",
]
TICKET_CATEGORIES = ["SIZE_ISSUE", "QUALITY", "BILLING", "OTHER", "DELIVERY"]


def emit_tickets(world, out: Path) -> dict[str, int]:
    """Event-level tickets. The category field is unreliable by design."""
    scenario = world.scenarios["2451"]
    rng = world.streams.fresh("tickets")
    calendar = world.calendar
    stores = world.stores
    dates = calendar["date"].to_numpy()
    period = calendar["period_month"].to_numpy()

    target = int(scenario["evidence"]["support_tickets_count"])
    increase = float(scenario["evidence"]["support_tickets_increase_pct"])
    baseline = int(round(target / (1.0 + increase / 100.0)))

    all_ids = stores["store_id"].to_numpy()
    west_ids = all_ids[stores["region"].to_numpy() == "West"]
    treated_ids = all_ids[world.mechanism.treated_mask]

    rows: list[dict] = []
    seq = 0

    def add(day, sid, body, is_size: bool) -> None:
        nonlocal seq
        seq += 1
        if is_size:
            category = (
                "SIZE_ISSUE" if rng.random() > 0.28 else str(rng.choice(TICKET_CATEGORIES))
            )
        else:
            category = str(rng.choice(TICKET_CATEGORIES))
        rows.append(
            {
                "ticket_id": f"T{seq:07d}",
                "store_id": str(sid),
                "created_date": np.datetime_as_string(day, unit="D"),
                "category": category,
                "body": body,
                "channel": str(rng.choice(["phone", "email", "app"])),
            }
        )

    for day in dates:
        for _ in range(int(rng.poisson(6))):
            add(day, rng.choice(all_ids), str(rng.choice(TICKET_OTHER_BODIES)), False)

    for month, count in (("2025-10", baseline), ("2025-11", target)):
        month_dates = dates[period == month]
        for _ in range(count):
            day = month_dates[int(rng.integers(0, len(month_dates)))]
            sid = rng.choice(treated_ids) if rng.random() < 0.78 else rng.choice(west_ids)
            add(day, sid, str(rng.choice(TICKET_SIZE_BODIES)), True)

    frame = pd.DataFrame(rows).sort_values(
        ["created_date", "ticket_id"], kind="stable"
    ).reset_index(drop=True)
    return {"store_ops/tickets.csv": write_csv(frame, out / "store_ops" / "tickets.csv")}


def emit_footfall(world, out: Path) -> dict[str, int]:
    """Daily footfall — NULL for 109 of 140 West stores.

    Only instrumented stores emit a count. The uninstrumented ones emit a
    row with a null, rather than no row at all, so the gap is visible
    rather than silently absent.
    """
    rng = world.streams.fresh("footfall")
    entity = world.entity
    stores = world.stores
    calendar = world.calendar

    window = int(entity["emission"]["sku_daily_window_days"])
    day_idx = np.arange(len(calendar))[-window:]
    dates = calendar["date"].dt.strftime("%Y-%m-%d").to_numpy()[day_idx]

    base_conversion = float(entity["footfall"]["conversion_rate_base_pt"]) / 100.0
    units = world.actual[:, day_idx]
    asp = float(np.average(world.skus["list_price_inr"], weights=world.skus["demand_share"]))
    transactions = units / (asp * entity["catalogue"]["units_per_bill_mean"])

    counted = transactions / base_conversion * np.exp(
        rng.normal(0.0, 0.06, size=transactions.shape)
    )

    has_counter = stores["has_footfall_counter"].to_numpy()
    n_stores, n_days = len(stores), len(day_idx)

    values = np.where(
        np.repeat(has_counter[:, None], n_days, axis=1),
        np.round(counted),
        np.nan,
    )

    frame = pd.DataFrame(
        {
            "store_id": np.repeat(stores["store_id"].to_numpy(), n_days),
            "footfall_date": np.tile(dates, n_stores),
            "footfall": values.reshape(-1),
            "counter_installed": np.repeat(has_counter, n_days),
        }
    )
    frame["footfall"] = frame["footfall"].astype("Int64")
    frame = frame.sort_values(["store_id", "footfall_date"], kind="stable").reset_index(
        drop=True
    )
    return {"store_ops/footfall_daily.csv": write_csv(frame, out / "store_ops" / "footfall_daily.csv")}


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------


def emit_calendar(world, out: Path) -> dict[str, int]:
    frame = world.calendar.copy()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return {"context/calendar.csv": write_csv(frame, out / "context" / "calendar.csv")}


def emit_festival_windows(world, out: Path) -> dict[str, int]:
    """One row per (festival occurrence, day). The bridge dim_calendar cannot be.

    `dim_calendar.festival` holds ONE label per day, and festivals overlap:
    Onam and Ganesh Chaturthi run together in South, Navratri and Durga
    Puja in East. A single label silently drops the second one, and a
    calendar model built on it cannot fit either.

    This table carries membership properly, plus the two things a model
    needs and a flag cannot express:

      day_index / window_days   where in the ramp the day falls. A festival
                                is not a switch; trade builds through the
                                window and peaks somewhere inside it.
      phase                     'window' or 'hangover'. The days after a
                                festival are suppressed, not normal.

    `occurrence` is the fiscal year the window opens in, so the two Diwalis
    are two rows and not one smeared average.
    """
    from data.generator.model import FESTIVAL_HANGOVER_DAYS, FESTIVALS

    calendar = world.calendar
    known = set(calendar["date"].dt.date.to_numpy())
    fiscal = dict(
        zip(calendar["date"].dt.date, calendar["fiscal_year_label"], strict=True)
    )

    rows = []
    for name, windows in sorted(FESTIVALS.items()):
        for window_start, window_end in windows:
            span = (window_end - window_start).days + 1
            occurrence = fiscal.get(window_start, "")
            for offset in range(span):
                day = window_start + timedelta(days=offset)
                if day not in known:
                    continue
                rows.append(
                    {
                        "date": day.isoformat(),
                        "festival": name,
                        "occurrence": occurrence or fiscal.get(day, ""),
                        "phase": "window",
                        "day_index": offset,
                        "phase_days": span,
                    }
                )
            for offset in range(FESTIVAL_HANGOVER_DAYS):
                day = window_end + timedelta(days=offset + 1)
                if day not in known:
                    continue
                rows.append(
                    {
                        "date": day.isoformat(),
                        "festival": name,
                        "occurrence": occurrence or fiscal.get(day, ""),
                        "phase": "hangover",
                        "day_index": offset,
                        "phase_days": FESTIVAL_HANGOVER_DAYS,
                    }
                )

    frame = pd.DataFrame(rows).sort_values(
        ["date", "festival", "phase"], kind="stable"
    ).reset_index(drop=True)
    return {
        "context/festival_windows.csv": write_csv(
            frame, out / "context" / "festival_windows.csv"
        )
    }


CAMPAIGNS = ["Always On", "Festive Push", "Weekend Offer", "New Range", "Clearance"]


def emit_marketing_spend(world, out: Path) -> dict[str, int]:
    """Region x campaign x ISO week.

    Carries #2451's H4: West spend drops 22%, but the cut lands ELEVEN
    DAYS AFTER the decline began. That ordering is what Test 1 uses to
    eliminate it, and it is the single best demo moment — H4 has the
    strongest correlation in the data and is still wrong.
    """
    rng = world.streams.fresh("marketing_spend")
    scenario = world.scenarios["2451"]
    h4 = scenario["competing_hypotheses"]["H4_marketing"]

    calendar = world.calendar
    onset = np.datetime64(scenario["mechanism"]["onset_date"])
    cut_from = onset + np.timedelta64(int(h4["onset_offset_days"]), "D")
    cut_pct = float(h4["spend_change_pct"]) / 100.0

    weeks = (
        calendar.groupby(["iso_year", "iso_week"], as_index=False)
        .agg(week_start=("date", "min"))
        .sort_values(["iso_year", "iso_week"])
    )

    rows = []
    from data.generator.model import REGIONS

    for region in REGIONS:
        annual = world.entity["regions"][region]["annual_net_revenue_inr_cr"]
        weekly_budget = annual * 1e7 * 0.035 / 52.0
        for _, week in weeks.iterrows():
            week_start = np.datetime64(week["week_start"].date())
            for campaign in CAMPAIGNS:
                share = {
                    "Always On": 0.42,
                    "Festive Push": 0.22,
                    "Weekend Offer": 0.15,
                    "New Range": 0.13,
                    "Clearance": 0.08,
                }[campaign]
                spend = weekly_budget * share * float(np.exp(rng.normal(0.0, 0.08)))
                if region == "West" and week_start >= cut_from:
                    spend *= 1.0 + cut_pct
                rows.append(
                    {
                        "region": region,
                        "campaign": campaign,
                        "iso_year": int(week["iso_year"]),
                        "iso_week": int(week["iso_week"]),
                        "week_start": np.datetime_as_string(week_start, unit="D"),
                        "spend_inr": round(float(spend), 2),
                    }
                )

    frame = pd.DataFrame(rows).sort_values(
        ["iso_year", "iso_week", "region", "campaign"], kind="stable"
    ).reset_index(drop=True)
    return {
        "context/marketing_spend.csv": write_csv(
            frame, out / "context" / "marketing_spend.csv"
        )
    }


#: Six items a quarter. NO pricing, NO footfall — the gap is the point.
#: These headlines are what makes competitor_action look plausible and
#: remain untestable.
NEWS_TEMPLATES = [
    "{competitor} opens flagship store in {city}",
    "{competitor} announces festive campaign across west India",
    "{competitor} expands quick-commerce partnership",
    "Footwear retail sees crowded festive season, say analysts",
    "{competitor} reports strong quarterly growth in western markets",
    "{competitor} signs celebrity endorsement for running range",
    "Discount war expected in footwear ahead of festive season",
    "{competitor} refreshes store format in metro locations",
]
COMPETITORS = ["Stride India", "PaceWear", "Corsa Footwear", "UrbanSole"]


def emit_competitor_news(world, out: Path) -> dict[str, int]:
    """Six items a quarter, headline and body only.

    There is no price column and no footfall column, because the
    organisation does not hold those feeds. That absence is what makes
    trigger T3 fire on #2467 and it must not be quietly filled in.
    """
    rng = world.streams.fresh("competitor_news")
    calendar = world.calendar
    quarters = sorted(
        {
            (int(r.fiscal_year), int(r.fiscal_quarter))
            for r in calendar.itertuples()
        }
    )
    cities = [c for region in world.entity["regions"].values() for c in region["cities"]]

    rows = []
    seq = 0
    for fy, fq in quarters:
        mask = (calendar["fiscal_year"].to_numpy() == fy) & (
            calendar["fiscal_quarter"].to_numpy() == fq
        )
        days = calendar["date"].to_numpy()[mask]
        if len(days) == 0:
            continue
        for _ in range(6):
            seq += 1
            competitor = str(rng.choice(COMPETITORS))
            headline = str(rng.choice(NEWS_TEMPLATES)).format(
                competitor=competitor, city=str(rng.choice(cities))
            )
            rows.append(
                {
                    "news_id": f"NW{seq:04d}",
                    "published_date": np.datetime_as_string(
                        days[int(rng.integers(0, len(days)))], unit="D"
                    ),
                    "competitor": competitor,
                    "headline": headline,
                    "body": headline + ". No pricing or footfall detail disclosed.",
                    "source": str(rng.choice(["TradeWire", "RetailDaily", "MarketPulse"])),
                }
            )

    frame = pd.DataFrame(rows).sort_values(
        ["published_date", "news_id"], kind="stable"
    ).reset_index(drop=True)
    return {
        "context/competitor_news.csv": write_csv(
            frame, out / "context" / "competitor_news.csv"
        )
    }


REVIEW_PLATFORMS = ["Maps", "AppStore", "Marketplace"]

#: Public review text. The lowest reliability tier there is (0.35), and
#: the only lane in which a competitor promotion is ever mentioned. #2467
#: turns on the fact that fourteen of these plus three news items is still
#: not enough to reach a verdict.
REVIEW_TEMPLATES = [
    "Staff were helpful but half the sizes I wanted were gone.",
    "Nice store, decent range. Billing was quick.",
    "Prices seem higher than {competitor} down the road right now.",
    "{competitor} is running a big discount this week, went there instead.",
    "Saw a {competitor} offer nearby, hard to justify full price here.",
    "Good collection, will come back.",
    "Waited a long time at the counter on a Sunday.",
    "Shoes are comfortable, no complaints.",
    "Asked for my size twice, they said stock is not there.",
    "Store is clean and the trial area is good.",
]

#: The subset above that names a competitor PROMOTION. Counting these is
#: what produces #2467's "14 review mentions", so they are placed rather
#: than drawn: background reviews never use them, or the count would be
#: whatever the noise happened to deal and the case would not be
#: reproducible. Everything else, including the review that merely
#: compares prices, is background.
COMPETITOR_REVIEW_TEMPLATES = REVIEW_TEMPLATES[3:5]
BACKGROUND_REVIEW_TEMPLATES = [
    body for body in REVIEW_TEMPLATES if body not in COMPETITOR_REVIEW_TEMPLATES
]


def emit_reviews(world, out: Path) -> dict[str, int]:
    """Public reviews at store grain — the weakest evidence lane there is.

    Two things matter about this table and neither is the volume:

      * it carries no rating breakdown, no verified-purchase flag and no
        competitor price, so it can corroborate a hypothesis and never
        test one
      * #2467 needs exactly the configured number of reviews naming a
        competitor promotion in East during the case period, because the
        whole point of that case is that fourteen weak mentions plus
        three news items still does not reach a verdict

    The competitor mentions are placed; the background reviews are drawn.
    """
    rng = world.streams.fresh("reviews")
    scenario = world.scenarios["2467"]
    calendar = world.calendar
    stores = world.stores

    period = scenario["period"]
    target_mentions = int(scenario["evidence"]["competitor_review_mentions"])
    scope = scenario["scope"]

    in_period = calendar["period_month"].to_numpy() == period
    period_days = calendar["date"].to_numpy()[in_period]
    all_days = calendar["date"].to_numpy()

    scope_stores = stores[stores["region"] == scope]["store_id"].to_numpy()
    every_store = stores["store_id"].to_numpy()

    rows = []
    seq = 0

    # Background: ordinary reviews across the estate and the whole window.
    background = int(len(every_store) * float(world.entity["reviews"]["per_store_over_window"]))
    for _ in range(background):
        seq += 1
        template = str(rng.choice(BACKGROUND_REVIEW_TEMPLATES))
        rows.append(
            {
                "review_id": f"RV{seq:05d}",
                "store_id": str(rng.choice(every_store)),
                "review_date": np.datetime_as_string(
                    all_days[int(rng.integers(0, len(all_days)))], unit="D"
                ),
                "rating": int(rng.integers(1, 6)),
                "body": template.format(competitor=str(rng.choice(COMPETITORS))),
                "platform": str(rng.choice(REVIEW_PLATFORMS)),
            }
        )

    # #2467: the competitor-promotion mentions, in scope and in period.
    for _ in range(target_mentions):
        seq += 1
        template = str(rng.choice(COMPETITOR_REVIEW_TEMPLATES))
        rows.append(
            {
                "review_id": f"RV{seq:05d}",
                "store_id": str(rng.choice(scope_stores)),
                "review_date": np.datetime_as_string(
                    period_days[int(rng.integers(0, len(period_days)))], unit="D"
                ),
                "rating": int(rng.integers(1, 4)),
                "body": template.format(competitor=str(rng.choice(COMPETITORS))),
                "platform": str(rng.choice(REVIEW_PLATFORMS)),
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        ["review_date", "review_id"], kind="stable"
    ).reset_index(drop=True)
    return {"store_ops/reviews.csv": write_csv(frame, out / "store_ops" / "reviews.csv")}


def emit_weather(world, out: Path) -> dict[str, int]:
    """Daily weather at city grain."""
    rng = world.streams.fresh("weather")
    calendar = world.calendar
    cities = sorted(
        {
            (city, region)
            for region, cfg in world.entity["regions"].items()
            for city in cfg["cities"]
        }
    )
    dates = calendar["date"].dt.strftime("%Y-%m-%d").to_numpy()
    doy = calendar["date"].dt.dayofyear.to_numpy()

    rows = []
    for city, region in cities:
        seasonal = 28.0 + 6.0 * np.sin(2 * np.pi * (doy - 100) / 365.25)
        temp = seasonal + rng.normal(0.0, 2.2, size=len(dates))
        monsoon = ((doy >= 160) & (doy <= 270)).astype(float)
        rain = rng.gamma(1.4, 6.0, size=len(dates)) * (0.25 + 2.2 * monsoon)
        rows.append(
            pd.DataFrame(
                {
                    "city": city,
                    "region": region,
                    "weather_date": dates,
                    "temp_c": np.round(temp, 1),
                    "rain_mm": np.round(rain, 1),
                }
            )
        )

    frame = pd.concat(rows, ignore_index=True)
    frame = frame.sort_values(["weather_date", "city"], kind="stable").reset_index(drop=True)
    return {"context/weather_daily.csv": write_csv(frame, out / "context" / "weather_daily.csv")}


def emit_qcomm_weekly(world, out: Path) -> dict[str, int]:
    """#2471: seven weekly points against twenty-six required."""
    scenario = world.scenarios["2471"]
    rng = world.streams.fresh("qcomm")
    series = scenario["series"]

    launch = pd.Timestamp(scenario["targets"]["launch_date"])
    end = world.calendar["date"].max()
    weeks = pd.date_range(launch, end, freq="7D")

    rows = []
    for i, week_start in enumerate(weeks):
        rate = series["fulfilment_rate_base_pct"] * (
            1.0 + rng.normal(0.0, series["weekly_noise_sigma"])
        )
        rows.append(
            {
                "week_start": week_start.strftime("%Y-%m-%d"),
                "week_index": i + 1,
                "scope": "All-India",
                "dark_stores": int(series["dark_stores"]),
                "orders": int(rng.integers(48000, 62000)),
                "fulfilment_rate_pct": round(float(rate), 3),
            }
        )

    frame = pd.DataFrame(rows)
    return {"context/qcomm_weekly.csv": write_csv(frame, out / "context" / "qcomm_weekly.csv")}
