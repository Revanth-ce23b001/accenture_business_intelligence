-- Seed the warehouse from data/raw.
--
-- `{raw}` is substituted with the raw directory by
-- engine/warehouse/load.py. Every column is named and every date is cast
-- explicitly: read_csv_auto's inference is convenient and is not a
-- contract, and a silently re-typed column is a bug that surfaces three
-- stages later.
--
-- NOTHING IS REPAIRED HERE. The B2B column is loaded, the returns column
-- is loaded, the unresolvable store_codes are loaded, and the weekly
-- marketing grain stays weekly. Reconciliation is a report
-- (engine/warehouse/reconcile.py), not a rewrite.

-- --- dimensions ------------------------------------------------------------

INSERT INTO dim_store
SELECT store_id, outlet_id, region, city, store_format, catchment_type,
       sqft, staff_headcount, size_index,
       has_footfall_counter, is_treated_2451, is_matched_control_2451
FROM read_csv_auto('{raw}/dim/dim_store.csv');

INSERT INTO dim_sku
SELECT sku_id, category, list_price_inr, demand_share, is_top20
FROM read_csv_auto('{raw}/dim/dim_sku.csv');

INSERT INTO dim_calendar
SELECT CAST(date AS DATE), year, month, day, dow,
       iso_year, iso_week, CAST(iso_week_start AS DATE), period_month,
       fiscal_year, fiscal_year_label, fiscal_quarter,
       fiscal_week, CAST(fiscal_week_start AS DATE),
       is_weekend, is_month_end, is_payday,
       NULLIF(CAST(festival AS VARCHAR), ''), is_festival,
       NULLIF(CAST(promo_window AS VARCHAR), ''), is_promo_window
FROM read_csv_auto('{raw}/context/calendar.csv');

INSERT INTO dim_festival_window
SELECT CAST(date AS DATE), festival, occurrence, phase, day_index, phase_days
FROM read_csv_auto('{raw}/context/festival_windows.csv');

INSERT INTO dim_store_xref
SELECT store_code, outlet_id,
       CAST(valid_from AS DATE),
       CAST(NULLIF(CAST(valid_to AS VARCHAR), '') AS DATE),
       mapping_source
FROM read_csv_auto('{raw}/dim/dim_store_xref.csv');

INSERT INTO dim_user
SELECT user_id, display_name, email, persona,
       NULLIF(CAST(region AS VARCHAR), ''),
       NULLIF(CAST(store_id AS VARCHAR), '')
FROM read_csv_auto('{raw}/dim/dim_user.csv');

-- --- facts -----------------------------------------------------------------

INSERT INTO fact_sales_daily
SELECT store_id, store_code, region, CAST(txn_date AS DATE),
       gross_amount_inr, returns_amount_inr, discount_amount_inr,
       tax_amount_inr, net_revenue_inr, b2b_net_revenue_inr,
       transfer_amount_inr, units, calendar_expected_inr
FROM read_csv_auto('{raw}/pos_erp/sales_daily.csv');

INSERT INTO fact_sales_daily_sku
SELECT store_id, CAST(txn_date AS DATE), sku_id, net_revenue_inr, units
FROM read_csv_auto('{raw}/pos_erp/sales_daily_sku.csv');

INSERT INTO fact_bill_lines
SELECT bill_id, line_no, store_id, store_code, sku_id,
       CAST(txn_ts AS TIMESTAMP), CAST(posted_date AS DATE),
       channel, txn_type, units,
       gross_amount, discount_amount, tax_amount, returns_amount, sample_rate
FROM read_csv_auto('{raw}/pos_erp/bill_lines.csv');

INSERT INTO fact_feed_status
SELECT CAST(feed_date AS DATE), store_id, region, source_system,
       status, rows_loaded, recoverable
FROM read_csv_auto('{raw}/pos_erp/feed_status.csv');

INSERT INTO restatement_register
SELECT restatement_id, kpi, scope, grain, period,
       CAST(flagged_at AS DATE),
       CAST(NULLIF(CAST(resolved_at AS VARCHAR), '') AS DATE),
       status, reason
FROM read_csv_auto('{raw}/pos_erp/restatements.csv');

INSERT INTO fact_inventory_snapshot
SELECT store_id, CAST(snapshot_date AS DATE), sku_id,
       snapshot_hour_ist, on_hand_units, is_ranged
FROM read_csv_auto('{raw}/store_ops/inventory_snapshot.csv');

INSERT INTO fact_footfall_daily
SELECT store_id, CAST(footfall_date AS DATE), footfall, counter_installed
FROM read_csv_auto('{raw}/store_ops/footfall_daily.csv');

INSERT INTO fact_marketing_spend_weekly
SELECT region, campaign, iso_year, iso_week, CAST(week_start AS DATE), spend_inr
FROM read_csv_auto('{raw}/context/marketing_spend.csv');

INSERT INTO fact_qcomm_weekly
SELECT CAST(week_start AS DATE), week_index, scope,
       dark_stores, orders, fulfilment_rate_pct
FROM read_csv_auto('{raw}/context/qcomm_weekly.csv');

-- --- documents -------------------------------------------------------------

INSERT INTO doc_store_notes
SELECT note_id, store_id, CAST(note_date AS DATE), author_role, body
FROM read_csv_auto('{raw}/store_ops/store_notes.csv');

INSERT INTO doc_tickets
SELECT ticket_id, store_id, CAST(created_date AS DATE), category, body, channel
FROM read_csv_auto('{raw}/store_ops/tickets.csv');

INSERT INTO doc_reviews
SELECT review_id, store_id, CAST(review_date AS DATE), rating, body, platform
FROM read_csv_auto('{raw}/store_ops/reviews.csv');

-- --- external --------------------------------------------------------------

INSERT INTO ext_competitor_news
SELECT news_id, CAST(published_date AS DATE), competitor, headline, body, source
FROM read_csv_auto('{raw}/context/competitor_news.csv');

INSERT INTO ext_weather_daily
SELECT city, region, CAST(weather_date AS DATE), temp_c, rain_mm
FROM read_csv_auto('{raw}/context/weather_daily.csv');
