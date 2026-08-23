-- Conformance views.
--
-- These are where the four reconciliation problems become queryable. Each
-- view SEPARATES the clean rows from the broken ones and keeps both. None
-- of them repairs anything, and none of them falls back to a second join
-- key when the first one fails — a silent fallback is exactly how a
-- mapping gap survives for years without anybody noticing.

-- ---------------------------------------------------------------------------
-- ENTITY KEY MISMATCH
--
-- POS keys on store_code. Store operations keys on outlet_id. dim_store_xref
-- is the contracted bridge and the only one. `resolved` is FALSE where the
-- POS code has no row in the bridge, which is what the re-fascia caused.
--
-- fact_sales_daily also carries store_id, and it is deliberately NOT used to
-- rescue the unresolved rows. store_id is the surrogate the POS system
-- stamped at install and never re-issued; joining on it would make the
-- integration look healthy while the mapping the business actually
-- maintains stays broken.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_sales_daily_keyed AS
SELECT
    f.*,
    x.outlet_id,
    x.outlet_id IS NOT NULL AS resolved
FROM fact_sales_daily AS f
LEFT JOIN dim_store_xref AS x
       ON f.store_code = x.store_code;

CREATE OR REPLACE VIEW v_sales_daily_conformed AS
SELECT * FROM v_sales_daily_keyed WHERE resolved;

CREATE OR REPLACE VIEW v_sales_daily_quarantined AS
SELECT * FROM v_sales_daily_keyed WHERE NOT resolved;

-- ---------------------------------------------------------------------------
-- DEFINITION CONFLICT
--
-- All three revenue figures for the same store-day, side by side. The KPI
-- contract's figure is `net_revenue_inr`; the other two are what the POS
-- ledger and the marketing team publish for the same period.
--
-- The expressions here MUST stay in step with
-- semantic_layer/warehouse.yaml -> reconciliation.definition_conflict, which
-- is the arbiter of what each team means. tests/test_reconciliation.py
-- checks that the two agree.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_revenue_definitions AS
SELECT
    store_id,
    store_code,
    region,
    txn_date,
    net_revenue_inr                        AS contract_inr,
    net_revenue_inr + b2b_net_revenue_inr  AS pos_ledger_inr,
    net_revenue_inr + returns_amount_inr   AS marketing_reported_inr,
    b2b_net_revenue_inr,
    returns_amount_inr,
    transfer_amount_inr
FROM fact_sales_daily;

-- ---------------------------------------------------------------------------
-- GRAIN MISMATCH
--
-- Marketing spend is booked by ISO week; everything it is compared against
-- is daily. This view spreads each week's spend evenly across the days of
-- that ISO week THAT EXIST IN dim_calendar, so a truncated week at either
-- end of the series divides by its real day count rather than by a nominal
-- seven.
--
-- `days_in_week` is carried on every row so a consumer can see when the
-- divisor was not seven. The allocation is still an assumption, and every
-- Evidence derived from it is tagged `flat_intraweek`.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_marketing_spend_daily AS
SELECT
    m.region,
    m.campaign,
    m.iso_year,
    m.iso_week,
    c.date                                        AS spend_date,
    m.spend_inr                                   AS week_spend_inr,
    COUNT(*) OVER (PARTITION BY m.region, m.campaign, m.iso_year, m.iso_week)
                                                  AS days_in_week,
    m.spend_inr
      / COUNT(*) OVER (PARTITION BY m.region, m.campaign, m.iso_year, m.iso_week)
                                                  AS allocated_spend_inr
FROM fact_marketing_spend_weekly AS m
JOIN dim_calendar AS c
  ON c.iso_year = m.iso_year
 AND c.iso_week = m.iso_week;

-- ---------------------------------------------------------------------------
-- CALENDAR MISMATCH
--
-- One row per day carrying both period keys, so any measure can be rolled
-- up either way and the two answers compared. `same_week_start` is TRUE
-- only where the fiscal week and the ISO week happen to begin on the same
-- day, which is what makes the divergence measurable rather than asserted.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_calendar_alignment AS
SELECT
    date,
    period_month,
    iso_year,
    iso_week,
    iso_week_start,
    fiscal_year,
    fiscal_year_label,
    fiscal_week,
    fiscal_week_start,
    festival,
    is_festival,
    iso_week_start = fiscal_week_start AS same_week_start
FROM dim_calendar;
