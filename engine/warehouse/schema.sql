-- CaseFile.ai warehouse schema.
--
-- Structure lives here rather than in Python for two reasons: CLAUDE.md
-- rule 2 forbids literals in engine/, and a warehouse whose shape is a
-- SQL file can be read by someone who does not read Python.
--
-- Three groups of tables:
--
--   dim_* / fact_* / doc_* / ext_*   the source systems, loaded as they
--                                    arrive. Nothing is repaired on load.
--   case_* and friends               the artefacts a case produces.
--   audit_log, data_gap_register     what the warehouse says about itself.
--
-- Two of the artefact tables are already live at P4: `audit_log`, written
-- by engine/db.py::execute_governed on every query, and
-- `data_gap_register`, written by engine/warehouse/reconcile.py with the
-- four disagreements it found. The rest are created empty and filled by
-- later stages.

-- ===========================================================================
-- Dimensions
-- ===========================================================================

-- The STORE OPERATIONS master. Its business key is `outlet_id`.
-- It holds no POS `store_code` on purpose: `dim_store_xref` is the only
-- mapping the organisation maintains between the two systems, and a second
-- private copy here could never go stale, which would hide the very gap
-- this warehouse exists to surface.
CREATE TABLE IF NOT EXISTS dim_store (
    store_id                  VARCHAR PRIMARY KEY,
    outlet_id                 VARCHAR NOT NULL,
    region                    VARCHAR NOT NULL,
    city                      VARCHAR NOT NULL,
    store_format              VARCHAR NOT NULL,
    catchment_type            VARCHAR NOT NULL,
    sqft                      INTEGER NOT NULL,
    staff_headcount           INTEGER NOT NULL,
    size_index                DOUBLE  NOT NULL,
    has_footfall_counter      BOOLEAN NOT NULL,
    is_treated_2451           BOOLEAN NOT NULL,
    is_matched_control_2451   BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_sku (
    sku_id          VARCHAR PRIMARY KEY,
    category        VARCHAR NOT NULL,
    list_price_inr  DOUBLE  NOT NULL,
    demand_share    DOUBLE  NOT NULL,
    is_top20        BOOLEAN NOT NULL
);

-- Both calendars, side by side. `iso_week_start` is always a Monday;
-- `fiscal_week_start` counts from 1 April whatever weekday that is. They
-- do not agree, and reconcile.py measures by how much.
CREATE TABLE IF NOT EXISTS dim_calendar (
    date               DATE PRIMARY KEY,
    year               INTEGER NOT NULL,
    month              INTEGER NOT NULL,
    day                INTEGER NOT NULL,
    dow                INTEGER NOT NULL,
    iso_year           INTEGER NOT NULL,
    iso_week           INTEGER NOT NULL,
    iso_week_start     DATE    NOT NULL,
    period_month       VARCHAR NOT NULL,
    fiscal_year        INTEGER NOT NULL,
    fiscal_year_label  VARCHAR NOT NULL,
    fiscal_quarter     INTEGER NOT NULL,
    fiscal_week        INTEGER NOT NULL,
    fiscal_week_start  DATE    NOT NULL,
    is_weekend         BOOLEAN NOT NULL,
    is_month_end       BOOLEAN NOT NULL,
    is_payday          BOOLEAN NOT NULL,
    festival           VARCHAR,
    is_festival        BOOLEAN NOT NULL
);

-- The bridge between the POS key and the operations key, and the only one.
-- It is INCOMPLETE by construction: the store_codes issued at the re-fascia
-- were never sent to operations, so no row here covers them.
CREATE TABLE IF NOT EXISTS dim_store_xref (
    store_code      VARCHAR PRIMARY KEY,
    outlet_id       VARCHAR NOT NULL,
    valid_from      DATE    NOT NULL,
    valid_to        DATE,
    mapping_source  VARCHAR NOT NULL
);

-- Personas. `persona` must be a key in the KPI contract's access_policy;
-- `region` and `store_id` are what its row predicate binds to.
CREATE TABLE IF NOT EXISTS dim_user (
    user_id       VARCHAR PRIMARY KEY,
    display_name  VARCHAR NOT NULL,
    email         VARCHAR NOT NULL,
    persona       VARCHAR NOT NULL,
    region        VARCHAR,
    store_id      VARCHAR
);

-- ===========================================================================
-- Facts
-- ===========================================================================

-- The authoritative revenue table.
--
-- `net_revenue_inr` is the KPI contract's figure. The columns beside it are
-- the ones the contract had to arbitrate between and they are kept so the
-- arbitration is reproducible:
--
--   + b2b_net_revenue_inr   -> the POS ledger's revenue figure
--   + returns_amount_inr    -> marketing's revenue figure
--   transfer_amount_inr     -> excluded by all three definitions
--
-- `store_code` is the POS key as it was stamped on the day. For three
-- stores it changes mid-series and stops resolving through dim_store_xref.
CREATE TABLE IF NOT EXISTS fact_sales_daily (
    store_id               VARCHAR NOT NULL,
    store_code             VARCHAR NOT NULL,
    region                 VARCHAR NOT NULL,
    txn_date               DATE    NOT NULL,
    gross_amount_inr       DOUBLE  NOT NULL,
    returns_amount_inr     DOUBLE  NOT NULL,
    discount_amount_inr    DOUBLE  NOT NULL,
    tax_amount_inr         DOUBLE  NOT NULL,
    net_revenue_inr        DOUBLE  NOT NULL,
    b2b_net_revenue_inr    DOUBLE  NOT NULL,
    transfer_amount_inr    DOUBLE  NOT NULL,
    units                  BIGINT  NOT NULL,
    calendar_expected_inr  DOUBLE  NOT NULL,
    PRIMARY KEY (store_id, txn_date)
);

CREATE TABLE IF NOT EXISTS fact_sales_daily_sku (
    store_id         VARCHAR NOT NULL,
    txn_date         DATE    NOT NULL,
    sku_id           VARCHAR NOT NULL,
    net_revenue_inr  DOUBLE  NOT NULL,
    units            BIGINT  NOT NULL,
    PRIMARY KEY (store_id, txn_date, sku_id)
);

-- Line grain, deterministically sampled. `sample_rate` is carried on every
-- row so nothing sums this table by mistake.
CREATE TABLE IF NOT EXISTS fact_bill_lines (
    bill_id          VARCHAR NOT NULL,
    line_no          INTEGER NOT NULL,
    store_id         VARCHAR NOT NULL,
    store_code       VARCHAR NOT NULL,
    sku_id           VARCHAR NOT NULL,
    txn_ts           TIMESTAMP NOT NULL,
    posted_date      DATE    NOT NULL,
    channel          VARCHAR NOT NULL,
    txn_type         VARCHAR NOT NULL,
    units            INTEGER NOT NULL,
    gross_amount     DOUBLE  NOT NULL,
    discount_amount  DOUBLE  NOT NULL,
    tax_amount       DOUBLE  NOT NULL,
    returns_amount   DOUBLE  NOT NULL,
    sample_rate      DOUBLE  NOT NULL,
    PRIMARY KEY (bill_id, line_no)
);

-- Case #2470: whether each store's POS feed loaded on a given date.
CREATE TABLE IF NOT EXISTS fact_feed_status (
    feed_date      DATE    NOT NULL,
    store_id       VARCHAR NOT NULL,
    region         VARCHAR NOT NULL,
    source_system  VARCHAR NOT NULL,
    status         VARCHAR NOT NULL,
    rows_loaded    INTEGER NOT NULL,
    recoverable    BOOLEAN NOT NULL,
    PRIMARY KEY (feed_date, store_id, source_system)
);

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot (
    store_id           VARCHAR NOT NULL,
    snapshot_date      DATE    NOT NULL,
    sku_id             VARCHAR NOT NULL,
    snapshot_hour_ist  INTEGER NOT NULL,
    on_hand_units      INTEGER NOT NULL,
    is_ranged          BOOLEAN NOT NULL,
    PRIMARY KEY (store_id, snapshot_date, sku_id, snapshot_hour_ist)
);

-- `footfall` is NULL where no counter is installed. A row still exists, so
-- the gap is visible rather than silently absent.
CREATE TABLE IF NOT EXISTS fact_footfall_daily (
    store_id          VARCHAR NOT NULL,
    footfall_date     DATE    NOT NULL,
    footfall          DOUBLE,
    counter_installed BOOLEAN NOT NULL,
    PRIMARY KEY (store_id, footfall_date)
);

-- GRAIN MISMATCH. Booked by ISO week; everything it is compared against is
-- daily. There is no daily column here and none is invented — the
-- allocation is a view, and it is tagged.
CREATE TABLE IF NOT EXISTS fact_marketing_spend_weekly (
    region      VARCHAR NOT NULL,
    campaign    VARCHAR NOT NULL,
    iso_year    INTEGER NOT NULL,
    iso_week    INTEGER NOT NULL,
    week_start  DATE    NOT NULL,
    spend_inr   DOUBLE  NOT NULL,
    PRIMARY KEY (region, campaign, iso_year, iso_week)
);

CREATE TABLE IF NOT EXISTS fact_qcomm_weekly (
    week_start           DATE    NOT NULL,
    week_index           INTEGER NOT NULL,
    scope                VARCHAR NOT NULL,
    dark_stores          INTEGER NOT NULL,
    orders               BIGINT  NOT NULL,
    fulfilment_rate_pct  DOUBLE  NOT NULL,
    PRIMARY KEY (scope, week_start)
);

-- ===========================================================================
-- Documents. No tags, no labels — a classifier has to read the body.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS doc_store_notes (
    note_id      VARCHAR PRIMARY KEY,
    store_id     VARCHAR NOT NULL,
    note_date    DATE    NOT NULL,
    author_role  VARCHAR NOT NULL,
    body         VARCHAR NOT NULL
);

-- `category` exists but is filled in by whoever took the call and is wrong
-- about a quarter of the time. The body is the truth.
CREATE TABLE IF NOT EXISTS doc_tickets (
    ticket_id     VARCHAR PRIMARY KEY,
    store_id      VARCHAR NOT NULL,
    created_date  DATE    NOT NULL,
    category      VARCHAR NOT NULL,
    body          VARCHAR NOT NULL,
    channel       VARCHAR NOT NULL
);

-- The weakest evidence lane there is. No verified-purchase flag, no
-- competitor price: it can corroborate a hypothesis and never test one.
CREATE TABLE IF NOT EXISTS doc_reviews (
    review_id    VARCHAR PRIMARY KEY,
    store_id     VARCHAR NOT NULL,
    review_date  DATE    NOT NULL,
    rating       INTEGER NOT NULL,
    body         VARCHAR NOT NULL,
    platform     VARCHAR NOT NULL
);

-- ===========================================================================
-- External feeds
-- ===========================================================================

-- No price column and no footfall column, because the organisation does not
-- hold those feeds. That absence is what fires trigger T3 on case #2467 and
-- it must not be quietly filled in.
CREATE TABLE IF NOT EXISTS ext_competitor_news (
    news_id         VARCHAR PRIMARY KEY,
    published_date  DATE    NOT NULL,
    competitor      VARCHAR NOT NULL,
    headline        VARCHAR NOT NULL,
    body            VARCHAR NOT NULL,
    source          VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS ext_weather_daily (
    city          VARCHAR NOT NULL,
    region        VARCHAR NOT NULL,
    weather_date  DATE    NOT NULL,
    temp_c        DOUBLE  NOT NULL,
    rain_mm       DOUBLE  NOT NULL,
    PRIMARY KEY (city, weather_date)
);

-- ===========================================================================
-- Case artefacts
-- ===========================================================================

CREATE TABLE IF NOT EXISTS case_registry (
    case_id                VARCHAR PRIMARY KEY,
    kpi                    VARCHAR NOT NULL,
    scope                  VARCHAR NOT NULL,
    grain                  VARCHAR NOT NULL,
    period                 VARCHAR NOT NULL,
    opened_at              TIMESTAMP NOT NULL,
    status                 VARCHAR NOT NULL,
    verdict                VARCHAR,
    reason_code            VARCHAR,
    reason_text            VARCHAR,
    coverage               DOUBLE,
    confidence_raw         DOUBLE,
    confidence_calibrated  DOUBLE,
    triggers_fired         VARCHAR,
    elapsed_ms             DOUBLE,
    closed_at              TIMESTAMP
);

CREATE TABLE IF NOT EXISTS case_hypothesis (
    case_id                 VARCHAR NOT NULL,
    hypothesis_id           VARCHAR NOT NULL,
    label                   VARCHAR NOT NULL,
    description             VARCHAR NOT NULL,
    status                  VARCHAR NOT NULL,
    elimination_reason      VARCHAR,
    verifiable              BOOLEAN NOT NULL,
    required_sources        VARCHAR,
    missing_sources         VARCHAR,
    attributed_share        DOUBLE,
    attributed_evidence_id  VARCHAR,
    residual_held_evidence_id VARCHAR,
    PRIMARY KEY (case_id, hypothesis_id)
);

-- Every number the UI shows resolves to a row here (CLAUDE.md rule 3).
-- `value_numeric` is NULL for model-produced records: rule 1 forbids the
-- model from being the source of a number.
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id      VARCHAR PRIMARY KEY,
    case_id          VARCHAR,
    kind             VARCHAR NOT NULL,
    produced_by      VARCHAR NOT NULL,
    label            VARCHAR NOT NULL,
    value_numeric    DOUBLE,
    value_text       VARCHAR,
    unit             VARCHAR,
    reliability      DOUBLE  NOT NULL,
    source_ref       VARCHAR NOT NULL,
    as_of            TIMESTAMP NOT NULL,
    freshness_hours  DOUBLE  NOT NULL,
    completeness     DOUBLE  NOT NULL,
    assumptions      VARCHAR,
    lineage_json     VARCHAR,
    notes            VARCHAR
);

CREATE TABLE IF NOT EXISTS test_result (
    case_id        VARCHAR NOT NULL,
    hypothesis_id  VARCHAR NOT NULL,
    test_id        INTEGER NOT NULL,
    name           VARCHAR NOT NULL,
    type           VARCHAR NOT NULL,
    passed         BOOLEAN NOT NULL,
    statistic      DOUBLE,
    p_value        DOUBLE,
    weight         DOUBLE,
    detail         VARCHAR NOT NULL,
    evidence_ids   VARCHAR,
    PRIMARY KEY (case_id, hypothesis_id, test_id)
);

CREATE TABLE IF NOT EXISTS recommendation (
    case_id                    VARCHAR PRIMARY KEY,
    action                     VARCHAR NOT NULL,
    playbook_ref               VARCHAR NOT NULL,
    cost_evidence_id           VARCHAR,
    recovery_low_evidence_id   VARCHAR,
    recovery_high_evidence_id  VARCHAR,
    roi_low                    DOUBLE,
    roi_high                   DOUBLE,
    recovery_confidence        VARCHAR NOT NULL,
    sample_size                INTEGER NOT NULL,
    effort                     VARCHAR,
    not_recommended            VARCHAR,
    linked_case_id             VARCHAR
);

-- What the human did with the case. The input to recalibration.
CREATE TABLE IF NOT EXISTS feedback_event (
    feedback_id  VARCHAR PRIMARY KEY,
    case_id      VARCHAR NOT NULL,
    user_id      VARCHAR NOT NULL,
    persona      VARCHAR NOT NULL,
    occurred_at  TIMESTAMP NOT NULL,
    action       VARCHAR NOT NULL,
    comment      VARCHAR
);

CREATE TABLE IF NOT EXISTS case_outcome (
    case_id                 VARCHAR PRIMARY KEY,
    recorded_at             TIMESTAMP NOT NULL,
    action_taken            VARCHAR,
    outcome                 VARCHAR NOT NULL,
    realised_recovery_inr   DOUBLE,
    horizon_weeks           INTEGER,
    was_correct             BOOLEAN,
    note                    VARCHAR
);

-- The closed cases the isotonic map is fitted on.
CREATE TABLE IF NOT EXISTS calibration_ledger (
    entry_id              VARCHAR PRIMARY KEY,
    case_id               VARCHAR NOT NULL,
    case_type             VARCHAR NOT NULL,
    closed_at             TIMESTAMP NOT NULL,
    confidence_raw        DOUBLE  NOT NULL,
    confidence_published  DOUBLE  NOT NULL,
    abstained             BOOLEAN NOT NULL,
    was_correct           BOOLEAN,
    notes                 VARCHAR
);

CREATE TABLE IF NOT EXISTS telemetry_event (
    event_id                     VARCHAR PRIMARY KEY,
    event                        VARCHAR NOT NULL,
    stage                        VARCHAR,
    case_id                      VARCHAR,
    started_at                   TIMESTAMP NOT NULL,
    duration_ms                  DOUBLE,
    model                        VARCHAR,
    input_tokens                 INTEGER,
    output_tokens                INTEGER,
    cache_read_input_tokens      INTEGER,
    cache_creation_input_tokens  INTEGER,
    cost_inr                     DOUBLE,
    mock                         BOOLEAN NOT NULL,
    detail                       VARCHAR
);

-- ===========================================================================
-- What the warehouse says about itself
-- ===========================================================================

-- Written by engine/db.py::execute_governed on every single query. There is
-- no ungoverned path, so there is no query missing from this table.
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id         VARCHAR PRIMARY KEY,
    occurred_at      TIMESTAMP NOT NULL,
    user_id          VARCHAR NOT NULL,
    persona          VARCHAR NOT NULL,
    kpi              VARCHAR NOT NULL,
    statement_hash   VARCHAR NOT NULL,
    row_predicate    VARCHAR NOT NULL,
    rows_returned    BIGINT  NOT NULL,
    rows_filtered    BIGINT  NOT NULL,
    columns_masked   VARCHAR NOT NULL,
    purpose          VARCHAR
);

-- Every disagreement the warehouse found and did not fix. A case that
-- touches an affected scope cites the row rather than working around it.
CREATE TABLE IF NOT EXISTS data_gap_register (
    gap_id           VARCHAR PRIMARY KEY,
    detected_at      TIMESTAMP NOT NULL,
    gap_code         VARCHAR NOT NULL,
    severity         VARCHAR NOT NULL,
    subject          VARCHAR NOT NULL,
    scope            VARCHAR,
    measure          VARCHAR NOT NULL,
    value_numeric    DOUBLE,
    unit             VARCHAR,
    detail           VARCHAR NOT NULL,
    resolution_hint  VARCHAR
);
