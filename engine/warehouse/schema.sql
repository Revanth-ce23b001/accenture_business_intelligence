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
    -- The two columns the access policy actually masks.
    --   staff_cost  monthly payroll. Compensation, so a regional head
    --               does not see it across their own estate.
    --   staff_id    roster key of the person accountable for the store.
    --               It identifies a human being.
    -- Both are real columns on purpose: a mask over a column that does
    -- not exist never fires, and the test that asserts it is absent from
    -- an LLM payload would pass for the wrong reason.
    staff_cost                BIGINT  NOT NULL,
    staff_id                  VARCHAR NOT NULL,
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
    is_festival        BOOLEAN NOT NULL,
    -- Meridian's own scheduled sales. A promo window is a calendar event,
    -- not a spend level: Gate 2's baseline may use this and must never use
    -- marketing spend, or a spend cut would hide inside "calendar-expected".
    promo_window       VARCHAR,
    is_promo_window    BOOLEAN NOT NULL
);

-- Festival membership, properly. `dim_calendar.festival` holds one label
-- per day and festivals OVERLAP — Onam with Ganesh Chaturthi in South,
-- Navratri with Durga Puja in East — so a single label silently drops the
-- second one and a calendar model built on it cannot fit either.
--
-- `day_index` over `phase_days` says where in the ramp a day falls, and
-- `phase` separates the window from the seven days of suppressed demand
-- after it. Gate 2 needs both: a festival is not a switch.
CREATE TABLE IF NOT EXISTS dim_festival_window (
    date        DATE    NOT NULL,
    festival    VARCHAR NOT NULL,
    occurrence  VARCHAR NOT NULL,
    phase       VARCHAR NOT NULL,
    day_index   INTEGER NOT NULL,
    phase_days  INTEGER NOT NULL,
    PRIMARY KEY (date, festival, phase)
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

-- Whether each store's POS feed loaded, and how many rows it carried.
-- One row per store per day for the whole window: Gate 1's row-count check
-- reads its eight-week median off this table, and a feed that failed is
-- only visible as a failure against a normal week. Case #2470 is 25 West
-- stores at zero on 12 Nov 2025.
CREATE TABLE IF NOT EXISTS fact_feed_status (
    feed_date      DATE    NOT NULL,
    store_id       VARCHAR NOT NULL,
    region         VARCHAR NOT NULL,
    source_system  VARCHAR NOT NULL,
    status         VARCHAR NOT NULL,
    rows_loaded    BIGINT  NOT NULL,
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

-- Periods finance has DECLARED open for restatement. Gate 1's fifth check
-- reads this rather than inferring a restatement from how young the data
-- is: late-arriving returns make a recent period incomplete, and
-- incompleteness is not a restatement. Somebody has to say the published
-- figure is wrong and will be reissued.
CREATE TABLE IF NOT EXISTS restatement_register (
    restatement_id  VARCHAR PRIMARY KEY,
    kpi             VARCHAR NOT NULL,
    scope           VARCHAR NOT NULL,
    grain           VARCHAR NOT NULL,
    period          VARCHAR NOT NULL,
    flagged_at      DATE    NOT NULL,
    resolved_at     DATE,
    status          VARCHAR NOT NULL,
    reason          VARCHAR NOT NULL
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
    -- How many times its own materiality limit the qualified residual is.
    -- Written by QUALIFY when the case opens. It is what ranks one case
    -- against another, and what tells restraint whether a later movement
    -- on the same scope is an escalation or a repeat.
    materiality_multiple   DOUBLE,
    coverage               DOUBLE,
    confidence_raw         DOUBLE,
    confidence_calibrated  DOUBLE,
    triggers_fired         VARCHAR,
    elapsed_ms             DOUBLE,
    closed_at              TIMESTAMP,
    -- The case this one was opened FROM, when a playbook's
    -- `linked_case_template` raised it: the cause of the cause. Written by
    -- engine/recommend/linked.py. NULL for a case a movement opened.
    linked_from_case_id    VARCHAR
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
--
-- Four columns are NOT NULL because a row missing any of them is a number
-- that cannot be chased:
--
--   source_system   where it came from
--   method          how it was made
--   source_as_of    how current the DATA is. Not when it was read —
--                   `retrieved_at` is that, and it is kept separately so
--                   the two can never be confused. A stale feed shown with
--                   today's date beside it reads as fresh.
--   lineage_json    what it was derived from, in order, with any SQL kept
--                   VERBATIM rather than hashed. audit_log stores a hash
--                   because it is a different table with different access
--                   rules; the statement travels with the number here.
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
    source_system    VARCHAR NOT NULL,
    method           VARCHAR NOT NULL,
    source_ref       VARCHAR NOT NULL,
    source_as_of     TIMESTAMP NOT NULL,
    retrieved_at     TIMESTAMP,
    freshness_hours  DOUBLE  NOT NULL,
    completeness     DOUBLE  NOT NULL,
    assumptions      VARCHAR,
    lineage_json     VARCHAR NOT NULL,
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
-- WHAT A READER SAID, AT ONE OF THREE LEVELS.
--
-- `target_kind` is which: 'verdict', 'driver' or 'action'. A reader can
-- accept a verdict and reject one driver inside it, and a table that only
-- recorded the verdict could not represent that — which matters, because
-- the three levels update three different things:
--
--   verdict  -> the isotonic calibration map
--   driver   -> the causal graph priors, per hypothesis per KPI
--   action   -> the playbook recovery curves
--
-- `target_id` names the driver or the playbook; NULL at verdict level.
-- `reason_code` is a closed set from learning.yaml, because free text is
-- a comment and a comment cannot be counted. Both travel.
--
-- The verdict and confidence AS PUBLISHED are stamped on the row. The case
-- can be re-run and reach a different answer; what the reader was looking
-- at when they clicked cannot be reconstructed afterwards, and the
-- calibration entry derived from this row depends on it.
CREATE TABLE IF NOT EXISTS feedback_event (
    feedback_id  VARCHAR PRIMARY KEY,
    case_id      VARCHAR NOT NULL,
    user_id      VARCHAR NOT NULL,
    persona      VARCHAR NOT NULL,
    occurred_at  TIMESTAMP NOT NULL,
    action       VARCHAR NOT NULL,
    comment      VARCHAR
);
ALTER TABLE feedback_event ADD COLUMN IF NOT EXISTS target_kind VARCHAR;
ALTER TABLE feedback_event ADD COLUMN IF NOT EXISTS target_id VARCHAR;
ALTER TABLE feedback_event ADD COLUMN IF NOT EXISTS reason_code VARCHAR;
ALTER TABLE feedback_event ADD COLUMN IF NOT EXISTS kpi VARCHAR;
ALTER TABLE feedback_event ADD COLUMN IF NOT EXISTS verdict_at_feedback VARCHAR;
ALTER TABLE feedback_event ADD COLUMN IF NOT EXISTS confidence_at_feedback DOUBLE;
ALTER TABLE feedback_event ADD COLUMN IF NOT EXISTS confidence_raw_at_feedback DOUBLE;
ALTER TABLE feedback_event ADD COLUMN IF NOT EXISTS case_type VARCHAR;

-- Bayesian counts behind every learned prior. One row per hypothesis per
-- KPI, because "stock-outs are usually the answer" is a claim about a KPI
-- and not about the business: stock-outs explain availability movements
-- far more often than they explain average selling price, and one pooled
-- count would blur the two into a number true of neither.
--
-- The counts are the state. The effective prior is DERIVED from them and
-- the declared prior in causal_graph.yaml, by `learning.yaml -> priors`,
-- and is never stored — a stored posterior is a number that goes stale
-- the moment either input changes.
CREATE TABLE IF NOT EXISTS hypothesis_prior (
    kpi          VARCHAR NOT NULL,
    hypothesis   VARCHAR NOT NULL,
    confirmed    BIGINT  NOT NULL DEFAULT 0,
    rejected     BIGINT  NOT NULL DEFAULT 0,
    first_seen_at TIMESTAMP NOT NULL,
    updated_at   TIMESTAMP NOT NULL,
    PRIMARY KEY (kpi, hypothesis)
);

-- What an action actually recovered, against what was promised.
--
-- Both halves are kept. Comparing realised against expected needs the
-- expectation as it was published, not as the curve would compute it
-- today — a curve that has since moved would make every past case look
-- better or worse than it was called at the time.
CREATE TABLE IF NOT EXISTS recovery_realisation (
    realisation_id     VARCHAR PRIMARY KEY,
    case_id            VARCHAR NOT NULL,
    playbook           VARCHAR NOT NULL,
    curve_ref          VARCHAR NOT NULL,
    attributable_inr   DOUBLE  NOT NULL,
    expected_low_inr   DOUBLE  NOT NULL,
    expected_high_inr  DOUBLE  NOT NULL,
    realised_inr       DOUBLE  NOT NULL,
    horizon_weeks      INTEGER NOT NULL,
    -- realised / attributable. What the curve is quoted in, so an update
    -- compares like with like.
    realised_share     DOUBLE  NOT NULL,
    -- True when the share is above `implausible_share_above`: recorded,
    -- and excluded from any curve update. Recovering 300% of an attributed
    -- loss means something else moved.
    implausible        BOOLEAN NOT NULL DEFAULT FALSE,
    recorded_at        TIMESTAMP NOT NULL
);

-- ONE ROW PER CASE PER HORIZON. D+14 asks whether the CAUSE held up;
-- D+56 asks whether the MONEY came back. They are different questions
-- with different answers — a cause can be right and the recovery still
-- fail — so they are different rows and the key is (case_id, horizon_days).
--
-- `was_correct` here is what actually happened, and it supersedes the
-- calibration entry the reader's feedback wrote. The two disagree
-- sometimes: somebody accepts a case that later fails to recover. Both
-- are kept, because "what we were told" and "what happened" are both
-- worth knowing and only the second should calibrate anything.
--
-- Recreated by engine/warehouse/migrate.py on a warehouse that predates
-- the composite key. Safe because the table never had a writer before
-- P16 and is provably empty; the migration checks that and refuses if it
-- is not.
CREATE TABLE IF NOT EXISTS case_outcome (
    case_id                 VARCHAR NOT NULL,
    horizon_days            INTEGER NOT NULL,
    horizon_name            VARCHAR NOT NULL,
    recorded_at             TIMESTAMP NOT NULL,
    action_taken            VARCHAR,
    outcome                 VARCHAR NOT NULL,
    -- D+14's answer. NULL at a horizon that does not measure it.
    cause_confirmed         BOOLEAN,
    -- D+56's answer, and the money behind it.
    recovered               BOOLEAN,
    realised_recovery_inr   DOUBLE,
    expected_low_inr        DOUBLE,
    expected_high_inr       DOUBLE,
    horizon_weeks           INTEGER,
    was_correct             BOOLEAN,
    note                    VARCHAR,
    PRIMARY KEY (case_id, horizon_days)
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

-- ONE ROW PER REQUEST. The unit of account.
--
-- `telemetry_event` below is the per-step detail — a stage boundary, a
-- model call. This is the roll-up, and it is the table the two published
-- performance claims are measured against: cost per case under INR 6,
-- P95 latency under 9 s warm (CLAUDE.md §"Definition of done").
--
-- THE FIVE STAGE LATENCIES ARE NOT NULL BY DESIGN. "Every request writes
-- a complete row with all five stage latencies" is the accept criterion,
-- and a criterion the database enforces cannot be forgotten by a caller.
-- A request killed at Gate 1 never enters ADJUDICATE; that stage records
-- 0.0 ms and `stages_entered` names the ones that actually ran, so a
-- zero is never ambiguous between "instant" and "never happened".
--
-- `elapsed_ms` here is the request. `case_registry.elapsed_ms` is the
-- case open -> verdict measurement, written by the same recorder — that
-- is the number the case header shows, and it is what replaced the
-- asserted "11 minutes" (resolved defect 6).
CREATE TABLE IF NOT EXISTS telemetry_request (
    request_id                   VARCHAR PRIMARY KEY,
    case_id                      VARCHAR,
    user_id                      VARCHAR NOT NULL,
    persona                      VARCHAR NOT NULL,
    question                     VARCHAR,
    verdict                      VARCHAR,
    confidence                   DOUBLE,

    started_at                   TIMESTAMP NOT NULL,
    total_latency_ms             DOUBLE  NOT NULL,
    latency_validate_ms          DOUBLE  NOT NULL,
    latency_qualify_ms           DOUBLE  NOT NULL,
    latency_gather_ms            DOUBLE  NOT NULL,
    latency_adjudicate_ms        DOUBLE  NOT NULL,
    latency_verdict_ms           DOUBLE  NOT NULL,
    -- Which of the five were actually entered, in order, comma-separated.
    stages_entered               VARCHAR NOT NULL,
    -- False for the first requests of a process, which pay for loading the
    -- semantic layer and opening the warehouse. The percentile is computed
    -- over warm rows, and this column is why that is checkable rather than
    -- a claim in a footnote.
    warm                         BOOLEAN NOT NULL,

    -- Which statistical methods ran, comma-separated. Not a count: the
    -- point is WHICH, so a case that skipped the DiD is visible as such.
    analytical_methods_executed  VARCHAR NOT NULL,

    llm_calls                    INTEGER NOT NULL,
    -- One model id per call, in call order, comma-separated. Proves the
    -- routing in llm/provider.py did what it says.
    model_per_call               VARCHAR NOT NULL,
    input_tokens                 BIGINT  NOT NULL,
    output_tokens                BIGINT  NOT NULL,
    cache_read_input_tokens      BIGINT  NOT NULL,
    cache_creation_input_tokens  BIGINT  NOT NULL,
    estimated_cost_inr           DOUBLE  NOT NULL,
    -- True when a call reported no usage and its tokens were estimated
    -- from the text. Always true offline, always false on a live run. A
    -- cost figure that does not say which it is cannot be defended.
    cost_estimated               BOOLEAN NOT NULL,
    -- Content-hash cache in `llm_cache`, not the prompt cache above.
    cache_hits                   BIGINT  NOT NULL,
    cache_misses                 BIGINT  NOT NULL,

    -- Rows the policy evaluated, rows it withheld, rows that crossed the
    -- trust boundary into a prompt. The third is the one an auditor asks
    -- about; `audit_log` holds the same figure per statement.
    rows_scanned                 BIGINT  NOT NULL,
    rows_filtered_by_policy      BIGINT  NOT NULL,
    rows_released_to_llm         BIGINT  NOT NULL,

    grounding_claims_checked     INTEGER NOT NULL,
    grounding_claims_stripped    INTEGER NOT NULL,

    mock                         BOOLEAN NOT NULL
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
    -- How many rows crossed the trust boundary into a model prompt.
    --
    -- A READ writes 0: at the moment execute_governed returns, nothing
    -- has gone anywhere. A RELEASE is a separate access event with its
    -- own row and its own count, because handing rows to a third party
    -- is a different thing from reading them and the log should not have
    -- to be interpreted to tell the two apart.
    rows_released_to_llm BIGINT NOT NULL DEFAULT 0,
    purpose          VARCHAR
);

-- What each KPI's definition hashed to, the last time Gate 1 ran it.
-- A movement computed under a new definition is not comparable with the
-- period before it, and the difference looks exactly like a real change.
-- `first_seen_at` and `last_seen_at` bracket the window in which a hash was
-- the live definition, so a case can be read back against the definition it
-- was actually decided under.
CREATE TABLE IF NOT EXISTS kpi_definition_log (
    kpi            VARCHAR NOT NULL,
    formula_hash   VARCHAR NOT NULL,
    kpi_version    INTEGER NOT NULL,
    hashed_fields  VARCHAR NOT NULL,
    first_seen_at  TIMESTAMP NOT NULL,
    last_seen_at   TIMESTAMP NOT NULL,
    PRIMARY KEY (kpi, formula_hash)
);

-- Model responses, keyed by a hash of the DOCUMENT rather than the batch.
--
-- The corpora repeat heavily — "Routine day, footfall normal, no issues to
-- report" is one document however many stores filed it — so hashing the
-- document is what makes the cache worth having. Batching is how the calls
-- are made efficient; hashing is how most of them stop being made at all.
--
-- The key covers everything that could change the answer: the text, the
-- candidate tag set, the prompt version and the model. Change any of them
-- and the old answer is correctly a miss.
CREATE TABLE IF NOT EXISTS llm_cache (
    content_hash    VARCHAR PRIMARY KEY,
    task            VARCHAR NOT NULL,
    model           VARCHAR NOT NULL,
    prompt_version  INTEGER NOT NULL,
    response_json   VARCHAR NOT NULL,
    created_at      TIMESTAMP NOT NULL
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
