CREATE TABLE IF NOT EXISTS bronze_sales (
    item_id            TEXT NOT NULL,
    store_id           TEXT NOT NULL,
    dept_id            TEXT NOT NULL,
    cat_id             TEXT NOT NULL,
    state_id           TEXT NOT NULL,
    d                  TEXT NOT NULL,
    sales              INTEGER NOT NULL CHECK (sales >= 0),

    run_date           DATE NOT NULL,
    _pipeline_version  TEXT NOT NULL,
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (item_id, store_id, run_date)
);

CREATE INDEX IF NOT EXISTS idx_bronze_sales_run_date
    ON bronze_sales(run_date);

CREATE INDEX IF NOT EXISTS idx_bronze_sales_item_store
    ON bronze_sales(item_id, store_id);


CREATE TABLE IF NOT EXISTS sell_prices (
    store_id           TEXT NOT NULL,
    item_id            TEXT NOT NULL,
    wm_yr_wk           INTEGER NOT NULL,
    sell_price         NUMERIC(10,4),
    CHECK (sell_price IS NULL OR sell_price >= 0)
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (store_id, item_id, wm_yr_wk),
);


CREATE TABLE IF NOT EXISTS calendar (
    d                  TEXT PRIMARY KEY,
    date               DATE NOT NULL,
    wm_yr_wk           INTEGER NOT NULL,
    weekday            TEXT NOT NULL,
    wday               INTEGER NOT NULL CHECK (wday BETWEEN 1 AND 7),
    month              INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    year               INTEGER NOT NULL,

    event_name_1       TEXT,
    event_type_1       TEXT,
    event_name_2       TEXT,
    event_type_2       TEXT,

    snap_CA            INTEGER NOT NULL CHECK (snap_CA IN (0,1)),
    snap_TX            INTEGER NOT NULL CHECK (snap_TX IN (0,1)),
    snap_WI            INTEGER NOT NULL CHECK (snap_WI IN (0,1)),

    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);