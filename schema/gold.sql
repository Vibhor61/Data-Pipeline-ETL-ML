CREATE TABLE IF NOT EXISTS gold_table (
    item_id            TEXT NOT NULL,
    store_id           TEXT NOT NULL,
    dept_id            TEXT NOT NULL,
    cat_id             TEXT NOT NULL,
    state_id           TEXT NOT NULL,
    d                  TEXT NOT NULL,

    sales              INTEGER NOT NULL CHECK (sales >= 0),
    sell_price         NUMERIC(10,4),
    run_date           DATE NOT NULL,  

    _processed_time    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    _pipeline_version  TEXT NOT NULL,

    sales_lag_1        INTEGER,
    sales_lag_3        INTEGER,
    sales_lag_7        INTEGER,
    sales_lag_14       INTEGER,
    sales_lag_28       INTEGER,

    sales_roll_mean_7  NUMERIC(10,4),
    sales_roll_mean_14 NUMERIC(10,4),
    sales_roll_mean_28 NUMERIC(10,4),
    sales_roll_std_7   NUMERIC(10,4),
    sales_roll_std_14  NUMERIC(10,4),
    sales_roll_std_28  NUMERIC(10,4),

    is_cold_start      BOOLEAN NOT NULL DEFAULT FALSE,
    is_weekend         INTEGER,
    quarter            INTEGER,
    month              SMALLINT,
    wday               SMALLINT CHECK (wday BETWEEN 1 AND 7),
    weekday            TEXT,
    year               SMALLINT,

    event_name_1       TEXT,
    event_type_1       TEXT,
    event_name_2       TEXT,
    event_type_2       TEXT,

    snap_CA            SMALLINT NOT NULL CHECK (snap_CA IN (0,1)),
    snap_TX            SMALLINT NOT NULL CHECK (snap_TX IN (0,1)),
    snap_WI            SMALLINT NOT NULL CHECK (snap_WI IN (0,1)),
    wm_yr_wk           INTEGER  NOT NULL,

    PRIMARY KEY (item_id, store_id, d, run_date)
);

CREATE INDEX IF NOT EXISTS idx_gold_table_run_date
    ON gold_table(run_date);

CREATE INDEX IF NOT EXISTS idx_gold_table_item_store
    ON gold_table(item_id, store_id);

CREATE INDEX IF NOT EXISTS idx_gold_table_wmyrwk
    ON gold_table(wm_yr_wk);