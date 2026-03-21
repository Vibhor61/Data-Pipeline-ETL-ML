CREATE TABLE IF NOT EXISTS silver_table (
    run_date           DATE        NOT NULL,   
    _processed_time    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    _pipeline_version  TEXT        NOT NULL,

    store_id           TEXT        NOT NULL,
    item_id            TEXT        NOT NULL,
    d                  TEXT        NOT NULL,
    date               DATE        NOT NULL,

    dept_id            TEXT        NOT NULL,
    cat_id             TEXT        NOT NULL,
    state_id           TEXT        NOT NULL,

    sales              INTEGER     NOT NULL CHECK (sales >= 0),
    sell_price         NUMERIC(10,4) CHECK (sell_price IS NULL OR sell_price >= 0),

    wm_yr_wk           INTEGER     NOT NULL,
    weekday            TEXT,        
    wday               SMALLINT    CHECK (wday IS NULL OR wday BETWEEN 1 AND 7),
    month              SMALLINT    CHECK (month IS NULL OR month BETWEEN 1 AND 12),
    year               SMALLINT,

    event_name_1       TEXT,
    event_type_1       TEXT,
    event_name_2       TEXT,
    event_type_2       TEXT,

    snap_CA            SMALLINT    NOT NULL CHECK (snap_CA IN (0,1)),
    snap_TX            SMALLINT    NOT NULL CHECK (snap_TX IN (0,1)),
    snap_WI            SMALLINT    NOT NULL CHECK (snap_WI IN (0,1)),

    PRIMARY KEY (run_date, store_id, item_id, d)
);


CREATE INDEX IF NOT EXISTS idx_silver_table_date   ON silver_table(date);
CREATE INDEX IF NOT EXISTS idx_silver_table_item   ON silver_table(item_id);
CREATE INDEX IF NOT EXISTS idx_silver_table_store  ON silver_table(store_id);