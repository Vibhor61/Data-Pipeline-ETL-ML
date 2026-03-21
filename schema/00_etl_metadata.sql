CREATE TABLE IF NOT EXISTS etl_pipeline_runs (
    run_id             BIGSERIAL PRIMARY KEY,
    dag_id             TEXT NOT NULL,  
    pipeline_name      TEXT NOT NULL,
    run_date           DATE NOT NULL,  

    status             TEXT NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    triggered_by       TEXT NOT NULL DEFAULT 'scheduler',

    started_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at           TIMESTAMPTZ,
    error_message      TEXT,

    rows_bronze        BIGINT NOT NULL DEFAULT 0 CHECK (rows_bronze >= 0),
    rows_silver        BIGINT NOT NULL DEFAULT 0 CHECK (rows_silver >= 0),
    rows_gold          BIGINT NOT NULL DEFAULT 0 CHECK (rows_gold >= 0),

    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (pipeline_name, run_date)
);

CREATE INDEX IF NOT EXISTS idx_etl_pipeline_runs_name_date
    ON etl_pipeline_runs(pipeline_name, run_date);

CREATE INDEX IF NOT EXISTS idx_etl_pipeline_runs_status
    ON etl_pipeline_runs(status);


CREATE TABLE IF NOT EXISTS etl_pipeline_steps (
    step_run_id        BIGSERIAL PRIMARY KEY,
    run_id             BIGINT NOT NULL REFERENCES etl_pipeline_runs(run_id) ON DELETE CASCADE,

    step_name          TEXT NOT NULL,
    status             TEXT NOT NULL CHECK (status IN ('running', 'success', 'failed', 'skipped')),

    started_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at           TIMESTAMPTZ,

    input_rows         BIGINT NOT NULL DEFAULT 0 CHECK (input_rows >= 0),
    output_rows        BIGINT NOT NULL DEFAULT 0 CHECK (output_rows >= 0),

    error_message      TEXT,

    UNIQUE (run_id, step_name)
);

CREATE INDEX IF NOT EXISTS idx_etl_pipeline_steps_run_id
    ON etl_pipeline_steps(run_id);

CREATE INDEX IF NOT EXISTS idx_etl_pipeline_steps_status
    ON etl_pipeline_steps(status);