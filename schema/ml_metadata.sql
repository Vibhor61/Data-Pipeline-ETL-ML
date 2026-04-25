CREATE TABLE IF NOT EXISTS ml_pipeline_runs (
    run_id            TEXT PRIMARY KEY,
    pipeline_name     TEXT NOT NULL,
    run_date          DATE NOT NULL,

    status            TEXT NOT NULL CHECK (status in ('running','success','failed')),
    triggered_by      TEXT,

    created_at        TIMESTAMP,
    ended_at          TIMESTAMP,
    error_message     TEXT
);

CREATE INDEX IF NOT EXISTS idx_ml_pipeline_name_run_date ON ml_pipeline_runs(pipeline_name, run_date);

CREATE TABLE IF NOT EXISTS ml_dataset (
    dataset_id            TEXT PRIMARY KEY,

    run_id                TEXT NOT NULL,
    pipeline_name         TEXT NOT NULL,

    source_table          TEXT NOT NULL,

    dataset_start_date    DATE,
    dataset_end_date      DATE,

    train_path            TEXT,
    val_path              TEXT,
    test_path             TEXT,

    train_row_count       INT,
    val_row_count         INT,
    test_row_count        INT,
    total_row_count       INT,

    feature_version       TEXT,
    feature_hash          TEXT,
    schema_hash           TEXT,

    created_at            TIMESTAMP,

    CONSTRAINT fk_dataset_run
        FOREIGN KEY (run_id)
        REFERENCES ml_pipeline_runs(run_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ml_dataset_run_id ON ml_dataset(run_id);

CREATE INDEX IF NOT EXISTS idx_ml_dataset_feature_hash ON ml_dataset(feature_hash);

CREATE TABLE IF NOT EXISTS ml_runs (
    ml_run_id         TEXT,

    run_id            TEXT NOT NULL,
    dataset_id        TEXT NOT NULL,

    stage             TEXT NOT NULL,
    status            TEXT NOT NULL,

    mlflow_run_id     TEXT,
    source_mlflow_run_id    TEXT,
    created_at        TIMESTAMP,
    ended_at          TIMESTAMP,
    error_message     TEXT,

    CONSTRAINT fk_pipeline_run
        FOREIGN KEY (run_id)
        REFERENCES ml_pipeline_runs(run_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_dataset
        FOREIGN KEY (dataset_id)
        REFERENCES ml_dataset(dataset_id)
        ON DELETE CASCADE,

    CONSTRAINT unique_stage_attempt
        UNIQUE (run_id, dataset_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_ml_runs_run_id ON ml_runs(run_id);

CREATE INDEX IF NOT EXISTS idx_ml_runs_dataset_id ON ml_runs(dataset_id);

CREATE INDEX IF NOT EXISTS idx_ml_runs_mlflow_run_id ON ml_runs(mlflow_run_id);
