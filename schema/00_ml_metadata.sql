CREATE TABLE IF NOT EXISTS ml_pipeline_runs(
    run_id              TEXT PRIMARY KEY,
    run_date            DATE NOT NULL,
    pipeline_name       TEXT,

    status              TEXT,
    triggered_by        TEXT,

    created_at          TIMESTAMP,
    ended_at            TIMESTAMP
)


CREATE TABLE IF NOT EXISTS ml_dataset(
    dataset_id           TEXT PRIMARY KEY,
    run_id               TEXT,
    run_date             DATE,
    
    dataset_type         TEXT,

    source_table         TEXT,     
    feature_query_hash   TEXT,     
    feature_version      TEXT,    

    row_count            INT,
    schema_hash          TEXT,

    created_at           TIMESTAMP
);


CREATE TABLE IF NOT EXISTS ml_runs(
    ml_run_id           TEXT PRIMARY KEY,

    run_id              TEXT NOT NULL,
    dataset_id          TEXT NOT NULL,
    stage               TEXT NOT NULL,

    model_name          TEXT,
    mlflow_run_id       TEXT,

    metric_name         TEXT,
    metric_value        DOUBLE PRECISION,

    slice_key           TEXT,
    slice_value         TEXT,

    prediction_count    INT,

    created_at          TIMESTAMP,

    CONSTRAINT fk_pipeline_run
        FOREIGN KEY (run_id)
        REFERENCES ml_pipeline_runs(run_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_dataset
        FOREIGN KEY (dataset_id)
        REFERENCES ml_dataset(dataset_id)
        ON DELETE CASCADE
)

