BEGIN;

-- 1. ml_pipeline_runs migration
ALTER TABLE ml_pipeline_runs
ADD COLUMN IF NOT EXISTS run_date DATE,
ADD COLUMN IF NOT EXISTS error_message TEXT;

ALTER TABLE ml_pipeline_runs
ALTER COLUMN pipeline_name SET NOT NULL,
ALTER COLUMN status SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ml_pipeline_name_run_date ON ml_pipeline_runs(pipeline_name, run_date);

-- 2. ml_dataset migration
ALTER TABLE ml_runs DROP CONSTRAINT IF EXISTS fk_dataset;
ALTER TABLE ml_dataset DROP CONSTRAINT IF EXISTS fk_dataset_run;

ALTER TABLE ml_dataset
ADD COLUMN IF NOT EXISTS pipeline_name TEXT,
ADD COLUMN IF NOT EXISTS dataset_start_date DATE,
ADD COLUMN IF NOT EXISTS dataset_end_date DATE,

ADD COLUMN IF NOT EXISTS train_path TEXT,
ADD COLUMN IF NOT EXISTS val_path TEXT,
ADD COLUMN IF NOT EXISTS test_path TEXT,

ADD COLUMN IF NOT EXISTS train_row_count INT,
ADD COLUMN IF NOT EXISTS val_row_count INT,
ADD COLUMN IF NOT EXISTS test_row_count INT,
ADD COLUMN IF NOT EXISTS total_row_count INT;

ALTER TABLE ml_dataset
DROP COLUMN IF EXISTS dataset_type,
DROP COLUMN IF EXISTS time_start,
DROP COLUMN IF EXISTS time_end,
DROP COLUMN IF EXISTS row_count;

ALTER TABLE ml_dataset
ALTER COLUMN run_id SET NOT NULL,
ALTER COLUMN pipeline_name SET NOT NULL,
ALTER COLUMN source_table SET NOT NULL;

ALTER TABLE ml_dataset ADD CONSTRAINT fk_dataset_run FOREIGN KEY (run_id) REFERENCES ml_pipeline_runs(run_id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_ml_dataset_run_id ON ml_dataset(run_id);

CREATE INDEX IF NOT EXISTS idx_ml_dataset_feature_hash ON ml_dataset(feature_hash);

-- 3. ml_runs migration
ALTER TABLE ml_runs
ADD COLUMN IF NOT EXISTS status TEXT,
ADD COLUMN IF NOT EXISTS ended_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS error_message TEXT,

ALTER TABLE ml_runs
DROP CONSTRAINT IF EXISTS unique_stage_runs;

ALTER TABLE ml_runs
DROP COLUMN IF EXISTS metric_name,
DROP COLUMN IF EXISTS metric_value,
DROP COLUMN IF EXISTS slice_key,
DROP COLUMN IF EXISTS slice_value,
DROP COLUMN IF EXISTS prediction_count;

ALTER TABLE ml_runs
ALTER COLUMN run_id SET NOT NULL,
ALTER COLUMN dataset_id SET NOT NULL,
ALTER COLUMN stage SET NOT NULL,
ALTER COLUMN status SET NOT NULL;

ALTER TABLE ml_runs ADD CONSTRAINT unique_stage_runs UNIQUE (run_id, dataset_id, stage);

ALTER TABLE ml_runs ADD CONSTRAINT fk_dataset FOREIGN KEY (dataset_id) REFERENCES ml_dataset(dataset_id) ON DELETE CASCADE;
 
CREATE INDEX IF NOT EXISTS idx_ml_runs_run_idON ml_runs(run_id);

CREATE INDEX IF NOT EXISTS idx_ml_runs_dataset_id ON ml_runs(dataset_id);

CREATE INDEX IF NOT EXISTS idx_ml_runs_mlflow_run_id ON ml_runs(mlflow_run_id);

COMMIT;