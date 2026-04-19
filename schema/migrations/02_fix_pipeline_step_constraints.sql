ALTER TABLE etl_pipeline_steps
ADD COLUMN IF NOT EXISTS dag_id TEXT;

UPDATE etl_pipeline_steps
SET dag_id = 'retail_etl_dag'
WHERE dag_id IS NULL;

ALTER TABLE etl_pipeline_steps
DROP CONSTRAINT IF EXISTS etl_pipeline_steps_run_id_step_name_key;

ALTER TABLE etl_pipeline_steps
DROP CONSTRAINT IF EXISTS uq_run_dag_step;

ALTER TABLE etl_pipeline_steps
ALTER COLUMN dag_id SET NOT NULL;

ALTER TABLE etl_pipeline_steps
ADD CONSTRAINT uq_run_dag_step
UNIQUE (run_id, dag_id, step_name);