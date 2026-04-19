ALTER TABLE etl_pipeline_runs
DROP CONSTRAINT IF EXISTS etl_pipeline_runs_pipeline_name_run_date_key;

ALTER TABLE etl_pipeline_runs
ADD CONSTRAINT uq_pipeline_dag_run
UNIQUE (pipeline_name, dag_id, run_date);