import logging
from datetime import datetime
from psycopg2 import sql
from typing import Optional

logger = logging.getLogger(__name__)


def get_etl_run_id(conn, run_date: str):
    sql = """
        SELECT run_id FROM etl_pipeline_runs " 
        WHERE run_date = %s
        ORDER BY created_at DESC
        LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (run_date,))
        result = cur.fetchone()
    if result is None:
        raise ValueError(f"No run_id found for run_date {run_date}")
    
    return result[0]


def create_or_get_ml_pipeline_run(
    conn,
    run_id: str,
    pipeline_name: str,
    run_date: str,
    triggered_by: str = "scheduler",
):
    query = """
    INSERT INTO ml_pipeline_runs (
        run_id,
        pipeline_name,
        run_date,
        status,
        triggered_by,
        created_at
    )
    VALUES (%s, %s, %s, 'running', %s, NOW())
    ON CONFLICT (run_id)
    DO UPDATE SET
        status = 'running',
        triggered_by = EXCLUDED.triggered_by,
        run_date = EXCLUDED.run_date,
        ended_at = NULL,
        error_message = NULL;
    """

    with conn.cursor() as cur:
        cur.execute(query, (run_id, pipeline_name, run_date, triggered_by))

    conn.commit()
    logger.info("ML pipeline run created: run_id=%s pipeline=%s run_date=%s", run_id, pipeline_name, run_date)


def update_ml_pipeline_status(
    conn,
    run_id: str,
    status: str,  
    error_message: Optional[str] = None,
):
    if status not in ("running", "success", "failed"):
        raise ValueError("Invalid status")

    query = """
        UPDATE ml_pipeline_runs
        SET status = %s,
            ended_at = NOW(),
            error_message = %s
        WHERE run_id = %s;
    """

    with conn.cursor() as cur:
        cur.execute(query, (status, error_message, run_id))

    conn.commit()

    logger.info("ML pipeline updated: run_id=%s status=%s",run_id,status)


def start_ml_stage(
    conn,
    ml_run_id: str,
    run_id: str,
    dataset_id: str,
    stage: str,
    mlflow_run_id: str,
):
    query = """
    INSERT INTO ml_runs (
        ml_run_id,
        run_id,
        dataset_id,
        stage,
        mlflow_run_id,
        status,
        created_at
    )
    VALUES (%s, %s, %s, %s, %s, 'running', NOW())
    ON CONFLICT (run_id, dataset_id, stage) 
    DO UPDATE SET
        status = 'running',
        mlflow_run_id = EXCLUDED.mlflow_run_id,
        ended_at = NULL,
        error_message = NULL;
    """
    with conn.cursor() as cur:
        cur.execute(query, (ml_run_id, run_id, dataset_id, stage, mlflow_run_id))

    conn.commit()
    logger.info("ML stage started: run_id=%s stage=%s mlflow_run_id=%s", run_id, stage, mlflow_run_id)


def finish_ml_stage(
    conn,
    run_id: str,
    dataset_id: str,
    stage: str,
    status: str,
    mlflow_run_id: str,
    error_message: Optional[str] = None,
):
    if status not in ("success", "failed"):
        raise ValueError("Invalid status")
    
    sql = """
        UPDATE ml_runs
        SET status = %s, error_message = %s, ended_at = NOW()
        WHERE run_id = %s AND dataset_id = %s AND stage = %s AND mlflow_run_id = %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (status, error_message, run_id, dataset_id, stage, mlflow_run_id))
    conn.commit()
    logger.info("ML stage finished: run_id=%s stage=%s mlflow_run_id=%s status=%s", run_id, stage, mlflow_run_id, status)



def log_dataset(conn, run_id: str, meta: dict):
    query = """
        INSERT INTO ml_dataset (
            dataset_id,
            run_id,
            pipeline_name,
            source_table,

            dataset_start_date,
            dataset_end_date,

            train_path,
            val_path,
            test_path,

            train_row_count,
            val_row_count,
            test_row_count,
            total_row_count,

            feature_version,
            feature_hash,
            schema_hash,

            created_at
        )
        VALUES (
            %(dataset_id)s,
            %(run_id)s,
            %(pipeline_name)s,
            %(source_table)s,

            %(dataset_start_date)s,
            %(dataset_end_date)s,

            %(train_path)s,
            %(val_path)s,
            %(test_path)s,

            %(train_row_count)s,
            %(val_row_count)s,
            %(test_row_count)s,
            %(total_row_count)s,

            %(feature_version)s,
            %(feature_hash)s,
            %(schema_hash)s,

            %(created_at)s
        )
        ON CONFLICT (dataset_id) DO NOTHING
    """

    payload = {
        "dataset_id": meta["dataset_id"],
        "run_id": run_id,
        "pipeline_name": meta["pipeline_name"],
        "source_table": meta["source_table"],

        "dataset_start_date": meta["dataset_start_date"],
        "dataset_end_date": meta["dataset_end_date"],

        "train_path": meta["paths"]["train"],
        "val_path": meta["paths"]["val"],
        "test_path": meta["paths"]["test"],

        "train_row_count": meta["row_counts"]["train"],
        "val_row_count": meta["row_counts"]["val"],
        "test_row_count": meta["row_counts"]["test"],
        "total_row_count": meta["row_counts"]["total"],

        "feature_version": meta["feature_version"],
        "feature_hash": meta["feature_hash"],
        "schema_hash": meta["schema_hash"],

        "created_at": datetime.now(datetime.timezone.utc)
    }

    with conn.cursor() as cur:
        cur.execute(query, payload)

    conn.commit()
    logger.info("Dataset logged with dataset_id=%s run_id=%s", meta["dataset_id"],run_id)