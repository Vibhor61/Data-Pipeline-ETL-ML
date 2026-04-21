import logging
from datetime import datetime
from psycopg2 import sql
from typing import Optional

logger = logging.getLogger(__name__)


def get_run_id(conn, run_date: str):
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
    triggered_by: str = "scheduler",
):
    query = """
    INSERT INTO ml_pipeline_runs (
        run_id,
        pipeline_name,
        status,
        triggered_by,
        created_at
    )
    VALUES (%s, %s, 'running', %s, NOW())
    ON CONFLICT (run_id)
    DO UPDATE SET
        status = 'running',
        triggered_by = EXCLUDED.triggered_by
        ended_at = NULL,
        error_message = NULL;
    """

    with conn.cursor() as cur:
        cur.execute(query, (run_id, pipeline_name, triggered_by))

    conn.commit()
    logger.info("ML pipeline run created: run_id=%s pipeline=%s triggered_by=%s", run_id, pipeline_name, triggered_by)


def update_ml_run_status(
    conn,
    run_id: str,
    status: str,  
    error_message: Optional[str] = None,
):
    if status not in ("running", "success", "failed"):
        raise ValueError("Invalid status")
 
    if status == "running":
        sql_q = """
            UPDATE ml_pipeline_runs
            SET status = %s,
                ended_at = NULL
            WHERE run_id = %s;
        """
        params = (status, run_id)
    else:
        sql_q = """
            UPDATE ml_pipeline_runs
            SET status = %s,
                ended_at = NOW(),
                error_message = %s
            WHERE run_id = %s;
        """
        params = (status, error_message, run_id)
 
    with conn.cursor() as cur:
        cur.execute(sql_q, params)
 
    conn.commit()
    logger.info("ML run %s updated to status=%s", run_id, status)


def start_ml_stage(
    conn,
    run_id: str,
    dataset_id: str,
    stage: str,
    mlflow_run_id: str,
):
    ml_run_id = f"{run_id}_{stage}"
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
        ml_run_id = EXCLUDED.ml_run_id,
        status = 'running',
        stage = EXCLUDED.stage,
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
    query = sql.SQL("""
        INSERT INTO ml_dataset (
            dataset_id,
            run_id,
            dataset_type,
            source_table,
            time_start,
            time_end,
            feature_version,
            feature_hash,
            row_count,
            schema_hash,
            created_at
        )
        VALUES (
            %(dataset_id)s,
            %(run_id)s,
            %(dataset_type)s,
            %(source_table)s,
            %(time_start)s,
            %(time_end)s,
            %(feature_version)s,
            %(feature_hash)s,
            %(row_count)s,
            %(schema_hash)s,
            %(created_at)s
        )
        ON CONFLICT (dataset_id) DO NOTHING
    """)

    payload = {
        "dataset_id": meta["dataset_id"],
        "run_id": run_id,
        "dataset_type": meta["dataset_type"],
        "source_table": meta["source_table"],
        "time_start": meta["time_start"],
        "time_end": meta["time_end"],
        "feature_version": meta["feature_version"],
        "feature_hash": meta["feature_hash"],
        "row_count": meta["row_count"],
        "schema_hash": meta["schema_hash"],
        "created_at": datetime.utcnow()
    }

    with conn.cursor() as cur:
        cur.execute(query, payload)

    conn.commit()

    logger.info("Dataset logged with dataset_id=%s run_id=%s", meta["dataset_id"],run_id)