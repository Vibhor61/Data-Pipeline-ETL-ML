import logging
from typing import Optional, Dict
import uuid
from ML.data_loader import DataLoader
logger = logging.getLogger(__name__)


import logging
import uuid

logger = logging.getLogger(__name__)


def create_or_get_ml_pipeline_run(
    conn,
    run_id: str,
    pipeline_name: str,
    triggered_by: str = "airflow",
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
        triggered_by = EXCLUDED.triggered_by;
    """

    with conn.cursor() as cur:
        cur.execute(query, (run_id, pipeline_name, triggered_by))

    conn.commit()
    logger.info("Pipeline run created: %s", run_id)


from datetime import datetime
from psycopg2 import sql


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


def log_training_run(
    conn,
    run_id: str,
    dataset_id: str,
    model_name: str,
    mlflow_run_id: str
):
    
    query = """
        INSERT INTO ml_runs (
            ml_run_id,
            run_id,
            dataset_id,
            stage,
            model_name,
            mlflow_run_id,
            created_at
        )
        VALUES (%s, %s, %s, 'train', %s, %s, NOW());
    """
    ml_run_id = f"{run_id}_train"
    with conn.cursor() as cur:
        cur.execute(
            query,
            (ml_run_id, run_id, dataset_id, model_name, mlflow_run_id),
        )

    conn.commit()
    logger.info("Training logged: model=%s with mlflow_run_id=%s", model_name, mlflow_run_id)


def log_prediction_run(
    conn,
    run_id: str,
    dataset_id: str,
    mlflow_run_id: str,
    model_name: str,
    prediction_count: int
):

    query = """
        INSERT INTO ml_runs (
            ml_run_id,
            run_id,
            dataset_id,
            stage,
            model_name,
            mlflow_run_id,
            prediction_count,
            created_at
        )
        VALUES (%s, %s, %s, 'predict', %s, %s, %s, NOW());
    """

    ml_run_id = f"{mlflow_run_id}_predict"

    with conn.cursor() as cur:
        cur.execute(
            query,
            (
                ml_run_id,
                run_id,
                dataset_id,
                model_name,
                mlflow_run_id,
                prediction_count
            ),
        )

    conn.commit()

def log_evaluation_run(
    conn,
    run_id: str,
    dataset_id: str,
    mlflow_run_id: str,
    metrics: dict,
    slice_key: str = "overall"
):

    query = """
        INSERT INTO ml_runs (
            ml_run_id,
            run_id,
            dataset_id,
            stage,
            mlflow_run_id,
            metric_name,
            metric_value,
            slice_key,
            created_at
        )
        VALUES (%s, %s, %s, 'evaluate', %s, %s, %s, %s, NOW());
    """

    with conn.cursor() as cur:
        for k, v in metrics.items():
            cur.execute(
                query,
                (
                    f"{run_id}_eval_{k}",
                    run_id,
                    dataset_id,
                    mlflow_run_id,
                    k,
                    float(v),
                    slice_key
                ),
            )

    conn.commit()


def update_ml_run_status(
    conn,
    run_id: str,
    status: str,
):
    if status not in ("running", "success", "failed"):
        raise ValueError("Invalid status")

    query = "UPDATE ml_pipeline_runs SET status = %s, ended_at = NOW() WHERE run_id = %s;"

    with conn.cursor() as cur:
        cur.execute(query, (status, run_id))

    conn.commit()
    logger.info("ML run updated: run_id=%s status=%s", run_id, status)