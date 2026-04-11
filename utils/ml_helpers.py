import logging
from typing import Optional, Dict
import uuid
from ML.data_loader import DataLoader
logger = logging.getLogger(__name__)


def create_or_get_ml_run(
    conn,
    run_id: str,
    pipeline_name: str,
    run_date: str,
    triggered_by: str = "scheduler",
):
    sql = """
    INSERT INTO ml_pipeline_runs run_id, pipeline_name, status, triggered_by,created_at 
    VALUES (%s, %s, 'running', %s, NOW())
    ON CONFLICT (run_id)
    DO UPDATE SET
        status = 'running',
        triggered_by = EXCLUDED.triggered_by,
        ended_at = NULL;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (run_id, run_date, pipeline_name, triggered_by))

    conn.commit()
    logger.info("ML pipeline run initialized: run_id=%s", run_id)


def write_ml_dataset(
    conn, 
    cfg: DataLoader, 
    meta: dict
):
    query = """
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
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
    """

    params = [
        meta["dataset_id"],
        meta["etl_run_id"],
        "training",
        cfg.table_name,
        cfg.start_date,
        cfg.end_date,
        cfg.feature_version,
        meta["feature_hash"],
        meta["row_count"],
        meta["schema_hash"]
    ]

    conn.execute(query, params)


def log_training_run(
    conn,
    run_id: str,
    dataset_id: str,
    model_name: str,
    mlflow_run_id: str,
):
    ml_run_id = str(uuid.uuid4())

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
    prediction_count: int,
):
    ml_run_id = str(uuid.uuid4())

    query = """
        INSERT INTO ml_runs (
            ml_run_id,
            run_id,
            dataset_id,
            stage,
            mlflow_run_id,
            prediction_count,
            created_at
        )
        VALUES (%s, %s, %s, 'predict', %s, %s, NOW());
    """

    with conn.cursor() as cur:
        cur.execute(
            query,
            (ml_run_id, run_id, dataset_id, mlflow_run_id, prediction_count),
        )

    conn.commit()
    logger.info("Prediction logged: mlflow_run_id=%s and rows=%s", mlflow_run_id, prediction_count)


def log_evaluation_metrics(
    conn,
    run_id: str,
    dataset_id: str,
    mlflow_run_id: str,
    metrics: Dict[str, float],
    slice_key: str = "overall",
    slice_value: Optional[str] = None,
):
    rows = []
    for metric_name, metric_value in metrics.items():
        rows.append(
            (
                str(uuid.uuid4()),
                run_id,
                dataset_id,
                "evaluate",
                mlflow_run_id,
                metric_name,
                metric_value,
                slice_key,
                slice_value,
            )
        )

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
            slice_value,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());
    """

    with conn.cursor() as cur:
        cur.executemany(query, rows)

    conn.commit()
    logger.info(
        "Evaluation metrics logged: mlflow_run_id=%s slice=%s", mlflow_run_id, slice_key,
    )


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