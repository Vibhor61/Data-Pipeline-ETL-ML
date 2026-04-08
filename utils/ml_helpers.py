import logging
from typing import Optional, Dict
import uuid

logger = logging.getLogger(__name__)


def create_or_get_ml_run(
    conn,
    run_id: str,
    pipeline_name: str,
    run_date: str,
    triggered_by: str = "scheduler",
):
    sql = """
    INSERT INTO ml_pipeline_runs run_id, run_date, pipeline_name, status, triggered_by,created_at 
    VALUES (%s, %s, %s, 'running', %s, NOW())
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


def dataset_log(
    conn,
    run_id: str,
    dataset_type: str,
    source_table: str,
    feature_query_hash: str,
    feature_version: str,
    row_count: int,
    schema_hash: str,
):
    dataset_id = str(uuid.uuid4())

    query = """
        INSERT INTO ml_dataset (
            dataset_id, run_id, run_date,
            dataset_type, source_table,
            feature_query_hash, feature_version,
            row_count, schema_hash, created_at
        )
        SELECT %s, %s, run_date,
               %s, %s, %s, %s, %s, %s, NOW()
        FROM ml_pipeline_runs
        WHERE run_id = %s;
    """

    with conn.cursor() as cur:
        cur.execute(
            query,
            (
                dataset_id,
                run_id,
                dataset_type,
                source_table,
                feature_query_hash,
                feature_version,
                row_count,
                schema_hash,
                run_id,
            ),
        )

    conn.commit()
    logger.info("Dataset logged: dataset_id=%s type=%s", dataset_id, dataset_type)
    return dataset_id


def log_training_run(
    conn,
    run_id: str,
    dataset_id: str,
    model_name: str,
    model_version: str,
    mlflow_run_id: str,
):
    ml_run_id = str(uuid.uuid4())

    query = """
        INSERT INTO ml_runs (
            ml_run_id, run_id, dataset_id, stage,
            model_name, model_version, mlflow_run_id, created_at
        )
        VALUES (%s, %s, %s, 'train', %s, %s, %s, NOW());
    """

    with conn.cursor() as cur:
        cur.execute(
            query,
            (ml_run_id, run_id, dataset_id, model_name, model_version, mlflow_run_id),
        )

    conn.commit()
    logger.info("Training logged: model=%s version=%s", model_name, model_version)

    return ml_run_id


def log_prediction_run(
    conn,
    run_id: str,
    dataset_id: str,
    model_version: str,
    prediction_count: int,
):
    ml_run_id = str(uuid.uuid4())

    query = """
        INSERT INTO ml_runs (
            ml_run_id, run_id, dataset_id, stage,
            model_version, prediction_count, created_at
        )
        VALUES (%s, %s, %s, 'predict', %s, %s, NOW());
    """

    with conn.cursor() as cur:
        cur.execute(
            query,
            (ml_run_id, run_id, dataset_id, model_version, prediction_count),
        )

    conn.commit()
    logger.info("Prediction logged: model_version=%s rows=%s", model_version, prediction_count)

    return ml_run_id


def log_evaluation_metrics(
    conn,
    run_id: str,
    dataset_id: str,
    model_version: str,
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
                model_version,
                metric_name,
                metric_value,
                slice_key,
                slice_value,
            )
        )

    query = """
        INSERT INTO ml_runs (
            ml_run_id, run_id, dataset_id, stage,
            model_version,
            metric_name, metric_value,
            slice_key, slice_value,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());
    """

    with conn.cursor() as cur:
        cur.executemany(query, rows)

    conn.commit()
    logger.info(
        "Evaluation metrics logged: run_id=%s slice=%s",
        run_id,
        slice_key,
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
        cur.execute(query, status, run_id)

    conn.commit()
    logger.info("ML run updated: run_id=%s status=%s", run_id, status)