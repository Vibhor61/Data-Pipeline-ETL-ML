from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import logging
import mlflow
import sys 
from pathlib import Path
from utils.db import get_connection
from utils.ml_helpers import(
    log_dataset, 
    create_or_get_ml_pipeline_run,
    get_etl_run_id,
    start_ml_stage,
    finish_ml_stage,
    update_ml_pipeline_status
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ML.train import train_main
from ML.data_loader import DataLoader,build_dataset_cfg
from ML.predict import predict_pipeline
from ML.evaluate import evaluate_pipeline

logger = logging.getLogger(__name__)

BASE_PATH = "/opt/airflow/data"
DAG_ID = "retail_ml_dag"
PIPELINE_NAME = "retail_pipeline"
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT = "retail_demand_forecasting"


def mlflow_run_context(mlflow_run_id: str):
    """
    Configures and initializes MLflow run context.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return mlflow.start_run(run_id=mlflow_run_id)


def run_stage(
    *,
    stage: str,
    run_id: str,
    dataset_id: str,
    mlflow_run_id: str,
    stage_fn,
    stage_kwargs: dict
):
    """
    Executes a pipeline stage (train/predict/evaluate) with metadata tracking.

    - Marks stage start in metadata
    - Runs ML stage function via provided callable
    - Captures and logs metrics/results
    - Marks stage success or failure with metadata updates
    - Updates pipeline-level status on error
    """
    stage_exc = None

    with mlflow_run_context(mlflow_run_id): 
    
        mlflow.set_tag(f"{stage}_status", "running")

        with get_connection() as conn:
            start_ml_stage(
                conn=conn,
                ml_run_id=f"{run_id}_{stage}",
                run_id=run_id,
                dataset_id=dataset_id,
                stage=stage,
                mlflow_run_id=mlflow_run_id,
            )

        try:
            result = stage_fn(**{**stage_kwargs, "mlflow_run_id": mlflow_run_id})
            mlflow.log_metric(f"{stage}_done", 1)
            mlflow.set_tag(f"{stage}_status", "success")
        except Exception as e:
            stage_exc = e
            mlflow.set_tag(f"{stage}_status", "failed")
            mlflow.log_metric(f"{stage}_failed", 1)

    if stage_exc is None:
        with get_connection() as conn:
            finish_ml_stage(
                conn=conn,
                run_id=run_id,
                dataset_id=dataset_id,
                stage=stage,
                status="success",
                mlflow_run_id=mlflow_run_id,
            )
        return {"result": result}
    else:
        with get_connection() as conn:
            finish_ml_stage(
                conn=conn,
                run_id=run_id,
                dataset_id=dataset_id,
                stage=stage,
                status="failed",
                mlflow_run_id=mlflow_run_id,
                error_message=str(stage_exc),
            )
            update_ml_pipeline_status(conn=conn, run_id=run_id, status="failed")
        raise stage_exc


def task_create_run(**context):
    """
    Initializes ML pipeline run metadata and MLflow tracking.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_date = context["ds"]

    with get_connection() as conn:
        run_id = get_etl_run_id(conn, run_date)

        if not run_id:
            raise ValueError(f"No ETL run found for date {run_date}")
        
        create_or_get_ml_pipeline_run(
            conn=conn,
            run_id=run_id,
            pipeline_name=PIPELINE_NAME,
            run_date=run_date,
            triggered_by="scheduler"
        )

    mlflow_run = mlflow.start_run(run_name=str(run_id))
    mlflow_run_id = str(mlflow_run.info.run_id)

    mlflow.set_tag("pipeline_name", PIPELINE_NAME)
    mlflow.set_tag("run_id", str(run_id))
    mlflow.set_tag("run_date", run_date)
    mlflow.end_run()

    logger.info("Created ML pipeline run: run_id=%s, mlflow_run_id=%s", run_id, mlflow_run_id)
    return {
        "run_id": run_id,
        "run_date": run_date,
        "mlflow_run_id": mlflow_run_id
    }


def task_build_dataset(**context):
    """
    Builds train/validation/test datasets from gold table.
    """

    ti = context["ti"]
    run_context = ti.xcom_pull(task_ids="create_run")
    run_date = run_context["run_date"]
    run_id = str(run_context["run_id"])   # airflow gives this
    mlflow_run_id = run_context["mlflow_run_id"]

    cfg = DataLoader(
        run_id=run_id,
        pipeline_name=PIPELINE_NAME,
        run_date=run_date,
        table_name="gold_table",
        date_column="run_date",
        feature_version="v1",
        output_dir="/opt/airflow/data/datasets"
    )

    meta = build_dataset_cfg(cfg)

    with get_connection() as conn:
        log_dataset(conn, meta)

    logger.info("Dataset built at paths: %s", meta["paths"])

    logger.info("Dataset built: dataset_id=%s, rows=%d",  meta["dataset_id"], meta["row_counts"]["total"])

    return {
        "run_id": str(run_id),
        "dataset_id": str(meta["dataset_id"]),
        "train_path": meta["paths"]["parquet"]["train"],
        "val_path": meta["paths"]["parquet"]["val"],
        "test_path": meta["paths"]["parquet"]["test"],
        "train_libsvm_path": meta["paths"]["libsvm"]["train"],
        "val_libsvm_path": meta["paths"]["libsvm"]["val"],
        "test_libsvm_path": meta["paths"]["libsvm"]["test"],
        "mlflow_run_id": mlflow_run_id,
        "dataset_dir": meta["dataset_dir"]
    }


def task_train(**context):
    """
    Triggers ML training module via run_stage.
    """
    
    ti = context["ti"]
    dataset_context = ti.xcom_pull(task_ids="build_dataset")
    
    run_stage(
        stage="train",
        run_id=str(dataset_context["run_id"]),
        dataset_id=str(dataset_context["dataset_id"]),
        mlflow_run_id=dataset_context["mlflow_run_id"],
        stage_fn=train_main,
        stage_kwargs={
            "run_id": str(dataset_context["run_id"]),
            "dataset_id": str(dataset_context["dataset_id"]),
            "train_libsvm_path": dataset_context["train_libsvm_path"],
            "val_libsvm_path": dataset_context["val_libsvm_path"],
            "dataset_dir": dataset_context["dataset_dir"]
        }
    )
    
    return {
        **dataset_context
    }


def task_predict(**context):
    """
    Triggers ML prediction module via run_stage.
    """
   
    ti = context["ti"]
    dataset_context = ti.xcom_pull(task_ids="train")
    
    result = run_stage(
        stage="predict",
        run_id=str(dataset_context["run_id"]),
        dataset_id=str(dataset_context["dataset_id"]),
        mlflow_run_id=dataset_context["mlflow_run_id"],
        stage_fn=predict_pipeline,
        stage_kwargs={
            "test_path": dataset_context["test_path"],
            "run_id": str(dataset_context["run_id"]),
            "dataset_id": str(dataset_context["dataset_id"]),
            "dataset_dir": dataset_context.get("dataset_dir")
        }
    )

    return {
        **dataset_context,
        "pred_path": result["result"]
    }


def task_evaluate(**context):
    """
    Triggers ML evaluation module via run_stage.
    """

    ti = context["ti"]
    dataset_context = ti.xcom_pull(task_ids="predict")

    run_stage(
        stage="evaluate",
        run_id=str(dataset_context["run_id"]),
        dataset_id=str(dataset_context["dataset_id"]),
        mlflow_run_id=dataset_context["mlflow_run_id"],
        stage_fn=evaluate_pipeline,
        stage_kwargs={
            "pred_path": dataset_context["pred_path"],
            "run_id": str(dataset_context["run_id"]),
            "dataset_id": str(dataset_context["dataset_id"]),
            "dataset_dir": dataset_context.get("dataset_dir")
        }
    )

    return dataset_context


def task_finalize(**context):
    """
    Finalizes pipeline run status based on task outcomes.
    """

    ti = context["ti"]
    dataset_context = ti.xcom_pull(task_ids="evaluate")

    run_id = str(dataset_context["run_id"])

    with get_connection() as conn:
        update_ml_pipeline_status(
            conn=conn,
            run_id=run_id,
            status="success"
        )

    logger.info("Pipeline finalized: run_id=%s", run_id)
    
    
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2012, 1, 1),
    # schedule_interval="@weekly",
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
) as dag:

    create_run = PythonOperator(
        
        task_id="create_run",
        python_callable=task_create_run
    )

    build_dataset = PythonOperator(
        task_id="build_dataset",
        python_callable=task_build_dataset
    )

    train = PythonOperator(
        task_id="train",
        python_callable=task_train
    )

    predict = PythonOperator(
        task_id="predict",
        python_callable=task_predict
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=task_evaluate
    )

    finalize = PythonOperator(
        task_id="finalize",
        python_callable=task_finalize
    )

    create_run >> build_dataset >> train >> predict >> evaluate >> finalize