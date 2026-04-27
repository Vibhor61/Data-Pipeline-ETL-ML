from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import logging
import mlflow
from typing import Optional
from utils.db import get_connection
from utils.ml_helpers import(
    log_dataset, 
    create_or_get_ml_pipeline_run,
    get_etl_run_id,
    start_ml_stage,
    finish_ml_stage,
    update_ml_pipeline_status,
    atomic_write_parquet
)

from ML.train import train_main
from ML.data_loader import DataLoader,build_dataset_cfg
from ML.predict import predict_pipeline
from ML.evaluate import evaluate_pipeline

logger = logging.getLogger(__name__)

BASE_PATH = "/opt/airflow/data"
DAG_ID = "retail_ml_dag"
PIPELINE_NAME = "retail_pipeline"
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT = "retail_sales_forecasting"


def run_stage(
    *,
    stage: str,
    run_id: str,
    dataset_id: str,
    parent_mlflow_run_id: str,
    source_mlflow_run_id: Optional[str],
    stage_fn,
    stage_kwargs: dict
):
    """
    Generic ML stage runner.

    Responsibilities:
    - starts MLflow run
    - records stage start metadata
    - executes stage function
    - records success/failure metadata
    - updates pipeline failure state if stage fails

    Returns:
        dict with:
        {
            "mlflow_run_id": str,
            "result": Any
        }
    """
    ml_run_id = f"{run_id}_{stage}"
    mlflow_run_id = None

    try:
        with mlflow.start_run(run_name=ml_run_id) as run:
            mlflow_run_id = run.info.run_id

            mlflow.set_tag("parent_run_id", parent_mlflow_run_id)
            mlflow.set_tag("stage", stage)

            with get_connection() as conn:
                start_ml_stage(
                    conn=conn,
                    ml_run_id=ml_run_id,
                    run_id=run_id,
                    dataset_id=dataset_id,
                    stage=stage,
                    mlflow_run_id=mlflow_run_id,
                    source_mlflow_run_id=source_mlflow_run_id
                )

            stage_kwargs = {**stage_kwargs, "mlflow_run_id": mlflow_run_id}
            result = stage_fn(**stage_kwargs)

            with get_connection() as conn:
                finish_ml_stage(
                    conn=conn,
                    run_id=run_id,
                    dataset_id=dataset_id,
                    stage=stage,
                    status="success",
                    mlflow_run_id=mlflow_run_id
                )

            return {
                "mlflow_run_id": mlflow_run_id,
                "result": result
            }

    except Exception as e:
        with get_connection() as conn:
            finish_ml_stage(
                conn=conn,
                run_id=run_id,
                dataset_id=dataset_id,
                stage=stage,
                status="failed",
                mlflow_run_id=mlflow_run_id,
                error_message=str(e)
            )

            update_ml_pipeline_status(
                conn=conn,
                run_id=run_id,
                status="failed"
            )

        raise


def task_create_run(**context):
    """
    Creates or retrieves the ML pipeline run record.
    Starts an MLflow parent run and returns run metadata for downstream tasks.
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

    with mlflow.start_run(run_name=run_id) as run:
        mlflow_run_id = run.info.run_id
        mlflow.set_tag("pipeline_name", PIPELINE_NAME)
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("run_date", run_date)
       
    logger.info("Created ML pipeline run: run_id=%s, mlflow_run_id=%s", run_id, mlflow_run_id)
    return {
        "run_id": run_id,
        "run_date": run_date,
        "parent_mlflow_run_id": mlflow_run_id
    }


def task_build_dataset(**context):
    """
    Builds train/validation/test datasets and writes them to parquet.
    Stores dataset metadata in the ML metadata table and returns context.
    """

    ti = context["ti"]
    run_context = ti.xcom_pull(task_ids="create_run")
    run_date = run_context["run_date"]
    run_id = run_context["run_id"]   # airflow gives this

    cfg = DataLoader(
        run_id=run_id,
        pipeline_name=PIPELINE_NAME,
        run_date=run_date,
        table_name="gold_table",
        date_column="run_date",
        feature_version="v1",
        output_dir="/opt/airflow/data/datasets"
    )

    datasets, meta = build_dataset_cfg(cfg)

    atomic_write_parquet(datasets["train"], meta["paths"]["train"])
    atomic_write_parquet(datasets["val"], meta["paths"]["val"])
    atomic_write_parquet(datasets["test"], meta["paths"]["test"])

    with get_connection() as conn:
        log_dataset(conn, meta)

    logger.info("Dataset built: dataset_id=%s, rows=%d",  meta["dataset_id"], meta["row_counts"]["total"])

    return {
        "run_id": run_id,
        "dataset_id": meta["dataset_id"],
        "train_path": meta["paths"]["train"],
        "val_path": meta["paths"]["val"],
        "test_path": meta["paths"]["test"],
        "parent_mlflow_run_id": run_context["parent_mlflow_run_id"],
        "dataset_start_date": meta["dataset_start_date"],
        "dataset_end_date": meta["dataset_end_date"]
    }


def task_train(**context):
    """
    Runs model training with MLflow tracking.
    Records the train stage metadata and returns the train MLflow run id.
    """
    
    ti = context["ti"]
    dataset_context = ti.xcom_pull(task_ids="build_dataset")

    result = run_stage(
        stage="train",
        run_id=dataset_context["run_id"],
        dataset_id=dataset_context["dataset_id"],
        parent_mlflow_run_id=dataset_context["parent_mlflow_run_id"],
        source_mlflow_run_id=None,
        stage_fn=train_main,
        stage_kwargs={
            "run_id": dataset_context["run_id"],
            "dataset_id": dataset_context["dataset_id"],
            "train_path": dataset_context["train_path"],
            "val_path": dataset_context["val_path"]
        }
    )

    return {
        **dataset_context,
        "train_mlflow_run_id": result["mlflow_run_id"]
    }



def task_predict(**context):
    """
    Runs model prediction and records the predict stage metadata.
    Returns the prediction file path for downstream evaluation.
    """
   
    ti = context["ti"]
    dataset_context = ti.xcom_pull(task_ids="train")

    result = run_stage(
        stage="predict",
        run_id=dataset_context["run_id"],
        dataset_id=dataset_context["dataset_id"],
        parent_mlflow_run_id=dataset_context["parent_mlflow_run_id"],
        source_mlflow_run_id=dataset_context["train_mlflow_run_id"],
        stage_fn=predict_pipeline,
        stage_kwargs={
            "test_path": dataset_context["test_path"],
            "run_id": dataset_context["run_id"],
            "dataset_id": dataset_context["dataset_id"],
            "train_mlflow_run_id": dataset_context["train_mlflow_run_id"]
        }
    )

    return {
        **dataset_context,
        "pred_path": result["result"],
        "pred_mlflow_run_id": result["mlflow_run_id"]
    }


def task_evaluate(**context):
    """
    Evaluates predictions and records the evaluate stage metadata.
    Returns the same dataset context after evaluation completes.
    """

    ti = context["ti"]
    dataset_context = ti.xcom_pull(task_ids="predict")

    run_stage(
        stage="evaluate",
        run_id=dataset_context["run_id"],
        dataset_id=dataset_context["dataset_id"],
        parent_mlflow_run_id=dataset_context["parent_mlflow_run_id"],
        source_mlflow_run_id=dataset_context["pred_mlflow_run_id"],
        stage_fn=evaluate_pipeline,
        stage_kwargs={
            "pred_path": dataset_context["pred_path"],
            "run_id": dataset_context["run_id"],
            "dataset_id": dataset_context["dataset_id"],
            "predict_mlflow_run_id": dataset_context["pred_mlflow_run_id"]
        }
    )

    return dataset_context


def task_finalize(**context):
    """
    Marks the ML pipeline run as successful in metadata.
    Finalizes the ML workflow for the current run.
    """

    ti = context["ti"]
    dataset_context = ti.xcom_pull(task_ids="evaluate")

    run_id = dataset_context["run_id"]

    with get_connection() as conn:
        update_ml_pipeline_status(
            conn=conn,
            run_id=run_id,
            status="success"
        )

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2013, 1, 29),
    # schedule_interval="@weekly",
    catchup=True,
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