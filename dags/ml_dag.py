from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import logging
import mlflow
import pandas as pd
from utils.db import get_connection
from utils.ml_helpers import(
    log_dataset, 
    create_or_get_ml_pipeline_run,
    get_etl_run_id,
    start_ml_stage,
    finish_ml_stage,
    update_ml_pipeline_status
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
MLFLOW_EXPERIMENT = "retail_demand_forecasting"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)


def task_create_run(**context):

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

    ti = context["ti"]
    run_context = ti.xcom_pull(task_ids="create_run")
    run_date = run_context["run_date"]
    run_id = run_context["run_id"]   # airflow gives this

    cfg = DataLoader(
        pipeline_name=PIPELINE_NAME,
        run_date=run_date,
        table_name="gold_table",
        date_column="run_date",
        feature_version="v1"
    )

    datasets, meta = build_dataset_cfg(cfg)

    datasets["train"].to_parquet(meta["paths"]["train"], index=False)
    datasets["val"].to_parquet(meta["paths"]["val"], index=False)
    datasets["test"].to_parquet(meta["paths"]["test"], index=False)

    with get_connection() as conn:
        log_dataset(conn, run_id, meta)

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
    
    ti = context["ti"]
    dataset_context = ti.xcom_pull(task_ids="build_dataset")
    run_id = dataset_context["run_id"]
    dataset_id = dataset_context["dataset_id"]

    train_path = dataset_context["train_path"]
    val_path = dataset_context["val_path"]

    ml_run_id = f"{run_id}_train"

    try:
        with mlflow.start_run(run_name=ml_run_id) as run:
            mlflow_run_id = run.info.run_id
            
            # Link to parent for traceability
            mlflow.set_tag("parent_run_id", dataset_context["parent_mlflow_run_id"])
            
            with get_connection() as conn:
                if conn is None:
                    raise RuntimeError("Database connection failed")
                
                start_ml_stage(
                    conn=conn,
                    ml_run_id=ml_run_id,
                    run_id=run_id,
                    dataset_id=dataset_id,
                    stage="train",
                    mlflow_run_id=mlflow_run_id
                )
            

            train_main(
                run_id,
                dataset_id,
                train_path,
                val_path,
                mlflow_run_id
            )
            
            with get_connection() as conn:
                if conn is None:
                    raise RuntimeError("Database connection failed")
                finish_ml_stage(
                    conn=conn,
                    run_id=run_id,
                    dataset_id=dataset_id,
                    stage="train",
                    status="success",
                    mlflow_run_id=mlflow_run_id
                )
        
        logger.info("Training completed: mlflow_run_id=%s", mlflow_run_id)
        
        return {
            **dataset_context,
            "train_mlflow_run_id": mlflow_run_id
        }

    except Exception as e:
        with get_connection() as conn:
            finish_ml_stage(
                conn=conn,
                run_id=run_id,
                dataset_id=dataset_id,
                stage="train",
                status="failed",
                mlflow_run_id=mlflow_run_id,
                error_message=str(e)
            )
        raise



def task_predict(**context):
   
    ti = context["ti"]
    dataset_context = ti.xcom_pull(task_ids="train")

    run_id = dataset_context["run_id"]
    dataset_id = dataset_context["dataset_id"]
    test_path = dataset_context["test_path"]

    ml_run_id = f"{run_id}_predict"

    conn = get_connection()
    try:
        with mlflow.start_run(run_name=ml_run_id) as run:
            mlflow_run_id = run.info.run_id

            mlflow.set_tag("parent_run_id", dataset_context["parent_mlflow_run_id"])
            
            start_ml_stage(
                conn=conn,
                ml_run_id=ml_run_id,
                run_id=run_id,
                dataset_id=dataset_id,
                stage="predict",
                mlflow_run_id=mlflow_run_id
            )

            pred_path = predict_pipeline(
                test_df_path=test_path,
                run_id=run_id,
                dataset_id=dataset_id,
                train_mlflow_run_id=dataset_context["train_mlflow_run_id"]
            )

            finish_ml_stage(
                conn=conn,
                run_id=run_id,
                dataset_id=dataset_id,
                stage="predict",
                status="success",
                mlflow_run_id=mlflow_run_id
            )

    except Exception as e:
        finish_ml_stage(
            conn=conn,
            run_id=run_id,
            dataset_id=dataset_id,
            stage="predict",
            status="failed",
            mlflow_run_id=mlflow_run_id,
            error_message=str(e)
        )
        raise

    finally:
        conn.close()

    return {
        **dataset_context, 
        "pred_path":pred_path 
    }

def task_evaluate(**context):

    ti = context["ti"]
    dataset_context = ti.xcom_pull(task_ids="predict")

    run_id = dataset_context["run_id"]
    dataset_id = dataset_context["dataset_id"]
    pred_path = dataset_context["pred_path"]

    ml_run_id = f"{run_id}_evaluate"

    conn = get_connection()
    try:
        with mlflow.start_run(run_name=ml_run_id) as run:
            mlflow_run_id = run.info.run_id
            
            mlflow.set_tag("parent_run_id", dataset_context["parent_mlflow_run_id"])
            
            start_ml_stage(
                conn=conn,
                ml_run_id=ml_run_id,
                run_id=run_id,
                dataset_id=dataset_id,
                stage="evaluate",
                mlflow_run_id=mlflow_run_id
            )

            evaluate_pipeline(
                test_df_path=pred_path,
                run_id=run_id,
                dataset_id=dataset_id
            )

            finish_ml_stage(
                conn=conn,
                run_id=run_id,
                dataset_id=dataset_id,
                stage="evaluate",
                status="success",
                mlflow_run_id=mlflow_run_id
            )

    except Exception as e:
        finish_ml_stage(
            conn=conn,
            run_id=run_id,
            dataset_id=dataset_id,
            stage="evaluate",
            status="failed",
            mlflow_run_id=mlflow_run_id,
            error_message=str(e)
        )
        raise

    finally:
        conn.close()

    return dataset_context


def task_finalize(**context):

    ti = context["ti"]
    dataset_context = ti.xcom_pull(task_ids="evaluate")

    run_id = dataset_context["run_id"]

    with get_connection() as conn:
        update_ml_pipeline_status(
            conn=conn,
            run_id=run_id,
            status="success"
        )


with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2013, 1, 29),
    schedule_interval="@weekly",
    catchup=False,
    max_active_runs=1,
    owner="airflow",
    retries=2
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