from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import uuid
import os
import pandas as pd
from utils.db import get_connection


BASE_PATH = "/opt/airflow/data"


def task_create_run(**context):
    
    from utils.ml_helpers import create_or_get_ml_run

    run_date = context["ds"]
    run_id = f"ml_pipeline_{run_date}"

    with get_connection() as conn:
        create_or_get_ml_run(
            conn=conn,
            run_id=run_id,
            pipeline_name="ml_pipeline",
            triggered_by="airflow"
        )

    return {
        "run_id": run_id,
        "run_date": run_date
    }


def task_build_dataset(**context):
    from utils.ml_helpers import log_dataset
    ti = context["ti"]
    run_context = ti.xcom_pull(task_ids="create_run")
    run_date = run_context["run_date"]
    run_id = run_context["run_id"]   # airflow gives this

    #Config driven in version 2

    cfg = DataLoader(
        pipeline_name="etl_pipeline",
        run_date=run_date,
        table_name="gold_table",
        date_column="run_date",
        feature_version="v1"
    )

    df, meta = build_dataset(cfg)
    dataset_id = meta["dataset_id"]
    dataset_path = meta["dataset_path"]
    time_start = meta["time_start"]
    time_end = meta["time_end"]

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    if not os.path.exists(dataset_path):
        df.to_parquet(dataset_path, index=False)

    with get_connection() as conn:
        log_dataset(
            conn,
            run_id,
            meta
        )

    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_path": dataset_path,
        "metadata": meta
    }


def task_train(**context):
    import pandas as pd
    from train import train_pipeline

    ti = context["ti"]

    run_context = ti.xcom_pull(task_ids="create_run")
    dataset_context = ti.xcom_pull(task_ids="build_dataset")
    dataset_path = dataset_context["dataset_path"]
    
    df = pd.read_parquet(dataset_path)

    train_pipeline(
        df=df,
        run_id=run_context["run_id"],
        dataset_id=dataset_context["dataset_id"],
        meta=dataset_context
    )

    return {
        "run_id": run_context["run_id"],
        "dataset_id": dataset_context["dataset_id"],
        "dataset_path": dataset_context["dataset_path"]
    }


def task_predict(**context):
    import pandas as pd
    from predict import predict_pipeline

    ti = context["ti"]
    train_context = ti.xcom_pull(task_ids="train")
    run_context = ti.xcom_pull(task_ids="create_run")

    run_id = run_context["run_id"]
    dataset_id = train_context["dataset_id"]
    dataset_path = train_context["dataset_path"]

    df = pd.read_parquet(dataset_path)

    predict_pipeline(
        df=df,
        run_id=run_id,
        dataset_id=dataset_id
    )

    return {
        "run_id": train_context["run_id"],
        "dataset_id": train_context["dataset_id"],
        "dataset_path": train_context["dataset_path"]
    }


def task_evaluate(**context):
    import pandas as pd
    from evaluate import evaluate_pipeline

    ti = context["ti"]

    run_context = ti.xcom_pull(task_ids="create_run")
    dataset_context = ti.xcom_pull(task_ids="build_dataset")

    run_id = run_context["run_id"]
    dataset_id = dataset_context["dataset_id"]
    dataset_path = dataset_context["dataset_path"]

    df = pd.read_parquet(dataset_path)

    # evaluation assumes prediction column already exists (from predict step)
    evaluate_pipeline(
        df=df,
        run_id=run_id,
        dataset_id=dataset_id
    )

    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
    }


def task_finalize(**context):
    from utils.db import get_connection
    from utils.ml_helpers import update_ml_run_status

    ti = context["ti"]
    meta = ti.xcom_pull(task_ids="train")

    with get_connection() as conn:
        update_ml_run_status(
            conn=conn,
            run_id=meta["run_id"],
            status="success"
        )


# --------------------------
# DAG
# --------------------------

with DAG(
    dag_id="ml_pipeline_final",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    max_active_runs=1,
    owner="airflow",
    retries=2
) as dag:

    create_run = PythonOperator(
        task_id="create_run",
        python_callable=task_create_run,
    )

    build_dataset = PythonOperator(
        task_id="build_dataset",
        python_callable=task_build_dataset,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=task_train,
    )

    predict = PythonOperator(
        task_id="predict",
        python_callable=task_predict,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=task_evaluate,
    )

    finalize = PythonOperator(
        task_id="finalize",
        python_callable=task_finalize,
    )

    create_run >> build_dataset >> train >> predict >> evaluate >> finalize