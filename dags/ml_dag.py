from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import uuid
import os

default_args = {
    "owner": "airflow",
    "retries": 2,
}

BASE_PATH = "/opt/airflow/data"


def task_create_run(**context):
    from utils.db import get_connection
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

    return {"run_id": run_id}


def task_build_dataset(**context):
    from data_loader import DataLoader, build_dataset
    
    ti = context["ti"]
    prev = ti.xcom_pull(task_ids="create_run")

    run_id = prev["run_id"]
    run_date = context["ds"]

    cfg = DataLoader(
        pipeline_name="etl_pipeline",
        run_date=run_date,
        table_name="gold_table",
        date_column="run_date",
        feature_version="v1"
    )

    df, meta = build_dataset(cfg)

    os.makedirs(BASE_PATH, exist_ok=True)
    path = f"{BASE_PATH}/{meta['dataset_id']}.parquet"

    if not os.path.exists(path):
        df.to_parquet(path)

    with get_connection() as conn:
        write_ml_dataset(
            conn=conn,
            run_id=run_id,
            cfg=cfg,
            meta=meta
        )
    
    return {
        "run_id": run_id,
        "dataset_id": meta["dataset_id"],
        "data_path": path
    }

def task_train(**context):
    import pandas as pd
    from train import train_pipeline

    ti = context["ti"]
    meta = ti.xcom_pull(task_ids="build_dataset")

    df = pd.read_parquet(meta["data_path"])

    result = train_pipeline(
        df=df,
        run_id=meta["run_id"],
        dataset_id=meta["dataset_id"]
    )

    # EXPECT train_pipeline RETURNS:
    # {
    #   "mlflow_run_id": ...,
    #   "train_path": ...,
    #   "val_path": ...,
    #   "test_path": ...
    # }

    return {
        "run_id": meta["run_id"],
        "dataset_id": meta["dataset_id"],
        **result
    }


# --------------------------
# TASK 4: PREDICT
# --------------------------

def task_predict(**context):
    import pandas as pd
    from predict import predict_pipeline

    ti = context["ti"]
    meta = ti.xcom_pull(task_ids="train")

    df_test = pd.read_parquet(meta["test_path"])

    df_pred = predict_pipeline(
        df=df_test,
        mlflow_run_id=meta["mlflow_run_id"],
        dataset_id=meta["dataset_id"]
    )

    # Persist predictions
    pred_path = f"{BASE_PATH}/pred_{meta['dataset_id']}.parquet"

    if not os.path.exists(pred_path):
        df_pred.to_parquet(pred_path)

    return {
        "run_id": meta["run_id"],
        "dataset_id": meta["dataset_id"],
        "mlflow_run_id": meta["mlflow_run_id"],
        "pred_path": pred_path
    }


# --------------------------
# TASK 5: EVALUATE
# --------------------------

def task_evaluate(**context):
    import pandas as pd
    from evaluate import evaluate_pipeline

    ti = context["ti"]
    meta = ti.xcom_pull(task_ids="predict")

    df_pred = pd.read_parquet(meta["pred_path"])

    evaluate_pipeline(
        df=df_pred,
        mlflow_run_id=meta["mlflow_run_id"],
        dataset_id=meta["dataset_id"]
    )


# --------------------------
# TASK 6: FINALIZE
# --------------------------

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
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
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

    # Dependencies
    create_run >> build_dataset >> train >> predict >> evaluate >> finalize