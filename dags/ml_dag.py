from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.state import State
from airflow.utils.trigger_rule import TriggerRule

from datetime import datetime, timedelta
import logging
import uuid
import psycopg2
import subprocess
import os

from ml_helpers import (
    create_or_get_ml_run,
    dataset_log,
    update_ml_run_status,
)

logger = logging.getLogger(__name__)

# -------------------------
# CONFIG
# -------------------------
DB_CONFIG = {
    "host": os.getenv("PGHOST", "postgres"),
    "port": int(os.getenv("PGPORT", "5432")),
    "database": os.getenv("PGDATABASE", "retail_dw"),
    "user": os.getenv("PGUSER", "airflow"),
    "password": os.getenv("PGPASSWORD", "airflow"),
}

DAG_ID = "retail_ml_dag"
PIPELINE_NAME = "retail_ml_pipeline"

GOLD_TABLE = "gold_table"
PRED_TABLE = "predictions_table"

# -------------------------
# UTILS
# -------------------------
def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def get_run_date(context):
    return context["ds"]

def get_run_id(context):
    return context["ti"].xcom_pull(task_ids="init", key="run_id")

# -------------------------
# INIT
# -------------------------
def init_ml_run(**context):
    conn = get_connection()
    run_date = get_run_date(context)

    run_id = f"ml_{run_date}_{uuid.uuid4().hex[:6]}"

    try:
        create_or_get_ml_run(
            conn=conn,
            run_id=run_id,
            pipeline_name=PIPELINE_NAME,
            run_date=run_date,
        )

        context["ti"].xcom_push(key="run_id", value=run_id)

    finally:
        conn.close()

# -------------------------
# GENERIC EXECUTOR
# -------------------------
def execute_ml_step(
    step_name: str,
    module: str,
    dataset_type: str,
    source_table: str,
    **context
):
    conn = get_connection()
    run_id = get_run_id(context)
    run_date = get_run_date(context)

    try:
        # log dataset
        dataset_id = dataset_log(
            conn=conn,
            run_id=run_id,
            dataset_type=dataset_type,
            source_table=source_table,
            feature_query_hash="auto",
            feature_version="v1",
            row_count=0,
            schema_hash="auto",
        )

        # execute script
        cmd = [
            "python", "-m", module,
            "--run-date", run_date,
            "--run-id", run_id,
            "--dataset-id", dataset_id,
        ]

        logger.info("Running %s: %s", step_name, cmd)
        subprocess.run(cmd, check=True)

    finally:
        conn.close()

# -------------------------
# TASK WRAPPERS
# -------------------------
def train_task(**context):
    execute_ml_step(
        step_name="train",
        module="ML.train",
        dataset_type="train",
        source_table=GOLD_TABLE,
        **context
    )

def predict_task(**context):
    execute_ml_step(
        step_name="predict",
        module="ML.predict",
        dataset_type="predict",
        source_table=GOLD_TABLE,
        **context
    )

def evaluate_task(**context):
    execute_ml_step(
        step_name="evaluate",
        module="ML.evaluate",
        dataset_type="evaluate",
        source_table=PRED_TABLE,
        **context
    )

# -------------------------
# FINALIZE
# -------------------------
def finalize(**context):
    conn = get_connection()

    try:
        run_id = get_run_id(context)
        dag_run = context["dag_run"]

        states = [
            dag_run.get_task_instance("train").state,
            dag_run.get_task_instance("predict").state,
            dag_run.get_task_instance("evaluate").state,
        ]

        if all(s == State.SUCCESS for s in states):
            update_ml_run_status(conn, run_id, "success")
        else:
            update_ml_run_status(conn, run_id, "failed")

    finally:
        conn.close()

# -------------------------
# DAG
# -------------------------
default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    start_date=datetime(2011, 1, 29),
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:

    init = PythonOperator(
        task_id="init",
        python_callable=init_ml_run,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=train_task,
    )

    predict = PythonOperator(
        task_id="predict",
        python_callable=predict_task,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=evaluate_task,
    )

    finalize_task = PythonOperator(
        task_id="finalize",
        python_callable=finalize,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    init >> train >> predict >> evaluate >> finalize_task