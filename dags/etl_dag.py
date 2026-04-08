from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.state import State
from airflow.utils.trigger_rule import TriggerRule

from datetime import timedelta, datetime
import logging
import os
import sys
from pathlib import Path
import psycopg2
import subprocess

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.etl_helpers import (
    report_table_count,
    update_run_rows,
    update_run_status,
    create_or_get_run,
    start_step,
    finish_step
)

import ETL.bronze, ETL.silver, ETL.gold

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": os.getenv("PGHOST", "postgres"),
    "port": int(os.getenv("PGPORT", "5432")),
    "database": os.getenv("PGDATABASE", "retail_dw"),
    "user": os.getenv("PGUSER", "airflow"),
    "password": os.getenv("PGPASSWORD", "airflow"),
}

DAG_ID = "retail_etl_dag"
PIPELINE_NAME = "retail_pipeline"

BRONZE_TABLE = os.getenv("BRONZE_TABLE", "bronze_sales")
SILVER_TABLE = os.getenv("SILVER_TABLE", "silver_table")
GOLD_TABLE = os.getenv("GOLD_TABLE", "gold_table")

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def get_run_date(context):
    return context["ds"]

def get_run_id(context) -> int:
    return context["ti"].xcom_pull(task_ids="init_run", key="run_id")

def run_init(**context):
    conn = get_connection()
    run_id = None

    try:
        run_date = get_run_date(context)

        run_id = create_or_get_run(
            conn=conn,
            dag_id=DAG_ID,
            pipeline_name=PIPELINE_NAME,
            run_date=run_date,
            triggered_by="scheduler",
        )

        update_run_status(conn, run_id, "running")

        context["ti"].xcom_push(key="run_id", value=run_id)
        logger.info("run_id=%s initialize for run_date=%s", run_id, run_date)
    except Exception:
        logger.exception(
            "Initialization failed for run_id=%s with run_date=%s",
            run_id,
            run_date,
        )
    finally:
        conn.close()


def execute_step(step_name:str, cmd:list[str], output_table:str, input_table:str|None, **context):
    
    conn = get_connection()
    run_id = get_run_id(context)
    run_date = get_run_date(context)

    input_rows = 0
    output_rows = 0

    try:
        start_step(conn, run_id, DAG_ID, step_name)

        if input_table is not None:
            input_rows = report_table_count(conn, input_table, run_date=run_date)

        logger.info("Running %s command: %s", step_name, cmd)
        subprocess.run(cmd, check=True)

        output_rows = report_table_count(conn, output_table, run_date=run_date)

        finish_step(
            conn=conn,
            run_id=run_id,
            step_name=step_name,
            status="success",
            input_rows=input_rows,
            output_rows=output_rows,
        )

        if step_name == "bronze":
            update_run_rows(conn, run_id, rows_bronze=output_rows)
        elif step_name == "silver":
            update_run_rows(conn, run_id, rows_silver=output_rows)
        elif step_name == "gold":
            update_run_rows(conn, run_id, rows_gold=output_rows)

        logger.info(
            "%s succeeded: run_id=%s input_rows=%s output_rows=%s", step_name, run_id, input_rows, output_rows,
        )

    except Exception as e:
        error_message = e
        try:
            finish_step(
                conn=conn,
                run_id=run_id,
                step_name=step_name,
                status="failed",
                input_rows=input_rows,
                output_rows=output_rows,
                error_message=error_message
            )
        
        except Exception:
            logger.exception(
                "Failure marking failed in metadata for step=%s and run_id=%s",
                step_name,
                run_id,
            )

        logger.exception("%s failed for run_id=%s", step_name, run_id)
        raise

    finally:
        conn.close()


def bronze_task(run_date:str, sales_csv_path: str, calendar_csv_path:str | None, sell_prices_csv_path: str |None):
    cmd = [
        "python",
        "-m",
        "ETL.bronze",
        "--run-date", run_date,
        "--sales-csv-path", sales_csv_path,
    ]

    if calendar_csv_path is not None:
        cmd += ["--calendar-csv-path", calendar_csv_path]

    if sell_prices_csv_path is not None:
        cmd += ["--sell-prices-csv-path", sell_prices_csv_path]

    execute_step(
        step_name="bronze",
        cmd=cmd,
        output_table=BRONZE_TABLE,
        input_table=None,
    )

def silver_task(run_date:str):
    execute_step(
        step_name="silver",
        cmd=[
            "python",
            "-m",
            "ETL.silver",
            "--run-date",run_date,
        ],
        output_table=SILVER_TABLE,
        input_table=BRONZE_TABLE
    )


def gold_task(run_date: str):
    execute_step(
        step_name="gold",
        cmd=[
            "python",
            "-m",
            "ETL.gold",
            "--run-date",run_date,
        ],
        output_table=GOLD_TABLE,
        input_table=SILVER_TABLE,
    )


def finalize_pipeline(**context):
    conn = get_connection()
    try:
        run_id = get_run_id(context)
        dag_run = context["dag_run"]

        bronze_state = dag_run.get_task_instance("bronze").state
        silver_state = dag_run.get_task_instance("silver").state
        gold_state = dag_run.get_task_instance("gold").state

        states = [bronze_state, silver_state, gold_state]

        logger.info(
            "Finalize states: bronze=%s silver=%s gold=%s",
            bronze_state,
            silver_state,
            gold_state,
        )

        if all(state == State.SUCCESS for state in states):
            update_run_status(conn, run_id, "success")
            logger.info("Pipeline marked success for run_id=%s", run_id)
        else:
            error_message = (
                f"Pipeline failed. Task states: bronze={bronze_state}, "
                f"silver={silver_state}, gold={gold_state}"
            )
            update_run_status(conn, run_id, "failed", error_message=error_message)
            logger.info("Pipeline marked failed for run_id=%s", run_id)

    finally:
        conn.close()


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Retail ETL DAG with run and step metadata tracking",
    start_date=datetime(2011,1,29),
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:
    
    init_run_task = PythonOperator(
        task_id="init_run",
        python_callable=run_init,
    )

    bronze_run = PythonOperator(
        task_id="bronze",
        python_callable=bronze_task,
    )

    silver_run = PythonOperator(
        task_id="silver",
        python_callable=silver_task,
    )

    gold_run = PythonOperator(
        task_id="gold",
        python_callable=gold_task,
    )

    finalize_pipeline_task = PythonOperator(
        task_id="finalize_pipeline",
        python_callable=finalize_pipeline,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    init_run_task >> bronze_run >> silver_run >> gold_run >> finalize_pipeline_task