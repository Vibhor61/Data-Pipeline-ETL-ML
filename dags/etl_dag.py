from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.state import State
from airflow.utils.trigger_rule import TriggerRule

from typing import List, Optional
from datetime import timedelta, datetime
import logging
import os
import sys
from pathlib import Path
import subprocess
from utils.db import get_connection

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

logger = logging.getLogger(__name__)

DAG_ID = "retail_etl_dag"
PIPELINE_NAME = "retail_pipeline"

BRONZE_TABLE = os.getenv("BRONZE_TABLE", "bronze_sales")
SILVER_TABLE = os.getenv("SILVER_TABLE", "silver_table")
GOLD_TABLE = os.getenv("GOLD_TABLE", "gold_table")


def get_run_date(context):
    return context["ds"]

def get_run_id(context) -> int:
    return context["ti"].xcom_pull(task_ids="init_run", key="run_id")

def run_init(**context):
    """
    Initializes pipeline run metadata and creates run_id.
    Pushes run_id to XCom for downstream tasks.
    """
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
        raise 
    finally:
        conn.close()


def execute_step(step_name:str, cmd:List[str], output_table:str, input_table:Optional[str], **context):
    """
    Executes a pipeline step (bronze/silver/gold) with metadata tracking.

    - Marks step start in metadata
    - Runs ETL module via subprocess
    - Captures input/output row counts
    - Marks step success or failure with metrics
    - Updates pipeline-level row counts
    """
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
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True
        )

        if result.stdout:
            logger.info("Subprocess stdout:\n%s", result.stdout)

        if result.stderr:
            logger.error("Subprocess stderr:\n%s", result.stderr)

        result.check_returncode()

        output_rows = report_table_count(conn, output_table, run_date=run_date)

        finish_step(
            conn=conn,
            run_id=run_id,
            dag_id=DAG_ID,
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
        error_message = str(e)
        try:
            finish_step(
                conn=conn,
                run_id=run_id,
                dag_id=DAG_ID,
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


def bronze_task(sales_csv_path: str, calendar_csv_path: Optional[str], sell_prices_csv_path: Optional[str], **context):
    """
    Triggers Bronze ETL module via CLI command.
    """ 
    run_date = get_run_date(context)

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
        **context
    )
    

def silver_task(**context):
    """
    Triggers Silver ETL module via CLI command.
    """
    run_date = get_run_date(context)

    execute_step(
        step_name="silver",
        cmd=[
            "python",
            "-m",
            "ETL.silver",
            "--run-date",run_date,
        ],
        output_table=SILVER_TABLE,
        input_table=BRONZE_TABLE,
        **context
    )


def gold_task(**context):
    """
    Triggers Gold ETL module via CLI command.
    """
    run_date = get_run_date(context)

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
        **context
    )


def finalize_pipeline(**context):
    """
    Finalizes pipeline run status based on task outcomes.

    Marks run as success only if all steps succeed,
    otherwise marks as failed with aggregated error state.
    """
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
    "retries": 0,
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
        op_kwargs={
            "sales_csv_path": "/opt/airflow/data/raw/sales_train_validation.csv",
            "calendar_csv_path": "/opt/airflow/data/raw/calendar.csv",
            "sell_prices_csv_path": "/opt/airflow/data/raw/sell_prices.csv",
        },
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