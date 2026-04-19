import logging
from typing import Optional, Tuple
from psycopg2 import sql

logger = logging.getLogger(__name__)


ALLOWED_TABLES = {"bronze_sales","silver_table","gold_table"}
def safe_table_identifier(table_name: str) -> sql.Identifier:
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Table '{table_name}' not in ALLOWED_TABLES.")
    return sql.Identifier(table_name)


def report_table_count(conn, table_name: str, run_date: Optional[str] = None) -> int:
    """ 
    Returns row count for a table, optionally filtered by run_date.
    """
    ident = safe_table_identifier(table_name)

    if run_date:
        query = sql.SQL("SELECT COUNT(*) FROM {} WHERE run_date = %s;").format(ident)
        params: Tuple = (run_date,)
    else:
        query = sql.SQL("SELECT COUNT(*) FROM {};").format(ident)
        params = ()

    with conn.cursor() as cur:
        cur.execute(query, params)
        cnt = cur.fetchone()[0]

    logger.info(
        "%s rows%s: %s",
        table_name,
        f" for run_date={run_date}" if run_date else "",
        cnt,
    )
    return cnt


def create_or_get_run(
    conn,
    dag_id: str,
    pipeline_name: str,
    run_date: str,
    triggered_by: str = "scheduler",
) -> int:
    """
    Creates or resets a pipeline run record for (pipeline_name, dag_id, run_date).
    Idempotent via ON CONFLICT; always returns run_id.
    """
    sql_q = """
        INSERT INTO etl_pipeline_runs (dag_id, pipeline_name, run_date, status, triggered_by)
        VALUES (%s, %s, %s, 'running', %s)
        ON CONFLICT (pipeline_name, dag_id, run_date)
        DO UPDATE SET
            status = 'running',
            triggered_by = EXCLUDED.triggered_by,
            ended_at = NULL,
            error_message = NULL
        RETURNING run_id;
    """
    with conn.cursor() as cur:
        cur.execute(sql_q, (dag_id, pipeline_name, run_date, triggered_by))
        run_id = cur.fetchone()[0]

    conn.commit()
    logger.info("Run upserted: run_id=%s pipeline=%s dag_id=%s run_date=%s", run_id, pipeline_name, dag_id, run_date)
    return run_id


def update_run_status(
    conn, 
    run_id: int, 
    status: str, 
    error_message: Optional[str] = None
):
    """
    Updates pipeline run status (running, success, failed).
    Sets ended_at only for terminal states.
    """
    if status not in ("running", "success", "failed"):
        raise ValueError(f"Invalid status: {status}")

    if status == "running":
        sql_q = """
            UPDATE etl_pipeline_runs
            SET status = %s,
                ended_at = NULL,
                error_message = NULL
            WHERE run_id = %s;
        """
        params = (status, run_id)
    else:
        sql_q = """
            UPDATE etl_pipeline_runs
            SET status = %s,
                ended_at = NOW(),
                error_message = %s
            WHERE run_id = %s;
        """
        params = (status, error_message, run_id)

    with conn.cursor() as cur:
        cur.execute(sql_q, params)

    conn.commit()
    logger.info("Run %s updated to status=%s", run_id, status)


def update_run_rows(
    conn,
    run_id: int,
    rows_bronze: Optional[int] = None,
    rows_silver: Optional[int] = None,
    rows_gold: Optional[int] = None,
):
    """
    Updates row count metrics (bronze/silver/gold) for a pipeline run.
    Only updates fields provided.
    """
    updates = []
    params = []

    if rows_bronze is not None:
        updates.append("rows_bronze = %s")
        params.append(rows_bronze)
    if rows_silver is not None:
        updates.append("rows_silver = %s")
        params.append(rows_silver)
    if rows_gold is not None:
        updates.append("rows_gold = %s")
        params.append(rows_gold)

    if not updates:
        return

    sql_q = f"UPDATE etl_pipeline_runs SET {', '.join(updates)} WHERE run_id = %s;"
    params.append(run_id)

    with conn.cursor() as cur:
        cur.execute(sql_q, tuple(params))

    conn.commit()
    logger.info(
        "Run %s row metrics updated: bronze=%s silver=%s gold=%s",
        run_id, rows_bronze, rows_silver, rows_gold
    )



def start_step(
    conn, 
    run_id: int, 
    dag_id: str, 
    step_name: str
):
    """
    Marks a pipeline step as running for a given run_id.
    Idempotent via ON CONFLICT on (run_id, step_name).
    """
    sql_q = """
        INSERT INTO etl_pipeline_steps (run_id, dag_id, step_name, status, started_at)
        VALUES (%s, %s, %s, 'running', NOW())
        ON CONFLICT (run_id, dag_id, step_name)
        DO UPDATE SET
            status = 'running',
            ended_at = NULL,
            input_rows = 0,
            output_rows = 0,
            error_message = NULL;
    """
    with conn.cursor() as cur:
        cur.execute(sql_q, (run_id, dag_id, step_name))

    conn.commit()
    logger.info("Step started: run_id=%s dag_id=%s step=%s", run_id, dag_id, step_name)


def finish_step(
    conn,
    run_id: int,
    dag_id : str,
    step_name: str,
    status: str,
    input_rows: int = 0,
    output_rows: int = 0,
    error_message: Optional[str] = None,
):
    """
    Marks a pipeline step as success or failed with row metrics.
    Records timing and optional error message.
    """
    if status not in ("success", "failed"):
        raise ValueError("finish_step status must be 'success' or 'failed'")

    sql_q = """
        UPDATE etl_pipeline_steps
        SET status = %s,
            ended_at = NOW(),
            input_rows = %s,
            output_rows = %s,
            error_message = %s
        WHERE run_id = %s
        AND dag_id = %s AND step_name = %s;
    """

    with conn.cursor() as cur:
        cur.execute(sql_q, (status, input_rows, output_rows, error_message, run_id, dag_id, step_name))

    conn.commit()
    logger.info("Step finished: run_id=%s step=%s status=%s", run_id, step_name, status)