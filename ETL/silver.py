import logging
import os
import argparse

import psycopg2
from psycopg2 import sql

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

DB_CONFIG = {
    "host": os.getenv("PGHOST", "postgres"),
    "port": int(os.getenv("PGPORT", "5432")),
    "database": os.getenv("PGDATABASE", "retail_dw"),
    "user": os.getenv("PGUSER", "airflow"),
    "password": os.getenv("PGPASSWORD", "airflow"),
}

BRONZE_TABLE = os.getenv("BRONZE_TABLE", "bronze_sales")
SILVER_TABLE = os.getenv("SILVER_TABLE", "silver_table")
CALENDAR_TABLE = os.getenv("CALENDAR_TABLE", "calendar")
SELL_PRICES_TABLE = os.getenv("SELL_PRICES_TABLE", "sell_prices")
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "v1")


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


ALLOWED_TABLES = {"bronze_sales", "silver_table", "calendar", "sell_prices"}
def validate_table_name(table_name: str) -> sql.Identifier:
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table_name}")
    return sql.Identifier(table_name)


def build_silver_partition(conn, run_date: str) -> int:
    bronze_ident = validate_table_name(BRONZE_TABLE)
    silver_ident = validate_table_name(SILVER_TABLE)
    calendar_ident = validate_table_name(CALENDAR_TABLE)
    sell_prices_ident = validate_table_name(SELL_PRICES_TABLE)

    delete_sql = sql.SQL("DELETE FROM {} WHERE run_date = %s;").format(silver_ident)

    insert_sql = sql.SQL(
        """
        WITH bronze AS (
            SELECT *
            FROM {bronze}
            WHERE run_date = %s
        )
        INSERT INTO {silver} (
            run_date,
            _processed_time,
            _pipeline_version,
            store_id,
            item_id,
            d,
            date,
            dept_id,
            cat_id,
            state_id,
            sales,
            sell_price,
            wm_yr_wk,
            weekday,
            wday,
            month,
            year,
            event_name_1,
            event_type_1,
            event_name_2,
            event_type_2,
            snap_CA,
            snap_TX,
            snap_WI
        )
        SELECT
            %s::date AS run_date,
            NOW() AS _processed_time,
            %s AS _pipeline_version,
            b.store_id,
            b.item_id,
            b.d,
            c.date,
            b.dept_id,
            b.cat_id,
            b.state_id,
            b.sales,
            s.sell_price,
            c.wm_yr_wk,
            c.weekday,
            c.wday,
            c.month,
            c.year,
            c.event_name_1,
            c.event_type_1,
            c.event_name_2,
            c.event_type_2,
            COALESCE(c.snap_CA, 0) AS snap_CA,
            COALESCE(c.snap_TX, 0) AS snap_TX,
            COALESCE(c.snap_WI, 0) AS snap_WI
        FROM bronze b
        JOIN {calendar} c ON b.d = c.d
        LEFT JOIN {sell_prices} s
            ON b.store_id = s.store_id
            AND b.item_id = s.item_id
            AND c.wm_yr_wk = s.wm_yr_wk
        ;
        """
    ).format(
        bronze=bronze_ident,  
        silver=silver_ident,
        calendar=calendar_ident,
        sell_prices=sell_prices_ident,
    )

    with conn.cursor() as cur:
        logger.info("Deleting existing silver partition for run_date=%s", run_date)
        cur.execute(delete_sql, (run_date,))

        logger.info("Inserting silver partition for run_date=%s", run_date)
        cur.execute(insert_sql, (run_date, run_date, PIPELINE_VERSION))

        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {} WHERE run_date = %s").format(silver_ident),
            (run_date,),
        )
        inserted = cur.fetchone()[0]

        if inserted == 0:
            raise ValueError(f"Silver partition is empty for run_date={run_date}")

    logger.info("Built silver partition for run_date=%s with %s rows", run_date, inserted)
    return inserted


def run_silver(run_date: str) -> int:
    conn = get_connection()
    try:
        inserted_rows = build_silver_partition(conn, run_date)
        conn.commit()

        logger.info(
            "Silver build completed successfully for run_date=%s, rows=%s",
            run_date,
            inserted_rows,
        )
        return inserted_rows
    except Exception:
        conn.rollback()
        logger.exception("Silver build failed for run_date=%s", run_date)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-date", required=True,  help="Business date for the run")
    args = parser.parse_args()

    run_silver(args.run_date)
