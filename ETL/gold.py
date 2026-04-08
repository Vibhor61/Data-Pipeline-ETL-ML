import logging
import os
import argparse
from psycopg2 import sql
from utils.db import get_connection

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


SILVER_TABLE = os.getenv("SILVER_TABLE", "silver_table")
GOLD_TABLE = os.getenv("GOLD_TABLE", "gold_table")


ALLOWED_TABLES = {"silver_table","gold_table"}
def validate_table_name(table_name: str) -> sql.Identifier:
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table_name}")
    return sql.Identifier(table_name)


def build_gold_partition(conn, run_date: str) -> int:
    silver_ident = validate_table_name(SILVER_TABLE)
    gold_ident = validate_table_name(GOLD_TABLE)

    delete_sql = sql.SQL("DELETE FROM {} WHERE run_date = %s;").format(gold_ident)

    insert_sql = sql.SQL(
        """
        WITH silver AS (SELECT * FROM {silver} WHERE run_date <= %s),

        dataset_start_date AS (SELECT MIN(run_date) from {silver}),

        silver_lags AS (
            SELECT *,
                LAG(sales, 1) OVER (PARTITION BY item_id, store_id ORDER BY date) AS sales_lag_1,
                LAG(sales, 3) OVER (PARTITION BY item_id, store_id ORDER BY date) AS sales_lag_3,
                LAG(sales, 7) OVER (PARTITION BY item_id, store_id ORDER BY date) AS sales_lag_7,
                LAG(sales, 14) OVER (PARTITION BY item_id, store_id ORDER BY date) AS sales_lag_14,
                LAG(sales, 28) OVER (PARTITION BY item_id, store_id ORDER BY date) AS sales_lag_28

                AVG(sales) OVER (PARTITION BY item_id, store_id ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS sales_roll_mean_7,
                AVG(sales) OVER (PARTITION BY item_id, store_id ORDER BY date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS sales_roll_mean_14,
                AVG(sales) OVER (PARTITION BY item_id, store_id ORDER BY date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS sales_roll_mean_28,
                STDDEV_SAMP(sales) OVER (PARTITION BY item_id, store_id ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS sales_roll_std_7,
                STDDEV_SAMP(sales) OVER (PARTITION BY item_id, store_id ORDER BY date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS sales_roll_std_14,
                STDDEV_SAMP(sales) OVER (PARTITION BY item_id, store_id ORDER BY date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS sales_roll_std_28,            
            
                CASE 
                    WHEN date < (SELECT MIN(run_date) FROM dataset_start_date) + INTERVAL '28 days' 
                    THEN TRUE ELSE FALSE 
                END AS is_cold_start
            FROM silver 
        ),
        silver_features AS (
            SELECT
                item_id, store_id, dept_id, cat_id, state_id, d, sales, sell_price, run_date, _processed_time, _pipeline_version,
                
                COALESCE(sales_lag_1,  sales_roll_mean_7, 0)  AS sales_lag_1,
                COALESCE(sales_lag_3,  sales_roll_mean_7, 0)  AS sales_lag_3,
                COALESCE(sales_lag_7,  sales_roll_mean_7, 0)  AS sales_lag_7,
                COALESCE(sales_lag_14, sales_roll_mean_14, 0) AS sales_lag_14,
                COALESCE(sales_lag_28, sales_roll_mean_28, 0) AS sales_lag_28,

                COALESCE(sales_roll_mean_7, 0)  AS sales_roll_mean_7,
                COALESCE(sales_roll_std_7, 0)   AS sales_roll_std_7,
                COALESCE(sales_roll_mean_14, 0) AS sales_roll_mean_14,
                COALESCE(sales_roll_std_14, 0)  AS sales_roll_std_14,
                COALESCE(sales_roll_mean_28, 0) AS sales_roll_mean_28,
                COALESCE(sales_roll_std_28, 0)  AS sales_roll_std_28,
                
                CASE WHEN wday IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
                EXTRACT(QUARTER FROM date)::int AS quarter,
                month, wday, weekday, year, event_name_1, event_type_1, event_name_2, event_type_2,
                
                COALESCE(snap_CA, 0) AS snap_CA, 
                COALESCE(snap_TX, 0) AS snap_TX, 
                COALESCE(snap_WI, 0) AS snap_WI,
                wm_yr_wk,
                is_cold_start
            FROM silver_lags
        )
        INSERT INTO {gold} (
            item_id, store_id, dept_id, cat_id, state_id, d, sales, sell_price, run_date, _processed_time, _pipeline_version,
            sales_lag_1, sales_lag_3, sales_lag_7, sales_lag_14, sales_lag_28,
            sales_roll_mean_7, sales_roll_std_7, sales_roll_mean_14, sales_roll_std_14, sales_roll_mean_28, sales_roll_std_28,
            is_weekend, quarter, month, wday, weekday, year, 
            event_name_1, event_type_1, event_name_2, event_type_2,
            snap_CA, snap_TX, snap_WI, wm_yr_wk, is_cold_start
        )
        SELECT
            item_id, store_id, dept_id, cat_id, state_id, d, sales, sell_price, run_date, _processed_time, _pipeline_version,
            sales_lag_1, sales_lag_3, sales_lag_7, sales_lag_14, sales_lag_28,
            sales_roll_mean_7, sales_roll_std_7, sales_roll_mean_14, sales_roll_std_14, sales_roll_mean_28, sales_roll_std_28,
            is_weekend, quarter, month, wday, weekday, year, 
            event_name_1, event_type_1, event_name_2, event_type_2,
            snap_CA, snap_TX, snap_WI, wm_yr_wk, is_cold_start
        FROM silver_features
        WHERE run_date = %s;
        """
    ).format(
        silver=silver_ident,
        gold=gold_ident,
    )

    with conn.cursor() as cur:
        logger.info("Deleting existing gold partition for run_date=%s", run_date)
        cur.execute(delete_sql, (run_date,))

        logger.info("Inserting gold partition for run_date=%s", run_date)
        
        cur.execute(insert_sql, (run_date,run_date))
        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {} WHERE run_date = %s").format(gold_ident),
            (run_date,),
        )
        inserted = cur.fetchone()[0]

        if inserted == 0:
            raise ValueError(f"Gold partition is empty for run_date={run_date}")

    logger.info("Built gold partition for run_date=%s with %s rows", run_date, inserted)
    return inserted


def run_gold(run_date: str) -> int:
    conn = get_connection()
    try:
        inserted_rows = build_gold_partition(conn, run_date)
        conn.commit()

        logger.info(
            "Gold build completed successfully for run_date=%s, rows=%s",
            run_date,
            inserted_rows,
        )
        return inserted_rows
    except Exception:
        conn.rollback()
        logger.exception("Gold build failed for run_date=%s", run_date)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-date", required=True,  help="Business date for the run")
    args = parser.parse_args()

    run_gold(args.run_date)
