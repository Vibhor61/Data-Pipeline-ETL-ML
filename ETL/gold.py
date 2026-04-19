"""
Gold Layer - ETL Pipeline

Builds feature-engineered dataset from Silver data.

Input: Silver data up to run_date
Output: Fully materialized Gold partition (same run_date)

Core design principles:
- run_date is the source of truth for all data slicing
- ingestion is partition-based (DELETE + INSERT per run_date)
- features are computed using only past data
- strict schema validation before commit

Invariants:
- exactly one Gold partition exists per run_date
- unique grain: (item_id, store_id, d, run_date)
- no future data is used in feature computation
- schema is strictly enforced (no extra/missing columns)
"""
import logging
import os
import argparse
from psycopg2 import sql
import pandas as pd
from ETL.gold_validation import GoldStorageSchema
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
    """
    Builds and validates Gold partition for a given run_date.
    Args:
        conn: active PostgreSQL connection
        run_date (str): execution date in YYYY-MM-DD format
    Returns:
        int: number of rows inserted into Gold partition

    Process:
        - Deletes existing partition for run_date (idempotent overwrite)
        - Extracts Silver data up to run_date for feature computation
        - Computes lag features (1, 3, 7, 14, 28 days)
        - Computes rolling statistics (mean, std) over historical windows
        - Handles missing lags using rolling aggregates
        - Derives additional features (is_weekend, quarter, cold_start)
        - Inserts fully materialized feature set into Gold table
        - Validates inserted data using GoldStorageSchema(Pandera based)

    Raises:
        ValueError: if partition is empty or validation fails
    """
    silver_ident = validate_table_name(SILVER_TABLE)
    gold_ident = validate_table_name(GOLD_TABLE)
 
    delete_sql = sql.SQL("DELETE FROM {} WHERE run_date = %s;").format(gold_ident)
 
    insert_sql = sql.SQL(
        """
        WITH silver_base AS (
        SELECT
            item_id,
            store_id,
            dept_id,
            cat_id,
            state_id,
            d,
            sales,
            sell_price,
            run_date,
            _processed_time,
            _pipeline_version,

            wday,
            weekday,
            month,
            year,
            date,
            wm_yr_wk,

            event_name_1,
            event_type_1,
            event_name_2,
            event_type_2,
            snap_ca,
            snap_tx,
            snap_wi

        FROM silver_table
        WHERE run_date <= %s
        ),

        silver_lags AS (
            SELECT
                *,

                LAG(sales, 1)  OVER (PARTITION BY item_id, store_id ORDER BY run_date)  AS sales_lag_1,
                LAG(sales, 3)  OVER (PARTITION BY item_id, store_id ORDER BY run_date)  AS sales_lag_3,
                LAG(sales, 7)  OVER (PARTITION BY item_id, store_id ORDER BY run_date)  AS sales_lag_7,
                LAG(sales, 14) OVER (PARTITION BY item_id, store_id ORDER BY run_date) AS sales_lag_14,
                LAG(sales, 28) OVER (PARTITION BY item_id, store_id ORDER BY run_date) AS sales_lag_28,

                AVG(sales) OVER (PARTITION BY item_id, store_id ORDER BY run_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)  AS sales_roll_mean_7,
                AVG(sales) OVER (PARTITION BY item_id, store_id ORDER BY run_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS sales_roll_mean_14,
                AVG(sales) OVER (PARTITION BY item_id, store_id ORDER BY run_date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS sales_roll_mean_28,

                STDDEV_SAMP(sales) OVER (PARTITION BY item_id, store_id ORDER BY run_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)  AS sales_roll_std_7,
                STDDEV_SAMP(sales) OVER (PARTITION BY item_id, store_id ORDER BY run_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING) AS sales_roll_std_14,
                STDDEV_SAMP(sales) OVER (PARTITION BY item_id, store_id ORDER BY run_date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS sales_roll_std_28

            FROM silver_base
        ),

        silver_features AS (
            SELECT
                item_id,
                store_id,
                dept_id,
                cat_id,
                state_id,

                d,
                sales,
                sell_price,
                run_date,
                _processed_time,
                _pipeline_version,

                wday,
                weekday,
                month,
                year,
                wm_yr_wk,

                event_name_1,
                event_type_1,
                event_name_2,
                event_type_2,

                COALESCE(snap_ca, 0) AS snap_ca,
                COALESCE(snap_tx, 0) AS snap_tx,
                COALESCE(snap_wi, 0) AS snap_wi,

                COALESCE(sales_lag_1,  sales_roll_mean_7, 0)  AS sales_lag_1,
                COALESCE(sales_lag_3,  sales_roll_mean_7, 0)  AS sales_lag_3,
                COALESCE(sales_lag_7,  sales_roll_mean_7, 0)  AS sales_lag_7,
                COALESCE(sales_lag_14, sales_roll_mean_14, 0) AS sales_lag_14,
                COALESCE(sales_lag_28, sales_roll_mean_28, 0) AS sales_lag_28,

                COALESCE(sales_roll_mean_7,  0) AS sales_roll_mean_7,
                COALESCE(sales_roll_std_7,   0) AS sales_roll_std_7,
                COALESCE(sales_roll_mean_14, 0) AS sales_roll_mean_14,
                COALESCE(sales_roll_std_14,  0) AS sales_roll_std_14,
                COALESCE(sales_roll_mean_28, 0) AS sales_roll_mean_28,
                COALESCE(sales_roll_std_28,  0) AS sales_roll_std_28,

                CASE WHEN EXTRACT(DOW FROM date)::int IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
                EXTRACT(QUARTER FROM date)::int AS quarter,

                CASE
                    WHEN run_date < (SELECT MIN(run_date) FROM silver_base) + INTERVAL '28 days'
                    THEN TRUE ELSE FALSE
                END AS is_cold_start

            FROM silver_lags
        )

        INSERT INTO gold_table (
            item_id, store_id, dept_id, cat_id, state_id,
            d, sales, sell_price, run_date, _processed_time, _pipeline_version,

            wday, weekday, month, year, wm_yr_wk,

            sales_lag_1, sales_lag_3, sales_lag_7, sales_lag_14, sales_lag_28,
            sales_roll_mean_7, sales_roll_std_7,
            sales_roll_mean_14, sales_roll_std_14,
            sales_roll_mean_28, sales_roll_std_28,

            is_weekend, quarter,

            event_name_1, event_type_1,
            event_name_2, event_type_2,

            snap_ca, snap_tx, snap_wi,
            is_cold_start
        )

        SELECT
            item_id, store_id, dept_id, cat_id, state_id,
            d, sales, sell_price, run_date, _processed_time, _pipeline_version,

            wday, weekday, month, year, wm_yr_wk,

            sales_lag_1, sales_lag_3, sales_lag_7, sales_lag_14, sales_lag_28,
            sales_roll_mean_7, sales_roll_std_7,
            sales_roll_mean_14, sales_roll_std_14,
            sales_roll_mean_28, sales_roll_std_28,

            is_weekend, quarter,

            event_name_1, event_type_1,
            event_name_2, event_type_2,

            snap_ca, snap_tx, snap_wi,
            is_cold_start
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
        
        cur.execute(insert_sql, (run_date,run_date,))
        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {} WHERE run_date = %s").format(gold_ident),
            (run_date,),
        )
        inserted = cur.fetchone()[0]

        if inserted == 0:
            raise ValueError(f"Gold partition is empty for run_date={run_date}")

        cur.execute(
            sql.SQL("SELECT * FROM {} WHERE run_date = %s").format(gold_ident),
            (run_date,)
        )

        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()

        df = pd.DataFrame(rows, columns=cols)
        print(df["_processed_time"].dtype)
        print(df["run_date"].dtype)
        print(df[["_processed_time", "run_date"]].head(5))
        logger.info("Validating gold schema for run_date=%s", run_date)
        GoldStorageSchema.validate(df)
        logger.info("Validation passed for run_date=%s", run_date)
    
    logger.info("Built gold partition for run_date=%s with %s rows", run_date, inserted)
    return inserted


def run_gold(run_date: str) -> int:
    """
    Executes Gold ETL pipeline for a given run_date.
    Args:
        run_date (str): execution date in YYYY-MM-DD format
    Returns:
        int: number of rows inserted into Gold table

    Process:
        - Establishes database connection
        - Builds Gold partition
        - Commits transaction on success
        - Rolls back on failure
    """
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
