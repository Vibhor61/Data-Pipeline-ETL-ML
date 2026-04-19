"""
Bronze Layer - ETL Pipeline

This module implements a batch ingestion pipeline that:
- Loads raw sales, calendar, and pricing data
- Transforms sales into a run_date-partitioned bronze table
- Ensures idempotent overwrites per run_date partition

Input: Raw CSV's of M5 data
Output: All tables stored in database(Postgres)

Core design principles:
- run_date is the source of truth for all data slicing
- ingestion is partition-based (DELETE + INSERT per run_date) with no incremental merging logic 

Invariants:
- bronze_sales is fully overwritten per run_date (no partial updates)
- run_date uniquely determines d via BASE_DATE offset
- downstream layer must not assume data correctness 
"""

import argparse
import logging
import os
from pathlib import Path
import datetime

from typing import List, Optional, Tuple
import pandas as pd
from psycopg2 import sql
from psycopg2.extras import execute_values
from utils.db import get_connection

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


SALES_CSV_PATH = Path(
    os.getenv("SALES_CSV_PATH", "/opt/airflow/data/raw/sales_train_validation.csv")
)
CALENDAR_CSV_PATH = Path(
    os.getenv("CALENDAR_CSV_PATH", "/opt/airflow/data/raw/calendar.csv")
)
SELL_PRICES_CSV_PATH = Path(
    os.getenv("SELL_PRICES_CSV_PATH", "/opt/airflow/data/raw/sell_prices.csv")
)

BRONZE_TABLE = os.getenv("BRONZE_TABLE", "bronze_sales")
CALENDAR_TABLE = os.getenv("CALENDAR_TABLE", "calendar")
SELL_PRICES_TABLE = os.getenv("SELL_PRICES_TABLE", "sell_prices")
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "v1")

BASE_DATE = os.getenv("BASE_DATE","2011-01-29")

ALLOWED_TABLES = {"bronze_sales", "calendar", "sell_prices"}


def validate_table_name(table_name: str) -> None:
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table_name}")    


def validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")


def run_date_to_d(run_date: str)-> str:
    """ 
    Converts business run_date into dataset column identifier(d_col).
    Args: 
        run_date (str): execution date in YYYY-MM-DD format
    Returns:
        str: dataset column identifier (e.g., d_1000)
    """
    base = datetime.datetime.strptime(BASE_DATE, "%Y-%m-%d").date()
    target = datetime.datetime.strptime(run_date, "%Y-%m-%d").date()
    # Mapping run_date to d
    delta_days = (target - base).days + 1
    if delta_days <= 0:
        raise ValueError(
            f"business_date {run_date} is before dataset start {BASE_DATE}"
        )

    return f"d_{delta_days}"


def extract_bronze_partition(run_date: str, d_col: str, sales_csv_path: Path) -> pd.DataFrame:
    """
    Extract sales data slice for a given run_date.
    Args:
        run_date (str): execution date
        d_col (str): dataset column identifier
        sales_csv_path (Path): raw sales dataset path
    Returns:
        pd.DataFrame: normalized bronze partition
    """
    required_id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

    usecols = required_id_cols + [d_col]
    sales_df = pd.read_csv(sales_csv_path, usecols=usecols)

    sales_df = sales_df.rename(columns={d_col: "sales"})
    sales_df["d"] = d_col
    sales_df["run_date"] = run_date
    sales_df["_pipeline_version"] = PIPELINE_VERSION

    sales_df = sales_df[
        [
            "item_id",
            "store_id",
            "dept_id",
            "cat_id",
            "state_id",
            "d",
            "sales",
            "run_date",
            "_pipeline_version",
        ]
    ].copy()

    # Safe parsing and no deduplication
    sales_df["sales"] = pd.to_numeric(sales_df["sales"], errors="coerce").fillna(0).astype(int)
    sales_df = sales_df.drop_duplicates(
        subset=["item_id", "store_id", "d", "run_date"]
    )
    
    logger.info(
        "Prepared bronze partition for run_date=%s with %s rows", run_date,len(sales_df),
    )
    return sales_df


def extract_calendar(calendar_csv_path: Path) -> pd.DataFrame:
    """
    Load calendar dimension dataset.
    Args:
        calendar_csv_path (Path): path to calendar CSV
    Returns:
        pd.DataFrame: cleaned calendar dimension table
    """
    df = pd.read_csv(calendar_csv_path)

    df.columns = df.columns.str.lower()

    expected_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_ca", "snap_tx", "snap_wi"
    ]

    df = df[expected_cols].copy()

    for snap_col in ["snap_ca", "snap_tx", "snap_wi"]:
        if snap_col in df:
            df[snap_col] = df[snap_col].fillna(0).astype(int)

    df.columns = df.columns.str.lower()
    logger.info("Loaded calendar dimension with %s rows", len(df))
    return df


def extract_sell_prices(sell_prices_csv_path: Path) -> pd.DataFrame:
    """
    Load sell_prices dataset.
    Args:
        sell_prices_csv_path (Path): path to sell prices CSV
    Returns:
        pd.DataFrame: cleaned sell prices table
    """
    df = pd.read_csv(sell_prices_csv_path)

    expected_cols = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    df = df[expected_cols].copy()

    df["sell_price"] = pd.to_numeric(df["sell_price"], errors="coerce")

    logger.info("Loaded sell prices with %s rows", len(df))
    return df


def overwrite_table(conn, table_name: str, df: pd.DataFrame, columns: List[str]):
    """
    Overwrites full content of a PostgreSQL table. 
    Activated only when either of this file is updatedand passed manually via CLI.
    Strategy used: DELETE + INSERT
    Args:
        conn: PostgreSQL connection
        table_name (str): target table
        df (pd.DataFrame): input data
        columns (list[str]): ordered column mapping
    """
    validate_table_name(table_name)

    delete_sql = sql.SQL("DELETE FROM {}").format(sql.Identifier(table_name))
    insert_sql = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(sql.Identifier(col) for col in columns),
    )

    with conn.cursor() as cur:
        logger.info("Deleting existing rows from %s", table_name)
        cur.execute(delete_sql)

        if df.empty:
            logger.info("No rows to insert into %s", table_name)
            return

        logger.info("Inserting %s rows into %s", len(df), table_name)
        execute_values(cur, insert_sql.as_string(conn), df[columns].values.tolist())


def overwrite_bronze_partition(conn, df: pd.DataFrame, run_date: str):
    """
    Idempotently overwrites bronze table partition for run_date.
    Strategy:
    - Deletes existing partition for run_date
    - Inserts fresh data for same partition
    Args:
        conn: PostgreSQL connection
        df (pd.DataFrame): prepared bronze dataset
        run_date (str): partition key
    """
    validate_table_name(BRONZE_TABLE)

    try:
        delete_sql = sql.SQL("DELETE FROM {} WHERE run_date = %s").format(
            sql.Identifier(BRONZE_TABLE)
        )

        insert_sql = sql.SQL(
            """
            INSERT INTO {} (
                item_id, store_id, dept_id, cat_id, state_id,
                d,
                sales,
                run_date,
                _pipeline_version
            )
            VALUES %s
            """
        ).format(sql.Identifier(BRONZE_TABLE))
    
        with conn.cursor() as cur:
            logger.info("Deleting existing bronze partition for run_date=%s", run_date)
            cur.execute(delete_sql, (run_date,))
            logger.info("Inserting %s rows into bronze partition for run_date=%s", len(df), run_date)
            bronze_cols = [
                "item_id", "store_id", "dept_id", "cat_id", "state_id",
                "d", "sales", "run_date", "_pipeline_version"
            ]
            execute_values(cur, insert_sql.as_string(conn), df[bronze_cols].values.tolist())
        
    except Exception:
        logger.exception("Error preparing SQL statements for bronze load")
        raise


def run_bronze(run_date: str, d_col: str, sales_csv_path:Path, calendar_csv_path: Optional[Path], sell_prices_csv_path: Optional[Path]):
    """
    Executes bronze ETL pipeline.
    Pipeline stages:
    1. Validate input files
    2. Extract sales partition
    3. Load bronze table (idempotent overwrite)
    4. Optionally load calendar dimension
    5. Optionally load sell prices dimension
    6. Commit transaction

    Args:
        run_date (str): execution date
        d_col (str): dataset column identifier
        sales_csv_path (Path): sales dataset path
        calendar_csv_path (Path | None): optional calendar dataset
        sell_prices_csv_path (Path | None): optional sell price dataset

    Raises:
        ValueError: if bronze extraction produces empty dataset
    """
    validate_file(sales_csv_path)

    bronze_df = extract_bronze_partition(run_date, d_col, sales_csv_path)
    if bronze_df.empty:
        raise ValueError(f"No bronze rows prepared for run_date={run_date}")

    conn = get_connection()
    try:
        overwrite_bronze_partition(conn, bronze_df, run_date)

        if calendar_csv_path is not None:
            validate_file(calendar_csv_path)
            
            calendar_df = extract_calendar(calendar_csv_path)
            overwrite_table(
                conn,
                CALENDAR_TABLE,
                calendar_df,
                [   
                    "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
                    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                    "snap_ca", "snap_tx", "snap_wi",
                ],
            )
            logger.info("Calendar dimension updated successfully for run_date=%s", run_date)    
        else:
            logger.info("No calendar CSV provided, skipping calendar dimension update for run_date=%s", run_date)

        if sell_prices_csv_path is not None:
            validate_file(sell_prices_csv_path)
            
            sell_prices_df = extract_sell_prices(sell_prices_csv_path)
            overwrite_table(
                conn,
                SELL_PRICES_TABLE,
                sell_prices_df,
                [
                    "store_id", "item_id", "wm_yr_wk", "sell_price"
                ],
            )
            logger.info("Sell prices dimension updated successfully for run_date=%s", run_date)
        else:  
            logger.info("No sell prices CSV provided, skipping sell prices dimension update for run_date=%s", run_date)

        conn.commit()
        logger.info("Bronze load successful for run_date=%s", run_date)
    except Exception:
        conn.rollback()
        logger.exception("Bronze load failed for run_date=%s", run_date)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-date", required=True, help="Business date for the run")
    parser.add_argument("--sales-csv-path", help="Path to sales_train_validation.csv", default=SALES_CSV_PATH)
    parser.add_argument("--calendar-csv-path", help="Path to calendar.csv for updating or first bootstrapping", default=None)
    parser.add_argument("--sell-prices-csv-path", help="Path to sell_prices.csv for updating or first bootstrapping", default=None)

    args = parser.parse_args()
    
    d_col = run_date_to_d(args.run_date)
    sales_csv_path = Path(args.sales_csv_path)
    calendar_csv_path = Path(args.calendar_csv_path) if args.calendar_csv_path else None
    sell_prices_csv_path = Path(args.sell_prices_csv_path) if args.sell_prices_csv_path else None

    run_bronze(
        run_date=args.run_date,
        d_col = d_col,
        sales_csv_path=sales_csv_path,
        calendar_csv_path=calendar_csv_path,
        sell_prices_csv_path=sell_prices_csv_path,
    )