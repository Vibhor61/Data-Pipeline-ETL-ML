import logging
from psycopg2 import sql
import pandas as pd
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
from utils.db import get_connection
import json
import os

logger = logging.getLogger(__name__)

BASE_DATE = pd.to_datetime("2011-29-01")

@dataclass
class DataLoader:
    pipeline_name: str  # ETL here
    run_date: str       # Airflow ds
    table_name: str     # gold table
    date_column: str    # typically "run_date"
    feature_version: str 


def compute_hash(payload: Dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()


def fetch_etl_run(conn, pipeline_name: str, run_date: str):
    query = """
        SELECT *
        FROM etl_pipeline_runs
        WHERE pipeline_name = %s
        AND run_date = %s
    """
    df = pd.read_sql_query(query, conn, params=[pipeline_name, run_date])

    if df.empty:
        raise ValueError("No ETL run found for given pipeline_name + run_date")

    row = df.iloc[0]

    if row["status"] != "success":
        raise ValueError(f"ETL not successful. Status={row['status']}")

    return row


def load_gold_dataset(conn, cfg: DataLoader, start_date, end_date) -> pd.DataFrame:
    query = sql.SQL("""
        SELECT * FROM {table} WHERE 
        {date_col} BETWEEN %s AND %s
    """).format(
        table=sql.Identifier(cfg.table_name),
        date_col=sql.Identifier(cfg.date_column)
    )

    df = pd.read_sql_query(
        query,
        conn,
        params=[cfg.run_date, start_date, end_date]
    )

    if df.empty:
        raise ValueError("Empty dataset after filtering")

    df = df.sort_values(by=cfg.date_column).reset_index(drop=True)

    return df


def split_dataset(df: pd.DataFrame):
    df = df.sort_values("run_date")

    n = len(df)

    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    if train.empty or val.empty or test.empty:
        raise ValueError("Invalid split sizes")

    return train, val, test


def build_dataset_cfg(cfg: DataLoader) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    with get_connection() as conn:

        # 1. validate ETL run
        fetch_etl_run(conn, cfg.pipeline_name, cfg.run_date)

        # 2. load gold dataset
        run_date = pd.to_datetime(cfg.run_date)
        end_date = run_date

        if run_date - pd.Timedelta(days=1000) < BASE_DATE:
            start_date = BASE_DATE
        else:
            start_date = run_date - pd.Timedelta(days=1000)

        df = load_gold_dataset(conn, cfg, start_date, end_date)
        train_df, val_df, test_df = split_dataset(df)

        # 3. schema hash
        schema_hash = compute_hash({
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns}
        })

        # 4. feature hash
        feature_hash = compute_hash({
            "feature_version": cfg.feature_version
        })

        # 5. dataset hash (CORE CONTRACT)
        dataset_id = compute_hash({
            "pipeline_name": cfg.pipeline_name,
            "run_date": cfg.run_date,
            "schema_hash": schema_hash,
            "feature_hash": feature_hash,
            "row_count": len(df)
        })

        dataset_dir = os.path.join(cfg.output_dir, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)

        train_path = os.path.join(dataset_dir, "train.parquet")
        val_path = os.path.join(dataset_dir, "val.parquet")
        test_path = os.path.join(dataset_dir, "test.parquet")

        metadata = {
            "dataset_id": dataset_id,
            "pipeline_name": cfg.pipeline_name,
            "paths": {
                "train": train_path,
                "val": val_path,
                "test": test_path
            },
            
            "run_date": cfg.run_date,

            "row_counts": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
                "total": len(df)
            },

            "dataset_start_date": str(start_date.date()),
            "dataset_end_date": str(end_date.date()),

            "schema_hash": schema_hash,
            "feature_hash": feature_hash,
            "feature_version": cfg.feature_version,
            "source_table": cfg.table_name
        }
        
        logger.info("Dataset built: %s", dataset_id)
        
        return {
            "train": train_df,
            "val": val_df,
            "test": test_df
            }, metadata
    