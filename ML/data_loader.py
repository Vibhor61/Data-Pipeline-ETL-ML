import logging
from psycopg2 import sql
import pandas as pd
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, Any
from utils.db import get_connection
import json
import os

logger = logging.getLogger(__name__)

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
        params=[cfg.run_date, cfg.start_date, cfg.end_date]
    )

    if df.empty:
        raise ValueError("Empty dataset after filtering")

    df = df.sort_values(by=cfg.date_column).reset_index(drop=True)

    return df


def build_dataset(cfg: DataLoader) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    with get_connection() as conn:

        # 1. validate ETL run
        etl_row = fetch_etl_run(conn, cfg.pipeline_name, cfg.run_date)

        # 2. load gold dataset
        run_date = pd.to_datetime(cfg.run_date)
        end_date = run_date
        start_date = run_date - pd.Timedelta(days=1000)

        df = load_gold_dataset(conn, cfg, start_date, end_date)

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

        dataset_path = os.path.join("/opt/airflow/data",f"{dataset_id}.parquet")

        metadata = {
            "dataset_id": dataset_id,
            "dataset_path": dataset_path,
            "pipeline_name": cfg.pipeline_name,
            "run_date": cfg.run_date,
            "row_count": len(df),

            "time_start": str(start_date.date()),
            "time_end": str(end_date.date()),

            "schema_hash": schema_hash,
            "feature_hash": feature_hash,
            "feature_version": cfg.feature_version,
            "source_table": cfg.table_name
        }

        return df, metadata
    