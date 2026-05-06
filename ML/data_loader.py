"""
Gold ML Data Loader

Builds training, validation, and test datasets from ETL gold output.

Input: validated ETL gold table and pipeline run metadata
Output: partitioned datasets, dataset metadata, and stable dataset hashes

Core design principles:
- dataset provenance is tracked through ETL run validation
- train/val/test splits are derived from run_date ordering
- schema and feature version hashes enable reproducible datasets
"""

import logging
import psycopg2 
import pandas as pd
import hashlib
from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq

from typing import Dict, Tuple, Optional, Any
from utils.db import get_connection
import json
import os

logger = logging.getLogger(__name__)

BASE_DATE = pd.to_datetime("2011-01-29")

@dataclass
class DataLoader:
    run_id: str
    pipeline_name: str  # ETL here
    run_date: str       # Airflow ds
    table_name: str     # gold table
    date_column: str    # typically "run_date"
    feature_version: str 
    output_dir: str


def compute_hash(payload: Dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()


def fetch_etl_run(conn, run_id:str) -> pd.Series:
    """
    Fetch and validate the exact ETL run chosen by DAG.
    Args:
        conn: active PostgreSQL connection
        run_id: unique identifier for the ETL run (from Airflow DAG)
    Returns:
        pandas.Series: metadata row for the validated ETL run
    Raises:
        ValueError: if no run exists or the ETL run did not succeed
    """
    query = """
        SELECT *
        FROM etl_pipeline_runs
        WHERE run_id = %s
    """

    df = pd.read_sql_query(query, conn, params=[run_id])

    if df.empty:
        raise ValueError(f"No ETL run found for run_id={run_id}")

    row = df.iloc[0]

    if row["status"] != "success":
        raise ValueError(f"ETL run {run_id} not successful. Status={row['status']}")

    return row


def split_dataset(conn, cfg: DataLoader, start_date, end_date):
    rows_query = f"""
        SELECT COUNT(*) FROM {cfg.table_name}
        WHERE {cfg.date_column} BETWEEN %s AND %s
    """

    with conn.cursor() as cur:
        cur.execute(rows_query, (start_date, end_date))
        total_rows = cur.fetchone()[0]

    if total_rows == 0:
        raise ValueError("Empty dataset after filtering")

    train_end = int(total_rows * 0.7)
    val_end = int(total_rows * 0.85)
    return total_rows, train_end, val_end

def load_gold_dataset(conn, cfg: DataLoader, start_date, end_date, train_end, val_end, tmp_paths):
    """
    Load and write the gold dataset partitions from the database into parquet files.
    Args:
        conn: active PostgreSQL connection
        cfg (DataLoader): dataset configuration object
        start_date: inclusive start date for filtering
        end_date: inclusive end date for filtering
        train_end: last row number for the training partition
        val_end: last row number for the validation partition
        tmp_paths: temporary parquet output paths for each partition
    Returns:
        tuple: (columns, row_counts) for the loaded dataset
    Raises:
        ValueError: if the loaded dataset is empty
    """

    query = f"""
        WITH ordered AS (
            SELECT *, ROW_NUMBER() OVER (ORDER BY {cfg.date_column}) AS row_num
            FROM {cfg.table_name}
            WHERE {cfg.date_column} BETWEEN %s AND %s
        )
        SELECT *,
            CASE 
                WHEN row_num <= %s THEN 'train'
                WHEN row_num <= %s THEN 'val'
                ELSE 'test'
            END AS partition
        FROM ordered
        ORDER BY {cfg.date_column}
    """

    cursor = conn.cursor(name="dataset_cursor")
    cursor.itersize = 10000
    cursor.execute(query, (start_date, end_date, train_end, val_end))
    
    columns = [desc[0] for desc in cursor.description]

    writers = {"train": None, "val": None, "test": None}
    row_counts = {"train": 0, "val": 0, "test": 0}

    while True:
        batch = cursor.fetchmany(10000)
        if not batch:
            break
        
        df_batch = pd.DataFrame(batch, columns=columns)

        for partition in ["train", "val", "test"]:
            partition_df = df_batch[df_batch["partition"] == partition].drop(columns=["partition"])
            
            if not partition_df.empty:
                table = pa.Table.from_pandas(partition_df)
                if writers[partition] is None:
                    writers[partition] = pq.ParquetWriter(tmp_paths[partition], table.schema)
                
                writers[partition].write_table(table)
                row_counts[partition] += len(partition_df)

    for w in writers.values():
        if w:
            w.close()

    return columns, row_counts


def atomic_commit(tmp_paths, final_paths):
    for key in final_paths:
        tmp = tmp_paths[key]
        final = final_paths[key]

        if os.path.exists(tmp):
            os.rename(tmp, final)


def clean_dir(dataset_dir: str):
    """
    Remove dangling or corrupted parquet files from a prior failed attempt.
    
    Args:
        dataset_dir: path to the dataset directory (e.g., /opt/airflow/data/datasets/hash123)
    """

    if not os.path.exists(dataset_dir):
        logger.info("Dataset dir doesn't exist no cleanup needed")
        return 
    files = ["train.parquet","val.parquet","test.parquet"]

    try: 
        for file in files:
            file_path = os.path.join(dataset_dir,file)
            if(os.path.exists(file_path)):
                os.remove(file_path)
    except Exception as e:
        logger.info(f"Error removing file {file_path}: {e}")
        raise

def build_dataset_cfg(cfg: DataLoader) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Build dataset parquet file paths and metadata from a dataset configuration.
    Args:
        cfg (DataLoader): dataset build configuration
    Returns:
        Tuple[Dict[str, str], Dict[str, Any]]: partition file paths and dataset metadata
    """
    with get_connection() as conn:

        # 1. validate ETL run
        fetch_etl_run(conn, cfg.run_id)

        # 2. load gold dataset
        run_date = pd.to_datetime(cfg.run_date)
        end_date = run_date

        if run_date - pd.Timedelta(days=1000) < BASE_DATE:
            start_date = BASE_DATE
        else:
            start_date = run_date - pd.Timedelta(days=1000)

        total_rows, train_end, val_end = split_dataset(
            conn, cfg, start_date, end_date
        )

        temp_id = compute_hash({
            "pipeline_name": cfg.pipeline_name,
            "run_date": cfg.run_date,
            "feature_version": cfg.feature_version,
            "row_count": total_rows,
            "run_id": cfg.run_id
        })

        dataset_dir = os.path.join(cfg.output_dir, temp_id)
        os.makedirs(dataset_dir, exist_ok=True)
        clean_dir(dataset_dir)

        final_paths = {
            "train": os.path.join(dataset_dir, "train.parquet"),
            "val": os.path.join(dataset_dir, "val.parquet"),
            "test": os.path.join(dataset_dir, "test.parquet"),
        }

        tmp_paths = {k: v + ".tmp" for k, v in final_paths.items()}

        columns, row_counts = load_gold_dataset(
            conn, cfg, start_date, end_date,
            train_end, val_end, tmp_paths
        )

        atomic_commit(tmp_paths, final_paths)

        # 3. schema hash
        schema_hash = compute_hash({
            "columns": columns,
        })

        # 4. feature hash
        feature_hash = compute_hash({
            "feature_version": cfg.feature_version
        })

        # 5. dataset hash (CORE CONTRACT)
        dataset_id = compute_hash({
            "pipeline_name": cfg.pipeline_name,
            "run_date": cfg.run_date,
            "etl_run_id": cfg.run_id,
            "schema_hash": schema_hash,
            "feature_hash": feature_hash,
            "row_count": total_rows
        })

        metadata = {
            "dataset_id": dataset_id,
            "pipeline_name": cfg.pipeline_name,
            "paths": final_paths,
            "run_date": cfg.run_date,
            "run_id": cfg.run_id,
            "row_counts": {**row_counts, "total": total_rows},
            "dataset_start_date": str(start_date.date()),
            "dataset_end_date": str(end_date.date()),
            "schema_hash": schema_hash,
            "feature_hash": feature_hash,
            "feature_version": cfg.feature_version,
            "source_table": cfg.table_name
        }
        
        logger.info("Dataset built: %s", dataset_id)

        return final_paths, metadata
    