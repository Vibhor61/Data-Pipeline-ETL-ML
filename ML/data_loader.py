from collections import defaultdict
import logging
import shutil
import pandas as pd
import hashlib
import gc
import psutil
import json
import os
import joblib

from sklearn.datasets import dump_svmlight_file
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq

from utils.db import get_connection
from ML.preprocess import preprocess, transform, CATEGORICAL_COLS, ALL_COLS

logger = logging.getLogger(__name__)
BASE_DATE = pd.to_datetime("2011-01-29")
TARGET_COL = "sales"

CURSOR_BATCH_SIZE = 10_000

def get_memory_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def compute_hash(payload: Dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass
class DataLoader:
    run_id: str
    pipeline_name: str  # ETL here
    run_date: str       # Airflow ds
    table_name: str     # gold table
    date_column: str    # typically "run_date"
    feature_version: str 
    output_dir: str


def fetch_etl_run(conn, run_id:str) -> pd.Series:
    query = """
        SELECT * FROM etl_pipeline_runs
        WHERE run_id = %s
    """

    df = pd.read_sql_query(query, conn, params=[run_id])

    if df.empty:
        raise ValueError(f"No ETL run found for run_id={run_id}")

    row = df.iloc[0]

    if row["status"] != "success":
        raise ValueError(f"ETL run {run_id} not successful. Status={row['status']}")

    return row


def compute_time_split(start_date: pd.Timestamp, end_date: pd.Timestamp):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    total_days = (end_date - start_date).days
    train_end = start_date + pd.Timedelta(days=int(total_days * 0.7))
    val_end = start_date + pd.Timedelta(days=int(total_days * 0.85))
    return train_end, val_end


def clean_dir(dataset_dir: str):
    if not os.path.exists(dataset_dir):
        logger.info("Dataset dir doesn't exist no cleanup needed")
        return 
    
    files = [
        "train.parquet","val.parquet","test.parquet",
        "train.parquet.tmp","val.parquet.tmp","test.parquet.tmp",
        "train.libsvm","val.libsvm","test.libsvm",
        "encoders.pkl","features.json","metadata.json"
    ]

    try: 
        for file in files:
            file_path = os.path.join(dataset_dir,file)
            if(os.path.exists(file_path)):
                os.remove(file_path)
                logger.info(f"Removed file: {file_path}")
    except Exception as e:
        logger.info(f"Error removing file {file_path}: {e}")
        raise


def finalize_tmp_files(tmp_paths: Dict[str, str], final_paths: Dict[str, str]):
    for k in tmp_paths:
        os.rename(tmp_paths[k], final_paths[k])


def build_encoder(conn, cfg: DataLoader, start_date, train_end_date):

    logger.info(f"Fitting encoder for {start_date} to {train_end_date}")
    
    vocab = defaultdict(set)

    cols = ", ".join(CATEGORICAL_COLS)
    query = f"""
        SELECT {cols} 
        FROM {cfg.table_name}
        WHERE {cfg.date_column} BETWEEN %s AND %s
    """

    cursor = conn.cursor(name="encoder_cursor")
    cursor.itersize = CURSOR_BATCH_SIZE
    cursor.execute(query, (start_date, train_end_date))

    columns = CATEGORICAL_COLS
    logger.info("Streaming data for encoder fitting")

    while True:
        batch = cursor.fetchmany(CURSOR_BATCH_SIZE)

        if not batch:
            break

        df = pd.DataFrame(batch, columns=columns)

        for col in CATEGORICAL_COLS:
            vocab[col].update(
                df[col]
                .dropna()
                .astype(str)
                .unique()
            )

        del df
    
    cursor.close()

    encoders = {}
    for col in CATEGORICAL_COLS:
        values = sorted(vocab[col])

        mapping = {v: i for i, v in enumerate(values)}
        mapping["__UNK__"] = len(mapping)

        encoders[col] = mapping

    gc.collect()

    logger.info(f"Encoder fitted for {start_date} to {train_end_date}")
    return encoders


def load_gold_dataset(conn, cfg: DataLoader, start_date, end_date, train_end, val_end, tmp_paths, libsvm_paths, encoders ):
    columns = ALL_COLS
    query = f"""
        SELECT {', '.join(ALL_COLS)}
        FROM {cfg.table_name}
        WHERE {cfg.date_column}
            BETWEEN %s AND %s
        ORDER BY item_id, store_id, {cfg.date_column}
    """

    cursor = conn.cursor(name="dataset_cursor")
    cursor.itersize = CURSOR_BATCH_SIZE
    cursor.execute(query, (start_date, end_date))
    
    FEATURES = None
    SCHEMA = None
    parquet_writers = {"train": None, "val": None, "test": None}
    libsvm_files = {"train": None, "val": None, "test": None}

    row_counts = {"train": 0, "val": 0, "test": 0}

    initial_memory = get_memory_usage_mb()
    logger.info(f"Streaming dataset initial memory usage: {initial_memory:.2f} MB")

    batch_num = 0
    while True:
        batch = cursor.fetchmany(CURSOR_BATCH_SIZE)
        if not batch:
            break
        
        logger.info(f"Batch size = {len(batch)} RAM = {get_memory_usage_mb():.2f}MB")
        df_batch = pd.DataFrame(batch, columns=columns)
        
        logger.info(f"Batch loaded into DataFrame RAM = {get_memory_usage_mb():.2f}MB")
        df_batch = preprocess(df_batch)
        
        logger.info(f"Batch preprocessed RAM = {get_memory_usage_mb():.2f}MB")
        df_batch["run_date"] = pd.to_datetime(df_batch[cfg.date_column])
        df_batch["partition"] = df_batch[cfg.date_column].apply(
            lambda x: "train" if x <= train_end else ("val" if x <= val_end else "test")
        )
        df_batch = df_batch.drop(columns=["run_date"])

        encoded_df = transform(df_batch, encoders)
        logger.info(f"Batch encoded RAM = {get_memory_usage_mb():.2f}MB")
        
        if FEATURES is None:
            
            EXCLUDE = {TARGET_COL, cfg.date_column, "partition"}
            FEATURES = [c for c in encoded_df.columns if c not in EXCLUDE]

        if SCHEMA is None:
            SCHEMA = pa.Schema.from_pandas(df_batch.drop(columns=["partition"]))

        for partition in ["train", "val", "test"]:
            
            parquet_df = df_batch[df_batch["partition"] == partition].drop(columns=["partition"])
            logger.info(f"Partition {partition} size: {len(parquet_df)}")
            
            if not parquet_df.empty:
                if parquet_writers[partition] is None:
                    parquet_writers[partition] = pq.ParquetWriter(
                        tmp_paths[partition],
                        SCHEMA
                    )
                                
                table = pa.Table.from_pandas(
                    parquet_df,
                    schema=SCHEMA,
                    preserve_index=False,
                    safe=True
                )

                parquet_writers[partition].write_table(table)
                row_counts[partition] += len(parquet_df)
               
            libsvm_df = encoded_df[encoded_df["partition"] == partition].drop(columns=["partition"])
            logger.info(f"Partition {partition} encoded size: {len(libsvm_df)}")
            
            if not libsvm_df.empty:
                if libsvm_files[partition] is None:
                    libsvm_files[partition] = open(libsvm_paths[partition], 'ab')
                
                feature_matrix = libsvm_df[FEATURES].fillna(0)
                dump_svmlight_file(feature_matrix, libsvm_df[TARGET_COL], libsvm_files[partition], zero_based=True)
        
        batch_num += 1

        del df_batch
        del encoded_df

        if batch_num % 50 == 0:
            gc.collect()
        logger.info(f"[GC_CLEANUP] batch={batch_num} RAM={get_memory_usage_mb():.2f} MB")

    for w in parquet_writers.values():
        if w:
            w.close()
    
    for f in libsvm_files.values():
        if f:
            f.close()

    final_memory = get_memory_usage_mb()
    logger.info(
        f"Final memory usage: {final_memory:.2f} MB (delta: {final_memory - initial_memory:.2f} MB)"
    )
    return columns, row_counts, FEATURES


def build_dataset_cfg(cfg: DataLoader) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    
    try:
        with get_connection() as conn:

            # 1. validate ETL run
            fetch_etl_run(conn, cfg.run_id)

            # 2. load gold dataset
            run_date = pd.to_datetime(cfg.run_date)
            end_date = run_date

            if run_date - pd.Timedelta(days=365) < BASE_DATE:
                start_date = BASE_DATE
            else:
                start_date = run_date - pd.Timedelta(days=365)

            train_end, val_end = compute_time_split(
                start_date, end_date
            )

            build_id = compute_hash({
                "pipeline_name": cfg.pipeline_name,
                "run_date": cfg.run_date,
                "feature_version": cfg.feature_version,
                "run_id": cfg.run_id
            })

            build_dir = os.path.join(cfg.output_dir, f"{build_id}_build")
            os.makedirs(build_dir, exist_ok=True)
            clean_dir(build_dir)

            parquet_paths = {
                "train": os.path.join(build_dir, "train.parquet"),
                "val": os.path.join(build_dir, "val.parquet"),
                "test": os.path.join(build_dir, "test.parquet"),
            }

            parquet_tmp_paths = {k: v + ".tmp" for k, v in parquet_paths.items()}
            
            libsvm_paths = {
                "train": os.path.join(build_dir, "train.libsvm"),
                "val": os.path.join(build_dir, "val.libsvm"),
                "test": os.path.join(build_dir, "test.libsvm"),
            }
            libsvm_tmp_paths = {k: v + ".tmp" for k, v in libsvm_paths.items()}
            
            encoders = build_encoder(conn, cfg, start_date, train_end)

            columns, row_counts, features = load_gold_dataset(
                conn, cfg, start_date, end_date,
                train_end, val_end, parquet_tmp_paths, libsvm_tmp_paths, encoders
            )

            total_rows = sum(row_counts.values())

            # 3. schema hash
            schema_hash = compute_hash({
                "columns": columns,
            })

            # 4. feature hash
            feature_hash = compute_hash({
                "feature_version": cfg.feature_version,
                "features": features
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

            final_dataset_dir = os.path.join(cfg.output_dir, dataset_id)

            final_parquet_paths = {
                "train": os.path.join(final_dataset_dir, "train.parquet"),
                "val": os.path.join(final_dataset_dir, "val.parquet"),
                "test": os.path.join(final_dataset_dir, "test.parquet"),
            }

            final_libsvm_paths = {
                "train": os.path.join(final_dataset_dir, "train.libsvm"),
                "val": os.path.join(final_dataset_dir, "val.libsvm"),
                "test": os.path.join(final_dataset_dir, "test.libsvm"),
            }

            encoder_path = os.path.join(build_dir, "encoders.pkl")
            features_path = os.path.join(build_dir, "features.json")
            metadata_path = os.path.join(build_dir, "metadata.json")

            joblib.dump(encoders, encoder_path)
            with open(features_path, "w") as f:
                json.dump(features, f)

            
            metadata = {
                "dataset_id": dataset_id,
                "pipeline_name": cfg.pipeline_name,
                "paths": {
                    "parquet": final_parquet_paths,
                    "libsvm": final_libsvm_paths
                },
                "run_date": cfg.run_date,
                "run_id": cfg.run_id,
                "row_counts": {
                    **row_counts, 
                    "total": total_rows
                },
                "dataset_start_date": str(start_date.date()),
                "dataset_end_date": str(end_date.date()),
                "schema_hash": schema_hash,
                "feature_hash": feature_hash,
                "feature_version": cfg.feature_version,
                "source_table": cfg.table_name,
                "artifacts": {
                    "encoder": os.path.basename(encoder_path),
                    "features": os.path.basename(features_path),
                    "metadata": os.path.basename(metadata_path)
                }
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            finalize_tmp_files(parquet_tmp_paths, parquet_paths)
            finalize_tmp_files(libsvm_tmp_paths, libsvm_paths)
            if os.path.exists(final_dataset_dir):
                shutil.rmtree(final_dataset_dir)

            os.rename(build_dir, final_dataset_dir)

            logger.info("Dataset built: %s", dataset_id)

            return final_parquet_paths, final_libsvm_paths, metadata
        
    except Exception as e:
        logger.error(f"Error building dataset: {e}")
        if build_dir and os.path.exists(build_dir):
            shutil.rmtree(build_dir, ignore_errors=True)
        raise
    