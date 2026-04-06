import numpy as np
import pandas as pd
import mlflow
import os
import psycopg2
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from dags.ml_helpers import log_evaluation_metrics
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DB_CONFIG = {
    "host": os.getenv("PGHOST", "postgres"),
    "port": int(os.getenv("PGPORT", "5432")),
    "database": os.getenv("PGDATABASE", "retail_dw"),
    "user": os.getenv("PGUSER", "airflow"),
    "password": os.getenv("PGPASSWORD", "airflow"),
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)


def evaluate_pipeline(df: pd.DataFrame, run_id: str ,dataset_id: str):

    mlflow.set_tracking_uri("http://mlflow:5000")

    y_true = df["sales"]
    y_pred = df["prediction"]

    test_rmse = rmse(y_true, y_pred)
    test_mae = mean_absolute_error(y_true, y_pred)
    test_mape = mape(y_true, y_pred)
    test_wmape = wmape(y_true, y_pred)

    df["errors"] = y_true - y_pred
    df["abs_errors"] = np.abs(df["errors"])

    error_mean = df["error"].mean()
    error_std = df["error"].std()

    

    
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_version FROM ml_runs WHERE run_id = %s and stage = 'train'
                ORDERED BY created_at DESC
                LIMIT 1;
            """, (run_id,))

        result = cur.fetchone()
        if result is None:
            raise ValueError(f"No model found for run_id={run_id}")
        model_version = result[0]

    finally:
        conn.close()

    
    try:
        conn = get_connection()
        log_evaluation_metrics(
            conn=conn,
            run_id=run_id,
            dataset_id=dataset_id,
            model_version=model_version,
            metrics={
                "rmse": test_rmse,
                "mae": test_mae,
                "mape": test_mape,
                "wmape": test_wmape,
                "error_mean": error_mean,
                "error_std": error_std,
            },
            slice_key="overall",
        )
    finally:
        conn.close()

    store_rmse = df.groupby("store_id").apply(
        lambda x: rmse(x["sales"], x["prediction"])
    )

    worst_stores = store_rmse.sort_values(ascending=False).head(5)
    best_stores = store_rmse.sort_values(ascending=True).head(5)
    
    dept_rmse = df.groupby("dept_id").apply(
        lambda x: rmse(x["sales"], x["prediction"])
    )
    
    worst_depts = dept_rmse.sort_values(ascending=False).head(5)
    best_depts = dept_rmse.sort_values(ascending=True).head(5)

    with mlflow.start_run(run_id=run_id):

        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mape", test_mape)
        mlflow.log_metric("test_wmape", test_wmape)

        mlflow.log_metric("error_mean", error_mean)
        mlflow.log_metric("error_std", error_std)

        mlflow.log_param("worst_stores", ",".join(map(str, worst_stores.index)))
        mlflow.log_param("best_stores", ",".join(map(str, best_stores.index)))

        mlflow.log_param("worst_depts", ",".join(map(str, worst_depts.index)))
        mlflow.log_param("best_depts", ",".join(map(str, best_depts.index)))