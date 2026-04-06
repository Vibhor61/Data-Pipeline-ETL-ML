import os
import json
import tempfile
import numpy as np
import pandas as pd
import logging
import joblib
import psycopg2

import mlflow
import mlflow.lightgbm
import mlflow.xgboost

from typing import Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from dags.ml_helpers import log_prediction_run

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TARGET_COL = "sales"
DATE = ["run_date"]

CATEGORICAL_COLS = [
    "item_id",
    "store_id",
    "dept_id",
    "cat_id",
    "state_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2"
]

DROP_COLS = [
    "sales",
    "run_date",
    "d",
    "_pipeline_version",
    "_processed_time",
    "weekday",
    "is_cold_start"
]

DB_CONFIG = {
    "host": os.getenv("PGHOST", "postgres"),
    "port": int(os.getenv("PGPORT", "5432")),
    "database": os.getenv("PGDATABASE", "retail_dw"),
    "user": os.getenv("PGUSER", "airflow"),
    "password": os.getenv("PGPASSWORD", "airflow"),
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def preprocess(df: pd.DataFrame):
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("None")

    df = df.drop(columns=DROP_COLS, errors="ignore")
    return df


def fit_encoders(train):
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        train[col] = train[col].astype(str)
        le.fit(train[col])
        encoders[col] = le

    return encoders


def transform(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    df = df.copy()

    for col, le in encoders.items():
        df[col] = df[col].astype(str)

        # handle unseen categories
        mask = ~df[col].isin(le.classes_)
        df.loc[mask, col] = "UNK"

        if "UNK" not in le.classes_:
            le.classes_ = np.append(le.classes_, "UNK")

        df[col] = le.transform(df[col])

    return df


def load_artifacts(run_id: str):
    client = mlflow.tracing.MLflowClient()
    
    with tempfile.TemporaryDirectory as tmpdir:
        client.download_articfacts(run_id, "encoders.pkl", tmpdir)
        client.download_artifacts(run_id, "features.json", tmpdir)

        encoders = joblib.load(os.path.join(tmpdir, "encoders.pkl"))

        with open(os.path.join(tmpdir, "features.json"), "r") as f:
            features = json.load(f)

    return encoders, features


def get_latest_run(experiment_name: str):

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)

    if exp is None:
        raise ValueError("Experiment not found")
    
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found")

    run = runs[0]

    return run.info.run_id, run.data.params.get("best_model")


def load_best_model(run_id: str, best_model: str):

    if best_model == "lgb":
        model_uri = f"runs:/{run_id}/lgb_model"
        return mlflow.lightgbm.load_model(model_uri)
    elif best_model == "xgb":
        model_uri = f"runs:/{run_id}/xgb_model"
        return mlflow.xgboost.load_model(model_uri)
    else:
        return None
    

def predict_pipeline(df: pd.DataFrame, run_id: str, dataset_id: str):

    logger.info("Prediction pipeline started for run_id=%s", run_id)
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_version, mlflow_run_id, model_name
                FROM ml_runs
                WHERE run_id = %s AND stage = 'train'
                ORDER BY created_at DESC
                LIMIT 1;
            """, (run_id,))

            result = cur.fetchone()

            if result is None:
                raise ValueError(f"No trained model found for run_id={run_id}")

            model_version, mlflow_run_id, model_name = result

    finally:
        conn.close()

    logger.info(
        "Loaded model metadata: model=%s version=%s",
        model_name,
        model_version
    )

    client = mlflow.tracking.MlflowClient()

    with tempfile.TemporaryDirectory() as tmpdir:
        client.download_artifacts(mlflow_run_id, "encoders.pkl", tmpdir)
        client.download_artifacts(mlflow_run_id, "features.json", tmpdir)

        encoders = joblib.load(os.path.join(tmpdir, "encoders.pkl"))

        with open(os.path.join(tmpdir, "features.json"), "r") as f:
            FEATURES = json.load(f)


    df = preprocess(df)
    df = transform(df, encoders)

    X = df[FEATURES]

  
    if model_name == "lgb":
        model_uri = f"runs:/{mlflow_run_id}/best_model"
        model = mlflow.lightgbm.load_model(model_uri)
        preds = model.predict(X)

    elif model_name == "xgb":
        model_uri = f"runs:/{mlflow_run_id}/best_model"
        model = mlflow.xgboost.load_model(model_uri)
        preds = model.predict(X)

    else:
        preds = df["sales_lag_7"]

    df["prediction"] = preds

    logger.info("Prediction completed: rows=%s", len(df))

    conn = get_connection()
    try:
        log_prediction_run(
            conn=conn,
            run_id=run_id,
            dataset_id=dataset_id,
            model_version=model_version,
            prediction_count=len(df),
        )
    finally:
        conn.close()

    return df