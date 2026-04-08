import os
import json
import tempfile
import numpy as np
import pandas as pd
import logging
import joblib

import mlflow
import mlflow.lightgbm
import mlflow.xgboost

from sklearn.preprocessing import LabelEncoder

from utils.ml_helpers import log_prediction_run
from utils.db import get_connection
from utils.features import CATEGORICAL_COLS, preprocess, transform

logger = logging.getLogger(__name__)

TARGET_COL = "sales"
DATE = ["run_date"]


def fit_encoders(train):
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        train[col] = train[col].astype(str)
        le.fit(train[col])
        encoders[col] = le

    return encoders


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