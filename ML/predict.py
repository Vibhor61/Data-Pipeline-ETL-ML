import os
import json
import tempfile
import pandas as pd
import logging
import joblib
import argparse

import mlflow
import mlflow.lightgbm
import mlflow.xgboost

from sklearn.preprocessing import LabelEncoder

from utils.ml_helpers import log_prediction_run
from utils.db import get_connection
from preprocess import CATEGORICAL_COLS, preprocess, transform

logger = logging.getLogger(__name__)

TARGET_COL = "sales"
DATE = ["run_date"]


def load_artifacts(mlflow_run_id: str):
    client = mlflow.tracking.MlflowClient()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        client.download_artifacts(mlflow_run_id, "encoders.pkl", tmpdir)
        client.download_artifacts(mlflow_run_id, "features.json", tmpdir)

        encoders = joblib.load(os.path.join(tmpdir, "encoders.pkl"))

        with open(os.path.join(tmpdir, "features.json"), "r") as f:
            features = json.load(f)

    return encoders, features


def load_model(run_id: str, best_model: str):
    if best_model == "lgb":
        model_uri = f"runs:/{run_id}/lgb_model"
        return mlflow.lightgbm.load_model(model_uri)
    elif best_model == "xgb":
        model_uri = f"runs:/{run_id}/xgb_model"
        return mlflow.xgboost.load_model(model_uri)
    else:
        return None
    

def predict_pipeline(df: pd.DataFrame, run_id: str, dataset_id: str):

    logger.info("Prediction pipeline started for mlflfow_run_id=%s and dataset_id=%s", mlflow_run_id, dataset_id)
    mlflow.set_tracking_uri("http://mlflow:5000")

    conn = get_connection()
    try:
        query = """
            SELECT mlflow_run_id, model_name
            FROM ml_runs
            WHERE run_id = %s AND stage = 'train'
        """

        with conn.cursor() as cur:
            cur.execute(query, (run_id,))
            row = cur.fetchone()

        if not row:
            raise ValueError(f"No training record found for run_id={run_id}")

        mlflow_run_id, best_model_name = row

    finally:
        conn.close()
    # Silent failure if trained on some dataset but evaluating on other without checking   
    run = mlflow.get_run(mlflow_run_id)
    trained_dataset_id = run.data.params.get("dataset_id")

    if trained_dataset_id != dataset_id:
        raise ValueError(
            f"Dataset mismatch: model trained on {trained_dataset_id}, "
            f"but prediction requested on {dataset_id}"
        )
    
    encoders, FEATURES = load_artifacts(mlflow_run_id)

    best_model_name = run.data.params.get("best_model")
    if best_model_name == "lgb":
        model = mlflow.lightgbm.load_model(f"runs:/{mlflow_run_id}/best_model")
    elif best_model_name == "xgb":
        model = mlflow.xgboost.load_model(f"runs:/{mlflow_run_id}/best_model")
    else:
        raise ValueError(f"Unknown model type: {best_model_name}")
    
    df = preprocess(df)
    df = transform(df, encoders)

    X = df[FEATURES]

    preds = model.predict(X)
    df["prediction"] = preds

    logger.info("Prediction completed rows=%s", len(df))
   
    conn = None
    try:
        conn = get_connection()
        log_prediction_run(
            conn=conn,
            run_id=run_id,
            dataset_id=dataset_id,
            mlflow_run_id=mlflow_run_id,
            model_name=best_model_name,
            prediction_count=len(df)
        )
    finally:
        if conn:
            conn.close()

    return df