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
    """Load encoders and features from MLflow artifacts."""
    client = mlflow.tracking.MlflowClient()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        client.download_artifacts(mlflow_run_id, "encoders.pkl", tmpdir)
        client.download_artifacts(mlflow_run_id, "features.json", tmpdir)

        encoders = joblib.load(os.path.join(tmpdir, "encoders.pkl"))

        with open(os.path.join(tmpdir, "features.json"), "r") as f:
            features = json.load(f)

    return encoders, features


def load_best_model(run_id: str, best_model: str):

    if best_model == "lgb":
        model_uri = f"runs:/{run_id}/lgb_model"
        return mlflow.lightgbm.load_model(model_uri)
    elif best_model == "xgb":
        model_uri = f"runs:/{run_id}/xgb_model"
        return mlflow.xgboost.load_model(model_uri)
    else:
        return None
    

def predict_pipeline(df: pd.DataFrame, mlflow_run_id: str, dataset_id: str):

    logger.info("Prediction pipeline started for run_id=%s and dataset_id=%s", mlflow_run_id, dataset_id)
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Silent failure if trained on some dataset but evaluating on other without checking   
    run = mlflow.get_run(mlflow_run_id)
    trained_dataset_id = run.data.params.get("dataset_id")

    if trained_dataset_id != dataset_id:
        raise ValueError(
            f"Dataset mismatch: model trained on {trained_dataset_id}, "
            f"but prediction requested on {dataset_id}"
        )
    
    encoders, FEATURES = load_artifacts(mlflow_run_id)

    model_uri = f"runs:/{mlflow_run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

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
            run_id=None,  
            dataset_id=dataset_id,
            model_version=mlflow_run_id,  
            prediction_count=len(df),
        )
    finally:
        if conn:
            conn.close()

    return df