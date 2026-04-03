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

from typing import Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

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
    

def predict_pipeline(df: pd.DataFrame):

    mlflow.set_tracking_uri("htt[://mlflow:5000")
    logger.info("Prediction pipeline started fetching best model from MLflow")
    
    run_id, best_model = get_latest_run(os.getenv("ml_run_id"))
    logger.info(f"Using run_id={run_id}, best_model={best_model}")
    
    encoders, FEATURES = load_artifacts(run_id)
    df = preprocess(df)
    df = transform(df, encoders)

    X = df[FEATURES]

    if best_model in ["lgb","xgb"]:
        model = load_best_model(run_id, best_model)
        preds = model.predict(X)
    else:
        preds = df["sales_lag_7"]

    df["prediction"] = preds

    logger.info("Prediction completed")

    return df