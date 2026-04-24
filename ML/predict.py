import os
import json
import tempfile
import pandas as pd
import logging
import joblib

import mlflow
import mlflow.lightgbm
import mlflow.xgboost

from preprocess import preprocess, transform

logger = logging.getLogger(__name__)

TARGET_COL = "sales"


def load_artifacts(mlflow_run_id: str):
    client = mlflow.tracking.MlflowClient()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        client.download_artifacts(mlflow_run_id, "encoders.pkl", tmpdir)
        client.download_artifacts(mlflow_run_id, "features.json", tmpdir)

        encoders = joblib.load(os.path.join(tmpdir, "encoders.pkl"))

        with open(os.path.join(tmpdir, "features.json"), "r") as f:
            features = json.load(f)

    return encoders, features


def load_model(mlflow_run_id: str):

    run = mlflow.get_run(mlflow_run_id)
    best_model = run.data.params.get("best_model")

    if best_model == "lgb":
        model_uri = f"runs:/{mlflow_run_id}/lgb_model"
        return mlflow.lightgbm.load_model(model_uri)
    elif best_model == "xgb":
        model_uri = f"runs:/{mlflow_run_id}/xgb_model"
        return mlflow.xgboost.load_model(model_uri)
    else:
        raise ValueError(f"Unknown best_model: {best_model}")
    

def predict_pipeline(test_path: str, run_id: str, dataset_id: str, train_mlflow_run_id: str):

    logger.info("Prediction pipeline started for run_id=%s, train_mlflow_run_id=%s and dataset_id=%s", run_id, train_mlflow_run_id, dataset_id)

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run found in predict stage")

    # Silent failure if trained on some dataset but evaluating on other without checking   
    run = mlflow.get_run(train_mlflow_run_id)
    best_model_name = run.data.params.get("best_model")

    trained_dataset_id = run.data.params.get("dataset_id")
    if trained_dataset_id != dataset_id:
        raise ValueError(
            f"Dataset mismatch: model trained on {trained_dataset_id}, "
            f"but prediction requested on {dataset_id}"
        )


    test_df = pd.read_parquet(test_path)
    test_df = preprocess(test_df)
    X_test = test_df.drop(columns=[TARGET_COL])

    encoders, features = load_artifacts(train_mlflow_run_id)
    X_test = transform(X_test, encoders)

    model = load_model(train_mlflow_run_id)
    
    preds = model.predict(X_test[features])
    
    pred_df = test_df.copy()
    pred_df["prediction"] = preds

    pred_path = os.path.join(os.path.dirname(test_path), "predictions.parquet")
    pred_df.to_parquet(pred_path, index=False)

    mlflow.log_artifact(pred_path, artifact_path="predictions")
    mlflow.log_metric("prediction_rows", len(pred_df))

    logger.info("Prediction completed rows=%s", len(pred_df))

    return pred_path