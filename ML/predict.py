"""
Prediction Module - ML Pipeline
 
Loads trained MLflow models and generates batch predictions
using LibSVM feature datasets.

Outputs:
- Prediction parquet artifact
- Baseline comparison predictions
- MLflow prediction artifacts and metrics
 
Core design principles:
- Memory-efficient batch prediction for large datasets
- Full lineage preservation via run_id + dataset_id
- Artifact logging for reproducibility
"""
import os
import json
import pandas as pd
import logging
import joblib
import gc

import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import xgboost as xgb

from sklearn.datasets import load_svmlight_file

logger = logging.getLogger(__name__)

TARGET_COL = "sales"


def load_artifacts_from_dataset(dataset_dir: str):
    """
    Load dataset artifacts required for inference.
    Includes:
    - encoders
    - feature registry
    - dataset metadata
    """
    encoder_path = os.path.join(dataset_dir, "encoders.pkl")
    features_path = os.path.join(dataset_dir, "features.json")
    metadata_path = os.path.join(dataset_dir, "metadata.json")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoders not found in {dataset_dir}")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features not found in {dataset_dir}")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found in {dataset_dir}")
    
    encoders = joblib.load(encoder_path)
    
    with open(features_path, "r") as f:
        features = json.load(f)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return encoders, features, metadata


def load_model(mlflow_run_id: str):
    """
    Load the best-performing model from MLflow.
    """
    run = mlflow.get_run(mlflow_run_id)
    best_model = run.data.params.get("best_model")

    if best_model is None:
        raise ValueError(f"Missing best_model parameter in MLflow run {mlflow_run_id}")

    if best_model == "lightgbm":
        return mlflow.lightgbm.load_model(f"runs:/{mlflow_run_id}/model_lgb"), best_model
    elif best_model == "xgboost":
        return mlflow.xgboost.load_model(f"runs:/{mlflow_run_id}/model_xgb"), best_model
    else:
        raise ValueError(f"Unknown best_model: {best_model}")
    

def predict_pipeline(test_path: str, run_id: str, dataset_id: str, mlflow_run_id: str, dataset_dir: str = None):
    """
    Execute batch prediction workflow.

    Pipeline stages:
    1. Load dataset artifacts
    2. Load best model from MLflow
    3. Generate predictions on test dataset
    4. Save prediction artifacts
    5. Log outputs to MLflow
    """
    logger.info("Prediction pipeline started for run_id=%s, mlflow_run_id=%s and dataset_id=%s", run_id, mlflow_run_id, dataset_id)

    if dataset_dir is None:
        dataset_dir = os.path.dirname(test_path)    

    _, features, _ = load_artifacts_from_dataset(dataset_dir)
    logger.info("Loaded dataset artifacts from %s", dataset_dir)

    model, model_type = load_model(mlflow_run_id)
    logger.info("Loaded model type %s from MLflow run %s", model_type, mlflow_run_id)

    test_libsvm_path = os.path.join(dataset_dir, "test.libsvm")

    X_test, y_test = load_svmlight_file(
        test_libsvm_path,
        n_features=len(features),
        zero_based=True
    )

    logger.info("Loaded test libsvm from %s with %s rows", test_libsvm_path, X_test.shape[0])
    logger.info("X_test shape: %s, model expects: %s", X_test.shape, len(features))

    if model_type == 'xgboost':
        dtest = xgb.DMatrix(X_test)
        preds = model.predict(dtest)
        del dtest
    elif model_type == 'lightgbm':
        preds = model.predict(X_test)

    test_df = pd.read_parquet(os.path.join(dataset_dir, "test.parquet"))

    prediction_df = pd.DataFrame({
        "prediction": preds,
        "baseline_prediction": test_df["sales_lag_7"],
        "sales": test_df[TARGET_COL],
        "store_id": test_df["store_id"],
        "dept_id": test_df["dept_id"],
    })

    pred_path = os.path.join(dataset_dir, "predictions.parquet")
    prediction_df.to_parquet(pred_path, index=False)
    logger.info("Saved prediction output to %s", pred_path)

    mlflow.log_artifact(pred_path, artifact_path="predictions")
    mlflow.log_metric("prediction_rows", len(prediction_df))

    logger.info("Prediction completed rows=%s output=%s", len(prediction_df), pred_path)
    del test_df, prediction_df, preds, model, X_test, y_test
    gc.collect()
    return pred_path