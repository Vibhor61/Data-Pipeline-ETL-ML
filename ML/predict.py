import os
import json
import tempfile
import pandas as pd
import logging
import joblib
import gc

import mlflow
import mlflow.lightgbm
import mlflow.xgboost

from ML.preprocess import preprocess, transform_xgb, categorical_cols_cast

logger = logging.getLogger(__name__)

TARGET_COL = "sales"


def load_artifacts_from_dataset(dataset_dir: str):
    encoder_path = os.path.join(dataset_dir, "encoders.pkl")
    features_path = os.path.join(dataset_dir, "features.json")
    metadata_path = os.path.join(dataset_dir, "metadata.json")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoders not found in {dataset_dir}")
    
    if  not os.path.exists(features_path):
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
    run = mlflow.get_run(mlflow_run_id)
    best_model = run.data.params.get("best_model")

    if best_model == "lgb":
        model_uri = f"runs:/{mlflow_run_id}/lgb_model"
        return mlflow.lightgbm.load_model(model_uri), best_model
    elif best_model == "xgb":
        model_uri = f"runs:/{mlflow_run_id}/xgb_model"
        return mlflow.xgboost.load_model(model_uri), best_model
    else:
        raise ValueError(f"Unknown best_model: {best_model}")
    

def predict_pipeline(test_path: str, run_id: str, dataset_id: str, train_mlflow_run_id: str, mlflow_run_id: str, dataset_dir: str = None):
    logger.info("Prediction pipeline started for run_id=%s, train_mlflow_run_id=%s and dataset_id=%s", run_id, train_mlflow_run_id, dataset_id)

    if dataset_dir is None:
        dataset_dir = os.path.dirname(test_path)

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run")
    
    if active_run.info.run_id != mlflow_run_id:
        raise RuntimeError(
            f"Active run {active_run.info.run_id} != expected {mlflow_run_id}"
        )
    
    mlflow.set_tag("source_train_run_id", train_mlflow_run_id)
    train_run = mlflow.get_run(train_mlflow_run_id)

    trained_dataset_id = train_run.data.params.get("dataset_id")
    if trained_dataset_id != dataset_id:
        raise ValueError(
            f"Dataset mismatch: model trained on {trained_dataset_id}, "
            f"but prediction requested on {dataset_id}"
        )

    encoders, features, _ = load_artifacts_from_dataset(dataset_dir)
    model, model_type = load_model(train_mlflow_run_id)
    
    test_df = pd.read_parquet(test_path)
    test_df = preprocess(test_df)
    missing_features = set(features) - set(test_df.columns)
    if missing_features:
        raise ValueError(f"Missing features in test data: {missing_features}")
    
    X_test = test_df[features]
    if model_type == "lgb":
        feature_df = categorical_cols_cast(X_test)

    elif model_type == "xgb":
        feature_df = transform_xgb(X_test, encoders)

    preds = model.predict(feature_df)
    
    prediction_df = pd.DataFrame({
        "prediction": preds,
        "actual": test_df[TARGET_COL],
        "store_id": test_df["store_id"],
        "dept_id": test_df["dept_id"],
    })

    pred_path = os.path.join(dataset_dir, "predictions.parquet")
    prediction_df.to_parquet(pred_path, index=False)

    mlflow.log_artifact(pred_path, artifact_path="predictions")
    mlflow.log_metric("prediction_rows", len(prediction_df))

    logger.info("Prediction completed rows=%s", len(prediction_df))

    del test_df, feature_df, prediction_df, preds, model
    gc.collect()
    return pred_path