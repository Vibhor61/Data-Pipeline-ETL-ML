import os
import json
import pandas as pd
import logging
import joblib
import gc

import mlflow
import mlflow.lightgbm
import mlflow.xgboost

from sklearn.datasets import load_svmlight_file

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

    if best_model is None:
        raise ValueError(f"Missing best_model parameter in MLflow run {mlflow_run_id}")

    model_uri = f"runs:/{mlflow_run_id}/model"
    if best_model == "lgb":
        return mlflow.lightgbm.load_model(model_uri), best_model
    elif best_model == "xgb":
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

    _, features, _ = load_artifacts_from_dataset(dataset_dir)
    logger.info("Loaded dataset artifacts from %s", dataset_dir)

    model, model_type = load_model(train_mlflow_run_id)
    logger.info("Loaded model type %s from MLflow run %s", model_type, train_mlflow_run_id)

    test_libsvm_path = os.path.join(dataset_dir, "test.libsvm")

    X_test, y_test = load_svmlight_file(
        test_libsvm_path,
        n_features=len(features),
        zero_based=True
    )

    logger.info("Loaded test libsvm from %s with %s rows", test_libsvm_path, X_test.shape[0])
    logger.info("X_test shape: %s, model expects: %s", X_test.shape, model.num_feature())
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