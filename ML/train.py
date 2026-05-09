import gc
import os
import json
import psutil
import numpy as np
import pandas as pd
import logging
import mlflow
import mlflow.lightgbm
import mlflow.xgboost

from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
from typing import Optional

from ML.preprocess import categorical_cols_cast
from ML.validate import validate_ml_dataset

logger = logging.getLogger(__name__)

TARGET_COL = "sales"


def log_memory_usage(stage: str, additional_info: str = ""):
    """Log current memory usage for tracing and debugging."""
    process = psutil.Process(os.getpid())

    rss_gb = process.memory_info().rss / (1024 ** 3)

    message = f"[{stage}] RSS Memory: {rss_gb:.3f} GB"

    if additional_info:
        message += f" | {additional_info}"

    logger.info(message)

    return rss_gb

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def lgbm_train(train_df, val_df, features):
    log_memory_usage("LGBM_TRAIN_START", f"Features: {len(features)}")

    X_train = categorical_cols_cast(train_df[features])
    y_train = train_df[TARGET_COL]

    X_val = categorical_cols_cast(val_df[features])
    y_val = val_df[TARGET_COL]

    del train_df, val_df
    gc.collect()

    log_memory_usage("LGBM_AFTER_DATA_EXTRACTION", "Train/val DFs deleted")

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature="auto",
        free_raw_data=True
    )

    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature="auto",
        free_raw_data=True
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1
    }

    log_memory_usage("LGBM_BEFORE_TRAIN", "Datasets created")

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    log_memory_usage("LGBM_AFTER_TRAIN", "Model trained")

    pred = model.predict(X_val)
    score = rmse(y_val, pred)
    del train_data, val_data, X_train, y_train
    del X_val, y_val, pred
    gc.collect()
    
    log_memory_usage("LGBM_TRAIN_END", f"RMSE: {score:.4f}")
    
    return model, params, score


def xgboost_train(train_libsvm_path, val_libsvm_path, y_val):
    log_memory_usage("XGBOOST_TRAIN_START", f"Train: {train_libsvm_path}")

    train_dmatrix = xgb.DMatrix(f"{train_libsvm_path}#train.cache")
    val_dmatrix = xgb.DMatrix(f"{val_libsvm_path}#val.cache")
    
    log_memory_usage("XGBOOST_DMATRIX_LOADED", "DMatrices created")

    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "eval_metric": "rmse"
    }

    model = xgb.train(
        params,
        train_dmatrix,
        num_boost_round=1000,
        evals=[(val_dmatrix, "validation")],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    log_memory_usage("XGBOOST_AFTER_TRAIN", "Model trained")
    
    preds = model.predict(val_dmatrix)
    score = rmse(y_val, preds)

    del train_dmatrix
    del val_dmatrix
    del preds
    gc.collect()
    log_memory_usage("XGBOOST_TRAIN_END", f"RMSE: {score:.4f}")
    
    return model, params, score
 

def load_dataset_artifacts(dataset_dir: str):

    metadata_path = os.path.join(dataset_dir, "metadata.json")
    features_path = os.path.join(dataset_dir, "features.json")

    with open(metadata_path) as f:
        metadata = json.load(f)

    with open(features_path) as f:
        features = json.load(f)

    return metadata, features


def train_pipeline(train_df: pd.DataFrame, val_df: pd.DataFrame, train_libsvm_path: str, val_libsvm_path: str, run_id: str, dataset_id: str, mlflow_run_id: str, dataset_dir: str):
    logger.info("Starting pipeline run for run_id=%s dataset_id=%s", run_id, dataset_id)
    log_memory_usage("PIPELINE_START", f"run_id={run_id}, dataset_id={dataset_id}")
    
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run")

    if active_run.info.run_id != mlflow_run_id:
        raise RuntimeError(
            f"Active run {active_run.info.run_id} != expected {mlflow_run_id}"
        )
    
    metadata, features = load_dataset_artifacts(dataset_dir)

    logger.info("Loaded dataset artifacts")
    log_memory_usage("PIPELINE_ARTIFACTS_LOADED", f"Features: {len(features)}")
    
    validate_ml_dataset(train_df, stage="train")
    validate_ml_dataset(val_df, stage="validation")

    mlflow.log_params({
        "run_id": run_id,
        "dataset_id": dataset_id,
        "mlflow_run_id": mlflow_run_id,
        "feature_version": metadata["feature_version"],
        "schema_hash": metadata["schema_hash"],
        "feature_hash": metadata["feature_hash"],
        "num_features": len(features)
    })

    
    baseline_pred = val_df["sales_lag_7"].values
    baseline_score = rmse(val_df[TARGET_COL], baseline_pred)
    mlflow.log_metric("baseline_rmse", baseline_score)
    del baseline_pred
    gc.collect()
    log_memory_usage("PIPELINE_BASELINE_COMPUTED", f"Baseline RMSE: {baseline_score:.4f}")

    # LightGBM with categorical feature support
    lgb_model, lgb_params, lgb_score = lgbm_train(train_df, val_df, features)
    
    mlflow.log_metric("lgb_val_rmse", lgb_score)
    mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})
    log_memory_usage("PIPELINE_LGBM_COMPLETED", f"LGB RMSE: {lgb_score:.4f}")
    del lgb_model
    gc.collect()

    # XGBoost with ordinal encoding for categorical features
    y_val_xgb = val_df[TARGET_COL].copy()
    xgb_model, xgb_params, xgb_score = xgboost_train(
        train_libsvm_path,
        val_libsvm_path,
        y_val_xgb
    )

    mlflow.log_metric("xgb_val_rmse", xgb_score)
    mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})
    log_memory_usage("PIPELINE_XGBOOST_COMPLETED", f"XGB RMSE: {xgb_score:.4f}")

    if lgb_score < xgb_score:
        best_model = lgb_model
        best_name = "lgb"
        best_rmse = lgb_score
        mlflow.lightgbm.log_model(best_model, "model")
    else:
        best_model = xgb_model
        best_name = "xgb"
        best_rmse = xgb_score
        mlflow.xgboost.log_model(best_model, "model")

    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_val_rmse", best_rmse)

    del xgb_model
    del y_val_xgb
    del train_df
    del val_df
    gc.collect()
    log_memory_usage("PIPELINE_MODELS_CLEANED", f"Best model: {best_name}, RMSE: {best_rmse:.4f}")

    mlflow.log_artifact(
        os.path.join(dataset_dir, "metadata.json"),
        artifact_path="dataset_artifacts"
    )

    mlflow.log_artifact(
        os.path.join(dataset_dir, "features.json"),
        artifact_path="dataset_artifacts"
    )

    gc.collect()
    logger.info("Training completed run_id=%s dataset_id=%s mlflow_run_id=%s", run_id, dataset_id, mlflow_run_id)
    log_memory_usage("PIPELINE_END", f"Best model: {best_name}")


def train_main(run_id: str, dataset_id: str, train_path: str, val_path: str, train_libsvm_path: str, val_libsvm_path: str, mlflow_run_id: Optional[str] = None, dataset_dir: Optional[str] = None):
    logger.info("Main entry: run_id=%s dataset_id=%s dataset_dir=%s mlflow_run_id=%s", run_id, dataset_id, dataset_dir, mlflow_run_id)

    if dataset_dir is None:
        dataset_dir = os.path.dirname(train_path)

    initial_memory_rss = log_memory_usage("MAIN_START", f"run_id={run_id}, dataset_id={dataset_id}")
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    log_memory_usage("MAIN_DATA_LOADED", f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")

    try:
        train_pipeline(
            train_df=train_df,
            val_df=val_df,
            train_libsvm_path=train_libsvm_path,
            val_libsvm_path=val_libsvm_path,
            run_id=run_id,
            dataset_id=dataset_id,
            mlflow_run_id=mlflow_run_id,
            dataset_dir=dataset_dir
        )
    finally:
        # Ensure proper cleanup even if training fails
        del train_df
        del val_df
        gc.collect()

    final_memory_rss = log_memory_usage("MAIN_END", f"Training completed")
    memory_delta_rss = final_memory_rss - initial_memory_rss
    logger.info(f"Memory delta - RSS: {memory_delta_rss:+.3f} GB")