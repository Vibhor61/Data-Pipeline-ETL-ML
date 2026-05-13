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


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)


def lgbm_train(train_libsvm_path, val_libsvm_path):
    log_memory_usage("LGBM_TRAIN_START")

    train_data = lgb.Dataset(train_libsvm_path, params={"format": "libsvm", "free_raw_data": True})
    val_data = lgb.Dataset(val_libsvm_path, params={"format": "libsvm", "free_raw_data": True})

    log_memory_usage("LGBM_AFTER_DATA_EXTRACTION", "Train/val DFs deleted")

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
        valid_names=["validation"],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )

    log_memory_usage("LGBM_AFTER_TRAIN", "Model trained")

    score = model.best_score["validation"]["rmse"]

    del train_data
    del val_data

    gc.collect()

    log_memory_usage("LGBM_TRAIN_END", f"RMSE: {score:.4f}")
    
    return model, params, score


def xgboost_train(train_libsvm_path, val_libsvm_path):
    log_memory_usage("XGBOOST_TRAIN_START", f"Train: {train_libsvm_path}")

    train_dmatrix = xgb.DMatrix(f"{train_libsvm_path}?format=libsvm#train.cache")
    val_dmatrix = xgb.DMatrix(f"{val_libsvm_path}?format=libsvm#val.cache")
    
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
    
    score = model.best_score

    del train_dmatrix
    del val_dmatrix
    
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


def train_pipeline(val_df: pd.DataFrame, train_libsvm_path: str, val_libsvm_path: str, run_id: str, dataset_id: str, mlflow_run_id: str, dataset_dir: str):
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

    mlflow.set_tags({
        "evaluation_metric": "wmape",
        "baseline_model": "sales_lag_7"
    })
    
    baseline_pred = val_df["sales_lag_7"].values
    baseline_rmse = rmse(val_df[TARGET_COL].values, baseline_pred)
    baseline_wmape = wmape(val_df[TARGET_COL].values, baseline_pred)
    mlflow.log_metrics({
        "baseline_rmse": baseline_rmse,
        "baseline_wmape": baseline_wmape
    })
    del baseline_pred
    gc.collect()
    log_memory_usage("PIPELINE_BASELINE_COMPUTED", f"Baseline RMSE: {baseline_rmse:.4f} and WMAPE: {baseline_wmape:.4f}")

    # LightGBM with categorical feature support
    lgb_model, lgb_params, lgb_rmse = lgbm_train(train_libsvm_path, val_libsvm_path)
    lgb_wmape = wmape(val_df[TARGET_COL].values, lgb_model.predict(val_df[features]))

    mlflow.log_metrics({
            "lgb_val_rmse": lgb_rmse,
            "lgb_val_wmape": lgb_wmape}
    )
    mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})
    log_memory_usage("PIPELINE_LGBM_COMPLETED", f"LGB RMSE: {lgb_rmse:.4f} and WMAPE: {lgb_wmape:.4f}")
   
    # XGBoost with ordinal encoding for categorical features
    xgb_model, xgb_params, xgb_rmse = xgboost_train(
        train_libsvm_path,
        val_libsvm_path
    )
    xgb_wmape = wmape(val_df[TARGET_COL].values, xgb_model.predict(xgb.DMatrix(val_df[features])))
    mlflow.log_metrics({
        "xgb_val_rmse": xgb_rmse,
        "xgb_val_wmape": xgb_wmape
    })
    mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})
    log_memory_usage("PIPELINE_XGBOOST_COMPLETED", f"XGB RMSE: {xgb_rmse:.4f} and WMAPE: {xgb_wmape:.4f}")

    if lgb_wmape < xgb_wmape:
        best_model = lgb_model
        best_name = "lgb"
        best_wmape = lgb_wmape
        mlflow.lightgbm.log_model(best_model, "model")
    else:
        best_model = xgb_model
        best_name = "xgb"
        best_wmape = xgb_wmape
        mlflow.xgboost.log_model(best_model, "model")

    mlflow.log_metric("training_feature_count", len(features))
    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_val_wmape", best_wmape)

    improvement_over_baseline = (baseline_wmape - best_wmape) / baseline_wmape * 100
    mlflow.log_metric("wmape_improvement_over_baseline_pct", improvement_over_baseline)

    del xgb_model
    del lgb_model
    del val_df
    gc.collect()
    log_memory_usage("PIPELINE_MODELS_CLEANED", f"Best model: {best_name}, WMAPE: {best_wmape:.4f}")

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
    
    val_df = pd.read_parquet(val_path)
    
    log_memory_usage("MAIN_DATA_LOADED, Val shape: {val_df.shape}")

    try:
        train_pipeline(
            val_df=val_df,
            train_libsvm_path=train_libsvm_path,
            val_libsvm_path=val_libsvm_path,
            run_id=run_id,
            dataset_id=dataset_id,
            mlflow_run_id=mlflow_run_id,
            dataset_dir=dataset_dir
        )
    finally:
        del val_df
        gc.collect()

    final_memory_rss = log_memory_usage("MAIN_END Training completed")
    memory_delta_rss = final_memory_rss - initial_memory_rss
    logger.info(f"Memory delta - RSS: {memory_delta_rss:+.3f} GB")