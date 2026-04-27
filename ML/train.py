"""
ML Training Pipeline

Trains and logs both LightGBM and XGBoost models on gold feature datasets.

Input: train and validation parquet datasets
Output: logged MLflow models, evaluation metrics, and artifact artifacts

Core design principles:
- model validation uses a held-out validation set
- encoders are persisted alongside feature definitions
- best model selection is based on validation RMSE
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import logging
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import joblib

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
from typing import Optional

from ML.preprocess import preprocess, categorical_cols_cast, fit_xgb_encoder, transform_xgb
from ML.validate import validate_ml_dataset

logger = logging.getLogger(__name__)

TARGET_COL = "sales"
DATE = ["run_date"]


def rmse(y_true, y_pred):
    """
    Compute root mean squared error for model evaluation.
    Args:
        y_true: true target values
        y_pred: predicted target values
    Returns:
        float: RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def baseline_train(df:pd.DataFrame):
    """
    Build a simple baseline prediction using the 7-day lag.
    """
    return df["sales_lag_7"]


def lgbm_train(X_train, y_train, X_val, y_val):
    """
    Train a LightGBM regression model with early stopping.
    Args:
        X_train: training features
        y_train: training labels
        X_val: validation features
        y_val: validation labels
    Returns:
        tuple: trained LightGBM model and parameter dictionary
    """
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature="auto"
    )

    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature="auto"
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

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    return model, params


def xgboost_train(X_train, y_train, X_val, y_val):
    """
    Train an XGBoost regression model with early stopping.
    Args:
        X_train: training features
        y_train: training labels
        X_val: validation features
        y_val: validation labels
    Returns:
        tuple: trained XGBoost model and parameter dictionary
    """
    params = {
        "n_estimators":1000,
        "learning_rate":0.05,
        "max_depth":8,
        "subsample":0.8,
        "colsample_bytree":0.8,
        "tree_method":"hist",
        "eval_metric":"rmse"
    }
    
    model = xgb.XGBRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )

    return model, params
 

def train_pipeline(train_df: pd.DataFrame, val_df: pd.DataFrame,run_id: str, dataset_id:str, mlflow_run_id: str):
    """
    Execute the end-to-end training pipeline and log results to MLflow.
    Args:
        train_df (pd.DataFrame): training dataset
        val_df (pd.DataFrame): validation dataset
        run_id (str): pipeline execution identifier
        dataset_id (str): dataset fingerprint identifier
        mlflow_run_id (str): active MLflow run identifier
    Returns:
        None
    Raises:
        RuntimeError: when no active MLflow run exists
    """

    logger.info("Starting pipeline run for run_id=%s dataset_id=%s", run_id, dataset_id)
    
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run")

    if active_run.info.run_id != mlflow_run_id:
        raise RuntimeError(
            f"Active run {active_run.info.run_id} != expected {mlflow_run_id}"
        )
    
    train_df = preprocess(train_df.copy())
    val_df = preprocess(val_df.copy())
    
    validate_ml_dataset(train_df, stage="train")
    validate_ml_dataset(val_df, stage="validation")
        
    EXCLUDE = {
        TARGET_COL, "run_date",
    }

    FEATURES = [c for c in train_df.columns if c not in EXCLUDE]
    
    X_train, y_train = train_df[FEATURES], train_df[TARGET_COL]
    X_val, y_val = val_df[FEATURES], val_df[TARGET_COL]


    mlflow.log_param("run_id", run_id)
    mlflow.log_param("dataset_id", dataset_id)
    mlflow.log_param("mlflow_run_id", mlflow_run_id)
    mlflow.log_param("num_features", len(FEATURES))

    baseline_pred = baseline_train(val_df)
    baseline_val_rmse = rmse(y_val, baseline_pred)
    mlflow.log_metric("baseline_val_rmse", baseline_val_rmse)
        
    # LightGBM with categorical feature support
    train_lgb = categorical_cols_cast(X_train)
    val_lgb = categorical_cols_cast(X_val)

    lgb_model, lgb_params = lgbm_train(train_lgb, y_train, val_lgb, y_val)
    lgb_val_pred = lgb_model.predict(val_lgb)
    lgb_rmse = rmse(y_val, lgb_val_pred)

    mlflow.log_metric("lgb_val_rmse", lgb_rmse)
    mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})
    
    # XGBoost with ordinal encoding for categorical features
    xgb_encoder = fit_xgb_encoder(train_df)
    train_xgb = transform_xgb(X_train, xgb_encoder)
    val_xgb = transform_xgb(X_val, xgb_encoder)

    xgb_model, xgb_params = xgboost_train(train_xgb, y_train, val_xgb, y_val)
    xgb_val_pred = xgb_model.predict(val_xgb)
    xgb_rmse = rmse(y_val, xgb_val_pred)

    mlflow.log_metric("xgb_val_rmse", xgb_rmse)
    mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})

    if lgb_rmse < xgb_rmse:
        best_model = lgb_model
        best_name = "lgb"
        best_rmse = lgb_rmse
    else:
        best_model = xgb_model
        best_name = "xgb"
        best_rmse = xgb_rmse

    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_val_rmse", best_rmse)

    if best_name == "lgb":
        mlflow.lightgbm.log_model(best_model, "best_model")
    else:
        mlflow.xgboost.log_model(best_model, "best_model")
        
    with tempfile.TemporaryDirectory() as tmpdir:
        enc_path = os.path.join(tmpdir, "encoders.pkl")
        feat_path = os.path.join(tmpdir, "features.json")

        joblib.dump(xgb_encoder, enc_path)

        with open(feat_path, "w") as f:
            json.dump(FEATURES, f)

        mlflow.log_artifact(enc_path)
        mlflow.log_artifact(feat_path)

    logger.info("Training completed run_id=%s dataset_id=%s mlflow_run_id=%s", run_id, dataset_id, mlflow_run_id)


def train_main(run_id: str, dataset_id: str, train_path: str, val_path: str, mlflow_run_id: Optional[str] = None):
    """
    Main entry point for training using parquet dataset files.
    Args:
        run_id (str): pipeline execution identifier
        dataset_id (str): dataset fingerprint identifier
        train_path (str): path to training parquet file
        val_path (str): path to validation parquet file
        mlflow_run_id (Optional[str]): active MLflow run identifier
    Returns:
        None
    """
    
    logger.info("Main entry: run_id=%s dataset_id=%s  mlflow_run_id=%s", run_id, dataset_id, mlflow_run_id)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    train_pipeline(train_df, val_df, run_id, dataset_id, mlflow_run_id)