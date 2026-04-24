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

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb

from preprocess import CATEGORICAL_COLS, preprocess, transform
from validate import GoldMLSchema

logger = logging.getLogger(__name__)

TARGET_COL = "sales"
DATE = ["run_date"]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

 
def fit_encoders(train):
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        train[col] = train[col].astype(str)
        le.fit(train[col].unique().tolist() + ["UNK"])
        encoders[col] = le

    return encoders


def baseline_train(df:pd.DataFrame):
    return df["sales_lag_7"]


def lgbm_train(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=CATEGORICAL_COLS
    )

    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature=CATEGORICAL_COLS
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

    logger.info("Starting pipeline run for run_id=%s dataset_id=%s", run_id, dataset_id)
    
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run found in training stage")
    
    GoldMLSchema.validate(train_df, lazy=True)
    GoldMLSchema.validate(val_df, lazy=True)

    train_df = preprocess(train_df.copy())
    val_df = preprocess(val_df.copy())

    if "run_date" in train_df.columns:
        train_df = train_df.drop(columns=["run_date"])
    if "run_date" in val_df.columns:
        val_df = val_df.drop(columns=["run_date"])

    encoders = fit_encoders(train_df)
        
    train_df = transform(train_df, encoders)
    val_df = transform(val_df, encoders)
   
    # FEATURES = [col for col in df.columns if col != TARGET_COL] Feature leakage if run_date and identifiers not dropped
    FEATURES = [col for col in train_df.columns if col != TARGET_COL]    
    X_train, y_train = train_df[FEATURES], train_df[TARGET_COL]
    X_val, y_val = val_df[FEATURES], val_df[TARGET_COL]


    mlflow.log_param("run_id", run_id)
    mlflow.log_param("dataset_id", dataset_id)
    mlflow.log_param("mlflow_run_id", mlflow_run_id)
    mlflow.log_param("num_features", len(FEATURES))

    baseline_pred = baseline_train(val_df)
    baseline_val_rmse = rmse(y_val, baseline_pred)
    mlflow.log_metric("baseline_val_rmse", baseline_val_rmse)
        
    lgb_model, lgb_params = lgbm_train(X_train, y_train, X_val, y_val)
    lgb_val_pred = lgb_model.predict(X_val)
    lgb_rmse = rmse(y_val, lgb_val_pred)

    mlflow.log_metric("lgb_val_rmse", lgb_rmse)
    mlflow.lightgbm.log_model(lgb_model, "lgb_model")
    mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})

    xgb_model, xgb_params = xgboost_train(X_train, y_train, X_val, y_val)
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_rmse = rmse(y_val, xgb_val_pred)

    mlflow.log_metric("xgb_val_rmse", xgb_rmse)
    mlflow.xgboost.log_model(xgb_model, "xgb_model")
    mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})

    best_model_name = "lgb" if lgb_rmse < xgb_rmse else "xgb"
    mlflow.log_param("best_model", best_model_name)

    best_model = lgb_model if best_model_name == "lgb" else xgb_model
    mlflow.log_metric("best_val_rmse", min(lgb_rmse, xgb_rmse))

    if best_model_name == "lgb":
        mlflow.lightgbm.log_model(best_model, "best_model") 
    else:
        mlflow.xgboost.log_model(best_model, "best_model")
        
    with tempfile.TemporaryDirectory() as tmpdir:
        enc_path = os.path.join(tmpdir, "encoders.pkl")
        feat_path = os.path.join(tmpdir, "features.json")

        joblib.dump(encoders, enc_path)

        with open(feat_path, "w") as f:
            json.dump(FEATURES, f)

        mlflow.log_artifact(enc_path)
        mlflow.log_artifact(feat_path)

    logger.info("Training completed run_id=%s dataset_id=%s mlflow_run_id=%s", run_id, dataset_id, mlflow_run_id)


def train_main(run_id: str, dataset_id: str, train_path: str, val_path: str, mlflow_run_id: str):

    logger.info("Main entry: run_id=%s dataset_id=%s", run_id, dataset_id)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    train_pipeline(train_df, val_df, run_id, dataset_id, mlflow_run_id)