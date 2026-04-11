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

from utils.ml_helpers import log_training_run
from utils.db import get_connection
from preprocess import CATEGORICAL_COLS, preprocess, transform
from validate import GoldMLSchema

logger = logging.getLogger(__name__)

TARGET_COL = "sales"
DATE = ["run_date"]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def split_data(df:pd.DataFrame, meta: dict):

    df = df.sort_values("run_date")
    end_date = pd.to_datetime(meta["time_end"])

    val_start  = end_date - pd.Timedelta(days=200)
    test_start = end_date - pd.Timedelta(days=100)

    train = df[df["run_date"] < val_start]
    val   = df[(df["run_date"] >= val_start) & (df["run_date"] < test_start)]
    test  = df[df["run_date"] >= test_start]

    return train, val, test


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


def xg_boost_train(X_train, y_train, X_val, y_val):
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
 

def train_pipeline(df: pd.DataFrame, run_id: str, dataset_id:str, meta: dict):

    logger.info("Starting pipeline run")
    mlflow.set_tracking_uri("http://mlflow:5000")  # or localhost if outside docker
    mlflow.set_experiment("retail_demand_forecasting")

    with mlflow.start_run(run_name=f"run_{run_id}") as run:
        mlflow_run_id = run.info.run_id

        df = df.copy()
        df = preprocess(df)

        train, val, test = split_data(df, meta)
        for d in [train, val, test]:
            if "run_date" in d.columns:
                d.drop(columns=["run_date"], inplace=True)

        encoders = fit_encoders(train)
        
        train = transform(train, encoders)
        val   = transform(val, encoders)
        test  = transform(test, encoders)

        # FEATURES = [col for col in df.columns if col != TARGET_COL] Feature leakage if run_date and identifiers not dropped
        FEATURES = [col for col in train.columns if col not in TARGET_COL]

        mlflow.log_param("run_id", run_id)
        mlflow.log_param("dataset_id", dataset_id)
        mlflow.log_param("feature_version", meta["feature_version"])
        mlflow.log_param("time_start", meta["time_start"])
        mlflow.log_param("time_end", meta["time_end"])

        mlflow.log_param("num_features", len(FEATURES))
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("val_rows", len(val))
        mlflow.log_param("test_rows", len(test))
        
        X_train, y_train = train[FEATURES], train[TARGET_COL]
        X_val, y_val     = val[FEATURES], val[TARGET_COL]
        X_test, y_test   = test[FEATURES], test[TARGET_COL]

        baseline_pred = baseline_train(val)
        baseline_val_rmse = rmse(y_val, baseline_pred)
        mlflow.log_metric("baseline_val_rmse", baseline_val_rmse)
        
        lgb_model, lgb_params = lgbm_train(X_train, y_train, X_val, y_val)
        val_lgb_pred = lgb_model.predict(X_val)
        lgb_val_rmse = rmse(y_val, val_lgb_pred)

        mlflow.log_metric("lgb_val_rmse", lgb_val_rmse)
        mlflow.lightgbm.log_model(lgb_model, "lgb_model")
        mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})

        xgb_model, xgb_params = xg_boost_train(X_train, y_train, X_val, y_val)
        val_xgb_pred = xgb_model.predict(X_val)
        xgb_val_rmse = rmse(y_val, val_xgb_pred)

        mlflow.log_metric("xgb_val_rmse", xgb_val_rmse)
        mlflow.xgboost.log_model(xgb_model, "xgb_model")
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})

        metrics = {
            "lgb": lgb_val_rmse,
            "xgb": xgb_val_rmse
        }

        best_model_name = min(metrics, key=metrics.get)
        mlflow.log_param("best_model", best_model_name)

        if best_model_name == "lgb":
            best_model = lgb_model
            test_pred = lgb_model.predict(X_test)
            mlflow.lightgbm.log_model(best_model, "best_model")

        else:
            best_model = xgb_model
            test_pred = xgb_model.predict(X_test)
            mlflow.xgboost.log_model(best_model, "best_model")

        test_rmse = rmse(y_test, test_pred)
        mlflow.log_metric("test_rmse", test_rmse)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            enc_path = os.path.join(tmpdir, "encoders.pkl")
            feat_path = os.path.join(tmpdir, "features.json")

            joblib.dump(encoders, enc_path)

            with open(feat_path, "w") as f:
                json.dump(FEATURES, f)

            mlflow.log_artifact(enc_path)
            mlflow.log_artifact(feat_path)

        logger.info(
            "Training pipeline completed successfully for run_id=%s best_model=%s rmse=%.4f",
            run_id,
            best_model_name,
            test_rmse
        )

        conn = None 
        try: 
            conn = get_connection()
            log_training_run(
                conn=conn,
                run_id=run_id,
                dataset_id=dataset_id,
                model_name=best_model_name,
                model_version=f"{best_model_name}_{run_id}",
                mlflow_run_id=mlflow_run_id,
            )  
        finally:
            if conn:
                conn.close()


def main(run_id: str, dataset_id: str, df: pd.DataFrame, meta: dict):
    logger.info("Main entry: run_id=%s dataset_id=%s", run_id, dataset_id)
    
    GoldMLSchema.validate(df, lazy=True)
    train_pipeline(df, run_id, dataset_id, meta)