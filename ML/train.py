import os
import json
import tempfile
import numpy as np
import pandas as pd
import logging

import mlflow
import mlflow.lightgbm
import mlflow.xgboost

from typing import Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb

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


def split_data(df:pd.DataFrame, end_date: str):
    df = df.sort_values("run_date")

    end_date = pd.to_datetime(end_date) 
    start_date = end_date - pd.Timedelta(days=1000)
    validation_start_date = end_date - pd.Timedelta(days=200)
    test_start_date = end_date - pd.Timedelta(days=100)
    
    train_data = df[(df["date"] >= start_date) & (df["date"] < validation_start_date)]
    validation_data = df[(df["date"] >= validation_start_date) & (df["date"] < test_start_date)]
    test_data = df[(df["date"] >= test_start_date) & (df["date"] <= end_date)]

    return train_data, validation_data, test_data


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


def baseline_train(df:pd.DataFrame , end_date:str):
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


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


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


def train_pipeline(df:pd.DataFrame, end_date: str):

    logger.info("Starting pipeline run")
    mlflow.set_tracking_uri("http://mlflow:5000")  # or localhost if outside docker
    mlflow.set_experiment("retail_demand_forecasting")
    
    with mlflow.start_run(run_name=f"run_{end_date}"):
        mlflow.log_param("end_date", end_date)
        mlflow.log_param("model_types", "baseline,lgbm,xgboost")
        mlflow.log_param("num_features", len(df.columns))
        
        df = df.copy()
        df = preprocess(df)

        train, val, test = split_data(df, end_date)

        encoders = fit_encoders(train)
        
        train = transform(train, encoders)
        val   = transform(val, encoders)
        test  = transform(test, encoders)

        FEATURES = [col for col in df.columns if col != TARGET_COL]
        
        X_train, y_train = train[FEATURES], train[TARGET_COL]
        X_val, y_val     = val[FEATURES], val[TARGET_COL]
        X_test, y_test   = test[FEATURES], test[TARGET_COL]

        baseline_pred = baseline_train(val)
        baseline_rmse = rmse(y_val, baseline_pred)
        mlflow.log_metric("baseline_rmse", baseline_rmse)
        
        lgb_model, lgb_params = lgbm_train(X_train, y_train, X_val, y_val)
        val_lgb_pred = lgb_model.predict(X_val)
        lgb_rmse = rmse(y_val, val_lgb_pred)

        mlflow.log_metric("lgb_rmse", lgb_rmse)
        mlflow.lightgbm.log_model(lgb_model, "lgb_model")
        mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})

        xgb_model, xgb_params = xg_boost_train(X_train, y_train, X_val, y_val)
        val_xgb_pred = xgb_model.predict(X_val)
        xgb_rmse = rmse(y_val, val_xgb_pred)

        mlflow.log_metric("xgb_rmse", xgb_rmse)
        mlflow.xgboost.log_model(xgb_model, "xgb_model")
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})

        metrics = {
            "baseline": baseline_rmse,
            "lgb": lgb_rmse,
            "xgb": xgb_rmse
        }

        best_model_name = min(metrics, key=metrics.get)
        mlflow.log_param("best_model", best_model_name)

        if best_model_name == "lgb":
            best_model = lgb_model
            test_pred = lgb_model.predict(X_test)
        elif best_model_name == "xgb":
            best_model = xgb_model
            test_pred = xgb_model.predict(X_test)
        else:
            test_pred = baseline_train(test)

        test_rmse = rmse(y_test, test_pred)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_model("best_model", best_model)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            enc_path = os.path.join(tmpdir, "encoders.pkl")
            feat_path = os.path.join(tmpdir, "features.json")

            import joblib
            joblib.dump(encoders, enc_path)

            with open(feat_path, "w") as f:
                json.dump(FEATURES, f)

            mlflow.log_artifact(enc_path)
            mlflow.log_artifact(feat_path)

        logger.info("Training pipeline completed successfully")