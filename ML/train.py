import os
import json
import joblib
import numpy as np
import pandas as pd
import logging

from typing import Tuple, Dict
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

MODEL_DIR = "artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)


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
    return df["sales_lag_1"]


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

    return model


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def xg_boost_train(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="rmse"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )

    return model


def train_pipeline(df:pd.DataFrame, end_date: str):
    logger.info("Starting pipeline run")

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

    val_baseline_pred = baseline_train(val)
    baseline_rmse = rmse(y_val, val_baseline_pred)

    lgb_model = lgbm_train(X_train, y_train, X_val, y_val)
    val_lgb_pred = lgb_model.predict(X_val)
    lgb_rmse = rmse(y_val, val_lgb_pred)

    xgb_model = xg_boost_train(X_train, y_train, X_val, y_val)
    val_xgb_pred = xgb_model.predict(X_val)
    xgb_rmse = rmse(y_val, val_xgb_pred)

    joblib.dump(lgb_model, os.path.join(MODEL_DIR, "lgb_model.pkl"))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

    with open(os.path.join(MODEL_DIR, "features.json"), "w") as f:
        json.dump(FEATURES, f)

    metrics = {
        "baseline_rmse": float(baseline_rmse),
        "lgb_rmse": float(lgb_rmse),
        "xgb_rmse": float(xgb_rmse)
    }

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)