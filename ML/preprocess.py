"""
ML Preprocessing Utilities

Prepares ML feature data for training and inference by cleaning, encoding,
and imputing missing values.

Input: raw gold feature dataset
Output: transformed feature-ready dataset

Core design principles:
- categorical features are label encoded consistently
- missing lags and rolling features are forward-filled and imputed
- dataset identifiers are preserved through feature generation
"""

import pandas as pd
from typing import Dict
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

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
    "run_date",
    "d",
    "_pipeline_version",
    "_processed_time",
    "weekday"
]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset in preparation for model training or inference.
    Args:
        df (pd.DataFrame): input gold feature dataset
    Returns:
        pd.DataFrame: cleaned and imputed dataset
    """
    df = df.copy()

    df = df.sort_values(["item_id", "store_id", "run_date"])

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    lag_cols = [col for col in df.columns if "lag" in col]
    roll_cols = [col for col in df.columns if "roll" in col]
    group_cols = ["item_id", "store_id"]

    df[lag_cols + roll_cols] = (
        df.groupby(group_cols)[lag_cols + roll_cols]
          .ffill()
    )

    for col in lag_cols + roll_cols:
        df[col] = df[col].fillna(df[col].mean())

    if "sell_price" in df.columns:
        df["sell_price"] = (
            df.groupby(group_cols)["sell_price"]
              .ffill()
        )

        df["sell_price"] = df["sell_price"].fillna(df["sell_price"].mean())

    df = df.drop(columns=DROP_COLS, errors="ignore")
    return df


def categorical_cols_cast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure categorical columns are of type 'category' for LightGBM.
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def fit_xgb_encoder(df: pd.DataFrame) -> OrdinalEncoder:
    """
    Fit ordinal encoder for XGBoost.
    """
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    encoder.fit(df[CATEGORICAL_COLS])
    return encoder


def transform_xgb(df: pd.DataFrame, enc: OrdinalEncoder) -> pd.DataFrame:
    """
    Apply ordinal encoding for XGBoost.
    """
    df = df.copy()
    df[CATEGORICAL_COLS] = enc.transform(df[CATEGORICAL_COLS])
    return df