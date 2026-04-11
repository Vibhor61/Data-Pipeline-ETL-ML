import pandas as pd
from typing import Dict
from sklearn.preprocessing import LabelEncoder

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
    df = df.copy()

    df = df.sort_values(["item_id", "store_id", "run_date"])

    for col in CATEGORICAL_COLS:
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
              .bfill()
        )

        df["sell_price"] = df["sell_price"].fillna(df["sell_price"].mean())

    df = df.drop(columns=DROP_COLS, errors="ignore")
    return df


def transform(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    df = df.copy()

    for col, le in encoders.items():
        df[col] = df[col].astype(str)

        mask = ~df[col].isin(le.classes_)
        df.loc[mask, col] = "UNK"

        df[col] = le.transform(df[col])

    return df