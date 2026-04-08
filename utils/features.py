import numpy as np
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
    "sales",
    "run_date",
    "d",
    "_pipeline_version",
    "_processed_time",
    "weekday",
    "is_cold_start"
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("None")

    df = df.drop(columns=DROP_COLS, errors="ignore")
    return df


def transform(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    df = df.copy()

    for col, le in encoders.items():
        df[col] = df[col].astype(str)

        mask = ~df[col].isin(le.classes_)
        df.loc[mask, col] = "UNK"

        if "UNK" not in le.classes_:
            le.classes_ = np.append(le.classes_, "UNK")

        df[col] = le.transform(df[col])

    return df