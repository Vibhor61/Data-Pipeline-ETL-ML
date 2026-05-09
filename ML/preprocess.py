import pandas as pd   
from sklearn.preprocessing import OrdinalEncoder

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
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df = df.drop(columns=DROP_COLS, errors="ignore")
    return df


def fit_encoder(df: pd.DataFrame) -> OrdinalEncoder:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    encoder.fit(df[CATEGORICAL_COLS])
    return encoder


def transform_xgb(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy(deep=False)
    for col in CATEGORICAL_COLS:

        if col not in df.columns:
            continue

        mapping = encoders[col]
        unk = mapping["__UNK__"]

        df[col] = (
            df[col].astype(str).map(lambda x: mapping.get(x, unk)).astype("int32")
        )

    return df


def categorical_cols_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to category dtype for LightGBM."""
    df = df.copy(deep=False)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df