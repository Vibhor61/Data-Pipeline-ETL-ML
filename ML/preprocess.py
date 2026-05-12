import pandas as pd   
from sklearn.preprocessing import OrdinalEncoder

ALL_COLS = [
    "item_id",
    "store_id",
    "dept_id",
    "cat_id",
    "state_id",
    "d",
    "sales",
    "sell_price",
    "run_date",
    "_processed_time",
    "_pipeline_version",
    "sales_lag_1",
    "sales_lag_3",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_roll_mean_7",
    "sales_roll_mean_14",
    "sales_roll_mean_28",
    "sales_roll_std_7",
    "sales_roll_std_14",
    "sales_roll_std_28",
    "is_cold_start",
    "is_weekend",
    "quarter",
    "month",
    "wday",
    "weekday",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_ca",
    "snap_tx",
    "snap_wi",
    "wm_yr_wk"
]

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
    "d",
    "_pipeline_version",
    "_processed_time",
    "weekday"
]

DECIMAL_COLS = [
    "sales",
    "sell_price",
    "sales_roll_mean_7",
    "sales_roll_mean_14",
    "sales_roll_mean_28",
    "sales_roll_std_7",
    "sales_roll_std_14",
    "sales_roll_std_28"
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = df[col].astype("int32")

    for col in df.select_dtypes(include=["floating"]).columns:
        df[col] = df[col].astype("float32")
    
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype("int8")

    if "run_date" in df.columns:
        df["run_date"] = pd.to_datetime(df["run_date"], errors="coerce")

    df[DECIMAL_COLS] = df[DECIMAL_COLS].astype("float32", errors="ignore")    
    df = df.drop(columns=DROP_COLS, errors="ignore")
    return df


def fit_encoder(df: pd.DataFrame) -> OrdinalEncoder:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    encoder.fit(df[CATEGORICAL_COLS])
    return encoder


def transform(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
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
