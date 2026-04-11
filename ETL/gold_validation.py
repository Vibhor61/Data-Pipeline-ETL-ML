import pandera as pa
from pandera import Column, Check

VALID_EVENT_TYPES = {"Sporting", "Cultural", "National", "Religious", None}

GoldStorageSchema = pa.DataFrameSchema(
    columns={
        # Identifiers
        "item_id": Column(pa.String, nullable=False),
        "store_id": Column(pa.String, nullable=False),
        "dept_id": Column(pa.String, nullable=False),
        "cat_id": Column(pa.String, nullable=False),
        "state_id": Column(pa.String, nullable=False),

        "d": Column(pa.String, nullable=False),

        # Core values
        "sales": Column(pa.Int, nullable=False, checks=Check.ge(0)),
        "sell_price": Column(pa.Float, nullable=True, checks=Check.ge(0)),

        "run_date": Column(pa.DateTime, nullable=False),
        "_processed_time": Column(pa.DateTime, nullable=False),
        "_pipeline_version": Column(pa.String, nullable=False),

        # Time features
        "wm_yr_wk": Column(pa.Int, nullable=False),
        "weekday": Column(pa.String, nullable=True),
        "wday": Column(pa.Int, nullable=True, checks=Check.in_range(1, 7)),
        "month": Column(pa.Int, nullable=True, checks=Check.in_range(1, 12)),
        "year": Column(pa.Int, nullable=True, checks=Check.ge(2000)),

        # Event features
        "event_name_1": Column(pa.String, nullable=True),
        "event_type_1": Column(pa.String, nullable=True),
        "event_name_2": Column(pa.String, nullable=True),
        "event_type_2": Column(pa.String, nullable=True),

        # SNAP
        "snap_CA": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
        "snap_TX": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
        "snap_WI": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),

        # Feature columns allowed but not strictly validated here
        "sales_lag_1": Column(pa.Float, nullable=True),
        "sales_lag_3": Column(pa.Float, nullable=True),
        "sales_lag_7": Column(pa.Float, nullable=True),
        "sales_lag_14": Column(pa.Float, nullable=True),
        "sales_lag_28": Column(pa.Float, nullable=True),

        "sales_roll_mean_7": Column(pa.Float, nullable=True),
        "sales_roll_mean_14": Column(pa.Float, nullable=True),
        "sales_roll_mean_28": Column(pa.Float, nullable=True),

        "sales_roll_std_7": Column(pa.Float, nullable=True),
        "sales_roll_std_14": Column(pa.Float, nullable=True),
        "sales_roll_std_28": Column(pa.Float, nullable=True),

        "is_cold_start": Column(pa.Bool, nullable=False),
        "is_weekend": Column(pa.Int, nullable=True),
        "quarter": Column(pa.Int, nullable=True),
    },

    checks=[
        # Uniqueness
        Check(
            lambda df: ~df.duplicated(
                subset=["item_id", "store_id", "d", "run_date"]
            ).any(),
            error="Duplicate grain detected"
        ),

        # Single batch guarantees
        Check(lambda df: df["run_date"].nunique() == 1),
        Check(lambda df: df["_pipeline_version"].nunique() == 1),

        # Time correctness
        Check(lambda df: (df["_processed_time"] >= df["run_date"]).all()),

        # Domain constraints
        Check(lambda df: df["state_id"].isin(["CA", "TX", "WI"]).all()),

        Check(lambda df:
            df["event_type_1"].isin(VALID_EVENT_TYPES).all()
            and df["event_type_2"].isin(VALID_EVENT_TYPES).all()
        ),
    ],

    strict=True,
    coerce=True,
)