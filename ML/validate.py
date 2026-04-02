import pandera as pa
from pandera import Column, Check
import pandas as pd

GoldTableSchema = pa.DataFrameSchema(
        columns={
                "item_id": Column(pa.String, nullable=False),
                "store_id": Column(pa.String, nullable=False),
                "dept_id": Column(pa.String, nullable=False),
                "cat_id": Column(pa.String, nullable=False),
                "state_id": Column(pa.String, nullable=False),
                
                "d": Column(pa.String, nullable=False),

                # Target Variable
                "sales": Column(pa.Int, nullable=False, checks=pa.Check.ge(0)),
                "sell_price": Column(pa.Float, nullable=True, checks=pa.Check.ge(0)),

                "run_date": Column(pa.DateTime, nullable=False),

                "_processed_time": Column(pa.DateTime, nullable=False),
                "_pipeline_version": Column(pa.String, nullable=False),

                # Lag Features
                "sales_lag_1": Column(pa.Float, nullable=True),
                "sales_lag_3": Column(pa.Float, nullable=True),
                "sales_lag_7": Column(pa.Float, nullable=True),
                "sales_lag_14": Column(pa.Float, nullable=True),
                "sales_lag_28": Column(pa.Float, nullable=True),
                
                # Rolling Features 
                "sales_roll_mean_7": Column(pa.Float, nullable=True),
                "sales_roll_mean_14": Column(pa.Float, nullable=True),
                "sales_roll_mean_28": Column(pa.Float, nullable=True),

                "sales_roll_std_7": Column(pa.Float, Check.ge(0), nullable=True),
                "sales_roll_std_14": Column(pa.Float, Check.ge(0), nullable=True),
                "sales_roll_std_28": Column(pa.Float, Check.ge(0), nullable=True),
                
                "is_cold_start": Column(pa.Bool, nullable=False),

                "is_weekend": Column(pa.Int, Check.isin([0, 1]), nullable=True),
                "quarter": Column(pa.Int, Check.isin([1,2,3,4]), nullable=True),
                "month": Column(pa.Int, Check.in_range(1, 13), nullable=True),
                "wday": Column(pa.Int, Check.in_range(1, 7), nullable=True),
                "weekday": Column(pa.String, nullable=True),
                "year": Column(pa.Int, Check.ge(2000), nullable=True),

                # Event Features
                "event_name_1": Column(pa.String, nullable=True),
                "event_type_1": Column(pa.String, nullable=True),
                "event_name_2": Column(pa.String, nullable=True),
                "event_type_2": Column(pa.String, nullable=True),

                #SNAP Features
                "snap_CA": Column(pa.Int, Check.isin([0,1]), nullable=False),
                "snap_TX": Column(pa.Int, Check.isin([0,1]), nullable=False),
                "snap_WI": Column(pa.Int, Check.isin([0,1]), nullable=False),

                "wm_yr_wk": Column(pa.Int, nullable=False),
        },

        
        checks = [
                pa.Check(lambda df: (df[["sales_lag_1","sales_lag_3","sales_lag_7","sales_lag_14","sales_lag_28"]] >= 0).all().all(),
                        error="Lag features contain negative values"),

                # Uniqueness: each (item_id, store_id, d) combo should appear once
                pa.Check(lambda df: ~df.duplicated(subset=["item_id", "store_id", "d", "run_date"]).any(),
                        error="Duplicate (item_id, store_id, d) found"),

                # processed_time should be >= run_date
                pa.Check(lambda df: (df["_processed_time"] >= df["run_date"]).all(),
                        error="_processed_time is earlier than run_date"),

                # sell_price > 0 when sales > 0
                pa.Check(lambda df: (df[df["sales"] > 0]["sell_price"] > 0).all(),
                        error="Items with positive sales have zero sell_price"),

                pa.Check(lambda df: (df[["sales_roll_mean_7","sales_roll_mean_14","sales_roll_mean_28","sales_roll_std_7","sales_roll_std_14","sales_roll_std_28"]] >= 0).all().all(),
                        error = "Rolling features contain negative values"),

                pa.Check(lambda df: df["_pipeline_version"].nunique() == 1,
                        error="Multiple pipeline versions in one batch"),

                pa.Check(lambda df: df["run_date"].nunique() == 1,
                        error="Multiple run_dates in one batch partial merge detected"),
                
                VALID_EVENT_TYPES = {"Sporting","Cultural","National","Religious", None}
                # M-5 Dataset Level Gaurantee
                pa.Check(lambda df:
                        df["event_type_1"].isin(VALID_EVENT_TYPES).all(),
                        df["event_type_2"].isin(VALID_EVENT_TYPES).all(),
                        error="Unknown event_type value M5 schema drift"),

                pa.Check(
                        lambda df: df["state_id"].isin(["CA","TX","WI"]).all(),
                        error="state_id outside M5 scope (CA/TX/WI)"
                )
        ],
        strict = True,
        coerce = True
)

def validate_gold_data(df: pd.DataFrame):
    try:
        GoldTableSchema.validate(df, lazy=True)
        print("Gold Table Validation Passed!")
    except pa.errors.SchemaErrors as err:
        print("Validation Failed!")
        print(err.failure_cases) 
        raise