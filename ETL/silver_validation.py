import pandera as pa
from pandera import Column, Check

SilverSchema = pa.DataFrameSchema(
    columns={
        "run_date": Column(pa.DateTime, nullable=False),

        "_processed_time": Column(pa.DateTime, nullable=False),
        "_pipeline_version": Column(pa.String, nullable=False),

        "store_id": Column(pa.String, nullable=False),
        "item_id": Column(pa.String, nullable=False),
        "d": Column(pa.String, nullable=False),

        "date": Column(pa.DateTime, nullable=False),

        "dept_id": Column(pa.String, nullable=False),
        "cat_id": Column(pa.String, nullable=False),
        "state_id": Column(pa.String, nullable=False),

        "sales": Column(
            pa.Int,
            nullable=False,
            checks=Check.ge(0)
        ),

        "sell_price": Column(
            pa.Float,
            nullable=True,
            checks=Check.ge(0)
        ),

        "wm_yr_wk": Column(pa.Int, nullable=False),

        "weekday": Column(pa.String, nullable=False),
        "wday": Column(pa.Int, nullable=False, checks=Check.in_range(1, 7)),
        "month": Column(pa.Int, nullable=False, checks=Check.in_range(1, 12)),
        "year": Column(pa.Int, nullable=False),

        "event_name_1": Column(pa.String, nullable=True),
        "event_type_1": Column(pa.String, nullable=True),
        "event_name_2": Column(pa.String, nullable=True),
        "event_type_2": Column(pa.String, nullable=True),

        "snap_CA": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
        "snap_TX": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
        "snap_WI": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
    },

    checks=[
        # One run_date per partition
        Check(lambda df: df["run_date"].nunique() == 1),

        # No duplicate grain
        Check(
            lambda df: df.duplicated(
                subset=["store_id", "item_id", "d"]
            ).sum() == 0,
            error="Duplicate grain detected"
        ),
    ],

    strict=True,
    coerce=True,
)