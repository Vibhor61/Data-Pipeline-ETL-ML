"""
Silver Schema Validation Layer

Defines a strict Pandera schema applied before database commit to:
- prevent schema drift from upstream (bronze) data
- enforce column-level types and constraints
- validate dataset-level invariants (partition + grain)

This acts as a contract boundary between transformation logic and storage.
"""

import pandera as pa
from pandera import Column, Check

"""
Silver Schema:
    Enforces:
        - Typed, fully materialized feature table (no implicit casting downstream)
        - Single run_date per batch (partition integrity)
        - Unique grain: (store_id, item_id, d)
        - Non-negative sales and prices
        - Valid calendar attributes (bounded ranges, categorical consistency)

    Design notes:
        - strict=True prevents unexpected columns (schema drift)
        - coerce=True ensures type normalization before validation
        - nullable fields are explicitly controlled (no implicit null propagation)
"""

SilverSchema = pa.DataFrameSchema(
    columns={
        "run_date": Column(pa.DateTime, nullable=False), # partition key

        "_processed_time": Column(pa.DateTime, nullable=False), # ETL processing timestamp
        "_pipeline_version": Column(pa.String, nullable=False), # version tracking

        # Identity cols
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
            checks=Check.ge(0) # Sales can never be negative
        ),

        "sell_price": Column(
            pa.Float,
            nullable=True,
            checks=Check.ge(0)  # Can be missing but never negative
        ),

        "wm_yr_wk": Column(pa.Int, nullable=False),
        "weekday": Column(pa.String, nullable=False),
        "wday": Column(pa.Int, nullable=False, checks=Check.in_range(1, 7)),
        "month": Column(pa.Int, nullable=False, checks=Check.in_range(1, 12)),
        "year": Column(pa.Int, nullable=False),

        # Event and Snap cols
        "event_name_1": Column(pa.String, nullable=True),
        "event_type_1": Column(pa.String, nullable=True),
        "event_name_2": Column(pa.String, nullable=True),
        "event_type_2": Column(pa.String, nullable=True),

        "snap_ca": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
        "snap_tx": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
        "snap_wi": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
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