"""
ML Validation Schema

Defines the post processing gold ML dataset contract and runtime sanity checks 

Core design principles:
- Critical checks must pass (fail pipeline)
- Soft checks logged for observability (do not fail)
"""

import pandera as pa
from pandera import Check
from pandera.errors import SchemaErrors
import logging
import pandas as pd

logger = logging.getLogger(__name__)


LAG_COLS = [
    "sales_lag_1", "sales_lag_3", "sales_lag_7",
    "sales_lag_14", "sales_lag_28"
]

ROLL_MEAN_COLS = [
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28"
]

ROLL_STD_COLS = [
    "sales_roll_std_7", "sales_roll_std_14", "sales_roll_std_28"
]


CriticalFeatureSchema = pa.DataFrameSchema(
        columns={},

        
        checks = [
        
                Check(
                        lambda df: (df["sales"].mean() > 0),
                        error="Mean sales are zero"
                ),
                Check(
                        lambda df: df["sales"].std() > 0,
                        error="Target has zero variance"
                ),
                Check(
                        lambda df: (df[LAG_COLS].dropna() >= 0).values.all(),
                        error="Lag features contain negative values"
                ),
                Check(
                        lambda df: (df[ROLL_MEAN_COLS] >= 0).values().all(),
                        error = "Rolling mean features contain negative values"
                ),
                Check(
                        lambda df: (df[ROLL_STD_COLS] >= 0).values().all(),
                        error = "Rolling std features contain negative values"
                ),
        ],
        strict = False,
        coerce = True
)


SoftFeatureSchema = pa.DataFrameSchema(
        columns={},
        checks = [
                Check(
                        lambda df: all(df.loc[df["is_cold_start"] == True, col].isna().mean() > 0.5 for col in LAG_COLS),
                        error="Cold-start rows should have missing lag features"
                ),
                
                Check(
                        lambda df: all(df[col].dropna().std() > 0 for col in LAG_COLS),
                        error="Some lag features have zero variance"
                ),

                Check(
                        lambda df: all(df[col].dropna().std() > 0 for col in ROLL_MEAN_COLS),
                        error="Rolling mean features have zero variance"
                ),

                Check(
                        lambda df:
                                df.loc[df["sales"] > 0, "sell_price"]
                                .notna().mean() > 0.95,
                        error="sell_price missing for active sales rows"
                ),
                Check(
                        lambda df: df["sales"].quantile(0.99) < 1000,
                        error="Extreme outliers in sales"
                ),       
        ],
        strict = False,
        coerce = True
)


def validate_ml_dataset(df: pd.DataFrame, stage: str) -> None:
    """
    Validates the ML dataset against critical and soft checks.

    Args:
        df (pd.DataFrame): The ML dataset to validate.
        stage (str): The pipeline stage (e.g., "train", "validation") for logging context.
    
    Raises:
        SchemaErrors: If critical checks fail.
    """
    try:
        CriticalFeatureSchema.validate(df, lazy=True)   
    except Exception as e:
        logger.error("Exception occured in critical schema at %s : %s", stage, e.failure_cases)  
        raise SchemaErrors(f"Critical schema validation failed at {stage}", e.failure_cases)    
    try:
        SoftFeatureSchema.validate(df, lazy=True)
    except Exception as e:
        logger.warning("Soft schema warnings at %s : %s", stage, e.failure_cases)
        raise SchemaErrors(f"Soft schema validation warnings at {stage}", e.failure_cases)