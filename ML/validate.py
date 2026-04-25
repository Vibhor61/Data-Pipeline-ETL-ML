"""
ML Validation Schema

Defines the gold ML dataset contract and runtime sanity checks.

Core design principles:
- validate target distribution and variance
- enforce non-negative lag and rolling features
- require consistent sell_price coverage for active sales
"""

import pandera as pa
from pandera import Check


GoldMLSchema = pa.DataFrameSchema(
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
                        lambda df: (df[["sales_lag_1","sales_lag_3","sales_lag_7","sales_lag_14","sales_lag_28"]] >= 0).values.all(),
                        error="Lag features contain negative values"
                ),
                Check(
                        lambda df: (df[["sales_roll_mean_7","sales_roll_mean_14","sales_roll_mean_28"]] >= 0).all().all(),
                        error = "Rolling mean features contain negative values"
                ),
                Check(
                        lambda df: (df[["sales_roll_std_7","sales_roll_std_14","sales_roll_std_28"]] >= 0).all().all(),
                        error = "Rolling std features contain negative values"
                ),

                Check(
                        lambda df:
                                df.loc[df["is_cold_start"] == True, "sales_lag_7"]
                                .isna().mean() > 0.5,
                        error="Cold-start rows should have missing lag features"
                ),
                
                Check(
                        lambda df: df["sales_lag_7"].std() > 0,
                        error="Lag feature has no variance"
                ),

                Check(
                        lambda df: df["sales_roll_mean_7"].std() > 0,
                        error="Rolling feature has no variance"
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

                Check(
                        lambda df: df["sales_lag_7"].std() > 0,
                        error="Lag feature has no variance"
                ),

                Check(
                        lambda df: df["sales_roll_mean_7"].std() > 0,
                        error="Rolling feature has no variance"
                ),
                
        ],
        strict = False,
        coerce = True
)
