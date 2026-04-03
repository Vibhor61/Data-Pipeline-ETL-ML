import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def evaluate_pipeline(df: pd.DataFrame, run_id: str):
    """
    df must contain:
    - prediction
    - actual sales (TARGET_COL)
    """

    mlflow.set_tracking_uri("http://mlflow:5000")

    y_true = df["sales"]
    y_pred = df["prediction"]

    rmse_val = rmse(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    # Log to SAME training run (important)
    with mlflow.start_run(run_id=run_id):

        mlflow.log_metric("real_rmse", rmse_val)
        mlflow.log_metric("real_mae", mae_val)
        mlflow.log_metric("real_mape", mape_val)

    return {
        "rmse": rmse_val,
        "mae": mae_val,
        "mape": mape_val
    }