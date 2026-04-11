import numpy as np
import pandas as pd
import mlflow

from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from utils.ml_helpers import log_evaluation_metrics
from utils.db import get_connection

logger = logging.getLogger(__name__)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)

# Slice key harcoded in version1
def evaluate_pipeline(df: pd.DataFrame, mlflow_run_id: str ,dataset_id: str):

    logger.info("Prediction pipeline started for mlflow_run_id=%s and dataset_id=%s", mlflow_run_id, dataset_id)
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Silent failure if trained on some dataset but evaluating on other without checking
    run = mlflow.get_run(mlflow_run_id)
    trained_dataset_id = run.data.params.get("dataset_id")

    if trained_dataset_id != dataset_id:
        raise ValueError(
            f"Dataset mismatch: model trained on {trained_dataset_id}, "
            f"but evaluation requested on {dataset_id}"
        )
    
    if "sales" not in df.columns or "prediction" not in df.columns:
        raise ValueError("Missing required columns: sales/prediction")
    
    y_true = df["sales"]
    y_pred = df["prediction"]

    test_rmse = rmse(y_true, y_pred)
    test_mae = mean_absolute_error(y_true, y_pred)
    test_mape = mape(y_true, y_pred)
    test_wmape = wmape(y_true, y_pred)

    df["errors"] = y_true - y_pred
    df["abs_errors"] = np.abs(df["errors"])

    error_mean = df["errors"].mean()
    error_std = df["errors"].std()

    conn = None
    try:
        conn = get_connection()
        log_evaluation_metrics(
            conn=conn,
            run_id=None,  
            dataset_id=dataset_id,
            model_version=mlflow_run_id,
            metrics={
                "rmse": test_rmse,
                "mae": test_mae,
                "mape": test_mape,
                "wmape": test_wmape,
                "error_mean": error_mean,
                "error_std": error_std,
            },
            slice_key="overall",
        )    
        
    finally:
        if conn:
            conn.close()

    store_rmse = df.groupby("store_id").apply(
        lambda x: rmse(x["sales"], x["prediction"])
    )

    worst_stores = store_rmse.sort_values(ascending=False).head(5)
    best_stores = store_rmse.sort_values(ascending=True).head(5)
    
    dept_rmse = df.groupby("dept_id").apply(
        lambda x: rmse(x["sales"], x["prediction"])
    )
    
    worst_depts = dept_rmse.sort_values(ascending=False).head(5)
    best_depts = dept_rmse.sort_values(ascending=True).head(5)

    with mlflow.start_run(run_id=mlflow_run_id):

        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mape", test_mape)
        mlflow.log_metric("test_wmape", test_wmape)

        mlflow.log_metric("error_mean", error_mean)
        mlflow.log_metric("error_std", error_std)

        mlflow.log_param("worst_stores", ",".join(map(str, worst_stores.index)))
        mlflow.log_param("best_stores", ",".join(map(str, best_stores.index)))

        mlflow.log_param("worst_depts", ",".join(map(str, worst_depts.index)))
        mlflow.log_param("best_depts", ",".join(map(str, best_depts.index)))
    
    logger.info("Evaluation completed")