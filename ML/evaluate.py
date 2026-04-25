"""
ML Evaluation Pipeline

Computes prediction quality metrics and logs evaluation artifacts to MLflow.

Input: model predictions and run metadata
Output: evaluation metrics and slice-level artifact reports

Core design principles:
- evaluation is traceable to a specific MLflow run
- metrics expose both global and slice-level performance
- failures are raised early for invalid input artifacts
"""

import numpy as np
import pandas as pd
import mlflow
import os
import tempfile
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

TARGET_COL = "sales"
STORE_COL = "store_id"
DEPT_COL = "dept_id"
def rmse(y_true, y_pred):
    """
    Compute root mean squared error.
    Args:
        y_true: true target values
        y_pred: predicted target values
    Returns:
        float: RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    """
    Compute mean absolute percentage error.
    Args:
        y_true: true target values
        y_pred: predicted target values
    Returns:
        float: MAPE percentage
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def wmape(y_true, y_pred):
    """
    Compute weighted mean absolute percentage error.
    Args:
        y_true: true target values
        y_pred: predicted target values
    Returns:
        float: WMAPE score
    """
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)


def compute_slice_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Compute evaluation metrics aggregated by slice.
    Args:
        df (pd.DataFrame): dataset containing actual and prediction columns
        group_col (str): column name to group by for slice metrics
    Returns:
        pd.DataFrame: aggregated slice performance metrics
    """
    metrics_df = (
        df.groupby(group_col).apply(
            lambda x: pd.Series({
                "rmse": rmse(x[TARGET_COL], x["prediction"]),
                "mae": mean_absolute_error(x[TARGET_COL], x["prediction"]),
                "wmape": wmape(x[TARGET_COL], x["prediction"]),
                "bias": (x[TARGET_COL] - x["prediction"]).mean(),
                "rows": len(x)
            })
        )
        .reset_index()
    )
    return metrics_df


def evaluate_pipeline(pred_path: str, run_id: str, dataset_id: str, predict_mlflow_run_id: str):
    """
    Run evaluation for a prediction artifact and log metrics to MLflow.
    Args:
        pred_path (str): path to prediction parquet file
        run_id (str): current evaluation pipeline run id
        dataset_id (str): dataset identifier used at training time
        predict_mlflow_run_id (str): MLflow run id for the trained model
    Returns:
        None
    Raises:
        RuntimeError: when no active MLflow run exists
        ValueError: when prediction file columns are invalid or dataset mismatch occurs
    """
    logger.info("Evaluation pipeline started for run_id=%s dataset_id=%s", run_id, dataset_id)
    
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run found in evaluate stage")
    
    run = mlflow.get_run(predict_mlflow_run_id)
    trained_dataset_id = run.data.params.get("dataset_id")
    if trained_dataset_id != dataset_id:
        raise ValueError(
            f"Dataset mismatch: model trained on {trained_dataset_id}, "
            f"evaluate requested on {dataset_id}"
        )

    pred_df = pd.read_parquet(pred_path)

    required_cols = {"actual", "prediction"}
    if not required_cols.issubset(pred_df.columns):
        raise ValueError(
            f"Predictions file must contain columns {required_cols}"
        )
    
    y_true = pred_df["actual"]
    y_pred = pred_df["prediction"]

    metrics = {
        "test_rmse": rmse(y_true, y_pred),
        "test_mae": mean_absolute_error(y_true, y_pred),
        "test_mape": mape(y_true, y_pred),
        "test_wmape": wmape(y_true, y_pred),
        "test_r2": r2_score(y_true, y_pred)
    }

    errors = y_true - y_pred
    metrics["error_mean"] = errors.mean()
    metrics["error_std"] = errors.std()

    mlflow.log_metrics(metrics)

    store_metrics = compute_slice_metrics(pred_df, STORE_COL)
    dept_metrics = compute_slice_metrics(pred_df, DEPT_COL)

    mlflow.log_metric("worst_store_wmape", store_metrics["wmape"].max())
    mlflow.log_metric("worst_dept_wmape", dept_metrics["wmape"].max())

    mlflow.log_metric("best_store_wmape", store_metrics["wmape"].min())
    mlflow.log_metric("best_dept_wmape", dept_metrics["wmape"].min())

    with tempfile.TemporaryDirectory() as tmpdir:

        store_path = os.path.join(tmpdir, "store_metrics.csv")
        dept_path = os.path.join(tmpdir, "department_metrics.csv")

        store_metrics.to_csv(store_path, index=False)
        dept_metrics.to_csv(dept_path, index=False)

        mlflow.log_artifact(store_path, artifact_path="slice_metrics")
        mlflow.log_artifact(dept_path, artifact_path="slice_metrics")

    logger.info("Evaluation completed run_id=%s", run_id)