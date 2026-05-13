import numpy as np
import pandas as pd
import mlflow
import os
import tempfile
import gc 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

TARGET_COL = "sales"
PREDICTION_COL = "prediction"
BASELINE_PREDICTION_COL = "baseline_prediction"

STORE_COL = "store_id"
DEPT_COL = "dept_id"

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)


def compute_slice_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    metrics_df = (
        df.groupby(group_col).apply(
            lambda x: pd.Series({
                "rmse": rmse(x[TARGET_COL], x[PREDICTION_COL]),
                "mae": mean_absolute_error(x[TARGET_COL], x[PREDICTION_COL]),
                "wmape": wmape(x[TARGET_COL], x[PREDICTION_COL]),
                "bias": (x[TARGET_COL] - x[PREDICTION_COL]).mean(),
                "rows": len(x)
            })
        )
        .reset_index()
    )
    return metrics_df


def log_slice_metrics(metrics_df: pd.DataFrame, group_col: str):

    mlflow.log_metrics({
        f"{group_col}_avg_wmape": metrics_df["wmape"].mean(),
        f"{group_col}_best_wmape": metrics_df["wmape"].min(),
        f"{group_col}_worst_wmape": metrics_df["wmape"].max(),
        f"{group_col}_median_wmape": metrics_df["wmape"].median(),
        f"{group_col}_p90_wmape": metrics_df["wmape"].quantile(0.9),
    })

def evaluate_pipeline(pred_path: str, run_id: str, dataset_id: str, train_mlflow_run_id: str, pred_mlflow_run_id: str, mlflow_run_id: str, dataset_dir: str = None):
    logger.info(
        "Evaluation pipeline started for run_id=%s dataset_id=%s pred_mlflow_run_id=%s",
        run_id,
        dataset_id,
        pred_mlflow_run_id,
    )
    
    if dataset_dir is None:
        dataset_dir = os.path.dirname(pred_path)

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run")
    
    if active_run.info.run_id != mlflow_run_id:
        raise RuntimeError(
            f"Active run {active_run.info.run_id} != expected {mlflow_run_id}"
        )
        
    mlflow.set_tag("source_train_run_id", train_mlflow_run_id)
    mlflow.set_tag("source_pred_run_id", pred_mlflow_run_id)
    
    train_run = mlflow.get_run(train_mlflow_run_id)
    trained_dataset_id = train_run.data.params.get("dataset_id")
    if trained_dataset_id != dataset_id:
        raise ValueError(
            f"Dataset mismatch: model trained on {trained_dataset_id}, "
            f"evaluate requested on {dataset_id}"
        )

    pred_run = mlflow.get_run(pred_mlflow_run_id)

    pred_source_train_run = pred_run.data.tags.get(
        "source_train_run_id"
    )

    if pred_source_train_run != train_mlflow_run_id:
        raise ValueError(
            "Prediction run was generated from a different training run"
        )
    
    pred_df = pd.read_parquet(pred_path)
    logger.info("Loaded prediction file %s with %s rows", pred_path, len(pred_df))

    required_cols = {TARGET_COL, PREDICTION_COL, BASELINE_PREDICTION_COL, STORE_COL, DEPT_COL}
    if not required_cols.issubset(pred_df.columns):
        raise ValueError(
            f"Predictions file must contain columns {required_cols}"
        )
    
    y_true = pred_df[TARGET_COL]
    y_pred = pred_df[PREDICTION_COL]
    baseline_pred = pred_df[BASELINE_PREDICTION_COL]

    baseline_metrics = {
        "baseline_rmse": rmse(y_true, baseline_pred),
        "baseline_mae": mean_absolute_error(y_true, baseline_pred),
        "baseline_wmape": wmape(y_true, baseline_pred),
    }

    metrics = {
        "test_rmse": rmse(y_true, y_pred),
        "test_mae": mean_absolute_error(y_true, y_pred),
        "test_wmape": wmape(y_true, y_pred),
        "test_r2": r2_score(y_true, y_pred)
    }

    errors = y_true - y_pred
    metrics["error_mean"] = errors.mean()
    metrics["error_std"] = errors.std()
    
    mlflow.log_metrics(metrics)
    mlflow.log_metrics(baseline_metrics)

    improvement_over_baseline = (baseline_metrics["baseline_wmape"] - metrics["test_wmape"]) / baseline_metrics["baseline_wmape"] * 100
    mlflow.log_metric("wmape_improvement_over_baseline_pct", improvement_over_baseline)

    model_abs_error = np.abs(y_true - y_pred).mean()
    baseline_abs_error = np.abs(y_true - baseline_pred).mean()
    improvement_over_baseline_abs_error_pct = (baseline_abs_error - model_abs_error) / baseline_abs_error * 100
    mlflow.log_metric("prediction_abs_error_improvement_over_baseline_pct", improvement_over_baseline_abs_error_pct)

    store_metrics = compute_slice_metrics(pred_df, STORE_COL)
    dept_metrics = compute_slice_metrics(pred_df, DEPT_COL)
    log_slice_metrics(store_metrics, STORE_COL)
    log_slice_metrics(dept_metrics, DEPT_COL)

    logger.info("Logged overall metrics: %s", metrics)

    with tempfile.TemporaryDirectory() as tmpdir:

        store_path = os.path.join(tmpdir, "store_metrics.csv")
        dept_path = os.path.join(tmpdir, "department_metrics.csv")

        store_metrics.to_csv(store_path, index=False)
        dept_metrics.to_csv(dept_path, index=False)

        mlflow.log_artifact(store_path, artifact_path="slice_metrics")
        mlflow.log_artifact(dept_path, artifact_path="slice_metrics")

    del pred_df, store_metrics, dept_metrics, y_true, y_pred, errors
    gc.collect()
    logger.info("Evaluation completed run_id=%s", run_id)