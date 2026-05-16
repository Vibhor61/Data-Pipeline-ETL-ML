# Data-Pipeline-ETL-ML

A production-grade **Airflow-based ETL + ML pipeline** for retail demand forecasting using the M5 dataset. Implements medallion architecture (Bronze/Silver/Gold) for data layering and multi-stage ML orchestration with PostgreSQL metadata tracking and MLflow experiment management.

**Focus:** End-to-end ML pipeline with full lineage from raw CSV → cleaned dataset → train/validation/test splits → model training → batch predictions → evaluation metrics.

---

## Architecture Diagram

![Architecture Diagram](images/architecture.png)

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | Apache Airflow | DAG scheduling and task orchestration |
| **Metadata & State** | PostgreSQL | All-data-in-database approach: ETL runs, ML stages, dataset metadata, feature stores |
| **ETL Processing** | SQL (PostgreSQL Window Functions, CTEs, Joins) | Bronze → Silver → Gold transformations with Pandera validation |
| **Schema Validation** | Pandera | Runtime schema and grain validation for ETL outputs |
| **Data Formats** | Parquet, LibSVM | Columnar storage (Parquet); model-agnostic libsvm for ML training |
| **ML Training** | XGBoost, LightGBM | Gradient boosting models with early stopping and hyperparameter tuning |
| **ML Pipeline** | scikit-learn | Encoding, preprocessing, metric computation |
| **Experiment Tracking** | MLflow | Run management, artifact storage, metrics logging |
| **Dataset Serialization** | PyArrow | Parquet I/O and Arrow table operations |
| **Feature Engineering** | SQL + pandas (local) | Lag/rolling features in SQL; validation in pandas |
| **Memory Management** | psutil, joblib, gc | Memory profiling, encoder persistence, garbage collection |
| **Containerization** | Docker + Docker Compose | Reproducible environment with Airflow, PostgreSQL, MLflow |
| **Monitoring** | Airflow UI, MLflow UI | Pipeline and experiment visualization |
 

---

## Dataset 

This project uses the **M5 Forecasting Dataset** from the Walmart retail forecasting competition on Kaggle.

The dataset contains historical daily sales data of Walmart products across multiple stores and categories in the United States. It also includes calendar events, pricing information, and product hierarchy data, making it suitable for retail demand forecasting and time-series analysis.

### Dataset Files

- `sales_train_validation.csv` — Historical sales data
- `calendar.csv` — Date and event information
- `sell_prices.csv` — Product pricing data

### Dataset Source

[M5 Forecasting Dataset (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy?utm_source=chatgpt.com)

---

## Project Structure

```
Data-Pipeline-ETL-ML/
├── dags/
│   ├── etl_dag.py              # Bronze → Silver → Gold ETL pipeline
│   └── ml_dag.py               # Dataset → Train → Predict → Evaluate ML pipeline
├── ETL/
│   ├── bronze.py               # Raw data ingestion (CSV → PostgreSQL)
│   ├── silver.py               # Data cleaning & transformation
│   └── gold.py                 # Business-ready aggregation (features, aggregations)
├── ML/
│   ├── data_loader.py          # Dataset building (train/val/test splits)
│   ├── train.py                # Model training (XGBoost/LightGBM)
│   ├── predict.py              # Batch prediction on test set
│   └── evaluate.py             # Evaluation metrics (RMSE, MAE, WMAPE)
├── utils/
│   ├── db.py                   # PostgreSQL connection management
│   ├── etl_helpers.py          # ETL metadata lifecycle
│   └── ml_helpers.py           # ML metadata lifecycle
├── schema/
│   ├── init.sql                # Database initialization
│   ├── bronze.sql              # Bronze layer schema (raw data)
│   ├── silver.sql              # Silver layer schema (cleaned data)
│   ├── gold.sql                # Gold layer schema (features)
│   ├── etl_metadata.sql        # ETL run and step tracking tables
│   └── ml_metadata.sql         # ML pipeline and stage tracking tables
├── requirements/
│   ├── airflow.txt             # Airflow + ML dependencies 
│   └── mlflow.txt              # MLflow server dependencies
├── docker/                     # Dockerfile configurations
├── docker-compose.yml          # Full stack orchestration (postgres, airflow, mlflow)
├── .env                        # Environment variables 
└── readme.md                   # (This file)
```

---

## Key Features

### ETL Architecture
- Medallion architecture (Bronze → Silver → Gold)
- SQL-first transformations using PostgreSQL window functions, joins, and CTEs
- Run-date partition isolation for deterministic and reproducible execution
- Pandera-based schema and grain validation
- Idempotent re-runs using `ON CONFLICT DO UPDATE`

### ML Pipeline
- Streaming dataset construction from Gold layer using cursor-based loading
- Deterministic train, validation and test dataset generation
- Dataset snapshotting to Parquet for reproducibility, lineage tracking, and downstream reuse
- Additional LibSVM serialization for efficient XGBoost and LightGBM training
- Idempotent ML stages with metadata-backed re-runs and deterministic dataset versioning
- Feature engineering:
  - lag features
  - rolling statistics
  - temporal features
  - categorical encoding
- XGBoost and LightGBM training with validation-based early stopping
- Batch prediction and evaluation with RMSE, MAE, WMAPE, and R² metrics
- Baseline comparison against lag-based forecasting

### Metadata, Lineage & Orchestration
- PostgreSQL-backed metadata architecture for ETL and ML lifecycle tracking
- Full lineage tracking across runs, datasets, models, predictions, and evaluations
- Nested MLflow runs for experiment management and artifact tracking
- Airflow XCom-based context propagation between pipeline stages
- Dataset lineage validation across train → predict → evaluate stages

### Reliability & Infrastructure
- Dockerized deployment using Airflow, PostgreSQL, and MLflow
- Memory profiling and explicit garbage collection for large workloads
- Step-level failure tracking and centralized logging
- SQL whitelist validation and safe dynamic query construction
- Monitoring through Airflow UI and MLflow UI

---
## Metadata Architecture

The project uses PostgreSQL as the central metadata and storage layer for ETL orchestration, ML workflow tracking, dataset lineage, and experiment management.

### Core Databases

| Database | Purpose |
|----------|---------|
| `retail_dw` | Main analytical warehouse for ETL + ML pipelines |
| `airflow_db` | Metadata backend for Apache Airflow |
| `mlflow_db` | Tracking backend for MLflow experiments and runs |

### Warehouse Tables (`retail_dw`)

| Table | Purpose |
|-------|---------|
| `bronze_sales` | Raw ingested sales data |
| `calendar` | Calendar and event reference data |
| `sell_prices` | Historical product pricing data |
| `silver_table` | Cleaned and validated intermediate layer |
| `gold_table` | Business-ready feature layer for ML |
| `etl_pipeline_runs` | Tracks ETL DAG executions |
| `etl_pipeline_steps` | Stores ETL stage-level metrics and status |
| `ml_pipeline_runs` | Tracks ML workflow executions |
| `ml_runs` | Stores train/predict/evaluate stage metadata |
| `ml_dataset` | Maintains dataset lineage, schema hashes, and feature versions |

---

## ETL Pipeline (etl_dag.py)

### DAG Flow
```
init_run → bronze → silver → gold → finalize_pipeline
```

### Task Descriptions

| Task | Responsibility | Input | Output |
|------|-----------------|-------|--------|
| `init_run` | Create/reset pipeline run metadata, push run_id to XCom | dag_run.conf, run_date | run_id |
| `bronze` | Load raw CSV files (sales, calendar, sell_prices) | CSV paths | bronze_sales table |
| `silver` | Clean, validate, transform bronze data | bronze_sales | silver_table |
| `gold` | Aggregate and prepare business-ready dataset | silver_table | gold_table |
| `finalize_pipeline` | Evaluate all step states, mark run success/failed | step results | pipeline status |

---

## ML Pipeline (ml_dag.py)

### DAG Flow
```
create_run → build_dataset → train → predict → evaluate → finalize
```

### Task Descriptions

| Task | Responsibility | Input | Output |
|------|-----------------|-------|--------|
| `create_run` | Fetch ETL run_id, create ML pipeline run, start MLflow parent run | run_date | run_id, parent_mlflow_run_id |
| `build_dataset` | Load gold table, build train/val/test splits, write parquet & libsvm | gold_table | dataset_id, split paths |
| `train` | Train model on train/val splits, log artifacts to MLflow | split paths | train_mlflow_run_id |
| `predict` | Generate predictions on test set | test_path, train_mlflow_run_id | pred_path, pred_mlflow_run_id |
| `evaluate` | Compute metrics, log results to MLflow | pred_path | evaluation summary |
| `finalize` | Mark ML pipeline run as success | evaluation results | pipeline status |

---

## Configuration

### Environment Variables

```bash
# PostgreSQL
PG_USER=airflow
PG_PASSWORD=airflow
PG_DB=retail_dw

# Airflow Metadata DB
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow_db

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_BACKEND_STORE_URI=postgresql://airflow:airflow@postgres:5432/mlflow_db

# ETL Tables
BRONZE_TABLE=bronze_sales
SILVER_TABLE=silver_table
GOLD_TABLE=gold_table

# Data Paths
DATA_DIR=/opt/airflow/data
DATASETS_DIR=/opt/airflow/data/datasets
```

## Running the Pipeline

### Prerequisites
- Docker & Docker Compose
- PostgreSQL (or use docker-compose version)
- Python 3.9+
- Place the M5 data files inside the `data/` directory

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vibhor61/Data-Pipeline-ETL-ML.git
   cd Data-Pipeline-ETL-ML
   ```

2. **Build and start containers**
   ```bash
   docker-compose up -d
   ```

3. **Access Airflow UI**
   - Navigate to `http://localhost:8080`
   - Default credentials: airflow / airflow

4. **Trigger ETL DAG**
   - In Airflow UI, enable `retail_etl_dag`
   - Click "Trigger DAG" or schedule via cron
   - Monitor task execution and logs

5. **Trigger ML DAG** (after ETL completion)
   - Enable `retail_ml_dag`
   - Trigger DAG (reads gold table from completed ETL run)
   - Monitor MLflow experiments at `http://localhost:5000`


Historical data ranges from `2011-01-29` to `2016-04-24`.

### Access Airflow Container

```bash
docker exec -it airflow_scheduler bash
```

### Trigger ETL Pipeline
```bash
airflow dags trigger retail_etl_dag \
  --exec-date 2011-01-29
```
During intial bootstrap, the Bronze layer ingests calendar.csv and sell_prices.csv which should be configured directly inside Airflow DAG

# Trigger ML Pipeline
```bash
airflow dags trigger retail_ml_dag \
  --exec-date 2012-01-29
```

### Access Services

| Service | URL |
|---|---|
| Airflow UI | http://localhost:8080 |
| MLflow UI | http://localhost:5000 |

Default Airflow credentials: airflow/airflow

---

## Future work and improvements

- Hyperparameter optimization pipeline
- CI/CD integration for automated testing and deployment
- Real-time inference pipeline
- Data drift and model drift monitoring
- Online feature store integration
- Distributed model training
- Monitoring dashboards and alerting using Grafana and Prometheus

---