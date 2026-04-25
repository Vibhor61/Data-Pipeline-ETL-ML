-- Airflow
CREATE DATABASE airflow_db;
-- MlFlow
CREATE DATABASE mlflow_db;

\c retail_dw;
-- Bronze
\i /docker-entrypoint-initdb.d/bronze.sql

-- Silver
\i /docker-entrypoint-initdb.d/silver.sql

-- Gold
\i /docker-entrypoint-initdb.d/gold.sql

-- ETL-Metadata
\i /docker-entrypoint-initdb.d/etl_metadata.sql

-- ML-Metadata
\i /docker-entrypoint-initdb.d/ml_metadata.sql