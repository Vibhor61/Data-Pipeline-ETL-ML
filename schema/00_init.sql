-- Bronze
\i /docker-entrypoint-initdb.d/00_bronze.sql

-- Silver
\i /docker-entrypoint-initdb.d/00_silver.sql

-- Gold
\i /docker-entrypoint-initdb.d/00_gold.sql

-- ETL-Metadata
\i /docker-entrypoint-initdb.d/00_etl_metadata.sql