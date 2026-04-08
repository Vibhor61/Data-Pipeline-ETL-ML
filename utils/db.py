import os
import psycopg2

DB_CONFIG = {
    "host": os.getenv("PGHOST", "postgres"),
    "port": int(os.getenv("PGPORT", "5432")),
    "database": os.getenv("PGDATABASE", "retail_dw"),
    "user": os.getenv("PGUSER", "airflow"),
    "password": os.getenv("PGPASSWORD", "airflow"),
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)
