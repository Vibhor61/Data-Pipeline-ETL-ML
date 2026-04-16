import os
import psycopg2


def get_connection():
    DB_CONFIG = {
        "host": os.getenv("ETL_HOST"),
        "port": int(os.getenv("ETL_PORT")),
        "database": os.getenv("ETL_DB"),
        "user": os.getenv("ETL_USER"),
        "password": os.getenv("ETL_PASSWORD"),
    }

    return psycopg2.connect(**DB_CONFIG)
