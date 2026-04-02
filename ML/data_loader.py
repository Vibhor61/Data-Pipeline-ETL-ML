import os 
import logging
import psycopg2
from psycopg2 import sql
import pandas as pd
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DB_CONFIG = {
    "host": os.getenv("PGHOST", "postgres"),
    "port": int(os.getenv("PGPORT", "5432")),
    "database": os.getenv("PGDATABASE", "retail_dw"),
    "user": os.getenv("PGUSER", "airflow"),
    "password": os.getenv("PGPASSWORD", "airflow"),
}


@dataclass
class DataLoader:
    table_name: str
    start_date: str
    end_date: str
    date_column: str 
    run_id: Optional[str] = None # For snapshotting a specific run, if needed


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_data(cfg: DataLoader) -> pd.DataFrame:
    with get_connection() as conn:
        base_query = sql.SQL("SELECT * FROM {table}").format(
                table=sql.Identifier(cfg.table_name)
            )

        where_clauses = [sql.SQL(
            "{date_col} BETWEEN %s AND %s"
        ).format(
            date_col=sql.Identifier(cfg.date_column)
        )]

        params = [cfg.start_date, cfg.end_date]

        if cfg.run_id is not None:
            where_clauses.append(sql.SQL("run_id = %s"))
            params.append(cfg.run_id)

        where_sql = sql.SQL("WHERE") + sql.SQL(" AND ").join(where_clauses)
        query = base_query + where_sql
        
        df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            logger.error("Loaded dataset is empty. Check time window / filters.")
            raise ValueError("Loaded dataset is empty. Check time window / filters.")
        
        return df