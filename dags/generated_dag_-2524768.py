from __future__ import annotations

import os

import pandas as pd
import pendulum

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook


def _extract_calculate_and_save_sales_data(**kwargs):
    """
    Connects to PostgreSQL, calculates total daily sales for the execution date,
    and saves the result to a CSV file.
    """
    # The execution date is available in the Airflow context
    execution_date = kwargs["ds"]
    output_path = "/opt/airflow/reports/daily_sales.csv"
    output_dir = os.path.dirname(output_path)

    # Ensure the report directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a hook to connect to the PostgreSQL database.
    # Assumes a connection with conn_id 'postgres_default' is configured in Airflow UI.
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")

    # SQL query to calculate the total sales for the given execution date.
    # Assumes a 'sales_data' table with 'sale_date' and 'amount' columns.
    # Using parameters for safety against SQL injection.
    sql_query = """
        SELECT
            %(execution_date)s AS report_date,
            SUM(amount) AS total_sales
        FROM sales_data
        WHERE sale_date = %(execution_date)s;
    """
    params = {"execution_date": execution_date}

    # Execute the query and fetch the result into a pandas DataFrame
    sales_df = pg_hook.get_pandas_df(sql=sql_query, parameters=params)

    # Save the DataFrame to the specified CSV file, overwriting if it exists.
    sales_df.to_csv(output_path, index=False, header=True)


with DAG(
    dag_id="daily_sales_report",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule_interval="@daily",
    catchup=False,
    default_args={
        "owner": "airflow",
        "retries": 1,
    },
    description="A DAG to generate a daily sales report from a PostgreSQL database.",
) as dag:
    generate_report_task = PythonOperator(
        task_id="generate_daily_sales_report",
        python_callable=_extract_calculate_and_save_sales_data,
    )