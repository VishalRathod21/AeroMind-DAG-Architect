from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook


def extract_calculate_and_save_sales_report():
    """
    Connects to a PostgreSQL database, calculates the total daily sales
    from the 'sales_data' table, and saves the result to a CSV file.
    """
    # Instantiate the PostgresHook to connect to the database.
    # This assumes an Airflow connection with the ID 'postgres_default' is configured.
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")

    # Define the SQL query to calculate the sum of sales for the execution date.
    # '{{ ds }}' is an Airflow template variable representing the logical date (YYYY-MM-DD).
    # This assumes the 'sales_data' table has 'sale_date' and 'amount' columns.
    sql_query = """
        SELECT
            '{{ ds }}' AS report_date,
            SUM(amount) AS total_sales
        FROM sales_data
        WHERE sale_date = '{{ ds }}';
    """

    # Execute the query. The hook handles the connection and renders the Jinja template.
    # The result is returned as a pandas DataFrame.
    sales_df = pg_hook.get_pandas_df(sql=sql_query)

    # Define the output file path
    output_path = "/opt/airflow/reports/daily_sales.csv"
    output_dir = os.path.dirname(output_path)

    # Ensure the target directory exists before writing the file
    os.makedirs(output_dir, exist_ok=True)

    # Save the DataFrame to a CSV file. If the file exists, it will be overwritten.
    sales_df.to_csv(output_path, index=False)

    print(f"Successfully saved daily sales report to {output_path}")


# Define the default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "retries": 1,
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="daily_sales_report",
    default_args=default_args,
    description="A simple DAG to generate a daily sales report.",
    schedule_interval="@daily",
    catchup=False,
    tags=["reporting", "postgres"],
) as dag:
    # Define the single task in the DAG
    generate_report_task = PythonOperator(
        task_id="generate_daily_sales_report",
        python_callable=extract_calculate_and_save_sales_report,
    )