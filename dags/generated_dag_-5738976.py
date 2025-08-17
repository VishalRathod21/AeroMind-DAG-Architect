import pendulum
import pandas as pd
import os

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

def _extract_calculate_and_save_sales_report(ds=None, **kwargs):
    """
    Connects to PostgreSQL, extracts sales data for the execution date,
    calculates the total sales, and saves the result to a CSV file.

    The execution date (ds) is passed by Airflow.
    """
    # 1. Set up the connection using the PostgresHook
    # Assumes a connection with conn_id 'postgres_default' is configured in Airflow
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')

    # 2. Define the SQL query to calculate total daily sales
    # Assumes a table 'sales_data' with columns 'sale_date' and 'amount'
    sql_query = f"""
        SELECT
            '{ds}' AS report_date,
            SUM(amount) AS total_sales
        FROM sales_data
        WHERE sale_date = '{ds}';
    """

    # 3. Execute the query and fetch the result into a pandas DataFrame
    print(f"Executing query for date: {ds}")
    sales_df = pg_hook.get_pandas_df(sql=sql_query)

    # 4. Define the output path and ensure the directory exists
    output_dir = '/opt/airflow/reports'
    output_file = os.path.join(output_dir, 'daily_sales.csv')
    os.makedirs(output_dir, exist_ok=True)

    # 5. Save the DataFrame to a CSV file, overwriting if it exists
    sales_df.to_csv(output_file, index=False)
    print(f"Successfully saved daily sales report to {output_file}")
    print(f"Report content:\n{sales_df.to_string()}")


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': pendulum.datetime(2023, 1, 1, tz="UTC"),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': pendulum.duration(minutes=5),
}

with DAG(
    dag_id='daily_sales_report',
    default_args=default_args,
    description='A DAG to calculate daily sales and save the report as a CSV file.',
    schedule_interval='@daily',
    catchup=False,
    tags=['reporting', 'postgres'],
) as dag:
    generate_report_task = PythonOperator(
        task_id='generate_daily_sales_report',
        python_callable=_extract_calculate_and_save_sales_report,
    )