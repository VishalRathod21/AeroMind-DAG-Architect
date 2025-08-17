import csv
import os
from datetime import datetime

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Define the output path as per the requirements
OUTPUT_FILE_PATH = "/opt/airflow/reports/daily_sales.csv"

def _extract_calculate_and_save_sales(**kwargs):
    """
    Connects to PostgreSQL, extracts sales data for the execution date,
    calculates the total, and saves it to a CSV file.
    """
    # The execution date is available in the Airflow context
    execution_date = kwargs["ds"]
    
    # Use the PostgresHook to connect to the database.
    # Assumes a connection with conn_id 'postgres_default' is configured in Airflow.
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    
    # SQL to calculate the total sales for the given execution date.
    # Assumes a table 'sales_data' with columns 'sale_date' and 'amount'.
    sql = f"SELECT SUM(amount) FROM sales_data WHERE sale_date = '{execution_date}';"
    
    # The hook's get_first() method executes the query and returns a single result.
    result = pg_hook.get_first(sql=sql)
    
    # The result is a tuple, e.g., (5000.00,). Handle case with no sales.
    total_sales = result[0] if result and result[0] is not None else 0
    
    print(f"Total sales for {execution_date}: {total_sales}")
    
    # Ensure the directory for the report exists.
    output_dir = os.path.dirname(OUTPUT_FILE_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the result to the specified CSV file.
    # This will overwrite the file each day with the new report.
    with open(OUTPUT_FILE_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        # Write the header
        writer.writerow(["date", "total_sales"])
        # Write the data row
        writer.writerow([execution_date, total_sales])
        
    print(f"Sales report saved to {OUTPUT_FILE_PATH}")


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "retries": 1,
    "catchup": False,
}

with DAG(
    dag_id="daily_sales_report",
    default_args=default_args,
    schedule_interval="@daily",
    description="A DAG to calculate daily total sales and save to a CSV.",
) as dag:
    
    generate_sales_report_task = PythonOperator(
        task_id="generate_sales_report",
        python_callable=_extract_calculate_and_save_sales,
    )