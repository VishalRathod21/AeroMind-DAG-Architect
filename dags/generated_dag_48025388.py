import csv
import os
from datetime import datetime

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook


def _extract_calculate_and_save_sales(ds=None, **kwargs):
    """
    Connects to PostgreSQL, extracts sales data for a given day,
    calculates the total, and saves it to a CSV file.
    """
    # Define the connection ID and the output file path
    postgres_conn_id = 'postgres_default'
    output_path = '/opt/airflow/reports/daily_sales.csv'
    
    # Instantiate the PostgresHook
    pg_hook = PostgresHook(postgres_conn_id=postgres_conn_id)
    
    # Define the SQL query to get the sum of sales for the execution date
    # Assumes a table 'sales_data' with columns 'sale_date' and 'amount'
    sql = "SELECT SUM(amount) FROM sales_data WHERE sale_date = %s"
    
    # Execute the query using the execution date (ds)
    # get_first() returns a single tuple, e.g., (Decimal('12345.67'),)
    result = pg_hook.get_first(sql, parameters=(ds,))
    
    # Process the result, handling the case of no sales
    total_sales = result[0] if result and result[0] is not None else 0
    
    print(f"Total sales for {ds}: {total_sales}")
    
    # Ensure the directory for the report exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the result to the specified CSV file
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date', 'total_sales'])
        writer.writerow([ds, total_sales])
        
    print(f"Report saved to {output_path}")


# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    dag_id='daily_sales_report',
    default_args=default_args,
    description='A DAG to calculate daily total sales and save to a CSV.',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # Define the single task using the PythonOperator
    generate_sales_report = PythonOperator(
        task_id='generate_sales_report',
        python_callable=_extract_calculate_and_save_sales,
    )