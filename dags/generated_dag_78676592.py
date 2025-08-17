import os
import pendulum
import pandas as pd

from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook

@dag(
    dag_id="daily_sales_report",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule_interval="@daily",
    catchup=False,
    default_args={
        "owner": "airflow",
        "retries": 3,
    },
    doc_md="""
    ### Daily Sales Report DAG
    This DAG connects to a PostgreSQL database, extracts data from the `sales_data` table,
    calculates the total daily sales for the execution date, and saves the result
    as a CSV file.
    """,
    tags=["reporting", "postgres"],
)
def daily_sales_report_dag():
    """
    A DAG to generate a daily sales report from a PostgreSQL database.
    """

    @task
    def extract_and_save_daily_sales_report(**kwargs):
        """
        Connects to PostgreSQL, calculates total daily sales for the execution date,
        and saves the result to a CSV file.
        """
        # Airflow's execution date is passed as 'ds' in the context
        execution_date = kwargs["ds"]
        output_path = "/opt/airflow/reports/daily_sales.csv"
        
        # 1. Connect to PostgreSQL using a hook
        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        
        # 2. Define the SQL query to calculate total sales for the execution date
        # Assumes a table 'sales_data' with columns 'sale_date' and 'amount'
        sql_query = f"""
            SELECT
                '{execution_date}' AS report_date,
                SUM(amount) AS total_sales
            FROM sales_data
            WHERE sale_date = '{execution_date}';
        """
        
        print(f"Executing query for date: {execution_date}")
        
        # 3. Execute the query and fetch the result into a pandas DataFrame
        sales_df = pg_hook.get_pandas_df(sql=sql_query)
        
        # Handle cases where there might be no sales for the day
        if sales_df.empty or pd.isna(sales_df.iloc[0]['total_sales']):
            print(f"No sales data found for {execution_date}. Creating a report with zero sales.")
            data = {'report_date': [execution_date], 'total_sales': [0]}
            sales_df = pd.DataFrame(data)
            
        # 4. Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 5. Save the DataFrame to the specified CSV file
        sales_df.to_csv(output_path, index=False)
        
        print(f"Successfully saved daily sales report to {output_path}")

    # Instantiate and run the task
    extract_and_save_daily_sales_report()

# Instantiate the DAG
daily_sales_report_dag()