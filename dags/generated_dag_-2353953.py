from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

# Define default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="hello_world_dag",
    default_args=default_args,
    description="A simple DAG that prints 'Hello, World!' every hour.",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule_interval="@hourly",
    catchup=False,
    tags=["example"],
) as dag:
    # Define the single task using the BashOperator
    print_hello_task = BashOperator(
        task_id="print_hello",
        bash_command="echo 'Hello, World!'",
    )