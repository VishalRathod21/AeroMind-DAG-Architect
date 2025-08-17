from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="minimal_example_dag",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    schedule=None,
    tags=["example"],
) as dag:
    # This is a placeholder DAG created because no specific requirements were provided.
    # It demonstrates a basic structure with a single task.
    start_task = BashOperator(
        task_id="start_task",
        bash_command="echo 'No requirements provided. This is a placeholder task.'",
    )