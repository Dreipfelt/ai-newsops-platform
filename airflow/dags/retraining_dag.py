"""
Airflow DAG — AI NewsOps retraining pipeline.

This DAG orchestrates the retraining workflow without embedding training logic in
Airflow itself. Each task calls a project script so the same commands can be run
locally, in CI, or from Airflow.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.empty import EmptyOperator
except ModuleNotFoundError:  # Allows static/unit tests without installing Airflow.
    DAG = None
    BashOperator = None
    EmptyOperator = None

PROJECT_ROOT = Path(os.getenv("NEWSOPS_PROJECT_ROOT", "/opt/airflow/project"))
PYTHON_BIN = os.getenv("NEWSOPS_PYTHON_BIN", "python")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DRIFT_THRESHOLD = os.getenv("DRIFT_THRESHOLD", "0.30")
MIN_F1 = os.getenv("MIN_CANDIDATE_F1", "0.65")

DEFAULT_ARGS = {
    "owner": "newsops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}


def project_cmd(command: str) -> str:
    """Build a shell command executed from the project root."""
    return f"cd {PROJECT_ROOT} && {command}"


if DAG is not None:
    with DAG(
        dag_id="newsops_retraining_pipeline",
        description="Detect drift, retrain model, evaluate and register candidate.",
        default_args=DEFAULT_ARGS,
        start_date=datetime(2026, 1, 1),
        schedule="0 2 * * *",
        catchup=False,
        max_active_runs=1,
        tags=["newsops", "mlops", "retraining"],
    ) as dag:
        start = EmptyOperator(task_id="start")

        validate_data = BashOperator(
            task_id="validate_data",
            bash_command=project_cmd(
                f"{PYTHON_BIN} scripts/validate_training_data.py "
                "--data-dir data/processed"
            ),
        )

        check_drift = BashOperator(
            task_id="check_drift",
            bash_command=project_cmd(
                f"DRIFT_THRESHOLD={DRIFT_THRESHOLD} "
                f"{PYTHON_BIN} src/retrain_trigger.py "
                "--drift-report monitoring/quick_drift_latest.json "
                "--output monitoring/retrain_decision.json"
            ),
        )

        retrain_candidate = BashOperator(
            task_id="retrain_candidate",
            bash_command=project_cmd(
                f"MLFLOW_TRACKING_URI={MLFLOW_URI} "
                f"MIN_CANDIDATE_F1={MIN_F1} "
                f"{PYTHON_BIN} src/retrain.py "
                "--decision-file monitoring/retrain_decision.json"
            ),
            execution_timeout=timedelta(hours=3),
        )

        register_model = BashOperator(
            task_id="register_model",
            bash_command=project_cmd(
                f"MLFLOW_TRACKING_URI={MLFLOW_URI} "
                f"{PYTHON_BIN} src/register_model.py"
            ),
        )

        end = EmptyOperator(task_id="end")

        start >> validate_data >> check_drift >> retrain_candidate >> register_model >> end
