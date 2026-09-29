"""Scheduled FD001 training workflow with an explicit human approval boundary."""

import os
from datetime import datetime, timedelta

from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor

from airflow import DAG

PROJECT_ROOT = os.environ.get("PM_AIRFLOW_PROJECT_ROOT", "/opt/predictive-maintenance")
RUN_KEY = "{{ dag_run.run_id | replace('/', '_') | replace(':', '_') }}"
RUN_NAME = f"fd001-airflow-{RUN_KEY}"
TRAINING_RUN = f"artifacts/training-runs/{RUN_NAME}"
MODEL_RELEASE = f"rul-airflow-{RUN_KEY}"
MANIFEST = f"artifacts/model-releases/{MODEL_RELEASE}/manifest.json"

DEFAULT_ARGS = {
    "owner": "predictive-maintenance",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def platform_command(command: str) -> str:
    """Run a project CLI command from the mounted application directory."""

    return f"set -euo pipefail\ncd {PROJECT_ROOT}\n.venv/bin/pm-platform {command}"


def approval_is_granted() -> bool:
    """Require an explicit operator-set variable before closing the DAG run."""

    return Variable.get("fd001_rul_approval_granted", default_var="false").lower() == "true"


with DAG(
    dag_id="fd001_rul_candidate_pipeline",
    description="Prepare, train, evaluate, package, and register an FD001 candidate",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["predictive-maintenance", "training", "fd001"],
) as dag:
    prepare_dataset = BashOperator(
        task_id="prepare_dataset",
        bash_command=platform_command("data prepare"),
    )

    train_and_compare = BashOperator(
        task_id="train_and_compare",
        bash_command=platform_command(
            f"model train-compare --run-name {RUN_NAME}"
        ),
    )

    package_candidate = BashOperator(
        task_id="package_candidate",
        bash_command=platform_command(
            f"model package-candidate --training-run-directory {TRAINING_RUN} "
            f"--model-release {MODEL_RELEASE}"
        ),
    )

    register_candidate = BashOperator(
        task_id="register_candidate",
        bash_command=platform_command(
            f"model register-candidate --manifest-path {MANIFEST}"
        ),
    )

    benchmark_candidate = BashOperator(
        task_id="benchmark_candidate",
        bash_command=platform_command(
            f"model benchmark-candidate --manifest-path {MANIFEST} "
            f"--output-path artifacts/approval-evidence/{MODEL_RELEASE}-benchmark.json"
        ),
    )

    manual_approval_required = PythonSensor(
        task_id="manual_approval_required",
        python_callable=approval_is_granted,
        poke_interval=60,
        timeout=7 * 24 * 60 * 60,
        mode="reschedule",
        doc_md="Review evidence, then set Airflow Variable fd001_rul_approval_granted=true.",
    )

    (
        prepare_dataset
        >> train_and_compare
        >> package_candidate
        >> register_candidate
        >> benchmark_candidate
        >> manual_approval_required
    )
