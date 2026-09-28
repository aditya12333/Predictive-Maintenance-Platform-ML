"""Scheduled FD001 training workflow with an explicit human approval boundary."""

from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

from airflow import DAG

PROJECT_ROOT = "/opt/predictive-maintenance"
RUN_NAME = "fd001-airflow-{{ ds_nodash }}"
TRAINING_RUN = f"artifacts/training-runs/{RUN_NAME}"
MODEL_RELEASE = "rul-airflow-{{ ds_nodash }}"
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

    manual_approval_required = EmptyOperator(
        task_id="manual_approval_required",
        doc_md=(
            "Review the candidate, evaluation report, and benchmark evidence. "
            "Run the audited approval command manually after a human decision. "
            "This DAG intentionally does not approve or promote models."
        ),
    )

    (
        prepare_dataset
        >> train_and_compare
        >> package_candidate
        >> register_candidate
        >> benchmark_candidate
        >> manual_approval_required
    )
