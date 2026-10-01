"""Scheduled operational monitoring report."""

import os
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT_ROOT = os.environ.get("PM_AIRFLOW_PROJECT_ROOT", "/opt/predictive-maintenance")


def request_candidate_if_needed() -> None:
    """Trigger candidate training once for a sustained failure sequence."""

    from airflow.api.common.trigger_dag import trigger_dag

    sys.path.insert(0, str(Path(PROJECT_ROOT) / "src"))
    from predictive_maintenance.monitoring.retraining import evaluate_report_history

    report_root = Path(PROJECT_ROOT) / "artifacts" / "monitoring"
    decision = evaluate_report_history(report_root / "history")
    request_path = report_root / "retraining-request.json"
    if decision.decision.value != "request_candidate" or request_path.exists():
        return
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        json.dumps(
            {
                "decision": decision.decision.value,
                "reason": decision.reason,
                "consecutive_failed_windows": decision.consecutive_failed_windows,
                "requested_at": datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
        )
        + "\n"
    )
    trigger_dag(
        dag_id="fd001_rul_candidate_pipeline",
        run_id=f"monitoring-retraining-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}",
        conf={"reason": decision.reason},
    )

with DAG(
    dag_id="operational_monitoring_pipeline",
    description="Evaluate persisted data quality and prediction availability",
    start_date=datetime(2026, 1, 1),
    schedule="*/15 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "predictive-maintenance",
        "depends_on_past": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["predictive-maintenance", "monitoring"],
) as dag:
    run_monitoring = BashOperator(
        task_id="run_monitoring_report",
        bash_command=(
            "set -euo pipefail\n"
            f"cd {PROJECT_ROOT}\n"
            ".venv/bin/pm-platform monitoring run "
            "--window-hours 24 "
            "--output-path artifacts/monitoring/latest.json"
        ),
    )

    request_retraining_candidate = PythonOperator(
        task_id="request_retraining_candidate",
        python_callable=request_candidate_if_needed,
    )

    run_monitoring >> request_retraining_candidate
