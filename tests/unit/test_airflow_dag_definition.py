"""Static safety checks for the Airflow training DAG definition."""

from pathlib import Path

DAG_SOURCE = Path("airflow/dags/rul_training_pipeline.py").read_text()


def test_training_dag_has_ordered_batch_tasks_and_manual_boundary() -> None:
    expected_tasks = (
        "task_id=\"prepare_dataset\"",
        "task_id=\"train_and_compare\"",
        "task_id=\"package_candidate\"",
        "task_id=\"register_candidate\"",
        "task_id=\"benchmark_candidate\"",
        "task_id=\"manual_approval_required\"",
    )
    for task in expected_tasks:
        assert task in DAG_SOURCE
    assert "model approve-candidate" not in DAG_SOURCE
    assert "model promote-approved" not in DAG_SOURCE
    assert "fd001_rul_approval_granted" in DAG_SOURCE
    assert "mode=\"reschedule\"" in DAG_SOURCE
    assert "schedule=\"0 2 * * *\"" in DAG_SOURCE
    assert "dag_run.run_id" in DAG_SOURCE
    assert "RUN_NAME = f\"fd001-airflow-{RUN_KEY}\"" in DAG_SOURCE
