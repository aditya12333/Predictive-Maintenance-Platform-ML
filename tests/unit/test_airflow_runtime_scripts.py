"""Static checks for the local Airflow runtime scripts."""

from pathlib import Path


def test_airflow_setup_script_is_pinned_and_initializes_metadata() -> None:
    source = Path("scripts/setup_airflow.sh").read_text()
    assert "raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}" in source
    assert "airflow\" db migrate" in source
    assert "PM_AIRFLOW_PROJECT_ROOT" in source


def test_airflow_runner_requires_setup_and_starts_services() -> None:
    source = Path("scripts/run_airflow.sh").read_text()
    assert "Airflow is not installed" in source
    assert 'PATH="${AIRFLOW_VENV}/bin:${PATH}"' in source
    assert 'airflow\" scheduler' in source
    assert 'airflow\" webserver --port 8080' in source
