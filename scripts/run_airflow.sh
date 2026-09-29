#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_VENV="${PROJECT_ROOT}/.airflow-venv"

if [[ ! -x "${AIRFLOW_VENV}/bin/airflow" ]]; then
  echo "Airflow is not installed. Run ./scripts/setup_airflow.sh first." >&2
  exit 2
fi

export AIRFLOW_HOME="${PROJECT_ROOT}/airflow/.local"
export AIRFLOW__CORE__DAGS_FOLDER="${PROJECT_ROOT}/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export PM_AIRFLOW_PROJECT_ROOT="${PROJECT_ROOT}"
export PATH="${AIRFLOW_VENV}/bin:${PATH}"

exec "${AIRFLOW_VENV}/bin/airflow" standalone
