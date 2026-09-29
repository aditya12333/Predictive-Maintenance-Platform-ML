#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_VENV="${PROJECT_ROOT}/.airflow-venv"
AIRFLOW_HOME_DIR="${PROJECT_ROOT}/airflow/.local"
AIRFLOW_VERSION="2.10.5"
PYTHON_VERSION="3.12"
CONSTRAINTS_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

if [[ ! -x "${AIRFLOW_VENV}/bin/python" ]]; then
  uv venv --python "${PYTHON_VERSION}" "${AIRFLOW_VENV}"
fi

uv pip install \
  --python "${AIRFLOW_VENV}/bin/python" \
  --constraint "${CONSTRAINTS_URL}" \
  -r "${PROJECT_ROOT}/airflow/requirements.txt"

export AIRFLOW_HOME="${AIRFLOW_HOME_DIR}"
export AIRFLOW__CORE__DAGS_FOLDER="${PROJECT_ROOT}/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export PM_AIRFLOW_PROJECT_ROOT="${PROJECT_ROOT}"
export PATH="${AIRFLOW_VENV}/bin:${PATH}"
mkdir -p "${AIRFLOW_HOME}"

"${AIRFLOW_VENV}/bin/airflow" db migrate
echo "Airflow is ready. DAG directory: ${AIRFLOW__CORE__DAGS_FOLDER}"
echo "Run: ./scripts/run_airflow.sh"
