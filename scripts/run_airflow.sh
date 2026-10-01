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

mkdir -p "${AIRFLOW_HOME}"
"${AIRFLOW_VENV}/bin/airflow" db migrate

cleanup() {
  kill "${SCHEDULER_PID:-}" "${TRIGGERER_PID:-}" "${WEBSERVER_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${AIRFLOW_VENV}/bin/airflow" scheduler &
SCHEDULER_PID=$!
"${AIRFLOW_VENV}/bin/airflow" triggerer &
TRIGGERER_PID=$!
"${AIRFLOW_VENV}/bin/airflow" webserver --port 8080 &
WEBSERVER_PID=$!

echo "Airflow dashboard: http://127.0.0.1:8080"
echo "Press Ctrl-C to stop Airflow."
wait "${SCHEDULER_PID}" "${TRIGGERER_PID}" "${WEBSERVER_PID}"
