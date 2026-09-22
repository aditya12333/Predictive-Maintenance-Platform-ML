#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIRECTORY/.." && pwd)"
MLFLOW_STATE_DIRECTORY="$PROJECT_ROOT/artifacts/mlflow"
MLFLOW_PORT="${PM_MLFLOW_PORT:-5000}"

mkdir -p "$MLFLOW_STATE_DIRECTORY/artifacts"

export MLFLOW_DISABLE_AGENT_HINT=1

exec "$PROJECT_ROOT/.venv/bin/mlflow" server \
  --backend-store-uri "sqlite:///$MLFLOW_STATE_DIRECTORY/mlflow.db" \
  --artifacts-destination "$MLFLOW_STATE_DIRECTORY/artifacts" \
  --host 127.0.0.1 \
  --port "$MLFLOW_PORT" \
  --workers 1
