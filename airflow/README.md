# Airflow orchestration

The `fd001_rul_candidate_pipeline` DAG schedules the repeatable batch portion of
the ML lifecycle:

```text
prepare → train and compare → package → register → benchmark → human approval
```

The DAG deliberately stops at `manual_approval_required`. Approval and promotion
remain audited human decisions and are performed with the existing CLI commands:

```bash
uv run pm-platform model approve-candidate ...
uv run pm-platform model promote-approved ...
```

The streaming consumer remains an always-on process and is not managed by this
DAG. Airflow is responsible for scheduled batch work, retries, task dependencies,
and run history.

## Local runtime setup

Install the pinned Airflow environment and initialize its local metadata database:

```bash
./scripts/setup_airflow.sh
```

Start the scheduler and webserver together:

```bash
./scripts/run_airflow.sh
```

Airflow will print a generated local login password. Open
`http://localhost:8080`, enable `fd001_rul_candidate_pipeline`, and trigger a
manual run from the UI. The run pauses at `manual_approval_required`; review the
candidate evidence, execute the audited approval command manually, and then set
the Airflow Variable `fd001_rul_approval_granted` to `true` if the workflow gate
should complete. The local runtime uses SQLite for learning and workflow
verification; a production deployment should use Airflow's external metadata
database and a separate executor.

Each run derives its training-run and model-release names from the Airflow run ID.
This keeps manual reruns immutable and prevents a second run on the same calendar
day from trying to overwrite the first run's artifacts. The approval sensor checks
the variable every 60 seconds and uses reschedule mode so it does not occupy a
worker slot while waiting.

## Local deployment assumption

The scheduler and worker mount the project at `/opt/predictive-maintenance` and
have the project virtual environment plus MLflow and PostgreSQL configuration
available. Airflow dependencies are intentionally kept separate from the main
runtime environment because Airflow is an orchestration service, not a prediction
runtime dependency.
