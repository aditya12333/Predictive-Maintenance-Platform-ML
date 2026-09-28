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

## Local deployment assumption

The scheduler and worker mount the project at `/opt/predictive-maintenance` and
have the project virtual environment plus MLflow and PostgreSQL configuration
available. Airflow dependencies are intentionally kept separate from the main
runtime environment because Airflow is an orchestration service, not a prediction
runtime dependency.
