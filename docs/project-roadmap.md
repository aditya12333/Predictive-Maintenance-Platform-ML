# Predictive Maintenance Platform: Daily Delivery Roadmap

## Goal

Build a production-like predictive-maintenance platform that a real operations
team could use to monitor fleet health, investigate data-quality problems, review
predictions, and make maintenance decisions.

The project is also a learning project. Every phase must include explanation,
implementation, verification, and documentation before the next phase begins.

## Current status

Completed:

- Phase 1: Production problem definition.
- Phase 2: Data foundation and validation rules.
- Phase 3: Production architecture.
- Phase 4: Streaming reliability, event-time buffering, dead-letter handling,
  idle flushing, and data-quality warnings.
- Phase 5: Versioned features, four-model RUL comparison, quality-aware inference,
  idempotent prediction persistence, and transactional consumer integration.

Next:

- Phase 6: package and approve a serving candidate, strengthen promotion gates, and
  introduce experiment and model-registry lifecycle support.

The exact daily checkpoint is maintained in
`docs/daily-progress.md`.

The Phase 4 documentation is available in
`docs/phases/04-streaming-reliability-and-data-quality.md`.

The Phase 5 documentation is available in
`docs/phases/05-feature-generation-model-training-and-inference.md`.

## Remaining phases

### Phase 6: Training, evaluation, and model approval

Estimated duration: 5–7 focused days.

- Package the Phase 5 experiment winner against the serving artifact contract.
- Add experiment tracking and model registry integration.
- Compare future candidates against the approved champion.
- Track MAE, early-warning performance, latency, and resource usage.
- Version models, scalers, feature definitions, and evaluation reports.
- Define approval and rollback rules.

### Phase 7: Fleet dashboard and operational workflows

Estimated duration: 7–10 focused days.

The dashboard is a first-class product surface, not a demonstration-only page.

It will support:

- Fleet overview with engine health and prediction status.
- Engine detail view with telemetry history, estimated RUL, risk, and quality flags.
- Last valid prediction display when a newer prediction is withheld.
- Separate equipment-risk and data-quality alerts.
- Pending, degraded, dead-lettered, and recently recovered event views.
- Time-range filtering, engine search, sorting, and alert acknowledgement.
- Clear explanations for why a prediction is unavailable or degraded.
- Responsive layout, accessible components, loading states, empty states, and
  failure states.

The frontend will consume documented backend APIs rather than querying the
database directly. Frontend technology will be selected before implementation,
with a production-oriented React/Next.js approach as the default candidate.

### Phase 8: Airflow orchestration

Estimated duration: 4–6 focused days.

- Schedule dataset preparation and retraining workflows.
- Add validation, training, evaluation, and approval tasks.
- Make tasks retryable and observable.
- Explain how Airflow differs from the always-on telemetry consumer.
- Support manual reruns and failure recovery.

### Phase 9: Monitoring and observability

Estimated duration: 4–6 focused days.

- Structured logs and correlation identifiers.
- Consumer lag, processing latency, retry, and DLQ metrics.
- Data-quality issue rates and prediction availability metrics.
- Model performance and drift checks.
- Health and readiness endpoints.
- Dashboard-facing operational status summaries.

### Phase 10: Production deployment and CI/CD

Estimated duration: 5–8 focused days.

- Production Docker images and environment configuration.
- CI checks for tests, Ruff, Mypy, migrations, and frontend builds.
- Containerized deployment architecture.
- Managed PostgreSQL, Kafka-compatible streaming, object storage, and secrets.
- Staging deployment before production deployment.
- Safe model and application rollback procedures.

### Phase 11: Hardening and final delivery

Estimated duration: 4–6 focused days.

- Load and failure testing under fleet traffic.
- Duplicate, out-of-order, delayed, malformed, and missing telemetry tests.
- Security and access-control review.
- Dashboard usability review.
- Runbooks, API documentation, architecture diagrams, and interview material.
- Final end-to-end demonstration from telemetry ingestion to dashboard decision.

## Daily execution method

Each working day has one primary outcome:

1. Learn the relevant concept.
2. Define the design decision.
3. Implement a small, reviewable change.
4. Run automated checks and a manual verification.
5. Record the result in documentation.

We will not advance simply because code exists. A phase is complete when its
behavior is understood, tested, manually verified where appropriate, and
documented.

## Time estimate

The remaining work is approximately 25–38 focused development days. At 2–3 hours
per day, this is roughly 5–8 weeks. The estimate includes the full dashboard,
Airflow, deployment, testing, and documentation. It can change as requirements
become more detailed, especially for frontend polish and cloud deployment.

## Definition of done

The project is complete when a user can:

1. Ingest telemetry safely.
2. See data-quality and equipment-risk conditions.
3. Review current and last-valid predictions in the dashboard.
4. Understand why a prediction was generated, degraded, or withheld.
5. Inspect model and data lineage.
6. Operate the system through documented deployment and incident procedures.
