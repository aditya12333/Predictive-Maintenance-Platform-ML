# Phase 3: Production Architecture Design

**Status:** Approved<br>
**Project:** AeroReliability Predictive Maintenance Platform<br>
**Architecture style:** Modular, event-driven and batch-first<br>
**Reference cloud:** AWS<br>
**Decision date:** 2026-09-05<br>
**Approved:** 2026-09-05

## 1. Purpose

Define the platform's responsibilities, boundaries, data flows, reliability behaviour and technology choices before adding more model or infrastructure code.

The architecture must support a client-facing system that is:

- Traceable from telemetry to maintenance decision.
- Resilient to connectivity and dependency failures.
- Safe under duplicate, late, missing or corrupt events.
- Scalable by component rather than only as one application.
- Maintainable by a small engineering team.
- Portable enough that logical contracts do not depend on one cloud vendor.
- Explicit about which outputs are current, stale, degraded or unavailable.

The target architecture is shown in [the Phase 3 architecture diagrams](../diagrams/03-production-architecture.md).

## 2. Architecture principles

1. **Logical design before vendor mapping.** Responsibilities remain stable even when a client changes infrastructure products.
2. **Asynchronous telemetry processing.** Ingestion acknowledges durable receipt; it does not wait for inference.
3. **At-least-once transport with idempotent business processing.** Retries are expected and cannot create duplicate predictions or alerts.
4. **Event time determines operational order.** Ingestion time measures delay and system behaviour.
5. **Immutable history with materialized current views.** Historical evidence is preserved while dashboards remain fast.
6. **Shared training and inference transformations.** One versioned feature package prevents training-serving skew.
7. **Human-controlled model promotion.** Drift or retraining never changes the champion automatically.
8. **Build once and promote exact artifacts.** Releases use immutable image, model, feature and configuration digests.
9. **Managed infrastructure where practical.** The product team builds predictive-maintenance capability, not database or broker control planes.
10. **Complexity must be earned.** No Kubernetes, separate feature store, distributed dataframe engine or excessive microservices without measured need.

## 3. Logical component boundaries

| Component | Responsibility | Initial deployment boundary |
| --- | --- | --- |
| Telemetry simulator or edge gateway | Emit versioned events, buffer during lost connectivity, retry until durable acknowledgement | Separate process/container |
| Ingestion API | Authenticate machines, enforce request limits and envelope contract, publish durably, return event identity | FastAPI service |
| Message broker | Buffer, order by partition, retain and replay events, distribute consumer work | Managed dependency; Redpanda locally |
| Raw-event archiver | Batch original events into immutable long-term objects | Background consumer |
| Telemetry-processing worker | Reorder, validate, build features, run inference, persist predictions, evaluate alerts | One modular Python worker initially |
| Notification worker | Deliver pending outbox notifications independently of alert transactions | Background worker |
| Operational API | Serve fleet state, predictions and alerts; enforce human workflows and authorization | Separate FastAPI service |
| Fleet dashboard | Display current and historical state; submit authorized human decisions | React and TypeScript application |
| Training pipeline | Build leakage-safe datasets, train and evaluate models, register candidates | Containerized batch job |
| Workflow orchestrator | Schedule and manage multi-step training jobs | Airflow when the workflow exists |
| Experiment tracker and registry | Record runs, artifacts and promotion lifecycle | MLflow service |
| Observability pipeline | Collect logs, metrics and traces and trigger operational alerts | OpenTelemetry plus monitoring backend |

Validation, sequence handling, feature transformation, inference and alert evaluation remain separate Python modules inside the initial processing worker. They become separate services only if scale, isolation or team ownership provides evidence for that split.

## 4. Online telemetry contract

Every event envelope must include at least:

- `event_id`: globally unique identity used for deduplication, tracking and audit.
- `engine_id`: equipment identity and broker partition key.
- `cycle`: sequence within an engine's observed lifecycle.
- `event_timestamp`: when the measurement occurred; used for operational ordering.
- `ingestion_timestamp`: assigned when the platform receives the event; used for latency and freshness monitoring.
- `schema_version`: selects the validated event contract.
- `source_id`: identifies the producing gateway or simulator.
- Operating settings and sensor measurements.
- A deterministic payload digest for conflict detection.

The HTTP endpoint returns `202 Accepted` only after the broker confirms durable receipt. The response contains `event_id` and status `RECEIVED`; it does not claim that validation or inference has finished.

Suggested initial endpoints:

| Method and path | Purpose |
| --- | --- |
| `POST /v1/telemetry` | Accept a versioned telemetry event |
| `GET /v1/events/{event_id}` | Return processing status and quality outcome |
| `GET /v1/fleet/health` | Return materialized current fleet state |
| `GET /v1/equipment/{engine_id}/predictions` | Return immutable prediction history |
| `GET /v1/alerts` | Query active and historical alerts |
| `PATCH /v1/alerts/{alert_id}` | Perform a version-checked alert transition |
| `POST /v1/inspections` | Record an engineer inspection |

## 5. Event delivery and ordering policy

### Delivery semantics

The broker provides at-least-once delivery. Consumers persist their business result before acknowledging the broker message.

A prediction uniqueness constraint such as `(event_id, model_version)` makes redelivery a safe no-op. Alert deduplication prevents repeated high-risk predictions from creating a new alert every cycle.

### Partitioning

Events use `engine_id` as the partition key. This preserves arrival order for one engine while allowing different engines to process in parallel. `event_id` is not a partition key because it would scatter one engine's cycles across partitions.

### Event-time reordering

Broker ordering cannot repair events that reached the broker out of event-time order. A per-engine reorder mechanism tracks:

- Next expected cycle.
- Highest observed cycle.
- Pending buffered cycles.
- Allowed lateness.
- Event-time watermark.

The allowed-lateness window is configurable. Waiting longer improves ordering but increases prediction latency. Once the watermark advances, newly arriving older events are classified as late and handled through an auditable correction policy.

### Duplicate and correction policy

| Condition | Behaviour |
| --- | --- |
| Same identity and same payload | Treat as retry; produce no duplicate result |
| Same identity and different payload | Flag conflict and quarantine; do not apply latest-arrival-wins |
| New cycle or legitimate newer event | Process in event-time sequence |
| Explicit authorized correction | Append a revision, preserve the original and recalculate affected results when required |

## 6. Missing, corrupt and stale data policy

If a cycle is missing but later events arrive, ingestion and sequence processing continue. The system:

- Records a `cycle_gap` issue.
- Marks data quality `DEGRADED`.
- Preserves the gap rather than inventing a value.
- Raises a separate data-quality alert.
- Produces a prediction only if the versioned model-input contract permits the available data.
- Otherwise records prediction status `WITHHELD` with an explicit reason.

When newer telemetry cannot be scored, the dashboard may show the last valid prediction only as historical context. It must show:

- Status `STALE`.
- The cycle and event time on which it was based.
- Its prediction generation time.
- The latest received cycle.
- The active data-quality concern.

The system does not silently decrement a stale RUL value or label an engine healthy because current telemetry is invalid. A pre-existing critical maintenance alert stays active until a human or a defined rule resolves it; corrupt data cannot close it.

Equipment-risk alerts and data-quality alerts remain separate because they describe different operational problems.

## 7. Storage ownership

| Storage | Owned information | Why |
| --- | --- | --- |
| Kafka-compatible broker | Events in motion, retained replay window and consumer offsets | Buffering, retry, ordering and fan-out |
| S3-compatible object storage | Original event batches, trusted Parquet datasets, quarantine artifacts, evaluation outputs and model artifacts | Durable, low-cost immutable history and batch access |
| PostgreSQL | Current equipment state, event metadata, predictions, alerts, inspections, users, audit history, deployments and outbox | Transactions, relationships, constraints and fast operational queries |
| MLflow | Experiment metadata, model versions, aliases and artifact references | Reproducible model lifecycle |

The broker is not the sole permanent source of telemetry. An archival consumer writes batched compressed raw events to object storage. Validated analytical tables use Parquet. One object per telemetry event is avoided because large numbers of tiny objects create inefficient storage and batch-processing behaviour.

The existing local `data/raw` and `data/interim` layout currently represents the immutable historical-storage responsibility.

## 8. Operational data model

Planned PostgreSQL entities include:

- `equipment`
- `telemetry_event`
- `prediction`
- `data_quality_issue`
- `alert`
- `alert_event`
- `inspection`
- `maintenance_action`
- `outbox_event`
- `model_deployment`

Predictions and alert events are append-only. A materialized equipment-current-state representation supports dashboard queries while retaining links to immutable evidence.

Alert state changes update the current alert row and append an audit event in one transaction. Optimistic concurrency uses a record version so simultaneous human updates cannot silently overwrite one another; a stale update receives `409 Conflict`.

## 9. Transaction and notification reliability

The processing worker performs the prediction, alert decision and outbox insertion in one PostgreSQL transaction. It acknowledges the broker only after commit.

Failure cases are safe:

- Crash before commit: broker redelivers and no partial result is visible.
- Crash after commit but before acknowledgement: broker redelivers and uniqueness constraints make processing a safe no-op.
- Notification failure: prediction and alert remain authoritative while the outbox worker retries separately.

Notification retries use backoff and an idempotency key. Repeated failures become an operational incident rather than silently deleting the request.

## 10. Offline ML and model lifecycle

New telemetry triggers inference using the current approved model; it does not trigger model training.

The training pipeline consumes trusted versioned data and performs:

1. Manifest and schema verification.
2. RUL and risk-label construction without future-data leakage.
3. Versioned feature construction.
4. Complete-engine data splitting.
5. Preprocessing fitted only on the training split.
6. Baseline and candidate training.
7. Multi-metric evaluation.
8. MLflow experiment logging and candidate registration.

Training and inference import the same versioned feature-transformation package. Inference loads preprocessing parameters fitted during training; it never refits a scaler or imputer on live telemetry.

A candidate cannot replace the champion based only on lower MAE. The current regression promotion gates include RMSE, NASA score, error stability, latency and compatibility. Human approval is required and audited. Classification and alert-policy gates will be defined later if that scope is added.

The registry retains immutable candidate, champion and previous versions. Rollback restores the exact prior release rather than rebuilding it.

## 11. Technology decisions and rationale

| Technology | Use | Reason for initial selection |
| --- | --- | --- |
| Python | Data, ML, workers and APIs | Existing expertise and ML ecosystem |
| FastAPI | Ingestion and operational APIs | Pydantic integration, OpenAPI, async I/O and testability |
| Pydantic | Event, API and metadata contracts | Runtime validation and explicit schemas |
| Polars | Batch data and feature transformations | Typed columnar execution, Parquet support and efficient local scaling |
| Redpanda locally / managed Kafka-compatible broker | Durable event log | Partitions, replay, consumer groups and back-pressure visibility |
| PostgreSQL | Operational and transactional state | Transactions, unique constraints, indexes and relational integrity |
| SQLAlchemy and Alembic | Database access and migrations | Explicit persistence layer and controlled schema evolution |
| S3-compatible object storage | Historical data and artifacts | Durable immutable storage and batch-tool compatibility |
| MLflow | Experiment tracking and model registry | Open model lifecycle with PostgreSQL and S3 integration |
| Airflow | Training workflow orchestration | Industry-relevant DAG scheduling, retries and operational history |
| Docker and Docker Compose | Packaging and local platform | Reproducible deployable units and local dependency integration |
| OpenTelemetry | Application telemetry | Vendor-neutral logs, metrics and traces |
| Prometheus and Grafana locally | Metrics and dashboards | Transparent, inspectable monitoring concepts |
| React, TypeScript and Vite | Fleet dashboard | Maintainable interactive client application without unnecessary server rendering |
| Terraform | Infrastructure as code | Reviewable, repeatable and portable infrastructure definitions |
| GitHub Actions | CI/CD | Automated quality gates, image build and auditable deployment flow |
| AWS reference services | Production deployment mapping | Managed containers, database, storage, identity and monitoring concepts |

Airflow is introduced only after plain Python training functions form a meaningful workflow. Spark, Kubernetes, a dedicated time-series database and a feature-store service remain deferred until performance or serving requirements justify them.

## 12. Observability

Every component emits structured logs, metrics and traces correlated by event, prediction and trace identifiers.

Monitoring is separated into:

- Service health: latency, errors, availability, worker restarts and database waits.
- Streaming health: arrival rate, processing rate, consumer lag and oldest-event age.
- Data health: missingness, schema failures, conflicts, lateness and staleness.
- Model health: prediction distribution, drift, performance when outcomes arrive and inference duration.
- Business health: RUL error, prediction stability and maintenance outcomes.

A healthy API does not imply healthy data or a healthy model. Dashboard states must distinguish `CURRENT`, `STALE`, `DEGRADED`, `WITHHELD` and `UNAVAILABLE` rather than silently substituting a healthy state.

Detailed reliability learning and failure-injection exercises are retained in [Project Learning Checkpoints](../learning-checkpoints.md).

## 13. Security and audit boundary

- Human authentication uses an OpenID Connect identity provider and short-lived tokens.
- Role-based authorization separates reliability managers, maintenance engineers, ML engineers, model approvers and administrators.
- Edge producers use machine credentials rather than human passwords.
- Internal services use workload identity and least-privilege permissions.
- Public access is limited to HTTPS dashboard and API entry points behind load balancing and rate limits.
- Databases, broker, workers, MLflow and orchestration remain private.
- Data is encrypted in transit and at rest.
- Secrets remain in a secrets manager, not images, source code or committed environment files.
- Model promotion, rollback and human alert decisions create immutable audit records.

## 14. Deployment and release architecture

Each API, worker, simulator and batch job is a containerized deployable. Stateful data remains outside containers.

Images are built once and identified by immutable digest. The same tested digest moves from staging to production. A release manifest pins:

- Container image digest.
- Model-bundle digest and registry version.
- Feature-schema version.
- Configuration version.
- Database migration version.

GitHub Actions provides CI quality gates. Production requires explicit approval. Database migrations use expand-and-contract changes to remain compatible with rolling deployments. Terraform plans require review before application.

The AWS reference maps containers to ECS Fargate, images to ECR, objects to S3, PostgreSQL to RDS, secrets to Secrets Manager, and application signals to managed monitoring through OpenTelemetry. Kubernetes is not part of the initial design.

## 15. Provisional capacity and SLO assumptions

These are versioned design targets, not permanent limits or real-aircraft claims.

| Requirement | Initial design target |
| --- | ---: |
| Logical fleet size | 10,000 engines |
| Sustained ingestion | 200 events per second |
| Short traffic burst | 1,000 events per second |
| Maximum event size | 32 KB |
| Ingestion acknowledgement p95 | Below 500 ms |
| Normal prediction freshness p95 | Below 10 seconds |
| Operational API p95 | Below 500 ms |
| Demonstration availability SLO | 99.5% |
| Accepted-event loss | Zero after durable acknowledgement |
| Recovery exercise | Process a 15-minute worker backlog within 30 minutes |

The load generator may synthesize additional equipment identities for service testing, but that data is isolated from model evaluation. Targets will change through a reviewed document version when client scale, telemetry cadence, cost or measured capacity supplies stronger evidence.

## 16. Failure behaviour

| Failure | Designed response |
| --- | --- |
| Edge connectivity loss | Persist locally and retry in order |
| Broker unavailable | Do not return `202`; edge retains and retries |
| Processing worker unavailable | Broker retains backlog; scale or recover consumers |
| PostgreSQL unavailable | Do not acknowledge; retry after recovery |
| Repeated poison event | Quarantine or dead-letter without blocking a partition forever |
| Model processing unavailable | Preserve event and retry; show prediction unavailable or stale explicitly |
| Notification provider unavailable | Keep authoritative alert and retry outbox |
| Dashboard unavailable | Continue ingestion and processing |
| Registry unavailable | Existing workers continue using their pinned loaded model |
| Raw archiver unavailable | Monitor archival lag while broker retention protects pending events |

## 17. Schema evolution

- APIs use major routes such as `/v1`.
- Events and stored artifacts include explicit schema versions.
- Additive optional fields may be backward compatible.
- Breaking changes receive a new contract and, when necessary, a new topic version.
- Producers and consumers are protected by contract tests.
- Historical records retain the schema version under which they were interpreted.

## 18. Review and completion criteria

Phase 3 is complete when:

- The logical, online, offline and deployment diagrams are reviewed.
- Component responsibilities and ownership are accepted.
- Event delivery, ordering, missingness and correction policies are accepted.
- Storage and operational data ownership are accepted.
- Training, inference, model-promotion and rollback boundaries are accepted.
- Security, observability, release and failure behaviours are accepted.
- Provisional capacity targets are accepted as changeable assumptions.

No target component in this architecture should be represented as already implemented. Implementation begins only after this document is approved and the next roadmap phase is selected.
