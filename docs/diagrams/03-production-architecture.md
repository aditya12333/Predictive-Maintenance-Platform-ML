# Phase 3: Target Production Architecture Diagrams

**Status:** Approved<br>
**Scope:** Target design only. These diagrams include planned components that have not yet been implemented.<br>
**Current implementation:** The working code currently ends at the validated FD001 interim dataset documented in [Implemented Data Flow](implemented-data-flow.md).

## 1. Logical platform architecture

```mermaid
flowchart LR
    subgraph EDGE[Equipment and edge]
        EQUIPMENT[Engine or telemetry simulator]
        BUFFER[Persistent connectivity buffer]
        EQUIPMENT --> BUFFER
    end

    subgraph ONLINE[Online telemetry and decisions]
        INGEST[FastAPI ingestion API]
        RAW_TOPIC[(telemetry.raw.v1)]
        ARCHIVER[Raw-event archiver]
        WORKER[Modular telemetry-processing worker]
        QUARANTINE[(Quarantine path)]
        DB[(PostgreSQL)]
        OUTBOX[Notification worker]

        INGEST -->|Durable publish| RAW_TOPIC
        RAW_TOPIC --> ARCHIVER
        RAW_TOPIC --> WORKER
        WORKER -->|Invalid or conflicting| QUARANTINE
        WORKER -->|Predictions, alerts and outbox| DB
        DB --> OUTBOX
    end

    subgraph STORAGE[Immutable storage]
        OBJECTS[(S3-compatible object storage)]
        ARCHIVER -->|Batched raw events| OBJECTS
        QUARANTINE -->|Reports and rejected data| OBJECTS
    end

    subgraph CLIENT[Client applications]
        DASHBOARD[React and TypeScript fleet dashboard]
        OPS_API[FastAPI operational API]
        DASHBOARD --> OPS_API
        OPS_API --> DB
    end

    subgraph ML[Offline ML lifecycle]
        AIRFLOW[Airflow training DAG]
        TRAIN[Versioned Python training pipeline]
        MLFLOW[MLflow tracking and registry]
        APPROVAL[Human model approval]

        OBJECTS -->|Trusted versioned data| AIRFLOW
        AIRFLOW --> TRAIN
        TRAIN -->|Runs and artifacts| MLFLOW
        MLFLOW --> APPROVAL
        APPROVAL -->|Champion bundle| MLFLOW
        MLFLOW -.->|Pinned model bundle| WORKER
    end

    subgraph OBS[Cross-cutting controls]
        OTEL[OpenTelemetry]
        MONITORING[Metrics, logs, traces and alerts]
        OTEL --> MONITORING
    end

    BUFFER -->|HTTPS telemetry| INGEST
    INGEST -->|202 only after broker acknowledgement| BUFFER
    WORKER -.-> OTEL
    INGEST -.-> OTEL
    OPS_API -.-> OTEL
    AIRFLOW -.-> OTEL
```

## 2. Online event-processing and failure flow

```mermaid
flowchart TD
    EVENT[Telemetry event arrives] --> ENVELOPE{Authentication, size,<br/>JSON and envelope valid?}
    ENVELOPE -- No --> REJECT_REQUEST[Return explicit 4xx response<br/>Record security or contract issue]
    ENVELOPE -- Yes --> BROKER{Broker durably accepted event?}
    BROKER -- No --> RETRY_RESPONSE[Return retryable failure<br/>Edge buffer retains event]
    BROKER -- Yes --> ACCEPTED[Return 202 Accepted<br/>with event_id]

    ACCEPTED --> CONSUME[Worker consumes raw event]
    CONSUME --> DUPLICATE{Event identity already processed?}
    DUPLICATE -- Exact retry --> SAFE_NOOP[No duplicate business result]
    DUPLICATE -- Conflicting payload --> CONFLICT[Quarantine and raise<br/>data-quality concern]
    DUPLICATE -- New event --> REORDER[Per-engine reorder buffer]

    REORDER --> ON_TIME{Expected cycle arrives<br/>within allowed lateness?}
    ON_TIME -- Yes --> DEEP_VALIDATION[Deep telemetry validation]
    ON_TIME -- No --> GAP[Record cycle gap<br/>Set quality DEGRADED]
    GAP --> DEEP_VALIDATION

    DEEP_VALIDATION --> USABLE{Model input contract satisfied?}
    USABLE -- No --> WITHHELD[Persist prediction status WITHHELD<br/>Raise data-quality alert]
    USABLE -- Yes --> SCORE[Shared feature package<br/>and pinned champion model]
    SCORE --> TRANSACTION[Database transaction:<br/>prediction, alert and outbox]
    TRANSACTION --> COMMIT{Transaction committed?}
    COMMIT -- No --> NO_ACK[Do not acknowledge broker<br/>Retry safely later]
    COMMIT -- Yes --> ACK[Acknowledge broker event]
    ACK --> NOTIFY[Notification worker processes outbox]
    NOTIFY --> SENT{Delivery successful?}
    SENT -- No --> RETRY_NOTIFY[Retain pending record<br/>Retry with backoff]
    SENT -- Yes --> COMPLETE[Mark notification delivered]
```

## 3. Offline training, approval and release flow

```mermaid
flowchart TD
    TRUSTED[Versioned trusted telemetry<br/>and dataset manifest] --> VERIFY[Verify source, schema<br/>and artifact digests]
    VERIFY --> TARGETS[Construct leakage-safe RUL<br/>and risk targets]
    TARGETS --> FEATURES[Shared versioned feature package]
    FEATURES --> SPLIT[Split by complete engine identities]
    SPLIT --> FIT[Fit preprocessing on training split only]
    FIT --> BASELINE[Train simple baseline]
    FIT --> CANDIDATE[Train candidate model]
    BASELINE --> TRACK[MLflow experiment tracking]
    CANDIDATE --> TRACK
    TRACK --> GATES{All promotion gates passed?}
    GATES -- No --> RETAIN[Retain candidate for analysis<br/>Keep current champion]
    GATES -- Yes --> REVIEW[Human approval with audit reason]
    REVIEW -- Rejected --> RETAIN
    REVIEW -- Approved --> REGISTRY[Assign immutable champion version]
    REGISTRY --> RELEASE[Release manifest pins:<br/>image, model, features and config]
    RELEASE --> STAGING[Deploy exact artifacts to staging]
    STAGING --> VERIFY_RELEASE{Smoke, contract and<br/>performance checks pass?}
    VERIFY_RELEASE -- No --> ROLLBACK[Restore exact previous release]
    VERIFY_RELEASE -- Yes --> PROD_APPROVAL[Production approval]
    PROD_APPROVAL --> PRODUCTION[Promote exact tested artifacts]
```

## 4. AWS reference deployment

```mermaid
flowchart TB
    subgraph PUBLIC[Public entry layer]
        USERS[Client users]
        DEVICES[Edge gateways or simulator]
        CDN[Static dashboard hosting and CDN]
        ALB[HTTPS load balancer]
        USERS --> CDN
        USERS --> ALB
        DEVICES --> ALB
    end

    subgraph PRIVATE[Private application network]
        INGEST_ECS[ECS ingestion API replicas]
        OPS_ECS[ECS operational API replicas]
        PROCESS_ECS[ECS processing workers]
        NOTIFY_ECS[ECS notification workers]
        TRAINING[ECS batch training tasks]
        BROKER_AWS[(Managed Kafka-compatible broker)]
        RDS[(RDS PostgreSQL)]
        MLFLOW_AWS[Private MLflow service]

        ALB --> INGEST_ECS
        ALB --> OPS_ECS
        INGEST_ECS --> BROKER_AWS
        BROKER_AWS --> PROCESS_ECS
        PROCESS_ECS --> RDS
        OPS_ECS --> RDS
        RDS --> NOTIFY_ECS
        TRAINING --> MLFLOW_AWS
        MLFLOW_AWS -.-> PROCESS_ECS
    end

    subgraph MANAGED[Managed platform services]
        S3[(S3 object storage)]
        ECR[ECR immutable images]
        SECRETS[Secrets Manager and workload identity]
        OBSERVABILITY[Cloud monitoring and OpenTelemetry]
        SCHEDULER[Scheduled workflow trigger]

        BROKER_AWS -->|Archival consumer| S3
        S3 --> TRAINING
        MLFLOW_AWS --> S3
        ECR --> INGEST_ECS
        ECR --> OPS_ECS
        ECR --> PROCESS_ECS
        ECR --> NOTIFY_ECS
        ECR --> TRAINING
        SCHEDULER --> TRAINING
    end

    INGEST_ECS -.-> SECRETS
    OPS_ECS -.-> SECRETS
    PROCESS_ECS -.-> SECRETS
    INGEST_ECS -.-> OBSERVABILITY
    OPS_ECS -.-> OBSERVABILITY
    PROCESS_ECS -.-> OBSERVABILITY
```

The cloud names describe the reference deployment, not hard dependencies in business logic. Contracts and application interfaces remain portable.
