# Production Predictive Maintenance and Equipment Health Platform

A production-oriented machine learning platform that processes turbofan telemetry,
predicts Remaining Useful Life (RUL), preserves data and model lineage, and handles
invalid, delayed, duplicate, and retryable events explicitly.

The project uses the NASA C-MAPSS FD001 dataset to build and verify the complete
path from trusted historical data to quality-aware streaming inference.

> This is a learning and engineering project built with simulated engine data. It
> supports maintenance decisions; it is not a certified aviation system and does
> not autonomously control equipment.

## Current status

| Phase | Outcome | Status |
| --- | --- | --- |
| 1 | Production problem definition and system boundaries | Completed |
| 2 | Reproducible C-MAPSS data foundation and validation | Completed |
| 3 | Production architecture and reliability design | Completed |
| 4 | Durable streaming, event ordering, DLQ, and quality warnings | Completed |
| 5 | Versioned features, model comparison, inference, and prediction persistence | Completed |
| 6 | Model packaging, approval gates, experiment tracking, and registry lifecycle | Completed |
| 7 | Alerts, operational APIs, and fleet dashboard | Planned |
| 8 | Airflow orchestration | Planned |
| 9 | Monitoring and drift detection | Planned |
| 10 | CI/CD and cloud deployment | Planned |
| 11 | Load testing, security review, and final delivery | Planned |

Phase 6 completed with an approved LightGBM registry version, an audited `champion`
alias, checksum-verified local caching, and a real FastAPI → Redpanda → consumer →
PostgreSQL prediction using that champion.

## What is implemented

### Reproducible data foundation

- Downloads and checksum-verifies the NASA C-MAPSS archive.
- Safely extracts nested archives with path, size, CRC, and member allowlist checks.
- Parses FD001 train, test, and test-RUL files into typed tables.
- Applies blocking validation rules and explicit warning handling.
- Publishes trusted Parquet artifacts and lineage manifests atomically.
- Quarantines rejected records and failed validation reports.

### Durable telemetry ingestion

- FastAPI ingestion endpoint with a strict `telemetry-v1` contract.
- PostgreSQL or Redpanda-backed event sink.
- Manual Kafka offset acknowledgement after durable staging.
- Dead-letter publication for malformed or contract-invalid messages.
- PostgreSQL-backed `pending_event` storage for restart recovery.
- Event-time reordering with an allowed-lateness window.
- Configurable idle flushing and idempotent ordering-window warnings.

### Feature generation and model training

- Shared `features-v1` contract used by training and inference.
- Deterministic feature order: `cycle`, followed by `sensor_1` through `sensor_21`.
- Rejection of missing, null, NaN, and infinite feature values.
- RUL target: `maximum engine cycle - current cycle`.
- Reproducible 80/20 split by complete engine identity.
- Comparison of Linear Regression, Adaptive Lasso, XGBoost, and LightGBM.
- Regression evaluation with MAE, RMSE, and the asymmetric NASA score.
- Immutable publication of the selected experiment and its evaluation evidence.

### Quality-aware inference

- Model-independent predictor interface.
- Candidate, approved, and retired artifact lifecycle states.
- Approved-only model loading with feature-version compatibility checks.
- Explicit prediction states:
  - `AVAILABLE`: valid input and an RUL estimate was produced.
  - `DEGRADED`: an RUL estimate was produced with known non-fatal quality issues.
  - `WITHHELD`: unsafe input was intentionally not scored.
- Retryable model execution failures remain separate from data-quality decisions.
- Telemetry and prediction persistence in one PostgreSQL transaction.
- Idempotency through `(event_id, model_release)` uniqueness.
- Last-valid prediction lookup for future dashboard use.

### Governed model lifecycle

- Optional MLflow tracking with one comparison parent and four model child runs.
- Immutable candidate packages with dataset, feature, metric, and checksum lineage.
- Separate candidate registration, evidence review, human approval, and promotion.
- Provisional MAE, RMSE, NASA score, latency, load-time, size, and memory gates.
- Audited `champion` and `previous` aliases with compensating promotion behavior.
- Registry-backed inference that verifies approval and promotion decision records.
- Versioned local serving cache with checksum and registry-lineage verification.
- Fail-closed startup when the alias, approval, audit evidence, or package is invalid.

## Model results

The four candidates used the same engine-level validation split and feature
contract.

| Rank | Model | Validation MAE | Validation RMSE | NASA score |
| ---: | --- | ---: | ---: | ---: |
| 1 | LightGBM | 23.419 | 30.821 | 270,305.533 |
| 2 | XGBoost | 23.638 | 31.171 | 313,218.453 |
| 3 | Linear Regression | 24.454 | 31.231 | 302,723.135 |
| 4 | Adaptive Lasso | 24.478 | 31.254 | 302,582.511 |

After selection, LightGBM was retrained on all 100 FD001 training engines and
evaluated once on the official 100-engine test set:

| Metric | Result |
| --- | ---: |
| MAE | 19.447 cycles |
| RMSE | 26.820 cycles |
| NASA score | 8,065.366 |

LightGBM release `rul-lightgbm-v1` is MLflow model `cmapss-fd001-rul` version `1`.
It is provisionally approved and holds the `champion` alias. Its limits must be
reviewed when domain-owner requirements are available.

## Runtime contracts

### Telemetry input

Each event contains:

```text
event_id
engine_id
cycle
event_timestamp
schema_version
source_id
measurements
```

The measurements object may be accepted and stored with incomplete sensors, but
`features-v1` requires all 21 sensor values before producing an RUL estimate.
Missing required features result in a traceable `WITHHELD` prediction.

### Prediction output

```json
{
  "event_id": "engine-17-cycle-82",
  "engine_id": 17,
  "cycle": 82,
  "estimated_rul": 26.0,
  "status": "degraded",
  "data_quality_status": "degraded",
  "quality_flags": ["cycle_gap"],
  "model_release": "rul-model-v1",
  "feature_version": "features-v1",
  "generated_at": "2026-09-16T12:00:00Z"
}
```

`estimated_rul` is an estimate of remaining operating cycles, not a guaranteed
failure date.

## Reliability behavior

| Situation | System response |
| --- | --- |
| Invalid telemetry envelope | Publish to the dead-letter topic and retain error context |
| Out-of-order valid event | Store durably and reorder by engine cycle |
| Missing cycle | Continue with `DEGRADED` quality and record a warning |
| Missing or invalid required feature | Persist a `WITHHELD` prediction without calling the model |
| Model execution failure | Roll back the processing transaction and retry later |
| Exact event or prediction retry | Return `DUPLICATE` without creating another row |
| Same identity with changed content | Return `CONFLICT` and preserve the original row |
| New event cannot be scored | Allow explicit retrieval of the older last-valid prediction |

## Technology stack

| Area | Technology |
| --- | --- |
| Language and packaging | Python 3.12, `uv`, Hatchling |
| API and contracts | FastAPI, Pydantic |
| Data processing | Polars, Parquet |
| Machine learning | scikit-learn, XGBoost, LightGBM |
| Streaming | Redpanda / Kafka-compatible client |
| Operational storage | PostgreSQL, SQLAlchemy, Psycopg |
| Schema migrations | Alembic |
| Testing and quality | Pytest, Ruff, strict Mypy |
| Local infrastructure | Docker Compose |
| Experiment and model lifecycle | MLflow |

## Repository structure

```text
src/predictive_maintenance/
├── api/            # Telemetry HTTP API
├── core/           # Runtime settings
├── data/           # Download, extraction, validation, and publication
├── features/       # Versioned feature contracts and builder
├── inference/      # Artifact, registry, predictor, worker, and transaction processor
├── storage/        # Telemetry and prediction persistence
├── streaming/      # Producer, consumer, DLQ, and event-time reorder logic
└── training/       # Data split, models, evaluation, and comparison pipeline

infrastructure/postgres/migrations/  # Operational database schema
tests/unit/                         # Isolated behavior tests
tests/integration/                  # PostgreSQL and pipeline integration tests
docs/phases/                        # Completed phase decisions and evidence
```

Local datasets, generated model binaries, environment files, and personal learning
notes are intentionally excluded from Git.

## Local setup

### Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Docker with Docker Compose

### Install dependencies

```bash
uv sync
cp .env.example .env
```

### Start infrastructure and apply migrations

```bash
docker compose up -d postgres redpanda
.venv/bin/alembic upgrade head
.venv/bin/alembic current
```

Expected database revision:

```text
0004_prediction_lineage (head)
```

### Prepare FD001

```bash
.venv/bin/pm-platform data download
.venv/bin/pm-platform data prepare
```

The trusted publication is written under `data/interim/cmapss/v1/fd001/` and is
excluded from Git.

### Train and compare the four models

Run names are immutable, so use a new name for each experiment:

```bash
.venv/bin/pm-platform model train-compare \
  --run-name fd001-four-model-v2
```

Generated model artifacts are stored locally under `artifacts/training-runs/` and
are excluded from Git.

### Run the telemetry API

```bash
.venv/bin/uvicorn predictive_maintenance.api.app:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /ready`
- `POST /v1/telemetry`

### Run the stream consumer

The consumer can run without inference by setting:

```bash
export PM_INFERENCE_MODEL_SOURCE=none
.venv/bin/pm-platform stream worker
```

To serve the approved MLflow champion, start MLflow and configure:

```bash
./scripts/run_mlflow_server.sh

export PM_MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export PM_MLFLOW_REGISTERED_MODEL_NAME=cmapss-fd001-rul
export PM_INFERENCE_MODEL_SOURCE=mlflow_champion
export PM_MODEL_CACHE_ROOT=artifacts/model-cache
.venv/bin/pm-platform model verify-champion
.venv/bin/pm-platform stream worker
```

The worker resolves the current alias on startup and rejects unapproved versions,
invalid decision records, lineage conflicts, and modified cached artifacts.

## Verification

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests infrastructure/postgres/migrations scripts
.venv/bin/mypy src
```

Phase 6 completion evidence:

```text
180 tests passed, including PostgreSQL integration tests
Ruff passed
Strict Mypy passed for 44 source files
PostgreSQL migration: 0004_prediction_lineage (head)
MLflow champion: cmapss-fd001-rul version 1 / rul-lightgbm-v1
Real streaming path: FastAPI → Redpanda → champion inference → PostgreSQL
```

## Documentation

- [Phase 1: Production problem definition](docs/phases/01-production-problem-definition.md)
- [Phase 2: Reproducible data foundation](docs/phases/02-data-foundation.md)
- [Phase 3: Production architecture](docs/phases/03-production-architecture.md)
- [Phase 4: Streaming reliability and data quality](docs/phases/04-streaming-reliability-and-data-quality.md)
- [Phase 5: Feature generation, RUL modelling, and inference](docs/phases/05-feature-generation-model-training-and-inference.md)
- [Delivery roadmap](docs/project-roadmap.md)

The detailed Phase 6 record, daily checkpoints, presentation runbook, and LinkedIn
draft are maintained locally and excluded from Git as personal working notes.



## Next phase

Phase 7 will add alert creation, dashboard-ready operational APIs, and the fleet
dashboard experience on top of persisted telemetry, quality issues, and RUL
predictions.
