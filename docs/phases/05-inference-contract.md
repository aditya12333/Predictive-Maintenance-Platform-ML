# Phase 5: Inference Contract

**Status:** Implemented and verified<br>
**Completion date:** 2026-09-16<br>
**Final phase document:**
[Phase 5: Feature Generation, RUL Modelling, and Inference](05-feature-generation-model-training-and-inference.md)

## Purpose

Define the contract between validated telemetry, feature generation, the approved
RUL model, and downstream dashboard/alert consumers before implementing inference
code.

The inference worker consumes durable, ordered telemetry. The ingestion API does not
wait for feature generation or model execution.

## Processing flow

```text
telemetry_event
      -> data-quality gate
      -> feature builder
      -> approved model + scaler
      -> RUL prediction
      -> prediction record
      -> dashboard and alerts
```

## Inference input

The worker reads a processed telemetry event with:

- `event_id`
- `engine_id`
- `cycle`
- validated sensor measurements
- quality status and quality flags
- schema version

The event must already be persisted and ordered by the streaming layer.

## Feature contract

The model receives the fixed, versioned `features-v1` vector:

- Current sensor values.
- Engine cycle number.

Rolling windows, differences, and trend features require a future feature-contract
version; they are not silently added to `features-v1`.

The feature builder used during training and inference must be the same version. The
feature names, ordering, data types, and missing-value behavior are part of the
model artifact contract.

## Data-quality gate

The worker checks the input before calling the model. It must detect at least:

- Missing required sensors.
- Invalid numeric values or types.
- An incompatible schema version.
- A feature vector with the wrong shape or ordering.
- Severe corruption or unresolved data-quality conditions.

The worker continues processing the event, but it must not create an unsafe
prediction. The resulting prediction status is explicit:

- `AVAILABLE`: prediction generated from acceptable input.
- `DEGRADED`: prediction generated while known non-fatal quality issues exist.
- `WITHHELD`: prediction intentionally not generated because the input contract is
  not safe to use.

The system must not silently lower confidence unless that behavior has been tested
and calibrated for the specific quality condition.

## Model artifact contract

An approved model release includes:

- Model file.
- Fitted scaler or preprocessing artifact.
- Feature schema and feature version.
- Model release identifier.
- Training dataset version.
- Evaluation metrics.
- Approval metadata.

The inference worker reuses the fitted training scaler. It must never fit a new
scaler on live production data.

## Prediction output

Each prediction record must be traceable to its source event and model release. The
minimum output fields are:

```json
{
  "event_id": "engine-17-cycle-82",
  "engine_id": 17,
  "cycle": 82,
  "estimated_rul": 26.0,
  "status": "AVAILABLE",
  "data_quality_status": "DEGRADED",
  "quality_flags": ["missing_cycle:80"],
  "model_release": "rul-model-v1",
  "feature_version": "features-v1",
  "generated_at": "2026-01-01T00:01:00Z"
}
```

RUL is an estimate of remaining useful operating cycles, not a guaranteed failure
date. Risk level and data-quality status remain separate concepts.

## Last-valid prediction behavior

If a newer event is withheld, the dashboard may continue showing the last valid
prediction, but must label it clearly:

```text
Current prediction: unavailable
Last valid RUL: 26 cycles at cycle 81
Reason: required sensor missing at cycle 82
Data quality: degraded
```

The system must never present an old prediction as if it were current.

## Persistence and idempotency

Predictions are unique by event and model release. Retrying the same event must not
create duplicate prediction records. Every prediction must retain model and feature
lineage for investigation and auditability.

## Explicit non-goals for this phase

- Automatic model retraining.
- Airflow scheduling.
- Automatic model promotion.
- Cloud deployment.
- Full dashboard implementation.
- Uncertainty calibration before the model has been evaluated for it.

## Phase 5 completion criteria

Phase 5 is complete when the contract is implemented and tested for available,
degraded, withheld, duplicate, and last-valid prediction behavior.

These criteria were satisfied on 2026-09-16. The final implementation also verifies
prediction conflicts, transactional rollback, approved-model loading, and ordered
consumer integration against PostgreSQL.
