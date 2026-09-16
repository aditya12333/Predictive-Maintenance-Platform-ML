# Phase 1: Production Problem Definition

**Status:** Approved<br>
**Project:** AeroReliability Predictive Maintenance Platform<br>
**Client:** AeroReliability Operations, a fictional organisation responsible for monitoring and maintaining a fleet of turbofan engines.

## 1. Problem statement

AeroReliability Operations relies on scheduled inspections and reactive maintenance. Its reliability team cannot consistently identify which engines are degrading, how urgent an emerging problem is, or where limited maintenance resources should be allocated first.

Unexpected failures can cause expensive repairs, unplanned downtime, and operational risk. Excessively early maintenance also wastes labour, parts, and equipment availability.

The client needs a fleet-level decision-support platform that converts engine telemetry into timely, traceable, and actionable maintenance insights.

## 2. Product objective

Build a cloud-based predictive-maintenance platform that:

- Detects abnormal engine behaviour.
- Estimates remaining useful life in operating cycles.
- Estimates failure risk within configurable cycle horizons.
- Prioritises engines requiring inspection.
- Explains the evidence behind warnings.
- Identifies unreliable or stale sensor data.
- Records alert decisions and maintenance outcomes.
- Monitors data, models, services, and business effectiveness.

The system will support human decisions. It will not autonomously control equipment.

## 3. Users

### Primary user: fleet reliability manager

- Monitors fleet health.
- Reviews prioritised alerts.
- Investigates degradation evidence.
- Assigns inspections.
- Acknowledges, dismisses, or escalates alerts.
- Evaluates fleet-level maintenance risk.

### Secondary user: maintenance engineer

- Reviews equipment history.
- Performs inspections.
- Records findings and maintenance actions.
- Confirms whether an alert identified a real problem.

### Supporting users

- Platform administrator
- ML engineer
- Model approver
- Technical director or fleet owner

## 4. Inputs

The platform will consume:

- Engine identifier
- Operating cycle
- Event timestamp
- Ingestion timestamp
- Operating-condition settings
- Sensor measurements
- Equipment metadata
- Data-source and schema versions
- Inspection results
- Maintenance outcomes
- Failure outcomes when available

NASA C-MAPSS will provide the initial historical data. Its engine histories will also be replayed as timestamped events to simulate a live fleet.

True RUL and future failure information are training labels and must never appear in production model inputs.

## 5. Outputs

For each engine, the system will provide:

- Current health status
- Estimated RUL in operating cycles
- Failure probability within configurable horizons
- Anomaly severity
- Alert severity and inspection urgency
- Important contributing sensor behaviour
- Data-quality and feature-freshness status
- Prediction, model, and feature versions
- Prediction history
- Active and historical alerts
- Maintenance and inspection history

The fleet view will rank engines by urgency so that the reliability manager can decide which assets to investigate first.

## 6. Business priorities

In descending order:

1. Avoid missed imminent failures.
2. Provide sufficient inspection lead time.
3. Keep false alerts operationally manageable.
4. Avoid unnecessary early maintenance.
5. Reduce unplanned downtime and maintenance cost.
6. Improve fleet availability and user confidence.

Late warnings will initially be penalised more heavily than equivalently early warnings.

## 7. System boundary

### Inside the initial project scope

- Reproducible C-MAPSS acquisition and verification
- Immutable raw-data storage
- Parsing, validation, and quarantine
- Versioned preprocessing and feature engineering
- Leakage-safe training datasets
- Automated training and evaluation
- Experiment tracking and model registry
- Batch fleet inference
- Simulated streaming inference
- Prediction and fleet-health APIs
- Alert creation, deduplication, and lifecycle
- Maintenance-feedback workflow
- Data, model, service, and business monitoring
- Authentication, authorisation, and audit history
- Containerisation, CI/CD, and cloud deployment
- Backup, recovery, and model rollback procedures

### Outside the initial system boundary

- Physical sensors and IoT hardware
- Aircraft or engine control
- Automatic shutdown or maintenance authorisation
- Physical inspection and repair
- Certified aviation-safety decisions
- Full enterprise maintenance-management replacement
- Real aircraft, industrial, or maritime deployment
- Insurance underwriting
- Hardware digital twins

## 8. Success metrics

These are provisional targets. Baseline experiments may revise them, but any revision must be documented.

### Model quality

- Evaluation uses completely held-out engines.
- No future data or target leakage.
- RUL MAE improves by at least 15% over an age-only baseline.
- The NASA asymmetric score improves over all simpler baselines.
- Recall for failures occurring within the 28-cycle planning horizon reaches at
  least 90%. In the project simulation, one operating cycle represents one
  simulated day; C-MAPSS itself does not contain calendar-day durations.
- Late-warning performance is reported separately from early-warning error.
- Prediction stability is measured across consecutive cycles.

### Operational usefulness

- High-risk engines are presented with actionable evidence.
- Confirmed high-risk cases receive useful warning lead time.
- Repeated predictions for one degradation episode do not create duplicate alerts.
- False alerts are measured per engine and operating period.
- Maintenance outcomes can be linked to the predictions and alerts that preceded them.

### Data reliability

- Every source archive is checksum-verified.
- Every accepted record satisfies a versioned schema.
- Invalid records are quarantined with an explicit reason.
- Duplicate and out-of-order events are handled safely.
- Stale or invalid data is never presented as healthy equipment.

### Service reliability

- Every prediction is traceable to its data, features, model, and code version.
- Batch scoring can process the demonstration fleet within its scheduled window.
- Initial API target: p95 response latency below 500 ms.
- Initial availability target: 99.5% for the demonstration deployment.
- Failed or degraded dependencies produce explicit status rather than silent errors.
- The previous champion model remains available for rollback.

## 9. Explicit non-goals

The first release will not:

- Claim that C-MAPSS validates performance on real engines.
- Provide autonomous or safety-critical control.
- Generalise to every type of industrial equipment.
- Optimise only for leaderboard performance.
- Begin with deep learning unless simpler models are insufficient.
- Introduce a feature store before online requirements justify it.
- Introduce Kubernetes before the managed-container architecture requires it.
- Include an LLM maintenance assistant.
- Automatically retrain and deploy models solely because drift was detected.
- Hide uncertainty, invalid data, or unsupported operating conditions.
- Silently modify historical predictions, alerts, or maintenance records.

## 10. Key assumptions and constraints

- AeroReliability Operations is fictional.
- C-MAPSS is simulated turbofan data.
- An operating cycle cannot automatically be translated into hours or days.
- Maintenance outcomes will initially be simulated or manually entered.
- Human users retain authority over inspections and maintenance.
- Batch inference will be completed before real-time inference.
- FD001 will be completed before expanding to more complex C-MAPSS subsets.
- The architecture will be client-ready, but the demonstration will not be represented as certified for real-world aviation use.
