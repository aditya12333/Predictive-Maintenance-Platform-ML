# Production Predictive Maintenance and Equipment Health Platform

## Project status

| Step | Description | Status |
| --- | --- | --- |
| 1 | Project definition and scope | Completed |
| 2 | Dataset understanding and reproducible data foundation | Completed |
| 3 | Production architecture design | Completed |
| 4 | Repository and local infrastructure | Completed |
| 5 | Data ingestion, storage, and validation | Completed |
| 6 | Preprocessing and feature engineering | Completed |
| 7 | Model training and experiment tracking | In progress: reproducible training complete |
| 8 | Model registry and promotion | Not started |
| 9 | Streaming inference and API | In progress: inference worker complete |
| 10 | Monitoring and automated retraining | Not started |
| 11 | CI/CD, cloud deployment, and testing | Not started |

## 1. Project overview

Industrial equipment produces sensor readings during operation. Gradual degradation can eventually cause failure, but it is difficult to determine the correct time to perform maintenance.

This project will build a production ML system that:

> Processes incoming equipment sensor data, estimates the equipment's remaining useful life, and identifies whether maintenance may be required within the next 30 operating cycles.

The system will begin with the NASA C-MAPSS turbofan engine degradation dataset. Historical sensor records will later be replayed as live events to simulate a fleet of operating machines.

Dataset: [NASA C-MAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

## 2. Intended user

The intended user is a **maintenance planner or reliability engineer** responsible for a fleet of equipment.

The system should help the user answer:

- Which equipment is currently healthy?
- Which equipment is deteriorating?
- Which equipment requires attention soon?
- Approximately how long can each unit continue operating?
- Which sensor signals contributed to a warning?

## 3. System inputs

Each incoming record will describe one equipment unit at one operating cycle.

```text
equipment_id
cycle_number
operating_conditions
sensor_1
sensor_2
...
sensor_21
event_timestamp
```

In C-MAPSS:

- Each engine represents one equipment unit.
- Each row represents one operating cycle.
- The columns contain operating settings and sensor measurements.
- Training engines are observed until failure, allowing RUL targets to be calculated.

## 4. System outputs

For each equipment unit, the system will generate an output similar to:

```json
{
  "equipment_id": 47,
  "estimated_rul": 24,
  "failure_within_30_cycles": true,
  "failure_probability": 0.82,
  "health_status": "critical",
  "important_sensors": ["sensor_11", "sensor_4", "sensor_15"],
  "model_version": "rul-model-v3",
  "prediction_timestamp": "2026-08-30T10:30:00Z"
}
```

## 5. Machine learning tasks

### 5.1 Remaining useful life regression

The primary ML task is to predict the number of operating cycles remaining before failure.

```text
Predicted RUL = 24 cycles
```

### 5.2 Failure-risk classification

The second task is to estimate whether the equipment is likely to fail within the next 30 operating cycles.

```text
Failure within 30 cycles = Yes
Probability = 82%
```

RUL is the main prediction target. Failure classification converts that prediction into an operationally useful risk signal.

## 6. Decision policy

The system will recommend one of three actions:

| Status | Meaning | Suggested action |
| --- | --- | --- |
| Healthy | No immediate failure indication | Continue normal monitoring |
| Warning | Degradation is developing | Increase inspection frequency |
| Critical | High risk within 30 cycles | Schedule a maintenance inspection |

The system will only recommend an action. A qualified human remains responsible for the final maintenance decision.

## 7. Success criteria

The project will be evaluated across model performance and production-system reliability.

### 7.1 RUL prediction

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- NASA asymmetric scoring metric

The asymmetric metric is important because predicting failure too late is more costly than producing a slightly early warning.

### 7.2 Failure detection

- Recall
- Precision
- F1 score
- False-alert rate
- Missed-failure rate

Recall will be particularly important because a missed failure can be more costly than an early maintenance warning.

### 7.3 Production system

- Prediction latency
- Data freshness
- Service availability
- Invalid sensor-event rate
- Pipeline failure rate
- Model and data drift

Exact acceptance thresholds will be established after the baseline model and load tests are available.

## 8. Initial scope

The first complete version will:

- Start with the simpler C-MAPSS `FD001` subset.
- Train a simple baseline model.
- Train a tree-based production candidate such as LightGBM or XGBoost.
- Replay historical sensor records as streaming events.
- Generate RUL and 30-cycle failure-risk predictions.
- Provide predictions through an API.
- Store sensor data, features, predictions, and model metadata.
- Monitor data quality, drift, prediction quality, and service health.
- Implement the ten production ML components described in the reference MLOps article.

After the full pipeline works with `FD001`, it will be tested against the more difficult `FD002`, `FD003`, and `FD004` subsets.

## 9. Production ML components

| Component | Planned implementation |
| --- | --- |
| Data storage | Raw sensor events, processed data, features, predictions, and outcomes |
| Data processing | Streaming sensor replay and offline batch processing |
| Preprocessing and feature engineering | Validated, reusable transformations for training and inference |
| Training pipeline | Time-aware training, tuning, evaluation, and registration |
| Inference pipeline | Generate predictions whenever new sensor data arrives |
| Feature store | Historical training features and current serving features |
| Experiment tracking | Record datasets, parameters, metrics, and model artifacts |
| Model registry | Manage candidate, champion, and previous model versions |
| Monitoring | Track drift, prediction error, latency, freshness, and failures |
| CI/CD | Test, package, deploy, and safely roll back system changes |

## 10. Initial technical approach

The initial modelling strategy will include:

1. A simple age-based or linear baseline.
2. A LightGBM or XGBoost model using rolling sensor features.
3. An optional temporal challenger such as an LSTM or temporal convolutional network.

The challenger will only replace the simpler production model if it provides a meaningful improvement while meeting latency, stability, and maintainability requirements.

Potential features include:

- Current sensor measurements
- Rolling means
- Rolling standard deviations
- Sensor trends and slopes
- Differences from initial operating conditions
- Equipment age
- Operating-condition variables

## 11. Current scope exclusions

The initial version will not:

- Automatically shut down equipment.
- Replace the judgement of a maintenance professional.
- Claim that simulated turbofan data represents every industrial machine.
- Include an LLM or maintenance chatbot.
- Start with Kubernetes before the core system works.
- Use a complex deep-learning model before establishing a reliable baseline.

## 12. Development principle

The project will be developed one completed step at a time. Each phase will be understood, implemented, tested, and documented before work begins on the next phase.

The next phase is **Step 4: establish the local platform foundation and repository boundaries needed to implement the approved architecture safely**.
