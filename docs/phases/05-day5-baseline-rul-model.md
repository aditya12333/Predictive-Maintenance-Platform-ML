# Phase 5, Day 5: RUL Model Training and Comparison

**Status:** Completed<br>
**Completion date:** 2026-09-15<br>
**Warning-metric update:** 2026-09-16<br>
**Training run:** `fd001-four-model-28cycle-v1`<br>
**Selected model:** LightGBM<br>
**Lifecycle state:** Experiment winner; not approved for production serving

## Objective

Build a readable, reproducible training pipeline for NASA C-MAPSS FD001 that
compares four regression approaches on the same leakage-safe validation split,
selects the best model using a declared policy, evaluates the selected approach on
the official test set, and publishes an immutable training record.

The four candidates are:

1. Linear Regression.
2. Adaptive Lasso.
3. XGBoost.
4. LightGBM.

## Training data and target

The pipeline verifies `dataset-manifest.json` before reading the published Parquet
files. Each file must be declared in the manifest and match its recorded size and
SHA-256 checksum.

For every row in a complete training-engine history:

```text
RUL = final failure cycle for that engine - current cycle
```

The target is stored as `rul_cycles`. It is not capped. A model prediction is
postprocessed with `max(0, predicted_rul)` because a negative remaining life has no
operational meaning.

The feature contract is `features-v1`:

```text
cycle, sensor_1, sensor_2, ..., sensor_21
```

`engine_id` is used to create the split and is never passed to a model as a feature.
The pipeline rejects missing columns, null values, NaN values, and infinite feature
values before training.

## Leakage-safe model selection

The 100 run-to-failure training engines are split by unique engine identity with
`random_seed=42`:

| Partition | Engines | Rows |
| --- | ---: | ---: |
| Training | 80 | 16,561 |
| Validation | 20 | 4,070 |

All cycles from one engine remain in one partition. The four candidates receive the
same training rows and are evaluated on the same validation rows.

Candidates are ranked using this declared policy:

1. Lowest validation MAE.
2. Lowest NASA score when MAE is tied.
3. Lowest RMSE when MAE and NASA score are tied.
4. Model name as a deterministic final tie-break.

The official FD001 test data is not used during candidate selection.

## Candidate models

### Linear Regression

Uses a scikit-learn pipeline:

```text
StandardScaler -> LinearRegression
```

Scaling is fitted only on training-engine rows during candidate comparison.

### Adaptive Lasso

Uses a scikit-learn pipeline:

```text
StandardScaler -> AdaptiveLassoRegressor
```

The adaptive estimator first fits Ridge regression to obtain stable initial
coefficients. It calculates a feature penalty weight:

```text
weight_j = 1 / (abs(initial_coefficient_j) + epsilon)^gamma
```

It then fits Lasso on the weighted features. Features with small initial
coefficients receive stronger penalties.

### XGBoost

Uses `XGBRegressor` with a squared-error objective, histogram tree construction,
300 estimators, learning rate `0.05`, maximum depth `6`, row and column subsampling,
L2 regularization, and the shared random seed.

### LightGBM

Uses `LGBMRegressor` with a regression objective, 300 estimators, learning rate
`0.05`, 31 leaves, row and column subsampling, L2 regularization, and the shared
random seed.

Tree models receive the original feature values because decision trees do not
require standard feature scaling.

## Evaluation metrics

Every candidate is evaluated using the same shared function.

- **MAE:** average absolute RUL error in operating cycles.
- **RMSE:** gives more weight to large RUL errors.
- **NASA score:** applies an asymmetric exponential penalty that penalizes
  overestimated RUL more heavily because it can delay maintenance.
- **Precision, recall, and F1:** measure the warning produced by applying the
  28-cycle planning horizon to actual and predicted RUL.
- **False-alert rate:** `FP / (FP + TN)`, the share of non-imminent rows incorrectly
  warned.
- **Missed-failure rate:** `FN / (TP + FN)`, the share of imminent rows for which no
  warning was produced.

For `error = predicted RUL - actual RUL`:

```text
error < 0:  exp(-error / 13) - 1
error >= 0: exp( error / 10) - 1
```

Lower values are better for the three regression metrics. Higher precision, recall,
and F1 are better; lower false-alert and missed-failure rates are better. MAE, RMSE,
precision, recall, F1, and the confusion matrix use scikit-learn. The NASA C-MAPSS
score is implemented locally because it is not a scikit-learn metric.

The warning is derived from the regression prediction rather than trained as a
separate classifier:

```text
actual warning    = actual RUL <= 28 cycles
predicted warning = predicted RUL <= 28 cycles
```

C-MAPSS records operating cycles, not calendar dates. For this project's replay
simulation, one cycle represents one simulated day. Therefore 28 cycles represents
the requested 28-day planning horizon only within the simulation.

## Validation results

| Rank | Model | MAE | RMSE | NASA score |
| ---: | --- | ---: | ---: | ---: |
| 1 | LightGBM | 23.419 | 30.821 | 270,305.533 |
| 2 | XGBoost | 23.638 | 31.171 | 313,218.453 |
| 3 | Linear Regression | 24.454 | 31.231 | 302,723.135 |
| 4 | Adaptive Lasso | 24.478 | 31.254 | 302,582.511 |

The corresponding 28-cycle warning results are:

| Model | Precision | Recall | F1 | False-alert rate | Missed-failure rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 92.41% | 83.97% | 87.99% | 1.15% | 16.03% |
| XGBoost | 92.44% | 82.24% | 87.04% | 1.12% | 17.76% |
| Linear Regression | 91.09% | 63.45% | 74.80% | 1.03% | 36.55% |
| Adaptive Lasso | 90.86% | 63.45% | 74.72% | 1.06% | 36.55% |

LightGBM was selected because it achieved the lowest validation MAE. It also
achieved the lowest validation RMSE and NASA score.

Compared with Linear Regression, LightGBM produced:

- 4.23% lower validation MAE.
- 1.31% lower validation RMSE.
- 10.71% lower validation NASA score.

XGBoost achieved the second-lowest MAE but the highest NASA score. Its average error
was competitive, while its error direction included more costly RUL
overestimations. Adaptive Lasso performed almost identically to Linear Regression
with the current feature contract.

## Official test evaluation

After model selection, a fresh LightGBM instance was trained on all 100 complete
training-engine histories. It was then evaluated on the final observed row from
each of the 100 official FD001 test engines. Each row was paired with NASA's
`additional_rul` label.

| Metric | Result |
| --- | ---: |
| Test engines | 100 |
| MAE | 19.447 cycles |
| RMSE | 26.820 cycles |
| NASA score | 8,065.366 |
| Precision | 94.44% |
| Recall | 70.83% |
| F1 | 80.95% |
| False-alert rate | 1.32% |
| Missed-failure rate | 29.17% |

The official warning confusion matrix contains 17 true positives, 1 false positive,
75 true negatives, and 7 false negatives. The 70.83% recall does not meet the
original 90% imminent-failure recall target. LightGBM remains the best experiment
candidate among these four models, but this result blocks production approval until
the model or feature approach improves and is evaluated again.

Validation and official-test NASA scores are not directly comparable. Validation
contains 4,070 cycle-level predictions, while the official test contains one final
prediction per engine. The NASA score is a sum and therefore grows with the number
of evaluated predictions.

The official test result is final confirmation for this fixed training approach. It
must not be used to choose model parameters or select a different candidate.

## Published training run

The immutable run is stored under:

```text
artifacts/training-runs/fd001-four-model-28cycle-v1/
```

It contains:

- `evaluation.json`: data lineage, exact engine split, feature contract, selection
  policy, validation ranking, and official-test result.
- `selected-model.joblib`: LightGBM retrained on all FD001 training engines.

The selected model remains an experiment artifact. Production serving still
requires packaging against the model artifact contract and explicit human approval.

## Manual execution

Run a new comparison with an automatically generated timestamp:

```bash
.venv/bin/pm-platform model train-compare
```

Use a descriptive immutable name when required:

```bash
.venv/bin/pm-platform model train-compare \
  --run-name fd001-four-model-v2
```

View a saved report with:

```bash
.venv/bin/python -m json.tool \
  artifacts/training-runs/fd001-four-model-28cycle-v1/evaluation.json
```

Reusing an existing run name fails instead of overwriting its evidence.

## Implementation

- `src/predictive_maintenance/training/data.py`
- `src/predictive_maintenance/training/models.py`
- `src/predictive_maintenance/training/evaluate.py`
- `src/predictive_maintenance/training/train.py`
- `src/predictive_maintenance/cli.py`
- `tests/unit/test_training_data.py`
- `tests/unit/test_training_models.py`
- `tests/unit/test_training_evaluate.py`
- `tests/unit/test_training_run.py`

## Verification

- The real training command completed successfully against the trusted FD001
  publication.
- The four models were trained and compared on one shared engine-level split.
- Only the validation-selected model was evaluated on the official test set.
- The report and selected model were atomically published to an immutable run
  directory.
- Pytest passed all 143 tests, including the PostgreSQL integration tests.
- Ruff passed for the complete source and test suite.
- Strict Mypy passed for 37 source files.

## Phase completion

Day 6 subsequently implemented the inference worker, quality gates, explicit
`AVAILABLE`, `DEGRADED`, and `WITHHELD` outcomes, idempotent prediction persistence,
last-valid lookup, and atomic consumer integration. The experiment winner was not
promoted automatically.

The complete phase is documented in
[Phase 5: Feature Generation, RUL Modelling, and Inference](05-feature-generation-model-training-and-inference.md).
