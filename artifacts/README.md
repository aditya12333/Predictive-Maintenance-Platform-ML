# Model artifacts

Each generated release is stored in an immutable versioned directory containing
the fitted model, fitted scaler, artifact manifest, and evaluation report.

Generated binary releases are intentionally excluded from version control. Create
the Day 5 baseline with:

```bash
pm-platform model train-baseline
```

Run the four-model FD001 comparison with:

```bash
.venv/bin/pm-platform model train-compare
```

Each comparison creates an immutable timestamped directory under
`artifacts/training-runs/`. It contains `evaluation.json` with the validation
ranking and official-test result, plus `selected-model.joblib` with the fitted
winner. Supply `--run-name` when a descriptive experiment name is useful; every
run name must be unique.
