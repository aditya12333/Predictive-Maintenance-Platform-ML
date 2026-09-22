"""Reproducible training for the first FD001 RUL baseline."""

import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from predictive_maintenance.data.download import calculate_sha256
from predictive_maintenance.data.manifest import DatasetManifest
from predictive_maintenance.features.contracts import FEATURE_NAMES, FEATURE_VERSION
from predictive_maintenance.inference.artifacts import (
    ModelApprovalStatus,
    ModelArtifactManifest,
    RegressionMetrics,
    calculate_artifact_sha256,
)
from predictive_maintenance.training.data import (
    DEFAULT_RANDOM_SEED,
    RUL_TARGET,
    add_rul_target,
    load_fd001_training_data,
    split_by_engine,
)
from predictive_maintenance.training.evaluate import evaluate_predictions

TARGET_NAME = RUL_TARGET


class BaselineTrainingError(ValueError):
    """Raised when trusted data cannot produce a safe baseline artifact."""


@dataclass(frozen=True)
class BaselineEvaluation:
    """Metrics and lineage recorded for one baseline training run."""

    model_release: str
    dataset_version: str
    dataset_manifest_sha256: str
    feature_version: str
    feature_names: tuple[str, ...]
    target_definition: str
    prediction_postprocessing: str
    random_seed: int
    training_engine_ids: tuple[int, ...]
    validation_engine_ids: tuple[int, ...]
    training_rows: int
    validation_rows: int
    model_mae_cycles: float
    model_rmse_cycles: float
    age_only_mae_cycles: float
    improvement_over_age_only: float
    generated_at: str


def load_trusted_training_data(dataset_dir: Path) -> tuple[pl.DataFrame, str]:
    """Verify the trusted dataset manifest and load its training artifact."""

    manifest_path = dataset_dir / "dataset-manifest.json"
    return load_fd001_training_data(dataset_dir), calculate_sha256(manifest_path)


def build_rul_training_frame(telemetry: pl.DataFrame) -> pl.DataFrame:
    """Build point-in-time features and RUL labels from run-to-failure histories."""

    required_columns = {"engine_id", *FEATURE_NAMES}
    missing_columns = sorted(required_columns.difference(telemetry.columns))
    if missing_columns:
        raise BaselineTrainingError(
            f"training telemetry is missing columns: {', '.join(missing_columns)}"
        )
    if telemetry.is_empty():
        raise BaselineTrainingError("training telemetry is empty")

    model_columns = list(FEATURE_NAMES)
    if telemetry.select(model_columns).null_count().row(0) != (0,) * len(model_columns):
        raise BaselineTrainingError("features-v1 contains null training values")

    non_finite = telemetry.select(
        pl.any_horizontal(pl.col(model_columns).is_infinite() | pl.col(model_columns).is_nan())
        .any()
        .alias("has_non_finite")
    ).item()
    if non_finite:
        raise BaselineTrainingError("features-v1 contains non-finite training values")

    return add_rul_target(telemetry).select("engine_id", *FEATURE_NAMES, TARGET_NAME)


def train_baseline(
    *,
    dataset_dir: Path,
    artifact_root: Path,
    model_release: str = "rul-model-v1",
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> BaselineEvaluation:
    """Train, evaluate, and atomically publish a candidate Ridge RUL model."""

    telemetry, dataset_manifest_sha256 = load_trusted_training_data(dataset_dir)
    training_frame = build_rul_training_frame(telemetry)
    training_rows, validation_rows = split_by_engine(
        training_frame,
        random_seed=random_seed,
    )
    training_ids = tuple(sorted(training_rows["engine_id"].unique().to_list()))
    validation_ids = tuple(sorted(validation_rows["engine_id"].unique().to_list()))
    feature_names = list(FEATURE_NAMES)

    train_features = training_rows.select(feature_names).to_numpy()
    train_targets = training_rows[TARGET_NAME].to_numpy()
    validation_features = validation_rows.select(feature_names).to_numpy()
    validation_targets = validation_rows[TARGET_NAME].to_numpy()

    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features)
    model = Ridge(alpha=1.0)
    model.fit(scaled_train_features, train_targets)
    predictions = np.maximum(
        0.0,
        model.predict(scaler.transform(validation_features)),
    )

    age_only_model = LinearRegression()
    age_only_model.fit(training_rows.select("cycle").to_numpy(), train_targets)
    age_only_predictions = np.maximum(
        0.0,
        age_only_model.predict(validation_rows.select("cycle").to_numpy()),
    )

    model_evaluation = evaluate_predictions(
        model_name="ridge",
        actual_rul=validation_targets,
        predicted_rul=predictions,
    )
    model_mae = model_evaluation.mae_cycles
    model_rmse = model_evaluation.rmse_cycles
    age_only_mae = float(mean_absolute_error(validation_targets, age_only_predictions))
    improvement = (age_only_mae - model_mae) / age_only_mae
    if not all(math.isfinite(value) for value in (model_mae, model_rmse, improvement)):
        raise BaselineTrainingError("baseline evaluation produced non-finite metrics")

    manifest_path = dataset_dir / "dataset-manifest.json"
    dataset_manifest = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    generated_at = datetime.now(UTC)
    evaluation = BaselineEvaluation(
        model_release=model_release,
        dataset_version=(
            f"{dataset_manifest.dataset_id}:{dataset_manifest.dataset_version}:"
            f"{dataset_manifest.subset_id}"
        ),
        dataset_manifest_sha256=dataset_manifest_sha256,
        feature_version=FEATURE_VERSION,
        feature_names=FEATURE_NAMES,
        target_definition="max_cycle_for_engine - current_cycle",
        prediction_postprocessing="max(0, predicted_rul)",
        random_seed=random_seed,
        training_engine_ids=training_ids,
        validation_engine_ids=validation_ids,
        training_rows=training_rows.height,
        validation_rows=validation_rows.height,
        model_mae_cycles=model_mae,
        model_rmse_cycles=model_rmse,
        age_only_mae_cycles=age_only_mae,
        improvement_over_age_only=improvement,
        generated_at=generated_at.isoformat(),
    )

    artifact_root.mkdir(parents=True, exist_ok=True)
    release_dir = artifact_root / model_release
    if release_dir.exists():
        raise BaselineTrainingError(f"immutable model release already exists: {release_dir}")

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{model_release}-", dir=artifact_root))
    try:
        joblib.dump(model, staging_dir / "model.joblib")
        joblib.dump(scaler, staging_dir / "scaler.joblib")
        (staging_dir / "evaluation.json").write_text(
            json.dumps(asdict(evaluation), indent=2) + "\n", encoding="utf-8"
        )
        manifest = ModelArtifactManifest(
            model_release=model_release,
            model_name="ridge",
            source_training_run=model_release,
            created_at=generated_at,
            feature_version=FEATURE_VERSION,
            feature_names=FEATURE_NAMES,
            training_dataset_version=evaluation.dataset_version,
            dataset_manifest_sha256=evaluation.dataset_manifest_sha256,
            target_definition=evaluation.target_definition,
            prediction_postprocessing=evaluation.prediction_postprocessing,
            model_path=Path("model.joblib"),
            model_sha256=calculate_artifact_sha256(staging_dir / "model.joblib"),
            preprocessing_path=Path("scaler.joblib"),
            preprocessing_sha256=calculate_artifact_sha256(
                staging_dir / "scaler.joblib"
            ),
            evaluation_report_path=Path("evaluation.json"),
            evaluation_report_sha256=calculate_artifact_sha256(
                staging_dir / "evaluation.json"
            ),
            validation_metrics=RegressionMetrics(
                model_name=model_evaluation.model_name,
                sample_count=model_evaluation.sample_count,
                mae_cycles=model_evaluation.mae_cycles,
                rmse_cycles=model_evaluation.rmse_cycles,
                nasa_score=model_evaluation.nasa_score,
            ),
            official_test_metrics=None,
            status=ModelApprovalStatus.CANDIDATE,
        )
        (staging_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        staging_dir.rename(release_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return evaluation
