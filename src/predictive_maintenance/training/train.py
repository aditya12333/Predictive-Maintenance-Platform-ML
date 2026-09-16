"""Train, compare, and persist the FD001 RUL model candidates."""

import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from predictive_maintenance.data.download import calculate_sha256
from predictive_maintenance.data.manifest import DatasetManifest
from predictive_maintenance.features.contracts import FEATURE_NAMES, FEATURE_VERSION
from predictive_maintenance.training.data import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_VALIDATION_FRACTION,
    RUL_TARGET,
    TrainingDataError,
    add_rul_target,
    load_fd001_test_data,
    load_fd001_training_data,
    split_by_engine,
)
from predictive_maintenance.training.evaluate import (
    FAILURE_HORIZON_CYCLES,
    ModelEvaluation,
    evaluate_predictions,
)
from predictive_maintenance.training.models import RULRegressor, create_candidate_models

RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class TrainingRunError(ValueError):
    """Raised when the model-comparison run cannot be completed safely."""


@dataclass(frozen=True)
class TrainingRunReport:
    """Data lineage, validation ranking, and held-out test result for one run."""

    run_name: str
    generated_at: str
    dataset_version: str
    dataset_manifest_sha256: str
    feature_version: str
    feature_names: tuple[str, ...]
    target_definition: str
    prediction_postprocessing: str
    failure_horizon_cycles: int
    simulation_cycle_duration_days: int
    failure_horizon_days: int
    failure_warning_definition: str
    random_seed: int
    validation_fraction: float
    training_engine_ids: tuple[int, ...]
    validation_engine_ids: tuple[int, ...]
    training_rows: int
    validation_rows: int
    selection_policy: str
    validation_ranking: tuple[ModelEvaluation, ...]
    selected_model_name: str
    official_test_evaluation: ModelEvaluation


def train_and_compare(
    *,
    dataset_dir: Path,
    output_root: Path,
    run_name: str,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> TrainingRunReport:
    """Compare four models on validation data and test the selected model once."""

    _validate_run_name(run_name)
    run_directory = output_root / run_name
    if run_directory.exists():
        raise TrainingRunError(f"immutable training run already exists: {run_directory}")

    telemetry = load_fd001_training_data(dataset_dir)
    full_training_frame = _build_training_frame(telemetry)
    training_rows, validation_rows = split_by_engine(
        full_training_frame,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
    )

    feature_names = list(FEATURE_NAMES)
    training_features = training_rows.select(feature_names).to_numpy()
    training_targets = training_rows[RUL_TARGET].to_numpy()
    validation_features = validation_rows.select(feature_names).to_numpy()
    validation_targets = validation_rows[RUL_TARGET].to_numpy()

    validation_evaluations = _train_validation_candidates(
        training_features=training_features,
        training_targets=training_targets,
        validation_features=validation_features,
        validation_targets=validation_targets,
        random_seed=random_seed,
    )
    validation_ranking = tuple(sorted(validation_evaluations, key=_ranking_key))
    selected_model_name = validation_ranking[0].model_name

    selected_model = create_candidate_models(random_seed=random_seed)[selected_model_name]
    selected_model.fit(
        full_training_frame.select(feature_names).to_numpy(),
        full_training_frame[RUL_TARGET].to_numpy(),
    )
    official_test_frame = _build_official_test_frame(dataset_dir)
    official_predictions = selected_model.predict(
        official_test_frame.select(feature_names).to_numpy()
    )
    official_test_evaluation = evaluate_predictions(
        model_name=selected_model_name,
        actual_rul=official_test_frame[RUL_TARGET].to_numpy(),
        predicted_rul=official_predictions,
    )

    manifest_path = dataset_dir / "dataset-manifest.json"
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    report = TrainingRunReport(
        run_name=run_name,
        generated_at=datetime.now(UTC).isoformat(),
        dataset_version=f"{manifest.dataset_id}:{manifest.dataset_version}:{manifest.subset_id}",
        dataset_manifest_sha256=calculate_sha256(manifest_path),
        feature_version=FEATURE_VERSION,
        feature_names=FEATURE_NAMES,
        target_definition="max_cycle_for_engine - current_cycle",
        prediction_postprocessing="max(0, predicted_rul)",
        failure_horizon_cycles=FAILURE_HORIZON_CYCLES,
        simulation_cycle_duration_days=1,
        failure_horizon_days=FAILURE_HORIZON_CYCLES,
        failure_warning_definition="rul_cycles <= failure_horizon_cycles",
        random_seed=random_seed,
        validation_fraction=validation_fraction,
        training_engine_ids=tuple(sorted(training_rows["engine_id"].unique().to_list())),
        validation_engine_ids=tuple(sorted(validation_rows["engine_id"].unique().to_list())),
        training_rows=training_rows.height,
        validation_rows=validation_rows.height,
        selection_policy="lowest validation MAE; NASA score, RMSE, and name break ties",
        validation_ranking=validation_ranking,
        selected_model_name=selected_model_name,
        official_test_evaluation=official_test_evaluation,
    )
    _publish_run(
        output_root=output_root,
        run_name=run_name,
        report=report,
        selected_model=selected_model,
    )
    return report


def _train_validation_candidates(
    *,
    training_features: np.ndarray,
    training_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    random_seed: int,
) -> list[ModelEvaluation]:
    evaluations = []
    for model_name, model in create_candidate_models(random_seed=random_seed).items():
        try:
            model.fit(training_features, training_targets)
            predictions = model.predict(validation_features)
        except Exception as error:
            raise TrainingRunError(f"{model_name} training or prediction failed") from error
        evaluations.append(
            evaluate_predictions(
                model_name=model_name,
                actual_rul=validation_targets,
                predicted_rul=predictions,
            )
        )
    return evaluations


def _build_training_frame(telemetry: pl.DataFrame) -> pl.DataFrame:
    _validate_feature_frame(telemetry, description="training telemetry")
    return add_rul_target(telemetry).select("engine_id", *FEATURE_NAMES, RUL_TARGET)


def _build_official_test_frame(dataset_dir: Path) -> pl.DataFrame:
    telemetry, rul_labels = load_fd001_test_data(dataset_dir)
    _validate_feature_frame(telemetry, description="test telemetry")

    required_label_columns = {"engine_id", "observed_final_cycle", "additional_rul"}
    missing_label_columns = sorted(required_label_columns.difference(rul_labels.columns))
    if missing_label_columns:
        raise TrainingDataError(
            f"test RUL data is missing columns: {', '.join(missing_label_columns)}"
        )
    if rul_labels.is_empty():
        raise TrainingDataError("test RUL data is empty")
    if rul_labels["engine_id"].n_unique() != rul_labels.height:
        raise TrainingDataError("test RUL data must contain one row per engine")

    final_rows = (
        telemetry.sort(["engine_id", "cycle"]).group_by("engine_id", maintain_order=True).tail(1)
    )
    if set(final_rows["engine_id"].to_list()) != set(rul_labels["engine_id"].to_list()):
        raise TrainingDataError("test telemetry and RUL labels must contain the same engines")

    joined = final_rows.join(
        rul_labels.select("engine_id", "observed_final_cycle", "additional_rul"),
        on="engine_id",
        how="inner",
    )
    if joined.filter(pl.col("cycle") != pl.col("observed_final_cycle")).height:
        raise TrainingDataError("test RUL labels do not match final observed cycles")
    if joined.filter(pl.col("additional_rul") < 0).height:
        raise TrainingDataError("test RUL labels must be non-negative")

    return joined.select("engine_id", *FEATURE_NAMES, pl.col("additional_rul").alias(RUL_TARGET))


def _validate_feature_frame(frame: pl.DataFrame, *, description: str) -> None:
    if frame.is_empty():
        raise TrainingDataError(f"{description} is empty")

    required_columns = {"engine_id", *FEATURE_NAMES}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise TrainingDataError(f"{description} is missing columns: {', '.join(missing_columns)}")

    feature_names = list(FEATURE_NAMES)
    if frame.select(feature_names).null_count().row(0) != (0,) * len(feature_names):
        raise TrainingDataError(f"{description} contains null feature values")
    has_non_finite = frame.select(
        pl.any_horizontal(pl.col(feature_names).is_infinite() | pl.col(feature_names).is_nan())
        .any()
        .alias("has_non_finite")
    ).item()
    if has_non_finite:
        raise TrainingDataError(f"{description} contains non-finite feature values")


def _ranking_key(evaluation: ModelEvaluation) -> tuple[float, float, float, str]:
    return (
        evaluation.mae_cycles,
        evaluation.nasa_score,
        evaluation.rmse_cycles,
        evaluation.model_name,
    )


def _validate_run_name(run_name: str) -> None:
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise TrainingRunError(
            "run_name must start with a letter or number and contain only letters, "
            "numbers, dots, underscores, or hyphens"
        )


def _publish_run(
    *,
    output_root: Path,
    run_name: str,
    report: TrainingRunReport,
    selected_model: RULRegressor,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    run_directory = output_root / run_name
    if run_directory.exists():
        raise TrainingRunError(f"immutable training run already exists: {run_directory}")

    staging_directory = Path(tempfile.mkdtemp(prefix=f".{run_name}-", dir=output_root))
    try:
        joblib.dump(selected_model, staging_directory / "selected-model.joblib")
        (staging_directory / "evaluation.json").write_text(
            json.dumps(asdict(report), indent=2) + "\n",
            encoding="utf-8",
        )
        staging_directory.rename(run_directory)
    except Exception:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise
