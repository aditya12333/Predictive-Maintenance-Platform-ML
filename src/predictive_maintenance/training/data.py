"""Load trusted datasets for model training."""

from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split

from predictive_maintenance.data.download import calculate_sha256
from predictive_maintenance.data.manifest import DatasetManifest

RUL_TARGET = "rul_cycles"
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_RANDOM_SEED = 42


class TrainingDataError(ValueError):
    """Raised when a trusted dataset cannot be used for training."""


def load_fd001_training_data(dataset_dir: Path) -> pl.DataFrame:
    """Verify and load the trusted FD001 training table."""

    manifest = _load_fd001_manifest(dataset_dir)
    return _load_verified_parquet(
        dataset_dir,
        manifest=manifest,
        relative_path="train.parquet",
        description="training data",
    )


def load_fd001_test_data(dataset_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Verify and load the official FD001 test telemetry and RUL labels."""

    manifest = _load_fd001_manifest(dataset_dir)
    telemetry = _load_verified_parquet(
        dataset_dir,
        manifest=manifest,
        relative_path="test.parquet",
        description="test data",
    )
    rul_labels = _load_verified_parquet(
        dataset_dir,
        manifest=manifest,
        relative_path="test-rul.parquet",
        description="test RUL data",
    )
    return telemetry, rul_labels


def _load_fd001_manifest(dataset_dir: Path) -> DatasetManifest:
    manifest_path = dataset_dir / "dataset-manifest.json"
    try:
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TrainingDataError("dataset manifest is missing or invalid") from error

    if manifest.subset_id != "FD001":
        raise TrainingDataError(f"expected FD001, received {manifest.subset_id}")
    return manifest


def _load_verified_parquet(
    dataset_dir: Path,
    *,
    manifest: DatasetManifest,
    relative_path: str,
    description: str,
) -> pl.DataFrame:
    artifact = next(
        (artifact for artifact in manifest.files if artifact.relative_path == relative_path),
        None,
    )
    if artifact is None:
        raise TrainingDataError(f"dataset manifest does not declare {relative_path}")

    artifact_path = dataset_dir / artifact.relative_path
    if not artifact_path.is_file():
        raise TrainingDataError(f"{description} is missing: {artifact_path}")
    if artifact_path.stat().st_size != artifact.size_bytes:
        raise TrainingDataError(f"{description} size does not match the manifest")
    if calculate_sha256(artifact_path) != artifact.sha256:
        raise TrainingDataError(f"{description} checksum does not match the manifest")

    try:
        return pl.read_parquet(artifact_path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise TrainingDataError(f"could not read {description}: {artifact_path}") from error


def add_rul_target(telemetry: pl.DataFrame) -> pl.DataFrame:
    """Add the true RUL label to complete run-to-failure engine histories."""

    if telemetry.is_empty():
        raise TrainingDataError("training telemetry is empty")

    required_columns = {"engine_id", "cycle"}
    missing_columns = sorted(required_columns.difference(telemetry.columns))
    if missing_columns:
        raise TrainingDataError(
            f"training telemetry is missing columns: {', '.join(missing_columns)}"
        )

    return telemetry.with_columns(
        (pl.col("cycle").max().over("engine_id") - pl.col("cycle")).alias(RUL_TARGET)
    )


def split_by_engine(
    data: pl.DataFrame,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split complete engine histories into reproducible training and validation sets."""

    if "engine_id" not in data.columns:
        raise TrainingDataError("training data is missing column: engine_id")
    if not 0 < validation_fraction < 1:
        raise TrainingDataError("validation_fraction must be between zero and one")

    engine_ids = sorted(data["engine_id"].unique().to_list())
    if len(engine_ids) < 2:
        raise TrainingDataError("at least two engines are required for a split")

    training_ids, validation_ids = train_test_split(
        engine_ids,
        test_size=validation_fraction,
        random_state=random_seed,
        shuffle=True,
    )
    training_data = data.filter(pl.col("engine_id").is_in(training_ids))
    validation_data = data.filter(pl.col("engine_id").is_in(validation_ids))

    if training_data.is_empty() or validation_data.is_empty():
        raise TrainingDataError("training and validation sets must both contain rows")
    if set(training_ids).intersection(validation_ids):
        raise TrainingDataError("an engine cannot appear in both training and validation")

    return training_data, validation_data
