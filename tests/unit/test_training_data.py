"""Tests for trusted training-data loading."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from predictive_maintenance.data.download import calculate_sha256
from predictive_maintenance.data.manifest import DatasetManifest, FileArtifact
from predictive_maintenance.training.data import (
    TrainingDataError,
    add_rul_target,
    load_fd001_training_data,
    split_by_engine,
)


def publish_dataset(dataset_dir: Path, *, subset_id: str = "FD001") -> Path:
    """Create a small manifest-backed dataset for one loader test."""

    dataset_dir.mkdir()
    train_path = dataset_dir / "train.parquet"
    pl.DataFrame({"engine_id": [1], "cycle": [1]}).write_parquet(train_path)
    manifest = DatasetManifest(
        dataset_id="nasa-cmapss-classic",
        dataset_version="v1",
        subset_id=subset_id,
        telemetry_schema_version="telemetry-v1",
        test_rul_schema_version="test-rul-v1",
        source_manifest_sha256="0" * 64,
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        validation_status="PASSED",
        files=(
            FileArtifact(
                relative_path="train.parquet",
                size_bytes=train_path.stat().st_size,
                sha256=calculate_sha256(train_path),
            ),
        ),
    )
    (dataset_dir / "dataset-manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return train_path


def test_loads_manifest_verified_fd001_training_data(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "fd001"
    publish_dataset(dataset_dir)

    result = load_fd001_training_data(dataset_dir)

    assert result.to_dict(as_series=False) == {"engine_id": [1], "cycle": [1]}


def test_rejects_training_data_changed_after_publication(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "fd001"
    train_path = publish_dataset(dataset_dir)
    train_path.write_bytes(b"changed")

    with pytest.raises(TrainingDataError, match="size does not match"):
        load_fd001_training_data(dataset_dir)


def test_rejects_the_wrong_cmapss_subset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "fd002"
    publish_dataset(dataset_dir, subset_id="FD002")

    with pytest.raises(TrainingDataError, match="expected FD001"):
        load_fd001_training_data(dataset_dir)


def test_adds_rul_target_independently_for_each_engine() -> None:
    telemetry = pl.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
        }
    )

    result = add_rul_target(telemetry)

    assert result["rul_cycles"].to_list() == [2, 1, 0, 1, 0]


def test_rul_target_rejects_empty_telemetry() -> None:
    with pytest.raises(TrainingDataError, match="empty"):
        add_rul_target(pl.DataFrame())


def test_rul_target_requires_engine_and_cycle_columns() -> None:
    with pytest.raises(TrainingDataError, match="cycle"):
        add_rul_target(pl.DataFrame({"engine_id": [1]}))


def test_split_by_engine_keeps_complete_histories_separate() -> None:
    data = pl.DataFrame(
        {
            "engine_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "cycle": [1, 2] * 5,
        }
    )

    first_training, first_validation = split_by_engine(data, random_seed=42)
    second_training, second_validation = split_by_engine(data, random_seed=42)

    training_ids = set(first_training["engine_id"].unique().to_list())
    validation_ids = set(first_validation["engine_id"].unique().to_list())
    assert training_ids.isdisjoint(validation_ids)
    assert training_ids | validation_ids == {1, 2, 3, 4, 5}
    assert first_training.equals(second_training)
    assert first_validation.equals(second_validation)


@pytest.mark.parametrize("invalid_fraction", [0.0, 1.0, -0.1, 1.1])
def test_split_by_engine_rejects_invalid_fraction(invalid_fraction: float) -> None:
    data = pl.DataFrame({"engine_id": [1, 2], "cycle": [1, 1]})

    with pytest.raises(TrainingDataError, match="between zero and one"):
        split_by_engine(data, validation_fraction=invalid_fraction)


def test_split_by_engine_requires_multiple_engines() -> None:
    data = pl.DataFrame({"engine_id": [1, 1], "cycle": [1, 2]})

    with pytest.raises(TrainingDataError, match="at least two engines"):
        split_by_engine(data)
