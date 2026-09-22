"""Tests for leakage-safe FD001 baseline preparation and training."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from predictive_maintenance.data.download import calculate_sha256
from predictive_maintenance.data.manifest import DatasetManifest, FileArtifact
from predictive_maintenance.features.builder import FeatureBuilder
from predictive_maintenance.features.contracts import FEATURE_NAMES
from predictive_maintenance.inference.artifacts import (
    ModelApprovalStatus,
    ModelArtifactManifest,
)
from predictive_maintenance.inference.sklearn_predictor import SklearnRULPredictor
from predictive_maintenance.training.baseline import (
    BaselineTrainingError,
    build_rul_training_frame,
    train_baseline,
)


def telemetry_frame() -> pl.DataFrame:
    rows = []
    for engine_id, final_cycle in ((1, 3), (2, 2)):
        for cycle in range(1, final_cycle + 1):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    **{f"sensor_{index}": float(index + cycle) for index in range(1, 22)},
                }
            )
    return pl.DataFrame(rows)


def test_rul_labels_use_only_each_engines_run_to_failure_history() -> None:
    frame = build_rul_training_frame(telemetry_frame())

    engine_one = frame.filter(pl.col("engine_id") == 1)
    engine_two = frame.filter(pl.col("engine_id") == 2)

    assert engine_one["rul_cycles"].to_list() == [2, 1, 0]
    assert engine_two["rul_cycles"].to_list() == [1, 0]
    assert frame.columns == ["engine_id", *FEATURE_NAMES, "rul_cycles"]


def test_training_frame_rejects_missing_model_feature() -> None:
    with pytest.raises(BaselineTrainingError, match="sensor_21"):
        build_rul_training_frame(telemetry_frame().drop("sensor_21"))


def publish_test_dataset(dataset_dir: Path) -> None:
    rows = []
    for engine_id in range(1, 11):
        for cycle in range(1, 6):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    **{
                        f"sensor_{index}": float(index + cycle + engine_id / 10)
                        for index in range(1, 22)
                    },
                }
            )
    dataset_dir.mkdir(parents=True)
    train_path = dataset_dir / "train.parquet"
    pl.DataFrame(rows).write_parquet(train_path)
    manifest = DatasetManifest(
        dataset_id="nasa-cmapss-classic",
        dataset_version="v1",
        subset_id="FD001",
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
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def test_training_publishes_auditable_candidate_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    artifact_root = tmp_path / "artifacts"
    publish_test_dataset(dataset_dir)

    evaluation = train_baseline(dataset_dir=dataset_dir, artifact_root=artifact_root)

    release_dir = artifact_root / "rul-model-v1"
    manifest = ModelArtifactManifest.model_validate_json(
        (release_dir / "manifest.json").read_text(encoding="utf-8")
    )
    evaluation_json = json.loads((release_dir / "evaluation.json").read_text())
    assert manifest.status.value == "candidate"
    assert manifest.validation_metrics.mae_cycles == evaluation.model_mae_cycles
    assert evaluation_json["prediction_postprocessing"] == "max(0, predicted_rul)"
    assert set(evaluation.training_engine_ids).isdisjoint(evaluation.validation_engine_ids)
    assert (release_dir / "model.joblib").is_file()
    assert (release_dir / "scaler.joblib").is_file()

    approved_manifest = manifest.model_copy(update={"status": ModelApprovalStatus.APPROVED})
    (release_dir / "manifest.json").write_text(
        approved_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    first_row = telemetry_frame().row(0, named=True)
    predictor = SklearnRULPredictor.load(release_dir / "manifest.json")
    prediction = predictor.predict(
        FeatureBuilder().build(
            cycle=int(first_row["cycle"]),
            measurements={name: float(first_row[name]) for name in FEATURE_NAMES[1:]},
        )
    )
    assert prediction >= 0

    with pytest.raises(BaselineTrainingError, match="already exists"):
        train_baseline(dataset_dir=dataset_dir, artifact_root=artifact_root)
