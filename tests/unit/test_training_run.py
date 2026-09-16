"""Tests for the complete RUL model-comparison training run."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from sklearn.dummy import DummyRegressor

from predictive_maintenance.data.download import calculate_sha256
from predictive_maintenance.data.manifest import DatasetManifest, FileArtifact
from predictive_maintenance.features.contracts import FEATURE_NAMES
from predictive_maintenance.training.train import TrainingRunError, train_and_compare


def _publish_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir()
    training_rows = []
    for engine_id in range(1, 7):
        for cycle in range(1, 4):
            training_rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    **{
                        f"sensor_{index}": float(engine_id + cycle + index)
                        for index in range(1, 22)
                    },
                }
            )

    test_rows = []
    for engine_id in range(1, 4):
        for cycle in range(1, 3):
            test_rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    **{
                        f"sensor_{index}": float(engine_id + cycle + index)
                        for index in range(1, 22)
                    },
                }
            )

    paths_and_frames = {
        "train.parquet": pl.DataFrame(training_rows),
        "test.parquet": pl.DataFrame(test_rows),
        "test-rul.parquet": pl.DataFrame(
            {
                "engine_id": [1, 2, 3],
                "observed_final_cycle": [2, 2, 2],
                "additional_rul": [1, 1, 1],
            }
        ),
    }
    artifacts = []
    for relative_path, frame in paths_and_frames.items():
        artifact_path = dataset_dir / relative_path
        frame.write_parquet(artifact_path)
        artifacts.append(
            FileArtifact(
                relative_path=relative_path,
                size_bytes=artifact_path.stat().st_size,
                sha256=calculate_sha256(artifact_path),
            )
        )

    manifest = DatasetManifest(
        dataset_id="nasa-cmapss-classic",
        dataset_version="v1",
        subset_id="FD001",
        telemetry_schema_version="telemetry-v1",
        test_rul_schema_version="test-rul-v1",
        source_manifest_sha256="0" * 64,
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        validation_status="PASSED",
        files=tuple(artifacts),
    )
    (dataset_dir / "dataset-manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def test_training_run_ranks_models_and_publishes_selected_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    output_root = tmp_path / "runs"
    _publish_dataset(dataset_dir)

    def fake_candidates(*, random_seed: int) -> dict[str, DummyRegressor]:
        assert random_seed == 42
        return {
            "linear_regression": DummyRegressor(strategy="constant", constant=10.0),
            "adaptive_lasso": DummyRegressor(strategy="constant", constant=8.0),
            "xgboost": DummyRegressor(strategy="constant", constant=4.0),
            "lightgbm": DummyRegressor(strategy="constant", constant=1.0),
        }

    monkeypatch.setattr(
        "predictive_maintenance.training.train.create_candidate_models",
        fake_candidates,
    )

    report = train_and_compare(
        dataset_dir=dataset_dir,
        output_root=output_root,
        run_name="test-run",
    )

    run_directory = output_root / "test-run"
    saved_report = json.loads((run_directory / "evaluation.json").read_text())
    assert report.selected_model_name == "lightgbm"
    assert [result.model_name for result in report.validation_ranking] == [
        "lightgbm",
        "xgboost",
        "adaptive_lasso",
        "linear_regression",
    ]
    assert report.official_test_evaluation.mae_cycles == 0.0
    assert report.feature_names == FEATURE_NAMES
    assert saved_report["selected_model_name"] == "lightgbm"
    assert (run_directory / "selected-model.joblib").is_file()

    with pytest.raises(TrainingRunError, match="already exists"):
        train_and_compare(
            dataset_dir=dataset_dir,
            output_root=output_root,
            run_name="test-run",
        )


@pytest.mark.parametrize("run_name", ["", "../escape", "contains spaces", "a" * 101])
def test_training_run_rejects_unsafe_run_name(tmp_path: Path, run_name: str) -> None:
    with pytest.raises(TrainingRunError, match="run_name"):
        train_and_compare(
            dataset_dir=tmp_path / "missing",
            output_root=tmp_path / "runs",
            run_name=run_name,
        )
