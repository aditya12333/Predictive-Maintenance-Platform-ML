"""Tests for immutable candidate packaging from a model-comparison run."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from predictive_maintenance.features.contracts import FEATURE_NAMES
from predictive_maintenance.inference.artifacts import (
    ArtifactLoadError,
    ModelApprovalStatus,
    calculate_artifact_sha256,
    load_approved_manifest,
)
from predictive_maintenance.training.package import (
    CandidatePackagingError,
    package_candidate,
)

RUN_NAME = "fd001-test-comparison"


def write_training_run(root: Path) -> Path:
    run_directory = root / RUN_NAME
    run_directory.mkdir()
    model = DummyRegressor(strategy="constant", constant=12.0)
    model.fit(np.zeros((2, len(FEATURE_NAMES))), np.asarray([10.0, 14.0]))
    joblib.dump(model, run_directory / "selected-model.joblib")

    report = {
        "run_name": RUN_NAME,
        "generated_at": "2026-09-20T10:00:00Z",
        "dataset_version": "nasa-cmapss-classic:v1:FD001",
        "dataset_manifest_sha256": "b" * 64,
        "feature_version": "features-v1",
        "feature_names": FEATURE_NAMES,
        "target_definition": "max_cycle_for_engine - current_cycle",
        "prediction_postprocessing": "max(0, predicted_rul)",
        "random_seed": 42,
        "validation_fraction": 0.2,
        "training_engine_ids": [1, 2],
        "validation_engine_ids": [3],
        "training_rows": 4,
        "validation_rows": 2,
        "selection_policy": "lowest validation MAE",
        "validation_ranking": [
            {
                "model_name": "lightgbm",
                "sample_count": 2,
                "mae_cycles": 10.0,
                "rmse_cycles": 12.0,
                "nasa_score": 30.0,
            }
        ],
        "selected_model_name": "lightgbm",
        "official_test_evaluation": {
            "model_name": "lightgbm",
            "sample_count": 1,
            "mae_cycles": 9.0,
            "rmse_cycles": 9.0,
            "nasa_score": 2.0,
        },
    }
    (run_directory / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_directory


def test_package_candidate_publishes_integrity_checked_unapproved_release(
    tmp_path: Path,
) -> None:
    training_run = write_training_run(tmp_path)
    artifact_root = tmp_path / "releases"
    source_model_sha256 = calculate_artifact_sha256(
        training_run / "selected-model.joblib"
    )
    source_report_sha256 = calculate_artifact_sha256(training_run / "evaluation.json")

    manifest = package_candidate(
        training_run_directory=training_run,
        artifact_root=artifact_root,
        model_release="rul-lightgbm-v1",
    )

    release_directory = artifact_root / "rul-lightgbm-v1"
    manifest_path = release_directory / "manifest.json"
    assert manifest.status is ModelApprovalStatus.CANDIDATE
    assert manifest.model_name == "lightgbm"
    assert manifest.source_training_run == RUN_NAME
    assert manifest.feature_names == FEATURE_NAMES
    assert manifest.preprocessing_path is None
    assert manifest.model_sha256 == source_model_sha256
    assert manifest.evaluation_report_sha256 == source_report_sha256
    assert manifest.model_path == release_directory / "model.joblib"
    assert manifest.validation_metrics.mae_cycles == 10.0
    assert manifest.official_test_metrics is not None
    assert manifest.official_test_metrics.mae_cycles == 9.0
    assert manifest_path.is_file()

    with pytest.raises(ArtifactLoadError, match="not approved"):
        load_approved_manifest(manifest_path)


def test_package_candidate_refuses_to_overwrite_release(tmp_path: Path) -> None:
    training_run = write_training_run(tmp_path)
    artifact_root = tmp_path / "releases"
    package_candidate(
        training_run_directory=training_run,
        artifact_root=artifact_root,
        model_release="rul-lightgbm-v1",
    )

    with pytest.raises(CandidatePackagingError, match="already exists"):
        package_candidate(
            training_run_directory=training_run,
            artifact_root=artifact_root,
            model_release="rul-lightgbm-v1",
        )


def test_package_candidate_rejects_incompatible_feature_order(tmp_path: Path) -> None:
    training_run = write_training_run(tmp_path)
    report_path = training_run / "evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["feature_names"] = list(reversed(FEATURE_NAMES))
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(CandidatePackagingError, match="ordering"):
        package_candidate(
            training_run_directory=training_run,
            artifact_root=tmp_path / "releases",
            model_release="rul-lightgbm-v1",
        )

    assert not (tmp_path / "releases" / "rul-lightgbm-v1").exists()
