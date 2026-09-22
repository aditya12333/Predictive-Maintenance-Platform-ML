"""Tests for model artifact metadata and integrity verification."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from predictive_maintenance.features.contracts import FEATURE_NAMES
from predictive_maintenance.inference.artifacts import (
    ArtifactLoadError,
    ModelApprovalStatus,
    ModelArtifactManifest,
    calculate_artifact_sha256,
    load_approved_manifest,
    load_model_manifest,
)

FAKE_SHA256 = "a" * 64


def valid_manifest_payload() -> dict[str, object]:
    return {
        "artifact_schema_version": "model-artifact-v1",
        "model_release": "rul-model-v1",
        "model_name": "ridge",
        "source_training_run": "fd001-test-run",
        "created_at": "2026-09-20T10:00:00Z",
        "feature_version": "features-v1",
        "feature_names": FEATURE_NAMES,
        "training_dataset_version": "cmapss-fd001-v1",
        "dataset_manifest_sha256": FAKE_SHA256,
        "target_definition": "max_cycle_for_engine - current_cycle",
        "prediction_postprocessing": "max(0, predicted_rul)",
        "model_path": "model.joblib",
        "model_sha256": FAKE_SHA256,
        "preprocessing_path": "scaler.joblib",
        "preprocessing_sha256": FAKE_SHA256,
        "evaluation_report_path": "evaluation.json",
        "evaluation_report_sha256": FAKE_SHA256,
        "validation_metrics": {
            "model_name": "ridge",
            "sample_count": 20,
            "mae_cycles": 12.4,
            "rmse_cycles": 16.2,
            "nasa_score": 55.0,
        },
        "official_test_metrics": None,
        "status": ModelApprovalStatus.APPROVED,
    }


def write_package(
    package_directory: Path,
    *,
    status: ModelApprovalStatus = ModelApprovalStatus.APPROVED,
) -> Path:
    model_path = package_directory / "model.joblib"
    preprocessing_path = package_directory / "scaler.joblib"
    evaluation_path = package_directory / "evaluation.json"
    model_path.write_bytes(b"model")
    preprocessing_path.write_bytes(b"scaler")
    evaluation_path.write_text("{}\n", encoding="utf-8")

    payload = valid_manifest_payload()
    payload.update(
        {
            "model_sha256": calculate_artifact_sha256(model_path),
            "preprocessing_sha256": calculate_artifact_sha256(preprocessing_path),
            "evaluation_report_sha256": calculate_artifact_sha256(evaluation_path),
            "status": status,
        }
    )
    manifest_path = package_directory / "manifest.json"
    manifest_path.write_text(
        ModelArtifactManifest(**payload).model_dump_json(),
        encoding="utf-8",
    )
    return manifest_path


def test_approved_manifest_is_valid() -> None:
    manifest = ModelArtifactManifest(**valid_manifest_payload())

    assert manifest.model_release == "rul-model-v1"
    assert manifest.status == ModelApprovalStatus.APPROVED
    assert manifest.validation_metrics.mae_cycles == 12.4


def test_manifest_rejects_negative_mae() -> None:
    payload = valid_manifest_payload()
    metrics_value = payload["validation_metrics"]
    assert isinstance(metrics_value, dict)
    metrics = dict(metrics_value)
    metrics["mae_cycles"] = -1.0
    payload["validation_metrics"] = metrics

    with pytest.raises(ValidationError):
        ModelArtifactManifest(**payload)


def test_manifest_rejects_empty_model_release() -> None:
    payload = valid_manifest_payload()
    payload["model_release"] = ""

    with pytest.raises(ValidationError):
        ModelArtifactManifest(**payload)


def test_manifest_rejects_incomplete_preprocessing_metadata() -> None:
    payload = valid_manifest_payload()
    payload["preprocessing_sha256"] = None

    with pytest.raises(ValidationError, match="must both be set"):
        ModelArtifactManifest(**payload)


def test_manifest_rejects_path_traversal() -> None:
    payload = valid_manifest_payload()
    payload["model_path"] = "../model.joblib"

    with pytest.raises(ValidationError, match="relative"):
        ModelArtifactManifest(**payload)


def test_loader_accepts_approved_manifest_with_valid_checksums(tmp_path: Path) -> None:
    manifest_path = write_package(tmp_path)

    manifest = load_approved_manifest(manifest_path)

    assert manifest.status is ModelApprovalStatus.APPROVED
    assert manifest.model_path == tmp_path / "model.joblib"
    assert manifest.preprocessing_path == tmp_path / "scaler.joblib"
    assert manifest.evaluation_report_path == tmp_path / "evaluation.json"


def test_general_loader_inspects_integrity_checked_candidate(tmp_path: Path) -> None:
    manifest_path = write_package(tmp_path, status=ModelApprovalStatus.CANDIDATE)

    manifest = load_model_manifest(manifest_path)

    assert manifest.status is ModelApprovalStatus.CANDIDATE


def test_approved_loader_rejects_candidate_before_serving(tmp_path: Path) -> None:
    manifest_path = write_package(tmp_path, status=ModelApprovalStatus.CANDIDATE)

    with pytest.raises(ArtifactLoadError, match="not approved"):
        load_approved_manifest(manifest_path)


def test_loader_rejects_missing_artifact_file(tmp_path: Path) -> None:
    manifest_path = write_package(tmp_path)
    (tmp_path / "model.joblib").unlink()

    with pytest.raises(ArtifactLoadError, match="file is missing"):
        load_approved_manifest(manifest_path)


def test_loader_rejects_modified_artifact(tmp_path: Path) -> None:
    manifest_path = write_package(tmp_path)
    (tmp_path / "model.joblib").write_bytes(b"modified-model")

    with pytest.raises(ArtifactLoadError, match="checksum does not match"):
        load_approved_manifest(manifest_path)


def test_manifest_supports_no_preprocessing_artifact() -> None:
    payload = valid_manifest_payload()
    payload["preprocessing_path"] = None
    payload["preprocessing_sha256"] = None

    manifest = ModelArtifactManifest(**payload)

    assert manifest.preprocessing_path is None
