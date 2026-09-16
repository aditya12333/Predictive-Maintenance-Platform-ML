"""Tests for model artifact metadata."""

import pytest
from pydantic import ValidationError

from predictive_maintenance.inference.artifacts import (
    ArtifactLoadError,
    ModelApprovalStatus,
    ModelArtifactManifest,
    load_approved_manifest,
)


def valid_manifest_payload() -> dict[str, object]:
    return {
        "model_release": "rul-model-v1",
        "feature_version": "features-v1",
        "training_dataset_version": "cmapss-fd001-v1",
        "model_path": "artifacts/rul-model-v1/model.joblib",
        "scaler_path": "artifacts/rul-model-v1/scaler.joblib",
        "mae_cycles": 12.4,
        "status": ModelApprovalStatus.APPROVED,
    }


def test_approved_manifest_is_valid() -> None:
    manifest = ModelArtifactManifest(**valid_manifest_payload())

    assert manifest.model_release == "rul-model-v1"
    assert manifest.status == ModelApprovalStatus.APPROVED
    assert manifest.mae_cycles == 12.4


def test_manifest_rejects_negative_mae() -> None:
    payload = valid_manifest_payload()
    payload["mae_cycles"] = -1.0

    with pytest.raises(ValidationError):
        ModelArtifactManifest(**payload)


def test_manifest_rejects_empty_model_release() -> None:
    payload = valid_manifest_payload()
    payload["model_release"] = ""

    with pytest.raises(ValidationError):
        ModelArtifactManifest(**payload)


def test_loader_accepts_approved_manifest_with_existing_files(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    scaler_path = tmp_path / "scaler.joblib"
    model_path.write_bytes(b"model")
    scaler_path.write_bytes(b"scaler")
    manifest_path = tmp_path / "manifest.json"
    payload = valid_manifest_payload()
    payload["model_path"] = model_path.name
    payload["scaler_path"] = scaler_path.name
    manifest_path.write_text(
        ModelArtifactManifest(**payload).model_dump_json(),
        encoding="utf-8",
    )

    manifest = load_approved_manifest(manifest_path)

    assert manifest.status is ModelApprovalStatus.APPROVED
    assert manifest.model_path == model_path
    assert manifest.scaler_path == scaler_path


def test_loader_rejects_unapproved_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload = valid_manifest_payload()
    payload["status"] = ModelApprovalStatus.CANDIDATE
    manifest_path.write_text(
        ModelArtifactManifest(**payload).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactLoadError, match="not approved"):
        load_approved_manifest(manifest_path)


def test_loader_rejects_missing_artifact_file(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        ModelArtifactManifest(**valid_manifest_payload()).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactLoadError, match="files are missing"):
        load_approved_manifest(manifest_path)
