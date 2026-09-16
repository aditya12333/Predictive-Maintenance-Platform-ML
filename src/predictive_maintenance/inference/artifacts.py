"""Contracts for versioned model artifacts."""

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class ModelApprovalStatus(StrEnum):
    """Lifecycle state of a model artifact."""

    CANDIDATE = "candidate"
    APPROVED = "approved"
    RETIRED = "retired"


class ModelArtifactManifest(BaseModel):
    """Metadata required to load and audit a model release."""

    model_release: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    training_dataset_version: str = Field(min_length=1)
    model_path: Path
    scaler_path: Path
    mae_cycles: float = Field(ge=0)
    status: ModelApprovalStatus


class ArtifactLoadError(ValueError):
    """Raised when an approved model artifact cannot be loaded safely."""


def load_approved_manifest(manifest_path: Path) -> ModelArtifactManifest:
    """Validate an approved manifest and verify its model files exist.

    The loader intentionally validates metadata before any model deserialization.
    Candidate and retired releases are not allowed into the production inference
    path, and missing files fail early with an actionable error.
    """

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactLoadError(f"could not read manifest: {manifest_path}") from error

    try:
        manifest = ModelArtifactManifest.model_validate(manifest_data)
    except ValueError as error:
        raise ArtifactLoadError("model manifest failed validation") from error

    if manifest.status is not ModelApprovalStatus.APPROVED:
        raise ArtifactLoadError(
            f"model release {manifest.model_release} is not approved"
        )

    # Manifest paths are relative to the directory containing manifest.json.
    model_path = (
        manifest.model_path
        if manifest.model_path.is_absolute()
        else manifest_path.parent / manifest.model_path
    )
    scaler_path = (
        manifest.scaler_path
        if manifest.scaler_path.is_absolute()
        else manifest_path.parent / manifest.scaler_path
    )

    # Fail before inference starts; do not deploy a manifest with missing artifacts.
    missing_paths = [path for path in (model_path, scaler_path) if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise ArtifactLoadError(f"model artifact files are missing: {missing}")

    return manifest.model_copy(update={"model_path": model_path, "scaler_path": scaler_path})
