"""Contracts and integrity checks for versioned model artifacts."""

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ARTIFACT_SCHEMA_VERSION: Literal["model-artifact-v1"] = "model-artifact-v1"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ModelApprovalStatus(StrEnum):
    """Review state of a model artifact."""

    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETIRED = "retired"


class RegressionMetrics(BaseModel):
    """Regression evidence recorded for one evaluation population."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model_name: str = Field(min_length=1)
    sample_count: int = Field(gt=0)
    mae_cycles: float = Field(ge=0)
    rmse_cycles: float = Field(ge=0)
    nasa_score: float = Field(ge=0)


class ModelArtifactManifest(BaseModel):
    """Portable metadata required to inspect and load one model release."""

    model_config = ConfigDict(extra="forbid")

    artifact_schema_version: Literal["model-artifact-v1"] = ARTIFACT_SCHEMA_VERSION
    model_release: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$",
    )
    model_name: str = Field(min_length=1)
    source_training_run: str = Field(min_length=1)
    created_at: datetime
    feature_version: str = Field(min_length=1)
    feature_names: tuple[str, ...] = Field(min_length=1)
    training_dataset_version: str = Field(min_length=1)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    target_definition: str = Field(min_length=1)
    prediction_postprocessing: str = Field(min_length=1)
    model_path: Path
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    preprocessing_path: Path | None = None
    preprocessing_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    evaluation_report_path: Path
    evaluation_report_sha256: str = Field(pattern=SHA256_PATTERN)
    validation_metrics: RegressionMetrics
    official_test_metrics: RegressionMetrics | None = None
    status: ModelApprovalStatus

    @field_validator("model_path", "preprocessing_path", "evaluation_report_path")
    @classmethod
    def validate_package_path(cls, value: Path | None) -> Path | None:
        """Require portable paths that cannot escape the release directory."""

        if value is None:
            return None
        if value.is_absolute() or value == Path(".") or ".." in value.parts:
            raise ValueError("artifact paths must be relative to the release directory")
        return value

    @model_validator(mode="after")
    def validate_related_fields(self) -> "ModelArtifactManifest":
        """Keep optional preprocessing metadata complete and timestamps auditable."""

        if (self.preprocessing_path is None) != (self.preprocessing_sha256 is None):
            raise ValueError(
                "preprocessing_path and preprocessing_sha256 must both be set or both be null"
            )
        if self.created_at.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature_names must not contain duplicates")
        if self.validation_metrics.model_name != self.model_name:
            raise ValueError("validation metrics do not match model_name")
        if (
            self.official_test_metrics is not None
            and self.official_test_metrics.model_name != self.model_name
        ):
            raise ValueError("official test metrics do not match model_name")
        return self


class ArtifactLoadError(ValueError):
    """Raised when a model artifact cannot be validated or loaded safely."""


def calculate_artifact_sha256(path: Path) -> str:
    """Return the SHA-256 digest for an artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_manifest(manifest_path: Path) -> ModelArtifactManifest:
    """Validate any lifecycle state and verify every declared artifact."""

    manifest = _read_manifest(manifest_path)
    return _resolve_and_verify_artifacts(manifest, manifest_path=manifest_path)


def load_approved_manifest(manifest_path: Path) -> ModelArtifactManifest:
    """Load an integrity-verified manifest only when it is explicitly approved."""

    manifest = _read_manifest(manifest_path)
    if manifest.status is not ModelApprovalStatus.APPROVED:
        raise ArtifactLoadError(
            f"model release {manifest.model_release} is not approved"
        )
    return _resolve_and_verify_artifacts(manifest, manifest_path=manifest_path)


def _read_manifest(manifest_path: Path) -> ModelArtifactManifest:
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactLoadError(f"could not read manifest: {manifest_path}") from error

    try:
        return ModelArtifactManifest.model_validate(manifest_data)
    except ValueError as error:
        raise ArtifactLoadError("model manifest failed validation") from error


def _resolve_and_verify_artifacts(
    manifest: ModelArtifactManifest,
    *,
    manifest_path: Path,
) -> ModelArtifactManifest:
    release_directory = manifest_path.parent.resolve()
    declared_artifacts = [
        ("model", manifest.model_path, manifest.model_sha256),
        (
            "evaluation report",
            manifest.evaluation_report_path,
            manifest.evaluation_report_sha256,
        ),
    ]
    if manifest.preprocessing_path is not None and manifest.preprocessing_sha256 is not None:
        declared_artifacts.append(
            ("preprocessing", manifest.preprocessing_path, manifest.preprocessing_sha256)
        )

    resolved_paths: dict[str, Path] = {}
    for artifact_name, relative_path, expected_sha256 in declared_artifacts:
        artifact_path = (release_directory / relative_path).resolve()
        if not artifact_path.is_relative_to(release_directory):
            raise ArtifactLoadError(
                f"{artifact_name} path escapes the model release directory"
            )
        if not artifact_path.is_file():
            raise ArtifactLoadError(f"model artifact file is missing: {artifact_path}")
        try:
            actual_sha256 = calculate_artifact_sha256(artifact_path)
        except OSError as error:
            raise ArtifactLoadError(
                f"could not read {artifact_name} artifact: {artifact_path}"
            ) from error
        if actual_sha256 != expected_sha256:
            raise ArtifactLoadError(f"{artifact_name} artifact checksum does not match")
        resolved_paths[artifact_name] = artifact_path

    return manifest.model_copy(
        update={
            "model_path": resolved_paths["model"],
            "evaluation_report_path": resolved_paths["evaluation report"],
            "preprocessing_path": resolved_paths.get("preprocessing"),
        }
    )
