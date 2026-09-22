"""Package an immutable training-run winner for candidate review."""

import json
import math
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import joblib
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from predictive_maintenance.features.contracts import FEATURE_NAMES, FEATURE_VERSION
from predictive_maintenance.inference.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ModelApprovalStatus,
    ModelArtifactManifest,
    RegressionMetrics,
    calculate_artifact_sha256,
    load_model_manifest,
)

MODEL_RELEASE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
EXPECTED_TARGET_DEFINITION = "max_cycle_for_engine - current_cycle"
EXPECTED_POSTPROCESSING = "max(0, predicted_rul)"


class CandidatePackagingError(ValueError):
    """Raised when a training run cannot produce a safe candidate package."""


class _Regressor(Protocol):
    def predict(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...


class _EvaluationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model_name: str = Field(min_length=1)
    sample_count: int = Field(gt=0)
    mae_cycles: float = Field(ge=0)
    rmse_cycles: float = Field(ge=0)
    nasa_score: float = Field(ge=0)

    def to_manifest_metrics(self) -> RegressionMetrics:
        return RegressionMetrics.model_validate(self.model_dump())


class _TrainingRunEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str = Field(min_length=1)
    generated_at: datetime
    dataset_version: str = Field(min_length=1)
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_version: str = Field(min_length=1)
    feature_names: tuple[str, ...] = Field(min_length=1)
    target_definition: str = Field(min_length=1)
    prediction_postprocessing: str = Field(min_length=1)
    random_seed: int
    validation_fraction: float = Field(gt=0, lt=1)
    training_engine_ids: tuple[int, ...] = Field(min_length=1)
    validation_engine_ids: tuple[int, ...] = Field(min_length=1)
    training_rows: int = Field(gt=0)
    validation_rows: int = Field(gt=0)
    selection_policy: str = Field(min_length=1)
    validation_ranking: tuple[_EvaluationEvidence, ...] = Field(min_length=1)
    selected_model_name: str = Field(min_length=1)
    official_test_evaluation: _EvaluationEvidence

    @model_validator(mode="after")
    def validate_selected_model_evidence(self) -> "_TrainingRunEvidence":
        ranking_names = [result.model_name for result in self.validation_ranking]
        if len(ranking_names) != len(set(ranking_names)):
            raise ValueError("validation ranking contains duplicate model names")
        if self.selected_model_name not in ranking_names:
            raise ValueError("selected model is missing from validation ranking")
        if self.validation_ranking[0].model_name != self.selected_model_name:
            raise ValueError("selected model is not first in the validation ranking")
        if self.official_test_evaluation.model_name != self.selected_model_name:
            raise ValueError("official test result does not match the selected model")
        if set(self.training_engine_ids).intersection(self.validation_engine_ids):
            raise ValueError("training and validation engine identities overlap")
        if self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return self


def package_candidate(
    *,
    training_run_directory: Path,
    artifact_root: Path,
    model_release: str,
) -> ModelArtifactManifest:
    """Copy and integrity-protect one training winner as an immutable candidate."""

    if not MODEL_RELEASE_PATTERN.fullmatch(model_release):
        raise CandidatePackagingError(
            "model_release must start with a letter or number and contain only letters, "
            "numbers, dots, underscores, or hyphens"
        )

    report_path = training_run_directory / "evaluation.json"
    selected_model_path = training_run_directory / "selected-model.joblib"
    report = _load_training_evidence(report_path)
    if report.run_name != training_run_directory.name:
        raise CandidatePackagingError(
            "training report run_name does not match its immutable directory"
        )
    _validate_serving_contract(report)
    if not selected_model_path.is_file():
        raise CandidatePackagingError(
            f"selected model artifact is missing: {selected_model_path}"
        )

    artifact_root.mkdir(parents=True, exist_ok=True)
    release_directory = artifact_root / model_release
    if release_directory.exists():
        raise CandidatePackagingError(
            f"immutable model release already exists: {release_directory}"
        )

    staging_directory = Path(
        tempfile.mkdtemp(prefix=f".{model_release}-", dir=artifact_root)
    )
    try:
        packaged_model_path = staging_directory / "model.joblib"
        packaged_report_path = staging_directory / "evaluation.json"
        shutil.copy2(selected_model_path, packaged_model_path)
        shutil.copy2(report_path, packaged_report_path)
        _smoke_test_model(packaged_model_path)

        validation_result = next(
            result
            for result in report.validation_ranking
            if result.model_name == report.selected_model_name
        )
        manifest = ModelArtifactManifest(
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            model_release=model_release,
            model_name=report.selected_model_name,
            source_training_run=report.run_name,
            created_at=datetime.now(UTC),
            feature_version=report.feature_version,
            feature_names=report.feature_names,
            training_dataset_version=report.dataset_version,
            dataset_manifest_sha256=report.dataset_manifest_sha256,
            target_definition=report.target_definition,
            prediction_postprocessing=report.prediction_postprocessing,
            model_path=Path("model.joblib"),
            model_sha256=calculate_artifact_sha256(packaged_model_path),
            preprocessing_path=None,
            preprocessing_sha256=None,
            evaluation_report_path=Path("evaluation.json"),
            evaluation_report_sha256=calculate_artifact_sha256(packaged_report_path),
            validation_metrics=validation_result.to_manifest_metrics(),
            official_test_metrics=report.official_test_evaluation.to_manifest_metrics(),
            status=ModelApprovalStatus.CANDIDATE,
        )
        manifest_path = staging_directory / "manifest.json"
        manifest_path.write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        load_model_manifest(manifest_path)
        staging_directory.rename(release_directory)
    except Exception:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise

    return load_model_manifest(release_directory / "manifest.json")


def _load_training_evidence(report_path: Path) -> _TrainingRunEvidence:
    try:
        raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidatePackagingError(
            f"could not read training evaluation report: {report_path}"
        ) from error
    try:
        return _TrainingRunEvidence.model_validate(raw_report)
    except ValueError as error:
        raise CandidatePackagingError("training evaluation report failed validation") from error


def _validate_serving_contract(report: _TrainingRunEvidence) -> None:
    if report.feature_version != FEATURE_VERSION:
        raise CandidatePackagingError(
            f"training run uses {report.feature_version}, runtime provides {FEATURE_VERSION}"
        )
    if report.feature_names != FEATURE_NAMES:
        raise CandidatePackagingError(
            "training feature names or ordering do not match features-v1"
        )
    if report.target_definition != EXPECTED_TARGET_DEFINITION:
        raise CandidatePackagingError("training target is not supported by the runtime")
    if report.prediction_postprocessing != EXPECTED_POSTPROCESSING:
        raise CandidatePackagingError("prediction postprocessing is not supported by the runtime")


def _smoke_test_model(model_path: Path) -> None:
    if not model_path.is_file():
        raise CandidatePackagingError(f"selected model artifact is missing: {model_path}")
    try:
        model = cast(_Regressor, joblib.load(model_path))
        features = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float64)
        predictions = np.asarray(model.predict(features), dtype=np.float64)
    except Exception as error:
        raise CandidatePackagingError(
            "selected model could not run a serving-contract prediction"
        ) from error
    if predictions.shape != (1,):
        raise CandidatePackagingError("selected model returned an invalid prediction shape")
    if not math.isfinite(float(predictions[0])):
        raise CandidatePackagingError("selected model returned a non-finite prediction")
