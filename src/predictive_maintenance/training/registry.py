"""Register integrity-verified candidate packages in MLflow without promoting them."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.sklearn import save_model

from predictive_maintenance.inference.artifacts import (
    ArtifactLoadError,
    ModelApprovalStatus,
    ModelArtifactManifest,
    load_model_manifest,
)

MODEL_RELEASE_TAG = "model_release"
APPROVAL_STATUS_TAG = "approval_status"
RESOURCE_DOES_NOT_EXIST = "RESOURCE_DOES_NOT_EXIST"


class CandidateRegistrationError(RuntimeError):
    """Raised when a candidate cannot be registered safely."""


@dataclass(frozen=True)
class CandidateRegistration:
    """Identity of one candidate version in the MLflow Model Registry."""

    registered_model_name: str
    model_version: str
    model_release: str
    approval_status: str
    source_run_id: str | None
    already_registered: bool


def register_candidate(
    *,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    manifest_path: Path,
) -> CandidateRegistration:
    """Register one immutable candidate package, or return its existing version."""

    try:
        manifest = load_model_manifest(manifest_path)
    except ArtifactLoadError as error:
        raise CandidateRegistrationError("candidate package failed integrity validation") from error
    if manifest.status is not ModelApprovalStatus.CANDIDATE:
        raise CandidateRegistrationError(
            f"model release {manifest.model_release} has status {manifest.status.value}; "
            "only candidates can be registered"
        )

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        _get_or_create_registered_model(client, registered_model_name)
        existing = _find_release_version(
            client,
            registered_model_name=registered_model_name,
            manifest=manifest,
        )
        if existing is not None:
            return existing
        experiment_id = _get_or_create_experiment(client, experiment_name)
        return _create_candidate_version(
            client,
            experiment_id=experiment_id,
            registered_model_name=registered_model_name,
            manifest=manifest,
            manifest_path=manifest_path,
        )
    except CandidateRegistrationError:
        raise
    except Exception as error:
        raise CandidateRegistrationError(
            f"could not register candidate with MLflow at {tracking_uri}"
        ) from error


def _get_or_create_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
        return
    except MlflowException as error:
        if error.error_code != RESOURCE_DOES_NOT_EXIST:
            raise

    client.create_registered_model(
        name,
        description="Remaining useful life regression candidates for NASA C-MAPSS FD001.",
        tags={
            "task": "rul-regression",
            "dataset": "nasa-cmapss-fd001",
        },
    )


def _get_or_create_experiment(client: MlflowClient, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    return experiment.experiment_id if experiment is not None else client.create_experiment(name)


def _find_release_version(
    client: MlflowClient,
    *,
    registered_model_name: str,
    manifest: ModelArtifactManifest,
) -> CandidateRegistration | None:
    versions = [
        version
        for version in client.search_model_versions(
            filter_string=f"name = '{registered_model_name}'"
        )
        if version.tags.get(MODEL_RELEASE_TAG) == manifest.model_release
    ]
    if not versions:
        return None
    if len(versions) > 1:
        raise CandidateRegistrationError(
            f"multiple registry versions use model release {manifest.model_release}"
        )

    version = versions[0]
    expected_immutable_tags = immutable_version_tags(manifest)
    mismatched = [
        key
        for key, expected in expected_immutable_tags.items()
        if version.tags.get(key) != expected
    ]
    if mismatched:
        fields = ", ".join(sorted(mismatched))
        raise CandidateRegistrationError(
            f"registered release {manifest.model_release} conflicts on: {fields}"
        )

    approval_status = version.tags.get(APPROVAL_STATUS_TAG)
    if approval_status not in {status.value for status in ModelApprovalStatus}:
        raise CandidateRegistrationError("registered release has an invalid approval status")
    return CandidateRegistration(
        registered_model_name=registered_model_name,
        model_version=str(version.version),
        model_release=manifest.model_release,
        approval_status=approval_status,
        source_run_id=version.run_id,
        already_registered=True,
    )


def _create_candidate_version(
    client: MlflowClient,
    *,
    experiment_id: str,
    registered_model_name: str,
    manifest: ModelArtifactManifest,
    manifest_path: Path,
) -> CandidateRegistration:
    registration_run = client.create_run(
        experiment_id,
        run_name=f"register-{manifest.model_release}",
        tags={
            "project": "predictive-maintenance-platform",
            "tracking_structure": "registry-source",
            "purpose": "candidate-registration",
            MODEL_RELEASE_TAG: manifest.model_release,
            APPROVAL_STATUS_TAG: ModelApprovalStatus.CANDIDATE.value,
        },
    )
    run_id = registration_run.info.run_id
    try:
        _log_registration_evidence(client, run_id=run_id, manifest=manifest)
        client.log_artifacts(
            run_id,
            str(manifest_path.parent),
            artifact_path="candidate-package",
        )
        with tempfile.TemporaryDirectory(prefix="mlflow-registry-model-") as temp_directory:
            model_directory = Path(temp_directory) / "model"
            _create_mlflow_model_directory(manifest, model_directory=model_directory)
            client.log_artifacts(run_id, str(model_directory), artifact_path="registry-model")

        version = client.create_model_version(
            registered_model_name,
            source=f"runs:/{run_id}/registry-model",
            run_id=run_id,
            tags={
                **immutable_version_tags(manifest),
                APPROVAL_STATUS_TAG: ModelApprovalStatus.CANDIDATE.value,
            },
            description=(
                f"Immutable candidate {manifest.model_release}; approval and serving "
                "promotion are separate operations."
            ),
        )
        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        try:
            client.set_terminated(run_id, status="FAILED")
        except Exception:
            pass
        raise

    return CandidateRegistration(
        registered_model_name=registered_model_name,
        model_version=str(version.version),
        model_release=manifest.model_release,
        approval_status=ModelApprovalStatus.CANDIDATE.value,
        source_run_id=run_id,
        already_registered=False,
    )


def _log_registration_evidence(
    client: MlflowClient,
    *,
    run_id: str,
    manifest: ModelArtifactManifest,
) -> None:
    parameters = {
        "model_release": manifest.model_release,
        "model_name": manifest.model_name,
        "source_training_run": manifest.source_training_run,
        "artifact_schema_version": manifest.artifact_schema_version,
        "feature_version": manifest.feature_version,
        "feature_names": ",".join(manifest.feature_names),
        "training_dataset_version": manifest.training_dataset_version,
        "target_definition": manifest.target_definition,
        "prediction_postprocessing": manifest.prediction_postprocessing,
    }
    for key, value in parameters.items():
        client.log_param(run_id, key, value)

    validation = manifest.validation_metrics
    client.log_metric(run_id, "validation_mae_cycles", validation.mae_cycles)
    client.log_metric(run_id, "validation_rmse_cycles", validation.rmse_cycles)
    client.log_metric(run_id, "validation_nasa_score", validation.nasa_score)
    if manifest.official_test_metrics is not None:
        official = manifest.official_test_metrics
        client.log_metric(run_id, "official_test_mae_cycles", official.mae_cycles)
        client.log_metric(run_id, "official_test_rmse_cycles", official.rmse_cycles)
        client.log_metric(run_id, "official_test_nasa_score", official.nasa_score)


def immutable_version_tags(manifest: ModelArtifactManifest) -> dict[str, str]:
    """Return lineage tags that must never change for a registered version."""

    return {
        MODEL_RELEASE_TAG: manifest.model_release,
        "model_name": manifest.model_name,
        "source_training_run": manifest.source_training_run,
        "artifact_schema_version": manifest.artifact_schema_version,
        "feature_version": manifest.feature_version,
        "training_dataset_version": manifest.training_dataset_version,
        "dataset_manifest_sha256": manifest.dataset_manifest_sha256,
        "model_sha256": manifest.model_sha256,
        "evaluation_report_sha256": manifest.evaluation_report_sha256,
    }


def _create_mlflow_model_directory(
    manifest: ModelArtifactManifest,
    *,
    model_directory: Path,
) -> None:
    try:
        model = cast(Any, joblib.load(manifest.model_path))
        input_example = np.zeros((1, len(manifest.feature_names)), dtype=np.float64)
        output_example = model.predict(input_example)
        signature = infer_signature(input_example, output_example)
        save_model(
            model,
            path=str(model_directory),
            serialization_format="cloudpickle",
            signature=signature,
            input_example=input_example,
            metadata={
                MODEL_RELEASE_TAG: manifest.model_release,
                "model_sha256": manifest.model_sha256,
            },
        )
    except Exception as error:
        raise CandidateRegistrationError(
            "could not convert candidate to an MLflow model artifact"
        ) from error
