"""Human approval records for registered model candidates."""

import hashlib
import tempfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from predictive_maintenance.features.contracts import FEATURE_NAMES, FEATURE_VERSION
from predictive_maintenance.inference.artifacts import (
    ArtifactLoadError,
    ModelApprovalStatus,
    ModelArtifactManifest,
    load_model_manifest,
)
from predictive_maintenance.training.package import (
    EXPECTED_POSTPROCESSING,
    EXPECTED_TARGET_DEFINITION,
)
from predictive_maintenance.training.registry import (
    APPROVAL_STATUS_TAG,
    immutable_version_tags,
)

APPROVAL_EVIDENCE_SCHEMA_VERSION: Literal["approval-evidence-v1"] = "approval-evidence-v1"
PROMOTION_POLICY_VERSION: Literal["model-promotion-policy-v1"] = "model-promotion-policy-v1"
RESOURCE_DOES_NOT_EXIST = "RESOURCE_DOES_NOT_EXIST"
INVALID_PARAMETER_VALUE = "INVALID_PARAMETER_VALUE"


class GateStatus(StrEnum):
    """Result of one required approval gate."""

    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"


class GateAssessment(BaseModel):
    """Auditable result and evidence reference for one promotion gate."""

    model_config = ConfigDict(extra="forbid")

    status: GateStatus
    details: str = Field(min_length=1)
    evidence_reference: str = Field(min_length=1)

    @field_validator("details", "evidence_reference")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class ApprovalEvidence(BaseModel):
    """Complete gate evidence required before a human approval decision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["approval-evidence-v1"] = APPROVAL_EVIDENCE_SCHEMA_VERSION
    promotion_policy_version: Literal["model-promotion-policy-v1"] = PROMOTION_POLICY_VERSION
    model_release: str = Field(min_length=1)
    assessed_at: datetime
    assessor: str = Field(min_length=1)
    provenance_integrity: GateAssessment
    serving_compatibility: GateAssessment
    prediction_validity: GateAssessment
    metric_validity: GateAssessment
    performance_policy: GateAssessment
    operational_readiness: GateAssessment
    automated_checks: GateAssessment

    @field_validator("model_release", "assessor")
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_timestamp(self) -> "ApprovalEvidence":
        if self.assessed_at.utcoffset() is None:
            raise ValueError("assessed_at must include a timezone")
        return self

    def gate_results(self) -> dict[str, GateAssessment]:
        """Return every required gate under its stable policy name."""

        return {
            "provenance_integrity": self.provenance_integrity,
            "serving_compatibility": self.serving_compatibility,
            "prediction_validity": self.prediction_validity,
            "metric_validity": self.metric_validity,
            "performance_policy": self.performance_policy,
            "operational_readiness": self.operational_readiness,
            "automated_checks": self.automated_checks,
        }


class ApprovalError(RuntimeError):
    """Raised when a registered candidate cannot be approved safely."""


class ApprovalDecision(BaseModel):
    """Immutable human decision record uploaded to the approval run."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved"] = "approved"
    registered_model_name: str
    model_version: str
    model_release: str
    approver: str
    decided_at: datetime
    reason: str
    source_training_run: str
    promotion_policy_version: str
    gates: dict[str, GateAssessment]
    model_sha256: str
    evaluation_report_sha256: str
    dataset_manifest_sha256: str
    approval_evidence_sha256: str
    current_champion_version: str | None
    intended_previous_version: str | None


class ApprovalResult(BaseModel):
    """Result returned after approving or finding an existing approval."""

    registered_model_name: str
    model_version: str
    model_release: str
    approval_status: str
    decision_run_id: str
    already_approved: bool


def approve_candidate(
    *,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    model_version: str,
    manifest_path: Path,
    evidence_path: Path,
    approver: str,
    reason: str,
) -> ApprovalResult:
    """Approve a registry candidate while leaving serving aliases unchanged."""

    normalized_approver = approver.strip()
    normalized_reason = reason.strip()
    if not normalized_approver:
        raise ApprovalError("approver must not be blank")
    if not normalized_reason:
        raise ApprovalError("approval reason must not be blank")

    manifest = load_candidate_manifest(manifest_path)
    verify_serving_package(manifest)
    evidence, evidence_sha256 = _load_approval_evidence(evidence_path)
    if evidence.model_release != manifest.model_release:
        raise ApprovalError("approval evidence does not match the model release")
    blocking_gates = [
        name
        for name, assessment in evidence.gate_results().items()
        if assessment.status is not GateStatus.PASSED
    ]
    if blocking_gates:
        raise ApprovalError(f"approval gates are not passed: {', '.join(blocking_gates)}")

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        version = client.get_model_version(registered_model_name, model_version)
        _verify_registry_version(version, manifest=manifest)
        current_status = version.tags.get(APPROVAL_STATUS_TAG)
        if current_status == ModelApprovalStatus.APPROVED.value:
            decision_run_id = version.tags.get("approval_decision_run_id")
            if not decision_run_id:
                raise ApprovalError("approved registry version has no decision record")
            return ApprovalResult(
                registered_model_name=registered_model_name,
                model_version=str(version.version),
                model_release=manifest.model_release,
                approval_status=current_status,
                decision_run_id=decision_run_id,
                already_approved=True,
            )
        if current_status != ModelApprovalStatus.CANDIDATE.value:
            raise ApprovalError(
                f"registry version has approval status {current_status}; expected candidate"
            )

        experiment_id = _get_or_create_experiment(client, experiment_name)
        champion_version = _get_alias_version(client, registered_model_name, "champion")
        decision = ApprovalDecision(
            registered_model_name=registered_model_name,
            model_version=str(version.version),
            model_release=manifest.model_release,
            approver=normalized_approver,
            decided_at=datetime.now(UTC),
            reason=normalized_reason,
            source_training_run=manifest.source_training_run,
            promotion_policy_version=evidence.promotion_policy_version,
            gates=evidence.gate_results(),
            model_sha256=manifest.model_sha256,
            evaluation_report_sha256=manifest.evaluation_report_sha256,
            dataset_manifest_sha256=manifest.dataset_manifest_sha256,
            approval_evidence_sha256=evidence_sha256,
            current_champion_version=champion_version,
            intended_previous_version=champion_version,
        )
        return _record_approval(
            client,
            experiment_id=experiment_id,
            decision=decision,
        )
    except ApprovalError:
        raise
    except Exception as error:
        raise ApprovalError(
            f"could not approve registry candidate at {tracking_uri}"
        ) from error


def load_candidate_manifest(manifest_path: Path) -> ModelArtifactManifest:
    """Load and integrity-check the unchanged local candidate package."""

    try:
        manifest = load_model_manifest(manifest_path)
    except ArtifactLoadError as error:
        raise ApprovalError("candidate package failed integrity validation") from error
    if manifest.status is not ModelApprovalStatus.CANDIDATE:
        raise ApprovalError("local immutable package must remain in candidate state")
    return manifest


def _load_approval_evidence(evidence_path: Path) -> tuple[ApprovalEvidence, str]:
    try:
        evidence_bytes = evidence_path.read_bytes()
        evidence = ApprovalEvidence.model_validate_json(evidence_bytes)
    except (OSError, ValueError) as error:
        raise ApprovalError("approval evidence failed validation") from error
    return evidence, hashlib.sha256(evidence_bytes).hexdigest()


def verify_serving_package(manifest: ModelArtifactManifest) -> None:
    """Re-run runtime compatibility and prediction-validity checks."""

    if manifest.feature_version != FEATURE_VERSION or manifest.feature_names != FEATURE_NAMES:
        raise ApprovalError("candidate feature contract is not supported by the runtime")
    if manifest.target_definition != EXPECTED_TARGET_DEFINITION:
        raise ApprovalError("candidate target definition is not supported by the runtime")
    if manifest.prediction_postprocessing != EXPECTED_POSTPROCESSING:
        raise ApprovalError("candidate prediction postprocessing is not supported by the runtime")
    try:
        model = joblib.load(manifest.model_path)
        features = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float64)
        predictions = np.asarray(model.predict(features), dtype=np.float64)
    except Exception as error:
        raise ApprovalError("candidate failed the serving-contract smoke test") from error
    if predictions.shape != (1,) or not np.isfinite(predictions).all():
        raise ApprovalError("candidate produced an invalid serving-contract prediction")


def _verify_registry_version(
    version: Any,
    *,
    manifest: ModelArtifactManifest,
) -> None:
    tags = getattr(version, "tags", {})
    mismatched = [
        key
        for key, expected in immutable_version_tags(manifest).items()
        if tags.get(key) != expected
    ]
    if mismatched:
        raise ApprovalError(
            f"registry version conflicts on: {', '.join(sorted(mismatched))}"
        )


def _get_or_create_experiment(client: MlflowClient, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    return experiment.experiment_id if experiment is not None else client.create_experiment(name)


def _get_alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        version = client.get_model_version_by_alias(name, alias)
    except MlflowException as error:
        alias_is_missing = (
            error.error_code == RESOURCE_DOES_NOT_EXIST
            or (
                error.error_code == INVALID_PARAMETER_VALUE
                and "alias" in str(error).lower()
                and "not found" in str(error).lower()
            )
        )
        if alias_is_missing:
            return None
        raise
    return str(version.version)


def _record_approval(
    client: MlflowClient,
    *,
    experiment_id: str,
    decision: ApprovalDecision,
) -> ApprovalResult:
    run = client.create_run(
        experiment_id,
        run_name=f"approve-{decision.model_release}",
        tags={
            "project": "predictive-maintenance-platform",
            "tracking_structure": "approval-decision",
            "purpose": "human-model-approval",
            "model_release": decision.model_release,
            "decision": decision.decision,
            "approver": decision.approver,
        },
    )
    run_id = run.info.run_id
    try:
        decision_bytes = (decision.model_dump_json(indent=2) + "\n").encode()
        decision_sha256 = hashlib.sha256(decision_bytes).hexdigest()
        with tempfile.TemporaryDirectory(prefix="model-approval-") as temp_directory:
            decision_path = Path(temp_directory) / "approval-decision.json"
            decision_path.write_bytes(decision_bytes)
            client.log_artifact(run_id, str(decision_path), artifact_path="decision")

        version_tags = {
            "approval_decision_run_id": run_id,
            "approval_record_sha256": decision_sha256,
            "approval_evidence_sha256": decision.approval_evidence_sha256,
            "approved_by": decision.approver,
            "approved_at": decision.decided_at.isoformat(),
            "approval_reason": decision.reason,
            "promotion_policy_version": decision.promotion_policy_version,
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(
                decision.registered_model_name,
                decision.model_version,
                key,
                value,
            )
        client.set_model_version_tag(
            decision.registered_model_name,
            decision.model_version,
            APPROVAL_STATUS_TAG,
            ModelApprovalStatus.APPROVED.value,
        )
        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        try:
            client.set_terminated(run_id, status="FAILED")
        except Exception:
            pass
        raise

    return ApprovalResult(
        registered_model_name=decision.registered_model_name,
        model_version=decision.model_version,
        model_release=decision.model_release,
        approval_status=ModelApprovalStatus.APPROVED.value,
        decision_run_id=run_id,
        already_approved=False,
    )
