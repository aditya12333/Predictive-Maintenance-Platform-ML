"""Audited MLflow alias promotion for approved model versions."""

import hashlib
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, ConfigDict

from predictive_maintenance.inference.artifacts import (
    ModelApprovalStatus,
    ModelArtifactManifest,
)
from predictive_maintenance.training.approval import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ApprovalError,
    load_candidate_manifest,
    verify_serving_package,
)
from predictive_maintenance.training.registry import (
    APPROVAL_STATUS_TAG,
    immutable_version_tags,
)


class PromotionError(RuntimeError):
    """Raised when an approved model cannot be promoted safely."""


class PromotionDecision(BaseModel):
    """Immutable record of one champion-alias decision."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["promoted"] = "promoted"
    registered_model_name: str
    model_version: str
    model_release: str
    promoted_by: str
    promoted_at: datetime
    reason: str
    approval_decision_run_id: str
    approval_record_sha256: str
    model_sha256: str
    evaluation_report_sha256: str
    old_champion_version: str | None
    old_previous_version: str | None
    new_champion_version: str
    new_previous_version: str | None


class PromotionResult(BaseModel):
    """Result returned after promotion or an idempotent repeat."""

    registered_model_name: str
    model_version: str
    model_release: str
    champion_version: str
    previous_version: str | None
    promotion_run_id: str
    already_promoted: bool


def promote_approved_model(
    *,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    model_version: str,
    manifest_path: Path,
    promoted_by: str,
    reason: str,
) -> PromotionResult:
    """Assign champion to an approved version and retain the old champion."""

    normalized_promoter = promoted_by.strip()
    normalized_reason = reason.strip()
    if not normalized_promoter:
        raise PromotionError("promoter must not be blank")
    if not normalized_reason:
        raise PromotionError("promotion reason must not be blank")

    try:
        manifest = load_candidate_manifest(manifest_path)
        verify_serving_package(manifest)
    except ApprovalError as error:
        raise PromotionError(str(error)) from error

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        target = client.get_model_version(registered_model_name, model_version)
        _verify_target(target, manifest=manifest)
        approval_run_id = target.tags.get("approval_decision_run_id")
        approval_record_sha256 = target.tags.get("approval_record_sha256")
        if not approval_run_id or not approval_record_sha256:
            raise PromotionError("approved version is missing its approval decision linkage")
        _verify_approval_record(
            client,
            run_id=approval_run_id,
            expected_sha256=approval_record_sha256,
        )

        old_champion = _get_alias_version(client, registered_model_name, "champion")
        old_previous = _get_alias_version(client, registered_model_name, "previous")
        if old_champion == str(target.version):
            promotion_run_id = target.tags.get("promotion_decision_run_id")
            if not promotion_run_id:
                raise PromotionError("champion alias has no promotion decision linkage")
            return PromotionResult(
                registered_model_name=registered_model_name,
                model_version=str(target.version),
                model_release=manifest.model_release,
                champion_version=str(target.version),
                previous_version=old_previous,
                promotion_run_id=promotion_run_id,
                already_promoted=True,
            )

        if old_champion is not None:
            current_champion = client.get_model_version(registered_model_name, old_champion)
            if current_champion.tags.get(APPROVAL_STATUS_TAG) != ModelApprovalStatus.APPROVED.value:
                raise PromotionError("current champion is not approved and cannot become previous")

        experiment_id = _get_or_create_experiment(client, experiment_name)
        decision = PromotionDecision(
            registered_model_name=registered_model_name,
            model_version=str(target.version),
            model_release=manifest.model_release,
            promoted_by=normalized_promoter,
            promoted_at=datetime.now(UTC),
            reason=normalized_reason,
            approval_decision_run_id=approval_run_id,
            approval_record_sha256=approval_record_sha256,
            model_sha256=manifest.model_sha256,
            evaluation_report_sha256=manifest.evaluation_report_sha256,
            old_champion_version=old_champion,
            old_previous_version=old_previous,
            new_champion_version=str(target.version),
            new_previous_version=old_champion,
        )
        return _record_and_apply_promotion(
            client,
            experiment_id=experiment_id,
            decision=decision,
        )
    except PromotionError:
        raise
    except Exception as error:
        raise PromotionError(f"could not promote model at {tracking_uri}") from error


def _verify_target(target: Any, *, manifest: ModelArtifactManifest) -> None:
    mismatched = [
        key
        for key, expected in immutable_version_tags(manifest).items()
        if target.tags.get(key) != expected
    ]
    if mismatched:
        raise PromotionError(
            f"registry version conflicts on: {', '.join(sorted(mismatched))}"
        )
    if target.tags.get(APPROVAL_STATUS_TAG) != ModelApprovalStatus.APPROVED.value:
        raise PromotionError("registry version is not approved")


def _verify_approval_record(
    client: MlflowClient,
    *,
    run_id: str,
    expected_sha256: str,
) -> None:
    approval_run = client.get_run(run_id)
    if approval_run.info.status != "FINISHED":
        raise PromotionError("approval decision run is not finished")
    with tempfile.TemporaryDirectory(prefix="verify-approval-") as temp_directory:
        downloaded_path = Path(
            client.download_artifacts(
                run_id,
                "decision/approval-decision.json",
                temp_directory,
            )
        )
        actual_sha256 = hashlib.sha256(downloaded_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise PromotionError("approval decision checksum does not match")


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


def _record_and_apply_promotion(
    client: MlflowClient,
    *,
    experiment_id: str,
    decision: PromotionDecision,
) -> PromotionResult:
    run = client.create_run(
        experiment_id,
        run_name=f"promote-{decision.model_release}",
        tags={
            "project": "predictive-maintenance-platform",
            "tracking_structure": "promotion-decision",
            "purpose": "champion-alias-promotion",
            "model_release": decision.model_release,
            "decision": decision.decision,
            "promoted_by": decision.promoted_by,
        },
    )
    run_id = run.info.run_id
    aliases_changed = False
    try:
        decision_bytes = (decision.model_dump_json(indent=2) + "\n").encode()
        decision_sha256 = hashlib.sha256(decision_bytes).hexdigest()
        with tempfile.TemporaryDirectory(prefix="model-promotion-") as temp_directory:
            decision_path = Path(temp_directory) / "promotion-decision.json"
            decision_path.write_bytes(decision_bytes)
            client.log_artifact(run_id, str(decision_path), artifact_path="decision")

        aliases_changed = True
        if decision.new_previous_version is not None:
            client.set_registered_model_alias(
                decision.registered_model_name,
                "previous",
                decision.new_previous_version,
            )
        client.set_registered_model_alias(
            decision.registered_model_name,
            "champion",
            decision.new_champion_version,
        )
        version_tags = {
            "promotion_decision_run_id": run_id,
            "promotion_record_sha256": decision_sha256,
            "promoted_by": decision.promoted_by,
            "promoted_at": decision.promoted_at.isoformat(),
            "promotion_reason": decision.reason,
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(
                decision.registered_model_name,
                decision.model_version,
                key,
                value,
            )
        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        if aliases_changed:
            _restore_alias(
                client,
                name=decision.registered_model_name,
                alias="champion",
                version=decision.old_champion_version,
            )
            _restore_alias(
                client,
                name=decision.registered_model_name,
                alias="previous",
                version=decision.old_previous_version,
            )
        try:
            client.set_terminated(run_id, status="FAILED")
        except Exception:
            pass
        raise

    return PromotionResult(
        registered_model_name=decision.registered_model_name,
        model_version=decision.model_version,
        model_release=decision.model_release,
        champion_version=decision.new_champion_version,
        previous_version=decision.new_previous_version,
        promotion_run_id=run_id,
        already_promoted=False,
    )


def _restore_alias(
    client: MlflowClient,
    *,
    name: str,
    alias: str,
    version: str | None,
) -> None:
    try:
        if version is None:
            client.delete_registered_model_alias(name, alias)
        else:
            client.set_registered_model_alias(name, alias, version)
    except Exception:
        pass
