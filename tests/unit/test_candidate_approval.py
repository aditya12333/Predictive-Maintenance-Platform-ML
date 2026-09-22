"""Tests for explicit human approval of registered model candidates."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pytest
from mlflow.exceptions import MlflowException
from sklearn.dummy import DummyRegressor

from predictive_maintenance.features.contracts import FEATURE_NAMES
from predictive_maintenance.inference.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ModelApprovalStatus,
    ModelArtifactManifest,
    RegressionMetrics,
    calculate_artifact_sha256,
    load_model_manifest,
)
from predictive_maintenance.training import approval
from predictive_maintenance.training.registry import immutable_version_tags


class ApprovalClient:
    def __init__(
        self,
        manifest: ModelArtifactManifest,
        *,
        missing_alias_error_code: int = 3002,
    ) -> None:
        self.version_tags = {
            **immutable_version_tags(manifest),
            "approval_status": "candidate",
        }
        self.version_tag_updates: dict[str, str] = {}
        self.decision_json: dict[str, Any] | None = None
        self.run_status: str | None = None
        self.run_created = False
        self.missing_alias_error_code = missing_alias_error_code

    def get_model_version(self, name: str, version: str) -> object:
        assert name == "cmapss-fd001-rul"
        assert version == "1"
        return SimpleNamespace(version="1", tags=self.version_tags)

    def get_experiment_by_name(self, name: str) -> object:
        return SimpleNamespace(experiment_id="experiment-1")

    def get_model_version_by_alias(self, name: str, alias: str) -> object:
        assert alias == "champion"
        raise MlflowException(
            "Registered model alias champion not found.",
            error_code=self.missing_alias_error_code,
        )

    def create_run(
        self,
        experiment_id: str,
        *,
        run_name: str,
        tags: dict[str, str],
    ) -> object:
        assert experiment_id == "experiment-1"
        assert run_name == "approve-rul-lightgbm-v1"
        assert tags["decision"] == "approved"
        self.run_created = True
        return SimpleNamespace(info=SimpleNamespace(run_id="approval-run-1"))

    def log_artifact(
        self,
        run_id: str,
        local_path: str,
        *,
        artifact_path: str,
    ) -> None:
        assert run_id == "approval-run-1"
        assert artifact_path == "decision"
        self.decision_json = json.loads(Path(local_path).read_text(encoding="utf-8"))

    def set_model_version_tag(
        self,
        name: str,
        version: str,
        key: str,
        value: str,
    ) -> None:
        assert name == "cmapss-fd001-rul"
        assert version == "1"
        self.version_tag_updates[key] = value
        self.version_tags[key] = value

    def set_terminated(self, run_id: str, *, status: str) -> None:
        assert run_id == "approval-run-1"
        self.run_status = status


def _write_candidate_package(root: Path) -> Path:
    release_directory = root / "rul-lightgbm-v1"
    release_directory.mkdir()
    model = DummyRegressor(strategy="constant", constant=12.0)
    model.fit(np.zeros((2, len(FEATURE_NAMES))), np.asarray([10.0, 14.0]))
    model_path = release_directory / "model.joblib"
    joblib.dump(model, model_path)
    report_path = release_directory / "evaluation.json"
    report_path.write_text("{}\n", encoding="utf-8")
    metrics = RegressionMetrics(
        model_name="lightgbm",
        sample_count=8,
        mae_cycles=10.0,
        rmse_cycles=12.0,
        nasa_score=30.0,
    )
    manifest = ModelArtifactManifest(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        model_release="rul-lightgbm-v1",
        model_name="lightgbm",
        source_training_run="fd001-comparison-v1",
        created_at=datetime(2026, 9, 22, tzinfo=UTC),
        feature_version="features-v1",
        feature_names=FEATURE_NAMES,
        training_dataset_version="nasa-cmapss-classic:v1:FD001",
        dataset_manifest_sha256="a" * 64,
        target_definition="max_cycle_for_engine - current_cycle",
        prediction_postprocessing="max(0, predicted_rul)",
        model_path=Path("model.joblib"),
        model_sha256=calculate_artifact_sha256(model_path),
        evaluation_report_path=Path("evaluation.json"),
        evaluation_report_sha256=calculate_artifact_sha256(report_path),
        validation_metrics=metrics,
        official_test_metrics=metrics,
        status=ModelApprovalStatus.CANDIDATE,
    )
    manifest_path = release_directory / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _write_evidence(root: Path, *, blocked_gate: str | None = None) -> Path:
    gate_names = (
        "provenance_integrity",
        "serving_compatibility",
        "prediction_validity",
        "metric_validity",
        "performance_policy",
        "operational_readiness",
        "automated_checks",
    )
    evidence = {
        "schema_version": "approval-evidence-v1",
        "promotion_policy_version": "model-promotion-policy-v1",
        "model_release": "rul-lightgbm-v1",
        "assessed_at": "2026-09-22T14:00:00Z",
        "assessor": "reviewer@example.com",
        **{
            gate_name: {
                "status": "pending" if gate_name == blocked_gate else "passed",
                "details": f"{gate_name} reviewed",
                "evidence_reference": f"evidence://{gate_name}",
            }
            for gate_name in gate_names
        },
    }
    evidence_path = root / "approval-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence_path


def test_approval_records_decision_without_assigning_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_candidate_package(tmp_path)
    evidence_path = _write_evidence(tmp_path)
    manifest = load_model_manifest(manifest_path)
    client = ApprovalClient(manifest)
    monkeypatch.setattr(approval, "MlflowClient", lambda *, tracking_uri: client)

    result = approval.approve_candidate(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="cmapss-fd001-rul",
        registered_model_name="cmapss-fd001-rul",
        model_version="1",
        manifest_path=manifest_path,
        evidence_path=evidence_path,
        approver=" approver@example.com ",
        reason=" Reviewed all required evidence. ",
    )

    assert result.approval_status == "approved"
    assert result.already_approved is False
    assert result.decision_run_id == "approval-run-1"
    assert client.run_status == "FINISHED"
    assert client.version_tag_updates["approval_status"] == "approved"
    assert client.version_tag_updates["approved_by"] == "approver@example.com"
    assert client.decision_json is not None
    assert client.decision_json["current_champion_version"] is None
    assert client.decision_json["intended_previous_version"] is None
    assert load_model_manifest(manifest_path).status is ModelApprovalStatus.CANDIDATE


def test_approval_accepts_mlflow_invalid_parameter_response_for_missing_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_candidate_package(tmp_path)
    evidence_path = _write_evidence(tmp_path)
    manifest = load_model_manifest(manifest_path)
    client = ApprovalClient(manifest, missing_alias_error_code=1000)
    monkeypatch.setattr(approval, "MlflowClient", lambda *, tracking_uri: client)

    result = approval.approve_candidate(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="cmapss-fd001-rul",
        registered_model_name="cmapss-fd001-rul",
        model_version="1",
        manifest_path=manifest_path,
        evidence_path=evidence_path,
        approver="approver@example.com",
        reason="Review complete.",
    )

    assert result.approval_status == "approved"


def test_approval_blocks_pending_gate_before_registry_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_candidate_package(tmp_path)
    evidence_path = _write_evidence(tmp_path, blocked_gate="operational_readiness")
    monkeypatch.setattr(
        approval,
        "MlflowClient",
        lambda **kwargs: pytest.fail("registry must not be mutated"),
    )

    with pytest.raises(approval.ApprovalError, match="operational_readiness"):
        approval.approve_candidate(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="cmapss-fd001-rul",
            registered_model_name="cmapss-fd001-rul",
            model_version="1",
            manifest_path=manifest_path,
            evidence_path=evidence_path,
            approver="approver@example.com",
            reason="Review complete.",
        )


def test_approval_is_idempotent_for_existing_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_candidate_package(tmp_path)
    evidence_path = _write_evidence(tmp_path)
    manifest = load_model_manifest(manifest_path)
    client = ApprovalClient(manifest)
    client.version_tags.update(
        {
            "approval_status": "approved",
            "approval_decision_run_id": "existing-approval-run",
        }
    )
    monkeypatch.setattr(approval, "MlflowClient", lambda *, tracking_uri: client)

    result = approval.approve_candidate(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="cmapss-fd001-rul",
        registered_model_name="cmapss-fd001-rul",
        model_version="1",
        manifest_path=manifest_path,
        evidence_path=evidence_path,
        approver="approver@example.com",
        reason="Review complete.",
    )

    assert result.already_approved is True
    assert result.decision_run_id == "existing-approval-run"
    assert client.run_created is False
