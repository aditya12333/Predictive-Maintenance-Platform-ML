"""Tests for audited champion-alias promotion."""

import hashlib
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
from predictive_maintenance.training import promotion
from predictive_maintenance.training.registry import immutable_version_tags


class PromotionClient:
    def __init__(self, manifest: ModelArtifactManifest, approval_path: Path) -> None:
        approval_sha256 = hashlib.sha256(approval_path.read_bytes()).hexdigest()
        self.target_tags = {
            **immutable_version_tags(manifest),
            "approval_status": "approved",
            "approval_decision_run_id": "approval-run-1",
            "approval_record_sha256": approval_sha256,
        }
        self.aliases: dict[str, str] = {}
        self.approval_path = approval_path
        self.promotion_runs = 0
        self.promotion_json: dict[str, Any] | None = None
        self.run_status: str | None = None

    def get_model_version(self, name: str, version: str) -> object:
        if version == "1":
            return SimpleNamespace(version="1", tags=self.target_tags)
        return SimpleNamespace(version=version, tags={"approval_status": "approved"})

    def get_run(self, run_id: str) -> object:
        assert run_id == "approval-run-1"
        return SimpleNamespace(info=SimpleNamespace(status="FINISHED"))

    def download_artifacts(self, run_id: str, path: str, dst_path: str) -> str:
        assert path == "decision/approval-decision.json"
        return str(self.approval_path)

    def get_model_version_by_alias(self, name: str, alias: str) -> object:
        version = self.aliases.get(alias)
        if version is None:
            raise MlflowException(
                f"Registered model alias {alias} not found.",
                error_code=1000,
            )
        return SimpleNamespace(version=version)

    def get_experiment_by_name(self, name: str) -> object:
        return SimpleNamespace(experiment_id="experiment-1")

    def create_run(
        self,
        experiment_id: str,
        *,
        run_name: str,
        tags: dict[str, str],
    ) -> object:
        assert run_name == "promote-rul-lightgbm-v1"
        assert tags["decision"] == "promoted"
        self.promotion_runs += 1
        return SimpleNamespace(info=SimpleNamespace(run_id="promotion-run-1"))

    def log_artifact(
        self,
        run_id: str,
        local_path: str,
        *,
        artifact_path: str,
    ) -> None:
        self.promotion_json = json.loads(Path(local_path).read_text(encoding="utf-8"))

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = version

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        self.aliases.pop(alias, None)

    def set_model_version_tag(
        self,
        name: str,
        version: str,
        key: str,
        value: str,
    ) -> None:
        self.target_tags[key] = value

    def set_terminated(self, run_id: str, *, status: str) -> None:
        self.run_status = status


def _write_candidate(root: Path) -> Path:
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
        sample_count=2,
        mae_cycles=1.0,
        rmse_cycles=2.0,
        nasa_score=3.0,
    )
    manifest = ModelArtifactManifest(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        model_release="rul-lightgbm-v1",
        model_name="lightgbm",
        source_training_run="comparison-v1",
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


def _approval_record(root: Path) -> Path:
    path = root / "approval-decision.json"
    path.write_text('{"decision":"approved"}\n', encoding="utf-8")
    return path


def test_first_promotion_assigns_only_champion_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_candidate(tmp_path)
    client = PromotionClient(load_model_manifest(manifest_path), _approval_record(tmp_path))
    monkeypatch.setattr(promotion, "MlflowClient", lambda *, tracking_uri: client)

    result = promotion.promote_approved_model(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="cmapss-fd001-rul",
        registered_model_name="cmapss-fd001-rul",
        model_version="1",
        manifest_path=manifest_path,
        promoted_by="Aditya Pandey",
        reason="First local champion.",
    )

    assert result.champion_version == "1"
    assert result.previous_version is None
    assert client.aliases == {"champion": "1"}
    assert client.run_status == "FINISHED"
    assert client.promotion_json is not None
    assert client.promotion_json["old_champion_version"] is None

    repeated = promotion.promote_approved_model(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="cmapss-fd001-rul",
        registered_model_name="cmapss-fd001-rul",
        model_version="1",
        manifest_path=manifest_path,
        promoted_by="Aditya Pandey",
        reason="First local champion.",
    )
    assert repeated.already_promoted is True
    assert client.promotion_runs == 1


def test_replacement_moves_existing_champion_to_previous(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_candidate(tmp_path)
    client = PromotionClient(load_model_manifest(manifest_path), _approval_record(tmp_path))
    client.aliases = {"champion": "2", "previous": "0"}
    monkeypatch.setattr(promotion, "MlflowClient", lambda *, tracking_uri: client)

    result = promotion.promote_approved_model(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="cmapss-fd001-rul",
        registered_model_name="cmapss-fd001-rul",
        model_version="1",
        manifest_path=manifest_path,
        promoted_by="Aditya Pandey",
        reason="Replace current champion.",
    )

    assert result.previous_version == "2"
    assert client.aliases == {"champion": "1", "previous": "2"}


def test_promotion_rejects_unapproved_target_before_alias_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_candidate(tmp_path)
    client = PromotionClient(load_model_manifest(manifest_path), _approval_record(tmp_path))
    client.target_tags["approval_status"] = "candidate"
    monkeypatch.setattr(promotion, "MlflowClient", lambda *, tracking_uri: client)

    with pytest.raises(promotion.PromotionError, match="not approved"):
        promotion.promote_approved_model(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="cmapss-fd001-rul",
            registered_model_name="cmapss-fd001-rul",
            model_version="1",
            manifest_path=manifest_path,
            promoted_by="Aditya Pandey",
            reason="Invalid attempt.",
        )

    assert client.aliases == {}
