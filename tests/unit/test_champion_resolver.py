"""Tests for approved MLflow champion resolution and immutable local caching."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from predictive_maintenance.features.contracts import FEATURE_NAMES, FeatureVector
from predictive_maintenance.inference import registry
from predictive_maintenance.inference.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ModelApprovalStatus,
    ModelArtifactManifest,
    RegressionMetrics,
    calculate_artifact_sha256,
    load_model_manifest,
)
from predictive_maintenance.training.registry import immutable_version_tags


def _write_candidate(root: Path) -> Path:
    package = root / "candidate-package"
    package.mkdir(parents=True)
    model = DummyRegressor(strategy="constant", constant=17.0)
    model.fit(np.zeros((2, len(FEATURE_NAMES))), np.asarray([10.0, 20.0]))
    model_path = package / "model.joblib"
    joblib.dump(model, model_path)
    report_path = package / "evaluation.json"
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
    manifest_path = package / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest_path


class ChampionClient:
    def __init__(self, manifest_path: Path) -> None:
        self.package = manifest_path.parent
        manifest = load_model_manifest(manifest_path)
        approval = {
            "decision": "approved",
            "registered_model_name": "cmapss-fd001-rul",
            "model_version": "1",
        }
        self.approval_path = manifest_path.parent.parent / "approval-decision.json"
        self.approval_path.write_text(json.dumps(approval) + "\n", encoding="utf-8")
        approval_sha = calculate_artifact_sha256(self.approval_path)
        promotion = {
            "decision": "promoted",
            "registered_model_name": "cmapss-fd001-rul",
            "model_version": "1",
            "new_champion_version": "1",
            "approval_decision_run_id": "approval-run",
            "approval_record_sha256": approval_sha,
        }
        self.promotion_path = manifest_path.parent.parent / "promotion-decision.json"
        self.promotion_path.write_text(json.dumps(promotion) + "\n", encoding="utf-8")
        self.tags = {
            **immutable_version_tags(manifest),
            "approval_status": "approved",
            "approval_decision_run_id": "approval-run",
            "approval_record_sha256": approval_sha,
            "promotion_decision_run_id": "promotion-run",
            "promotion_record_sha256": calculate_artifact_sha256(self.promotion_path),
        }
        self.package_downloads = 0

    def get_model_version_by_alias(self, name: str, alias: str) -> object:
        assert name == "cmapss-fd001-rul"
        assert alias == "champion"
        return SimpleNamespace(version="1", run_id="source-run", tags=self.tags)

    def get_run(self, run_id: str) -> object:
        assert run_id in {"source-run", "approval-run", "promotion-run"}
        return SimpleNamespace(info=SimpleNamespace(status="FINISHED"))

    def download_artifacts(self, run_id: str, path: str, dst_path: str) -> str:
        if run_id == "approval-run":
            return str(self.approval_path)
        if run_id == "promotion-run":
            return str(self.promotion_path)
        assert run_id == "source-run"
        assert path == "candidate-package"
        self.package_downloads += 1
        destination = Path(dst_path) / "candidate-package"
        shutil.copytree(self.package, destination)
        return str(destination)


def _features() -> FeatureVector:
    return FeatureVector(
        feature_version="features-v1",
        feature_names=FEATURE_NAMES,
        values=tuple(0.0 for _ in FEATURE_NAMES),
    )


def test_resolver_verifies_champion_and_reuses_versioned_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = ChampionClient(_write_candidate(tmp_path / "source"))
    monkeypatch.setattr(registry, "MlflowClient", lambda *, tracking_uri: client)
    cache_root = tmp_path / "cache"

    first = registry.resolve_champion(
        tracking_uri="http://127.0.0.1:5000",
        registered_model_name="cmapss-fd001-rul",
        cache_root=cache_root,
    )
    second = registry.resolve_champion(
        tracking_uri="http://127.0.0.1:5000",
        registered_model_name="cmapss-fd001-rul",
        cache_root=cache_root,
    )

    assert first.model_version == "1"
    assert first.model_release == "rul-lightgbm-v1"
    assert first.predictor.predict(_features()) == 17.0
    assert second.cache_directory == first.cache_directory
    assert client.package_downloads == 1


def test_resolver_rejects_unapproved_champion_before_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = ChampionClient(_write_candidate(tmp_path / "source"))
    client.tags["approval_status"] = "candidate"
    monkeypatch.setattr(registry, "MlflowClient", lambda *, tracking_uri: client)

    with pytest.raises(registry.ChampionResolutionError, match="not approved"):
        registry.resolve_champion(
            tracking_uri="http://127.0.0.1:5000",
            registered_model_name="cmapss-fd001-rul",
            cache_root=tmp_path / "cache",
        )
    assert client.package_downloads == 0


def test_resolver_rejects_modified_cached_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = ChampionClient(_write_candidate(tmp_path / "source"))
    monkeypatch.setattr(registry, "MlflowClient", lambda *, tracking_uri: client)
    champion = registry.resolve_champion(
        tracking_uri="http://127.0.0.1:5000",
        registered_model_name="cmapss-fd001-rul",
        cache_root=tmp_path / "cache",
    )
    (champion.cache_directory / "model.joblib").write_bytes(b"tampered")

    with pytest.raises(registry.ChampionResolutionError, match="integrity checks"):
        registry.resolve_champion(
            tracking_uri="http://127.0.0.1:5000",
            registered_model_name="cmapss-fd001-rul",
            cache_root=tmp_path / "cache",
        )


def test_resolver_rejects_modified_promotion_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = ChampionClient(_write_candidate(tmp_path / "source"))
    client.promotion_path.write_text('{"decision":"modified"}\n', encoding="utf-8")
    monkeypatch.setattr(registry, "MlflowClient", lambda *, tracking_uri: client)

    with pytest.raises(registry.ChampionResolutionError, match="checksum"):
        registry.resolve_champion(
            tracking_uri="http://127.0.0.1:5000",
            registered_model_name="cmapss-fd001-rul",
            cache_root=tmp_path / "cache",
        )
