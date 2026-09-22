"""Tests for idempotent MLflow registration of packaged candidates."""

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
)
from predictive_maintenance.training import registry


class RecordingClient:
    def __init__(self, *, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self.registered_model_created = False
        self.registration_run_created = False
        self.parameters: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.logged_artifacts: list[tuple[str, str]] = []
        self.version_request: dict[str, Any] | None = None
        self.run_status: str | None = None

    def get_registered_model(self, name: str) -> object:
        raise MlflowException("missing", error_code=3002)

    def create_registered_model(
        self,
        name: str,
        *,
        description: str,
        tags: dict[str, str],
    ) -> object:
        self.registered_model_created = True
        return SimpleNamespace(name=name, description=description, tags=tags)

    def search_model_versions(self, *, filter_string: str) -> list[object]:
        assert filter_string == "name = 'cmapss-fd001-rul'"
        return []

    def get_experiment_by_name(self, name: str) -> None:
        assert name == "cmapss-fd001-rul"
        return None

    def create_experiment(self, name: str) -> str:
        assert name == "cmapss-fd001-rul"
        return "experiment-1"

    def create_run(
        self,
        experiment_id: str,
        *,
        run_name: str,
        tags: dict[str, str],
    ) -> object:
        assert experiment_id == "experiment-1"
        assert run_name == "register-rul-lightgbm-v1"
        assert tags["approval_status"] == "candidate"
        self.registration_run_created = True
        return SimpleNamespace(info=SimpleNamespace(run_id="registration-run-1"))

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        assert run_id == "registration-run-1"
        self.parameters[key] = value

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        assert run_id == "registration-run-1"
        self.metrics[key] = value

    def log_artifacts(self, run_id: str, local_dir: str, *, artifact_path: str) -> None:
        assert run_id == "registration-run-1"
        self.logged_artifacts.append((Path(local_dir).name, artifact_path))

    def create_model_version(
        self,
        name: str,
        *,
        source: str,
        run_id: str,
        tags: dict[str, str],
        description: str,
    ) -> object:
        self.version_request = {
            "name": name,
            "source": source,
            "run_id": run_id,
            "tags": tags,
            "description": description,
        }
        return SimpleNamespace(version="1")

    def set_terminated(self, run_id: str, *, status: str) -> None:
        assert run_id == "registration-run-1"
        self.run_status = status


def _write_candidate_package(
    root: Path,
    *,
    status: ModelApprovalStatus = ModelApprovalStatus.CANDIDATE,
) -> Path:
    release_directory = root / "rul-lightgbm-v1"
    release_directory.mkdir()
    model = DummyRegressor(strategy="constant", constant=12.0)
    model.fit(np.zeros((2, len(FEATURE_NAMES))), np.asarray([10.0, 14.0]))
    model_path = release_directory / "model.joblib"
    joblib.dump(model, model_path)
    evaluation_path = release_directory / "evaluation.json"
    evaluation_path.write_text("{}\n", encoding="utf-8")

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
        evaluation_report_sha256=calculate_artifact_sha256(evaluation_path),
        validation_metrics=RegressionMetrics(
            model_name="lightgbm",
            sample_count=8,
            mae_cycles=10.0,
            rmse_cycles=12.0,
            nasa_score=30.0,
        ),
        official_test_metrics=RegressionMetrics(
            model_name="lightgbm",
            sample_count=3,
            mae_cycles=9.0,
            rmse_cycles=11.0,
            nasa_score=20.0,
        ),
        status=status,
    )
    manifest_path = release_directory / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest_path


def test_register_candidate_creates_unpromoted_registry_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_candidate_package(tmp_path)
    client = RecordingClient(tracking_uri="http://127.0.0.1:5000")
    monkeypatch.setattr(registry, "MlflowClient", lambda *, tracking_uri: client)

    def fake_create_model(
        manifest: ModelArtifactManifest,
        *,
        model_directory: Path,
    ) -> None:
        assert manifest.model_release == "rul-lightgbm-v1"
        model_directory.mkdir()
        (model_directory / "MLmodel").write_text("model\n", encoding="utf-8")

    monkeypatch.setattr(registry, "_create_mlflow_model_directory", fake_create_model)

    result = registry.register_candidate(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="cmapss-fd001-rul",
        registered_model_name="cmapss-fd001-rul",
        manifest_path=manifest_path,
    )

    assert result.model_version == "1"
    assert result.approval_status == "candidate"
    assert result.already_registered is False
    assert client.registered_model_created is True
    assert client.run_status == "FINISHED"
    assert client.version_request is not None
    assert client.version_request["source"] == "runs:/registration-run-1/registry-model"
    assert client.version_request["tags"]["approval_status"] == "candidate"
    assert client.version_request["tags"]["model_release"] == "rul-lightgbm-v1"
    assert client.metrics["official_test_mae_cycles"] == 9.0
    assert {artifact_path for _, artifact_path in client.logged_artifacts} == {
        "candidate-package",
        "registry-model",
    }


def test_register_candidate_is_idempotent_for_matching_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_candidate_package(tmp_path)
    manifest = registry.load_model_manifest(manifest_path)
    client = RecordingClient(tracking_uri="http://127.0.0.1:5000")
    client.get_registered_model = lambda name: SimpleNamespace(name=name)  # type: ignore[method-assign]
    existing_tags = {
        **registry.immutable_version_tags(manifest),
        "approval_status": "approved",
    }
    client.search_model_versions = lambda **kwargs: [  # type: ignore[method-assign]
        SimpleNamespace(version="3", tags=existing_tags, run_id="existing-run")
    ]
    monkeypatch.setattr(registry, "MlflowClient", lambda *, tracking_uri: client)

    result = registry.register_candidate(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="cmapss-fd001-rul",
        registered_model_name="cmapss-fd001-rul",
        manifest_path=manifest_path,
    )

    assert result.model_version == "3"
    assert result.approval_status == "approved"
    assert result.already_registered is True
    assert client.registration_run_created is False


def test_register_candidate_rejects_conflicting_existing_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_candidate_package(tmp_path)
    manifest = registry.load_model_manifest(manifest_path)
    client = RecordingClient(tracking_uri="http://127.0.0.1:5000")
    client.get_registered_model = lambda name: SimpleNamespace(name=name)  # type: ignore[method-assign]
    conflicting_tags = {
        **registry.immutable_version_tags(manifest),
        "model_sha256": "f" * 64,
        "approval_status": "candidate",
    }
    client.search_model_versions = lambda **kwargs: [  # type: ignore[method-assign]
        SimpleNamespace(version="1", tags=conflicting_tags, run_id="conflicting-run")
    ]
    monkeypatch.setattr(registry, "MlflowClient", lambda *, tracking_uri: client)

    with pytest.raises(registry.CandidateRegistrationError, match="model_sha256"):
        registry.register_candidate(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="cmapss-fd001-rul",
            registered_model_name="cmapss-fd001-rul",
            manifest_path=manifest_path,
        )

    assert client.registration_run_created is False


def test_register_candidate_rejects_non_candidate_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_candidate_package(
        tmp_path,
        status=ModelApprovalStatus.APPROVED,
    )
    monkeypatch.setattr(
        registry,
        "MlflowClient",
        lambda **kwargs: pytest.fail("MLflow must not be called"),
    )

    with pytest.raises(registry.CandidateRegistrationError, match="only candidates"):
        registry.register_candidate(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="cmapss-fd001-rul",
            registered_model_name="cmapss-fd001-rul",
            manifest_path=manifest_path,
        )
