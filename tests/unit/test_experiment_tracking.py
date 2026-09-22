"""Tests for the MLflow experiment-tracking boundary."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from predictive_maintenance.training import tracking
from predictive_maintenance.training.evaluate import ModelEvaluation
from predictive_maintenance.training.train import TrainingRunReport


class AvailableClient:
    def __init__(self, *, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def get_experiment_by_name(self, name: str) -> object:
        return SimpleNamespace(name=name)


class MissingExperimentClient(AvailableClient):
    def get_experiment_by_name(self, name: str) -> None:
        return None


class UnavailableClient(AvailableClient):
    def get_experiment_by_name(self, name: str) -> object:
        raise ConnectionError("server unavailable")


class RecordingClient:
    instances: list["RecordingClient"] = []

    def __init__(self, *, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self.parameters: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.tags: dict[str, str] = {}
        self.artifacts: list[tuple[str, str | None]] = []
        self.parameters_by_run: dict[str, dict[str, Any]] = {"run-1": self.parameters}
        self.metrics_by_run: dict[str, dict[str, float]] = {"run-1": self.metrics}
        self.tags_by_run: dict[str, dict[str, str]] = {"run-1": self.tags}
        self.artifacts_by_run: dict[str, list[tuple[str, str | None]]] = {
            "run-1": self.artifacts
        }
        self.status_by_run: dict[str, str] = {}
        self.created_run_names: list[str] = []
        self.terminated_status: str | None = None
        self.__class__.instances.append(self)

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
        run_id = f"run-{len(self.created_run_names) + 1}"
        self.created_run_names.append(run_name)
        self.parameters_by_run.setdefault(run_id, {})
        self.metrics_by_run.setdefault(run_id, {})
        self.tags_by_run.setdefault(run_id, {}).update(tags)
        self.artifacts_by_run.setdefault(run_id, [])
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        self.parameters_by_run[run_id][key] = value

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        self.metrics_by_run[run_id][key] = value

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags_by_run[run_id][key] = value

    def log_artifact(
        self,
        run_id: str,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None:
        self.artifacts_by_run[run_id].append((Path(local_path).name, artifact_path))

    def set_terminated(self, run_id: str, *, status: str) -> None:
        self.status_by_run[run_id] = status
        if run_id == "run-1":
            self.terminated_status = status


class FailingLogClient(RecordingClient):
    def log_metric(self, run_id: str, key: str, value: float) -> None:
        raise ConnectionError("tracking server stopped")


def _training_report() -> TrainingRunReport:
    validation_results = (
        ModelEvaluation("lightgbm", 8, 1.0, 2.0, 3.0),
        ModelEvaluation("linear_regression", 8, 4.0, 5.0, 6.0),
    )
    return TrainingRunReport(
        run_name="comparison-1",
        generated_at=datetime(2026, 9, 22, tzinfo=UTC).isoformat(),
        dataset_version="nasa-cmapss-classic:v1:FD001",
        dataset_manifest_sha256="a" * 64,
        feature_version="features-v1",
        feature_names=("cycle", "sensor_1"),
        target_definition="max_cycle_for_engine - current_cycle",
        prediction_postprocessing="max(0, predicted_rul)",
        random_seed=42,
        validation_fraction=0.2,
        training_engine_ids=(1, 2, 3, 4),
        validation_engine_ids=(5,),
        training_rows=32,
        validation_rows=8,
        selection_policy="lowest validation MAE",
        validation_ranking=validation_results,
        selected_model_name="lightgbm",
        official_test_evaluation=ModelEvaluation("lightgbm", 3, 1.5, 2.5, 4.0),
    )


def _publish_training_artifacts(run_directory: Path) -> None:
    run_directory.mkdir()
    (run_directory / "evaluation.json").write_text("{}\n", encoding="utf-8")
    (run_directory / "selected-model.joblib").write_bytes(b"model")


def test_tracking_check_reports_existing_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracking, "MlflowClient", AvailableClient)

    result = tracking.check_tracking_connection(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="cmapss-fd001-rul",
    )

    assert result.experiment_exists is True


def test_tracking_check_allows_experiment_to_be_created_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracking, "MlflowClient", MissingExperimentClient)

    result = tracking.check_tracking_connection(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="cmapss-fd001-rul",
    )

    assert result.experiment_exists is False


def test_tracking_check_converts_client_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracking, "MlflowClient", UnavailableClient)

    with pytest.raises(tracking.ExperimentTrackingError, match="could not connect"):
        tracking.check_tracking_connection(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="cmapss-fd001-rul",
        )


def test_parent_run_logs_report_and_finishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    RecordingClient.instances.clear()
    monkeypatch.setattr(tracking, "MlflowClient", RecordingClient)
    run_directory = tmp_path / "comparison-1"
    _publish_training_artifacts(run_directory)

    with tracking.start_comparison_run(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="cmapss-fd001-rul",
        run_name="comparison-1",
    ) as tracker:
        tracker.log_training_report(
            report=_training_report(),
            run_directory=run_directory,
        )

    client = RecordingClient.instances[-1]
    assert client.terminated_status == "FINISHED"
    assert client.parameters["training_engine_count"] == 4
    assert client.parameters["validation_engine_count"] == 1
    assert client.parameters["feature_names"] == "cycle,sensor_1"
    assert client.metrics["validation_lightgbm_mae_cycles"] == 1.0
    assert client.metrics["validation_linear_regression_nasa_score"] == 6.0
    assert client.metrics["official_test_rmse_cycles"] == 2.5
    assert client.tags["tracking_structure"] == "parent"
    assert client.tags["selected_model"] == "lightgbm"
    assert client.artifacts == [
        ("evaluation.json", "training-run"),
        ("selected-model.joblib", "training-run"),
    ]


def test_candidate_runs_are_linked_to_parent_and_log_validation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RecordingClient.instances.clear()
    monkeypatch.setattr(tracking, "MlflowClient", RecordingClient)
    monkeypatch.setattr(
        tracking,
        "create_candidate_models",
        lambda *, random_seed: {
            "lightgbm": Pipeline(
                [("scaler", StandardScaler()), ("model", DummyRegressor(strategy="mean"))]
            ),
            "linear_regression": DummyRegressor(strategy="median"),
        },
    )

    with tracking.start_comparison_run(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="cmapss-fd001-rul",
        run_name="comparison-1",
    ) as tracker:
        tracker.log_candidate_runs(report=_training_report())

    client = RecordingClient.instances[-1]
    assert client.created_run_names == [
        "comparison-1",
        "comparison-1.lightgbm",
        "comparison-1.linear_regression",
    ]
    assert client.tags_by_run["run-2"]["mlflow.parentRunId"] == "run-1"
    assert client.tags_by_run["run-2"]["selected_model"] == "true"
    assert client.tags_by_run["run-3"]["selected_model"] == "false"
    assert client.parameters_by_run["run-2"]["model_model__strategy"] == "mean"
    assert client.parameters_by_run["run-2"]["model_scaler_class"].endswith(
        ".StandardScaler"
    )
    assert client.parameters_by_run["run-2"]["model_model_class"].endswith(
        ".DummyRegressor"
    )
    assert client.parameters_by_run["run-3"]["validation_rows"] == 8
    assert client.metrics_by_run["run-2"] == {
        "validation_mae_cycles": 1.0,
        "validation_rmse_cycles": 2.0,
        "validation_nasa_score": 3.0,
    }
    assert "official_test_mae_cycles" not in client.metrics_by_run["run-2"]
    assert client.status_by_run == {
        "run-1": "FINISHED",
        "run-2": "FINISHED",
        "run-3": "FINISHED",
    }


def test_parent_run_marks_training_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RecordingClient.instances.clear()
    monkeypatch.setattr(tracking, "MlflowClient", RecordingClient)

    with pytest.raises(RuntimeError, match="training stopped"):
        with tracking.start_comparison_run(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="cmapss-fd001-rul",
            run_name="comparison-1",
        ):
            raise RuntimeError("training stopped")

    assert RecordingClient.instances[-1].terminated_status == "FAILED"


def test_parent_run_converts_logging_failure_and_marks_run_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FailingLogClient.instances.clear()
    monkeypatch.setattr(tracking, "MlflowClient", FailingLogClient)
    run_directory = tmp_path / "comparison-1"
    _publish_training_artifacts(run_directory)

    with pytest.raises(tracking.ExperimentTrackingError, match="could not log"):
        with tracking.start_comparison_run(
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="cmapss-fd001-rul",
            run_name="comparison-1",
        ) as tracker:
            tracker.log_training_report(
                report=_training_report(),
                run_directory=run_directory,
            )

    assert FailingLogClient.instances[-1].terminated_status == "FAILED"
