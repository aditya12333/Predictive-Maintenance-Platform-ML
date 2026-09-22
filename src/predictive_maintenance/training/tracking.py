"""MLflow boundary used by training commands."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mlflow import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from predictive_maintenance.training.models import create_candidate_models

if TYPE_CHECKING:
    from predictive_maintenance.training.evaluate import ModelEvaluation
    from predictive_maintenance.training.train import TrainingRunReport


class ExperimentTrackingError(RuntimeError):
    """Raised when the configured MLflow service cannot be reached safely."""


@dataclass(frozen=True)
class TrackingConnection:
    """Result of checking one configured tracking destination."""

    tracking_uri: str
    experiment_name: str
    experiment_exists: bool


@dataclass(frozen=True)
class ComparisonRunTracker:
    """Record one complete model-comparison execution in an MLflow parent run."""

    client: MlflowClient
    experiment_id: str
    run_id: str
    run_name: str

    def log_candidate_runs(self, *, report: "TrainingRunReport") -> None:
        """Create one child run for every completed validation candidate."""

        models = create_candidate_models(random_seed=report.random_seed)
        for evaluation in report.validation_ranking:
            model = models.get(evaluation.model_name)
            if model is None:
                raise ExperimentTrackingError(
                    f"no model definition exists for {evaluation.model_name}"
                )
            self._log_candidate_run(
                report=report,
                evaluation=evaluation,
                model=model,
            )

    def log_training_report(
        self,
        *,
        report: "TrainingRunReport",
        run_directory: Path,
    ) -> None:
        """Log lineage, comparison metrics, selection outcome, and local artifacts."""

        artifact_paths = (
            run_directory / "evaluation.json",
            run_directory / "selected-model.joblib",
        )
        missing_artifacts = [path.name for path in artifact_paths if not path.is_file()]
        if missing_artifacts:
            missing = ", ".join(missing_artifacts)
            raise ExperimentTrackingError(f"training run is missing artifacts: {missing}")

        parameters: dict[str, str | int | float] = {
            "random_seed": report.random_seed,
            "validation_fraction": report.validation_fraction,
            "training_engine_count": len(report.training_engine_ids),
            "validation_engine_count": len(report.validation_engine_ids),
            "training_rows": report.training_rows,
            "validation_rows": report.validation_rows,
            "official_test_sample_count": report.official_test_evaluation.sample_count,
            "dataset_version": report.dataset_version,
            "feature_version": report.feature_version,
            "feature_names": ",".join(report.feature_names),
            "target_definition": report.target_definition,
            "prediction_postprocessing": report.prediction_postprocessing,
            "selection_policy": report.selection_policy,
        }
        tags = {
            "selected_model": report.selected_model_name,
            "dataset_manifest_sha256": report.dataset_manifest_sha256,
            "generated_at": report.generated_at,
        }

        try:
            for key, value in parameters.items():
                self.client.log_param(self.run_id, key, value)
            for key, value in tags.items():
                self.client.set_tag(self.run_id, key, value)

            for evaluation in report.validation_ranking:
                prefix = f"validation_{evaluation.model_name}"
                self.client.log_metric(
                    self.run_id,
                    f"{prefix}_mae_cycles",
                    evaluation.mae_cycles,
                )
                self.client.log_metric(
                    self.run_id,
                    f"{prefix}_rmse_cycles",
                    evaluation.rmse_cycles,
                )
                self.client.log_metric(
                    self.run_id,
                    f"{prefix}_nasa_score",
                    evaluation.nasa_score,
                )

            official = report.official_test_evaluation
            self.client.log_metric(
                self.run_id,
                "official_test_mae_cycles",
                official.mae_cycles,
            )
            self.client.log_metric(
                self.run_id,
                "official_test_rmse_cycles",
                official.rmse_cycles,
            )
            self.client.log_metric(
                self.run_id,
                "official_test_nasa_score",
                official.nasa_score,
            )
            for artifact_path in artifact_paths:
                self.client.log_artifact(
                    self.run_id,
                    str(artifact_path),
                    artifact_path="training-run",
                )
        except Exception as error:
            raise ExperimentTrackingError(
                f"could not log training run {report.run_name} to MLflow"
            ) from error

    def _log_candidate_run(
        self,
        *,
        report: "TrainingRunReport",
        evaluation: "ModelEvaluation",
        model: object,
    ) -> None:
        child_run_id: str | None = None
        try:
            child_run = self.client.create_run(
                self.experiment_id,
                run_name=f"{self.run_name}.{evaluation.model_name}",
                tags={
                    MLFLOW_PARENT_RUN_ID: self.run_id,
                    "project": "predictive-maintenance-platform",
                    "tracking_structure": "child",
                    "purpose": "validation-comparison",
                    "model_name": evaluation.model_name,
                    "selected_model": str(
                        evaluation.model_name == report.selected_model_name
                    ).lower(),
                },
            )
            child_run_id = child_run.info.run_id
            shared_parameters: dict[str, str | int | float] = {
                "random_seed": report.random_seed,
                "training_engine_count": len(report.training_engine_ids),
                "validation_engine_count": len(report.validation_engine_ids),
                "training_rows": report.training_rows,
                "validation_rows": report.validation_rows,
                "validation_sample_count": evaluation.sample_count,
                "dataset_version": report.dataset_version,
                "feature_version": report.feature_version,
            }
            for key, value in shared_parameters.items():
                self.client.log_param(child_run_id, key, value)
            for key, value in _extract_model_parameters(model).items():
                self.client.log_param(child_run_id, key, value)

            self.client.log_metric(child_run_id, "validation_mae_cycles", evaluation.mae_cycles)
            self.client.log_metric(
                child_run_id,
                "validation_rmse_cycles",
                evaluation.rmse_cycles,
            )
            self.client.log_metric(
                child_run_id,
                "validation_nasa_score",
                evaluation.nasa_score,
            )
            self.client.set_terminated(child_run_id, status="FINISHED")
        except Exception as error:
            if child_run_id is not None:
                try:
                    self.client.set_terminated(child_run_id, status="FAILED")
                except Exception:
                    pass
            raise ExperimentTrackingError(
                f"could not log MLflow child run for {evaluation.model_name}"
            ) from error


def check_tracking_connection(
    *,
    tracking_uri: str,
    experiment_name: str,
) -> TrackingConnection:
    """Confirm that MLflow responds and inspect the target experiment."""

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
    except Exception as error:
        raise ExperimentTrackingError(
            f"could not connect to MLflow tracking server at {tracking_uri}"
        ) from error

    return TrackingConnection(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        experiment_exists=experiment is not None,
    )


@contextmanager
def start_comparison_run(
    *,
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
) -> Iterator[ComparisonRunTracker]:
    """Create and manage the parent MLflow run for one comparison execution."""

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else client.create_experiment(experiment_name)
        )
        run = client.create_run(
            experiment_id,
            run_name=run_name,
            tags={
                "project": "predictive-maintenance-platform",
                "tracking_structure": "parent",
                "source_training_run": run_name,
            },
        )
    except Exception as error:
        raise ExperimentTrackingError(
            f"could not start MLflow run at {tracking_uri}"
        ) from error

    tracker = ComparisonRunTracker(
        client=client,
        experiment_id=experiment_id,
        run_id=run.info.run_id,
        run_name=run_name,
    )
    try:
        yield tracker
    except BaseException:
        try:
            client.set_terminated(tracker.run_id, status="FAILED")
        except Exception:
            pass
        raise
    else:
        try:
            client.set_terminated(tracker.run_id, status="FINISHED")
        except Exception as error:
            raise ExperimentTrackingError(
                f"could not finish MLflow run {tracker.run_id}"
            ) from error


def _extract_model_parameters(model: object) -> dict[str, str | int | float | bool]:
    """Return stable scalar estimator parameters suitable for MLflow."""

    get_params = getattr(model, "get_params", None)
    if not callable(get_params):
        raise ExperimentTrackingError("candidate model does not expose get_params()")

    raw_parameters = get_params(deep=True)
    if not isinstance(raw_parameters, dict):
        raise ExperimentTrackingError("candidate model returned invalid parameters")

    parameters: dict[str, str | int | float | bool] = {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
    }
    for key, value in raw_parameters.items():
        if isinstance(value, (str, int, float, bool)):
            parameters[f"model_{key}"] = value
        elif "__" not in key and callable(getattr(value, "get_params", None)):
            parameters[f"model_{key}_class"] = (
                f"{type(value).__module__}.{type(value).__qualname__}"
            )
    return parameters
