"""Command-line entry point for local and automated platform operations."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from predictive_maintenance.core.settings import PlatformSettings, load_settings
from predictive_maintenance.data.download import DatasetDownloadError, download_archive
from predictive_maintenance.data.extract import DatasetExtractionError
from predictive_maintenance.data.prepare import DatasetPreparationError, prepare_fd001
from predictive_maintenance.storage.database import create_database_engine

app = typer.Typer(
    name="pm-platform",
    help="Operate the predictive-maintenance platform.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Acquire and prepare versioned datasets.")
config_app = typer.Typer(help="Inspect and validate runtime configuration.")
stream_app = typer.Typer(help="Consume and process telemetry events.")
model_app = typer.Typer(help="Train and inspect versioned RUL models.")
app.add_typer(data_app, name="data")
app.add_typer(config_app, name="config")
app.add_typer(stream_app, name="stream")
app.add_typer(model_app, name="model")


def _load_settings_or_exit() -> PlatformSettings:
    """Load settings or stop with a concise, non-secret validation error."""

    try:
        return load_settings()
    except ValidationError as error:
        typer.echo("Configuration is invalid:", err=True)
        for issue in error.errors(include_input=False, include_url=False):
            setting_name = "_".join(str(part) for part in issue["loc"]).upper()
            typer.echo(f"- PM_{setting_name}: {issue['msg']}", err=True)
        raise typer.Exit(code=2) from error


@config_app.command("check")
def check_config() -> None:
    """Validate common settings and print a non-secret summary."""

    settings = _load_settings_or_exit()
    typer.echo("Configuration is valid.")
    typer.echo(f"Environment: {settings.environment.value}")
    typer.echo(f"Log level: {settings.log_level}")
    typer.echo(f"Data root: {settings.data_root.resolve()}")
    typer.echo(
        f"MLflow tracking: {settings.mlflow_tracking_uri}"
        if settings.mlflow_tracking_uri is not None
        else "MLflow tracking: disabled"
    )
    typer.echo(f"MLflow experiment: {settings.mlflow_experiment_name}")
    typer.echo(f"MLflow registered model: {settings.mlflow_registered_model_name}")
    typer.echo(f"Inference model source: {settings.inference_model_source}")
    typer.echo(f"Model cache: {settings.model_cache_root.resolve()}")


@data_app.command("download")
def download_data(
    force: Annotated[
        bool,
        typer.Option(help="Download again even when a verified archive exists."),
    ] = False,
) -> None:
    """Download and checksum-verify the NASA C-MAPSS source archive."""

    settings = _load_settings_or_exit()
    destination_dir = settings.data_root / "raw" / "cmapss"

    try:
        archive_path = download_archive(destination_dir, force=force)
    except DatasetDownloadError as error:
        typer.echo(f"Download failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Verified dataset archive: {archive_path}")


@data_app.command("prepare")
def prepare_data(
    allow_warnings: Annotated[
        bool,
        typer.Option(
            help="Permit publication with recorded warnings after explicit review.",
        ),
    ] = False,
) -> None:
    """Validate and atomically publish the trusted C-MAPSS FD001 dataset."""

    settings = _load_settings_or_exit()
    cmapss_root = settings.data_root / "raw" / "cmapss"
    interim_dir = settings.data_root / "interim" / "cmapss" / "v1" / "fd001"
    quarantine_dir = settings.data_root / "quarantine" / "cmapss" / "v1" / "fd001"

    try:
        manifest = prepare_fd001(
            source_archive=cmapss_root / "nasa_turbofan_outer.zip",
            raw_version_dir=cmapss_root / "v1",
            interim_dir=interim_dir,
            quarantine_dir=quarantine_dir,
            allow_warnings=allow_warnings,
        )
    except (DatasetExtractionError, DatasetPreparationError) as error:
        typer.echo(f"Preparation failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Trusted FD001 dataset: {interim_dir}")
    typer.echo(f"Validation status: {manifest.validation_status}")


@stream_app.command("consume-once")
def consume_once() -> None:
    """Consume one telemetry event and persist it after successful processing."""
    from predictive_maintenance.streaming.consumer import TelemetryConsumer

    settings = _load_settings_or_exit()
    engine = create_database_engine(settings)
    consumer = TelemetryConsumer(settings, engine)
    try:
        outcome = consumer.consume_once()
    finally:
        consumer.close()
        engine.dispose()

    typer.echo(f"Consumer result: {outcome.value}")


@stream_app.command("worker")
def run_worker(
    max_messages: Annotated[
        int | None,
        typer.Option(help="Stop after this many messages; omit for continuous operation."),
    ] = None,
) -> None:
    """Run the telemetry consumer continuously or for a bounded test session."""
    from predictive_maintenance.streaming.consumer import TelemetryConsumer

    settings = _load_settings_or_exit()
    engine = create_database_engine(settings)
    consumer = TelemetryConsumer(settings, engine)
    try:
        processed = consumer.run(max_messages=max_messages)
    finally:
        consumer.close()
        engine.dispose()
    typer.echo(f"Worker processed {processed} message(s).")


@model_app.command("train-baseline")
def train_baseline_model(
    dataset_dir: Annotated[
        Path,
        typer.Option(help="Trusted FD001 dataset publication."),
    ] = Path("data/interim/cmapss/v1/fd001"),
    artifact_root: Annotated[
        Path,
        typer.Option(help="Destination for immutable model releases."),
    ] = Path("artifacts"),
    model_release: Annotated[
        str,
        typer.Option(help="Unique version for this candidate model."),
    ] = "rul-model-v1",
) -> None:
    """Train and evaluate the first leakage-safe FD001 RUL baseline."""
    from predictive_maintenance.training.baseline import (
        BaselineTrainingError,
        train_baseline,
    )

    try:
        evaluation = train_baseline(
            dataset_dir=dataset_dir,
            artifact_root=artifact_root,
            model_release=model_release,
        )
    except BaselineTrainingError as error:
        typer.echo(f"Baseline training failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Candidate model: {evaluation.model_release}")
    typer.echo(f"Validation MAE: {evaluation.model_mae_cycles:.3f} cycles")
    typer.echo(f"Validation RMSE: {evaluation.model_rmse_cycles:.3f} cycles")
    typer.echo(f"Age-only MAE: {evaluation.age_only_mae_cycles:.3f} cycles")
    typer.echo(f"Improvement over age-only: {evaluation.improvement_over_age_only * 100:.2f}%")
    typer.echo("Status: candidate (review required before approval)")


@model_app.command("tracking-check")
def check_model_tracking() -> None:
    """Verify the configured MLflow server without creating an experiment."""
    settings = _load_settings_or_exit()
    if settings.mlflow_tracking_uri is None:
        typer.echo(
            "MLflow tracking is disabled; set PM_MLFLOW_TRACKING_URI first.",
            err=True,
        )
        raise typer.Exit(code=2)

    from predictive_maintenance.training.tracking import (
        ExperimentTrackingError,
        check_tracking_connection,
    )

    try:
        connection = check_tracking_connection(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
        )
    except ExperimentTrackingError as error:
        typer.echo(f"MLflow connection failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"MLflow server is available: {connection.tracking_uri}")
    typer.echo(f"Target experiment: {connection.experiment_name}")
    typer.echo(
        "Experiment already exists."
        if connection.experiment_exists
        else "Experiment does not exist yet; the first tracked run will create it."
    )


@model_app.command("train-compare")
def train_model_comparison(
    dataset_dir: Annotated[
        Path,
        typer.Option(help="Trusted FD001 dataset publication."),
    ] = Path("data/interim/cmapss/v1/fd001"),
    output_root: Annotated[
        Path,
        typer.Option(help="Destination for immutable model-comparison runs."),
    ] = Path("artifacts/training-runs"),
    run_name: Annotated[
        str | None,
        typer.Option(help="Unique run name; defaults to a UTC timestamp."),
    ] = None,
    random_seed: Annotated[
        int,
        typer.Option(help="Seed used for the engine-level validation split."),
    ] = 42,
) -> None:
    """Train, compare, and test the four FD001 RUL model candidates."""
    from datetime import UTC, datetime

    from predictive_maintenance.training.data import TrainingDataError
    from predictive_maintenance.training.evaluate import EvaluationError
    from predictive_maintenance.training.train import TrainingRunError, train_and_compare

    resolved_run_name = run_name or datetime.now(UTC).strftime("fd001-%Y%m%dT%H%M%SZ")
    settings = _load_settings_or_exit()
    try:
        if settings.mlflow_tracking_uri is None:
            report = train_and_compare(
                dataset_dir=dataset_dir,
                output_root=output_root,
                run_name=resolved_run_name,
                random_seed=random_seed,
            )
        else:
            from predictive_maintenance.training.tracking import (
                ExperimentTrackingError,
                start_comparison_run,
            )

            try:
                with start_comparison_run(
                    tracking_uri=settings.mlflow_tracking_uri,
                    experiment_name=settings.mlflow_experiment_name,
                    run_name=resolved_run_name,
                ) as tracker:
                    report = train_and_compare(
                        dataset_dir=dataset_dir,
                        output_root=output_root,
                        run_name=resolved_run_name,
                        random_seed=random_seed,
                    )
                    tracker.log_candidate_runs(report=report)
                    tracker.log_training_report(
                        report=report,
                        run_directory=output_root / resolved_run_name,
                    )
            except ExperimentTrackingError as error:
                typer.echo(f"MLflow tracking failed: {error}", err=True)
                raise typer.Exit(code=1) from error
    except (TrainingDataError, EvaluationError, TrainingRunError) as error:
        typer.echo(f"Model comparison failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo("Validation ranking (lower is better):")
    typer.echo(f"{'Rank':<6}{'Model':<22}{'MAE':>12}{'RMSE':>12}{'NASA score':>16}")
    for rank, evaluation in enumerate(report.validation_ranking, start=1):
        typer.echo(
            f"{rank:<6}{evaluation.model_name:<22}"
            f"{evaluation.mae_cycles:>12.3f}"
            f"{evaluation.rmse_cycles:>12.3f}"
            f"{evaluation.nasa_score:>16.3f}"
        )

    test_result = report.official_test_evaluation
    typer.echo(f"Selected model: {report.selected_model_name}")
    typer.echo("Official test result (selected model only):")
    typer.echo(f"- MAE: {test_result.mae_cycles:.3f} cycles")
    typer.echo(f"- RMSE: {test_result.rmse_cycles:.3f} cycles")
    typer.echo(f"- NASA score: {test_result.nasa_score:.3f}")
    typer.echo(f"Saved run: {(output_root / resolved_run_name).resolve()}")


@model_app.command("package-candidate")
def package_model_candidate(
    training_run_directory: Annotated[
        Path,
        typer.Option(help="Immutable training run containing the selected model."),
    ] = Path("artifacts/training-runs/fd001-four-model-regression-v2"),
    artifact_root: Annotated[
        Path,
        typer.Option(help="Destination for immutable candidate model releases."),
    ] = Path("artifacts/model-releases"),
    model_release: Annotated[
        str,
        typer.Option(help="Unique release name for the candidate package."),
    ] = "rul-lightgbm-v1",
) -> None:
    """Package a training-run winner without approving it for production."""
    from predictive_maintenance.training.package import (
        CandidatePackagingError,
        package_candidate,
    )

    try:
        manifest = package_candidate(
            training_run_directory=training_run_directory,
            artifact_root=artifact_root,
            model_release=model_release,
        )
    except CandidatePackagingError as error:
        typer.echo(f"Candidate packaging failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Candidate release: {manifest.model_release}")
    typer.echo(f"Model: {manifest.model_name}")
    typer.echo(f"Feature contract: {manifest.feature_version}")
    typer.echo(f"Model SHA-256: {manifest.model_sha256}")
    typer.echo(f"Status: {manifest.status.value} (approval required before serving)")
    typer.echo(f"Saved package: {(artifact_root / model_release).resolve()}")


@model_app.command("register-candidate")
def register_model_candidate(
    manifest_path: Annotated[
        Path,
        typer.Option(help="Integrity-protected candidate manifest to register."),
    ] = Path("artifacts/model-releases/rul-lightgbm-v1/manifest.json"),
) -> None:
    """Register a packaged candidate without approving or promoting it."""

    settings = _load_settings_or_exit()
    if settings.mlflow_tracking_uri is None:
        typer.echo(
            "MLflow tracking is disabled; set PM_MLFLOW_TRACKING_URI first.",
            err=True,
        )
        raise typer.Exit(code=2)

    from predictive_maintenance.training.registry import (
        CandidateRegistrationError,
        register_candidate,
    )

    try:
        registration = register_candidate(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            registered_model_name=settings.mlflow_registered_model_name,
            manifest_path=manifest_path,
        )
    except CandidateRegistrationError as error:
        typer.echo(f"Candidate registration failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Registered model: {registration.registered_model_name}")
    typer.echo(f"Model version: {registration.model_version}")
    typer.echo(f"Model release: {registration.model_release}")
    typer.echo(f"Approval status: {registration.approval_status}")
    typer.echo(
        "Result: existing matching version"
        if registration.already_registered
        else "Result: new candidate version created"
    )
    typer.echo("Serving promotion: not performed")


@model_app.command("approve-candidate")
def approve_model_candidate(
    approver: Annotated[
        str,
        typer.Option(help="Identity of the human making the approval decision."),
    ],
    reason: Annotated[
        str,
        typer.Option(help="Auditable reason for approving this candidate."),
    ],
    evidence_path: Annotated[
        Path,
        typer.Option(help="Validated approval-gate evidence JSON."),
    ],
    manifest_path: Annotated[
        Path,
        typer.Option(help="Local immutable candidate manifest to verify again."),
    ] = Path("artifacts/model-releases/rul-lightgbm-v1/manifest.json"),
    model_version: Annotated[
        str,
        typer.Option(help="MLflow model version associated with the candidate."),
    ] = "1",
) -> None:
    """Record human approval without assigning a serving alias."""

    settings = _load_settings_or_exit()
    if settings.mlflow_tracking_uri is None:
        typer.echo(
            "MLflow tracking is disabled; set PM_MLFLOW_TRACKING_URI first.",
            err=True,
        )
        raise typer.Exit(code=2)

    from predictive_maintenance.training.approval import ApprovalError, approve_candidate

    try:
        result = approve_candidate(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            registered_model_name=settings.mlflow_registered_model_name,
            model_version=model_version,
            manifest_path=manifest_path,
            evidence_path=evidence_path,
            approver=approver,
            reason=reason,
        )
    except ApprovalError as error:
        typer.echo(f"Candidate approval failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Registered model: {result.registered_model_name}")
    typer.echo(f"Model version: {result.model_version}")
    typer.echo(f"Model release: {result.model_release}")
    typer.echo(f"Approval status: {result.approval_status}")
    typer.echo(f"Decision run: {result.decision_run_id}")
    typer.echo(
        "Result: existing approval retained"
        if result.already_approved
        else "Result: immutable approval decision recorded"
    )
    typer.echo("Serving alias: unchanged")


@model_app.command("benchmark-candidate")
def benchmark_model_candidate(
    manifest_path: Annotated[
        Path,
        typer.Option(help="Integrity-protected candidate manifest to benchmark."),
    ] = Path("artifacts/model-releases/rul-lightgbm-v1/manifest.json"),
    output_path: Annotated[
        Path,
        typer.Option(help="Immutable destination for the benchmark report."),
    ] = Path("artifacts/approval-evidence/rul-lightgbm-v1-serving-benchmark.json"),
    repetitions: Annotated[
        int,
        typer.Option(help="Number of measured single-record predictions."),
    ] = 1_000,
) -> None:
    """Benchmark the packaged model used by the serving worker."""

    from predictive_maintenance.training.benchmark import (
        ServingBenchmarkError,
        benchmark_candidate,
    )

    try:
        report = benchmark_candidate(
            manifest_path=manifest_path,
            output_path=output_path,
            repetitions=repetitions,
        )
    except ServingBenchmarkError as error:
        typer.echo(f"Candidate benchmark failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Model release: {report.model_release}")
    typer.echo(f"Model size: {report.model_size_bytes} bytes")
    typer.echo(f"Load latency: {report.load_latency_ms:.3f} ms")
    typer.echo(f"Prediction p50: {report.prediction_p50_ms:.3f} ms")
    typer.echo(f"Prediction p95: {report.prediction_p95_ms:.3f} ms")
    typer.echo(f"Prediction p99: {report.prediction_p99_ms:.3f} ms")
    typer.echo(f"Peak process RSS: {report.process_peak_rss_bytes} bytes")
    typer.echo(f"Saved benchmark: {output_path.resolve()}")


@model_app.command("promote-approved")
def promote_approved_model_version(
    promoted_by: Annotated[
        str,
        typer.Option(help="Identity of the human authorizing promotion."),
    ],
    reason: Annotated[
        str,
        typer.Option(help="Auditable reason for assigning the champion alias."),
    ],
    manifest_path: Annotated[
        Path,
        typer.Option(help="Local immutable package manifest to verify again."),
    ] = Path("artifacts/model-releases/rul-lightgbm-v1/manifest.json"),
    model_version: Annotated[
        str,
        typer.Option(help="Approved MLflow model version to promote."),
    ] = "1",
) -> None:
    """Promote an approved version to champion with an audit record."""

    settings = _load_settings_or_exit()
    if settings.mlflow_tracking_uri is None:
        typer.echo(
            "MLflow tracking is disabled; set PM_MLFLOW_TRACKING_URI first.",
            err=True,
        )
        raise typer.Exit(code=2)

    from predictive_maintenance.training.promotion import (
        PromotionError,
        promote_approved_model,
    )

    try:
        result = promote_approved_model(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            registered_model_name=settings.mlflow_registered_model_name,
            model_version=model_version,
            manifest_path=manifest_path,
            promoted_by=promoted_by,
            reason=reason,
        )
    except PromotionError as error:
        typer.echo(f"Model promotion failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Registered model: {result.registered_model_name}")
    typer.echo(f"Model release: {result.model_release}")
    typer.echo(f"Champion version: {result.champion_version}")
    typer.echo(f"Previous version: {result.previous_version or 'none'}")
    typer.echo(f"Promotion run: {result.promotion_run_id}")
    typer.echo(
        "Result: existing promotion retained"
        if result.already_promoted
        else "Result: champion alias updated"
    )


@model_app.command("verify-champion")
def verify_champion() -> None:
    """Verify and cache the approved MLflow champion used by inference."""

    settings = _load_settings_or_exit()
    if settings.mlflow_tracking_uri is None:
        typer.echo(
            "MLflow tracking is disabled; set PM_MLFLOW_TRACKING_URI first.",
            err=True,
        )
        raise typer.Exit(code=2)

    from predictive_maintenance.inference.registry import (
        ChampionResolutionError,
        resolve_champion,
    )

    try:
        champion = resolve_champion(
            tracking_uri=settings.mlflow_tracking_uri,
            registered_model_name=settings.mlflow_registered_model_name,
            cache_root=settings.model_cache_root,
        )
    except ChampionResolutionError as error:
        typer.echo(f"Champion verification failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Registered model: {champion.registered_model_name}")
    typer.echo(f"Champion version: {champion.model_version}")
    typer.echo(f"Model release: {champion.model_release}")
    typer.echo(f"Registry source run: {champion.source_run_id}")
    typer.echo(f"Verified cache: {champion.cache_directory}")
    typer.echo("Result: champion is approved, auditable, integrity-checked, and loadable")


if __name__ == "__main__":
    app()
