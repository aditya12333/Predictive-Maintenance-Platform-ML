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
    try:
        report = train_and_compare(
            dataset_dir=dataset_dir,
            output_root=output_root,
            run_name=resolved_run_name,
            random_seed=random_seed,
        )
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

    typer.echo("28-cycle warning metrics:")
    typer.echo(
        f"{'Model':<22}{'Precision':>12}{'Recall':>12}{'F1':>12}"
        f"{'False alert':>14}{'Missed':>12}"
    )
    for evaluation in report.validation_ranking:
        warning = evaluation.failure_horizon
        typer.echo(
            f"{evaluation.model_name:<22}"
            f"{warning.precision:>12.3f}"
            f"{warning.recall:>12.3f}"
            f"{warning.f1_score:>12.3f}"
            f"{warning.false_alert_rate:>14.3f}"
            f"{warning.missed_failure_rate:>12.3f}"
        )

    test_result = report.official_test_evaluation
    typer.echo(f"Selected model: {report.selected_model_name}")
    typer.echo("Official test result (selected model only):")
    typer.echo(f"- MAE: {test_result.mae_cycles:.3f} cycles")
    typer.echo(f"- RMSE: {test_result.rmse_cycles:.3f} cycles")
    typer.echo(f"- NASA score: {test_result.nasa_score:.3f}")
    typer.echo("- 28-cycle warning metrics:")
    typer.echo(f"  - Precision: {test_result.failure_horizon.precision:.3f}")
    typer.echo(f"  - Recall: {test_result.failure_horizon.recall:.3f}")
    typer.echo(f"  - F1: {test_result.failure_horizon.f1_score:.3f}")
    typer.echo(f"  - False-alert rate: {test_result.failure_horizon.false_alert_rate:.3f}")
    typer.echo(f"  - Missed-failure rate: {test_result.failure_horizon.missed_failure_rate:.3f}")
    typer.echo(f"Saved run: {(output_root / resolved_run_name).resolve()}")


if __name__ == "__main__":
    app()
