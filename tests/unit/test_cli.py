"""Tests for the project command-line entry point."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from predictive_maintenance.cli import app
from predictive_maintenance.data.manifest import DatasetManifest

runner = CliRunner()


def test_root_help_lists_command_groups() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "data" in result.stdout
    assert "config" in result.stdout


def test_config_check_prints_validated_non_secret_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PM_ENVIRONMENT", "test")
    monkeypatch.setenv("PM_LOG_LEVEL", "warning")
    monkeypatch.setenv("PM_DATA_ROOT", str(tmp_path))

    result = runner.invoke(app, ["config", "check"])

    assert result.exit_code == 0
    assert "Configuration is valid." in result.stdout
    assert "Environment: test" in result.stdout
    assert "Log level: WARNING" in result.stdout
    assert f"Data root: {tmp_path}" in result.stdout


def test_config_check_fails_clearly_for_invalid_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PM_ENVIRONMENT", "unknown")

    result = runner.invoke(app, ["config", "check"])

    assert result.exit_code == 2
    assert "Configuration is invalid:" in result.stderr
    assert "PM_ENVIRONMENT" in result.stderr


def test_data_download_uses_configured_data_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded: dict[str, object] = {}

    def fake_download(destination_dir: Path, *, force: bool = False) -> Path:
        recorded["destination_dir"] = destination_dir
        recorded["force"] = force
        return destination_dir / "nasa_turbofan_outer.zip"

    monkeypatch.setenv("PM_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("predictive_maintenance.cli.download_archive", fake_download)

    result = runner.invoke(app, ["data", "download", "--force"])

    assert result.exit_code == 0
    assert recorded == {
        "destination_dir": tmp_path / "raw" / "cmapss",
        "force": True,
    }


def test_data_prepare_uses_configured_data_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded: dict[str, object] = {}
    manifest = DatasetManifest(
        dataset_id="nasa-cmapss-classic",
        dataset_version="v1",
        subset_id="FD001",
        telemetry_schema_version="telemetry-v1",
        test_rul_schema_version="test-rul-v1",
        source_manifest_sha256="a" * 64,
        published_at="2026-09-05T00:00:00Z",
        validation_status="PASSED",
        files=(),
    )

    def fake_prepare(**kwargs: object) -> DatasetManifest:
        recorded.update(kwargs)
        return manifest

    monkeypatch.setenv("PM_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("predictive_maintenance.cli.prepare_fd001", fake_prepare)

    result = runner.invoke(app, ["data", "prepare"])

    assert result.exit_code == 0
    assert recorded == {
        "source_archive": tmp_path / "raw" / "cmapss" / "nasa_turbofan_outer.zip",
        "raw_version_dir": tmp_path / "raw" / "cmapss" / "v1",
        "interim_dir": tmp_path / "interim" / "cmapss" / "v1" / "fd001",
        "quarantine_dir": tmp_path / "quarantine" / "cmapss" / "v1" / "fd001",
        "allow_warnings": False,
    }
    assert "Validation status: PASSED" in result.stdout
