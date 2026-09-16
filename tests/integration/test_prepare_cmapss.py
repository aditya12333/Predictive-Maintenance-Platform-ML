"""Integration tests for atomic FD001 dataset preparation."""

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import polars as pl
import pytest

from predictive_maintenance.data.contracts import EXPECTED_INNER_MEMBERS
from predictive_maintenance.data.download import calculate_sha256
from predictive_maintenance.data.prepare import DatasetPreparationError, prepare_fd001


def _telemetry_row(engine_id: int, *, malformed: bool = False) -> str:
    values = [
        str(engine_id),
        "1",
        "0.0",
        "0.0",
        "100.0",
        *[str(500.0 + index) for index in range(21)],
    ]
    if malformed:
        values.pop()
    return " ".join(values)


def _build_source_archive(destination: Path, *, malformed_train: bool = False) -> None:
    train = "\n".join(
        _telemetry_row(engine_id, malformed=malformed_train and engine_id == 42)
        for engine_id in range(1, 101)
    )
    test = "\n".join(_telemetry_row(engine_id) for engine_id in range(1, 101))
    rul = "\n".join("10" for _ in range(100))

    contents = {member: f"placeholder for {member}\n" for member in EXPECTED_INNER_MEMBERS}
    contents["train_FD001.txt"] = train + "\n"
    contents["test_FD001.txt"] = test + "\n"
    contents["RUL_FD001.txt"] = rul + "\n"

    inner_buffer = BytesIO()
    with ZipFile(inner_buffer, "w", compression=ZIP_STORED) as inner:
        for member, content in contents.items():
            inner.writestr(member, content)

    with ZipFile(destination, "w", compression=ZIP_STORED) as outer:
        outer.writestr(
            "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip",
            inner_buffer.getvalue(),
        )


def test_prepare_fd001_publishes_parquet_and_is_idempotent(tmp_path: Path) -> None:
    source_archive = tmp_path / "source.zip"
    raw_dir = tmp_path / "raw" / "v1"
    interim_dir = tmp_path / "interim" / "fd001"
    quarantine_dir = tmp_path / "quarantine" / "fd001"
    _build_source_archive(source_archive)

    first_manifest = prepare_fd001(
        source_archive,
        raw_dir,
        interim_dir,
        quarantine_dir,
        expected_source_sha256=calculate_sha256(source_archive),
    )
    second_manifest = prepare_fd001(
        source_archive,
        raw_dir,
        interim_dir,
        quarantine_dir,
        expected_source_sha256=calculate_sha256(source_archive),
    )

    assert first_manifest == second_manifest
    assert first_manifest.validation_status == "PASSED"
    assert pl.read_parquet(interim_dir / "train.parquet").shape == (100, 29)
    assert pl.read_parquet(interim_dir / "test-rul.parquet").shape == (100, 6)
    assert not quarantine_dir.exists()


def test_prepare_fd001_blocks_and_reports_invalid_publication(tmp_path: Path) -> None:
    source_archive = tmp_path / "source.zip"
    raw_dir = tmp_path / "raw" / "v1"
    interim_dir = tmp_path / "interim" / "fd001"
    quarantine_dir = tmp_path / "quarantine" / "fd001"
    _build_source_archive(source_archive, malformed_train=True)

    with pytest.raises(DatasetPreparationError, match="publication blocked"):
        prepare_fd001(
            source_archive,
            raw_dir,
            interim_dir,
            quarantine_dir,
            expected_source_sha256=calculate_sha256(source_archive),
        )

    assert not interim_dir.exists()
    assert (quarantine_dir / "validation-report.json").is_file()
    assert (quarantine_dir / "rejected-records.jsonl").is_file()
