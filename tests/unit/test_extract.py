"""Tests for safe, idempotent C-MAPSS extraction."""

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest

from predictive_maintenance.data.contracts import EXPECTED_INNER_MEMBERS
from predictive_maintenance.data.download import calculate_sha256
from predictive_maintenance.data.extract import (
    DatasetExtractionError,
    extract_cmapss_archive,
)


def _build_outer_archive(
    destination: Path,
    *,
    extra_inner_members: tuple[str, ...] = (),
) -> None:
    inner_buffer = BytesIO()
    with ZipFile(inner_buffer, "w", compression=ZIP_STORED) as inner:
        for member in EXPECTED_INNER_MEMBERS:
            inner.writestr(member, f"content for {member}\n")
        for member in extra_inner_members:
            inner.writestr(member, "unsafe\n")

    with ZipFile(destination, "w", compression=ZIP_STORED) as outer:
        outer.writestr(
            "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip",
            inner_buffer.getvalue(),
        )


def test_extract_archive_is_idempotent(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    destination = tmp_path / "v1"
    _build_outer_archive(archive_path)
    checksum = calculate_sha256(archive_path)

    first_manifest = extract_cmapss_archive(
        archive_path,
        destination,
        expected_source_sha256=checksum,
    )
    second_manifest = extract_cmapss_archive(
        archive_path,
        destination,
        expected_source_sha256=checksum,
    )

    assert first_manifest == second_manifest
    assert (destination / "source-manifest.json").is_file()
    assert (destination / "extracted" / "train_FD001.txt").is_file()


def test_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    _build_outer_archive(archive_path, extra_inner_members=("../outside.txt",))

    with pytest.raises(DatasetExtractionError, match="unsafe member"):
        extract_cmapss_archive(
            archive_path,
            tmp_path / "v1",
            expected_source_sha256=calculate_sha256(archive_path),
        )

    assert not (tmp_path / "outside.txt").exists()


def test_extract_detects_modified_published_file(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    destination = tmp_path / "v1"
    _build_outer_archive(archive_path)
    checksum = calculate_sha256(archive_path)
    extract_cmapss_archive(
        archive_path,
        destination,
        expected_source_sha256=checksum,
    )
    extracted_file = destination / "extracted" / "train_FD001.txt"
    extracted_file.write_text("modified\n", encoding="utf-8")

    with pytest.raises(DatasetExtractionError, match="size changed|checksum changed"):
        extract_cmapss_archive(
            archive_path,
            destination,
            expected_source_sha256=checksum,
        )
