"""Tests for the C-MAPSS downloader."""

import hashlib

import pytest

from predictive_maintenance.data.download import (
    ARCHIVE_NAME,
    DatasetDownloadError,
    calculate_sha256,
    download_archive,
    verify_checksum,
)


def test_calculate_sha256(tmp_path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"predictive-maintenance")

    expected = hashlib.sha256(b"predictive-maintenance").hexdigest()

    assert calculate_sha256(file_path) == expected


def test_verify_checksum_rejects_corrupted_file(tmp_path) -> None:
    file_path = tmp_path / "corrupted.zip"
    file_path.write_bytes(b"incorrect-content")

    with pytest.raises(DatasetDownloadError, match="Checksum verification failed"):
        verify_checksum(file_path, "invalid-checksum")


def test_valid_existing_archive_is_reused(tmp_path) -> None:
    archive_path = tmp_path / ARCHIVE_NAME
    archive_content = b"existing-valid-archive"
    archive_path.write_bytes(archive_content)

    expected_checksum = hashlib.sha256(archive_content).hexdigest()

    result = download_archive(
        tmp_path,
        expected_sha256=expected_checksum,
    )

    assert result == archive_path
    assert result.read_bytes() == archive_content
