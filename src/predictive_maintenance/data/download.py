"""Download the NASA C-MAPSS dataset reproducibly."""

import hashlib
import shutil
import ssl
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import certifi

NASA_DATASET_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)

NASA_ARCHIVE_SHA256 = (
    "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
)

ARCHIVE_NAME = "nasa_turbofan_outer.zip"


class DatasetDownloadError(RuntimeError):
    """Raised when the dataset cannot be downloaded or verified."""


def calculate_sha256(file_path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_checksum(file_path: Path, expected_sha256: str) -> None:
    """Verify that a file matches its expected checksum."""

    actual_sha256 = calculate_sha256(file_path)

    if actual_sha256 != expected_sha256:
        raise DatasetDownloadError(
            f"Checksum verification failed for {file_path}. "
            f"Expected {expected_sha256}, received {actual_sha256}."
        )


def download_archive(
    destination_dir: Path,
    *,
    url: str = NASA_DATASET_URL,
    expected_sha256: str = NASA_ARCHIVE_SHA256,
    force: bool = False,
) -> Path:
    """Download and verify the NASA archive.

    A temporary file is downloaded first. The final archive is only replaced
    after checksum validation succeeds.
    """

    destination_dir.mkdir(parents=True, exist_ok=True)
    archive_path = destination_dir / ARCHIVE_NAME

    if archive_path.exists() and not force:
        verify_checksum(archive_path, expected_sha256)
        return archive_path

    request = Request(
        url,
        headers={"User-Agent": "predictive-maintenance-platform/0.1"},
    )

    temporary_path: Path | None = None
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination_dir,
            prefix="cmapss-",
            suffix=".part",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            with urlopen(request, timeout=120, context=ssl_context,
            ) as response:
                shutil.copyfileobj(response, temporary_file)

        verify_checksum(temporary_path, expected_sha256)
        temporary_path.replace(archive_path)

    except Exception as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

        if isinstance(error, DatasetDownloadError):
            raise

        raise DatasetDownloadError(
            f"Failed to download the C-MAPSS dataset from {url}"
        ) from error

    return archive_path


def main() -> None:
    """Download the dataset into the raw data directory."""

    archive_path = download_archive(Path("data/raw/cmapss"))
    print(f"Verified dataset archive: {archive_path}")


if __name__ == "__main__":
    main()
