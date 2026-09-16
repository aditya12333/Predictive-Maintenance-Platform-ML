"""Canonical contracts for the NASA C-MAPSS source data."""

from typing import Final

import polars as pl

DATASET_ID: Final = "nasa-cmapss-classic"
DATASET_VERSION: Final = "v1"
TELEMETRY_SCHEMA_VERSION: Final = "telemetry-v1"
TEST_RUL_SCHEMA_VERSION: Final = "test-rul-v1"

SUBSETS: Final = ("FD001", "FD002", "FD003", "FD004")
SPLITS: Final = ("train", "test")

IDENTIFIER_COLUMNS: Final = ("engine_id", "cycle")
OPERATING_SETTING_COLUMNS: Final = tuple(
    f"operating_setting_{index}" for index in range(1, 4)
)
SENSOR_COLUMNS: Final = tuple(f"sensor_{index}" for index in range(1, 22))
SOURCE_TELEMETRY_COLUMNS: Final = (
    *IDENTIFIER_COLUMNS,
    *OPERATING_SETTING_COLUMNS,
    *SENSOR_COLUMNS,
)
CANONICAL_TELEMETRY_COLUMNS: Final = (
    "dataset_version",
    "subset_id",
    "split",
    *SOURCE_TELEMETRY_COLUMNS,
)

TELEMETRY_SCHEMA: Final = {
    "dataset_version": pl.String,
    "subset_id": pl.String,
    "split": pl.String,
    "engine_id": pl.Int32,
    "cycle": pl.Int32,
    **{column: pl.Float64 for column in OPERATING_SETTING_COLUMNS},
    **{column: pl.Float64 for column in SENSOR_COLUMNS},
}

TEST_RUL_SCHEMA: Final = {
    "dataset_version": pl.String,
    "subset_id": pl.String,
    "engine_id": pl.Int32,
    "observed_final_cycle": pl.Int32,
    "additional_rul": pl.Int32,
    "true_failure_cycle": pl.Int32,
}

EXPECTED_INNER_MEMBERS: Final = (
    "Damage Propagation Modeling.pdf",
    "RUL_FD001.txt",
    "RUL_FD002.txt",
    "RUL_FD003.txt",
    "RUL_FD004.txt",
    "readme.txt",
    "test_FD001.txt",
    "test_FD002.txt",
    "test_FD003.txt",
    "test_FD004.txt",
    "train_FD001.txt",
    "train_FD002.txt",
    "train_FD003.txt",
    "train_FD004.txt",
)

FD001_EXPECTED_ENGINE_COUNTS: Final = {"train": 100, "test": 100}
