"""Strict parsing and structural validation for NASA C-MAPSS tables."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import polars as pl

from predictive_maintenance.data.contracts import (
    CANONICAL_TELEMETRY_COLUMNS,
    DATASET_ID,
    DATASET_VERSION,
    FD001_EXPECTED_ENGINE_COUNTS,
    OPERATING_SETTING_COLUMNS,
    SOURCE_TELEMETRY_COLUMNS,
    TELEMETRY_SCHEMA,
    TELEMETRY_SCHEMA_VERSION,
    TEST_RUL_SCHEMA,
)
from predictive_maintenance.data.manifest import (
    DatasetStatistics,
    QualityIssue,
    RejectedRecord,
    ValidationReport,
    utc_now,
)

MISSING_TOKENS = frozenset({"", "na", "nan", "null", "none"})


@dataclass(frozen=True)
class TelemetryParseResult:
    """Canonical telemetry plus issues found during parsing."""

    frame: pl.DataFrame
    issues: tuple[QualityIssue, ...]
    rejected_records: tuple[RejectedRecord, ...]


@dataclass(frozen=True)
class RulParseResult:
    """Canonical test labels plus relationship issues."""

    frame: pl.DataFrame
    issues: tuple[QualityIssue, ...]
    rejected_records: tuple[RejectedRecord, ...]


def _empty_telemetry_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=TELEMETRY_SCHEMA)


def _empty_rul_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=TEST_RUL_SCHEMA)


def parse_telemetry_file(
    source_file: Path,
    *,
    subset_id: str,
    split: Literal["train", "test"],
) -> TelemetryParseResult:
    """Parse a whitespace-delimited telemetry file using the canonical schema."""

    rows: list[tuple[object, ...]] = []
    issues: list[QualityIssue] = []
    rejected_records: list[RejectedRecord] = []

    with source_file.open(encoding="utf-8") as source:
        for line_number, raw_line_with_newline in enumerate(source, start=1):
            raw_line = raw_line_with_newline.rstrip("\r\n")
            if not raw_line.strip():
                continue
            tokens = raw_line.split()
            if len(tokens) != len(SOURCE_TELEMETRY_COLUMNS):
                reason = (
                    f"Expected {len(SOURCE_TELEMETRY_COLUMNS)} columns, "
                    f"received {len(tokens)}"
                )
                issues.append(
                    QualityIssue(
                        rule_id="telemetry_column_count",
                        severity="error",
                        scope="record",
                        message=reason,
                        source_file=source_file.name,
                        line_number=line_number,
                        observed_value=str(len(tokens)),
                    )
                )
                rejected_records.append(
                    RejectedRecord(
                        source_file=source_file.name,
                        line_number=line_number,
                        raw_record=raw_line,
                        reason=reason,
                    )
                )
                continue

            try:
                engine_id = int(tokens[0])
                cycle = int(tokens[1])
            except ValueError:
                reason = "Engine ID and cycle must be integers"
                issues.append(
                    QualityIssue(
                        rule_id="telemetry_identifier_type",
                        severity="error",
                        scope="record",
                        message=reason,
                        source_file=source_file.name,
                        line_number=line_number,
                    )
                )
                rejected_records.append(
                    RejectedRecord(
                        source_file=source_file.name,
                        line_number=line_number,
                        raw_record=raw_line,
                        reason=reason,
                    )
                )
                continue

            if engine_id <= 0 or cycle <= 0:
                reason = "Engine ID and cycle must be positive"
                issues.append(
                    QualityIssue(
                        rule_id="telemetry_identifier_range",
                        severity="error",
                        scope="record",
                        message=reason,
                        source_file=source_file.name,
                        line_number=line_number,
                        engine_id=engine_id if engine_id > 0 else None,
                        cycle=cycle if cycle > 0 else None,
                    )
                )
                rejected_records.append(
                    RejectedRecord(
                        source_file=source_file.name,
                        line_number=line_number,
                        raw_record=raw_line,
                        reason=reason,
                    )
                )
                continue

            numeric_values: list[float | None] = []
            row_has_error = False
            for index, (column_name, token) in enumerate(
                zip(SOURCE_TELEMETRY_COLUMNS[2:], tokens[2:], strict=True),
                start=2,
            ):
                if token.lower() in MISSING_TOKENS:
                    if column_name in OPERATING_SETTING_COLUMNS:
                        reason = f"Required operating setting is missing: {column_name}"
                        issues.append(
                            QualityIssue(
                                rule_id="operating_setting_missing",
                                severity="error",
                                scope="record",
                                message=reason,
                                source_file=source_file.name,
                                line_number=line_number,
                                engine_id=engine_id,
                                cycle=cycle,
                                column_name=column_name,
                            )
                        )
                        row_has_error = True
                    else:
                        issues.append(
                            QualityIssue(
                                rule_id="sensor_missing",
                                severity="warning",
                                scope="record",
                                message=f"Sensor value is missing: {column_name}",
                                source_file=source_file.name,
                                line_number=line_number,
                                engine_id=engine_id,
                                cycle=cycle,
                                column_name=column_name,
                            )
                        )
                    numeric_values.append(None)
                    continue

                try:
                    value = float(token)
                except ValueError:
                    reason = f"Value is not numeric: {column_name}"
                    issues.append(
                        QualityIssue(
                            rule_id="telemetry_numeric_type",
                            severity="error",
                            scope="record",
                            message=reason,
                            source_file=source_file.name,
                            line_number=line_number,
                            engine_id=engine_id,
                            cycle=cycle,
                            column_name=column_name,
                            observed_value=tokens[index],
                        )
                    )
                    numeric_values.append(None)
                    row_has_error = True
                    continue

                if not math.isfinite(value):
                    reason = f"Value must be finite: {column_name}"
                    issues.append(
                        QualityIssue(
                            rule_id="telemetry_numeric_finite",
                            severity="error",
                            scope="record",
                            message=reason,
                            source_file=source_file.name,
                            line_number=line_number,
                            engine_id=engine_id,
                            cycle=cycle,
                            column_name=column_name,
                            observed_value=token,
                        )
                    )
                    row_has_error = True
                numeric_values.append(value)

            if row_has_error:
                rejected_records.append(
                    RejectedRecord(
                        source_file=source_file.name,
                        line_number=line_number,
                        raw_record=raw_line,
                        reason="Record failed one or more required field validations",
                    )
                )
                continue

            rows.append(
                (
                    DATASET_VERSION,
                    subset_id,
                    split,
                    engine_id,
                    cycle,
                    *numeric_values,
                )
            )

    frame = (
        pl.DataFrame(
            rows,
            schema=list(TELEMETRY_SCHEMA.items()),
            orient="row",
        )
        if rows
        else _empty_telemetry_frame()
    )
    if tuple(frame.columns) != CANONICAL_TELEMETRY_COLUMNS:
        raise AssertionError("Canonical telemetry columns differ from the contract")

    return TelemetryParseResult(
        frame=frame,
        issues=tuple(issues),
        rejected_records=tuple(rejected_records),
    )


def _validate_telemetry_structure(
    frame: pl.DataFrame,
    *,
    source_file: str,
    split: Literal["train", "test"],
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    if frame.is_empty():
        return [
            QualityIssue(
                rule_id="telemetry_not_empty",
                severity="error",
                scope="file",
                message="Telemetry file contains no valid records",
                source_file=source_file,
            )
        ]

    duplicate_rows = (
        frame.group_by(["engine_id", "cycle"])
        .len()
        .filter(pl.col("len") > 1)
        .select(["engine_id", "cycle"])
    )
    for engine_id, cycle in duplicate_rows.iter_rows():
        issues.append(
            QualityIssue(
                rule_id="engine_cycle_unique",
                severity="error",
                scope="sequence",
                message="Engine-cycle pair is duplicated",
                source_file=source_file,
                engine_id=engine_id,
                cycle=cycle,
            )
        )

    expected_engine_count = FD001_EXPECTED_ENGINE_COUNTS[split]
    engine_ids = sorted(frame.get_column("engine_id").unique().to_list())
    expected_engine_ids = list(range(1, expected_engine_count + 1))
    if engine_ids != expected_engine_ids:
        issues.append(
            QualityIssue(
                rule_id="fd001_engine_ids",
                severity="error",
                scope="dataset",
                message=(
                    f"FD001 {split} engine IDs must be 1 through "
                    f"{expected_engine_count}"
                ),
                source_file=source_file,
            )
        )

    for engine_frame in frame.partition_by("engine_id", maintain_order=True):
        engine_id = int(engine_frame.item(0, "engine_id"))
        cycles = sorted(engine_frame.get_column("cycle").to_list())
        expected_cycles = list(range(1, max(cycles) + 1))
        if cycles != expected_cycles:
            issues.append(
                QualityIssue(
                    rule_id="engine_cycles_contiguous",
                    severity="error",
                    scope="sequence",
                    message="Engine cycles must be unique and contiguous from cycle 1",
                    source_file=source_file,
                    engine_id=engine_id,
                )
            )

    return issues


def parse_test_rul_file(
    source_file: Path,
    *,
    subset_id: str,
    test_frame: pl.DataFrame,
) -> RulParseResult:
    """Parse ordered test RUL labels and join them to test engine identities."""

    values: list[int] = []
    issues: list[QualityIssue] = []
    rejected_records: list[RejectedRecord] = []

    with source_file.open(encoding="utf-8") as source:
        for line_number, raw_line_with_newline in enumerate(source, start=1):
            raw_line = raw_line_with_newline.rstrip("\r\n")
            if not raw_line.strip():
                continue
            tokens = raw_line.split()
            if len(tokens) != 1:
                reason = f"Expected one RUL value, received {len(tokens)} columns"
                issues.append(
                    QualityIssue(
                        rule_id="rul_column_count",
                        severity="error",
                        scope="record",
                        message=reason,
                        source_file=source_file.name,
                        line_number=line_number,
                    )
                )
                rejected_records.append(
                    RejectedRecord(
                        source_file=source_file.name,
                        line_number=line_number,
                        raw_record=raw_line,
                        reason=reason,
                    )
                )
                continue
            try:
                value = int(tokens[0])
            except ValueError:
                value = 0
            if value <= 0:
                reason = "Additional test RUL must be a positive integer"
                issues.append(
                    QualityIssue(
                        rule_id="rul_positive_integer",
                        severity="error",
                        scope="record",
                        message=reason,
                        source_file=source_file.name,
                        line_number=line_number,
                        observed_value=tokens[0],
                    )
                )
                rejected_records.append(
                    RejectedRecord(
                        source_file=source_file.name,
                        line_number=line_number,
                        raw_record=raw_line,
                        reason=reason,
                    )
                )
                continue
            values.append(value)

    engine_final_cycles = (
        test_frame.group_by("engine_id")
        .agg(pl.col("cycle").max().alias("observed_final_cycle"))
        .sort("engine_id")
    )
    if len(values) != engine_final_cycles.height:
        issues.append(
            QualityIssue(
                rule_id="rul_engine_count_match",
                severity="error",
                scope="relationship",
                message=(
                    f"RUL label count {len(values)} does not match test engine count "
                    f"{engine_final_cycles.height}"
                ),
                source_file=source_file.name,
            )
        )
        return RulParseResult(
            frame=_empty_rul_frame(),
            issues=tuple(issues),
            rejected_records=tuple(rejected_records),
        )

    rul_rows = []
    for (engine_id, observed_final_cycle), additional_rul in zip(
        engine_final_cycles.iter_rows(),
        values,
        strict=True,
    ):
        rul_rows.append(
            (
                DATASET_VERSION,
                subset_id,
                engine_id,
                observed_final_cycle,
                additional_rul,
                observed_final_cycle + additional_rul,
            )
        )

    frame = pl.DataFrame(
        rul_rows,
        schema=list(TEST_RUL_SCHEMA.items()),
        orient="row",
    )
    return RulParseResult(
        frame=frame,
        issues=tuple(issues),
        rejected_records=tuple(rejected_records),
    )


def validate_fd001(
    train_result: TelemetryParseResult,
    test_result: TelemetryParseResult,
    rul_result: RulParseResult,
) -> ValidationReport:
    """Apply all FD001 publication gates and return a complete report."""

    issues = [
        *train_result.issues,
        *test_result.issues,
        *rul_result.issues,
        *_validate_telemetry_structure(
            train_result.frame,
            source_file="train_FD001.txt",
            split="train",
        ),
        *_validate_telemetry_structure(
            test_result.frame,
            source_file="test_FD001.txt",
            split="test",
        ),
    ]

    statistics: list[DatasetStatistics] = []
    split_frames: tuple[
        tuple[Literal["train", "test"], pl.DataFrame],
        ...,
    ] = (("train", train_result.frame), ("test", test_result.frame))
    for split, frame in split_frames:
        if frame.is_empty():
            continue
        statistics.append(
            DatasetStatistics(
                split=split,
                row_count=frame.height,
                engine_count=frame.get_column("engine_id").n_unique(),
                minimum_cycle=cast(int, frame.get_column("cycle").min()),
                maximum_cycle=cast(int, frame.get_column("cycle").max()),
            )
        )

    if any(issue.severity == "error" for issue in issues):
        status: Literal["PASSED", "PASSED_WITH_WARNINGS", "FAILED"] = "FAILED"
    elif issues:
        status = "PASSED_WITH_WARNINGS"
    else:
        status = "PASSED"

    return ValidationReport(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        subset_id="FD001",
        schema_version=TELEMETRY_SCHEMA_VERSION,
        status=status,
        generated_at=utc_now(),
        statistics=tuple(statistics),
        issues=tuple(issues),
    )
