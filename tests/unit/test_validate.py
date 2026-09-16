"""Tests for the C-MAPSS canonical data contract."""

from pathlib import Path

from predictive_maintenance.data.contracts import CANONICAL_TELEMETRY_COLUMNS
from predictive_maintenance.data.validate import (
    parse_telemetry_file,
    parse_test_rul_file,
    validate_fd001,
)


def _telemetry_row(engine_id: int = 1, cycle: int = 1) -> list[str]:
    return [
        str(engine_id),
        str(cycle),
        "0.0",
        "0.0",
        "100.0",
        *[str(500.0 + index) for index in range(21)],
    ]


def test_parse_valid_telemetry_uses_canonical_schema(tmp_path: Path) -> None:
    source_path = tmp_path / "train_FD001.txt"
    source_path.write_text(" ".join(_telemetry_row()) + "\n", encoding="utf-8")

    result = parse_telemetry_file(source_path, subset_id="FD001", split="train")

    assert result.frame.shape == (1, 29)
    assert tuple(result.frame.columns) == CANONICAL_TELEMETRY_COLUMNS
    assert not result.issues
    assert not result.rejected_records


def test_parse_preserves_known_missing_sensor_with_warning(tmp_path: Path) -> None:
    source_path = tmp_path / "train_FD001.txt"
    row = _telemetry_row()
    row[-1] = "NA"
    source_path.write_text(" ".join(row) + "\n", encoding="utf-8")

    result = parse_telemetry_file(source_path, subset_id="FD001", split="train")

    assert result.frame.item(0, "sensor_21") is None
    assert [issue.rule_id for issue in result.issues] == ["sensor_missing"]
    assert not result.rejected_records


def test_parse_quarantines_ambiguous_column_count(tmp_path: Path) -> None:
    source_path = tmp_path / "train_FD001.txt"
    source_path.write_text(" ".join(_telemetry_row()[:-1]) + "\n", encoding="utf-8")

    result = parse_telemetry_file(source_path, subset_id="FD001", split="train")

    assert result.frame.is_empty()
    assert result.issues[0].rule_id == "telemetry_column_count"
    assert len(result.rejected_records) == 1


def test_validation_rejects_cycle_gap(tmp_path: Path) -> None:
    train_path = tmp_path / "train_FD001.txt"
    test_path = tmp_path / "test_FD001.txt"
    rul_path = tmp_path / "RUL_FD001.txt"
    train_path.write_text(
        "\n".join(
            (
                " ".join(_telemetry_row(cycle=1)),
                " ".join(_telemetry_row(cycle=3)),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    test_path.write_text(" ".join(_telemetry_row()) + "\n", encoding="utf-8")
    rul_path.write_text("10\n", encoding="utf-8")
    train_result = parse_telemetry_file(train_path, subset_id="FD001", split="train")
    test_result = parse_telemetry_file(test_path, subset_id="FD001", split="test")
    rul_result = parse_test_rul_file(
        rul_path,
        subset_id="FD001",
        test_frame=test_result.frame,
    )

    report = validate_fd001(train_result, test_result, rul_result)

    assert report.status == "FAILED"
    assert "engine_cycles_contiguous" in {issue.rule_id for issue in report.issues}


def test_test_rul_count_must_match_test_engines(tmp_path: Path) -> None:
    test_path = tmp_path / "test_FD001.txt"
    rul_path = tmp_path / "RUL_FD001.txt"
    test_path.write_text(
        "\n".join(
            (
                " ".join(_telemetry_row(engine_id=1)),
                " ".join(_telemetry_row(engine_id=2)),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    rul_path.write_text("10\n", encoding="utf-8")
    test_result = parse_telemetry_file(test_path, subset_id="FD001", split="test")

    result = parse_test_rul_file(
        rul_path,
        subset_id="FD001",
        test_frame=test_result.frame,
    )

    assert result.frame.is_empty()
    assert "rul_engine_count_match" in {issue.rule_id for issue in result.issues}
