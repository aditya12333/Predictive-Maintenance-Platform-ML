"""Tests for the initial numeric drift signal."""

import pytest

from predictive_maintenance.monitoring.drift import calculate_psi


def test_psi_is_low_for_matching_populations() -> None:
    result = calculate_psi(range(100), range(100), bins=10)

    assert result.psi == pytest.approx(0.0)
    assert result.drifted is False


def test_psi_detects_shifted_population() -> None:
    result = calculate_psi(range(100), range(1000, 1100), bins=10)

    assert result.psi > 0.2
    assert result.drifted is True


def test_psi_rejects_invalid_populations() -> None:
    with pytest.raises(ValueError, match="finite"):
        calculate_psi([1.0, float("nan")], [1.0, 2.0])
