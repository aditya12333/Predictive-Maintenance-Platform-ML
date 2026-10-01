"""Small, explainable distribution-drift checks for numeric features."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite, log


@dataclass(frozen=True)
class DriftResult:
    """Population Stability Index result for one monitored feature."""

    psi: float
    threshold: float
    drifted: bool
    reference_count: int
    current_count: int


def calculate_psi(
    reference: Iterable[float],
    current: Iterable[float],
    *,
    bins: int = 10,
    threshold: float = 0.2,
) -> DriftResult:
    """Compare two numeric populations using reference quantile bins.

    PSI is a monitoring signal, not a retraining decision. The caller must choose
    a reference window and investigate the affected feature before acting.
    """

    if bins < 2:
        raise ValueError("bins must be at least 2")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    reference_values = _finite_values(reference, "reference")
    current_values = _finite_values(current, "current")
    if len(reference_values) < bins:
        raise ValueError("reference population must contain at least bins values")
    if not current_values:
        raise ValueError("current population must not be empty")

    ordered = sorted(reference_values)
    edges = [ordered[(index * len(ordered)) // bins] for index in range(1, bins)]
    edges = sorted(set(edges))
    boundaries = [-float("inf"), *edges, float("inf")]
    reference_counts = _histogram(reference_values, boundaries)
    current_counts = _histogram(current_values, boundaries)
    reference_total = float(len(reference_values))
    current_total = float(len(current_values))
    epsilon = 1e-6
    psi = 0.0
    for reference_count, current_count in zip(reference_counts, current_counts, strict=True):
        reference_rate = max(reference_count / reference_total, epsilon)
        current_rate = max(current_count / current_total, epsilon)
        psi += (current_rate - reference_rate) * log(current_rate / reference_rate)
    return DriftResult(
        psi=psi,
        threshold=threshold,
        drifted=psi >= threshold,
        reference_count=len(reference_values),
        current_count=len(current_values),
    )


def _finite_values(values: Iterable[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not isfinite(value) for value in result):
        raise ValueError(f"{name} values must be finite and non-empty")
    return result


def _histogram(values: list[float], boundaries: list[float]) -> list[int]:
    counts = [0] * (len(boundaries) - 1)
    for value in values:
        for index in range(len(counts)):
            if boundaries[index] <= value < boundaries[index + 1] or (
                index == len(counts) - 1 and value == boundaries[index + 1]
            ):
                counts[index] += 1
                break
    return counts
