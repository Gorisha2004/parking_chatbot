"""Reusable statistics helpers shared by the evaluation scripts."""
from __future__ import annotations

import math


def percentile(values: list[float], pct: float) -> float:
    """Return the pct-th percentile (0-100) of values using linear interpolation."""
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list.")
    if not 0 <= pct <= 100:
        raise ValueError("pct must be between 0 and 100.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_latencies(values: list[float]) -> dict[str, float]:
    """Return mean, p95 and max for a list of latency measurements, in seconds."""
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p95": percentile(values, 95),
        "max": max(values),
    }
