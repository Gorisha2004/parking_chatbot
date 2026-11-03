"""A generic, reusable load-test harness: run one callable many times at a given
concurrency, and summarize how long it took and how often it failed.

Deliberately has no idea what it is calling - scenario-specific setup (building an
agent, seeding bookings, starting the MCP app) lives in load_scenarios.py, which is
what keeps this module small enough to unit test on its own (SOLID: single
responsibility) and reusable for any future component that needs load numbers.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from parking_chatbot.evaluation.metrics import percentile


@dataclass(frozen=True)
class CallOutcome:
    """What happened on one call: how long it took, and its error if it raised."""

    latency_seconds: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class LoadResult:
    """Every call's outcome, plus the wall-clock time the whole run took."""

    outcomes: list[CallOutcome]
    wall_clock_seconds: float

    @property
    def total_calls(self) -> int:
        return len(self.outcomes)

    @property
    def failures(self) -> list[CallOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.ok]

    @property
    def error_rate(self) -> float:
        return len(self.failures) / self.total_calls if self.total_calls else 0.0

    @property
    def throughput_per_second(self) -> float:
        return self.total_calls / self.wall_clock_seconds if self.wall_clock_seconds > 0 else 0.0


def run_concurrent(call: Callable[[int], None], total_calls: int, concurrency: int) -> LoadResult:
    """Run `call(i)` once for each i in range(total_calls), `concurrency` at a time.

    A failing call is recorded as an outcome, not raised - one slow or broken
    request must not abort the rest of the run, which is also exactly the
    property being measured (the error rate).
    """
    if total_calls < 1:
        raise ValueError("total_calls must be at least 1.")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1.")

    def _timed(index: int) -> CallOutcome:
        started = time.perf_counter()
        try:
            call(index)
        except Exception as error:  # noqa: BLE001 - deliberately broad: any failure is a load-test data point
            return CallOutcome(latency_seconds=time.perf_counter() - started, error=f"{type(error).__name__}: {error}")
        return CallOutcome(latency_seconds=time.perf_counter() - started)

    started_run = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_timed, i) for i in range(total_calls)]
        outcomes = [future.result() for future in as_completed(futures)]
    return LoadResult(outcomes=outcomes, wall_clock_seconds=time.perf_counter() - started_run)


def summarize(result: LoadResult) -> dict:
    """Latency percentiles (successful calls only) plus the error rate and
    throughput for a load result - the numbers the report table shows."""
    latencies = [outcome.latency_seconds for outcome in result.outcomes if outcome.ok]
    if not latencies:
        percentiles = {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    else:
        percentiles = {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "mean": sum(latencies) / len(latencies),
            "max": max(latencies),
        }
    return {
        "total_calls": result.total_calls,
        "errors": len(result.failures),
        "error_rate": result.error_rate,
        "throughput_per_second": result.throughput_per_second,
        "sample_errors": sorted({outcome.error for outcome in result.failures})[:5],
        **percentiles,
    }
