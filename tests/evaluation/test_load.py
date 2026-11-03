"""Tests for the generic concurrent load-test harness."""
import threading
import time

import pytest

from parking_chatbot.evaluation.load import run_concurrent, summarize


def test_run_concurrent_runs_every_call_exactly_once():
    seen = []
    lock = threading.Lock()

    def _call(index: int) -> None:
        with lock:
            seen.append(index)

    result = run_concurrent(_call, total_calls=20, concurrency=4)

    assert result.total_calls == 20
    assert sorted(seen) == list(range(20))
    assert result.error_rate == 0.0


def test_run_concurrent_overlaps_calls_up_to_the_given_concurrency():
    """A low-concurrency run of calls that each sleep must take noticeably
    longer than a high-concurrency run of the same calls - proof that work is
    actually running in parallel, not serialized despite the concurrency argument."""

    def _call(index: int) -> None:
        time.sleep(0.03)

    serial = run_concurrent(_call, total_calls=8, concurrency=1)
    parallel = run_concurrent(_call, total_calls=8, concurrency=8)

    assert parallel.wall_clock_seconds < serial.wall_clock_seconds / 2


def test_run_concurrent_records_a_failure_without_aborting_the_rest():
    def _call(index: int) -> None:
        if index == 3:
            raise ValueError("boom")

    result = run_concurrent(_call, total_calls=6, concurrency=2)

    assert result.total_calls == 6
    assert len(result.failures) == 1
    assert result.error_rate == pytest.approx(1 / 6)
    assert "boom" in result.failures[0].error


def test_run_concurrent_rejects_a_non_positive_total_or_concurrency():
    with pytest.raises(ValueError):
        run_concurrent(lambda i: None, total_calls=0, concurrency=1)
    with pytest.raises(ValueError):
        run_concurrent(lambda i: None, total_calls=5, concurrency=0)


def test_summarize_reports_percentiles_error_rate_and_sample_errors():
    def _call(index: int) -> None:
        time.sleep(0.01)
        if index < 2:
            raise RuntimeError(f"failure {index}")

    result = run_concurrent(_call, total_calls=10, concurrency=5)
    stats = summarize(result)

    assert stats["total_calls"] == 10
    assert stats["errors"] == 2
    assert stats["error_rate"] == pytest.approx(0.2)
    assert stats["mean"] > 0
    assert stats["p50"] <= stats["p95"] <= stats["p99"] <= stats["max"]
    assert len(stats["sample_errors"]) == 2


def test_summarize_of_an_all_failing_run_reports_zeroed_latencies_not_a_crash():
    def _call(index: int) -> None:
        raise RuntimeError("always fails")

    stats = summarize(run_concurrent(_call, total_calls=3, concurrency=3))

    assert stats["error_rate"] == 1.0
    assert stats["mean"] == 0.0
    assert stats["p99"] == 0.0
