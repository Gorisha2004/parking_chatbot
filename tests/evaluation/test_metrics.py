import pytest

from parking_chatbot.evaluation.metrics import percentile, summarize_latencies


def test_percentile_of_sorted_values():
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 0) == 1
    assert percentile(values, 50) == 3
    assert percentile(values, 100) == 5


def test_percentile_interpolates_between_points():
    assert percentile([1, 2], 50) == 1.5


def test_percentile_rejects_empty_list():
    with pytest.raises(ValueError):
        percentile([], 95)


def test_summarize_latencies_reports_mean_p95_and_max():
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["mean"] == 3.0
    assert summary["max"] == 5.0
    assert summary["p95"] == pytest.approx(4.8)


def test_summarize_latencies_of_empty_list_returns_zeros():
    assert summarize_latencies([]) == {"mean": 0.0, "p95": 0.0, "max": 0.0}
