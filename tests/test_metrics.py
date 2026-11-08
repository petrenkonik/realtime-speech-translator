"""In-memory counters and histograms."""

from __future__ import annotations

import threading

import pytest

from pipeline.metrics import Counter, Histogram, Metrics, MetricsError

pytestmark = pytest.mark.unit


def test_counter_counts_and_refuses_to_go_down() -> None:
    counter = Counter()
    counter.inc()
    counter.inc(4)
    assert counter.value == 5
    with pytest.raises(MetricsError, match="decrease"):
        counter.inc(-1)


def test_counter_is_exact_under_concurrent_increments() -> None:
    counter = Counter()

    def work() -> None:
        for _ in range(5_000):
            counter.inc()

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert counter.value == 20_000


def test_empty_histogram_summary_is_all_zero() -> None:
    summary = Histogram().summary()
    assert (summary.count, summary.mean, summary.p50, summary.p95, summary.max) == (0, 0, 0, 0, 0)


def test_histogram_percentiles_use_nearest_rank() -> None:
    histogram = Histogram()
    for value in range(1, 101):
        histogram.record(float(value))
    summary = histogram.summary()
    assert summary.count == 100
    assert summary.mean == pytest.approx(50.5)
    assert summary.p50 == 50.0
    assert summary.p95 == 95.0
    assert summary.max == 100.0


def test_histogram_window_forgets_old_values_but_keeps_the_lifetime_count() -> None:
    histogram = Histogram(window=3)
    for value in (100.0, 1.0, 2.0, 3.0):
        histogram.record(value)
    summary = histogram.summary()
    assert summary.count == 4
    assert summary.max == 3.0  # the 100 fell out of the window


def test_histogram_window_must_be_positive() -> None:
    with pytest.raises(MetricsError, match="window"):
        Histogram(0)


def test_registry_returns_the_same_metric_for_the_same_name() -> None:
    metrics = Metrics()
    assert metrics.counter("overflow") is metrics.counter("overflow")
    assert metrics.histogram("rtf") is metrics.histogram("rtf", window=5)


def test_a_name_cannot_be_both_counter_and_histogram() -> None:
    metrics = Metrics()
    metrics.counter("x")
    with pytest.raises(MetricsError, match="counter"):
        metrics.histogram("x")
    metrics.histogram("y")
    with pytest.raises(MetricsError, match="histogram"):
        metrics.counter("y")


@pytest.mark.parametrize("name", ["", " lead", "trail "])
def test_bad_names_are_rejected(name: str) -> None:
    with pytest.raises(MetricsError, match="name"):
        Metrics().counter(name)


def test_snapshot_flattens_everything_in_name_order() -> None:
    metrics = Metrics()
    metrics.counter("b.drops").inc(2)
    metrics.histogram("a.latency").record(10.0)
    snapshot = metrics.snapshot()
    assert list(snapshot) == sorted(snapshot)
    assert snapshot["b.drops"] == 2.0
    assert snapshot["a.latency.count"] == 1.0
    assert snapshot["a.latency.p95"] == 10.0
