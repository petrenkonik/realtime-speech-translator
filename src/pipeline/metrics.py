"""In-memory counters and histograms. Nothing is ever written to disk.

Used by the stages that are not already counted elsewhere (per-stage latency, whisper
real-time factor, lane delays). Everything here is called from worker threads, never from
the audio callback, so a plain lock per metric is fine.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


class MetricsError(ValueError):
    """A metric was requested with a name or window that cannot work."""


class Counter:
    """A number that only goes up."""

    def __init__(self) -> None:
        """Start at zero."""
        self._lock = threading.Lock()
        self._value = 0

    def inc(self, amount: int = 1) -> None:
        """Add ``amount`` (default 1); negative amounts are rejected."""
        if amount < 0:
            raise MetricsError(f"a counter cannot decrease, got {amount}")
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        """The current total."""
        with self._lock:
            return self._value


@dataclass(frozen=True, slots=True)
class HistogramSummary:
    """Statistics of the values currently inside a histogram window.

    ``count`` is the lifetime number of recorded values; the other fields describe
    only the most recent ``window`` values. All are 0.0 while the histogram is empty.
    """

    count: int
    mean: float
    p50: float
    p95: float
    max: float


class Histogram:
    """The last ``window`` recorded values with percentile summaries."""

    def __init__(self, window: int = 1000) -> None:
        """Create an empty histogram.

        Args:
            window: How many recent values to keep (at least 1).
        """
        if window < 1:
            raise MetricsError(f"window must be at least 1, got {window}")
        self._lock = threading.Lock()
        self._values: deque[float] = deque(maxlen=window)
        self._count = 0

    def record(self, value: float) -> None:
        """Add one value, forgetting the oldest if the window is full."""
        with self._lock:
            self._values.append(value)
            self._count += 1

    def summary(self) -> HistogramSummary:
        """Return statistics of the current window."""
        with self._lock:
            values = sorted(self._values)
            count = self._count
        if not values:
            return HistogramSummary(count, 0.0, 0.0, 0.0, 0.0)
        return HistogramSummary(
            count=count,
            mean=sum(values) / len(values),
            p50=_percentile(values, 50),
            p95=_percentile(values, 95),
            max=values[-1],
        )


def _percentile(sorted_values: list[float], percent: float) -> float:
    # Nearest-rank: the smallest value with at least `percent` of the data at or below it.
    rank = max(1, -(-len(sorted_values) * percent // 100))
    return sorted_values[int(rank) - 1]


class Metrics:
    """A registry of named counters and histograms."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._lock = threading.Lock()
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}

    def counter(self, name: str) -> Counter:
        """Return the counter called ``name``, creating it on first use."""
        _check_name(name)
        with self._lock:
            if name in self._histograms:
                raise MetricsError(f"{name!r} is already a histogram")
            return self._counters.setdefault(name, Counter())

    def histogram(self, name: str, window: int = 1000) -> Histogram:
        """Return the histogram called ``name``, creating it on first use.

        ``window`` only matters when the histogram is created.
        """
        _check_name(name)
        with self._lock:
            if name in self._counters:
                raise MetricsError(f"{name!r} is already a counter")
            if name not in self._histograms:
                self._histograms[name] = Histogram(window)
            return self._histograms[name]

    def snapshot(self) -> dict[str, float]:
        """Flatten everything into ``name -> number`` (histograms become ``name.p50`` etc.)."""
        with self._lock:
            counters = dict(self._counters)
            histograms = dict(self._histograms)
        flat: dict[str, float] = {name: float(c.value) for name, c in counters.items()}
        for name, histogram in histograms.items():
            summary = histogram.summary()
            flat[f"{name}.count"] = float(summary.count)
            flat[f"{name}.mean"] = summary.mean
            flat[f"{name}.p50"] = summary.p50
            flat[f"{name}.p95"] = summary.p95
            flat[f"{name}.max"] = summary.max
        return dict(sorted(flat.items()))


def _check_name(name: str) -> None:
    if not name or name != name.strip():
        raise MetricsError(f"metric name must be non-empty without surrounding spaces: {name!r}")
