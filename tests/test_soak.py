"""S1 / A1 / A2: the soak run's verdict logic (unit) and the run itself (marker ``soak``).

The unit tests cover the pure parts of ``scripts/soak.py``: percentiles, how latencies are
collected from the capture's sliding window, the RSS growth rule and the pass / fail verdict.
The soak test plays looped speech through a VB-Cable for ``PIPELINE_SOAK_MINUTES`` minutes
(default 60) with the other cores busy; run it only on request: ``pytest -m soak -s``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("soak", ROOT / "scripts" / "soak.py")
assert _SPEC is not None
assert _SPEC.loader is not None
soak = importlib.util.module_from_spec(_SPEC)
# dataclasses look the module up by name while the class body runs
import sys  # noqa: E402

sys.modules["soak"] = soak
_SPEC.loader.exec_module(soak)


def good_report(**changes: object) -> object:
    values: dict[str, object] = {
        "minutes": 60.0,
        "chunks": 1000,
        "expected_chunks": 1000,
        "input_overflow": 0,
        "ring_lost_frames": 0,
        "max_callback_ms": 1.2,
        "p99_finalization_ms": 31.0,
        "max_finalization_ms": 40.0,
        "output_underruns": 0,
        "driver_underflows": 0,
        "reconnects": 0,
        "rss_growth_mb_per_h": 3.0,
        "write_operations": 0,
        "write_bytes": 0,
    }
    values.update(changes)
    return soak.SoakReport(**values)


# --- percentile -----------------------------------------------------------------------------


@pytest.mark.unit
def test_percentile_of_a_hundred_values() -> None:
    values = [float(i) for i in range(1, 101)]
    assert soak.percentile(values, 50) == pytest.approx(50.5)
    assert soak.percentile(values, 99) == pytest.approx(99.01)
    assert soak.percentile(values, 100) == 100.0


@pytest.mark.unit
def test_percentile_of_one_value_is_that_value() -> None:
    assert soak.percentile([7.0], 99) == 7.0


@pytest.mark.unit
def test_percentile_of_nothing_is_zero() -> None:
    assert soak.percentile([], 99) == 0.0


# --- collecting latencies from a sliding window ---------------------------------------------


@pytest.mark.unit
def test_new_values_are_what_the_window_gained() -> None:
    assert soak.new_values((1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0, 5.0)) == [4.0, 5.0]


@pytest.mark.unit
def test_nothing_new_gives_nothing() -> None:
    assert soak.new_values((1.0, 2.0), (1.0, 2.0)) == []


@pytest.mark.unit
def test_chunks_that_ended_without_silence_add_no_value() -> None:
    # Only chunks that ended on silence record a latency, so the window gains fewer values than
    # there are new chunks; counting by length keeps the old values out.
    assert soak.new_values((1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0)) == [4.0]


@pytest.mark.unit
def test_a_full_window_cannot_be_compared_by_length_so_the_chunk_gain_is_used() -> None:
    before = tuple(float(i) for i in range(1000))
    after = tuple(float(i) for i in range(2, 1002))
    assert soak.new_values(before, after, window=1000, chunks_gained=2) == [1000.0, 1001.0]


@pytest.mark.unit
def test_a_chunk_gain_larger_than_the_window_returns_the_whole_window() -> None:
    after = tuple(float(i) for i in range(5))
    assert soak.new_values(after, after, window=5, chunks_gained=50) == list(after)


# --- RSS growth -----------------------------------------------------------------------------


@pytest.mark.unit
def test_rss_growth_is_measured_from_after_the_warm_up() -> None:
    samples = [(0.0, 100.0), (300.0, 150.0), (3600.0, 156.0)]
    # 100 -> 150 is start-up (models, caches); only 150 -> 156 over 3300 s is growth.
    assert soak.rss_growth_mb_per_hour(samples, warmup_s=300.0) == pytest.approx(6.0 * 3600 / 3300)


@pytest.mark.unit
def test_rss_that_shrinks_is_no_growth() -> None:
    assert soak.rss_growth_mb_per_hour([(0.0, 100.0), (400.0, 90.0), (3600.0, 80.0)], 300.0) == 0.0


@pytest.mark.unit
def test_rss_with_too_little_data_gives_zero() -> None:
    assert soak.rss_growth_mb_per_hour([(0.0, 100.0)], 300.0) == 0.0
    assert soak.rss_growth_mb_per_hour([(0.0, 100.0), (100.0, 400.0)], 300.0) == 0.0


# --- verdict --------------------------------------------------------------------------------


@pytest.mark.unit
def test_a_clean_run_has_no_failures() -> None:
    assert soak.evaluate(good_report()) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"input_overflow": 1}, "overflow"),
        ({"ring_lost_frames": 160}, "ring"),
        ({"max_callback_ms": 5.0}, "callback"),
        ({"p99_finalization_ms": 50.5}, "finalization"),
        ({"rss_growth_mb_per_h": 20.0}, "RSS"),
        ({"output_underruns": 1}, "underrun"),
        ({"driver_underflows": 1}, "underflow"),
        ({"reconnects": 1}, "reconnect"),
        ({"write_operations": 1, "write_bytes": 10}, "write"),
        ({"chunks": 900}, "chunks"),
        ({"chunks": 1100}, "chunks"),
    ],
)
def test_each_criterion_fails_on_its_own(changes: dict[str, object], fragment: str) -> None:
    failures = soak.evaluate(good_report(**changes))
    assert len(failures) == 1
    assert fragment in failures[0]


@pytest.mark.unit
def test_the_limits_themselves_pass() -> None:
    report = good_report(max_callback_ms=4.99, p99_finalization_ms=50.0, rss_growth_mb_per_h=19.9)
    assert soak.evaluate(report) == []


@pytest.mark.unit
def test_a_few_chunks_of_difference_are_tolerated() -> None:
    assert soak.evaluate(good_report(chunks=998)) == []


@pytest.mark.unit
def test_every_failure_is_listed() -> None:
    failures = soak.evaluate(good_report(input_overflow=3, max_callback_ms=9.0))
    assert len(failures) == 2


@pytest.mark.unit
def test_a_run_that_heard_nothing_fails_even_if_the_counters_are_zero() -> None:
    assert soak.evaluate(good_report(chunks=0, expected_chunks=0))


# --- the report text ------------------------------------------------------------------------


@pytest.mark.unit
def test_the_report_lists_every_criterion_with_its_limit() -> None:
    text = soak.format_report(good_report())
    for fragment in ("overflow", "callback", "finalization", "RSS", "underrun", "write"):
        assert fragment in text
    assert "PASS" in text


@pytest.mark.unit
def test_the_report_says_fail_and_why() -> None:
    text = soak.format_report(good_report(input_overflow=2))
    assert "FAIL" in text
    assert "overflow" in text


# --- the run (marker soak, only on request) -------------------------------------------------


@pytest.mark.soak
def test_s1_looped_speech_through_the_cable_under_load() -> None:
    minutes = float(os.environ.get("PIPELINE_SOAK_MINUTES", "60"))
    report = soak.run_soak(soak.SoakOptions(minutes=minutes))
    print(soak.format_report(report))
    assert soak.evaluate(report) == []
