"""Dashboard rendering and the background thread."""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable

import pytest
from fakes import silence
from rich.console import Console
from test_capture import build, play

from pipeline.bus import Bus
from pipeline.dashboard import (
    Dashboard,
    Row,
    bus_rows,
    capture_rows,
    metrics_rows,
    output_rows,
    render_live,
    render_plain,
)
from pipeline.metrics import Metrics

pytestmark = pytest.mark.unit


def sections() -> dict[str, Callable[[], list[Row]]]:
    return {
        "capture": lambda: [("device", "V380"), ("overflow", "0")],
        "bus": lambda: [("whisper", "depth 1")],
    }


def wait_for(predicate: Callable[[], bool], timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def output_of(console: Console) -> str:
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


def test_plain_rendering_is_one_line_with_all_sections() -> None:
    line = render_plain(sections())
    assert "\n" not in line
    assert line == "[capture] device=V380 overflow=0 [bus] whisper=depth 1"


def test_a_failing_section_shows_the_error_and_the_rest_still_renders() -> None:
    def broken() -> list[Row]:
        raise RuntimeError("boom")

    line = render_plain({"bad": broken, "good": lambda: [("a", "1")]})
    assert "RuntimeError: boom" in line
    assert "[good] a=1" in line


def test_live_rendering_contains_titles_labels_and_values() -> None:
    console = Console(file=io.StringIO(), width=80, force_terminal=False)
    console.print(render_live(sections()))
    text = output_of(console)
    for expected in ("capture", "device", "V380", "bus", "whisper", "depth 1"):
        assert expected in text


def test_capture_rows_describe_the_device_and_counters() -> None:
    rig = build()
    play(rig, silence(200, 48_000))
    rows = dict(capture_rows(rig.session))
    assert rows["device"] == "Microphone (V380 USB)"
    assert rows["connected"] == "yes"
    assert rows["callbacks"] == "20"
    assert rows["overflow"] == "0"


def test_capture_rows_flag_a_lost_stream_and_the_last_error() -> None:
    rig = build()
    rig.factory.current.fail()
    rig.session.tick()
    rig.devices.remove("Microphone (V380 USB)")
    rig.session.tick()
    rows = dict(capture_rows(rig.session))
    assert rows["connected"] == "NO"
    assert rows["device"] == "not connected"
    assert "no input device" in rows["last error"]


def test_bus_rows_show_depth_and_drops_per_subscriber() -> None:
    bus: Bus[int] = Bus()
    bus.subscribe("meter", maxsize=1)
    bus.publish(1)
    bus.publish(2)
    ((name, value),) = bus_rows(bus)
    assert name == "meter"
    assert "depth 1" in value
    assert "dropped 1" in value


def test_metrics_rows_list_every_metric() -> None:
    metrics = Metrics()
    metrics.counter("underruns").inc(3)
    assert ("underruns", "3") in metrics_rows(metrics)


def test_plain_mode_keeps_printing_until_stopped() -> None:
    lines: list[str] = []
    dashboard = Dashboard(sections(), mode="plain", plain_interval_s=0.01, write=lines.append)
    dashboard.start()
    assert wait_for(lambda: len(lines) >= 3)
    dashboard.stop()
    count = len(lines)
    time.sleep(0.05)
    assert len(lines) == count
    assert lines[0].startswith("[capture]")


def test_live_mode_draws_and_stops_cleanly() -> None:
    console = Console(file=io.StringIO(), width=80, force_terminal=True)
    dashboard = Dashboard(sections(), mode="live", console=console)
    dashboard.start()
    assert wait_for(lambda: "V380" in output_of(console))
    started = time.perf_counter()
    dashboard.stop()
    assert time.perf_counter() - started < 1.0
    assert not any(t.name == "dashboard" for t in threading.enumerate())


def test_start_twice_is_an_error() -> None:
    dashboard = Dashboard(sections(), mode="plain", plain_interval_s=0.05, write=lambda _: None)
    dashboard.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            dashboard.start()
    finally:
        dashboard.stop()


def test_output_rows_show_underruns_and_the_lost_device() -> None:
    from test_output import build as build_output

    rig = build_output(prebuffer_ms=0)
    rows = dict(output_rows(rig.session))
    assert rows["device"] == "Speakers (Realtek Audio)"
    assert rows["connected"] == "yes"
    assert rows["underruns"] == "0"

    rig.factory.current.fail()
    rig.session.tick()
    rig.devices.remove("Speakers (Realtek Audio)")
    rig.session.tick()
    rows = dict(output_rows(rig.session))
    assert rows["connected"] == "NO"
    assert rows["device"] == "not connected"
    assert "no output device" in rows["last error"]
