"""Live status display (``rich``) and a plain one-line mode for long runs.

The dashboard knows nothing about the stages. It is given *sections*, each a function
returning ``(label, value)`` rows, and redraws them. Stages add their own sections as they
are built; helpers here cover capture, buses and metrics. Rows hold numbers and device
names only: utterance text is shown by a separate section that honors ``--hide-text``.
Nothing is written to a file.
"""

from __future__ import annotations

import statistics
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from pipeline.bus import Bus
from pipeline.capture import CaptureSession
from pipeline.metrics import Metrics
from pipeline.output import OutputSession

Row = tuple[str, str]
RowProvider = Callable[[], Sequence[Row]]
DashboardMode = Literal["live", "plain"]

REFRESH_HZ = 4


def capture_rows(session: CaptureSession) -> list[Row]:
    """Rows describing the microphone: device, health counters, finalization latency."""
    stats = session.stats()
    device = session.device
    latencies = stats.finalization_latency_ms
    rows: list[Row] = [
        ("device", device.name if device else "not connected"),
        ("connected", "yes" if stats.connected else "NO"),
        ("callbacks", str(stats.callbacks)),
        ("overflow", str(stats.input_overflow)),
        ("ring lost", str(stats.ring_lost_frames)),
        ("callback max", f"{stats.max_callback_ms:.2f} ms"),
        ("reconnects", str(stats.reconnects)),
        ("chunks", str(stats.chunks)),
    ]
    if latencies:
        rows.append(
            (
                "finalize +",
                f"median {statistics.median(latencies):.0f} ms, max {max(latencies):.0f} ms",
            )
        )
    if stats.last_error:
        rows.append(("last error", stats.last_error))
    if session.fatal_error is not None:
        rows.append(("FATAL", f"{type(session.fatal_error).__name__}: {session.fatal_error}"))
    return rows


def output_rows(session: OutputSession) -> list[Row]:
    """Rows describing one output device: connection, underruns, queued audio."""
    stats = session.stats()
    device = session.device
    rows: list[Row] = [
        ("device", device.name if device else "not connected"),
        ("connected", "yes" if stats.connected else "NO"),
        ("underruns", str(stats.underruns)),
        ("driver underflow", str(stats.driver_underflows)),
        ("queued", f"{stats.queued_ms:.0f} ms"),
        ("dropped", f"{stats.dropped_frames} frames"),
        ("callback max", f"{stats.max_callback_ms:.2f} ms"),
        ("reconnects", str(stats.reconnects)),
    ]
    if stats.last_error:
        rows.append(("last error", stats.last_error))
    if session.fatal_error is not None:
        rows.append(("FATAL", f"{type(session.fatal_error).__name__}: {session.fatal_error}"))
    return rows


def bus_rows(bus: Bus[Any]) -> list[Row]:
    """One row per subscriber: queue depth, items delivered and dropped."""
    return [
        (name, f"depth {s.depth} (max {s.max_depth}), delivered {s.delivered}, dropped {s.dropped}")
        for name, s in bus.stats().items()
    ]


def metrics_rows(metrics: Metrics) -> list[Row]:
    """Every metric as a row, in name order."""
    return [(name, f"{value:g}") for name, value in metrics.snapshot().items()]


def _safe_rows(provider: RowProvider) -> list[Row]:
    # A broken section must not take the whole display down.
    try:
        return list(provider())
    except Exception as error:
        return [("error", f"{type(error).__name__}: {error}")]


def render_plain(sections: Mapping[str, RowProvider]) -> str:
    """Render all sections as one line, for ``--no-ui``."""
    parts = []
    for title, provider in sections.items():
        fields = " ".join(f"{label}={value}" for label, value in _safe_rows(provider))
        parts.append(f"[{title}] {fields}")
    return " ".join(parts)


def render_live(sections: Mapping[str, RowProvider]) -> RenderableType:
    """Render all sections as stacked ``rich`` panels."""
    panels = []
    for title, provider in sections.items():
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        for label, value in _safe_rows(provider):
            table.add_row(label, value)
        panels.append(Panel(table, title=title, title_align="left"))
    return Group(*panels)


class Dashboard:
    """Redraws the sections from a background thread until ``stop()``."""

    def __init__(
        self,
        sections: Mapping[str, RowProvider],
        mode: DashboardMode = "live",
        plain_interval_s: float = 1.0,
        console: Console | None = None,
        write: Callable[[str], None] = print,
    ) -> None:
        """Create a stopped dashboard.

        Args:
            sections: Title -> row provider, shown in order.
            mode: ``"live"`` for the ``rich`` display at 4 Hz, ``"plain"`` for one text
                line per ``plain_interval_s`` (long soak runs, redirected output).
            plain_interval_s: Seconds between plain lines.
            console: Console for live mode; a default one is created when omitted.
            write: Where plain lines go.
        """
        self._sections = sections
        self._mode = mode
        self._plain_interval_s = plain_interval_s
        self._console = console
        self._write = write
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start drawing in a background thread."""
        if self._thread is not None:
            raise RuntimeError("the dashboard is already started")
        self._stop.clear()
        target = self._run_live if self._mode == "live" else self._run_plain
        self._thread = threading.Thread(target=target, name="dashboard", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop drawing and wait for the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            self._thread = None

    def _run_plain(self) -> None:
        while True:
            self._write(render_plain(self._sections))
            if self._stop.wait(self._plain_interval_s):
                return

    def _run_live(self) -> None:
        with Live(
            render_live(self._sections),
            console=self._console,
            auto_refresh=False,
            transient=False,
        ) as live:
            while not self._stop.wait(1 / REFRESH_HZ):
                live.update(render_live(self._sections), refresh=True)
            live.update(render_live(self._sections), refresh=True)
