"""Keeps one audio stream alive: watchdog, close, search by name, reopen.

Capture and every output need the same behavior when a USB device disappears, so it
lives here once. The supervisor knows nothing about what flows through the stream; the
owner supplies a *link* (the stream plus whatever state belongs to one connection) and
hooks to flush or account for it when the connection ends.

Time comes from an injected clock, so tests drive the watchdog with a fake one.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pipeline.devices import (
    AmbiguousDeviceError,
    AudioStream,
    DeviceInfo,
    DeviceKind,
    DeviceListing,
    DeviceNotFoundError,
    StreamOpenError,
    find_device,
)


class SupervisedLink(Protocol):
    """One open connection: its device, its stream and a count of audio callbacks."""

    @property
    def device(self) -> DeviceInfo:
        """The device this connection belongs to."""
        ...

    @property
    def stream(self) -> AudioStream:
        """The open stream."""
        ...

    @property
    def callbacks(self) -> int:
        """How many audio callbacks happened; the watchdog watches it grow."""
        ...


L = TypeVar("L", bound=SupervisedLink)


@dataclass(frozen=True, slots=True)
class SupervisorStats:
    """Connection health counters.

    Attributes:
        connected: A stream is open.
        losses: Times the stream was declared lost.
        reconnects: Successful reconnections after a loss (not the first connection).
        last_loss_reason: Why the last loss was declared, or ``""``.
        last_error: The last failed connection attempt, or ``""``.
    """

    connected: bool
    losses: int
    reconnects: int
    last_loss_reason: str
    last_error: str


class Supervisor(Generic[L]):
    """Opens a device by name and reconnects it when the stream dies."""

    def __init__(
        self,
        query: str,
        kind: DeviceKind,
        devices: DeviceListing,
        open_link: Callable[[DeviceInfo], L],
        watchdog_ms: int,
        retry_ms: int,
        clock: Callable[[], float] = time.monotonic,
        before_close: Callable[[L, str], None] | None = None,
        after_close: Callable[[L], None] | None = None,
    ) -> None:
        """Create a supervisor; nothing is opened until ``connect()``.

        Args:
            query: Substring of the device name.
            kind: ``"input"`` or ``"output"``.
            devices: Device list source (PortAudio or a fake).
            open_link: Builds a link (including its stream) for a device. It may raise
                ``StreamOpenError``; other exceptions are not retried.
            watchdog_ms: No callback for this long means the stream is lost.
            retry_ms: Pause between reconnection attempts.
            clock: Monotonic seconds.
            before_close: Called with the link and the reason when a loss is declared,
                while the link is still current (flush pending data here).
            after_close: Called whenever a link has been closed (loss or ``close()``).
        """
        self._query = query
        self._kind: DeviceKind = kind
        self._devices = devices
        self._open_link = open_link
        self._watchdog_ms = watchdog_ms
        self._retry_s = retry_ms / 1000
        self._clock = clock
        self._before_close = before_close
        self._after_close = after_close
        self._link: L | None = None
        self._ever_connected = False
        self._next_attempt_s = 0.0
        self._seen_callbacks = 0
        self._seen_at_s = 0.0
        self._losses = 0
        self._reconnects = 0
        self._last_loss_reason = ""
        self._last_error = ""

    @property
    def link(self) -> L | None:
        """The current connection, or ``None`` while disconnected."""
        return self._link

    def stats(self) -> SupervisorStats:
        """Return a snapshot of the health counters."""
        return SupervisorStats(
            connected=self._link is not None,
            losses=self._losses,
            reconnects=self._reconnects,
            last_loss_reason=self._last_loss_reason,
            last_error=self._last_error,
        )

    def connect(self) -> None:
        """Find the device and open its stream; raise if that is not possible.

        Raises:
            DeviceNotFoundError: No device matches the name.
            AmbiguousDeviceError: Several devices match.
            StreamOpenError: The device was found but could not be opened or started.
        """
        device = find_device(self._devices.list_devices(), self._query, self._kind)
        link = self._open_link(device)
        try:
            link.stream.start()
        except StreamOpenError:
            _close_quietly(link.stream)
            raise
        self._link = link
        self._seen_callbacks = 0
        self._seen_at_s = self._clock()
        self._last_error = ""
        if self._ever_connected:
            self._reconnects += 1
        self._ever_connected = True

    def poll(self) -> None:
        """One supervision pass: detect a lost stream, or retry when disconnected."""
        now = self._clock()
        link = self._link
        if link is None:
            if now >= self._next_attempt_s:
                self._try_reconnect(now)
            return
        reason = self._lost_reason(link, now)
        if not reason:
            return
        self._losses += 1
        self._last_loss_reason = reason
        if self._before_close is not None:
            self._before_close(link, reason)
        self.close()
        self._next_attempt_s = now  # first retry immediately, then every retry_ms

    def close(self) -> None:
        """Close the current stream, if any, without counting it as a loss."""
        link = self._link
        if link is None:
            return
        self._link = None
        _close_quietly(link.stream)
        if self._after_close is not None:
            self._after_close(link)

    def _lost_reason(self, link: L, now: float) -> str:
        if not link.stream.active:
            return "stream inactive"
        if link.callbacks != self._seen_callbacks:
            self._seen_callbacks = link.callbacks
            self._seen_at_s = now
            return ""
        if (now - self._seen_at_s) * 1000 >= self._watchdog_ms:
            return f"no audio callback for {self._watchdog_ms} ms"
        return ""

    def _try_reconnect(self, now: float) -> None:
        # PortAudio caches its device list; rescan so a replugged device is visible.
        self._devices.reinitialize()
        try:
            self.connect()
        except (DeviceNotFoundError, AmbiguousDeviceError, StreamOpenError) as error:
            self._last_error = str(error)
            self._next_attempt_s = now + self._retry_s


class TickThread:
    """Calls ``tick()`` every ``poll_s`` seconds in a daemon thread until stopped.

    An exception from ``tick()`` ends the thread and is kept in ``fatal_error`` for the
    dashboard instead of being lost in a dead thread.
    """

    def __init__(self, tick: Callable[[], None], poll_s: float, name: str) -> None:
        """Create a stopped thread."""
        self._tick = tick
        self._poll_s = poll_s
        self._name = name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fatal_error: BaseException | None = None

    @property
    def running(self) -> bool:
        """True between ``start()`` and ``stop()``."""
        return self._thread is not None

    def start(self) -> None:
        """Start the thread; raise if it is already running."""
        if self._thread is not None:
            raise RuntimeError(f"{self._name} is already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Ask the thread to end and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            self._thread = None

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._tick()
                self._stop.wait(self._poll_s)
        except Exception as error:  # a bug or an unsupported device; shown on the dashboard
            self.fatal_error = error


def _close_quietly(stream: AudioStream) -> None:
    """Close a stream that may already be broken; there is nothing more to do on failure."""
    with contextlib.suppress(StreamOpenError):
        stream.stop()
    with contextlib.suppress(StreamOpenError):
        stream.close()
