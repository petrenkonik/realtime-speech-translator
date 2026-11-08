"""Audio output: a ring buffer behind an output stream, kept connected by a supervisor.

One ``OutputSession`` per output device. A language lane (producer thread) hands it TTS
audio at the device's sample rate; the audio callback copies it out. If the ring is empty
the callback plays silence. An empty ring *during speech* is an underrun and is counted;
idle silence between utterances is not.

Outputs share nothing: each has its own ring, stream and supervisor, so one device
failing (a pulled USB dongle) never touches another.

The producer must not block the real-time callback and vice versa, so the ring is
lock-free for one producer and one consumer, like the capture ring.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pipeline.devices import (
    AudioStream,
    DeviceInfo,
    DeviceListing,
    OutputStreamFactory,
)
from pipeline.supervisor import Supervisor, TickThread

Float32Array = NDArray[np.float32]

_WRITE_WAIT_S = 0.005


class OutputConfigError(ValueError):
    """An output setting is out of range."""


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output settings.

    Attributes:
        device_query: Substring of the output device name.
        block_ms: Length of one audio callback block.
        ring_seconds: Capacity of the ring between the lane and the callback.
        prebuffer_ms: Audio collected at the start of an utterance before playback
            begins, so that the first callbacks do not run dry (spec O1).
        watchdog_ms: No callback for this long means the stream is lost.
        retry_ms: Pause between attempts to find and reopen a lost device.
        poll_ms: How often the supervisor thread checks the stream.
    """

    device_query: str
    block_ms: int = 10
    ring_seconds: float = 10.0
    prebuffer_ms: int = 20
    watchdog_ms: int = 500
    retry_ms: int = 1000
    poll_ms: int = 20

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not self.device_query.strip():
            raise OutputConfigError("device_query must not be empty")
        if not 2 <= self.block_ms <= 100:
            raise OutputConfigError(f"block_ms must be 2..100, got {self.block_ms}")
        if not 0.5 <= self.ring_seconds <= 60:
            raise OutputConfigError(f"ring_seconds must be 0.5..60, got {self.ring_seconds}")
        if not 0 <= self.prebuffer_ms <= 100:
            raise OutputConfigError(f"prebuffer_ms must be 0..100, got {self.prebuffer_ms}")
        if self.watchdog_ms < 100:
            raise OutputConfigError(f"watchdog_ms must be at least 100, got {self.watchdog_ms}")
        if self.retry_ms < 1:
            raise OutputConfigError(f"retry_ms must be positive, got {self.retry_ms}")
        if not 1 <= self.poll_ms <= 200:
            raise OutputConfigError(f"poll_ms must be 1..200, got {self.poll_ms}")


class OutputRing:
    """Mono float32 ring, one producer thread and one consumer (the audio callback).

    The producer stores samples and then publishes the new write position; the consumer
    only moves the read position. ``write`` never overwrites unplayed audio: it accepts
    what fits and reports how much that was, so the producer decides whether to wait.
    """

    def __init__(self, capacity_frames: int) -> None:
        """Allocate the ring once."""
        if capacity_frames < 1:
            raise OutputConfigError("the ring needs at least one frame")
        self._capacity = capacity_frames
        self._data = np.zeros(capacity_frames, dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._discard_until = 0

    @property
    def capacity_frames(self) -> int:
        """Ring size in frames."""
        return self._capacity

    @property
    def available(self) -> int:
        """Frames waiting to be played."""
        return max(0, self._write_pos - max(self._read_pos, self._discard_until))

    @property
    def free(self) -> int:
        """Frames the producer can write right now."""
        return self._capacity - self.available

    def write(self, samples: Float32Array) -> int:
        """Copy as many samples as fit; return how many were accepted. Producer side."""
        accepted = min(len(samples), self.free)
        if accepted == 0:
            return 0
        position = self._write_pos
        start = position % self._capacity
        first = min(accepted, self._capacity - start)
        self._data[start : start + first] = samples[:first]
        self._data[: accepted - first] = samples[first:accepted]
        self._write_pos = position + accepted
        return accepted

    def discard_queued(self) -> None:
        """Drop everything written so far (late speech is worse than skipped speech).

        Only moves a marker; the consumer honors it at its next read, and audio written
        after this call is kept.
        """
        self._discard_until = self._write_pos

    def read_into(self, out: Any, frames: int) -> int:  # noqa: ANN401
        """Copy up to ``frames`` frames into ``out[:n]`` on every channel. Consumer side.

        Returns:
            The number of frames written; the caller fills the rest with silence.
        """
        if self._read_pos < self._discard_until:
            self._read_pos = self._discard_until
        count = min(frames, self._write_pos - self._read_pos)
        if count <= 0:
            return 0
        start = self._read_pos % self._capacity
        first = min(count, self._capacity - start)
        out[:first] = self._data[start : start + first, np.newaxis]
        if count > first:
            out[first:count] = self._data[: count - first, np.newaxis]
        self._read_pos += count
        return count


class _OutputLink:
    """Everything that belongs to one open output stream."""

    def __init__(
        self,
        device: DeviceInfo,
        config: OutputConfig,
        factory: OutputStreamFactory,
        streaming: bool,
    ) -> None:
        rate_hz = device.default_sample_rate_hz
        self.device = device
        self.ring = OutputRing(int(config.ring_seconds * rate_hz))
        self.prebuffer_frames = config.prebuffer_ms * rate_hz // 1000
        self.streaming = streaming
        self.primed = False
        self.callbacks = 0
        self.underruns = 0
        self.underrun_frames = 0
        self.driver_underflows = 0
        self.frames_played = 0
        self.max_callback_s = 0.0
        blocksize = max(1, round(rate_hz * config.block_ms / 1000))
        self.stream: AudioStream = factory(device, self.on_audio, blocksize)

    def on_audio(self, out: Any, driver_underflow: bool) -> None:  # noqa: ANN401
        """Audio callback: copy from the ring or play silence. Nothing else."""
        started = time.perf_counter()
        frames = len(out)
        ring = self.ring
        if driver_underflow:
            self.driver_underflows += 1
        if not self.primed:
            waiting = ring.available
            # Wait for the prebuffer while speech is still arriving; a short tail
            # after the end of an utterance plays as soon as it exists.
            if waiting >= self.prebuffer_frames or (waiting > 0 and not self.streaming):
                self.primed = True
        played = ring.read_into(out, frames) if self.primed else 0
        if played < frames:
            out[played:] = 0.0
            if self.primed and self.streaming:
                self.underruns += 1
                self.underrun_frames += frames - played
            if ring.available == 0:
                self.primed = False  # collect a fresh prebuffer for the next burst
        self.frames_played += played
        self.callbacks += 1
        elapsed = time.perf_counter() - started
        if elapsed > self.max_callback_s:
            self.max_callback_s = elapsed


@dataclass(frozen=True, slots=True)
class OutputStats:
    """Snapshot of one output's counters.

    Attributes:
        connected: A stream is open.
        callbacks: Audio callbacks since the first connection.
        underruns: Callbacks that ran dry in the middle of an utterance.
        underrun_frames: Frames of silence played because of those underruns.
        driver_underflows: Underflows reported by PortAudio itself.
        frames_played: Audio frames delivered to the device.
        dropped_frames: Frames discarded because the device was not connected or the
            ring stayed full.
        queued_ms: Audio waiting in the ring.
        max_callback_ms: Longest time spent inside the audio callback.
        reconnects: Successful reconnections after a loss.
        losses: Times the stream was declared lost.
        last_loss_reason: Why the last loss was declared, or ``""``.
        last_error: The last failed connection attempt, or ``""``.
    """

    connected: bool
    callbacks: int
    underruns: int
    underrun_frames: int
    driver_underflows: int
    frames_played: int
    dropped_frames: int
    queued_ms: float
    max_callback_ms: float
    reconnects: int
    losses: int
    last_loss_reason: str
    last_error: str


class OutputSession:
    """One output device: accepts audio from a lane and keeps the device connected."""

    def __init__(
        self,
        config: OutputConfig,
        devices: DeviceListing,
        stream_factory: OutputStreamFactory,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wire the session; nothing is opened until ``connect()`` or ``start()``.

        Args:
            config: Output settings.
            devices: Device list source (PortAudio or a fake).
            stream_factory: Opens an output stream for a device.
            clock: Monotonic seconds; only the watchdog and the retry timer use it.
        """
        self._config = config
        self._factory = stream_factory
        self._streaming = False
        self._generation = 0
        self._dropped_frames = 0
        self._closed_callbacks = 0
        self._closed_underruns = 0
        self._closed_underrun_frames = 0
        self._closed_driver_underflows = 0
        self._closed_frames_played = 0
        self._max_callback_s = 0.0
        self._supervisor: Supervisor[_OutputLink] = Supervisor(
            query=config.device_query,
            kind="output",
            devices=devices,
            open_link=self._open_link,
            watchdog_ms=config.watchdog_ms,
            retry_ms=config.retry_ms,
            clock=clock,
            after_close=self._after_close,
        )
        self._thread = TickThread(self.tick, config.poll_ms / 1000, f"output-{config.device_query}")

    # --- connection ----------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True while a stream is open."""
        return self._supervisor.link is not None

    @property
    def device(self) -> DeviceInfo | None:
        """The device currently open, if any."""
        link = self._supervisor.link
        return None if link is None else link.device

    @property
    def sample_rate_hz(self) -> int | None:
        """Rate the audio passed to ``write`` must have; ``None`` while disconnected."""
        device = self.device
        return None if device is None else device.default_sample_rate_hz

    @property
    def generation(self) -> int:
        """Number of connections so far; a lane rebuilds its resampler when it changes."""
        return self._generation

    @property
    def fatal_error(self) -> BaseException | None:
        """The exception that ended the supervisor thread, if any."""
        return self._thread.fatal_error

    def connect(self) -> None:
        """Find the device and open its stream; raise if that is not possible.

        Raises:
            DeviceNotFoundError: No output device matches ``device_query``.
            AmbiguousDeviceError: Several devices match.
            StreamOpenError: The device was found but could not be opened.
        """
        self._supervisor.connect()

    def tick(self) -> None:
        """One supervision pass: watchdog and reconnect."""
        self._supervisor.poll()

    def start(self) -> None:
        """Connect (raising on failure) and run the supervisor thread."""
        if self._thread.running:
            raise RuntimeError("the output session is already started")
        self.connect()
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the thread and close the stream."""
        self._thread.stop(timeout_s)
        self._supervisor.close()

    # --- audio in ------------------------------------------------------------------

    def begin_utterance(self) -> None:
        """Mark the start of speech: from now on an empty ring counts as an underrun."""
        self._streaming = True
        link = self._supervisor.link
        if link is not None:
            link.streaming = True

    def end_utterance(self) -> None:
        """Mark the end of speech: the remaining audio plays out, then silence."""
        self._streaming = False
        link = self._supervisor.link
        if link is not None:
            link.streaming = False

    def write(self, samples: Float32Array) -> int:
        """Queue mono audio at ``sample_rate_hz`` without blocking.

        Returns:
            How many frames were accepted. While the device is disconnected everything
            is discarded (and counted) and the full length is returned, so the lane
            does not wait for a device that may stay away.
        """
        link = self._supervisor.link
        if link is None:
            self._dropped_frames += len(samples)
            return len(samples)
        return link.ring.write(samples)

    def write_all(self, samples: Float32Array, timeout_s: float = 5.0) -> bool:
        """Queue all of ``samples``, waiting for room in the ring.

        Returns:
            True if everything was queued; False if the ring stayed full for
            ``timeout_s`` (the rest is discarded and counted).
        """
        deadline = time.monotonic() + timeout_s
        remaining = samples
        while len(remaining) > 0:
            accepted = self.write(remaining)
            remaining = remaining[accepted:]
            if len(remaining) == 0:
                return True
            if time.monotonic() >= deadline:
                self._dropped_frames += len(remaining)
                return False
            time.sleep(_WRITE_WAIT_S)
        return True

    def clear(self) -> None:
        """Drop audio that was queued but not yet played."""
        link = self._supervisor.link
        if link is not None:
            link.ring.discard_queued()

    # --- statistics ----------------------------------------------------------------

    def stats(self) -> OutputStats:
        """Return a snapshot of the counters."""
        link = self._supervisor.link
        health = self._supervisor.stats()
        rate_hz = 0 if link is None else link.device.default_sample_rate_hz
        queued_ms = 0.0 if link is None else link.ring.available * 1000.0 / rate_hz
        return OutputStats(
            connected=health.connected,
            callbacks=self._closed_callbacks + (0 if link is None else link.callbacks),
            underruns=self._closed_underruns + (0 if link is None else link.underruns),
            underrun_frames=self._closed_underrun_frames
            + (0 if link is None else link.underrun_frames),
            driver_underflows=self._closed_driver_underflows
            + (0 if link is None else link.driver_underflows),
            frames_played=self._closed_frames_played + (0 if link is None else link.frames_played),
            dropped_frames=self._dropped_frames,
            queued_ms=queued_ms,
            max_callback_ms=max(self._max_callback_s, 0.0 if link is None else link.max_callback_s)
            * 1000.0,
            reconnects=health.reconnects,
            losses=health.losses,
            last_loss_reason=health.last_loss_reason,
            last_error=health.last_error,
        )

    # --- internals -----------------------------------------------------------------

    def _open_link(self, device: DeviceInfo) -> _OutputLink:
        link = _OutputLink(device, self._config, self._factory, self._streaming)
        self._generation += 1
        return link

    def _after_close(self, link: _OutputLink) -> None:
        self._dropped_frames += link.ring.available  # queued but never played
        self._closed_callbacks += link.callbacks
        self._closed_underruns += link.underruns
        self._closed_underrun_frames += link.underrun_frames
        self._closed_driver_underflows += link.driver_underflows
        self._closed_frames_played += link.frames_played
        self._max_callback_s = max(self._max_callback_s, link.max_callback_s)
