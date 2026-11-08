"""Microphone capture: ring buffer, processing chain, watchdog and reconnect.

Data flow for one connection::

    audio callback --> RingBuffer --> ChannelMixer --> StreamingResampler --> VadStage --> ChunkBus
    (copy only)        (2 s)          (this thread: everything below the ring)

The audio callback only copies the block into a preallocated ring and bumps counters.
Everything heavier (mixing, resampling, the VAD model, publishing) runs in the
processing thread, so a slow stage can overflow the ring (counted) but can never make
the audio driver drop input.

Time is the stream sample counter, not the wall clock, so results do not depend on
machine speed. The only wall-clock use is the watchdog, and its clock is injected.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pipeline.bus import ChunkBus
from pipeline.devices import AudioStream, DeviceInfo, DeviceListing, StreamFactory
from pipeline.pcm import ChannelMixer, DownmixMode
from pipeline.resample import OUT_RATE_HZ, StreamingResampler
from pipeline.supervisor import Supervisor, TickThread
from pipeline.vad import Chunk

Float32Array = NDArray[np.float32]


class CaptureConfigError(ValueError):
    """A capture setting is out of range."""


class ChunkStage(Protocol):
    """What the session needs from the VAD stage (``VadStage`` satisfies it)."""

    @property
    def position_samples(self) -> int:
        """Stream position in 16 kHz samples, on the clock of the chunk coordinates."""
        ...

    def push(self, samples: Float32Array) -> list[Chunk]:
        """Feed 16 kHz samples; return the chunks they completed."""
        ...

    def flush(self) -> Chunk | None:
        """Finalize the open chunk and reset (end of stream or interruption)."""
        ...


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Capture settings (all times in milliseconds).

    Attributes:
        device_query: Substring of the input device name.
        block_ms: Length of one audio callback block.
        ring_seconds: Capacity of the ring between the callback and the processing thread.
        watchdog_ms: No callback for this long means the stream is lost.
        retry_ms: Pause between attempts to find and reopen a lost device.
        poll_ms: How often the processing thread looks at the ring.
        downmix: ``"auto"`` or ``"mean"``, see ``ChannelMixer``.
        silence_ms: The segmenter's silence threshold; used only to report how far
            beyond it a chunk was finalized.
    """

    device_query: str
    block_ms: int = 10
    ring_seconds: float = 2.0
    watchdog_ms: int = 500
    retry_ms: int = 1000
    poll_ms: int = 5
    downmix: DownmixMode = "auto"
    silence_ms: int = 450

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not self.device_query.strip():
            raise CaptureConfigError("device_query must not be empty")
        if not 5 <= self.block_ms <= 100:
            raise CaptureConfigError(f"block_ms must be 5..100, got {self.block_ms}")
        if not 0.5 <= self.ring_seconds <= 30:
            raise CaptureConfigError(f"ring_seconds must be 0.5..30, got {self.ring_seconds}")
        if self.watchdog_ms < 100:
            raise CaptureConfigError(f"watchdog_ms must be at least 100, got {self.watchdog_ms}")
        if self.retry_ms < 1:
            raise CaptureConfigError(f"retry_ms must be positive, got {self.retry_ms}")
        if not 1 <= self.poll_ms <= 100:
            raise CaptureConfigError(f"poll_ms must be 1..100, got {self.poll_ms}")
        if self.downmix not in ("auto", "mean"):
            raise CaptureConfigError(f"downmix must be 'auto' or 'mean', got {self.downmix!r}")
        if self.silence_ms < 1:
            raise CaptureConfigError(f"silence_ms must be positive, got {self.silence_ms}")


class RingBuffer:
    """Single-producer, single-consumer ring of audio frames without locks.

    The producer (audio callback) only calls ``write``; the consumer (processing
    thread) only calls ``read``. The producer stores the data first and publishes the
    new write position afterwards, so the consumer never sees positions whose data is
    not there yet. If the consumer falls more than a full ring behind, the oldest
    frames are gone: ``read`` skips to the oldest surviving frame and counts the loss.
    """

    def __init__(self, capacity_frames: int, channels: int) -> None:
        """Allocate the ring once; nothing is allocated afterwards on the write path."""
        if capacity_frames < 1 or channels < 1:
            raise CaptureConfigError("ring needs at least one frame and one channel")
        self._capacity = capacity_frames
        self._channels = channels
        self._data = np.zeros((capacity_frames, channels), dtype=np.float32)
        self._write_pos = 0  # total frames ever written; only the producer changes it
        self._read_pos = 0  # total frames ever read or skipped; only the consumer changes it
        self._lost_frames = 0

    @property
    def capacity_frames(self) -> int:
        """Ring size in frames."""
        return self._capacity

    @property
    def frames_written(self) -> int:
        """Total frames written so far (the capture stream clock)."""
        return self._write_pos

    @property
    def lost_frames(self) -> int:
        """Frames overwritten before the consumer read them."""
        return self._lost_frames

    def write(self, block: Float32Array) -> None:
        """Copy one block into the ring. Producer side; never blocks or allocates."""
        frames = len(block)
        position = self._write_pos
        if frames >= self._capacity:
            # A block larger than the whole ring: only its tail can be kept.
            tail = block[frames - self._capacity :]
            start = (position + frames) % self._capacity  # where the oldest kept frame goes
            self._data[start:] = tail[: self._capacity - start]
            self._data[:start] = tail[self._capacity - start :]
            self._write_pos = position + frames
            return
        start = position % self._capacity
        first = min(frames, self._capacity - start)
        self._data[start : start + first] = block[:first]
        self._data[: frames - first] = block[first:]
        self._write_pos = position + frames

    def read(self) -> Float32Array:
        """Return all frames written since the last call, oldest first. Consumer side."""
        end = self._write_pos
        start = self._read_pos
        if end - start > self._capacity:
            self._lost_frames += end - self._capacity - start
            start = end - self._capacity
        indices = np.arange(start, end) % self._capacity
        copy: Float32Array = self._data[indices]
        # The producer may have lapped us while we copied; the oldest frames of the
        # copy could then be partly overwritten. Drop them instead of passing on noise.
        end_after = self._write_pos
        if end_after - start > self._capacity:
            clobbered = end_after - self._capacity - start
            self._lost_frames += clobbered
            copy = copy[clobbered:]
        self._read_pos = end
        return copy


@dataclass(frozen=True, slots=True)
class CaptureStats:
    """Snapshot of the capture counters, for the dashboard and for tests.

    Attributes:
        connected: A stream is open and being read.
        callbacks: Audio callbacks since the first connection.
        input_overflow: Callbacks that reported a PortAudio input overflow.
        ring_lost_frames: Frames the processing thread was too slow to read.
        max_callback_ms: Longest time spent inside the audio callback.
        reconnects: Successful reconnections after a loss (not counting the first).
        losses: Times the stream was declared lost.
        last_loss_reason: Why the last loss was declared, or ``""``.
        last_error: Last failed connection attempt, or ``""``.
        chunks: Chunks published.
        finalization_latency_ms: How far past the silence threshold the most recent
            chunks were published, in stream time (oldest first, at most 1000).
    """

    connected: bool
    callbacks: int
    input_overflow: int
    ring_lost_frames: int
    max_callback_ms: float
    reconnects: int
    losses: int
    last_loss_reason: str
    last_error: str
    chunks: int
    finalization_latency_ms: tuple[float, ...]


class _Link:
    """Everything that belongs to one open stream; replaced as a whole on reconnect."""

    def __init__(
        self,
        device: DeviceInfo,
        config: CaptureConfig,
        stream_factory: StreamFactory,
    ) -> None:
        rate_hz = device.default_sample_rate_hz
        channels = device.max_input_channels
        self.device = device
        self.ring = RingBuffer(int(config.ring_seconds * rate_hz), channels)
        self.mixer = ChannelMixer(rate_hz, config.downmix)
        self.resampler = StreamingResampler(rate_hz, OUT_RATE_HZ)
        self.callbacks = 0
        self.input_overflow = 0
        self.max_callback_s = 0.0
        blocksize = max(1, round(rate_hz * config.block_ms / 1000))
        self.stream: AudioStream = stream_factory(device, self.on_block, blocksize)

    def on_block(self, block: Any, overflow: bool) -> None:  # noqa: ANN401
        """Audio callback: copy, count, time. Nothing else."""
        started = time.perf_counter()
        if overflow:
            self.input_overflow += 1
        self.ring.write(block)
        self.callbacks += 1
        elapsed = time.perf_counter() - started
        if elapsed > self.max_callback_s:
            self.max_callback_s = elapsed


class CaptureSession:
    """Keeps a microphone connected and turns its audio into chunks on a ``ChunkBus``.

    ``tick()`` is one pass: read the ring, run the chain, publish chunks, check the
    watchdog and reconnect if needed (the connection part is a ``Supervisor``).
    ``start()`` runs ``tick()`` in a thread; tests call it directly with a fake clock,
    which keeps them deterministic.
    """

    def __init__(
        self,
        config: CaptureConfig,
        devices: DeviceListing,
        stream_factory: StreamFactory,
        stage: ChunkStage,
        bus: ChunkBus,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wire the session; nothing is opened until ``connect()`` or ``start()``.

        Args:
            config: Capture settings.
            devices: Device list source (PortAudio or a fake).
            stream_factory: Opens an input stream for a device.
            stage: The VAD stage that turns 16 kHz audio into chunks.
            bus: Where finished chunks are published.
            clock: Monotonic seconds; only the watchdog and the retry timer use it.
        """
        self._config = config
        self._factory = stream_factory
        self._stage = stage
        self._bus = bus
        self._supervisor: Supervisor[_Link] = Supervisor(
            query=config.device_query,
            kind="input",
            devices=devices,
            open_link=self._open_link,
            watchdog_ms=config.watchdog_ms,
            retry_ms=config.retry_ms,
            clock=clock,
            before_close=self._before_close,
            after_close=self._after_close,
        )
        # Counters of closed links are added here so totals survive a reconnect.
        self._closed_callbacks = 0
        self._closed_overflow = 0
        self._closed_lost_frames = 0
        self._max_callback_s = 0.0
        self._chunks = 0
        self._latencies: deque[float] = deque(maxlen=1000)
        self._thread = TickThread(self.tick, config.poll_ms / 1000, "capture")

    # --- public API ----------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True while a stream is open."""
        return self._supervisor.link is not None

    @property
    def fatal_error(self) -> BaseException | None:
        """The exception that ended the processing thread, if any."""
        return self._thread.fatal_error

    @property
    def device(self) -> DeviceInfo | None:
        """The device currently open, if any."""
        link = self._supervisor.link
        return None if link is None else link.device

    def stats(self) -> CaptureStats:
        """Return a snapshot of the counters."""
        link = self._supervisor.link
        health = self._supervisor.stats()
        live_callbacks = 0 if link is None else link.callbacks
        live_overflow = 0 if link is None else link.input_overflow
        live_lost = 0 if link is None else link.ring.lost_frames
        live_max_cb = 0.0 if link is None else link.max_callback_s
        return CaptureStats(
            connected=health.connected,
            callbacks=self._closed_callbacks + live_callbacks,
            input_overflow=self._closed_overflow + live_overflow,
            ring_lost_frames=self._closed_lost_frames + live_lost,
            max_callback_ms=max(self._max_callback_s, live_max_cb) * 1000.0,
            reconnects=health.reconnects,
            losses=health.losses,
            last_loss_reason=health.last_loss_reason,
            last_error=health.last_error,
            chunks=self._chunks,
            finalization_latency_ms=tuple(self._latencies),
        )

    def connect(self) -> None:
        """Find the device and open its stream; raise if that is not possible.

        Raises:
            DeviceNotFoundError: No device matches ``device_query``.
            AmbiguousDeviceError: Several devices match.
            StreamOpenError: The device was found but could not be opened.
        """
        self._supervisor.connect()

    def tick(self) -> None:
        """One pass: process what arrived, then check the connection."""
        if self._supervisor.link is not None:
            self._drain()
        self._supervisor.poll()

    def start(self) -> None:
        """Connect (raising on failure) and run the processing thread.

        Raises:
            DeviceNotFoundError: The configured device does not exist at startup.
            AmbiguousDeviceError: The name matches several devices.
            StreamOpenError: The device could not be opened.
        """
        if self._thread.running:
            raise RuntimeError("the capture session is already started")
        self.connect()
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the thread, close the stream and publish the pending chunk."""
        self._thread.stop(timeout_s)
        if self._supervisor.link is not None:
            self._drain()
            self._supervisor.close()
        self._publish(self._stage.flush())

    # --- internals -----------------------------------------------------------------

    def _open_link(self, device: DeviceInfo) -> _Link:
        return _Link(device, self._config, self._factory)

    def _before_close(self, link: _Link, reason: str) -> None:
        self._drain()  # whatever arrived before the failure still counts
        self._publish(self._stage.flush())  # finalize the pending chunk

    def _after_close(self, link: _Link) -> None:
        self._closed_callbacks += link.callbacks
        self._closed_overflow += link.input_overflow
        self._closed_lost_frames += link.ring.lost_frames
        self._max_callback_s = max(self._max_callback_s, link.max_callback_s)

    def _drain(self) -> None:
        link = self._supervisor.link
        if link is None:
            return
        block = link.ring.read()
        if len(block) == 0:
            return
        mono = link.mixer.process(block)
        samples = link.resampler.process(mono)
        chunks = self._stage.push(samples)
        position = self._stage.position_samples
        for chunk in chunks:
            self._publish(chunk, position)

    def _publish(self, chunk: Chunk | None, position_samples: int | None = None) -> None:
        if chunk is None:
            return
        if chunk.reason == "silence" and position_samples is not None:
            beyond_end_ms = (position_samples - chunk.speech_end_sample) * 1000 / OUT_RATE_HZ
            self._latencies.append(beyond_end_ms - self._config.silence_ms)
        self._chunks += 1
        self._bus.publish(chunk)
