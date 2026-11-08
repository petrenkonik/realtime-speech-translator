"""Test doubles for audio hardware: devices, input streams, clock and a model-free VAD."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pipeline.devices import BlockCallback, DeviceInfo, StreamOpenError

Float32Array = NDArray[np.float32]


def make_device(
    name: str,
    index: int = 0,
    rate_hz: int = 48_000,
    inputs: int = 2,
    outputs: int = 0,
    host_api: str = "Windows WASAPI",
) -> DeviceInfo:
    """Build a ``DeviceInfo`` with sensible defaults for an input device."""
    return DeviceInfo(
        name=name,
        index=index,
        host_api=host_api,
        max_input_channels=inputs,
        max_output_channels=outputs,
        default_sample_rate_hz=rate_hz,
    )


class FakeClock:
    """Monotonic clock that only moves when the test says so."""

    def __init__(self, start_s: float = 100.0) -> None:
        self.now_s = start_s

    def __call__(self) -> float:
        return self.now_s

    def advance(self, seconds: float) -> None:
        self.now_s += seconds


class FakeDeviceList:
    """Scriptable ``DeviceListing``: tests reorder, remove and add devices."""

    def __init__(self, devices: Sequence[DeviceInfo]) -> None:
        self.devices = list(devices)
        self.reinitialize_calls = 0
        self.list_calls = 0

    def list_devices(self) -> Sequence[DeviceInfo]:
        self.list_calls += 1
        return list(self.devices)

    def reinitialize(self) -> None:
        self.reinitialize_calls += 1

    def renumber(self, order: Sequence[str]) -> None:
        """Reorder devices by name and give them new indexes, like Windows does."""
        by_name = {device.name: device for device in self.devices}
        self.devices = [
            DeviceInfo(
                name=by_name[name].name,
                index=index + 10,
                host_api=by_name[name].host_api,
                max_input_channels=by_name[name].max_input_channels,
                max_output_channels=by_name[name].max_output_channels,
                default_sample_rate_hz=by_name[name].default_sample_rate_hz,
            )
            for index, name in enumerate(order)
        ]

    def remove(self, name: str) -> None:
        self.devices = [device for device in self.devices if device.name != name]


class FakeInputStream:
    """An input stream that delivers whatever the test feeds it, as fast as the test likes."""

    def __init__(self, callback: BlockCallback, blocksize_frames: int, channels: int) -> None:
        self.callback = callback
        self.blocksize_frames = blocksize_frames
        self.channels = channels
        self.active = False
        self.closed = False
        self.fail_on_start = False

    def start(self) -> None:
        if self.fail_on_start:
            raise StreamOpenError("scripted start failure")
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.closed = True

    def fail(self) -> None:
        """Simulate the driver killing the stream (device unplugged, PortAudioError)."""
        self.active = False

    def feed(
        self,
        audio: Float32Array,
        rng: np.random.Generator | None = None,
        overflow_blocks: Sequence[int] = (),
    ) -> int:
        """Deliver ``audio`` (shape ``(frames, channels)``) through the callback.

        Args:
            audio: Samples to deliver.
            rng: When given, block lengths are random (1 to 2x the nominal size) like
                real drivers; otherwise they are the nominal size.
            overflow_blocks: Indexes of blocks that carry the overflow flag.

        Returns:
            The number of blocks delivered.
        """
        position = 0
        index = 0
        while position < len(audio):
            size = self.blocksize_frames
            if rng is not None:
                size = int(rng.integers(1, 2 * self.blocksize_frames + 1))
            block = audio[position : position + size]
            self.callback(block, index in overflow_blocks)
            position += len(block)
            index += 1
        return index


class FakeOutputStream(FakeInputStream):
    """An output stream the test pulls audio from, like a sound card would."""

    SENTINEL = 7.0  # proves the callback overwrites every sample it is given

    def __init__(self, callback: BlockCallback, blocksize_frames: int, channels: int) -> None:
        super().__init__(callback, blocksize_frames, channels)
        self.played: list[Float32Array] = []

    def pull(self, blocks: int = 1, underflow_blocks: Sequence[int] = ()) -> Float32Array:
        """Run the callback ``blocks`` times; return (and record) what it produced."""
        produced = []
        for index in range(blocks):
            out = np.full((self.blocksize_frames, self.channels), self.SENTINEL, dtype=np.float32)
            self.callback(out, index in underflow_blocks)
            produced.append(out)
        audio = np.concatenate(produced) if produced else np.zeros((0, self.channels), np.float32)
        self.played.append(audio)
        return audio


class FakeStreamFactory:
    """``StreamFactory`` (or output factory) that records every stream it opened."""

    def __init__(self, stream_cls: type[FakeInputStream] = FakeInputStream) -> None:
        self.stream_cls = stream_cls
        self.streams: list[Any] = []
        self.opened_devices: list[DeviceInfo] = []
        self.fail_opens = 0  # the next N opens raise StreamOpenError
        self.fail_starts = 0  # the next N streams fail in start()

    def __call__(self, device: DeviceInfo, callback: BlockCallback, blocksize_frames: int) -> Any:
        if self.fail_opens > 0:
            self.fail_opens -= 1
            raise StreamOpenError("scripted open failure")
        channels = device.max_input_channels or device.max_output_channels
        stream = self.stream_cls(
            callback,
            blocksize_frames,
            min(channels, 2) if self.stream_cls is FakeOutputStream else channels,
        )
        if self.fail_starts > 0:
            self.fail_starts -= 1
            stream.fail_on_start = True
        self.streams.append(stream)
        self.opened_devices.append(device)
        return stream

    @property
    def current(self) -> Any:
        return self.streams[-1]


class EnergyVad:
    """Model-free stand-in for ``SileroVad``: loud frame = speech, quiet frame = silence."""

    def __init__(self, threshold_rms: float = 0.01) -> None:
        self.threshold_rms = threshold_rms
        self.resets = 0

    def __call__(self, frame: Float32Array) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        return 0.95 if rms > self.threshold_rms else 0.02

    def reset(self) -> None:
        self.resets += 1


def speech(duration_ms: int, rate_hz: int, channels: int = 2, level: float = 0.3) -> Float32Array:
    """A loud tone burst standing in for speech, shape ``(frames, channels)``."""
    frames = duration_ms * rate_hz // 1000
    t = np.arange(frames) / rate_hz
    tone = (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.repeat(tone[:, np.newaxis], channels, axis=1)


def silence(duration_ms: int, rate_hz: int, channels: int = 2) -> Float32Array:
    """Digital silence, shape ``(frames, channels)``."""
    return np.zeros((duration_ms * rate_hz // 1000, channels), dtype=np.float32)


def stream_of(*parts: Float32Array) -> Float32Array:
    """Concatenate parts into one stream."""
    return np.concatenate(parts).astype(np.float32)
