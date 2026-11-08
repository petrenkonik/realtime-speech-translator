"""Capture: ring buffer, whole chain on fakes, watchdog and reconnect (spec F1-F5)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pytest
from fakes import (
    EnergyVad,
    FakeClock,
    FakeDeviceList,
    FakeStreamFactory,
    Float32Array,
    make_device,
    silence,
    speech,
    stream_of,
)

from pipeline.bus import ChunkBus, Subscription
from pipeline.capture import (
    CaptureConfig,
    CaptureConfigError,
    CaptureSession,
    RingBuffer,
)
from pipeline.devices import DeviceNotFoundError, StreamOpenError
from pipeline.resample import UnsupportedSampleRateError
from pipeline.vad import Chunk, Segmenter, SegmenterConfig, VadStage

pytestmark = pytest.mark.unit

MIC = "Microphone (V380 USB)"


@dataclass
class Rig:
    session: CaptureSession
    devices: FakeDeviceList
    factory: FakeStreamFactory
    clock: FakeClock
    sub: Subscription[Chunk]
    rate_hz: int
    channels: int


def build(
    rate_hz: int = 48_000,
    channels: int = 2,
    silence_ms: int = 450,
    ring_seconds: float = 2.0,
    connect: bool = True,
) -> Rig:
    clock = FakeClock()
    devices = FakeDeviceList(
        [
            make_device(MIC, index=1, rate_hz=rate_hz, inputs=channels),
            make_device("Speakers (Realtek Audio)", index=2, inputs=0, outputs=2),
        ]
    )
    factory = FakeStreamFactory()
    bus = ChunkBus()
    sub = bus.subscribe("test", maxsize=100)
    stage = VadStage(
        EnergyVad(),  # type: ignore[arg-type]
        Segmenter(SegmenterConfig(silence_ms=silence_ms)),
    )
    config = CaptureConfig("V380", silence_ms=silence_ms, ring_seconds=ring_seconds)
    session = CaptureSession(config, devices, factory, stage, bus, clock)
    if connect:
        session.connect()
    return Rig(session, devices, factory, clock, sub, rate_hz, channels)


def play(rig: Rig, audio: Float32Array, step_ms: int = 10) -> None:
    """Deliver ``audio`` in real-time-sized steps, running the supervisor after each."""
    step = rig.rate_hz * step_ms // 1000
    for start in range(0, len(audio), step):
        rig.factory.current.feed(audio[start : start + step])
        rig.clock.advance(step_ms / 1000)
        rig.session.tick()


def collect(sub: Subscription[Chunk]) -> list[Chunk]:
    chunks = []
    while (chunk := sub.get(timeout_s=0)) is not None:
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------- ring buffer


def test_ring_returns_frames_in_order_across_the_wrap_point() -> None:
    ring = RingBuffer(capacity_frames=10, channels=1)
    out: list[float] = []
    value = 0.0
    for size in (4, 4, 4, 3, 5, 2):
        block = (value + np.arange(size, dtype=np.float32))[:, np.newaxis]
        value += size
        ring.write(block)
        out += ring.read()[:, 0].tolist()
    assert out == list(range(int(value)))
    assert ring.lost_frames == 0
    assert ring.frames_written == int(value)


def test_ring_read_when_empty_returns_no_frames() -> None:
    ring = RingBuffer(8, channels=2)
    assert ring.read().shape == (0, 2)


def test_ring_that_is_lapped_skips_to_the_oldest_surviving_frames() -> None:
    ring = RingBuffer(capacity_frames=10, channels=1)
    for start in range(0, 25, 5):  # 25 frames into a ring of 10 without reading
        ring.write((start + np.arange(5, dtype=np.float32))[:, np.newaxis])
    frames = ring.read()[:, 0].tolist()
    assert frames == list(range(15, 25))
    assert ring.lost_frames == 15


def test_ring_block_larger_than_the_ring_keeps_its_tail() -> None:
    ring = RingBuffer(capacity_frames=4, channels=1)
    ring.write(np.arange(10, dtype=np.float32)[:, np.newaxis])
    assert ring.read()[:, 0].tolist() == [6.0, 7.0, 8.0, 9.0]
    assert ring.lost_frames == 6


def test_ring_rejects_empty_dimensions() -> None:
    with pytest.raises(CaptureConfigError):
        RingBuffer(0, 1)


# ---------------------------------------------------------------- config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device_query": " "},
        {"block_ms": 4},
        {"block_ms": 101},
        {"ring_seconds": 0.1},
        {"ring_seconds": 31},
        {"watchdog_ms": 99},
        {"retry_ms": 0},
        {"poll_ms": 0},
        {"downmix": "median"},
        {"silence_ms": 0},
    ],
)
def test_config_rejects_out_of_range_values(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {"device_query": "mic"}
    with pytest.raises(CaptureConfigError):
        CaptureConfig(**{**base, **kwargs})  # type: ignore[arg-type]


def test_config_defaults_match_the_spec() -> None:
    config = CaptureConfig("mic")
    assert (config.block_ms, config.ring_seconds, config.watchdog_ms) == (10, 2.0, 500)
    assert (config.retry_ms, config.downmix) == (1000, "auto")


# ---------------------------------------------------------------- F1


@pytest.mark.parametrize(
    ("rate_hz", "channels"), [(48_000, 2), (44_100, 1), (16_000, 1), (32_000, 2)]
)
def test_f1_whole_chain_on_fakes_gives_the_expected_chunks(rate_hz: int, channels: int) -> None:
    rig = build(rate_hz, channels)
    audio = stream_of(
        silence(500, rate_hz, channels),
        speech(1000, rate_hz, channels),
        silence(1200, rate_hz, channels),
        speech(800, rate_hz, channels),
        silence(1200, rate_hz, channels),
    )
    rng = np.random.default_rng(3)
    stream = rig.factory.current
    for start in range(0, len(audio), rate_hz // 10):  # 100 ms at a time, random block sizes
        stream.feed(audio[start : start + rate_hz // 10], rng=rng)
        rig.clock.advance(0.1)
        rig.session.tick()

    chunks = collect(rig.sub)
    assert [chunk.id for chunk in chunks] == [1, 2]
    assert all(chunk.reason == "silence" for chunk in chunks)
    first, second = chunks
    # Truth in 16 kHz samples; the VAD works on 32 ms frames, the segmenter keeps 100 ms
    # of pre-roll and trims the tail to 160 ms, so allow two frames of slack.
    slack = 2 * 512
    assert abs(first.start_sample - (8_000 - 1_600)) <= slack
    assert abs(first.speech_end_sample - 24_000) <= slack
    assert abs(second.speech_end_sample - (8_000 + 16_000 + 19_200 + 12_800)) <= slack
    assert first.samples.dtype == np.float32
    assert not first.samples.flags.writeable
    assert rig.session.stats().chunks == 2


def test_the_chain_downmixes_a_mono_microphone_on_a_stereo_device() -> None:
    # Right channel dead: a plain mean would halve the level, "auto" keeps it.
    rig = build(48_000, 2)
    loud = speech(600, 48_000, 2, level=0.05)
    loud[:, 1] = 0.0
    play(rig, stream_of(silence(300, 48_000), loud, silence(1000, 48_000)))
    (chunk,) = collect(rig.sub)
    peak = float(np.max(np.abs(chunk.samples)))
    assert peak > 0.04


# ---------------------------------------------------------------- F2


def test_f2_overflow_flags_are_counted() -> None:
    rig = build()
    rig.factory.current.feed(silence(100, 48_000), overflow_blocks={1, 3, 5})
    rig.session.tick()
    assert rig.session.stats().input_overflow == 3


def test_a_slow_processing_thread_loses_frames_in_the_ring_not_in_the_driver() -> None:
    rig = build(ring_seconds=0.5)
    rig.factory.current.feed(silence(1500, 48_000))  # 1.5 s arrives before any tick
    rig.session.tick()
    stats = rig.session.stats()
    assert stats.ring_lost_frames == 72_000 - 24_000  # only the last 0.5 s fits
    assert stats.input_overflow == 0


def test_the_callback_is_fast_and_counted() -> None:
    rig = build()
    play(rig, silence(500, 48_000))
    stats = rig.session.stats()
    assert stats.callbacks == 50
    assert 0.0 < stats.max_callback_ms < 5.0


# ---------------------------------------------------------------- F3


def test_f3_stream_failure_finalizes_the_pending_chunk_and_reconnects() -> None:
    rig = build()
    play(rig, stream_of(silence(300, 48_000), speech(1000, 48_000)))
    assert collect(rig.sub) == []  # still mid-utterance

    rig.factory.current.fail()
    rig.clock.advance(0.01)
    rig.session.tick()

    (pending,) = collect(rig.sub)
    assert pending.reason == "flush"
    assert not rig.session.connected
    assert rig.session.stats().losses == 1
    assert rig.session.stats().last_loss_reason == "stream inactive"
    assert rig.factory.streams[0].closed

    rig.clock.advance(0.01)
    rig.session.tick()  # immediate first retry
    assert rig.session.connected
    assert len(rig.factory.streams) == 2
    assert rig.devices.reinitialize_calls == 1

    play(rig, stream_of(silence(300, 48_000), speech(800, 48_000), silence(1000, 48_000)))
    (resumed,) = collect(rig.sub)
    assert resumed.reason == "silence"
    assert resumed.id == pending.id + 1
    stats = rig.session.stats()
    assert stats.reconnects == 1
    assert stats.chunks == 2


def test_audio_from_before_the_gap_is_not_glued_to_speech_after_it() -> None:
    rig = build()
    play(rig, stream_of(silence(500, 48_000), speech(600, 48_000)))
    rig.factory.current.fail()
    rig.session.tick()
    (pending,) = collect(rig.sub)
    rig.session.tick()
    play(rig, stream_of(speech(600, 48_000), silence(1000, 48_000)))
    (chunk,) = collect(rig.sub)
    # Pre-roll was cleared by the flush, so the new chunk starts at the new stream's
    # first sample instead of reaching back into audio from before the gap.
    assert chunk.start_sample >= pending.end_sample


def test_watchdog_declares_the_stream_lost_after_500_ms_without_callbacks() -> None:
    rig = build()
    play(rig, silence(100, 48_000))
    rig.clock.advance(0.4)
    rig.session.tick()
    assert rig.session.connected  # 400 ms of silence from the driver is still tolerated
    rig.clock.advance(0.2)
    rig.session.tick()
    assert not rig.session.connected
    assert "no audio callback" in rig.session.stats().last_loss_reason


def test_a_steady_trickle_of_callbacks_keeps_the_watchdog_quiet() -> None:
    rig = build()
    for _ in range(20):
        rig.factory.current.feed(silence(10, 48_000))
        rig.clock.advance(0.3)
        rig.session.tick()
    assert rig.session.connected
    assert rig.session.stats().losses == 0


def test_a_stream_that_never_delivers_is_lost_after_the_watchdog_time() -> None:
    rig = build()
    rig.clock.advance(0.6)
    rig.session.tick()
    assert not rig.session.connected


def test_retries_once_per_second_until_the_device_returns() -> None:
    rig = build()
    play(rig, silence(100, 48_000))
    rig.devices.remove(MIC)
    rig.factory.current.fail()
    rig.session.tick()  # loss
    rig.session.tick()  # first retry: device missing
    assert not rig.session.connected
    assert "no input device" in rig.session.stats().last_error
    calls = rig.devices.list_calls

    rig.clock.advance(0.5)
    rig.session.tick()
    assert rig.devices.list_calls == calls  # not yet: retries are 1 s apart

    rig.clock.advance(0.6)
    rig.session.tick()
    assert rig.devices.list_calls == calls + 1
    assert not rig.session.connected

    rig.devices.devices.append(make_device(MIC, index=5))
    rig.clock.advance(1.1)
    rig.session.tick()
    assert rig.session.connected
    assert rig.session.stats().reconnects == 1
    assert rig.session.stats().last_error == ""


def test_f4_reconnect_finds_the_device_after_windows_renumbered_it() -> None:
    rig = build()
    first_index = rig.session.device.index  # type: ignore[union-attr]
    rig.devices.renumber(["Speakers (Realtek Audio)", MIC])
    rig.factory.current.fail()
    rig.session.tick()
    rig.session.tick()
    assert rig.session.connected
    assert rig.factory.opened_devices[1].name == MIC
    assert rig.factory.opened_devices[1].index != first_index


def test_failures_to_open_or_start_the_stream_are_retried() -> None:
    rig = build()
    rig.factory.current.fail()
    rig.session.tick()  # loss
    rig.factory.fail_opens = 1
    rig.session.tick()
    assert not rig.session.connected
    assert "scripted open failure" in rig.session.stats().last_error

    rig.factory.fail_starts = 1
    rig.clock.advance(1.1)
    rig.session.tick()
    assert not rig.session.connected
    assert rig.factory.current.closed  # a half-open stream is released

    rig.clock.advance(1.1)
    rig.session.tick()
    assert rig.session.connected


def test_counters_survive_a_reconnect() -> None:
    rig = build()
    play(rig, silence(200, 48_000))
    rig.factory.current.feed(silence(20, 48_000), overflow_blocks={0})
    rig.session.tick()
    rig.factory.current.fail()
    rig.session.tick()
    rig.session.tick()
    play(rig, silence(100, 48_000))
    stats = rig.session.stats()
    assert stats.callbacks == 20 + 2 + 10
    assert stats.input_overflow == 1
    assert stats.reconnects == 1


def test_an_unsupported_device_rate_is_not_retried_silently() -> None:
    rig = build(rate_hz=96_000, connect=False)
    with pytest.raises(UnsupportedSampleRateError, match="96000"):
        rig.session.connect()


# ---------------------------------------------------------------- F5


@pytest.mark.parametrize("silence_ms", [450, 525, 600])
def test_f5_finalization_latency_stays_within_50_ms_of_the_threshold(silence_ms: int) -> None:
    rig = build(silence_ms=silence_ms)
    audio = stream_of(
        silence(400, 48_000),
        speech(900, 48_000),
        silence(1500, 48_000),
        speech(700, 48_000),
        silence(1500, 48_000),
        speech(1300, 48_000),
        silence(1500, 48_000),
    )
    play(rig, audio)

    latencies = rig.session.stats().finalization_latency_ms
    assert len(latencies) == 3
    # The segmenter rounds the threshold up to whole 32 ms frames, so even a perfect
    # detector reports up to 31 ms beyond it; the budget for everything else is 50 ms.
    assert all(0.0 <= value <= 50.0 for value in latencies), latencies


# ---------------------------------------------------------------- start / stop / thread


def test_start_fails_loudly_when_the_device_is_missing() -> None:
    rig = build(connect=False)
    rig.devices.remove(MIC)
    with pytest.raises(DeviceNotFoundError, match="V380"):
        rig.session.start()


def test_start_fails_loudly_when_the_stream_cannot_start() -> None:
    rig = build(connect=False)
    rig.factory.fail_starts = 1
    with pytest.raises(StreamOpenError):
        rig.session.start()
    assert not rig.session.connected
    assert rig.factory.current.closed


def test_stop_publishes_the_pending_chunk() -> None:
    rig = build()
    play(rig, stream_of(silence(300, 48_000), speech(1000, 48_000)))
    assert collect(rig.sub) == []
    rig.session.stop()
    (chunk,) = collect(rig.sub)
    assert chunk.reason == "flush"
    assert not rig.session.connected
    assert rig.factory.current.closed


def test_the_thread_turns_audio_into_chunks_and_stops_cleanly() -> None:
    rig = build(connect=False, ring_seconds=10)
    rig.session = CaptureSession(  # real clock for the real thread
        CaptureConfig("V380", ring_seconds=10, poll_ms=2),
        rig.devices,
        rig.factory,
        VadStage(EnergyVad(), Segmenter()),  # type: ignore[arg-type]
        _bus_of(rig),
    )
    rig.session.start()
    with pytest.raises(RuntimeError, match="already started"):
        rig.session.start()

    audio = stream_of(silence(300, 48_000), speech(800, 48_000), silence(1200, 48_000))
    rig.factory.current.feed(audio)
    chunk = rig.sub.get(timeout_s=3.0)
    assert chunk is not None
    assert chunk.reason == "silence"

    started = time.perf_counter()
    rig.session.stop()
    assert time.perf_counter() - started < 1.0
    assert rig.session.fatal_error is None


def test_a_fatal_error_in_the_thread_is_kept_for_the_dashboard() -> None:
    rig = build(connect=False)
    bus = _bus_of(rig)
    session = CaptureSession(
        CaptureConfig("V380", poll_ms=2, watchdog_ms=100),
        rig.devices,
        rig.factory,
        VadStage(EnergyVad(), Segmenter()),  # type: ignore[arg-type]
        bus,
    )
    session.start()
    rig.devices.devices = [make_device(MIC, rate_hz=96_000)]  # the "new" device is unsupported
    rig.factory.current.fail()
    deadline = time.monotonic() + 3.0
    while session.fatal_error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert isinstance(session.fatal_error, UnsupportedSampleRateError)
    session.stop()


def _bus_of(rig: Rig) -> ChunkBus:
    """Return a fresh bus whose subscriber replaces the rig's, for threaded tests."""
    bus = ChunkBus()
    rig.sub = bus.subscribe("threaded", maxsize=100)
    return bus
