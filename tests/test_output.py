"""Output ring, underrun accounting, reconnect and independence of outputs (spec F6)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest
from fakes import (
    FakeClock,
    FakeDeviceList,
    FakeOutputStream,
    FakeStreamFactory,
    Float32Array,
    make_device,
)

from pipeline.devices import DeviceNotFoundError, StreamOpenError
from pipeline.output import (
    OutputConfig,
    OutputConfigError,
    OutputRing,
    OutputSession,
)

pytestmark = pytest.mark.unit

RATE = 48_000
BLOCK = 480  # 10 ms at 48 kHz
REALTEK = "Speakers (Realtek Audio)"
HS100W = "Speakers (HS100W)"


def ramp(frames: int, start: int = 0) -> Float32Array:
    """Distinct, easily checked samples (all small integers, exact in float32)."""
    return (start + np.arange(frames)).astype(np.float32)


@dataclass
class Rig:
    session: OutputSession
    devices: FakeDeviceList
    factory: FakeStreamFactory
    clock: FakeClock


def build(
    query: str = "Realtek",
    prebuffer_ms: int = 20,
    ring_seconds: float = 10.0,
    devices: FakeDeviceList | None = None,
    connect: bool = True,
    rate_hz: int = RATE,
) -> Rig:
    clock = FakeClock()
    if devices is None:
        devices = FakeDeviceList(
            [
                make_device(REALTEK, index=1, rate_hz=rate_hz, inputs=0, outputs=2),
                make_device(HS100W, index=2, rate_hz=rate_hz, inputs=0, outputs=2),
                make_device("Microphone (V380)", index=3, inputs=1),
            ]
        )
    factory = FakeStreamFactory(FakeOutputStream)
    config = OutputConfig(query, ring_seconds=ring_seconds, prebuffer_ms=prebuffer_ms)
    session = OutputSession(config, devices, factory, clock)
    if connect:
        session.connect()
    return Rig(session, devices, factory, clock)


# ---------------------------------------------------------------- ring


def test_ring_returns_samples_in_order_across_the_wrap_point() -> None:
    ring = OutputRing(10)
    out = np.zeros((4, 1), dtype=np.float32)
    got: list[float] = []
    value = 0
    for size in (4, 4, 4, 3, 4, 4):
        assert ring.write(ramp(size, value)) == size
        value += size
        count = ring.read_into(out, 4)
        got += out[:count, 0].tolist()
    assert got == list(range(value))


def test_ring_accepts_only_what_fits_and_never_overwrites_unplayed_audio() -> None:
    ring = OutputRing(8)
    assert ring.write(ramp(6)) == 6
    assert ring.write(ramp(6, 100)) == 2
    assert ring.free == 0
    out = np.zeros((8, 1), dtype=np.float32)
    assert ring.read_into(out, 8) == 8
    assert out[:, 0].tolist() == [0, 1, 2, 3, 4, 5, 100, 101]


def test_ring_fills_every_channel_with_the_mono_signal() -> None:
    ring = OutputRing(8)
    ring.write(ramp(3, 1))
    out = np.zeros((4, 2), dtype=np.float32)
    assert ring.read_into(out, 4) == 3
    assert out[:3].tolist() == [[1, 1], [2, 2], [3, 3]]


def test_discarding_drops_queued_audio_but_keeps_what_comes_later() -> None:
    ring = OutputRing(16)
    ring.write(ramp(5))
    ring.discard_queued()
    assert ring.available == 0
    ring.write(ramp(2, 50))
    out = np.zeros((4, 1), dtype=np.float32)
    assert ring.read_into(out, 4) == 2
    assert out[:2, 0].tolist() == [50, 51]


def test_an_empty_ring_reads_nothing() -> None:
    assert OutputRing(4).read_into(np.zeros((4, 1), dtype=np.float32), 4) == 0


def test_a_ring_needs_room_for_at_least_one_frame() -> None:
    with pytest.raises(OutputConfigError):
        OutputRing(0)


# ---------------------------------------------------------------- config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device_query": " "},
        {"block_ms": 1},
        {"block_ms": 101},
        {"ring_seconds": 0.1},
        {"ring_seconds": 61},
        {"prebuffer_ms": -1},
        {"prebuffer_ms": 101},
        {"watchdog_ms": 99},
        {"retry_ms": 0},
        {"poll_ms": 0},
    ],
)
def test_config_rejects_out_of_range_values(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {"device_query": "out"}
    with pytest.raises(OutputConfigError):
        OutputConfig(**{**base, **kwargs})  # type: ignore[arg-type]


def test_config_defaults() -> None:
    config = OutputConfig("out")
    assert (config.prebuffer_ms, config.ring_seconds, config.watchdog_ms) == (20, 10.0, 500)


# ---------------------------------------------------------------- playback


def test_audio_reaches_the_device_unchanged_on_both_channels() -> None:
    rig = build(prebuffer_ms=0)
    stream = rig.factory.current
    audio = ramp(BLOCK * 3)
    rig.session.begin_utterance()
    assert rig.session.write(audio) == len(audio)
    played = stream.pull(3)
    rig.session.end_utterance()
    assert played.shape == (BLOCK * 3, 2)
    assert np.array_equal(played[:, 0], audio)
    assert np.array_equal(played[:, 1], audio)
    stats = rig.session.stats()
    assert stats.frames_played == BLOCK * 3
    assert stats.underruns == 0


def test_the_sentinel_is_always_overwritten_so_silence_is_real_silence() -> None:
    rig = build()
    played = rig.factory.current.pull(2)
    assert not np.any(played)  # idle: exact zeros, no stale data
    assert rig.session.stats().underruns == 0


def test_a_device_with_more_channels_is_opened_with_two() -> None:
    devices = FakeDeviceList([make_device("Surround", inputs=0, outputs=8)])
    rig = build("Surround", devices=devices)
    assert rig.factory.current.channels == 2


def test_prebuffer_holds_playback_back_until_enough_audio_has_arrived() -> None:
    rig = build(prebuffer_ms=20)
    stream = rig.factory.current
    rig.session.begin_utterance()
    rig.session.write(ramp(480))  # 10 ms: less than the 20 ms prebuffer
    assert not np.any(stream.pull(1))
    rig.session.write(ramp(480, 480))  # now 20 ms are waiting
    played = stream.pull(2)
    assert played[0, 0] == 0.0  # starts with the first sample, in order
    assert np.array_equal(played[:, 0], ramp(960))
    assert rig.session.stats().underruns == 0  # waiting for the prebuffer is not an underrun


def test_a_short_utterance_below_the_prebuffer_still_plays_after_it_ends() -> None:
    rig = build(prebuffer_ms=50)
    stream = rig.factory.current
    rig.session.begin_utterance()
    rig.session.write(ramp(200))
    rig.session.end_utterance()
    played = stream.pull(1)
    assert np.array_equal(played[:200, 0], ramp(200))
    assert not np.any(played[200:])
    assert rig.session.stats().underruns == 0


def test_running_dry_during_speech_is_an_underrun_and_idle_silence_is_not() -> None:
    rig = build(prebuffer_ms=0)
    stream = rig.factory.current
    stream.pull(3)  # idle before the utterance: not counted
    rig.session.begin_utterance()
    rig.session.write(ramp(BLOCK + 100))
    stream.pull(2)  # the second block is short by 380 frames
    stats = rig.session.stats()
    assert stats.underruns == 1
    assert stats.underrun_frames == BLOCK - 100

    rig.session.end_utterance()
    stream.pull(3)  # silence after the utterance: not counted
    assert rig.session.stats().underruns == 1


def test_after_an_underrun_playback_waits_for_a_fresh_prebuffer() -> None:
    rig = build(prebuffer_ms=20)
    stream = rig.factory.current
    rig.session.begin_utterance()
    rig.session.write(ramp(960))
    stream.pull(2)
    stream.pull(1)  # ran dry once
    rig.session.write(ramp(240, 1000))  # 5 ms: below the prebuffer, so still silent
    assert not np.any(stream.pull(1))
    assert rig.session.stats().underruns == 1  # waiting is not a second underrun


def test_driver_underflow_flags_are_counted_separately() -> None:
    rig = build()
    rig.factory.current.pull(4, underflow_blocks={1, 3})
    assert rig.session.stats().driver_underflows == 2
    assert rig.session.stats().underruns == 0


def test_clear_drops_audio_that_has_not_played_yet() -> None:
    rig = build(prebuffer_ms=0)
    rig.session.begin_utterance()
    rig.session.write(ramp(BLOCK * 4))
    rig.factory.current.pull(1)
    rig.session.clear()
    assert rig.session.stats().queued_ms == 0.0


def test_queued_time_is_reported_in_milliseconds() -> None:
    rig = build()
    rig.session.write(ramp(RATE // 2))
    assert rig.session.stats().queued_ms == pytest.approx(500.0)


def test_the_callback_is_fast_and_counted() -> None:
    rig = build(prebuffer_ms=0)
    rig.session.write(ramp(RATE))
    rig.factory.current.pull(50)
    stats = rig.session.stats()
    assert stats.callbacks == 50
    assert 0.0 < stats.max_callback_ms < 5.0


def test_write_all_waits_for_room_while_a_consumer_drains_the_ring() -> None:
    rig = build(prebuffer_ms=0, ring_seconds=0.5)
    stream = rig.factory.current
    rig.session.begin_utterance()
    stop = threading.Event()

    def consume() -> None:
        while not stop.is_set():
            stream.pull(1)
            time.sleep(0.001)

    thread = threading.Thread(target=consume)
    thread.start()
    try:
        assert rig.session.write_all(ramp(RATE), timeout_s=5.0)  # 1 s into a 0.5 s ring
    finally:
        stop.set()
        thread.join()
    assert rig.session.stats().dropped_frames == 0


def test_write_all_gives_up_after_the_timeout_and_counts_the_rest() -> None:
    rig = build(ring_seconds=0.5)
    assert not rig.session.write_all(ramp(RATE), timeout_s=0.05)  # nobody drains the ring
    assert rig.session.stats().dropped_frames == RATE - RATE // 2


# ---------------------------------------------------------------- connection


def test_startup_fails_loudly_for_a_missing_output_device() -> None:
    rig = build("Nonexistent", connect=False)
    with pytest.raises(DeviceNotFoundError, match="Nonexistent"):
        rig.session.start()


def test_input_only_devices_are_not_outputs() -> None:
    rig = build("V380", connect=False)
    with pytest.raises(DeviceNotFoundError):
        rig.session.connect()


def test_startup_fails_loudly_when_the_stream_cannot_start() -> None:
    rig = build(connect=False)
    rig.factory.fail_starts = 1
    with pytest.raises(StreamOpenError):
        rig.session.start()
    assert rig.factory.current.closed


def test_rate_and_generation_follow_the_connected_device() -> None:
    rig = build(rate_hz=44_100)
    assert rig.session.sample_rate_hz == 44_100
    assert rig.session.generation == 1
    rig.factory.current.fail()
    rig.session.tick()
    assert rig.session.sample_rate_hz is None
    rig.devices.devices = [make_device(REALTEK, rate_hz=48_000, inputs=0, outputs=2)]
    rig.session.tick()
    assert rig.session.sample_rate_hz == 48_000
    assert rig.session.generation == 2


def test_audio_written_while_disconnected_is_discarded_and_counted() -> None:
    rig = build()
    rig.factory.current.fail()
    rig.session.tick()
    rig.devices.remove(REALTEK)
    rig.session.tick()  # reconnect attempt fails
    assert rig.session.write(ramp(1000)) == 1000
    assert rig.session.stats().dropped_frames == 1000


def test_queued_audio_is_counted_as_dropped_when_the_device_is_lost() -> None:
    rig = build(prebuffer_ms=0)
    rig.session.write(ramp(BLOCK * 2))
    rig.factory.current.fail()
    rig.session.tick()
    assert rig.session.stats().dropped_frames == BLOCK * 2


def test_the_watchdog_declares_the_output_lost_after_500_ms_without_callbacks() -> None:
    rig = build()
    rig.factory.current.pull(2)
    rig.session.tick()  # sees the callbacks; the silence starts from here
    rig.clock.advance(0.4)
    rig.session.tick()
    assert rig.session.connected
    rig.clock.advance(0.2)
    rig.session.tick()
    assert not rig.session.connected
    assert "no audio callback" in rig.session.stats().last_loss_reason


def test_counters_survive_a_reconnect() -> None:
    rig = build(prebuffer_ms=0)
    rig.session.begin_utterance()
    rig.session.write(ramp(BLOCK))
    rig.factory.current.pull(2)  # one block played, one underrun
    rig.factory.current.fail()
    rig.session.tick()
    rig.session.tick()
    rig.factory.current.pull(1)
    stats = rig.session.stats()
    assert stats.callbacks == 3
    assert stats.frames_played == BLOCK
    assert stats.underruns == 2  # the utterance is still open on the new stream
    assert stats.reconnects == 1


def test_start_runs_a_supervisor_thread_and_stop_closes_the_stream() -> None:
    rig = build(connect=False)
    session = OutputSession(
        OutputConfig("Realtek", poll_ms=2), rig.devices, rig.factory
    )  # real clock for the real thread
    session.start()
    with pytest.raises(RuntimeError, match="already started"):
        session.start()
    stream = rig.factory.current
    stream.fail()
    deadline = time.monotonic() + 3.0
    while len(rig.factory.streams) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(rig.factory.streams) == 2  # the thread reconnected by itself
    session.stop()
    assert rig.factory.current.closed
    assert session.fatal_error is None


# ---------------------------------------------------------------- F6


def test_f6_one_output_failing_does_not_disturb_the_other() -> None:
    devices = FakeDeviceList(
        [
            make_device(REALTEK, index=1, inputs=0, outputs=2),
            make_device(HS100W, index=2, inputs=0, outputs=2),
        ]
    )
    a = build("Realtek", prebuffer_ms=0, devices=devices)
    b = build("HS100W", prebuffer_ms=0, devices=devices)
    a.session.begin_utterance()
    b.session.begin_utterance()

    heard_b: list[Float32Array] = []
    offset = 0
    for step in range(30):
        if step == 10:
            a.factory.current.fail()  # the Realtek output dies mid-speech
        if step == 12:
            devices.remove(REALTEK)  # and stays away for a while
        if step == 20:
            devices.devices.append(make_device(REALTEK, index=9, inputs=0, outputs=2))
        for rig in (a, b):
            rig.clock.advance(0.1)
            rig.session.write(ramp(BLOCK, offset))
        offset += BLOCK
        for rig in (a, b):
            rig.session.tick()
        if a.session.connected:
            a.factory.current.pull(1)
        heard_b.append(b.factory.current.pull(1))

    # B played every sample exactly once, in order, without a gap.
    heard = np.concatenate(heard_b)
    assert np.array_equal(heard[:, 0], ramp(BLOCK * 30))
    b_stats = b.session.stats()
    assert (b_stats.losses, b_stats.reconnects, b_stats.underruns) == (0, 0, 0)
    assert b_stats.dropped_frames == 0

    # A noticed, dropped the audio meant for the outage, and came back.
    a_stats = a.session.stats()
    assert a_stats.losses == 1
    assert a_stats.reconnects == 1
    assert a.session.connected
    assert a_stats.dropped_frames > 0


def test_f6_a_failing_open_on_one_output_leaves_the_other_untouched() -> None:
    devices = FakeDeviceList(
        [
            make_device(REALTEK, index=1, inputs=0, outputs=2),
            make_device(HS100W, index=2, inputs=0, outputs=2),
        ]
    )
    a = build("Realtek", prebuffer_ms=0, devices=devices)
    b = build("HS100W", prebuffer_ms=0, devices=devices)
    a.factory.current.fail()
    a.session.tick()
    a.factory.fail_opens = 5
    for _ in range(3):
        a.clock.advance(1.1)
        a.session.tick()
    assert not a.session.connected
    b.session.write(ramp(BLOCK))
    assert np.array_equal(b.factory.current.pull(1)[:, 0], ramp(BLOCK))
    assert b.session.connected
