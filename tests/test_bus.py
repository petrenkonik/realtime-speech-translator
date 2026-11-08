"""Tests for the multi-consumer bus (spec U11-U14 plus edge cases)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pipeline.bus import Bus, BusClosedError, BusConfigError, ChunkBus, Subscription
from pipeline.vad import Chunk

pytestmark = pytest.mark.unit


def make_chunk(chunk_id: int, n_samples: int = 16) -> Chunk:
    samples = np.zeros(n_samples, dtype=np.float32)
    samples.flags.writeable = False
    return Chunk(
        id=chunk_id,
        samples=samples,
        start_sample=0,
        end_sample=n_samples,
        speech_end_sample=n_samples,
        finalized_sample=n_samples,
        reason="silence",
    )


def drain(sub: Subscription[Chunk], out: list[int]) -> None:
    """Consumer loop used by tests: collect ids until the bus ends."""
    while (item := sub.get()) is not None:
        out.append(item.id)


def join_all(threads: list[threading.Thread], timeout_s: float = 5.0) -> None:
    for thread in threads:
        thread.join(timeout_s)
        assert not thread.is_alive()


# ---------------------------------------------------------------- U11


def test_u11_fan_out_to_eight_subscribers() -> None:
    bus = ChunkBus()
    received: list[list[int]] = [[] for _ in range(8)]
    threads = []
    for i in range(8):
        # maxsize large enough that nothing may be dropped in this test.
        sub = bus.subscribe(f"s{i}", maxsize=10_000)
        thread = threading.Thread(target=drain, args=(sub, received[i]))
        thread.start()
        threads.append(thread)

    for chunk_id in range(1, 10_001):
        bus.publish(make_chunk(chunk_id))
    bus.close(drain=True)
    join_all(threads)

    expected = list(range(1, 10_001))
    for got in received:
        assert got == expected  # all received, ordered, no duplicates


def test_same_object_is_shared_between_subscribers() -> None:
    bus = ChunkBus()
    a = bus.subscribe("a")
    b = bus.subscribe("b")
    chunk = make_chunk(1)
    bus.publish(chunk)
    assert a.get(timeout_s=1) is chunk
    assert b.get(timeout_s=1) is chunk


def test_subscriber_added_later_sees_only_new_items() -> None:
    bus = ChunkBus()
    early = bus.subscribe("early")
    bus.publish(make_chunk(1))
    late = bus.subscribe("late")
    bus.publish(make_chunk(2))
    assert [early.get(timeout_s=1).id, early.get(timeout_s=1).id] == [1, 2]  # type: ignore[union-attr]
    first = late.get(timeout_s=1)
    assert first is not None
    assert first.id == 2


def test_publish_without_subscribers_is_fine() -> None:
    bus = ChunkBus()
    bus.publish(make_chunk(1))
    assert bus.published == 1


# ---------------------------------------------------------------- U12


def test_u12_slow_subscriber_never_blocks_publisher() -> None:
    bus = ChunkBus()
    fast_ids: list[int] = []
    fast = bus.subscribe("fast", maxsize=10_000)
    slow = bus.subscribe("slow", maxsize=4)  # never consumed during the run
    thread = threading.Thread(target=drain, args=(fast, fast_ids))
    thread.start()

    durations_ms: list[float] = []
    for chunk_id in range(1, 2_001):
        started = time.perf_counter()
        bus.publish(make_chunk(chunk_id))
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    bus.close(drain=True)
    join_all([thread])

    durations_ms.sort()
    # The 99th percentile is checked instead of the maximum: one OS scheduling
    # hiccup on a busy CI machine must not fail an otherwise correct run.
    assert durations_ms[int(len(durations_ms) * 0.99)] < 1.0
    assert fast_ids == list(range(1, 2_001))  # the others got everything
    stats = slow.stats()
    assert stats.dropped == 2_000 - 4
    assert stats.depth == 4
    assert stats.delivered == 0


def test_drop_oldest_keeps_the_newest_items() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("s", maxsize=3)
    for chunk_id in range(1, 8):
        bus.publish(make_chunk(chunk_id))
    bus.close(drain=True)
    got: list[int] = []
    drain(sub, got)
    assert got == [5, 6, 7]
    assert sub.stats().dropped == 4


def test_drop_newest_keeps_the_oldest_items() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("s", maxsize=3, policy="drop_newest")
    for chunk_id in range(1, 8):
        bus.publish(make_chunk(chunk_id))
    bus.close(drain=True)
    got: list[int] = []
    drain(sub, got)
    assert got == [1, 2, 3]
    assert sub.stats().dropped == 4


def test_stats_track_depth_delivered_and_high_water_mark() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("s", maxsize=8)
    for chunk_id in range(1, 6):
        bus.publish(make_chunk(chunk_id))
    assert sub.stats().depth == 5
    assert sub.get(timeout_s=1) is not None
    assert sub.get(timeout_s=1) is not None
    stats = sub.stats()
    assert (stats.delivered, stats.dropped, stats.depth, stats.max_depth) == (2, 0, 3, 5)
    assert sub.name == "s"


def test_bus_stats_lists_every_subscriber() -> None:
    bus = ChunkBus()
    bus.subscribe("a", maxsize=1)
    bus.subscribe("b", maxsize=2)
    for chunk_id in range(1, 4):
        bus.publish(make_chunk(chunk_id))
    stats = bus.stats()
    assert set(stats) == {"a", "b"}
    assert stats["a"].dropped == 2
    assert stats["b"].dropped == 1
    assert bus.published == 3


# ---------------------------------------------------------------- U13


def test_u13_close_stops_every_consumer_within_one_second() -> None:
    bus = ChunkBus()
    threads = []
    for i in range(8):
        sub = bus.subscribe(f"s{i}")
        thread = threading.Thread(target=drain, args=(sub, []), daemon=True)
        thread.start()
        threads.append(thread)
    time.sleep(0.05)  # let every consumer block in get()

    started = time.perf_counter()
    bus.close()
    for thread in threads:
        thread.join(1.0)
    elapsed_s = time.perf_counter() - started

    assert all(not thread.is_alive() for thread in threads)
    assert elapsed_s < 1.0


def test_u13_close_does_not_wait_for_a_slow_consumer_backlog() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("slow", maxsize=32)
    handled: list[int] = []

    def slow_consumer() -> None:
        while (item := sub.get()) is not None:
            handled.append(item.id)
            time.sleep(0.5)  # the demo --slow-consumer sleeps even longer

    thread = threading.Thread(target=slow_consumer, daemon=True)
    thread.start()
    for chunk_id in range(1, 21):
        bus.publish(make_chunk(chunk_id))
    time.sleep(0.05)

    started = time.perf_counter()
    bus.close()
    thread.join(1.5)  # one in-flight item (0.5 s) may finish, the backlog may not
    assert not thread.is_alive()
    assert time.perf_counter() - started < 1.0
    assert len(handled) < 20


def test_close_with_drain_delivers_the_backlog_first() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("s")
    for chunk_id in range(1, 5):
        bus.publish(make_chunk(chunk_id))
    bus.close(drain=True)
    got: list[int] = []
    drain(sub, got)
    assert got == [1, 2, 3, 4]


def test_close_without_drain_discards_the_backlog() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("s")
    for chunk_id in range(1, 5):
        bus.publish(make_chunk(chunk_id))
    bus.close()
    assert sub.get() is None
    assert sub.stats().depth == 0


def test_get_returns_none_forever_after_close() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("s")
    bus.close()
    assert sub.get() is None
    assert sub.get() is None
    assert bus.closed


def test_close_is_idempotent() -> None:
    bus = ChunkBus()
    bus.close()
    bus.close()
    assert bus.closed


def test_publish_after_close_is_ignored() -> None:
    # The audio thread can race with shutdown; it must not crash.
    bus = ChunkBus()
    sub = bus.subscribe("s")
    bus.close()
    bus.publish(make_chunk(1))
    assert sub.get() is None
    assert bus.published == 0


def test_subscribe_after_close_raises() -> None:
    bus = ChunkBus()
    bus.close()
    with pytest.raises(BusClosedError, match="closed"):
        bus.subscribe("late")


def test_get_timeout_returns_none_when_idle_but_open() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("s")
    started = time.perf_counter()
    assert sub.get(timeout_s=0.05) is None
    assert time.perf_counter() - started >= 0.04
    assert not sub.finished


def test_finished_flag_follows_close_and_backlog() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("s")
    bus.publish(make_chunk(1))
    bus.close(drain=True)
    assert not sub.finished  # one item still waits
    assert sub.get(timeout_s=1) is not None
    assert sub.finished
    assert sub.get(timeout_s=1) is None


def test_close_wakes_a_blocked_consumer() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("s")
    results: list[Chunk | None] = []
    thread = threading.Thread(target=lambda: results.append(sub.get()))
    thread.start()
    time.sleep(0.05)
    bus.close()
    join_all([thread], timeout_s=1.0)
    assert results == [None]


def test_iteration_ends_at_close() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("s")
    for chunk_id in range(1, 4):
        bus.publish(make_chunk(chunk_id))
    bus.close(drain=True)
    assert [chunk.id for chunk in sub] == [1, 2, 3]


# ---------------------------------------------------------------- U14


def test_u14_chunk_samples_are_read_only() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("s")
    bus.publish(make_chunk(1))
    chunk = sub.get(timeout_s=1)
    assert chunk is not None
    with pytest.raises(ValueError, match="read-only"):
        chunk.samples[0] = 1.0


# ---------------------------------------------------------------- validation


def test_subscriber_names_must_be_unique() -> None:
    bus = ChunkBus()
    bus.subscribe("a")
    with pytest.raises(BusConfigError, match="already"):
        bus.subscribe("a")


def test_empty_name_is_rejected() -> None:
    with pytest.raises(BusConfigError, match="name"):
        ChunkBus().subscribe("")


@pytest.mark.parametrize("maxsize", [0, -1])
def test_maxsize_must_be_positive(maxsize: int) -> None:
    with pytest.raises(BusConfigError, match="maxsize"):
        ChunkBus().subscribe("a", maxsize=maxsize)


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(BusConfigError, match="policy"):
        ChunkBus().subscribe("a", policy="block")  # type: ignore[arg-type]


def test_generic_bus_carries_any_payload() -> None:
    # TextBus is the same class carrying utterances.
    bus: Bus[str] = Bus()
    sub = bus.subscribe("s")
    bus.publish("hello")
    assert sub.get(timeout_s=1) == "hello"


def test_concurrent_publishers_and_subscribe_do_not_corrupt_state() -> None:
    bus = ChunkBus()
    stop = threading.Event()
    errors: list[BaseException] = []

    def publisher() -> None:
        try:
            chunk_id = 0
            while not stop.is_set():
                chunk_id += 1
                bus.publish(make_chunk(chunk_id))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=publisher) for _ in range(3)]
    for thread in threads:
        thread.start()
    for i in range(20):
        sub = bus.subscribe(f"late{i}", maxsize=2)
        assert sub.get(timeout_s=1) is not None
    stop.set()
    bus.close()
    join_all(threads)
    assert errors == []


# --- unsubscribe (U15) ----------------------------------------------------------------------


def test_u15_unsubscribe_ends_one_subscriber_and_leaves_the_others_alone() -> None:
    bus = ChunkBus()
    leaving = bus.subscribe("leaving", maxsize=4)
    staying = bus.subscribe("staying", maxsize=4)
    bus.publish(make_chunk(1))
    stats_before = staying.stats()

    assert bus.unsubscribe(leaving) is True
    assert leaving.finished  # closed and its backlog discarded
    assert leaving.get(timeout_s=0) is None
    assert "leaving" not in bus.stats()
    assert bus.stats()["staying"] == stats_before  # counters of the others unchanged

    bus.publish(make_chunk(2))
    assert leaving.get(timeout_s=0) is None  # nothing reaches it any more
    assert [item.id for item in (staying.get(timeout_s=0), staying.get(timeout_s=0)) if item] == [
        1,
        2,
    ]
    assert not bus.closed


def test_u15_unsubscribe_wakes_a_blocked_consumer() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("a")
    seen: list[int] = []
    thread = threading.Thread(target=drain, args=(sub, seen))
    thread.start()
    time.sleep(0.05)
    bus.unsubscribe(sub)
    join_all([thread], timeout_s=1.0)
    assert seen == []


def test_u15_unsubscribe_is_idempotent_and_the_name_can_be_reused() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("a")
    assert bus.unsubscribe(sub) is True
    assert bus.unsubscribe(sub) is False
    foreign = ChunkBus().subscribe("a")
    assert bus.unsubscribe(foreign) is False  # not one of ours: nothing happens
    assert not foreign.finished
    again = bus.subscribe("a")
    bus.publish(make_chunk(1))
    assert again.get(timeout_s=0) is not None


def test_u15_unsubscribe_after_close_is_harmless() -> None:
    bus = ChunkBus()
    sub = bus.subscribe("a")
    bus.close()
    bus.unsubscribe(sub)
    assert sub.finished
