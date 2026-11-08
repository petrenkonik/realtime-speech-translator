"""The Smart Turn classifier wrapper, the two links and their use inside VadStage."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fakes import EnergyVad

from pipeline.turn import (
    OnnxTurnClassifier,
    SyncTurnLink,
    ThreadedTurnLink,
    TurnModelError,
)
from pipeline.vad import (
    FRAME_SAMPLES,
    ModelNotFoundError,
    Segmenter,
    SegmenterConfig,
    TurnAnswer,
    TurnRequest,
    VadConfigError,
    VadStage,
)

pytestmark = pytest.mark.unit


def request(request_id: int = 1, samples: int = 1_000) -> TurnRequest:
    window = np.zeros(samples, dtype=np.float32)
    return TurnRequest(request_id, window, 0.5, "turn")


class Constant:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.calls = 0

    def predict(self, window: Any) -> float:
        self.calls += 1
        return self.probability


def wait_for(predicate: Any, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# --- OnnxTurnClassifier ------------------------------------------------------------


class FakeSession:
    def __init__(self, output: Any) -> None:
        self.output = output
        self.feeds: list[dict[str, Any]] = []

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]:
        self.feeds.append(input_feed)
        return [np.asarray(self.output, dtype=np.float32)]


def test_the_classifier_feeds_features_of_the_right_shape_and_reads_the_probability() -> None:
    session = FakeSession([[0.8]])
    classifier = OnnxTurnClassifier(Path("unused.onnx"), session=session)
    probability = classifier.predict(np.zeros(16_000, dtype=np.float32))

    assert probability == pytest.approx(0.8)
    feed = session.feeds[-1]
    assert set(feed) == {"input_features"}
    assert feed["input_features"].shape == (1, 80, 800)
    assert feed["input_features"].dtype == np.float32


def test_the_session_asks_for_the_non_saturating_int8_kernels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without this, int8 matmuls on CPUs that lack VNNI (e.g. Zen1) can saturate and give
    # answers that differ a lot from the fp32 model (measured at step 2b follow-up).
    import onnxruntime as ort

    seen: dict[str, Any] = {}

    class RecordingSession(FakeSession):
        def __init__(self, path: str, sess_options: Any, providers: list[str]) -> None:
            super().__init__([[0.5]])
            seen["options"] = sess_options
            seen["providers"] = providers

    monkeypatch.setattr(ort, "InferenceSession", RecordingSession)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not a real model")
    OnnxTurnClassifier(model)
    assert seen["options"].get_session_config_entry("session.x64quantprecision") == "1"
    assert seen["providers"] == ["CPUExecutionProvider"]


def test_the_classifier_warms_up_when_it_is_created() -> None:
    session = FakeSession([0.3])
    OnnxTurnClassifier(Path("unused.onnx"), session=session)
    assert len(session.feeds) == 1


def test_an_output_that_is_not_a_probability_is_rejected_at_load() -> None:
    with pytest.raises(TurnModelError, match="not a probability"):
        OnnxTurnClassifier(Path("unused.onnx"), session=FakeSession([[7.3]]))


def test_a_missing_model_file_names_the_fetch_script() -> None:
    with pytest.raises(ModelNotFoundError, match=r"fetch_models\.py"):
        OnnxTurnClassifier(Path("does-not-exist.onnx"))


@pytest.mark.parametrize("threads", [0, 3, -1])
def test_the_thread_count_is_limited(threads: int) -> None:
    with pytest.raises(TurnModelError, match="threads"):
        OnnxTurnClassifier(Path("unused.onnx"), threads=threads, session=FakeSession([0.5]))


# --- SyncTurnLink ------------------------------------------------------------------


def test_the_sync_link_answers_after_the_configured_number_of_frames() -> None:
    link = SyncTurnLink(Constant(0.7), delay_frames=3)
    assert link.poll() is None  # frame 1
    link.submit(request(5))
    assert [link.poll(), link.poll()] == [None, None]
    answer = link.poll()
    assert answer == TurnAnswer(5, pytest.approx(0.7))  # type: ignore[arg-type]
    assert link.poll() is None


def test_the_sync_link_reset_forgets_pending_answers() -> None:
    link = SyncTurnLink(Constant(0.7), delay_frames=2)
    link.submit(request())
    link.reset()
    assert [link.poll() for _ in range(5)] == [None] * 5


def test_an_instant_answer_is_not_allowed() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        SyncTurnLink(Constant(0.5), delay_frames=0)


# --- ThreadedTurnLink --------------------------------------------------------------


class Gated:
    """A classifier that blocks until the test lets it go."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.seen: list[int] = []

    def predict(self, window: Any) -> float:
        self.seen.append(len(window))
        self.started.set()
        assert self.release.wait(5.0)
        return 0.9


def test_the_threaded_link_delivers_the_answer_without_blocking_the_caller() -> None:
    gate = Gated()
    link = ThreadedTurnLink(gate)
    try:
        started = time.perf_counter()
        link.submit(request(7))
        assert time.perf_counter() - started < 0.05  # submit never waits for the model
        assert gate.started.wait(2.0)
        assert link.poll() is None  # still computing; poll does not block either
        gate.release.set()
        assert wait_for(lambda: link.poll() == TurnAnswer(7, pytest.approx(0.9)))  # type: ignore[arg-type]
    finally:
        link.close()


def test_a_newer_request_replaces_one_the_model_has_not_started() -> None:
    gate = Gated()
    link = ThreadedTurnLink(gate)
    try:
        link.submit(request(1, samples=100))
        assert gate.started.wait(2.0)  # request 1 is being computed
        link.submit(request(2, samples=200))
        link.submit(request(3, samples=300))  # replaces 2 in the one-slot mailbox
        gate.release.set()
        assert wait_for(lambda: len(gate.seen) == 2)
        assert gate.seen == [100, 300]
        assert link.superseded == 1
    finally:
        link.close()


def test_reset_discards_a_prediction_that_was_running() -> None:
    gate = Gated()
    link = ThreadedTurnLink(gate)
    try:
        link.submit(request(1))
        assert gate.started.wait(2.0)
        link.reset()
        gate.release.set()
        assert wait_for(lambda: len(link.model_times_ms()) == 1)
        assert link.poll() is None  # the answer belonged to the interrupted stream
    finally:
        link.close()


def test_a_failing_classifier_costs_an_answer_but_not_the_thread() -> None:
    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, window: Any) -> float:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model crashed")
            return 0.6

    link = ThreadedTurnLink(Flaky())
    try:
        link.submit(request(1))
        assert wait_for(lambda: link.errors == 1)
        assert "model crashed" in link.last_error
        link.submit(request(2))
        assert wait_for(lambda: link.poll() == TurnAnswer(2, pytest.approx(0.6)))  # type: ignore[arg-type]
    finally:
        link.close()


def test_prediction_times_are_recorded_and_close_stops_the_thread() -> None:
    clock_values = iter([10.0, 10.025])
    link = ThreadedTurnLink(Constant(0.5), clock=lambda: next(clock_values))
    link.submit(request())
    assert wait_for(lambda: link.model_times_ms() == [pytest.approx(25.0)])
    started = time.perf_counter()
    link.close()
    assert time.perf_counter() - started < 1.0
    assert not any(t.name == "turn-model" and t.is_alive() for t in threading.enumerate())


# --- VadStage with a link ----------------------------------------------------------


def tone(frames: int, level: float = 0.3) -> Any:
    t = np.arange(frames * FRAME_SAMPLES) / 16_000
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(frames: int) -> Any:
    return np.zeros(frames * FRAME_SAMPLES, dtype=np.float32)


def test_smart_mode_without_a_link_is_a_config_error() -> None:
    with pytest.raises(VadConfigError, match="turn link"):
        VadStage(EnergyVad(), Segmenter(SegmenterConfig(turn_mode="smart")))  # type: ignore[arg-type]


def test_a_link_is_ignored_in_fixed_mode() -> None:
    link = SyncTurnLink(Constant(0.99))
    stage = VadStage(EnergyVad(), Segmenter(), link)  # type: ignore[arg-type]
    stage.push(np.concatenate([tone(20), quiet(40)]))
    assert link.poll() is None


def test_the_stage_ends_the_chunk_earlier_in_smart_mode_than_in_fixed_mode() -> None:
    audio = np.concatenate([tone(20), quiet(40)])

    fixed = VadStage(EnergyVad(), Segmenter())  # type: ignore[arg-type]
    smart = VadStage(
        EnergyVad(),  # type: ignore[arg-type]
        Segmenter(SegmenterConfig(turn_mode="smart")),
        SyncTurnLink(Constant(0.95)),
    )
    (late,) = fixed.push(audio)
    (early,) = smart.push(audio)

    assert early.reason == "turn"
    assert late.reason == "silence"
    assert late.finalized_sample - early.finalized_sample >= 6 * FRAME_SAMPLES
    np.testing.assert_array_equal(early.samples, late.samples)


def test_a_flush_resets_the_link() -> None:
    link = SyncTurnLink(Constant(0.95), delay_frames=50)
    stage = VadStage(
        EnergyVad(),  # type: ignore[arg-type]
        Segmenter(SegmenterConfig(turn_mode="smart")),
        link,
    )
    stage.push(np.concatenate([tone(20), quiet(10)]))
    stage.flush()
    assert [link.poll() for _ in range(60)] == [None] * 60
