"""SileroVad and VadStage with a scripted fake session (no model needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pipeline.vad import (
    FRAME_SAMPLES,
    ModelNotFoundError,
    Segmenter,
    SileroVad,
    VadStage,
)

pytestmark = pytest.mark.unit


class FakeSession:
    """Returns a scripted probability and adds 1 to the state so carry-over is visible."""

    def __init__(self, probs: list[float] | None = None) -> None:
        self.probs = probs or []
        self.calls: list[dict[str, Any]] = []

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        self.calls.append({k: np.array(v, copy=True) for k, v in input_feed.items()})
        index = len(self.calls) - 1
        prob = self.probs[index] if index < len(self.probs) else 0.0
        return [np.array([[prob]], dtype=np.float32), input_feed["state"] + 1.0]


def _frame(value: float) -> np.ndarray:
    return np.full(FRAME_SAMPLES, value, dtype=np.float32)


def test_each_call_prepends_the_last_64_samples_of_the_previous_frame() -> None:
    session = FakeSession()
    vad = SileroVad(Path("unused"), session=session)

    vad(_frame(0.1))
    vad(_frame(0.2))

    first, second = session.calls
    assert first["input"].shape == (1, 64 + FRAME_SAMPLES)
    assert np.all(first["input"][0, :64] == 0.0)  # nothing before the first frame
    assert np.all(second["input"][0, :64] == np.float32(0.1))
    assert np.all(second["input"][0, 64:] == np.float32(0.2))


def test_the_recurrent_state_is_carried_between_frames() -> None:
    session = FakeSession()
    vad = SileroVad(Path("unused"), session=session)

    for _ in range(3):
        vad(_frame(0.0))

    assert [float(c["state"].max()) for c in session.calls] == [0.0, 1.0, 2.0]
    assert session.calls[0]["state"].shape == (2, 1, 128)
    assert int(session.calls[0]["sr"]) == 16_000


def test_reset_clears_state_and_context() -> None:
    session = FakeSession()
    vad = SileroVad(Path("unused"), session=session)
    vad(_frame(0.3))
    vad.reset()

    vad(_frame(0.4))

    assert float(session.calls[1]["state"].max()) == 0.0
    assert np.all(session.calls[1]["input"][0, :64] == 0.0)


def test_the_probability_is_returned_as_a_float() -> None:
    vad = SileroVad(Path("unused"), session=FakeSession([0.8]))

    assert vad(_frame(0.0)) == pytest.approx(0.8)


def test_a_frame_of_the_wrong_size_is_rejected() -> None:
    vad = SileroVad(Path("unused"), session=FakeSession())

    with pytest.raises(ValueError, match="512"):
        vad(np.zeros(300, dtype=np.float32))


def test_a_missing_model_file_names_the_fetch_script() -> None:
    with pytest.raises(ModelNotFoundError, match="fetch_models"):
        SileroVad(Path("does-not-exist.onnx"))


def test_the_stage_frames_arbitrary_blocks_and_keeps_the_remainder() -> None:
    session = FakeSession()
    stage = VadStage(SileroVad(Path("unused"), session=session), Segmenter())
    rng = np.random.default_rng(0)

    stage.push(rng.standard_normal(300).astype(np.float32))
    assert len(session.calls) == 0
    stage.push(rng.standard_normal(1500).astype(np.float32))  # 1800 samples total -> 3 frames

    assert len(session.calls) == 3


def test_the_stage_gives_the_same_chunks_for_any_block_size() -> None:
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(60 * FRAME_SAMPLES).astype(np.float32)
    probs = [0.9] * 20 + [0.05] * 40

    def run(block: int) -> list[tuple[int, int]]:
        stage = VadStage(SileroVad(Path("unused"), session=FakeSession(probs)), Segmenter())
        chunks = []
        for start in range(0, len(audio), block):
            chunks += stage.push(audio[start : start + block])
        return [(c.start_sample, c.end_sample) for c in chunks]

    assert run(60 * FRAME_SAMPLES) == run(160) == run(777) == run(FRAME_SAMPLES)
    assert len(run(777)) == 1


def test_flush_ends_the_open_chunk_drops_the_partial_frame_and_resets_the_model() -> None:
    session = FakeSession([0.9] * 20)
    stage = VadStage(SileroVad(Path("unused"), session=session), Segmenter())
    stage.push(np.zeros(20 * FRAME_SAMPLES + 100, dtype=np.float32))

    chunk = stage.flush()
    stage.push(np.zeros(FRAME_SAMPLES, dtype=np.float32))

    assert chunk is not None
    assert chunk.reason == "flush"
    assert chunk.end_sample == 20 * FRAME_SAMPLES
    assert float(session.calls[-1]["state"].max()) == 0.0  # model was reset
    assert stage.segmenter.dropped_short == 0


def test_position_counts_pending_samples_and_drops_them_at_flush() -> None:
    stage = VadStage(SileroVad(Path("unused"), session=FakeSession()), Segmenter())
    stage.push(np.zeros(FRAME_SAMPLES + 100, dtype=np.float32))
    assert stage.segmenter.position_samples == FRAME_SAMPLES
    assert stage.position_samples == FRAME_SAMPLES + 100

    stage.flush()  # the partial frame is dropped, so the clock falls back to the frame edge
    assert stage.position_samples == FRAME_SAMPLES
