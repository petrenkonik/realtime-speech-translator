"""M1: the real Silero model plus Segmenter on speech with known pauses.

The speech is synthesized offline with the Windows SAPI voices, joined with pauses of
known length, so the expected chunk count and boundaries are exact. (The spec's
LibriSpeech variant needs a 300 MB download; this one needs nothing but the model.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from sapi_speech import build_stream, synthesize

from pipeline.vad import FRAME_SAMPLES, Segmenter, SegmenterConfig, SileroVad, VadStage

pytestmark = pytest.mark.model

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"


@pytest.fixture(scope="module")
def utterances(tmp_path_factory: pytest.TempPathFactory) -> list[NDArray[np.float32]]:
    out, error = synthesize(tmp_path_factory.mktemp("sapi"))
    if not out:
        pytest.skip(f"SAPI voices unavailable: {error}")
    return out


def _stage() -> VadStage:
    return VadStage(SileroVad(MODEL_PATH), Segmenter(SegmenterConfig()))


def test_m1_silero_scores_speech_high_and_noise_low(utterances: list[NDArray[np.float32]]) -> None:
    vad = SileroVad(MODEL_PATH)
    speech = np.concatenate([np.zeros(4 * FRAME_SAMPLES, dtype=np.float32), utterances[0]])
    speech = speech[: len(speech) // FRAME_SAMPLES * FRAME_SAMPLES]
    probs = [vad(speech[i : i + FRAME_SAMPLES]) for i in range(0, len(speech), FRAME_SAMPLES)]
    vad.reset()
    noise = np.random.default_rng(0).standard_normal(20 * FRAME_SAMPLES).astype(np.float32) * 0.001
    noise_probs = [vad(noise[i : i + FRAME_SAMPLES]) for i in range(0, len(noise), FRAME_SAMPLES)]

    assert max(probs) > 0.9
    assert max(noise_probs) < 0.2
    assert float(np.mean(probs[:4])) < 0.2  # leading silence


def test_m1_chunk_count_and_boundaries_match_the_known_pauses(
    utterances: list[NDArray[np.float32]],
) -> None:
    pre_roll = SegmenterConfig().pre_roll_ms * 16
    matches = 0
    errors_ms: list[float] = []
    for seed in range(10):
        audio = build_stream(utterances, seed)
        stage = _stage()
        chunks = stage.push(audio)
        tail = stage.flush()
        if tail is not None:
            chunks.append(tail)
        expected = _expected_spans(audio)
        if len(chunks) == len(expected):
            matches += 1
            for chunk, (start, end) in zip(chunks, expected, strict=True):
                errors_ms.append(abs(chunk.start_sample - (start - pre_roll)) / 16)
                errors_ms.append(abs(chunk.speech_end_sample - end) / 16)

    assert matches >= 9
    assert float(np.median(errors_ms)) <= 100.0


def _expected_spans(audio: NDArray[np.float32]) -> list[tuple[int, int]]:
    """Ground truth from the signal itself: loud stretches joined when the gap is < 450 ms."""
    frame = 160
    n = len(audio) // frame
    level = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    loud = np.flatnonzero(level > 0.01)
    spans: list[list[int]] = []
    for index in loud:
        if spans and (index - spans[-1][1]) * 10 < 450:
            spans[-1][1] = int(index)
        else:
            spans.append([int(index), int(index)])
    return [(a * frame, (b + 1) * frame) for a, b in spans]
