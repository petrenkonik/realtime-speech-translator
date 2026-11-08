"""The sweep's Reference port: must match the real Segmenter exactly, and be checkable by hand."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from pipeline.vad import FRAME_SAMPLES, Segmenter, SegmenterConfig, VadStage

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent


def load_sweep() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sweep_endpointing", ROOT / "scripts/sweep_endpointing.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["sweep_endpointing"] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


sweep = load_sweep()


class ReplayProbabilities:
    """Plays back given probabilities, like the evaluation does with Silero's output."""

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = probabilities
        self._next = 0

    def __call__(self, frame: np.ndarray) -> float:
        value = self._probabilities[self._next]
        self._next += 1
        return value

    def reset(self) -> None:
        """Nothing to reset."""


def random_stream(seed: int) -> tuple[np.ndarray, list[float]]:
    """Speech, silence and in-between runs, plus one speech run long enough to force a cut."""
    rng = np.random.default_rng(seed)
    probabilities: list[float] = []
    while len(probabilities) < 2_500:
        kind = rng.integers(3)
        length = int(rng.integers(1, 60))
        if kind == 0:
            probabilities += list(rng.uniform(0.6, 1.0, length))
        elif kind == 1:
            probabilities += list(rng.uniform(0.0, 0.3, length))
        else:
            probabilities += list(rng.uniform(0.35, 0.5, length))
    probabilities += [0.9] * 520  # longer than max_utterance_ms, so the hard cut runs
    probabilities += [0.1] * 40
    amplitude = rng.uniform(0.001, 0.2, len(probabilities))
    noise = rng.standard_normal(len(probabilities) * FRAME_SAMPLES)
    audio = (noise * np.repeat(amplitude, FRAME_SAMPLES)).astype(np.float32)
    return audio, probabilities


@pytest.mark.parametrize(
    ("speech", "exit_threshold", "silence_ms"),
    [(0.5, 0.35, 450), (0.6, 0.6, 600), (0.3, 0.15, 500)],
)
def test_reference_port_matches_the_real_segmenter(
    speech: float, exit_threshold: float, silence_ms: int
) -> None:
    rule = sweep.Rule("hysteresis", speech, exit_threshold, silence_ms)
    hard_cuts = 0
    for seed in (1, 2, 3):
        audio, probabilities = random_stream(seed)
        config = SegmenterConfig(
            silence_ms=silence_ms,
            speech_threshold=speech,
            exit_threshold=exit_threshold,
            turn_mode="fixed",
        )
        segmenter = Segmenter(config)
        stage = VadStage(ReplayProbabilities(probabilities), segmenter)  # type: ignore[arg-type]
        chunks = stage.push(audio)
        last = stage.flush()
        if last is not None:
            chunks.append(last)
        hard_cuts += segmenter.turn_hard_cuts
        expected = [(c.speech_end_sample, c.finalized_sample) for c in chunks]

        rms = sweep.frame_rms(audio)
        assert sweep.Reference(rule).run(probabilities, rms) == expected
    assert hard_cuts >= 1  # the long speech run must reach the max-length cut path


def test_integral_equals_hysteresis_for_binary_probabilities() -> None:
    # With p in {0, 1} every silent frame counts 1 and the clock starts right after the last
    # speech frame, so both silence tests fire on the same frame.
    rng = np.random.default_rng(7)
    probabilities = [float(v) for v in rng.integers(0, 2, 3_000)]
    probabilities += [1.0] * 520 + [0.0] * 40
    rms = [0.01] * len(probabilities)
    for silence_ms in (450, 500, 600):
        frames = sweep.Rule("integral", 0.5, 0.5, silence_ms).silence_frames
        integral = sweep.Rule("integral", 0.5, 0.5, frames * 32)
        hysteresis = sweep.Rule("hysteresis", 0.5, 0.35, silence_ms)
        assert sweep.Reference(integral).run(probabilities, rms) == sweep.Reference(hysteresis).run(
            probabilities, rms
        )


def test_integral_counts_uncertain_frames_as_less_silence() -> None:
    # Frames at p = 0.2 add 0.8 each, so a 15-frame integral needs 19 of them (15 / 0.8 = 18.75).
    # Stall rule (2 x 15 frames) is not reached, so only the integral decides here.
    probabilities = [1.0] * 5 + [0.2] * 40
    rms = [0.01] * len(probabilities)
    rule = sweep.Rule("integral", 0.7, 0.7, 450)  # 15 frames
    cuts = sweep.Reference(rule).run(probabilities, rms)
    assert cuts == [(5 * FRAME_SAMPLES, (5 + 19) * FRAME_SAMPLES)]


def test_drift_cusum_and_band_equal_hysteresis_on_binary_probabilities() -> None:
    # With p in {0, 1} the level c = 0.35 adds 0.35 per silent frame, so c * N fires at frame N,
    # the same frame as the hysteresis clock; the band adds 1 per silent frame.
    rng = np.random.default_rng(11)
    probabilities = [float(v) for v in rng.integers(0, 2, 3_000)]
    probabilities += [1.0] * 520 + [0.0] * 40
    rms = [0.01] * len(probabilities)
    for frames in (15, 16, 19):
        expected = sweep.Reference(sweep.Rule("hysteresis", 0.5, 0.35, frames * 32)).run(
            probabilities, rms
        )
        for kind in ("drift", "cusum", "band"):
            rule = sweep.Rule(kind, 0.5, 0.35, frames * 32)
            assert sweep.Reference(rule).run(probabilities, rms) == expected, kind


def _speech_doubt_silence() -> tuple[list[float], list[float]]:
    # 5 speech frames, 3 doubtful frames at p = 0.45 (between c = 0.35 and 0.5), then silence.
    probabilities = [1.0] * 5 + [0.45] * 3 + [0.0] * 20
    return probabilities, [0.01] * len(probabilities)


def test_drift_is_pulled_back_by_doubtful_frames() -> None:
    # Each doubtful frame adds 0.35 - 0.45 = -0.10, so S = -0.30 before the silence starts.
    # Silence adds 0.35 per frame and the chunk ends at S >= 0.35 * 15 = 5.25: 16 frames.
    probabilities, rms = _speech_doubt_silence()
    rule = sweep.Rule("drift", 0.5, 0.35, 450)
    assert sweep.Reference(rule).run(probabilities, rms) == [
        (5 * FRAME_SAMPLES, 24 * FRAME_SAMPLES)
    ]


def test_cusum_never_goes_below_zero() -> None:
    # The same doubtful frames leave S at 0 instead of -0.30, so silence needs 15 frames.
    probabilities, rms = _speech_doubt_silence()
    rule = sweep.Rule("cusum", 0.5, 0.35, 450)
    assert sweep.Reference(rule).run(probabilities, rms) == [
        (5 * FRAME_SAMPLES, 23 * FRAME_SAMPLES)
    ]


def test_band_freezes_doubtful_frames_like_the_hysteresis_clock() -> None:
    probabilities, rms = _speech_doubt_silence()
    expected = [(5 * FRAME_SAMPLES, 23 * FRAME_SAMPLES)]
    assert sweep.Reference(sweep.Rule("band", 0.5, 0.35, 450)).run(probabilities, rms) == expected
    hysteresis = sweep.Rule("hysteresis", 0.5, 0.35, 450)
    assert sweep.Reference(hysteresis).run(probabilities, rms) == expected
