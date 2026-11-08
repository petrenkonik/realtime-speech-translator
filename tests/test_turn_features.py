"""U23: the Smart Turn input features have the right shape, padding and window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from turn_reference import reference_clips

from pipeline.turn_features import (
    N_FRAMES,
    N_MELS,
    WINDOW_SAMPLES,
    fit_window,
    turn_features,
)

pytestmark = pytest.mark.unit


def noise(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * 16_000)) * 0.1).astype(np.float32)


def test_u23_the_feature_shape_and_type_are_fixed() -> None:
    for seconds in (0.2, 3.0, 8.0, 12.0):
        features = turn_features(noise(seconds))
        assert features.shape == (N_MELS, N_FRAMES) == (80, 800)
        assert features.dtype == np.float32
        assert np.isfinite(features).all()


def test_u23_short_audio_is_padded_with_zeros_at_the_front() -> None:
    audio = noise(3.0)
    window = fit_window(audio)
    assert len(window) == WINDOW_SAMPLES
    split = WINDOW_SAMPLES - len(audio)
    assert not window[:split].any()
    np.testing.assert_array_equal(window[split:], audio)


def test_u23_long_audio_keeps_the_last_eight_seconds() -> None:
    audio = noise(12.0)
    np.testing.assert_array_equal(fit_window(audio), audio[-WINDOW_SAMPLES:])
    # Whatever came before the last 8 s has no influence.
    changed = audio.copy()
    changed[: len(audio) - WINDOW_SAMPLES] += 5.0
    np.testing.assert_array_equal(turn_features(audio), turn_features(changed))


def test_u23_padding_shows_up_as_a_flat_start_and_speech_at_the_end() -> None:
    features = turn_features(noise(2.0))
    start, end = features[:, :100], features[:, -100:]
    assert float(start.std(axis=1).mean()) < 0.05  # constant floor
    assert float(end.std(axis=1).mean()) > float(start.std(axis=1).mean())


def test_features_are_deterministic_and_do_not_modify_the_input() -> None:
    audio = noise(4.0)
    before = audio.copy()
    first = turn_features(audio)
    np.testing.assert_array_equal(first, turn_features(audio))
    np.testing.assert_array_equal(audio, before)


def test_scaling_the_signal_does_not_change_the_features() -> None:
    # The waveform is normalized to unit variance before the mel transform.
    audio = noise(5.0)
    np.testing.assert_allclose(turn_features(audio), turn_features(audio * 4.0), atol=1e-3)


def test_m4_features_match_the_vendor_reference() -> None:
    """Parity with ``WhisperFeatureExtractor(chunk_length=8, do_normalize=True)``.

    The reference file comes from ``scripts/make_turn_reference.py``; the spec allows 1e-3.
    """
    reference = np.load(Path(__file__).parent / "data" / "smart_turn_reference.npz")
    clips = reference_clips()
    assert set(reference.files) == set(clips)
    for name, clip in clips.items():
        difference = np.abs(turn_features(clip) - reference[name]).max()
        assert difference < 1e-3, f"{name}: max abs difference {difference}"
