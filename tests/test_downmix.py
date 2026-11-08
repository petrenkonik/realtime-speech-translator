"""Q1-Q4: stereo to mono must keep the level when one channel is dead and when both play."""

from __future__ import annotations

import numpy as np
import pytest
from signals import feed_blocks, rms, sine, to_db, tone_amplitude

from pipeline.pcm import ChannelMixer, to_mono

pytestmark = pytest.mark.unit

RATE_HZ = 48_000
BLOCK_FRAMES = 480  # 10 ms, as captured


def _stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([left, right], axis=1)


def _level_error_db(out: np.ndarray, reference: np.ndarray) -> float:
    return to_db(rms(out) / rms(reference))


def test_q1_plain_mean_loses_6_db_with_a_dead_channel() -> None:
    # Documents the problem the mixer exists to solve.
    left = sine(1_000, RATE_HZ)
    silent = np.zeros_like(left)

    out = to_mono(_stereo(left, silent))

    assert _level_error_db(out, left) == pytest.approx(-6.02, abs=0.05)


@pytest.mark.parametrize("dead_side", ["right", "left"])
def test_q1_auto_keeps_level_when_one_channel_is_silent(dead_side: str) -> None:
    live = sine(1_000, RATE_HZ)
    silent = np.zeros_like(live)
    block = _stereo(live, silent) if dead_side == "right" else _stereo(silent, live)

    out = feed_blocks(ChannelMixer(RATE_HZ), block, BLOCK_FRAMES)

    assert abs(_level_error_db(out, live)) < 0.05
    np.testing.assert_allclose(out, live, atol=1e-6)


def test_q1_auto_ignores_a_hissing_dead_channel() -> None:
    rng = np.random.default_rng(3)
    live = sine(1_000, RATE_HZ)
    hiss = (rng.standard_normal(len(live)) * 10 ** (-90 / 20)).astype(np.float32)

    out = feed_blocks(ChannelMixer(RATE_HZ), _stereo(live, hiss), BLOCK_FRAMES)

    assert abs(_level_error_db(out, live)) < 0.1


def test_q1_auto_ignores_a_dead_channel_with_dc_offset() -> None:
    live = sine(1_000, RATE_HZ)
    stuck = np.full_like(live, 0.2)

    out = feed_blocks(ChannelMixer(RATE_HZ), _stereo(live, stuck), BLOCK_FRAMES)

    assert abs(_level_error_db(out - np.mean(out), live)) < 0.1


def test_q2_identical_channels_pass_through_unchanged() -> None:
    signal = sine(1_000, RATE_HZ)

    out = feed_blocks(ChannelMixer(RATE_HZ), _stereo(signal, signal), BLOCK_FRAMES)

    np.testing.assert_allclose(out, signal, atol=1e-6)


def test_q2_independent_tones_in_both_channels_are_both_kept_without_clipping() -> None:
    left = sine(1_000, RATE_HZ, amplitude=1.0)
    right = sine(2_000, RATE_HZ, amplitude=1.0)

    out = feed_blocks(ChannelMixer(RATE_HZ), _stereo(left, right), BLOCK_FRAMES)

    assert np.max(np.abs(out)) <= 1.0
    for freq_hz in (1_000, 2_000):
        assert to_db(tone_amplitude(out, freq_hz, RATE_HZ)) == pytest.approx(-6.0, abs=1.0)


def test_q2_a_quieter_second_channel_does_not_drag_the_level_down() -> None:
    left = sine(1_000, RATE_HZ, amplitude=0.5)
    right = sine(1_000, RATE_HZ, amplitude=0.05)  # 20 dB lower, same phase

    out = feed_blocks(ChannelMixer(RATE_HZ), _stereo(left, right), BLOCK_FRAMES)

    assert _level_error_db(out, left) > -1.0


def test_q3_antiphase_channels_cancel_in_mean_mode_but_not_in_auto() -> None:
    signal = sine(1_000, RATE_HZ)
    block = _stereo(signal, -signal)

    mean_out = feed_blocks(ChannelMixer(RATE_HZ, mode="mean"), block, BLOCK_FRAMES)
    mixer = ChannelMixer(RATE_HZ)
    auto_out = feed_blocks(mixer, block, BLOCK_FRAMES)

    assert np.all(mean_out == 0.0)
    assert abs(_level_error_db(auto_out, signal)) < 0.1
    assert mixer.polarity_inverted


def test_q3_uncorrelated_channels_are_not_polarity_flipped() -> None:
    rng = np.random.default_rng(5)
    noise = rng.standard_normal((RATE_HZ, 2)).astype(np.float32) * 0.1
    mixer = ChannelMixer(RATE_HZ)

    feed_blocks(mixer, noise, BLOCK_FRAMES)

    assert not mixer.polarity_inverted


def test_q4_a_channel_that_switches_on_midway_causes_no_click() -> None:
    left = sine(1_000, RATE_HZ, seconds=2.0)
    right = np.roll(left, 12)  # 90 degrees apart, so the mix really changes
    right[: RATE_HZ // 2] = 0.0
    pure_sine_step = float(np.max(np.abs(np.diff(left))))

    out = feed_blocks(ChannelMixer(RATE_HZ), _stereo(left, right), BLOCK_FRAMES)

    assert float(np.max(np.abs(np.diff(out)))) <= 1.5 * pure_sine_step


def test_q4_polarity_flip_midway_causes_no_click() -> None:
    left = sine(1_000, RATE_HZ, seconds=2.0)
    right = left.copy()
    right[RATE_HZ:] *= -1  # cable swapped halfway
    pure_sine_step = float(np.max(np.abs(np.diff(left))))

    out = feed_blocks(ChannelMixer(RATE_HZ, tau_ms=50), _stereo(left, right), BLOCK_FRAMES)

    assert float(np.max(np.abs(np.diff(out)))) <= 2.0 * pure_sine_step


def test_three_identical_channels_pass_through_unchanged() -> None:
    signal = sine(1_000, RATE_HZ, seconds=0.2)

    out = feed_blocks(ChannelMixer(RATE_HZ), np.stack([signal] * 3, axis=1), BLOCK_FRAMES)

    np.testing.assert_allclose(out, signal, atol=1e-6)


def test_digital_silence_stays_silent() -> None:
    out = feed_blocks(ChannelMixer(RATE_HZ), np.zeros((4_800, 2), dtype=np.float32), BLOCK_FRAMES)

    assert np.all(out == 0.0)


def test_mono_blocks_pass_through_and_empty_blocks_are_handled() -> None:
    mixer = ChannelMixer(RATE_HZ)
    mono = sine(440, RATE_HZ, seconds=0.01)

    np.testing.assert_array_equal(mixer.process(mono), mono)
    np.testing.assert_array_equal(mixer.process(mono[:, np.newaxis]), mono)
    assert mixer.process(np.zeros((0, 2), dtype=np.float32)).shape == (0,)


def test_channel_count_change_resets_the_state() -> None:
    mixer = ChannelMixer(RATE_HZ)
    stereo = _stereo(sine(1_000, RATE_HZ, seconds=0.1), np.zeros(4_800, dtype=np.float32))
    triple = np.stack([sine(1_000, RATE_HZ, seconds=0.1)] * 3, axis=1)

    mixer.process(stereo)
    out = mixer.process(triple)

    np.testing.assert_allclose(out, triple[:, 0], atol=1e-6)


@pytest.mark.parametrize(("rate_hz", "tau_ms"), [(0, 300.0), (48_000, 0.0)])
def test_invalid_arguments_are_rejected(rate_hz: int, tau_ms: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ChannelMixer(rate_hz, tau_ms=tau_ms)


def test_three_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        ChannelMixer(RATE_HZ).process(np.zeros((2, 2, 2), dtype=np.float32))
