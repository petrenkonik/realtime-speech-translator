"""Tests for RateConverter: speech from the TTS rate to the rate of an output device."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pipeline.resample import RateConverter, StreamingResampler, UnsupportedSampleRateError

pytestmark = pytest.mark.unit

# SAPI voices give 16000, 22050 or 24000 Hz; devices run at 44100 or 48000 Hz.
PAIRS_HZ = [(22_050, 48_000), (22_050, 44_100), (24_000, 48_000), (16_000, 48_000)]


def _sine(
    freq_hz: float, rate_hz: int, seconds: float, amplitude: float = 0.5
) -> NDArray[np.float32]:
    t = np.arange(round(rate_hz * seconds)) / rate_hz
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _tone_db(signal: NDArray[np.float32], rate_hz: int, freq_hz: float, ref: float) -> float:
    """Level of one frequency, in dB relative to ``ref``, from a Hann-windowed DFT bin."""
    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window)) * 2 / window.sum()
    bin_index = round(freq_hz * len(signal) / rate_hz)
    return float(20 * np.log10(max(spectrum[bin_index - 2 : bin_index + 3].max(), 1e-12) / ref))


@pytest.mark.parametrize(("in_hz", "out_hz"), PAIRS_HZ)
def test_the_output_length_is_the_ceiling_of_the_ratio(in_hz: int, out_hz: int) -> None:
    signal = _sine(440, in_hz, seconds=0.8)

    out = RateConverter(in_hz, out_hz).process(signal)

    assert out.dtype == np.float32
    assert len(out) == -(-len(signal) * out_hz // in_hz)


@pytest.mark.parametrize(("in_hz", "out_hz"), PAIRS_HZ)
def test_the_result_does_not_depend_on_the_block_split(in_hz: int, out_hz: int) -> None:
    rng = np.random.default_rng(in_hz + out_hz)
    signal = rng.uniform(-0.8, 0.8, in_hz).astype(np.float32)
    one_shot = RateConverter(in_hz, out_hz).process(signal)

    converter = RateConverter(in_hz, out_hz)
    parts, pos = [], 0
    while pos < len(signal):
        size = int(rng.integers(0, 1500))
        parts.append(converter.process(signal[pos : pos + size]))
        pos += size

    streamed = np.concatenate(parts)
    assert streamed.shape == one_shot.shape
    assert np.max(np.abs(streamed - one_shot)) < 1e-5


@pytest.mark.parametrize(("in_hz", "out_hz"), PAIRS_HZ)
@pytest.mark.parametrize("freq_hz", [300, 1_000, 4_000])
def test_a_tone_keeps_its_frequency_and_level(in_hz: int, out_hz: int, freq_hz: int) -> None:
    out = RateConverter(in_hz, out_hz).process(_sine(freq_hz, in_hz, seconds=1.0))[out_hz // 5 :]

    assert _tone_db(out, out_hz, freq_hz, ref=0.5) == pytest.approx(0.0, abs=0.5)


@pytest.mark.parametrize(("in_hz", "out_hz"), [(22_050, 48_000), (24_000, 48_000)])
def test_the_upsampling_image_is_rejected_by_60db(in_hz: int, out_hz: int) -> None:
    # A 1 kHz tone sampled at in_hz has a mirror image at in_hz - 1 kHz after upsampling.
    out = RateConverter(in_hz, out_hz).process(_sine(1_000, in_hz, seconds=1.0))[out_hz // 5 :]

    assert _tone_db(out, out_hz, in_hz - 1_000, ref=0.5) <= -60.0


@pytest.mark.parametrize(("in_hz", "out_hz"), PAIRS_HZ)
def test_the_group_delay_is_at_most_3_ms(in_hz: int, out_hz: int) -> None:
    impulse = np.zeros(in_hz // 10, dtype=np.float32)
    impulse[0] = 1.0

    out = RateConverter(in_hz, out_hz).process(impulse)

    assert np.argmax(np.abs(out)) * 1000 / out_hz <= 3.0


def test_equal_rates_pass_through_unchanged() -> None:
    signal = _sine(440, 22_050, seconds=0.1)

    np.testing.assert_array_equal(RateConverter(22_050, 22_050).process(signal), signal)


def test_downsampling_is_available_too() -> None:
    out = RateConverter(48_000, 22_050).process(_sine(1_000, 48_000, seconds=2.0))
    settled = out[2_000 : 2_000 + 22_050]  # whole seconds, so the tone sits on a DFT bin

    assert _tone_db(settled, 22_050, 1_000, ref=0.5) == pytest.approx(0.0, abs=0.5)


@pytest.mark.parametrize(("in_hz", "out_hz"), [(0, 48_000), (7_999, 48_000), (22_050, 200_000)])
def test_a_rate_outside_8_to_192_khz_is_a_clear_error(in_hz: int, out_hz: int) -> None:
    with pytest.raises(UnsupportedSampleRateError, match="8000"):
        RateConverter(in_hz, out_hz)


def test_the_capture_resampler_stays_strict() -> None:
    # Capture only knows four input rates and a 16 kHz output; the speech path must not
    # loosen that: an odd microphone rate is still reported, not guessed.
    with pytest.raises(UnsupportedSampleRateError):
        StreamingResampler(22_050)
