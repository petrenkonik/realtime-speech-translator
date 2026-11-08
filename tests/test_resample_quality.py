"""Q5-Q7: aliasing, passband flatness and dynamics of the streaming resampler."""

from __future__ import annotations

import numpy as np
import pytest
from signals import rms, sine, to_db, tone_amplitude

from pipeline.resample import StreamingResampler

pytestmark = pytest.mark.unit

OUT_RATE_HZ = 16_000
SETTLE = 2_000  # output samples dropped to skip the filter start-up transient
RATES_HZ = [32_000, 44_100, 48_000]


def _alias_cases() -> list[tuple[int, int]]:
    freqs = [8_100, 9_000, 10_000, 12_000, 15_000]
    return [(rate, f) for rate in RATES_HZ for f in freqs if f < rate / 2 - 200]


def test_q5_naive_decimation_aliases_so_the_test_can_fail() -> None:
    # Control: without a low-pass the 10 kHz tone folds to 6 kHz at full level.
    folded = sine(10_000, 48_000)[::3][SETTLE:]

    assert to_db(float(np.max(np.abs(folded))) / 0.5) > -1.0


@pytest.mark.parametrize(("rate_hz", "freq_hz"), _alias_cases())
def test_q5_tones_above_8_khz_are_rejected_by_60_db(rate_hz: int, freq_hz: int) -> None:
    out = StreamingResampler(rate_hz).process(sine(freq_hz, rate_hz))[SETTLE:]

    assert to_db(float(np.max(np.abs(out))) / 0.5) <= -60.0


@pytest.mark.parametrize("rate_hz", RATES_HZ)
def test_q5_band_limited_noise_above_8_khz_is_rejected_by_60_db(rate_hz: int) -> None:
    rng = np.random.default_rng(rate_hz)
    noise = rng.standard_normal(rate_hz * 2)
    spectrum = np.fft.rfft(noise)
    spectrum[np.fft.rfftfreq(len(noise), 1 / rate_hz) < 8_500] = 0  # keep only 8.5 kHz and up
    high_band = np.fft.irfft(spectrum, len(noise)).astype(np.float32)

    out = StreamingResampler(rate_hz).process(high_band)[SETTLE:]

    assert to_db(rms(out) / rms(high_band)) <= -60.0


@pytest.mark.parametrize("rate_hz", RATES_HZ)
@pytest.mark.parametrize("freq_hz", [100, 1_000, 3_000, 5_000, 7_000])
def test_q6_passband_is_flat_within_a_tenth_of_a_decibel(rate_hz: int, freq_hz: int) -> None:
    out = StreamingResampler(rate_hz).process(sine(freq_hz, rate_hz, seconds=2.0))[4_000:]

    level_db = to_db(tone_amplitude(out[:16_000], freq_hz, OUT_RATE_HZ) / 0.5)
    assert level_db == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("rate_hz", RATES_HZ)
def test_q7_distortion_and_noise_stay_80_db_below_the_tone(rate_hz: int) -> None:
    out = StreamingResampler(rate_hz).process(sine(1_000, rate_hz, seconds=2.0))[4_000:20_000]
    spectrum = np.abs(np.fft.rfft(out.astype(np.float64) * np.hanning(len(out)))) ** 2
    peak = int(np.argmax(spectrum))
    tone_power = spectrum[peak - 4 : peak + 5].sum()
    residual_power = spectrum.sum() - tone_power - spectrum[:3].sum()

    assert 10 * np.log10(residual_power / tone_power) <= -80.0


@pytest.mark.parametrize("rate_hz", RATES_HZ)
def test_q7_a_very_quiet_tone_keeps_its_level(rate_hz: int) -> None:
    amplitude = 10 ** (-90 / 20)  # -90 dBFS: about 1 LSB of 16-bit audio
    out = StreamingResampler(rate_hz).process(sine(1_000, rate_hz, 2.0, amplitude))[4_000:]

    assert to_db(np.sqrt(2) * rms(out) / amplitude) == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("rate_hz", RATES_HZ)
def test_q7_a_quiet_detail_next_to_a_loud_signal_is_preserved(rate_hz: int) -> None:
    quiet = 10 ** (-80 / 20)
    mix = sine(300, rate_hz, 2.0, 0.5) + sine(2_500, rate_hz, 2.0, quiet)

    out = StreamingResampler(rate_hz).process(mix)[4_000:20_000]

    assert to_db(tone_amplitude(out, 2_500, OUT_RATE_HZ) / quiet) == pytest.approx(0.0, abs=0.5)


@pytest.mark.parametrize("rate_hz", RATES_HZ)
def test_q7_a_full_scale_tone_does_not_overshoot_once_settled(rate_hz: int) -> None:
    out = StreamingResampler(rate_hz).process(sine(7_000, rate_hz, 1.0, 1.0))[SETTLE:]

    assert float(np.max(np.abs(out))) <= 1.001
