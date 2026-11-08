"""Q8-Q11: quantization to int16 must not turn small detail into distortion."""

from __future__ import annotations

import numpy as np
import pytest
from signals import sine, tone_amplitude

from pipeline.pcm import INT16_FULL_SCALE, float_to_int16, unpack_int24

pytestmark = pytest.mark.unit

RATE_HZ = 16_000
LSB = 1.0 / INT16_FULL_SCALE


def _peak_to_median_db(samples: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2
    spectrum = spectrum[2:]  # ignore DC
    return 10 * float(np.log10(spectrum.max() / np.median(spectrum)))


def _error_in_lsb(signal: np.ndarray, quantized: np.ndarray) -> np.ndarray:
    return quantized.astype(np.float64) - signal.astype(np.float64) * INT16_FULL_SCALE


def test_q8_extremes_and_zero_are_exact_without_dither() -> None:
    out = float_to_int16(np.array([0.0, 1.0, -1.0], dtype=np.float32), dither=False)

    np.testing.assert_array_equal(out, [0, 32767, -32767])
    assert out.dtype == np.int16


def test_q8_out_of_range_input_saturates_instead_of_wrapping() -> None:
    out = float_to_int16(np.array([1.5, -1.5, 40.0], dtype=np.float32), dither=False)

    np.testing.assert_array_equal(out, [32767, -32768, 32767])


def test_q8_undithered_rounding_loses_a_sub_lsb_dc_level() -> None:
    out = float_to_int16(np.full(100_000, 0.3 * LSB, dtype=np.float32), dither=False)

    assert out.mean() == 0.0


def test_q8_dither_preserves_a_sub_lsb_dc_level_on_average() -> None:
    rng = np.random.default_rng(1)

    out = float_to_int16(np.full(200_000, 0.3 * LSB, dtype=np.float32), rng=rng)

    assert out.mean() == pytest.approx(0.3, abs=0.01)


def test_q8_a_tone_below_one_lsb_vanishes_without_dither_and_survives_with_it() -> None:
    tone = sine(1_000, RATE_HZ, amplitude=0.4 * LSB)

    plain = float_to_int16(tone, dither=False)
    dithered = float_to_int16(tone, rng=np.random.default_rng(2))

    assert not plain.any()
    assert _peak_to_median_db(dithered.astype(np.float64)) > 15.0
    # The recovered amplitude is the input amplitude, in LSB.
    assert tone_amplitude(dithered, 1_000, RATE_HZ) == pytest.approx(0.4, rel=0.15)


def test_q9_dither_turns_harmonic_distortion_into_flat_noise() -> None:
    tone = sine(1_000, RATE_HZ, amplitude=10.3 * LSB)  # periodic error without dither

    plain_error = _error_in_lsb(tone, float_to_int16(tone, dither=False))
    dither_error = _error_in_lsb(tone, float_to_int16(tone, rng=np.random.default_rng(4)))

    assert _peak_to_median_db(plain_error) > 30.0  # distinct harmonic lines
    assert _peak_to_median_db(dither_error) < 15.0  # no lines, just a floor


def test_q10_error_power_matches_theory() -> None:
    rng = np.random.default_rng(6)
    signal = rng.uniform(-0.5, 0.5, 400_000).astype(np.float32)

    plain = _error_in_lsb(signal, float_to_int16(signal, dither=False))
    dithered = _error_in_lsb(signal, float_to_int16(signal, rng=np.random.default_rng(7)))

    assert np.var(plain) == pytest.approx(1 / 12, rel=0.05)
    assert np.var(dithered) == pytest.approx(1 / 4, rel=0.05)  # 1/12 rounding + 1/6 dither


def test_q10_dither_is_unbiased_and_decorrelated_from_the_signal() -> None:
    rng = np.random.default_rng(8)
    signal = rng.uniform(-0.5, 0.5, 400_000).astype(np.float32)

    error = _error_in_lsb(signal, float_to_int16(signal, rng=np.random.default_rng(9)))

    assert abs(error.mean()) < 0.01
    assert abs(np.corrcoef(error, signal)[0, 1]) < 0.01


def test_q10_a_seeded_generator_makes_the_output_reproducible() -> None:
    signal = sine(440, RATE_HZ, seconds=0.1)

    first = float_to_int16(signal, rng=np.random.default_rng(11))
    second = float_to_int16(signal, rng=np.random.default_rng(11))

    np.testing.assert_array_equal(first, second)


def test_q10_digital_silence_gets_no_dither_noise() -> None:
    out = float_to_int16(np.zeros(4_800, dtype=np.float32))

    assert not out.any()


def test_q11_int24_to_float32_keeps_every_bit_within_one_step() -> None:
    rng = np.random.default_rng(12)
    values = rng.integers(-0x800000, 0x800000, 20_000)
    raw = b"".join(int(v & 0xFFFFFF).to_bytes(3, "little") for v in values)

    recovered = np.rint(unpack_int24(raw, channels=1)[:, 0].astype(np.float64) * 0x7FFFFF)

    # float32 has a 24-bit mantissa, so 24-bit audio survives; no dither is needed there.
    assert np.max(np.abs(recovered - np.maximum(values, -0x7FFFFF))) <= 1
