"""Tests for StreamingResampler (U3-U6)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pipeline.resample import StreamingResampler, UnsupportedSampleRateError

pytestmark = pytest.mark.unit

OUT_RATE_HZ = 16_000
SUPPORTED_RATES_HZ = [16_000, 32_000, 44_100, 48_000]


def _sine(
    freq_hz: float, rate_hz: int, seconds: float, amplitude: float = 0.5
) -> NDArray[np.float32]:
    t = np.arange(round(rate_hz * seconds)) / rate_hz
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _stream(
    resampler: StreamingResampler, signal: NDArray[np.float32], block_sizes: list[int]
) -> NDArray[np.float32]:
    parts = []
    pos = 0
    for size in block_sizes:
        parts.append(resampler.process(signal[pos : pos + size]))
        pos += size
    assert pos == len(signal)
    return np.concatenate(parts)


def _random_blocks(total: int, rng: np.random.Generator) -> list[int]:
    sizes = []
    left = total
    while left > 0:
        size = int(min(left, rng.integers(0, 2000)))
        sizes.append(size)
        left -= size
    return sizes


@pytest.mark.parametrize("frames", [48_000, 48_003, 3 * 12_345])
def test_u3_48k_to_16k_length_is_exact(frames: int) -> None:
    rng = np.random.default_rng(1)
    signal = rng.uniform(-0.5, 0.5, frames).astype(np.float32)

    out = _stream(StreamingResampler(48_000), signal, _random_blocks(frames, rng))

    assert len(out) == -(-frames // 3)  # ceil: exactly N/3 when N is a multiple of 3


def test_u3_44k1_to_16k_length_within_one_sample() -> None:
    rng = np.random.default_rng(2)
    frames = 44_100 * 3 + 17
    signal = rng.uniform(-0.5, 0.5, frames).astype(np.float32)

    out = _stream(StreamingResampler(44_100), signal, _random_blocks(frames, rng))

    assert abs(len(out) - frames * OUT_RATE_HZ / 44_100) <= 1


@pytest.mark.parametrize("rate_hz", SUPPORTED_RATES_HZ)
def test_u4_block_size_invariance(rate_hz: int) -> None:
    rng = np.random.default_rng(rate_hz)
    signal = rng.uniform(-0.8, 0.8, rate_hz * 2).astype(np.float32)

    one_shot = StreamingResampler(rate_hz).process(signal)
    streamed = _stream(StreamingResampler(rate_hz), signal, _random_blocks(len(signal), rng))

    assert one_shot.dtype == np.float32
    assert streamed.shape == one_shot.shape
    assert np.max(np.abs(streamed - one_shot)) < 1e-5


@pytest.mark.parametrize("rate_hz", SUPPORTED_RATES_HZ)
def test_u5_1khz_sine_keeps_its_frequency(rate_hz: int) -> None:
    out = StreamingResampler(rate_hz).process(_sine(1_000, rate_hz, seconds=1.0))

    spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    freqs = np.fft.rfftfreq(len(out), d=1 / OUT_RATE_HZ)
    bin_hz = freqs[1]
    assert abs(freqs[np.argmax(spectrum)] - 1_000) <= bin_hz


@pytest.mark.parametrize("rate_hz", [32_000, 44_100, 48_000])
def test_u6_10khz_tone_is_rejected_by_60db(rate_hz: int) -> None:
    amplitude = 0.5
    tone = _sine(10_000, rate_hz, seconds=1.0, amplitude=amplitude)
    out = StreamingResampler(rate_hz).process(tone)

    settled = out[OUT_RATE_HZ // 10 :]  # skip the filter start-up transient
    peak_db = 20 * np.log10(np.max(np.abs(settled)) / amplitude)
    assert peak_db <= -60


def test_passband_7khz_is_kept() -> None:
    amplitude = 0.5
    tone = _sine(7_000, 48_000, seconds=1.0, amplitude=amplitude)
    out = StreamingResampler(48_000).process(tone)

    settled = out[OUT_RATE_HZ // 10 :]
    assert np.max(np.abs(settled)) == pytest.approx(amplitude, rel=0.02)


@pytest.mark.parametrize("rate_hz", [32_000, 44_100, 48_000])
def test_u27_group_delay_is_at_most_3_ms(rate_hz: int) -> None:
    impulse = np.zeros(rate_hz // 10, dtype=np.float32)
    impulse[0] = 1.0

    out = StreamingResampler(rate_hz).process(impulse)

    delay_ms = np.argmax(np.abs(out)) * 1000 / OUT_RATE_HZ
    assert delay_ms <= 3.0


def test_16k_passes_through_unchanged() -> None:
    signal = _sine(440, OUT_RATE_HZ, seconds=0.1)

    np.testing.assert_array_equal(StreamingResampler(OUT_RATE_HZ).process(signal), signal)


def test_stereo_input_is_mixed_to_mono() -> None:
    signal = _sine(1_000, 48_000, seconds=0.2)
    stereo = np.stack([signal, signal], axis=1)

    mono_out = StreamingResampler(48_000).process(signal)
    stereo_out = StreamingResampler(48_000).process(stereo)

    np.testing.assert_allclose(stereo_out, mono_out, atol=1e-7)


@pytest.mark.parametrize("rate_hz", [8_000, 22_050, 96_000])
def test_unsupported_rate_is_a_clear_error(rate_hz: int) -> None:
    with pytest.raises(UnsupportedSampleRateError, match=f"{rate_hz} Hz is not supported"):
        StreamingResampler(rate_hz)


def test_empty_block_yields_empty_output_and_keeps_state() -> None:
    resampler = StreamingResampler(48_000)
    signal = _sine(1_000, 48_000, seconds=0.1)

    first = resampler.process(signal[:2000])
    empty = resampler.process(np.zeros(0, dtype=np.float32))
    second = resampler.process(signal[2000:])

    assert len(empty) == 0
    one_shot = StreamingResampler(48_000).process(signal)
    np.testing.assert_array_equal(np.concatenate([first, second]), one_shot)
