"""Synthetic signals and level measurements shared by the audio-quality tests."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]


class Mixer(Protocol):
    """Anything with a ``process(block) -> mono`` method."""

    def process(self, block: Float32Array) -> Float32Array:
        """Process one block."""
        ...


def sine(
    freq_hz: float, rate_hz: int, seconds: float = 1.0, amplitude: float = 0.5
) -> Float32Array:
    """Return a sine wave starting at phase zero."""
    t = np.arange(round(rate_hz * seconds)) / rate_hz
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def rms(samples: NDArray[np.floating]) -> float:
    """Root-mean-square level, computed in float64."""
    return float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))


def to_db(ratio: float) -> float:
    """Convert an amplitude ratio to decibels, with a floor to avoid log(0)."""
    return 20 * float(np.log10(max(ratio, 1e-20)))


def tone_amplitude(samples: NDArray[np.floating], freq_hz: float, rate_hz: int) -> float:
    """Estimate the amplitude of one tone with a Hann-windowed FFT."""
    window = np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(np.asarray(samples, dtype=np.float64) * window))
    centre = round(freq_hz * len(samples) / rate_hz)
    return float(spectrum[centre - 2 : centre + 3].max() * 2 / window.sum())


def feed_blocks(mixer: Mixer, block: Float32Array, block_frames: int) -> Float32Array:
    """Push a signal through ``mixer.process`` in fixed-size blocks."""
    parts = [mixer.process(block[i : i + block_frames]) for i in range(0, len(block), block_frames)]
    return np.concatenate(parts)
