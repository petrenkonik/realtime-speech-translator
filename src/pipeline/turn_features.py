"""Log-mel features for the Smart Turn model, in plain NumPy.

The model was trained on the output of ``transformers.WhisperFeatureExtractor`` (8 s
window), so this module reproduces that pipeline exactly instead of importing it
(``transformers`` and ``torch`` are heavy and not allowed as dependencies):

1. keep the last 8 s of the audio, or pad zeros at the *front* up to 8 s;
2. normalize the whole 8 s vector to zero mean and unit variance (the vendor passes
   ``do_normalize=True``; the padding is part of the vector);
3. STFT with a periodic Hann window, ``n_fft`` 400, hop 160, reflect padding;
4. power spectrum -> 80 Slaney mel bins -> ``log10(max(x, 1e-10))``;
5. clamp to ``max - 8``, scale ``(x + 4) / 4``, drop the last frame.

Parity with the vendor extractor is checked by test M4 against stored reference output.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE_HZ = 16_000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE_HZ  # 128 000
N_FFT = 400
HOP_SAMPLES = 160
N_MELS = 80
N_FRAMES = WINDOW_SAMPLES // HOP_SAMPLES  # 800
_NORMALIZE_EPS = 1e-7

Float32Array = NDArray[np.float32]

# Slaney mel scale: linear below 1 kHz, logarithmic above.
_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOG_STEP = np.log(6.4) / 27.0


def _hz_to_mel(hz: NDArray[np.float64]) -> NDArray[np.float64]:
    mel = hz / _F_SP
    above = hz >= _MIN_LOG_HZ
    mel[above] = _MIN_LOG_MEL + np.log(hz[above] / _MIN_LOG_HZ) / _LOG_STEP
    return mel


def _mel_to_hz(mel: NDArray[np.float64]) -> NDArray[np.float64]:
    hz = mel * _F_SP
    above = mel >= _MIN_LOG_MEL
    hz[above] = _MIN_LOG_HZ * np.exp(_LOG_STEP * (mel[above] - _MIN_LOG_MEL))
    return hz


@lru_cache(maxsize=1)
def _mel_filters() -> NDArray[np.float64]:
    """Triangular filters, shape ``(N_FFT // 2 + 1, N_MELS)``, area-normalized (Slaney)."""
    fft_freqs = np.linspace(0.0, SAMPLE_RATE_HZ / 2, N_FFT // 2 + 1)
    mel_edges = np.linspace(
        _hz_to_mel(np.array([0.0]))[0],
        _hz_to_mel(np.array([SAMPLE_RATE_HZ / 2]))[0],
        N_MELS + 2,
    )
    edges = _mel_to_hz(mel_edges)
    widths = np.diff(edges)
    ramps = edges[np.newaxis, :] - fft_freqs[:, np.newaxis]
    down = -ramps[:, :-2] / widths[:-1]
    up = ramps[:, 2:] / widths[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    filters *= (2.0 / (edges[2 : N_MELS + 2] - edges[:N_MELS]))[np.newaxis, :]
    return filters


@lru_cache(maxsize=1)
def _window() -> NDArray[np.float64]:
    """Periodic Hann window."""
    return np.hanning(N_FFT + 1)[:-1]


def fit_window(audio: Float32Array) -> Float32Array:
    """Return exactly ``WINDOW_SAMPLES`` samples: the last 8 s, zero-padded at the front."""
    if len(audio) >= WINDOW_SAMPLES:
        return audio[-WINDOW_SAMPLES:].astype(np.float32, copy=False)
    padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    padded[WINDOW_SAMPLES - len(audio) :] = audio
    return padded


def turn_features(audio: Float32Array) -> Float32Array:
    """Compute the model input for the end of ``audio`` (16 kHz mono float32).

    Returns:
        Array of shape ``(80, 800)``, float32.
    """
    signal = fit_window(audio).astype(np.float64)
    signal = (signal - signal.mean()) / np.sqrt(signal.var() + _NORMALIZE_EPS)

    padded = np.pad(signal, N_FFT // 2, mode="reflect")
    starts = np.arange(N_FRAMES + 1) * HOP_SAMPLES
    frames = padded[starts[:, np.newaxis] + np.arange(N_FFT)[np.newaxis, :]]
    spectrum = np.fft.rfft(frames * _window(), axis=1)
    power = spectrum.real**2 + spectrum.imag**2  # (frames, bins)

    mel = np.maximum(1e-10, power @ _mel_filters())  # (frames, mels)
    log_mel = np.log10(mel).T[:, :N_FRAMES]  # drop the last frame
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    scaled: Float32Array = ((log_mel + 4.0) / 4.0).astype(np.float32)
    return scaled
