"""Five fixed clips for the Smart Turn feature parity test (M4).

The clips are generated, not recorded, so the repository holds only the reference
output. ``tests/data/smart_turn_reference.npz`` was produced once by the vendor
pipeline (``truncate_audio_to_last_n_seconds`` + ``WhisperFeatureExtractor(chunk_length=8)``
with ``do_normalize=True``, see ``scripts/make_turn_reference.py``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

RATE_HZ = 16_000


def _time(seconds: float) -> NDArray[np.float64]:
    return np.arange(int(seconds * RATE_HZ)) / RATE_HZ


def _voiced(seconds: float, f0: float, seed: int) -> NDArray[np.float32]:
    t = _time(seconds)
    rng = np.random.default_rng(seed)
    signal = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (0.2 * envelope * signal + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def reference_clips() -> dict[str, NDArray[np.float32]]:
    """Name -> 16 kHz mono float32 clip."""
    rng = np.random.default_rng(7)
    t_chirp = _time(3.5)
    chirp = 0.3 * np.sin(2 * np.pi * (100 * t_chirp + 0.5 * (2900 / 3.5) * t_chirp**2))
    gappy = np.concatenate(
        [
            _voiced(3.0, 140, 1),
            np.zeros(RATE_HZ, np.float32),
            _voiced(5.0, 210, 2),
            _voiced(2.0, 95, 3),
        ]
    )
    return {
        "short_noise": (0.05 * rng.standard_normal(RATE_HZ // 2)).astype(np.float32),
        "voiced_2s": _voiced(2.0, 120, 4),
        "chirp_3_5s": (chirp + 0.02 * rng.standard_normal(len(t_chirp))).astype(np.float32),
        "exactly_8s": _voiced(8.0, 180, 5),
        "gappy_11s": gappy.astype(np.float32),
    }
