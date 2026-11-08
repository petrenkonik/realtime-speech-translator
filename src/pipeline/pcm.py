"""PCM conversion helpers: 24-bit integer unpacking and channel mixing."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

_BYTES_PER_SAMPLE = 3
_INT24_FULL_SCALE = 0x7FFFFF
_SIGN_BIT = 0x800000


def unpack_int24(raw: bytes, channels: int) -> NDArray[np.float32]:
    """Decode little-endian packed signed 24-bit PCM into float32 frames.

    Args:
        raw: Interleaved samples, 3 bytes each, little-endian.
        channels: Number of interleaved channels.

    Returns:
        Array of shape ``(frames, channels)`` in ``[-1.0, 1.0]``. ``0x7FFFFF``
        maps to exactly ``+1.0`` and ``0x800000`` to exactly ``-1.0``.

    Raises:
        ValueError: If ``channels`` is below 1 or ``raw`` holds a partial frame.
    """
    if channels < 1:
        raise ValueError(f"channels must be at least 1, got {channels}")
    frame_bytes = _BYTES_PER_SAMPLE * channels
    if len(raw) % frame_bytes != 0:
        raise ValueError(
            f"int24 data must be a multiple of {frame_bytes} bytes for {channels} "
            f"channel(s), got {len(raw)} bytes"
        )
    triples = np.frombuffer(raw, dtype=np.uint8).reshape(-1, _BYTES_PER_SAMPLE).astype(np.int32)
    unsigned = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    signed = (unsigned ^ _SIGN_BIT) - _SIGN_BIT
    # Divide in float64 and clip: -0x800000 / 0x7FFFFF would round to a hair
    # below -1.0 in float32, and downstream stages assume the closed interval.
    scaled = np.clip(signed / _INT24_FULL_SCALE, -1.0, 1.0)
    return scaled.astype(np.float32).reshape(-1, channels)


def to_mono(block: NDArray[np.float32]) -> NDArray[np.float32]:
    """Mix a block down to mono by averaging its channels.

    Args:
        block: Shape ``(frames,)`` or ``(frames, channels)``.

    Returns:
        A float32 array of shape ``(frames,)``. Antiphase channels cancel to
        exactly zero.
    """
    samples = np.asarray(block, dtype=np.float32)
    if samples.ndim == 1:
        return samples
    if samples.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D block, got {samples.ndim}-D")
    if samples.shape[1] == 1:
        return np.ascontiguousarray(samples[:, 0])
    mixed: NDArray[np.float32] = samples.mean(axis=1, dtype=np.float32)
    return mixed


DownmixMode = Literal["mean", "auto"]

_ACTIVE_FLOOR = 1e-9  # total RMS below this counts as digital silence
_INVERT_BELOW = -0.5  # correlation that flips a channel's polarity ...
_RESTORE_ABOVE = 0.0  # ... and the one that flips it back (hysteresis)


class ChannelMixer:
    """Stateful multi-channel to mono mixer that survives a dead channel.

    A plain channel mean loses 6 dB when one channel is silent (a mono mic on
    a stereo device) and cancels to nothing when the channels are in antiphase
    (a miswired cable). In ``auto`` mode the channels are weighted by their
    smoothed RMS, so a silent or hissing channel contributes almost nothing,
    and for stereo a channel that is strongly anti-correlated with the other
    is polarity-flipped. Weights are ramped across each block, so a change of
    state never produces a click. ``mean`` is the plain average of ``to_mono``.
    """

    def __init__(self, rate_hz: int, mode: DownmixMode = "auto", tau_ms: float = 300.0) -> None:
        """Create a mixer.

        Args:
            rate_hz: Sample rate of the incoming blocks.
            mode: ``"auto"`` for adaptive weighting, ``"mean"`` for a plain mean.
            tau_ms: Time constant of the level and correlation estimates.

        Raises:
            ValueError: If ``rate_hz`` or ``tau_ms`` is not positive.
        """
        if rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        if tau_ms <= 0:
            raise ValueError(f"tau_ms must be positive, got {tau_ms}")
        self._mode = mode
        self._tau_samples = tau_ms / 1000 * rate_hz
        self._power: NDArray[np.float64] | None = None
        self._cross = 0.0
        self._weights: NDArray[np.float32] | None = None
        self._inverted = False

    @property
    def polarity_inverted(self) -> bool:
        """Whether the second channel is currently being polarity-flipped."""
        return self._inverted

    def process(self, block: NDArray[np.float32]) -> NDArray[np.float32]:
        """Mix one block down to mono.

        Args:
            block: Shape ``(frames,)`` or ``(frames, channels)``.

        Returns:
            A float32 array of shape ``(frames,)``.
        """
        samples = np.asarray(block, dtype=np.float32)
        if samples.ndim == 1:
            return samples
        if samples.ndim != 2:
            raise ValueError(f"expected a 1-D or 2-D block, got {samples.ndim}-D")
        frames, channels = samples.shape
        if channels == 1 or self._mode == "mean":
            return to_mono(samples)
        if frames == 0:
            return np.zeros(0, dtype=np.float32)

        self._update_estimates(samples)
        target = self._target_weights(channels)
        previous = target if self._weights is None else self._weights
        ramp = (np.arange(1, frames + 1, dtype=np.float32) / frames)[:, np.newaxis]
        weights = previous + (target - previous) * ramp
        self._weights = target
        mixed: NDArray[np.float32] = np.sum(samples * weights, axis=1, dtype=np.float32)
        return mixed

    def _update_estimates(self, samples: NDArray[np.float32]) -> None:
        frames, channels = samples.shape
        # Mean removal first: a dead channel with a DC offset must stay dead.
        centred = samples.astype(np.float64) - samples.mean(axis=0, dtype=np.float64)
        power = np.mean(centred**2, axis=0)
        cross = float(np.mean(centred[:, 0] * centred[:, 1])) if channels == 2 else 0.0
        if self._power is None or self._power.shape[0] != channels:
            self._power, self._cross = power, cross
            self._weights, self._inverted = None, False
        else:
            alpha = 1.0 - float(np.exp(-frames / self._tau_samples))
            self._power = self._power + alpha * (power - self._power)
            self._cross += alpha * (cross - self._cross)
        if channels == 2:
            denominator = float(np.sqrt(self._power[0] * self._power[1]))
            if denominator > _ACTIVE_FLOOR:
                correlation = self._cross / denominator
                if correlation < _INVERT_BELOW:
                    self._inverted = True
                elif correlation > _RESTORE_ABOVE:
                    self._inverted = False

    def _target_weights(self, channels: int) -> NDArray[np.float32]:
        assert self._power is not None
        rms = np.sqrt(self._power)
        total = float(rms.sum())
        base = rms / total if total > _ACTIVE_FLOOR else np.full(channels, 1.0 / channels)
        if self._inverted:
            base = base * np.array([1.0, -1.0])
        return base.astype(np.float32)


INT16_FULL_SCALE = 32767


def float_to_int16(
    block: NDArray[np.float32],
    *,
    dither: bool = True,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int16]:
    """Quantize float samples to int16, optionally with TPDF dither.

    Plain rounding makes the error a function of the signal: low-level detail
    below one step vanishes and louder tones grow harmonic distortion. Adding
    triangular noise of +-1 step before rounding makes the error independent
    of the signal (a flat hiss of about -98 dBFS instead of distortion).
    Digital silence is left untouched so pauses stay silent.

    Args:
        block: Float samples nominally in ``[-1.0, 1.0]``.
        dither: Apply TPDF dither (default) or round only.
        rng: Random source; pass a seeded generator for reproducible output.

    Returns:
        int16 samples; out-of-range input saturates instead of wrapping.
    """
    scaled = np.asarray(block, dtype=np.float64) * INT16_FULL_SCALE
    if dither and np.any(scaled):
        generator = rng if rng is not None else np.random.default_rng()
        scaled = scaled + generator.random(scaled.shape) - generator.random(scaled.shape)
    info = np.iinfo(np.int16)
    quantized: NDArray[np.int16] = np.clip(np.rint(scaled), info.min, info.max).astype(np.int16)
    return quantized
