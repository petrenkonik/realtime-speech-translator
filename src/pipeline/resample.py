"""Streaming polyphase resamplers: microphone audio to 16 kHz, speech to a device rate.

``scipy.signal.resample_poly`` is stateless and clicks at block edges, so the
filter history is carried across calls here instead.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.signal import firwin, kaiserord

from pipeline.pcm import to_mono

OUT_RATE_HZ = 16_000
SUPPORTED_IN_RATES_HZ = (16_000, 32_000, 44_100, 48_000)
# ``RateConverter`` accepts any rate in this range, on both sides.
MIN_RATE_HZ = 8_000
MAX_RATE_HZ = 192_000

# Passband and stopband edges as fractions of the output rate: 7 kHz and 8 kHz
# at 16 kHz. Anything above 8 kHz would alias into the speech band.
_PASSBAND_END = 7_000 / OUT_RATE_HZ
_STOPBAND_START = 8_000 / OUT_RATE_HZ
# A 5 dB margin over the required 60 dB rejection.
_STOPBAND_DB = 65.0


class UnsupportedSampleRateError(ValueError):
    """Raised when the input rate has no tested resampling path."""


class StreamingResampler:
    """Block-by-block resampler with a linear-phase FIR and carried state.

    The output length over a whole stream is ``ceil(N * out / in)``, and the
    result does not depend on how the input is split into blocks.
    """

    def __init__(self, in_rate_hz: int, out_rate_hz: int = OUT_RATE_HZ) -> None:
        """Design the filter for one input rate.

        Args:
            in_rate_hz: Sample rate of the incoming blocks.
            out_rate_hz: Sample rate of the output; only 16 kHz is tested.

        Raises:
            UnsupportedSampleRateError: If ``in_rate_hz`` is not supported.
        """
        self._check_rates(in_rate_hz, out_rate_hz)
        self._passthrough = in_rate_hz == out_rate_hz
        divisor = gcd(in_rate_hz, out_rate_hz)
        self._up = out_rate_hz // divisor
        self._down = in_rate_hz // divisor
        self._consumed = 0  # input samples seen so far
        self._next_output = 0  # index of the next output sample to produce
        if self._passthrough:
            self._taps_per_phase = 1
            self._phases = np.ones((1, 1), dtype=np.float32)
            self._history = np.zeros(0, dtype=np.float32)
            return
        self._phases = self._design(in_rate_hz, out_rate_hz)
        self._taps_per_phase = self._phases.shape[1]
        self._history = np.zeros(self._taps_per_phase - 1, dtype=np.float32)

    def _check_rates(self, in_rate_hz: int, out_rate_hz: int) -> None:
        """Reject a pair of rates that has no tested path."""
        if in_rate_hz not in SUPPORTED_IN_RATES_HZ:
            supported = ", ".join(str(rate) for rate in SUPPORTED_IN_RATES_HZ)
            raise UnsupportedSampleRateError(
                f"{in_rate_hz} Hz is not supported; supported input rates: {supported} Hz"
            )

    def _design(self, in_rate_hz: int, out_rate_hz: int) -> NDArray[np.float32]:
        """Build the polyphase bank, shape ``(up, taps_per_phase)``, oldest tap first."""
        cutoff_rate_hz = min(in_rate_hz, out_rate_hz)
        fs_up_hz = in_rate_hz * self._up
        transition_hz = (_STOPBAND_START - _PASSBAND_END) * cutoff_rate_hz
        numtaps, beta = kaiserord(_STOPBAND_DB, transition_hz / (fs_up_hz / 2))
        numtaps |= 1  # odd length keeps a whole-sample group delay at the upsampled rate
        centre_hz = (_PASSBAND_END + _STOPBAND_START) / 2 * cutoff_rate_hz
        prototype = firwin(numtaps, centre_hz, window=("kaiser", beta), fs=fs_up_hz)
        prototype *= self._up  # zero-stuffing divides the amplitude by ``up``
        taps = -(-numtaps // self._up)
        padded = np.zeros(taps * self._up)
        padded[:numtaps] = prototype
        # Row ``phase`` holds h[phase + j * up]; reverse j so the dot product
        # with an ascending window of input samples is a convolution.
        bank = padded.reshape(taps, self._up)[::-1].T
        return np.ascontiguousarray(bank, dtype=np.float32)

    def process(self, block: NDArray[np.float32]) -> NDArray[np.float32]:
        """Resample one block.

        Args:
            block: Shape ``(frames,)`` or ``(frames, channels)``; channels are
                averaged to mono first.

        Returns:
            The output samples that became computable with this block, possibly
            an empty array.
        """
        mono = to_mono(block)
        if self._passthrough:
            return mono.copy()
        if len(mono) == 0:
            return np.zeros(0, dtype=np.float32)

        history = np.concatenate((self._history, mono))
        total = self._consumed + len(mono)
        last_output = (total * self._up - 1) // self._down
        outputs = np.arange(self._next_output, last_output + 1, dtype=np.int64)

        position = outputs * self._down  # position on the upsampled time grid
        newest = position // self._up  # newest input sample each output needs
        phase = position % self._up
        windows = sliding_window_view(history, self._taps_per_phase)[newest - self._consumed]
        products = np.einsum("ij,ij->i", windows, self._phases[phase])
        result: NDArray[np.float32] = products.astype(np.float32)

        keep = self._taps_per_phase - 1
        self._history = history[len(history) - keep :].copy()
        self._consumed = total
        self._next_output = last_output + 1
        return result


class RateConverter(StreamingResampler):
    """Converts between any two rates; used for synthesized speech on its way to a device.

    The filter keeps 87.5 % of the lower rate's Nyquist band (7 kHz of 8 kHz at 16 kHz, 9.6 kHz
    of 11 kHz for a 22.05 kHz voice) and rejects the rest by 65 dB, which is what stops the
    mirror images of an upsampled voice from reaching the speaker. The output length over a
    whole stream is still ``ceil(N * out / in)`` and block boundaries do not show.
    """

    def _check_rates(self, in_rate_hz: int, out_rate_hz: int) -> None:
        for rate_hz in (in_rate_hz, out_rate_hz):
            if not MIN_RATE_HZ <= rate_hz <= MAX_RATE_HZ:
                raise UnsupportedSampleRateError(
                    f"{rate_hz} Hz is not supported; rates must be {MIN_RATE_HZ}-{MAX_RATE_HZ} Hz"
                )
