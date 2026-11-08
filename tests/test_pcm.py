"""Tests for PCM conversion: int24 unpacking (U1) and stereo to mono (U2)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.pcm import to_mono, unpack_int24

pytestmark = pytest.mark.unit


def _pack_int24(values: list[int]) -> bytes:
    """Encode signed 24-bit integers as little-endian 3-byte samples."""
    return b"".join((v & 0xFFFFFF).to_bytes(3, "little") for v in values)


def test_u1_int24_extremes_and_zero_are_exact() -> None:
    raw = _pack_int24([0x7FFFFF, -0x800000, 0])

    out = unpack_int24(raw, channels=1)

    assert out.dtype == np.float32
    assert out.shape == (3, 1)
    assert out[0, 0] == 1.0
    assert out[1, 0] == -1.0
    assert out[2, 0] == 0.0


def test_u1_int24_is_little_endian() -> None:
    # 0x000001 little-endian is b"\x01\x00\x00"; big-endian would read 0x010000.
    out = unpack_int24(b"\x01\x00\x00", channels=1)

    assert out[0, 0] == np.float32(1 / 0x7FFFFF)


def test_u1_int24_interleaved_channels() -> None:
    raw = _pack_int24([0x7FFFFF, -0x800000, 0, 0x7FFFFF])

    out = unpack_int24(raw, channels=2)

    np.testing.assert_array_equal(out, np.array([[1.0, -1.0], [0.0, 1.0]], dtype=np.float32))


def test_u1_int24_rejects_partial_frame() -> None:
    with pytest.raises(ValueError, match="multiple of 6 bytes"):
        unpack_int24(b"\x00" * 7, channels=2)


def test_u2_stereo_to_mono_is_channel_mean() -> None:
    block = np.array([[0.5, 0.25], [1.0, -1.0], [-0.5, 0.0]], dtype=np.float32)

    mono = to_mono(block)

    assert mono.dtype == np.float32
    np.testing.assert_array_equal(mono, np.array([0.375, 0.0, -0.25], dtype=np.float32))


def test_u2_antiphase_stereo_gives_zero() -> None:
    signal = np.sin(np.linspace(0, 20 * np.pi, 1000)).astype(np.float32)
    block = np.stack([signal, -signal], axis=1)

    assert np.all(to_mono(block) == 0.0)


def test_u2_mono_input_passes_through() -> None:
    block = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    np.testing.assert_array_equal(to_mono(block), block)
    np.testing.assert_array_equal(to_mono(block[:, np.newaxis]), block)


def test_u1_int24_rejects_zero_channels() -> None:
    with pytest.raises(ValueError, match="channels must be at least 1"):
        unpack_int24(b"", channels=0)


def test_u2_rejects_three_dimensional_block() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        to_mono(np.zeros((2, 2, 2), dtype=np.float32))
