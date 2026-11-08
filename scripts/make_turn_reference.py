"""Regenerate ``tests/data/smart_turn_reference.npz`` with the vendor feature pipeline.

Run it in a throwaway environment that has ``transformers`` installed (it is NOT a
project dependency and must never be added to ``pyproject.toml``):

    python scripts/make_turn_reference.py

The vendor code is copied in spirit from ``pipecat-ai/smart-turn`` ``inference.py``
(BSD-2-Clause): keep the last 8 s or zero-pad the front, then
``WhisperFeatureExtractor(chunk_length=8)`` with ``do_normalize=True``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from turn_reference import reference_clips  # noqa: E402


def vendor_features(audio: np.ndarray) -> np.ndarray:
    """The vendor ``predict_endpoint`` pre-processing, up to the ONNX call."""
    from transformers import WhisperFeatureExtractor

    max_samples = 8 * 16_000
    if len(audio) > max_samples:
        audio = audio[-max_samples:]
    elif len(audio) < max_samples:
        audio = np.pad(audio, (max_samples - len(audio), 0), mode="constant", constant_values=0)
    extractor = WhisperFeatureExtractor(chunk_length=8)
    inputs = extractor(
        audio,
        sampling_rate=16_000,
        return_tensors="np",
        padding="max_length",
        max_length=max_samples,
        truncation=True,
        do_normalize=True,
    )
    return inputs.input_features.squeeze(0).astype(np.float32)


def main() -> None:
    """Write the reference file."""
    reference = {name: vendor_features(clip) for name, clip in reference_clips().items()}
    target = ROOT / "tests" / "data" / "smart_turn_reference.npz"
    target.parent.mkdir(exist_ok=True)
    np.savez_compressed(target, **reference)
    print(f"wrote {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
