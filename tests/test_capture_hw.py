"""Real-microphone smoke test. Run only on request: ``pytest -m hw`` with the V380 plugged in."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pipeline.bus import ChunkBus
from pipeline.capture import CaptureConfig, CaptureSession
from pipeline.devices import SoundDeviceListing, open_sounddevice_input
from pipeline.vad import Segmenter, SileroVad, VadStage

pytestmark = pytest.mark.hw


def test_the_real_microphone_delivers_callbacks_for_one_second() -> None:
    """Counts callbacks only; no audio is stored or inspected beyond the VAD chain."""
    stage = VadStage(SileroVad(Path("models/silero_vad.onnx")), Segmenter())  # type: ignore[arg-type]
    session = CaptureSession(
        CaptureConfig("V380"), SoundDeviceListing(), open_sounddevice_input, stage, ChunkBus()
    )
    session.start()
    try:
        time.sleep(1.0)
        stats = session.stats()
    finally:
        session.stop()
    assert stats.connected
    assert stats.callbacks >= 90  # 10 ms blocks, one second, some start-up slack
    assert stats.input_overflow == 0
    assert stats.max_callback_ms < 5.0
    assert session.fatal_error is None
