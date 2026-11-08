"""V1-V3: the real capture and output code through a VB-Cable loopback.

Run only on request: ``pytest -m virtual``. The speech goes into the virtual cable, not
the speakers. Device names come from the environment so a machine with two cables can
use them:

* ``PIPELINE_CABLE_IN``: output device the test plays into (default ``CABLE Input``).
* ``PIPELINE_CABLE_OUT``: input device that hears it (default ``CABLE Output``).
* ``PIPELINE_CABLE_PNP_ID``: instance id of the cable driver for V3 (optional).

With a single cable V1 and V2 run one after another, which is all they need. T7 and the
full A+B bench need two cables.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from sapi_speech import build_stream, synthesize
from scipy.signal import fftconvolve

from pipeline.bus import ChunkBus
from pipeline.capture import CaptureConfig, CaptureSession
from pipeline.devices import (
    DeviceNotFoundError,
    SoundDeviceListing,
    find_device,
    open_sounddevice_input,
    open_sounddevice_output,
)
from pipeline.output import OutputConfig, OutputSession
from pipeline.resample import StreamingResampler
from pipeline.vad import Chunk, Segmenter, SileroVad, VadStage

pytestmark = pytest.mark.virtual

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"
CABLE_IN = os.environ.get("PIPELINE_CABLE_IN", "CABLE Input")
CABLE_OUT = os.environ.get("PIPELINE_CABLE_OUT", "CABLE Output")
RATE_HZ = 16_000
PLAY_BLOCK = 1_600  # 100 ms at 16 kHz
RESTORE_LIMIT_S = 3.0


class Recorder:
    """A ``ChunkStage`` that keeps the raw 16 kHz signal instead of looking for speech."""

    def __init__(self) -> None:
        self.parts: list[NDArray[np.float32]] = []
        self._position = 0

    @property
    def position_samples(self) -> int:
        return self._position

    def push(self, samples: NDArray[np.float32]) -> list[Chunk]:
        self.parts.append(samples.copy())
        self._position += len(samples)
        return []

    def flush(self) -> Chunk | None:
        return None

    def signal(self) -> NDArray[np.float32]:
        return np.concatenate(self.parts) if self.parts else np.zeros(0, dtype=np.float32)


@pytest.fixture(scope="module")
def utterances(tmp_path_factory: pytest.TempPathFactory) -> list[NDArray[np.float32]]:
    out, error = synthesize(tmp_path_factory.mktemp("sapi"))
    if not out:
        pytest.skip(f"SAPI voices unavailable: {error}")
    return out


@pytest.fixture(scope="module", autouse=True)
def cable() -> None:
    devices = SoundDeviceListing().list_devices()
    try:
        find_device(devices, CABLE_IN, kind="output")
        find_device(devices, CABLE_OUT, kind="input")
    except DeviceNotFoundError as error:
        pytest.skip(f"VB-Cable not installed: {error}")


@pytest.fixture
def output() -> Iterator[OutputSession]:
    session = OutputSession(OutputConfig(CABLE_IN), SoundDeviceListing(), open_sounddevice_output)
    session.start()
    yield session
    session.stop()


def play(session: OutputSession, signal: NDArray[np.float32]) -> None:
    """Play a 16 kHz signal in real time and wait until the ring is empty."""
    rate = session.sample_rate_hz
    assert rate is not None
    resampler = StreamingResampler(RATE_HZ, rate)
    session.begin_utterance()
    for start in range(0, len(signal), PLAY_BLOCK):
        block = resampler.process(signal[start : start + PLAY_BLOCK])
        assert session.write_all(block, timeout_s=30.0)
    session.end_utterance()
    deadline = time.monotonic() + 30.0
    while session.stats().queued_ms > 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.3)  # the device buffer still holds the last block


def capture(stage: VadStage | Recorder) -> CaptureSession:
    return CaptureSession(
        CaptureConfig(CABLE_OUT), SoundDeviceListing(), open_sounddevice_input, stage, ChunkBus()
    )


def offline_chunk_count(signal: NDArray[np.float32]) -> int:
    stage = VadStage(SileroVad(MODEL_PATH), Segmenter())
    count = len(stage.push(signal))
    return count + (1 if stage.flush() is not None else 0)


def peak_correlation(recording: NDArray[np.float32], reference: NDArray[np.float32]) -> float:
    """Best normalized correlation of ``reference`` anywhere inside ``recording`` (0..1)."""
    cross = fftconvolve(recording, reference[::-1], mode="valid")
    energy = np.concatenate([[0.0], np.cumsum(recording.astype(np.float64) ** 2)])
    window = energy[len(reference) :] - energy[: -len(reference)]
    norm = np.sqrt(window * float(np.sum(reference.astype(np.float64) ** 2)))
    return float(np.max(np.abs(cross) / np.maximum(norm, 1e-9)))


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run scripts/fetch_models.py first")
def test_v1_loopback_gives_the_same_chunks_as_the_offline_run(
    utterances: list[NDArray[np.float32]], output: OutputSession
) -> None:
    signal = np.concatenate([build_stream(utterances, 0), build_stream(utterances, 1)])
    expected = offline_chunk_count(signal)
    assert expected >= 4  # six utterances; a run that expects nothing proves nothing

    session = capture(VadStage(SileroVad(MODEL_PATH), Segmenter()))
    session.start()
    try:
        time.sleep(0.3)
        play(output, signal)
    finally:
        session.stop()
    stats = session.stats()

    assert session.fatal_error is None
    assert stats.chunks == expected
    assert stats.input_overflow == 0
    assert stats.ring_lost_frames == 0
    assert output.stats().underruns == 0


def test_v2_the_output_reaches_the_device_without_underruns(
    utterances: list[NDArray[np.float32]], output: OutputSession
) -> None:
    reference = utterances[0]
    padded = np.concatenate([np.zeros(RATE_HZ // 2, np.float32), reference])

    recorder = Recorder()
    session = capture(recorder)
    session.start()
    try:
        time.sleep(0.3)
        play(output, padded)
    finally:
        session.stop()

    stats = output.stats()
    assert stats.underruns == 0
    assert stats.driver_underflows == 0
    correlation = peak_correlation(recorder.signal(), reference)
    assert correlation >= 0.8, f"correlation {correlation:.2f}"


def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined,unused-ignore]
    except (AttributeError, OSError):
        return False


def _cable_instance_id() -> str:
    if os.environ.get("PIPELINE_CABLE_PNP_ID"):
        return os.environ["PIPELINE_CABLE_PNP_ID"]
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "(Get-PnpDevice -Class MEDIA"
            " | Where-Object FriendlyName -like '*VB-Audio Virtual Cable*'"
            " | Select-Object -First 1).InstanceId",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result.stdout.strip()


@pytest.mark.skipif(not _is_admin(), reason="needs an elevated shell (pnputil /restart-device)")
def test_v3_the_stream_is_restored_after_the_device_restarts() -> None:
    instance_id = _cable_instance_id()
    if not instance_id:
        pytest.skip("cable driver not found; set PIPELINE_CABLE_PNP_ID")

    session = capture(Recorder())
    session.start()
    try:
        time.sleep(0.5)
        subprocess.run(["pnputil", "/restart-device", instance_id], check=True, capture_output=True)
        started = time.monotonic()
        restored = False
        while time.monotonic() - started < 10.0:
            stats = session.stats()
            if stats.connected and stats.reconnects >= 1:
                restored = True
                break
            time.sleep(0.05)
        elapsed = time.monotonic() - started
        callbacks_before = session.stats().callbacks
        time.sleep(0.5)
        callbacks_after = session.stats().callbacks
    finally:
        session.stop()

    assert restored
    assert elapsed <= RESTORE_LIMIT_S
    assert callbacks_after > callbacks_before  # audio flows again, not just "connected"
    assert session.fatal_error is None
