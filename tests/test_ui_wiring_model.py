"""The real wiring on the real local models (WD11); needs ``models/`` from fetch_models.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeDeviceList, FakeOutputStream, FakeStreamFactory, make_device

from pipeline.config import parse_settings
from pipeline.ui.wiring import RealWiring

pytestmark = pytest.mark.model

MODELS = Path(__file__).resolve().parent.parent / "models"


@pytest.mark.parametrize("turn_mode", ["fixed", "smart"])
def test_wd11_the_real_wiring_builds_and_starts_on_the_local_models(turn_mode: str) -> None:
    if not (MODELS / "whisper-tiny.en" / "model.bin").is_file():
        pytest.skip("run scripts/fetch_models.py first")
    settings = parse_settings(
        [
            "--input",
            "V380",
            "--route",
            "de=Realtek,fr=HS100W",
            "--models-dir",
            str(MODELS),
            "--whisper-model",
            "tiny.en",
            "--turn-mode",
            turn_mode,
        ]
    )
    devices = FakeDeviceList(
        [
            make_device("Microphone (V380 USB)", index=1, inputs=2),
            make_device("Speakers (Realtek)", index=2, inputs=0, outputs=2),
            make_device("Headset (HS100W)", index=3, inputs=0, outputs=2),
        ]
    )
    wiring = RealWiring(
        devices=devices,
        input_stream_factory=FakeStreamFactory(),
        output_stream_factory=FakeStreamFactory(FakeOutputStream),
    )
    assert wiring.problems(settings) == ()
    built = wiring.build(settings)
    try:
        assert built.pipeline.start(lenient=True) == ()
        assert built.pipeline.running
    finally:
        built.pipeline.stop()
        built.close()
