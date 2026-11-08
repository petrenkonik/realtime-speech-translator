"""Pipeline orchestrator on fakes: start/stop, runtime control, lane isolation (P1-P20)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from fakes import (
    EnergyVad,
    FakeClock,
    FakeDeviceList,
    FakeOutputStream,
    FakeStreamFactory,
    make_device,
    silence,
    speech,
    stream_of,
)

from pipeline.app import DeviceState, Pipeline, PipelineStateError, UnknownTargetError
from pipeline.capture import CaptureConfig, ChunkStage
from pipeline.config import ConfigError, LaneRoute, Settings
from pipeline.devices import DeviceInfo, DeviceNotFoundError
from pipeline.lanes import Speech, Translation
from pipeline.output import OutputConfig
from pipeline.transcribe import Utterance
from pipeline.vad import Float32Array, Segmenter, SegmenterConfig, VadStage

pytestmark = pytest.mark.unit

MIC = "Microphone (V380 USB)"
MIC_B = "Microphone (Backup Mic)"
REALTEK = "Speakers (Realtek Audio)"
HS100W = "Headset (HS100W)"
SPARE = "Speakers (USB Spare)"
INPUT_RATE_HZ = 48_000
OUTPUT_RATE_HZ = 16_000
HUGE_WATCHDOG_MS = 10**9  # the fake clock jumps in these tests; only scripted failures count


def wait_for(predicate: Callable[[], bool], what: str, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        time.sleep(0.002)


class FakeWhisper:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(
        self, samples: Float32Array, *, hotwords: str | None, initial_prompt: str | None
    ) -> str:
        self.calls += 1
        return "hello world"


class FakeTranslator:
    def __init__(self, target: str) -> None:
        self.target = target
        self.calls: list[str] = []

    def translate(self, text: str) -> str:
        self.calls.append(text)
        return f"[{self.target}] {text}"


class FakeTts:
    def synthesize(self, text: str) -> Speech:
        return Speech(samples=np.full(len(text) * 100, 0.25, dtype=np.float32), rate_hz=16_000)


@dataclass
class Rig:
    pipeline: Pipeline
    devices: FakeDeviceList
    clock: FakeClock
    input_factory: FakeStreamFactory
    output_factory: FakeStreamFactory
    whisper: FakeWhisper
    translators: dict[str, FakeTranslator] = field(default_factory=dict)
    stages: list[ChunkStage] = field(default_factory=list)
    broken_models: set[str] = field(default_factory=set)  # targets whose translator will not load

    def say(self, speech_ms: int = 600) -> None:
        """Speak into the current input stream, followed by enough silence to end the chunk."""
        stream = self.input_factory.current
        stream.feed(stream_of(speech(speech_ms, INPUT_RATE_HZ), silence(700, INPUT_RATE_HZ)))

    def out_streams(self, device_name: str) -> list[Any]:
        return [
            stream
            for stream, device in zip(
                self.output_factory.streams, self.output_factory.opened_devices, strict=True
            )
            if device.name == device_name
        ]

    def counter(self, name: str) -> int:
        return self.pipeline.metrics.counter(name).value

    def heard(self, device_name: str) -> bool:
        """True if the newest stream of that output has played any non-silent audio."""
        stream = self.out_streams(device_name)[-1]
        return bool(np.any(stream.pull(blocks=40) != 0.0))


def settings_for(
    input_query: str = "V380", routes: tuple[tuple[str, str], ...] = (("de", "Realtek"),)
) -> Settings:
    return Settings(
        capture=CaptureConfig(input_query, watchdog_ms=HUGE_WATCHDOG_MS),
        routes=tuple(
            LaneRoute(target, OutputConfig(query, watchdog_ms=HUGE_WATCHDOG_MS))
            for target, query in routes
        ),
        targets=tuple(target for target, _ in routes),
        segmenter=SegmenterConfig(),
    )


def build(
    routes: tuple[tuple[str, str], ...] = (("de", "Realtek"), ("fr", "HS100W")),
    extra_devices: tuple[DeviceInfo, ...] = (),
) -> Rig:
    broken_models: set[str] = set()
    clock = FakeClock()
    devices = FakeDeviceList(
        [
            make_device(MIC, index=1, rate_hz=INPUT_RATE_HZ, inputs=2),
            make_device(MIC_B, index=5, rate_hz=INPUT_RATE_HZ, inputs=2),
            make_device(REALTEK, index=2, rate_hz=OUTPUT_RATE_HZ, inputs=0, outputs=2),
            make_device(HS100W, index=3, rate_hz=OUTPUT_RATE_HZ, inputs=0, outputs=2),
            make_device(SPARE, index=4, rate_hz=OUTPUT_RATE_HZ, inputs=0, outputs=2),
            *extra_devices,
        ]
    )
    input_factory = FakeStreamFactory()
    output_factory = FakeStreamFactory(FakeOutputStream)
    whisper = FakeWhisper()
    translators: dict[str, FakeTranslator] = {}
    stages: list[ChunkStage] = []

    def stage_factory(settings: Settings) -> ChunkStage:
        stage = VadStage(EnergyVad(), Segmenter(settings.segmenter))  # type: ignore[arg-type]
        stages.append(stage)
        return stage

    def translator_factory(target: str) -> FakeTranslator:
        if target in broken_models:
            raise RuntimeError(f"model for {target} is missing")
        translators[target] = FakeTranslator(target)
        return translators[target]

    pipeline = Pipeline(
        settings_for(routes=routes),
        devices=devices,
        input_stream_factory=input_factory,
        output_stream_factory=output_factory,
        stage_factory=stage_factory,
        speech_to_text=whisper,
        translator_factory=translator_factory,
        tts_factory=lambda target: FakeTts(),
        clock=clock,
    )
    return Rig(
        pipeline,
        devices,
        clock,
        input_factory,
        output_factory,
        whisper,
        translators,
        stages,
        broken_models,
    )


@pytest.fixture
def rig() -> Rig:
    return build()


@pytest.fixture
def running() -> Any:
    """A started rig that is stopped again whatever the test does."""
    built = build()
    built.pipeline.start()
    yield built
    built.pipeline.stop()


# ---------------------------------------------------------------- P1 lifecycle


def test_p1_start_opens_every_device_and_stop_closes_them(rig: Rig) -> None:
    pipeline = rig.pipeline
    assert not pipeline.running
    assert pipeline.input_state() is DeviceState.DISCONNECTED
    pipeline.start()
    assert pipeline.running
    assert pipeline.input_state() is DeviceState.CONNECTED
    assert pipeline.output_state("de") is DeviceState.CONNECTED
    assert pipeline.output_state("fr") is DeviceState.CONNECTED
    assert [d.name for d in rig.output_factory.opened_devices] == [REALTEK, HS100W]
    pipeline.stop()
    assert not pipeline.running
    assert rig.input_factory.current.closed
    assert all(stream.closed for stream in rig.output_factory.streams)
    assert pipeline.input_state() is DeviceState.DISCONNECTED
    assert pipeline.output_state("de") is DeviceState.DISCONNECTED
    assert pipeline.chunk_bus.closed
    assert pipeline.text_bus.closed


def test_p1_stop_is_idempotent_and_safe_before_start(rig: Rig) -> None:
    rig.pipeline.stop()  # never started
    rig.pipeline.stop()
    assert not rig.input_factory.streams

    other = build()
    other.pipeline.start()
    other.pipeline.stop()
    other.pipeline.stop()
    assert len(other.input_factory.streams) == 1


def test_p1_start_twice_and_start_after_stop_are_refused(rig: Rig) -> None:
    rig.pipeline.start()
    with pytest.raises(PipelineStateError, match="already started"):
        rig.pipeline.start()
    rig.pipeline.stop()
    with pytest.raises(PipelineStateError, match="build a new Pipeline"):
        rig.pipeline.start()


def test_p1_threads_end_with_stop(running: Rig) -> None:
    names_before = {t.name for t in threading.enumerate()}
    assert "transcribe" in names_before
    assert {"lane-de", "lane-fr", "capture"} <= names_before
    running.pipeline.stop()
    remaining = {t.name for t in threading.enumerate()}
    assert not remaining & {"transcribe", "lane-de", "lane-fr", "capture"}


# ---------------------------------------------------------------- P2 end to end


def test_p2_utterance_reaches_every_lane_and_the_ui_subscription(running: Rig) -> None:
    ui = running.pipeline.text_bus.subscribe("ui", maxsize=8)
    running.say()
    wait_for(lambda: running.counter("lane.de.utterances") == 1, "de lane")
    wait_for(lambda: running.counter("lane.fr.utterances") == 1, "fr lane")
    assert running.whisper.calls == 1  # one decode, however many languages
    utterance = ui.get(timeout_s=1.0)
    assert isinstance(utterance, Utterance)
    assert utterance.text == "hello world"
    assert running.translators["de"].calls == ["hello world"]
    assert running.translators["fr"].calls == ["hello world"]
    assert running.heard(REALTEK)
    assert running.heard(HS100W)


def test_p2_accessors_expose_stats_for_the_ui(running: Rig) -> None:
    running.say()
    wait_for(lambda: running.counter("lane.de.utterances") == 1, "de lane")
    pipeline = running.pipeline
    assert pipeline.capture_stats().chunks == 1
    assert pipeline.capture_stats().connected
    assert pipeline.output_stats("de").connected
    stats = pipeline.bus_stats()
    assert "whisper" in stats.chunks
    assert {"lane-de", "lane-fr"} <= set(stats.text)
    assert stats.chunks["whisper"].delivered == 1
    device = pipeline.input_device()
    assert device is not None
    assert device.name == MIC
    output = pipeline.output_device("fr")
    assert output is not None
    assert output.name == HS100W
    assert pipeline.targets == ("de", "fr")
    assert pipeline.settings.capture.device_query == "V380"
    assert pipeline.position_samples > 0


# ---------------------------------------------------------------- P3 stop flushes


def test_p3_stop_publishes_the_pending_chunk(rig: Rig) -> None:
    rig.pipeline.start()
    tap = rig.pipeline.chunk_bus.subscribe("tap", maxsize=4)
    rig.input_factory.current.feed(speech(600, INPUT_RATE_HZ))  # no silence: chunk stays open
    rig.pipeline.stop()
    chunk = tap.get(timeout_s=0)
    assert chunk is not None
    assert chunk.reason == "flush"


# ---------------------------------------------------------------- P4 input control


def test_p4_input_can_be_disconnected_and_reconnected_with_the_same_stage(running: Rig) -> None:
    pipeline = running.pipeline
    ui = pipeline.text_bus.subscribe("ui", maxsize=8)
    tap = pipeline.chunk_bus.subscribe("tap", maxsize=8)
    running.say()
    first = ui.get(timeout_s=5.0)
    assert first is not None

    first_stream = running.input_factory.current
    running.input_factory.current.feed(speech(300, INPUT_RATE_HZ))  # pending when we disconnect
    pipeline.disconnect_input()
    assert pipeline.input_state() is DeviceState.DISCONNECTED
    assert first_stream.closed
    assert pipeline.input_device() is None
    assert pipeline.running  # only the input went away
    assert tap.get(timeout_s=0) is not None  # the first normal chunk
    flushed = tap.get(timeout_s=0)
    assert flushed is not None
    assert flushed.reason == "flush"

    # Let the flushed chunk be decoded first so the next one cannot be merged with it.
    second = ui.get(timeout_s=5.0)
    assert second is not None

    pipeline.disconnect_input()  # idempotent
    pipeline.connect_input()
    assert pipeline.input_state() is DeviceState.CONNECTED
    assert len(running.input_factory.streams) == 2
    pipeline.connect_input()  # idempotent
    assert len(running.input_factory.streams) == 2

    running.say()
    third = ui.get(timeout_s=5.0)
    assert third is not None
    assert (first.id, second.id, third.id) == (1, 2, 3)  # chunk ids continue: same stage
    assert second.start_sample >= first.speech_end_sample
    assert third.start_sample >= second.speech_end_sample  # the stream clock kept running
    assert len(running.stages) == 1  # one stage for the whole life of the pipeline
    wait_for(lambda: running.counter("lane.de.utterances") == 3, "utterances after reconnect")
    assert running.whisper.calls == 3


def test_p4_counters_survive_an_input_reconnect(running: Rig) -> None:
    running.say()
    wait_for(lambda: running.pipeline.capture_stats().chunks == 1, "first chunk")
    callbacks_before = running.pipeline.capture_stats().callbacks
    assert callbacks_before > 0
    running.pipeline.disconnect_input()
    parked = running.pipeline.capture_stats()
    assert not parked.connected
    assert parked.callbacks == callbacks_before
    assert parked.chunks == 1
    running.pipeline.connect_input()
    running.say()
    wait_for(lambda: running.pipeline.capture_stats().chunks == 2, "second chunk")
    after = running.pipeline.capture_stats()
    assert after.connected
    assert after.callbacks > callbacks_before


def test_p4_control_calls_need_a_running_pipeline(rig: Rig) -> None:
    for call in (
        rig.pipeline.connect_input,
        rig.pipeline.disconnect_input,
        lambda: rig.pipeline.connect_output("de"),
        lambda: rig.pipeline.disconnect_output("de"),
    ):
        with pytest.raises(PipelineStateError, match="not running"):
            call()
    rig.pipeline.start()
    rig.pipeline.stop()
    with pytest.raises(PipelineStateError, match="not running"):
        rig.pipeline.connect_input()


def test_p4_connect_input_raises_when_the_device_is_missing(running: Rig) -> None:
    running.pipeline.disconnect_input()
    running.devices.remove(MIC)
    with pytest.raises(DeviceNotFoundError):
        running.pipeline.connect_input()
    assert running.pipeline.input_state() is DeviceState.DISCONNECTED


# ---------------------------------------------------------------- P5 reconnecting state


def test_p5_a_lost_input_is_reconnecting_until_the_device_returns(running: Rig) -> None:
    pipeline = running.pipeline
    mic = next(d for d in running.devices.devices if d.name == MIC)
    running.devices.remove(MIC)
    running.input_factory.current.fail()
    wait_for(lambda: pipeline.input_state() is DeviceState.RECONNECTING, "reconnecting")
    assert pipeline.output_state("de") is DeviceState.CONNECTED  # others are not involved
    running.devices.devices.append(mic)
    running.clock.advance(1.5)
    wait_for(lambda: pipeline.input_state() is DeviceState.CONNECTED, "reconnected")
    stats = pipeline.capture_stats()
    assert stats.losses == 1
    assert stats.reconnects == 1


def test_p5_a_lost_output_is_reconnecting_and_the_other_lane_keeps_playing(running: Rig) -> None:
    pipeline = running.pipeline
    headset = next(d for d in running.devices.devices if d.name == HS100W)
    running.devices.remove(HS100W)
    running.out_streams(HS100W)[0].fail()
    wait_for(lambda: pipeline.output_state("fr") is DeviceState.RECONNECTING, "fr reconnecting")
    assert pipeline.output_state("de") is DeviceState.CONNECTED

    running.say()
    wait_for(lambda: running.counter("lane.de.utterances") == 1, "de lane")
    wait_for(lambda: running.counter("lane.fr.skipped_offline") == 1, "fr skipped")
    assert running.translators["fr"].calls == []
    assert running.heard(REALTEK)

    running.devices.devices.append(headset)
    running.clock.advance(1.5)
    wait_for(lambda: pipeline.output_state("fr") is DeviceState.CONNECTED, "fr back")


# ---------------------------------------------------------------- P6 output control


def test_p6_a_disconnected_output_skips_utterances_and_others_are_unaffected(
    running: Rig,
) -> None:
    pipeline = running.pipeline
    running.say()
    wait_for(lambda: running.counter("lane.fr.utterances") == 1, "first utterance")
    wait_for(lambda: running.counter("lane.de.utterances") == 1, "first utterance de")

    de_stream = running.out_streams(REALTEK)[0]
    fr_stream = running.out_streams(HS100W)[0]
    pipeline.disconnect_output("de")
    assert pipeline.output_state("de") is DeviceState.DISCONNECTED
    assert pipeline.output_state("fr") is DeviceState.CONNECTED
    assert de_stream.closed
    assert not fr_stream.closed
    assert pipeline.output_device("de") is None
    pipeline.disconnect_output("de")  # idempotent

    running.say()
    wait_for(lambda: running.counter("lane.fr.utterances") == 2, "fr second utterance")
    wait_for(lambda: running.counter("lane.de.skipped_offline") == 1, "de skipped")
    assert running.translators["de"].calls == ["hello world"]  # nothing new
    assert running.counter("lane.de.utterances") == 1
    assert not pipeline.output_stats("de").connected

    pipeline.connect_output("de")
    assert pipeline.output_state("de") is DeviceState.CONNECTED
    assert len(running.out_streams(REALTEK)) == 2
    pipeline.connect_output("de")  # idempotent
    assert len(running.out_streams(REALTEK)) == 2
    running.say()
    wait_for(lambda: running.counter("lane.de.utterances") == 2, "de after reconnect")
    assert running.heard(REALTEK)
    assert len(running.out_streams(HS100W)) == 1  # fr was never reopened


def test_p6_connect_output_raises_when_the_device_is_missing(running: Rig) -> None:
    running.pipeline.disconnect_output("fr")
    running.devices.remove(HS100W)
    with pytest.raises(DeviceNotFoundError):
        running.pipeline.connect_output("fr")
    assert running.pipeline.output_state("fr") is DeviceState.DISCONNECTED


def test_p6_unknown_targets_are_rejected(running: Rig) -> None:
    for call in (
        lambda: running.pipeline.output_state("xx"),
        lambda: running.pipeline.output_stats("xx"),
        lambda: running.pipeline.output_device("xx"),
        lambda: running.pipeline.connect_output("xx"),
        lambda: running.pipeline.disconnect_output("xx"),
        lambda: running.pipeline.set_output("xx", "Realtek"),
    ):
        with pytest.raises(UnknownTargetError, match="xx"):
            call()


# ---------------------------------------------------------------- P7 change devices by name


def test_p7_set_output_moves_one_lane_and_leaves_the_others_alone(running: Rig) -> None:
    pipeline = running.pipeline
    old_stream = running.out_streams(REALTEK)[0]
    fr_stream = running.out_streams(HS100W)[0]
    pipeline.set_output("de", "USB Spare")
    assert [d.name for d in running.output_factory.opened_devices][-1] == SPARE
    assert old_stream.closed
    assert not fr_stream.closed
    assert len(running.out_streams(HS100W)) == 1
    assert pipeline.output_state("de") is DeviceState.CONNECTED
    device = pipeline.output_device("de")
    assert device is not None
    assert device.name == SPARE
    assert pipeline.settings.routes[0].output.device_query == "USB Spare"
    assert pipeline.settings.routes[1].output.device_query == "HS100W"

    running.say()
    wait_for(lambda: running.counter("lane.de.utterances") == 1, "de lane on the new device")
    wait_for(lambda: running.counter("lane.fr.utterances") == 1, "fr lane")
    assert running.heard(SPARE)
    assert running.heard(HS100W)


def test_p7_set_output_with_an_unknown_device_keeps_the_old_output(running: Rig) -> None:
    with pytest.raises(DeviceNotFoundError):
        running.pipeline.set_output("de", "No Such Speaker")
    assert running.pipeline.output_state("de") is DeviceState.CONNECTED
    assert not running.out_streams(REALTEK)[0].closed
    assert running.pipeline.settings.routes[0].output.device_query == "Realtek"
    running.say()
    wait_for(lambda: running.counter("lane.de.utterances") == 1, "de lane")


def test_p7_set_output_while_disconnected_only_stores_the_choice(running: Rig) -> None:
    running.pipeline.disconnect_output("de")
    opened = len(running.output_factory.streams)
    running.pipeline.set_output("de", "USB Spare")
    assert len(running.output_factory.streams) == opened
    assert running.pipeline.output_state("de") is DeviceState.DISCONNECTED
    running.pipeline.connect_output("de")
    assert running.output_factory.opened_devices[-1].name == SPARE


def test_p7_set_output_to_a_bad_query_is_a_validation_error(running: Rig) -> None:
    with pytest.raises(ValueError, match="device_query"):
        running.pipeline.set_output("de", "  ")


def test_p7_set_input_device_switches_the_microphone(running: Rig) -> None:
    pipeline = running.pipeline
    old_stream = running.input_factory.current
    pipeline.set_input_device("Backup")
    assert old_stream.closed
    assert pipeline.input_state() is DeviceState.CONNECTED
    device = pipeline.input_device()
    assert device is not None
    assert device.name == MIC_B
    assert pipeline.settings.capture.device_query == "Backup"
    assert pipeline.settings.segmenter.silence_ms == pipeline.settings.capture.silence_ms

    running.say()
    wait_for(lambda: running.counter("lane.de.utterances") == 1, "utterance from the new mic")


def test_p7_set_input_device_with_an_unknown_name_keeps_the_old_microphone(running: Rig) -> None:
    old_stream = running.input_factory.current
    with pytest.raises(DeviceNotFoundError):
        running.pipeline.set_input_device("No Such Mic")
    assert not old_stream.closed
    assert running.pipeline.input_state() is DeviceState.CONNECTED
    assert running.pipeline.settings.capture.device_query == "V380"


def test_p7_set_input_device_while_disconnected_only_stores_the_choice(running: Rig) -> None:
    running.pipeline.disconnect_input()
    running.pipeline.set_input_device("Backup")
    assert len(running.input_factory.streams) == 1
    running.pipeline.connect_input()
    assert running.input_factory.opened_devices[-1].name == MIC_B


# ---------------------------------------------------------------- P8 failed start


def test_p8_a_failed_start_releases_everything(rig: Rig) -> None:
    rig.devices.remove(HS100W)
    with pytest.raises(DeviceNotFoundError):
        rig.pipeline.start()
    assert not rig.pipeline.running
    assert all(stream.closed for stream in rig.output_factory.streams)
    assert not rig.input_factory.streams  # the input is opened last
    with pytest.raises(PipelineStateError):
        rig.pipeline.start()


def test_p8_a_missing_microphone_releases_the_outputs(rig: Rig) -> None:
    rig.devices.remove(MIC)
    with pytest.raises(DeviceNotFoundError):
        rig.pipeline.start()
    assert len(rig.output_factory.streams) == 2
    assert all(stream.closed for stream in rig.output_factory.streams)


# ---------------------------------------------------------------- P9 slow UI consumer


def test_p9_a_stalled_ui_subscriber_does_not_hold_back_the_lanes(running: Rig) -> None:
    ui = running.pipeline.text_bus.subscribe("ui", maxsize=1)  # never read
    for _ in range(3):
        before = running.counter("lane.de.utterances")
        running.say()
        wait_for(lambda before=before: running.counter("lane.de.utterances") == before + 1, "lane")
    assert ui.stats().dropped == 2


# ---------------------------------------------------------------- P10 no disk writes


def test_p10_a_run_with_reconnects_touches_no_disk(no_disk_writes: list[Any]) -> None:
    built = build()
    built.pipeline.start()
    built.say()
    wait_for(lambda: built.counter("lane.de.utterances") == 1, "lane")
    built.pipeline.disconnect_input()
    built.pipeline.connect_input()
    built.pipeline.disconnect_output("fr")
    built.pipeline.set_output("de", "USB Spare")
    built.pipeline.connect_output("fr")
    built.pipeline.stop()
    assert no_disk_writes == []


# ---------------------------------------------------------------- P11 construction


def test_p11_a_single_route_builds_a_single_lane() -> None:
    built = build(routes=(("fr", "HS100W"),))
    built.pipeline.start()
    try:
        assert built.pipeline.targets == ("fr",)
        assert [d.name for d in built.output_factory.opened_devices] == [HS100W]
    finally:
        built.pipeline.stop()


# ---------------------------------------------------------------- P15 translations for the UI


def test_p15_every_lane_publishes_its_translation_on_the_translation_bus(running: Rig) -> None:
    ui = running.pipeline.translation_bus.subscribe("ui", maxsize=8)
    running.say()
    wait_for(lambda: running.counter("lane.de.utterances") == 1, "de lane")
    wait_for(lambda: running.counter("lane.fr.utterances") == 1, "fr lane")
    received = [ui.get(timeout_s=2.0), ui.get(timeout_s=2.0)]
    translations = {item.target: item for item in received if isinstance(item, Translation)}
    assert set(translations) == {"de", "fr"}
    assert translations["de"].text == "[de] hello world"
    assert translations["fr"].text == "[fr] hello world"
    assert translations["de"].utterance_id == translations["fr"].utterance_id == 1
    assert "hello" not in repr(running.pipeline.metrics.snapshot())  # text stays out of metrics
    assert ui.get(timeout_s=0) is None


def test_p15_the_translation_bus_closes_with_the_pipeline(rig: Rig) -> None:
    assert not rig.pipeline.translation_bus.closed
    rig.pipeline.start()
    rig.pipeline.stop()
    assert rig.pipeline.translation_bus.closed


def test_p15_a_stalled_translation_subscriber_does_not_hold_back_the_lanes(running: Rig) -> None:
    ui = running.pipeline.translation_bus.subscribe("ui", maxsize=1)  # never read
    for _ in range(3):
        before = running.counter("lane.de.utterances")
        running.say()
        wait_for(lambda before=before: running.counter("lane.de.utterances") == before + 1, "lane")
        wait_for(lambda before=before: running.counter("lane.fr.utterances") == before + 1, "fr")
    assert ui.stats().dropped == 6 - 1  # two lanes x three utterances into one slot


def test_p15_an_output_that_is_away_publishes_no_translation(running: Rig) -> None:
    # Documented rule: a lane whose output is away skips before translating, so there is
    # no translation to publish for it; the other lane still publishes.
    ui = running.pipeline.translation_bus.subscribe("ui", maxsize=8)
    running.pipeline.disconnect_output("de")
    running.say()
    wait_for(lambda: running.counter("lane.fr.utterances") == 1, "fr lane")
    wait_for(lambda: running.counter("lane.de.skipped_offline") == 1, "de skipped")
    only = ui.get(timeout_s=2.0)
    assert only is not None
    assert only.target == "fr"
    assert ui.get(timeout_s=0) is None


# ---------------------------------------------------------------- P16 change a lane's language


@pytest.fixture
def three_languages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend a third language exists so two lanes can run while a third is a valid choice."""
    monkeypatch.setattr("pipeline.config.SUPPORTED_TARGETS", ("de", "fr", "es"))
    monkeypatch.setattr("pipeline.app.SUPPORTED_TARGETS", ("de", "fr", "es"))


def started(routes: tuple[tuple[str, str], ...]) -> Rig:
    built = build(routes=routes)
    built.pipeline.start()
    return built


def test_p16_a_running_lane_changes_language_on_the_same_output() -> None:
    built = started((("de", "Realtek"),))
    try:
        stream = built.out_streams(REALTEK)[0]
        mic_stream = built.input_factory.current
        built.say()
        wait_for(lambda: built.counter("lane.de.utterances") == 1, "de lane")

        built.pipeline.set_output_target("de", "fr")

        assert built.pipeline.targets == ("fr",)
        assert built.pipeline.settings.targets == ("fr",)
        assert built.pipeline.settings.routes[0].target == "fr"
        assert built.pipeline.settings.routes[0].output.device_query == "Realtek"
        assert built.pipeline.output_state("fr") is DeviceState.CONNECTED
        with pytest.raises(UnknownTargetError):
            built.pipeline.output_state("de")
        names = {t.name for t in threading.enumerate()}
        assert "lane-fr" in names
        assert "lane-de" not in names
        assert set(built.pipeline.bus_stats().text) == {"lane-fr"}

        # Same device binding: the stream is neither closed nor reopened; the mic is untouched.
        assert len(built.out_streams(REALTEK)) == 1
        assert not stream.closed
        assert built.input_factory.current is mic_stream
        assert not mic_stream.closed
        assert len(built.input_factory.streams) == 1

        ui = built.pipeline.translation_bus.subscribe("ui", maxsize=4)
        built.say()
        wait_for(lambda: built.counter("lane.fr.utterances") == 1, "fr lane")
        translation = ui.get(timeout_s=2.0)
        assert translation is not None
        assert translation.target == "fr"
        assert translation.text == "[fr] hello world"
        assert built.counter("lane.de.utterances") == 1  # the old lane did nothing more
        assert built.heard(REALTEK)
        assert built.pipeline.running
    finally:
        built.pipeline.stop()


def test_p16_only_the_changed_lane_is_rebuilt(three_languages: None) -> None:
    built = started((("de", "Realtek"), ("fr", "HS100W")))
    try:
        built.say()
        wait_for(lambda: built.counter("lane.fr.utterances") == 1, "fr lane")
        fr_translator = built.translators["fr"]
        fr_stats = built.pipeline.bus_stats().text["lane-fr"]
        hs_stream = built.out_streams(HS100W)[0]

        built.pipeline.set_output_target("de", "es")

        assert built.pipeline.targets == ("es", "fr")  # same position as the old lane
        assert built.pipeline.settings.targets == ("es", "fr")
        assert built.translators["fr"] is fr_translator  # not rebuilt
        assert built.pipeline.bus_stats().text["lane-fr"] == fr_stats  # counters untouched
        assert not hs_stream.closed
        assert len(built.out_streams(HS100W)) == 1
        device = built.pipeline.output_device("es")
        assert device is not None
        assert device.name == REALTEK

        built.say()
        wait_for(lambda: built.counter("lane.es.utterances") == 1, "es lane")
        wait_for(lambda: built.counter("lane.fr.utterances") == 2, "fr lane again")
        assert built.translators["es"].calls == ["hello world"]
    finally:
        built.pipeline.stop()


def test_p16_the_device_can_still_be_changed_after_the_language_changed(
    three_languages: None,
) -> None:
    built = started((("de", "Realtek"), ("fr", "HS100W")))
    try:
        built.pipeline.set_output_target("de", "es")
        built.pipeline.set_output("es", "USB Spare")
        assert built.pipeline.settings.routes[0].target == "es"
        assert built.pipeline.settings.routes[0].output.device_query == "USB Spare"
        built.say()
        wait_for(lambda: built.counter("lane.es.utterances") == 1, "es lane")
        assert built.heard(SPARE)
    finally:
        built.pipeline.stop()


def test_p16_the_radio_offset_stays_with_the_output() -> None:
    built = build(routes=(("de", "Realtek"),))
    route = built.pipeline.settings.routes[0]
    assert route.radio_offset_ms == 0
    built.pipeline.set_output_target("de", "fr")
    assert built.pipeline.settings.routes[0].radio_offset_ms == route.radio_offset_ms


# ---------------------------------------------------------------- P17 validation


def test_p17_bad_language_choices_are_rejected_and_change_nothing(running: Rig) -> None:
    pipeline = running.pipeline
    before = (pipeline.settings, pipeline.targets, pipeline.bus_stats().text.keys())
    with pytest.raises(ConfigError, match="unsupported"):
        pipeline.set_output_target("de", "xx")
    with pytest.raises(ConfigError, match="source"):
        pipeline.set_output_target("de", "en")
    with pytest.raises(ConfigError, match="already"):
        pipeline.set_output_target("de", "fr")  # fr has its own lane and output
    with pytest.raises(UnknownTargetError, match="es"):
        pipeline.set_output_target("es", "fr")
    assert (pipeline.settings, pipeline.targets, pipeline.bus_stats().text.keys()) == before
    running.say()
    wait_for(lambda: running.counter("lane.de.utterances") == 1, "de lane still works")


def test_p17_choosing_the_current_language_changes_nothing(running: Rig) -> None:
    de_translator = running.translators["de"]
    running.pipeline.set_output_target("de", "de")
    assert running.translators["de"] is de_translator
    assert running.pipeline.targets == ("de", "fr")
    assert running.pipeline.running


# ---------------------------------------------------------------- P18 other situations


def test_p18_a_translator_that_cannot_load_leaves_the_old_lane_working() -> None:
    built = started((("de", "Realtek"),))
    try:
        built.broken_models.add("fr")
        with pytest.raises(RuntimeError, match="model for fr"):
            built.pipeline.set_output_target("de", "fr")
        assert built.pipeline.targets == ("de",)
        assert built.pipeline.settings.targets == ("de",)
        assert "lane-de" in {t.name for t in threading.enumerate()}
        built.say()
        wait_for(lambda: built.counter("lane.de.utterances") == 1, "de lane still works")
    finally:
        built.pipeline.stop()


def test_p18_a_stopped_pipeline_refuses_the_change() -> None:
    built = started((("de", "Realtek"),))
    built.pipeline.stop()
    with pytest.raises(PipelineStateError, match="stopped"):
        built.pipeline.set_output_target("de", "fr")


def test_p18_before_start_the_choice_is_applied_to_the_lane_that_will_start() -> None:
    built = build(routes=(("de", "Realtek"),))
    built.pipeline.set_output_target("de", "fr")
    assert built.pipeline.targets == ("fr",)
    assert not built.output_factory.streams  # nothing is opened before start
    built.pipeline.start()
    try:
        built.say()
        wait_for(lambda: built.counter("lane.fr.utterances") == 1, "fr lane")
        assert built.translators["fr"].calls == ["hello world"]
    finally:
        built.pipeline.stop()


def test_p18_a_disconnected_output_keeps_its_state_across_the_change() -> None:
    built = started((("de", "Realtek"),))
    try:
        built.pipeline.disconnect_output("de")
        built.pipeline.set_output_target("de", "fr")
        assert built.pipeline.output_state("fr") is DeviceState.DISCONNECTED
        built.say()
        wait_for(lambda: built.counter("lane.fr.skipped_offline") == 1, "fr skipped")
        built.pipeline.connect_output("fr")
        built.say()
        wait_for(lambda: built.counter("lane.fr.utterances") == 1, "fr plays")
    finally:
        built.pipeline.stop()


def test_p18_threads_end_with_stop_after_a_language_change() -> None:
    built = started((("de", "Realtek"),))
    built.pipeline.set_output_target("de", "fr")
    built.pipeline.stop()
    remaining = {t.name for t in threading.enumerate()}
    assert not remaining & {"transcribe", "lane-de", "lane-fr", "capture"}


# ---------------------------------------------------------------- P19 lenient start


def test_p19_a_lenient_start_leaves_a_missing_output_disconnected() -> None:
    built = build()
    headset = next(d for d in built.devices.devices if d.name == HS100W)
    built.devices.remove(HS100W)
    try:
        assert built.pipeline.start(lenient=True) == ("fr",)
        assert built.pipeline.running
        assert built.pipeline.output_state("de") is DeviceState.CONNECTED
        assert built.pipeline.output_state("fr") is DeviceState.DISCONNECTED
        assert built.pipeline.input_state() is DeviceState.CONNECTED

        built.say()
        wait_for(lambda: built.counter("lane.de.utterances") == 1, "de lane")
        wait_for(lambda: built.counter("lane.fr.skipped_offline") == 1, "fr skipped")
        assert built.translators["fr"].calls == []

        built.devices.devices.append(headset)
        built.pipeline.connect_output("fr")  # the device came back: the normal call works
        assert built.pipeline.output_state("fr") is DeviceState.CONNECTED
        built.say()
        wait_for(lambda: built.counter("lane.fr.utterances") == 1, "fr lane")
    finally:
        built.pipeline.stop()


def test_p19_start_stays_strict_by_default(rig: Rig) -> None:
    assert rig.pipeline.start() == ()  # nothing was left disconnected
    rig.pipeline.stop()
    other = build()
    other.devices.remove(HS100W)
    with pytest.raises(DeviceNotFoundError):
        other.pipeline.start()
    assert not other.pipeline.running


def test_p19_a_lenient_start_still_needs_the_microphone() -> None:
    built = build()
    built.devices.remove(MIC)
    with pytest.raises(DeviceNotFoundError):
        built.pipeline.start(lenient=True)
    assert not built.pipeline.running
    assert all(stream.closed for stream in built.output_factory.streams)


# ---------------------------------------------------------------- P20 no disk writes


def test_p20_changing_a_language_and_translating_touches_no_disk(
    no_disk_writes: list[Any],
) -> None:
    built = build(routes=(("de", "Realtek"),))
    built.pipeline.start()
    built.say()
    wait_for(lambda: built.counter("lane.de.utterances") == 1, "lane")
    built.pipeline.set_output_target("de", "fr")
    built.say()
    wait_for(lambda: built.counter("lane.fr.utterances") == 1, "lane after the change")
    built.pipeline.stop()
    assert no_disk_writes == []
