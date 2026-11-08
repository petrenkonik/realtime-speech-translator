"""Tk-free view-model of the window, on fake and real pipelines (VM1-VM16)."""

from __future__ import annotations

import ast
import dataclasses
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeClock
from test_app import MIC, Rig, build, settings_for, wait_for

from pipeline.app import BusStats, DeviceState
from pipeline.bus import Bus, ChunkBus
from pipeline.capture import CaptureStats
from pipeline.config import Settings
from pipeline.devices import DeviceInfo
from pipeline.lanes import Translation, TranslationBus
from pipeline.metrics import Metrics
from pipeline.output import OutputStats
from pipeline.transcribe import TextBus, Utterance
from pipeline.ui import model as model_module
from pipeline.ui.model import (
    Health,
    HealthThresholds,
    Snapshot,
    TranscriptRow,
    ViewModel,
    ViewModelError,
    render_report,
)

pytestmark = pytest.mark.unit

SAMPLES_PER_MS = 16
SECRET = "xyzzy-secret-sentence"
SECRET_DE = "geheimer-satz-de"


def capture_stats(**changes: Any) -> CaptureStats:
    base = CaptureStats(
        connected=True,
        callbacks=10,
        input_overflow=0,
        ring_lost_frames=0,
        max_callback_ms=0.2,
        reconnects=0,
        losses=0,
        last_loss_reason="",
        last_error="",
        chunks=0,
        finalization_latency_ms=(),
    )
    return dataclasses.replace(base, **changes)


def output_stats(**changes: Any) -> OutputStats:
    base = OutputStats(
        connected=True,
        callbacks=5,
        underruns=0,
        underrun_frames=0,
        driver_underflows=0,
        frames_played=0,
        dropped_frames=0,
        queued_ms=0.0,
        max_callback_ms=0.1,
        reconnects=0,
        losses=0,
        last_loss_reason="",
        last_error="",
    )
    return dataclasses.replace(base, **changes)


def device(name: str) -> DeviceInfo:
    return DeviceInfo(
        name=name,
        index=1,
        host_api="Windows WASAPI",
        max_input_channels=1,
        max_output_channels=1,
        default_sample_rate_hz=48_000,
    )


class FakePipeline:
    """Everything the view-model reads, scripted by the test; buses and metrics are real."""

    def __init__(self, targets: tuple[str, ...] = ("de", "fr")) -> None:
        self.settings: Settings = settings_for(
            routes=tuple((target, f"Out-{target}") for target in targets)
        )
        self.metrics = Metrics()
        self.text_bus: TextBus = TextBus()
        self.translation_bus: TranslationBus = TranslationBus()
        self.chunk_bus: ChunkBus = Bus()
        self.targets = targets
        self.running = True
        self.position_samples = 0
        self.capture = capture_stats()
        self.outputs = {target: output_stats() for target in targets}
        self.input_dev: DeviceInfo | None = device("Mic One")
        self.input_st = DeviceState.CONNECTED
        self.output_st = {target: DeviceState.CONNECTED for target in targets}
        self.chunk_subs = {"whisper": self.chunk_bus.subscribe("whisper", maxsize=4)}
        self.lane_subs = {t: self.text_bus.subscribe(f"lane-{t}", maxsize=2) for t in targets}

    def capture_stats(self) -> CaptureStats:
        return self.capture

    def output_stats(self, target: str) -> OutputStats:
        return self.outputs[target]

    def bus_stats(self) -> BusStats:
        return BusStats(chunks=self.chunk_bus.stats(), text=self.text_bus.stats())

    def input_device(self) -> DeviceInfo | None:
        return self.input_dev

    def output_device(self, target: str) -> DeviceInfo | None:
        return device(f"Out-{target}") if self.output_st[target] is DeviceState.CONNECTED else None

    def input_state(self) -> DeviceState:
        return self.input_st

    def output_state(self, target: str) -> DeviceState:
        return self.output_st[target]


def utt(
    uid: int,
    start_ms: int,
    end_ms: int,
    *,
    text: str = SECRET,
    finalized_ms: int | None = None,
    decode_ms: float = 120.0,
    reason: str = "silence",
) -> Utterance:
    final = end_ms + 450 if finalized_ms is None else finalized_ms
    return Utterance(
        id=uid,
        chunk_ids=(uid,),
        text=text,
        start_sample=start_ms * SAMPLES_PER_MS,
        speech_end_sample=end_ms * SAMPLES_PER_MS,
        finalized_sample=final * SAMPLES_PER_MS,
        audio_ms=end_ms - start_ms,
        decode_ms=decode_ms,
        reason=reason,  # type: ignore[arg-type]
    )


def tr(
    uid: int,
    target: str = "de",
    *,
    text: str = SECRET_DE,
    latency_ms: float | None = 800.0,
    translate_ms: float = 30.0,
    tts_ms: float | None = 50.0,
) -> Translation:
    return Translation(
        utterance_id=uid,
        target=target,
        text=text,
        translate_ms=translate_ms,
        tts_ms=tts_ms,
        latency_ms=latency_ms,
    )


@pytest.fixture
def fake() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def vm(fake: FakePipeline, clock: FakeClock) -> Iterator[ViewModel]:
    model = ViewModel(fake, clock=clock)
    yield model
    model.close()


# ---------------------------------------------------------------- VM1 snapshot shape


def test_vm1_a_fresh_snapshot_has_cards_and_no_rows(vm: ViewModel) -> None:
    snap = vm.poll()
    assert snap.rows == ()
    assert snap.running
    assert snap.targets == ("de", "fr")
    assert snap.input.device_name == "Mic One"
    assert snap.input.device_query == "V380"
    assert snap.input.state is DeviceState.CONNECTED
    assert snap.input.level is None  # capture counters carry no level; nothing is invented
    assert [card.target for card in snap.outputs] == ["de", "fr"]
    assert snap.output("de").device_name == "Out-de"
    assert snap.output("de").device_query == "Out-de"
    assert snap.output("de").lane.utterances == 0
    assert snap.output("de").lane.latency_p50_ms is None
    assert snap.summary.utterances_seen == 0
    assert snap.summary.latency_mean_ms is None
    assert snap.hide_text is False


def test_vm1_snapshots_are_frozen_and_tuples(vm: ViewModel, fake: FakePipeline) -> None:
    fake.text_bus.publish(utt(1, 0, 500))
    first = vm.poll()
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.running = False  # type: ignore[misc]
    assert isinstance(first.rows, tuple)
    assert isinstance(first.outputs, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.rows[0].text = "x"  # type: ignore[misc]
    fake.text_bus.publish(utt(2, 1000, 1500))
    second = vm.poll()
    assert len(first.rows) == 1  # an old snapshot never changes under the UI
    assert len(second.rows) == 2
    assert second.seq == first.seq + 1


def test_vm1_values_that_cannot_be_right_are_rejected(vm: ViewModel) -> None:
    snap = vm.poll()
    with pytest.raises(ViewModelError, match="pause_ms"):
        dataclasses.replace(
            TranscriptRow(
                utterance_id=1,
                start_s=0.0,
                pause_ms=None,
                audio_ms=1,
                decode_ms=1.0,
                finalize_wait_ms=0,
                reason="silence",
                text=None,
                translations=(),
            ),
            pause_ms=-5,
        )
    with pytest.raises(ViewModelError, match="seq"):
        dataclasses.replace(snap, seq=-1)
    with pytest.raises(ViewModelError, match="max_rows"):
        ViewModel(FakePipeline(), max_rows=0)
    with pytest.raises(ViewModelError, match="warn_hold_s"):
        HealthThresholds(warn_hold_s=0.0)


def test_vm1_the_model_module_does_not_import_tk() -> None:
    # Static scan: a subprocess import would be slow, and sys.modules may hold tkinter from
    # other tests of the window.
    source = Path(model_module.__file__ or "").read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"tkinter", "_tkinter", "Tkinter", "turtle"}


# ---------------------------------------------------------------- VM2 transcript timings


def test_vm2_rows_follow_the_stream_clock(vm: ViewModel, fake: FakePipeline) -> None:
    fake.text_bus.publish(utt(1, 1000, 2500, decode_ms=210.0, reason="silence"))
    fake.text_bus.publish(utt(5, 3000, 4000, finalized_ms=4900, reason="max_length"))
    rows = vm.poll().rows
    assert [row.utterance_id for row in rows] == [1, 5]
    first, second = rows
    assert first.pause_ms is None  # nothing came before
    assert first.start_s == pytest.approx(1.0)
    assert first.audio_ms == 1500
    assert first.decode_ms == 210.0
    assert first.finalize_wait_ms == 450
    assert first.reason == "silence"
    assert second.pause_ms == 500  # 3000 ms start minus 2500 ms end of the row before
    assert second.start_s == pytest.approx(3.0)
    assert second.finalize_wait_ms == 900
    assert second.reason == "max_length"
    assert first.text == SECRET


def test_vm2_a_pause_is_never_negative_and_survives_polls(
    vm: ViewModel, fake: FakePipeline
) -> None:
    fake.text_bus.publish(utt(1, 0, 2000))
    vm.poll()
    fake.text_bus.publish(utt(2, 1900, 2600))  # overlap after a merge: no negative pause
    fake.text_bus.publish(utt(3, 3600, 4000))
    rows = vm.poll().rows
    assert [row.pause_ms for row in rows] == [None, 0, 1000]


# ---------------------------------------------------------------- VM3 translations


def test_vm3_translations_join_by_utterance_id_per_target(
    vm: ViewModel, fake: FakePipeline
) -> None:
    fake.text_bus.publish(utt(1, 0, 500))
    fake.text_bus.publish(utt(2, 1000, 1500))
    fake.translation_bus.publish(tr(2, "fr", text="fr-two", latency_ms=700.0))
    fake.translation_bus.publish(tr(1, "de", text="de-one"))
    rows = vm.poll().rows
    one, two = rows
    assert one.cell("de").text == "de-one"
    assert one.cell("de").arrived
    assert one.cell("de").latency_ms == 800.0
    assert one.cell("de").translate_ms == 30.0
    assert one.cell("de").tts_ms == 50.0
    assert not one.cell("fr").arrived  # pending
    assert one.cell("fr").text is None
    assert two.cell("fr").text == "fr-two"
    assert not two.cell("de").arrived
    assert [cell.target for cell in one.translations] == ["de", "fr"]


def test_vm3_a_late_translation_updates_its_row(vm: ViewModel, fake: FakePipeline) -> None:
    fake.text_bus.publish(utt(1, 0, 500))
    before = vm.poll()
    assert not before.rows[0].cell("de").arrived
    fake.translation_bus.publish(tr(1, "de"))
    after = vm.poll()
    assert after.rows[0].cell("de").text == SECRET_DE
    assert not before.rows[0].cell("de").arrived  # the earlier snapshot stays as it was
    assert after.summary.translations_received == 1


def test_vm3_a_translation_seen_before_its_utterance_waits_for_it(
    vm: ViewModel, fake: FakePipeline
) -> None:
    fake.translation_bus.publish(tr(7, "de"))
    assert vm.poll().rows == ()
    fake.text_bus.publish(utt(7, 0, 500))
    rows = vm.poll().rows
    assert rows[0].cell("de").arrived


def test_vm3_a_translation_for_a_row_that_was_capped_is_ignored(fake: FakePipeline) -> None:
    model = ViewModel(fake, max_rows=1)
    fake.text_bus.publish(utt(1, 0, 500))
    fake.text_bus.publish(utt(2, 1000, 1500))
    model.poll()
    fake.translation_bus.publish(tr(1, "de"))
    snap = model.poll()
    assert [row.utterance_id for row in snap.rows] == [2]
    assert not snap.rows[0].cell("de").arrived
    model.close()


def test_vm3_a_translation_without_audio_keeps_none_timings(
    vm: ViewModel, fake: FakePipeline
) -> None:
    fake.text_bus.publish(utt(1, 0, 500))
    fake.translation_bus.publish(tr(1, "de", latency_ms=None, tts_ms=None))
    cell = vm.poll().rows[0].cell("de")
    assert cell.arrived
    assert cell.latency_ms is None
    assert cell.tts_ms is None


# ---------------------------------------------------------------- VM4 row cap


def test_vm4_rows_are_capped_dropping_the_oldest(fake: FakePipeline) -> None:
    model = ViewModel(fake, max_rows=3)
    for index in range(6):
        fake.text_bus.publish(utt(index + 1, index * 1000, index * 1000 + 500))
        model.poll()
    snap = model.poll()
    assert [row.utterance_id for row in snap.rows] == [4, 5, 6]
    assert snap.rows[0].pause_ms == 500  # kept from when the row was created
    assert snap.summary.utterances_seen == 6
    assert snap.summary.rows_shown == 3
    assert snap.summary.rows_dropped == 3
    model.close()


# ---------------------------------------------------------------- VM5 hide text


def test_vm5_hide_text_masks_text_and_keeps_timings(vm: ViewModel, fake: FakePipeline) -> None:
    fake.text_bus.publish(utt(1, 0, 500))
    fake.translation_bus.publish(tr(1, "de"))
    shown = vm.poll().rows[0]
    assert shown.text == SECRET
    vm.set_hide_text(True)
    assert vm.hide_text
    snap = vm.poll()
    row = snap.rows[0]
    assert snap.hide_text
    assert row.text is None
    assert row.cell("de").text is None
    assert row.cell("de").arrived  # the dash is for "not arrived", not for "hidden"
    assert row.audio_ms == shown.audio_ms
    assert row.decode_ms == shown.decode_ms
    assert row.cell("de").latency_ms == 800.0
    assert SECRET not in repr(snap)
    assert SECRET_DE not in repr(snap)
    vm.set_hide_text(False)
    assert vm.poll().rows[0].text == SECRET  # the text was only masked, not lost


def test_vm5_rows_arriving_while_hidden_are_masked_from_the_start(
    vm: ViewModel, fake: FakePipeline
) -> None:
    vm.set_hide_text(True)
    fake.text_bus.publish(utt(1, 0, 500))
    fake.translation_bus.publish(tr(1, "fr"))
    row = vm.poll().rows[0]
    assert row.text is None
    assert row.cell("fr").text is None
    assert vm.export_text().count(SECRET) == 0


def test_vm5_text_never_reaches_metrics_or_exception_messages(
    vm: ViewModel, fake: FakePipeline
) -> None:
    fake.text_bus.publish(utt(1, 0, 500))
    fake.translation_bus.publish(tr(1, "de"))
    snap = vm.poll()
    assert SECRET not in repr(fake.metrics.snapshot())
    assert SECRET not in repr(snap.stages)
    assert SECRET not in repr(snap.summary)
    assert SECRET not in repr(snap.buses)
    assert SECRET not in repr(snap.rows[0])  # text fields are out of repr


# ---------------------------------------------------------------- VM6 export


def test_vm6_export_text_has_a_summary_and_a_table(vm: ViewModel, fake: FakePipeline) -> None:
    fake.text_bus.publish(utt(1, 1000, 2500, text="first\tline\nwith breaks"))
    fake.text_bus.publish(utt(2, 3000, 4000))
    fake.translation_bus.publish(tr(1, "de", text="eins"))
    report = vm.export_text()
    lines = report.splitlines()
    assert any(line.startswith("utterances: 2") for line in lines)
    assert any("latency" in line for line in lines)
    header_index = next(i for i, line in enumerate(lines) if line.startswith("#\t"))
    header = lines[header_index].split("\t")
    assert header[:8] == [
        "#",
        "start_s",
        "pause_ms",
        "audio_ms",
        "decode_ms",
        "finalize_wait_ms",
        "reason",
        "text",
    ]
    assert "de" in header
    assert "fr" in header
    rows = [line.split("\t") for line in lines[header_index + 1 :] if line]
    assert len(rows) == 2
    assert all(len(cells) == len(header) for cells in rows)  # no stray tabs or newlines
    assert rows[0][header.index("text")] == "first line with breaks"
    assert rows[0][header.index("de")] == "eins"
    assert rows[0][header.index("fr")] == "-"
    assert rows[0][header.index("pause_ms")] == "-"
    assert rows[1][header.index("pause_ms")] == "500"


def test_vm6_export_respects_hide_text(vm: ViewModel, fake: FakePipeline) -> None:
    fake.text_bus.publish(utt(1, 0, 500))
    fake.translation_bus.publish(tr(1, "de"))
    vm.set_hide_text(True)
    report = vm.export_text()
    assert SECRET not in report
    assert SECRET_DE not in report
    assert "hidden" in report
    assert "800" in report  # the latency is a timing and stays


def test_vm6_render_report_is_a_pure_function_of_the_snapshot(
    vm: ViewModel, fake: FakePipeline
) -> None:
    fake.text_bus.publish(utt(1, 0, 500))
    snap = vm.poll()
    assert render_report(snap) == render_report(snap)


# ---------------------------------------------------------------- VM7 cards and counters


def test_vm7_cards_show_device_state_and_lane_counters(vm: ViewModel, fake: FakePipeline) -> None:
    metrics = fake.metrics
    metrics.counter("lane.de.utterances").inc(4)
    metrics.counter("lane.de.errors").inc(1)
    metrics.counter("lane.de.skipped_offline").inc(2)
    for value in (100.0, 200.0, 300.0, 400.0):
        metrics.histogram("lane.de.latency_ms").record(value)
    fake.output_st["fr"] = DeviceState.RECONNECTING
    fake.outputs["fr"] = output_stats(connected=False, losses=1, last_error="device gone")
    snap = vm.poll()
    de = snap.output("de")
    assert (de.lane.utterances, de.lane.errors, de.lane.skipped_offline) == (4, 1, 2)
    assert de.lane.latency_p50_ms == 200.0
    assert de.lane.latency_p95_ms == 400.0
    assert de.state is DeviceState.CONNECTED
    fr = snap.output("fr")
    assert fr.state is DeviceState.RECONNECTING
    assert fr.device_name is None
    assert fr.losses == 1
    assert fr.last_error == "device gone"
    assert fr.lane.utterances == 0


def test_vm7_stage_health_numbers_come_from_the_pipeline(vm: ViewModel, fake: FakePipeline) -> None:
    fake.capture = capture_stats(
        callbacks=99, input_overflow=2, ring_lost_frames=320, reconnects=1, losses=1, chunks=7
    )
    fake.metrics.counter("transcribe.errors").inc(3)
    fake.metrics.counter("transcribe.utterances").inc(5)
    for value in (100.0, 300.0):
        fake.metrics.histogram("transcribe.decode_ms").record(value)
    for value in (0.1, 0.3):
        fake.metrics.histogram("transcribe.rtf").record(value)
    fake.outputs["de"] = output_stats(underruns=4, driver_underflows=1)
    snap = vm.poll()
    assert snap.input.callbacks == 99
    assert snap.input.input_overflow == 2
    assert snap.input.ring_lost_frames == 320
    assert (snap.input.reconnects, snap.input.losses, snap.input.chunks) == (1, 1, 7)
    assert snap.transcribe.errors == 3
    assert snap.transcribe.utterances == 5
    assert snap.transcribe.decode_p50_ms == 100.0
    assert snap.transcribe.decode_p95_ms == 300.0
    assert snap.transcribe.rtf_p95 == 0.3
    assert snap.output("de").underruns == 4
    assert snap.output("de").driver_underflows == 1
    names = {(row.bus, row.name) for row in snap.buses}
    assert {("chunks", "whisper"), ("text", "lane-de"), ("text", "lane-fr")} <= names
    assert any(
        row.bus == "translations" and row.name.startswith("view-model") for row in snap.buses
    )


# ---------------------------------------------------------------- VM8 health colours


def health(snap: Snapshot, key: str) -> Health:
    return snap.stage(key).health


def test_vm8_stage_keys_follow_the_signal_path(vm: ViewModel) -> None:
    keys = [stage.key for stage in vm.poll().stages]
    assert keys == ["input", "transcribe", "translate:de", "output:de", "translate:fr", "output:fr"]


def test_vm8_a_stopped_pipeline_is_idle_everywhere(vm: ViewModel, fake: FakePipeline) -> None:
    fake.running = False
    fake.input_st = DeviceState.DISCONNECTED
    snap = vm.poll()
    assert {stage.health for stage in snap.stages} == {Health.IDLE}


def test_vm8_a_healthy_pipeline_is_ok_once_it_has_work(vm: ViewModel, fake: FakePipeline) -> None:
    fake.metrics.counter("transcribe.utterances").inc()
    fake.metrics.histogram("transcribe.decode_ms").record(100.0)
    fake.metrics.histogram("transcribe.rtf").record(0.2)
    for target in ("de", "fr"):
        fake.metrics.counter(f"lane.{target}.utterances").inc()
    snap = vm.poll()
    assert {stage.health for stage in snap.stages} == {Health.OK}


def test_vm8_before_any_speech_the_work_stages_are_idle(vm: ViewModel) -> None:
    snap = vm.poll()
    assert health(snap, "input") is Health.OK
    assert health(snap, "transcribe") is Health.IDLE
    assert health(snap, "translate:de") is Health.IDLE
    assert health(snap, "output:de") is Health.OK


def test_vm8_disconnected_is_bad_and_reconnecting_is_warn(
    vm: ViewModel, fake: FakePipeline
) -> None:
    fake.input_st = DeviceState.DISCONNECTED
    fake.output_st["de"] = DeviceState.DISCONNECTED
    fake.output_st["fr"] = DeviceState.RECONNECTING
    snap = vm.poll()
    assert health(snap, "input") is Health.BAD
    assert health(snap, "output:de") is Health.BAD
    assert health(snap, "output:fr") is Health.WARN
    # A lane whose output is away only skips; that is not a translation fault.
    assert health(snap, "translate:de") is Health.IDLE
    fake.input_st = DeviceState.RECONNECTING
    assert health(vm.poll(), "input") is Health.WARN


def test_vm8_a_new_overflow_warns_for_the_hold_time_then_clears(
    vm: ViewModel, fake: FakePipeline, clock: FakeClock
) -> None:
    fake.capture = capture_stats(input_overflow=3)  # history from before the window opened
    assert health(vm.poll(), "input") is Health.OK
    fake.capture = capture_stats(input_overflow=4)
    clock.advance(1.0)
    snap = vm.poll()
    assert health(snap, "input") is Health.WARN
    assert "overflow" in snap.stage("input").detail
    clock.advance(HealthThresholds().warn_hold_s - 2.0)
    assert health(vm.poll(), "input") is Health.WARN
    clock.advance(2.5)
    assert health(vm.poll(), "input") is Health.OK
    fake.capture = capture_stats(input_overflow=4, ring_lost_frames=10)
    assert health(vm.poll(), "input") is Health.WARN


def test_vm8_a_new_underrun_warns_the_output(
    vm: ViewModel, fake: FakePipeline, clock: FakeClock
) -> None:
    vm.poll()
    fake.outputs["fr"] = output_stats(underruns=1)
    snap = vm.poll()
    assert health(snap, "output:fr") is Health.WARN
    assert health(snap, "output:de") is Health.OK
    assert "underrun" in snap.stage("output:fr").detail
    clock.advance(HealthThresholds().warn_hold_s + 1.0)
    assert health(vm.poll(), "output:fr") is Health.OK


def test_vm8_transcribe_warns_on_new_errors_and_slow_decodes(
    vm: ViewModel, fake: FakePipeline
) -> None:
    fake.metrics.counter("transcribe.utterances").inc()
    vm.poll()
    fake.metrics.counter("transcribe.errors").inc()
    assert health(vm.poll(), "transcribe") is Health.WARN

    other = FakePipeline()
    other.metrics.counter("transcribe.utterances").inc()
    other.metrics.histogram("transcribe.decode_ms").record(5000.0)
    other.metrics.histogram("transcribe.rtf").record(1.4)
    model = ViewModel(other)
    snap = model.poll()
    assert health(snap, "transcribe") is Health.WARN
    assert "real time" in snap.stage("transcribe").detail
    model.close()


def test_vm8_a_lane_that_falls_behind_or_fails_warns(vm: ViewModel, fake: FakePipeline) -> None:
    fake.metrics.counter("lane.de.utterances").inc()
    vm.poll()
    fake.metrics.counter("lane.de.errors").inc()
    for index in range(5):  # the lane's queue holds two: three are dropped for fr
        fake.text_bus.publish(utt(index + 1, index * 1000, index * 1000 + 500))
    snap = vm.poll()
    assert health(snap, "translate:de") is Health.WARN
    assert health(snap, "translate:fr") is Health.WARN
    assert "behind" in snap.stage("translate:fr").detail


def test_vm8_details_never_carry_text(vm: ViewModel, fake: FakePipeline) -> None:
    fake.text_bus.publish(utt(1, 0, 500))
    fake.capture = capture_stats(input_overflow=1, last_error="no device")
    snap = vm.poll()
    assert all(SECRET not in stage.detail for stage in snap.stages)


# ---------------------------------------------------------------- VM9 summary


def test_vm9_summary_counts_latency_drops_and_errors(vm: ViewModel, fake: FakePipeline) -> None:
    fake.text_bus.publish(utt(1, 0, 500))
    fake.text_bus.publish(utt(2, 1000, 1500))
    for uid, value in ((1, 100.0), (2, 300.0)):
        fake.translation_bus.publish(tr(uid, "de", latency_ms=value))
    fake.translation_bus.publish(tr(1, "fr", latency_ms=None))  # no audio, no latency
    fake.capture = capture_stats(input_overflow=2, ring_lost_frames=5)
    fake.outputs["de"] = output_stats(underruns=3)
    fake.metrics.counter("transcribe.errors").inc(2)
    fake.metrics.counter("lane.fr.errors").inc(1)
    for index in range(5):  # overflows the size-2 lane queues: 3 + 3 dropped
        fake.text_bus.publish(utt(10 + index, 5000 + index * 100, 5050 + index * 100))
    summary = vm.poll().summary
    assert summary.utterances_seen == 7
    assert summary.translations_received == 3
    assert summary.latency_count == 2
    assert summary.latency_mean_ms == pytest.approx(200.0)
    assert summary.latency_p95_ms == 300.0
    assert summary.input_overflow == 2
    assert summary.ring_lost_frames == 5
    assert summary.underruns == 3
    assert summary.errors == 3
    assert (
        summary.bus_dropped
        == fake.lane_subs["de"].stats().dropped + fake.lane_subs["fr"].stats().dropped
    )


# ---------------------------------------------------------------- VM10 slow or absent view


def test_vm10_a_never_polled_view_model_holds_back_nobody(fake: FakePipeline) -> None:
    model = ViewModel(fake, text_queue=2, translation_queue=2)
    started = time.monotonic()
    for index in range(2000):
        fake.text_bus.publish(utt(index + 1, index * 100, index * 100 + 50))
        fake.translation_bus.publish(tr(index + 1, "de"))
    assert time.monotonic() - started < 2.0  # publishing never waited for the reader
    assert fake.lane_subs["de"].stats().dropped == 1998  # its own bound, as before
    snap = model.poll()
    assert len(snap.rows) == 2  # only the newest two survived in the view-model's own queue
    assert snap.summary.view_dropped >= 2 * 1998
    assert [row.utterance_id for row in snap.rows] == [1999, 2000]
    model.close()


def test_vm10_one_poll_drains_a_bounded_amount(fake: FakePipeline) -> None:
    model = ViewModel(fake, drain_cap=10, text_queue=100)
    for index in range(35):
        fake.text_bus.publish(utt(index + 1, index * 100, index * 100 + 50))
    assert len(model.poll().rows) == 10
    assert len(model.poll().rows) == 20
    assert len(model.poll().rows) == 30
    assert len(model.poll().rows) == 35
    model.close()


def test_vm10_poll_is_fast_with_a_full_transcript(fake: FakePipeline) -> None:
    model = ViewModel(fake, max_rows=500, text_queue=600)
    for index in range(500):
        fake.text_bus.publish(utt(index + 1, index * 1000, index * 1000 + 500))
    model.poll()
    started = time.monotonic()
    for _ in range(20):
        model.poll()
    assert (time.monotonic() - started) / 20 < 0.05  # well inside a 100 ms UI tick
    model.close()


# ---------------------------------------------------------------- VM11 close


def test_vm11_close_frees_both_subscriptions(fake: FakePipeline) -> None:
    before_text = set(fake.text_bus.stats())
    before_translation = set(fake.translation_bus.stats())
    model = ViewModel(fake)
    assert set(fake.text_bus.stats()) - before_text
    assert set(fake.translation_bus.stats()) - before_translation
    model.close()
    assert set(fake.text_bus.stats()) == before_text
    assert set(fake.translation_bus.stats()) == before_translation
    model.close()  # idempotent
    fake.text_bus.publish(utt(1, 0, 500))  # nobody is listening any more; nothing breaks
    assert fake.text_bus.published == 1


def test_vm11_two_view_models_get_distinct_names(fake: FakePipeline) -> None:
    first = ViewModel(fake)
    second = ViewModel(fake)
    names = [n for n in fake.text_bus.stats() if n.startswith("view-model")]
    assert len(names) == 2
    first.close()
    third = ViewModel(fake)  # a freed name may come back; it must not collide
    assert len([n for n in fake.text_bus.stats() if n.startswith("view-model")]) == 2
    second.close()
    third.close()


def test_vm11_poll_after_close_returns_the_rows_it_has(fake: FakePipeline) -> None:
    model = ViewModel(fake)
    fake.text_bus.publish(utt(1, 0, 500))
    assert len(model.poll().rows) == 1
    model.close()
    fake.text_bus.publish(utt(2, 1000, 1500))
    assert len(model.poll().rows) == 1  # no more intake after close


def test_vm11_a_closed_bus_is_reported_clearly() -> None:
    pipeline = FakePipeline()
    pipeline.text_bus.close()
    with pytest.raises(ViewModelError, match="closed"):
        ViewModel(pipeline)
    assert not [n for n in pipeline.translation_bus.stats() if n.startswith("view-model")]


# ---------------------------------------------------------------- VM12 real pipeline


@pytest.fixture
def running() -> Iterator[Rig]:
    built = build()
    built.pipeline.start()
    yield built
    built.pipeline.stop()


def test_vm12_a_real_utterance_becomes_a_row_with_both_translations(running: Rig) -> None:
    model = ViewModel(running.pipeline)
    running.say()
    wait_for(lambda: running.counter("lane.de.utterances") == 1, "de lane")
    wait_for(lambda: running.counter("lane.fr.utterances") == 1, "fr lane")
    deadline = time.monotonic() + 5.0
    snap = model.poll()
    while time.monotonic() < deadline and not (
        snap.rows and all(cell.arrived for cell in snap.rows[0].translations)
    ):
        time.sleep(0.01)
        snap = model.poll()
    row = snap.rows[0]
    assert row.text == "hello world"
    assert row.cell("de").text == "[de] hello world"
    assert row.cell("fr").text == "[fr] hello world"
    assert row.pause_ms is None
    assert row.audio_ms > 0
    assert row.finalize_wait_ms >= 0
    assert snap.summary.utterances_seen == 1
    assert snap.summary.latency_count == 2
    assert snap.input.state is DeviceState.CONNECTED
    assert snap.input.device_name == MIC
    assert snap.output("de").lane.utterances == 1
    assert snap.output("de").lane.latency_p50_ms is not None
    assert snap.transcribe.utterances == 1
    assert {stage.health for stage in snap.stages} == {Health.OK}
    assert any(r.bus == "text" and r.name.startswith("view-model") for r in snap.buses)
    model.close()


def test_vm12_a_second_utterance_gets_a_pause(running: Rig) -> None:
    model = ViewModel(running.pipeline)
    for expected in (1, 2):
        running.say()
        wait_for(lambda expected=expected: running.counter("lane.de.utterances") == expected, "de")
    deadline = time.monotonic() + 5.0
    snap = model.poll()
    while len(snap.rows) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
        snap = model.poll()
    assert snap.rows[0].pause_ms is None
    second = snap.rows[1].pause_ms
    assert second is not None
    assert second >= 300  # about the silence the fake microphone fed between the two
    model.close()


# ---------------------------------------------------------------- VM13 device states


def test_vm13_input_disconnect_and_reconnect_show_in_the_card_and_health(running: Rig) -> None:
    model = ViewModel(running.pipeline)
    assert model.poll().input.state is DeviceState.CONNECTED
    running.pipeline.disconnect_input()
    snap = model.poll()
    assert snap.input.state is DeviceState.DISCONNECTED
    assert snap.input.device_name is None
    assert snap.stage("input").health is Health.BAD
    running.pipeline.connect_input()
    snap = model.poll()
    assert snap.input.state is DeviceState.CONNECTED
    assert snap.input.device_name == MIC
    assert snap.stage("input").health is Health.OK
    model.close()


def test_vm13_a_lost_input_shows_reconnecting_then_connected(running: Rig) -> None:
    model = ViewModel(running.pipeline)
    mic = next(d for d in running.devices.devices if d.name == MIC)
    running.devices.remove(MIC)
    running.input_factory.current.fail()
    wait_for(lambda: model.poll().input.state is DeviceState.RECONNECTING, "reconnecting")
    snap = model.poll()
    assert snap.stage("input").health is Health.WARN
    assert snap.output("de").state is DeviceState.CONNECTED
    running.devices.devices.append(mic)
    running.clock.advance(1.5)
    wait_for(lambda: model.poll().input.state is DeviceState.CONNECTED, "connected again")
    final = model.poll()
    assert final.input.losses == 1
    assert final.input.reconnects == 1
    model.close()


def test_vm13_an_output_that_is_away_shows_up_and_its_lane_skips(running: Rig) -> None:
    model = ViewModel(running.pipeline)
    running.pipeline.disconnect_output("de")
    running.say()
    wait_for(lambda: running.counter("lane.fr.utterances") == 1, "fr lane")
    wait_for(lambda: running.counter("lane.de.skipped_offline") == 1, "de skipped")
    deadline = time.monotonic() + 5.0
    snap = model.poll()
    while time.monotonic() < deadline and not (snap.rows and snap.rows[0].cell("fr").arrived):
        time.sleep(0.01)
        snap = model.poll()
    assert snap.output("de").state is DeviceState.DISCONNECTED
    assert snap.output("de").lane.skipped_offline == 1
    assert snap.stage("output:de").health is Health.BAD
    row = snap.rows[0]
    assert row.cell("fr").arrived
    assert not row.cell("de").arrived  # an offline output publishes no translation: a dash
    model.close()


# ---------------------------------------------------------------- VM14 language change


def test_vm14_a_language_change_is_reflected_in_cards_stages_and_rows() -> None:
    built = build(routes=(("de", "Realtek"),))
    built.pipeline.start()
    try:
        model = ViewModel(built.pipeline)
        built.say()
        wait_for(lambda: built.counter("lane.de.utterances") == 1, "de lane")
        snap = model.poll()
        while not (snap.rows and snap.rows[0].cell("de").arrived):
            time.sleep(0.01)
            snap = model.poll()
        assert [card.target for card in snap.outputs] == ["de"]

        built.pipeline.set_output_target("de", "fr")
        built.say()
        wait_for(lambda: built.counter("lane.fr.utterances") == 1, "fr lane")
        deadline = time.monotonic() + 5.0
        snap = model.poll()
        while time.monotonic() < deadline and not (
            len(snap.rows) == 2 and snap.rows[1].cell("fr").arrived
        ):
            time.sleep(0.01)
            snap = model.poll()
        assert snap.targets == ("fr",)
        assert [card.target for card in snap.outputs] == ["fr"]
        assert snap.output("fr").state is DeviceState.CONNECTED
        assert snap.output("fr").device_query == "Realtek"
        assert snap.output("fr").lane.utterances == 1
        assert [stage.key for stage in snap.stages if stage.key.startswith("output")] == [
            "output:fr"
        ]
        first, second = snap.rows
        assert [cell.target for cell in first.translations] == ["de"]  # history stays as it was
        assert first.cell("de").arrived
        assert [cell.target for cell in second.translations] == ["fr"]
        assert second.cell("fr").text == "[fr] hello world"
        assert {r.name for r in snap.buses if r.bus == "text"} >= {"lane-fr"}
        assert "lane-de" not in {r.name for r in snap.buses}
        model.close()
    finally:
        built.pipeline.stop()


# ---------------------------------------------------------------- VM15 close on a real pipeline


def test_vm15_close_frees_subscriptions_of_a_real_pipeline(running: Rig) -> None:
    text_before = set(running.pipeline.text_bus.stats())
    translation_before = set(running.pipeline.translation_bus.stats())
    model = ViewModel(running.pipeline)
    model.close()
    assert set(running.pipeline.text_bus.stats()) == text_before
    assert set(running.pipeline.translation_bus.stats()) == translation_before
    running.say()  # the pipeline keeps working without a window
    wait_for(lambda: running.counter("lane.de.utterances") == 1, "de lane")


def test_vm15_a_stalled_view_model_does_not_slow_a_real_pipeline(running: Rig) -> None:
    model = ViewModel(running.pipeline, text_queue=1, translation_queue=1)  # never polled
    for expected in (1, 2, 3):
        running.say()
        wait_for(lambda expected=expected: running.counter("lane.de.utterances") == expected, "de")
        wait_for(lambda expected=expected: running.counter("lane.fr.utterances") == expected, "fr")
    snap = model.poll()
    assert snap.summary.view_dropped > 0
    model.close()


# ---------------------------------------------------------------- VM16 no disk writes


def test_vm16_a_session_with_polls_and_export_touches_no_disk(no_disk_writes: list[Any]) -> None:
    built = build()
    built.pipeline.start()
    model = ViewModel(built.pipeline)
    built.say()
    wait_for(lambda: built.counter("lane.de.utterances") == 1, "lane")
    model.poll()
    model.set_hide_text(True)
    model.export_text()
    built.pipeline.disconnect_input()
    built.pipeline.set_output_target("de", "de")
    model.poll()
    model.close()
    built.pipeline.stop()
    assert no_disk_writes == []
