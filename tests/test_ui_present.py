"""Window presenter: pure formatting of snapshots, device lists and options (WD2-WD4, WD14)."""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
from fakes import make_device
from test_ui_model import SECRET, SECRET_DE, FakePipeline, tr, utt

from pipeline.app import DeviceState
from pipeline.ui import present
from pipeline.ui.model import Health, Snapshot, ViewModel

pytestmark = pytest.mark.unit

WASAPI = "Windows WASAPI"


def snapshot_of(fake: FakePipeline) -> Snapshot:
    """A snapshot of a fake that has nothing published (the view-model starts late)."""
    vm = ViewModel(fake)
    try:
        return vm.poll()
    finally:
        vm.close()


@pytest.fixture
def fake() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def vm(fake: FakePipeline) -> Iterator[ViewModel]:
    """Created before anything is published, so it receives what the test publishes."""
    model = ViewModel(fake)
    yield model
    model.close()


# ---------------------------------------------------------------- WD2 device options


def test_wd2_input_options_are_wasapi_input_devices_only() -> None:
    devices = [
        make_device("Mic B", inputs=2),
        make_device("Speakers", inputs=0, outputs=2),
        make_device("Mic A", inputs=1, outputs=0),
        make_device("Mic MME", inputs=2, host_api="MME"),
        make_device("Both", inputs=2, outputs=2),
    ]
    assert present.input_device_names(devices) == ("Both", "Mic A", "Mic B")
    assert present.output_device_names(devices) == ("Both", "Speakers")


def test_wd2_duplicate_names_are_listed_once() -> None:
    devices = [make_device("Mic", index=1), make_device("Mic", index=7)]
    assert present.input_device_names(devices) == ("Mic",)


def test_wd2_an_empty_device_list_gives_empty_options() -> None:
    assert present.input_device_names([]) == ()
    assert present.output_device_names([]) == ()


def test_wd2_the_displayed_device_is_the_one_the_query_finds() -> None:
    names = ("Microphone (V380 USB)", "Microphone (Backup)", "Speakers (Realtek Audio)")
    assert present.display_device("Microphone (V380 USB)", names) == "Microphone (V380 USB)"
    assert present.display_device("v380", names) == "Microphone (V380 USB)"
    assert present.display_device("Microphone", names) == "Microphone"  # ambiguous: show the query
    assert present.display_device("Gone", names) == "Gone"  # not listed: show what is configured


def test_wd2_an_unselected_device_shows_an_empty_box() -> None:
    from pipeline.config import UNSELECTED_DEVICE

    assert present.display_device(UNSELECTED_DEVICE, ("Mic",)) == ""


def test_wd2_a_state_has_a_word_and_a_kind_so_colour_is_never_alone() -> None:
    connected = present.device_status(DeviceState.CONNECTED)
    reconnecting = present.device_status(DeviceState.RECONNECTING)
    away = present.device_status(DeviceState.DISCONNECTED)
    assert (connected.text, connected.kind) == ("Connected", "ok")
    assert (reconnecting.text, reconnecting.kind) == ("Reconnecting", "warn")
    assert (away.text, away.kind) == ("Not connected", "off")


def test_wd2_the_connect_button_follows_the_state_and_the_busy_flag() -> None:
    cases = [
        (DeviceState.DISCONNECTED, "Connect"),
        (DeviceState.CONNECTED, "Disconnect"),
        (DeviceState.RECONNECTING, "Disconnect"),  # stop retrying
    ]
    for state, label in cases:
        button = present.connect_button(state, busy=False, can_connect=True)
        assert (button.text, button.enabled) == (label, True)
        assert not present.connect_button(state, busy=True, can_connect=True).enabled
    assert not present.connect_button(
        DeviceState.DISCONNECTED, busy=False, can_connect=False
    ).enabled
    # Disconnect is always possible while not busy, even with nothing to connect to.
    assert present.connect_button(DeviceState.CONNECTED, busy=False, can_connect=False).enabled


# ---------------------------------------------------------------- WD3 language options


def test_wd3_a_lane_may_pick_its_own_language_only_when_the_other_is_taken() -> None:
    assert present.language_options("de", ("de", "fr")) == ("de",)
    assert present.language_options("fr", ("de", "fr")) == ("fr",)


def test_wd3_a_single_lane_may_switch_to_any_supported_language() -> None:
    assert present.language_options("de", ("de",)) == ("de", "fr")
    assert present.language_options("fr", ("fr",)) == ("de", "fr")


def test_wd3_the_source_language_is_never_offered() -> None:
    options = present.language_options("de", ("de",), supported=("en", "de", "fr"), source="en")
    assert options == ("de", "fr")


def test_wd3_language_labels_round_trip() -> None:
    assert present.language_label("de") == "de - German"
    assert present.language_label("xx") == "xx"
    assert present.language_code("de - German") == "de"
    assert present.language_code("fr") == "fr"


# ---------------------------------------------------------------- WD4 snapshot formatting


def test_wd4_every_stage_card_names_its_health_in_words() -> None:
    fake = FakePipeline()
    fake.input_st = DeviceState.DISCONNECTED
    fake.output_st["fr"] = DeviceState.RECONNECTING
    cards = {card.key: card for card in present.stage_cards(snapshot_of(fake))}
    assert list(cards) == [
        "input",
        "transcribe",
        "translate:de",
        "output:de",
        "translate:fr",
        "output:fr",
    ]
    assert (cards["input"].health, cards["input"].word) == (Health.BAD, "Problem")
    assert (cards["output:fr"].health, cards["output:fr"].word) == (Health.WARN, "Warning")
    assert (cards["transcribe"].health, cards["transcribe"].word) == (Health.IDLE, "Idle")
    words = {card.word for card in cards.values()}
    assert words <= set(present.HEALTH_WORDS.values())


def test_wd4_healthy_cards_say_ok(fake: FakePipeline, vm: ViewModel) -> None:
    fake.text_bus.publish(utt(1, 1000, 2000))
    fake.metrics.counter("transcribe.utterances").inc()
    fake.metrics.histogram("transcribe.decode_ms").record(100.0)
    cards = {card.key: card for card in present.stage_cards(vm.poll())}
    assert (cards["transcribe"].health, cards["transcribe"].word) == (Health.OK, "OK")
    assert cards["input"].word == "OK"


def test_wd4_cards_carry_latency_drops_overflows_and_underruns() -> None:
    fake = FakePipeline()
    fake.metrics.histogram("lane.de.latency_ms").record(900.0)
    fake.metrics.histogram("lane.de.latency_ms").record(1100.0)
    fake.metrics.histogram("transcribe.decode_ms").record(400.0)
    fake.capture = dataclasses.replace(fake.capture, input_overflow=3, ring_lost_frames=7)
    fake.outputs["de"] = dataclasses.replace(fake.outputs["de"], underruns=2, dropped_frames=5)
    cards = {card.key: card for card in present.stage_cards(snapshot_of(fake))}
    assert "overflows 3" in cards["input"].metrics
    assert "lost frames 7" in cards["input"].metrics
    assert "decode p50/p95 400/400 ms" in cards["transcribe"].metrics
    assert "latency p50/p95" in cards["translate:de"].metrics
    assert "1100" in cards["translate:de"].metrics
    assert "underruns 2" in cards["output:de"].metrics
    assert "dropped frames 5" in cards["output:de"].metrics
    assert "latency p50/p95 -/- ms" in cards["translate:fr"].metrics  # no sample yet


def test_wd4_bus_drops_show_on_the_stage_that_lost_the_items(
    fake: FakePipeline, vm: ViewModel
) -> None:
    for index in range(5):  # the lane queue holds 2, so three are dropped
        fake.text_bus.publish(utt(index, index * 1000, index * 1000 + 500))
    cards = {card.key: card for card in present.stage_cards(vm.poll())}
    assert "drops 3" in cards["translate:de"].metrics


def test_wd4_stream_time_format() -> None:
    assert present.format_stream_time(0.0) == "00:00.0"
    assert present.format_stream_time(12.34) == "00:12.3"
    assert present.format_stream_time(75.0) == "01:15.0"
    assert present.format_stream_time(3725.5) == "1:02:05.5"


def test_wd4_a_row_has_time_pause_text_translations_and_processing(
    fake: FakePipeline, vm: ViewModel
) -> None:
    fake.text_bus.publish(utt(1, 1000, 2000, decode_ms=380.0))
    fake.text_bus.publish(utt(2, 2520, 3500, decode_ms=210.0))
    fake.translation_bus.publish(tr(2, "de", latency_ms=910.0))
    snap = vm.poll()
    first, second = snap.rows
    one = present.transcript_cells(first, snap.targets)
    two = present.transcript_cells(second, snap.targets)
    assert one[:3] == ("00:01.0", "-", SECRET)  # the first row has no pause
    assert two[:3] == ("00:02.5", "520 ms", SECRET)
    assert two[3] == SECRET_DE  # de arrived
    assert two[4] == "-"  # fr is pending
    assert "210 ms" in two[5]
    assert "de 910 ms" in two[5]
    assert "fr -" in two[5]
    assert len(one) == len(present.transcript_columns(snap.targets))


def test_wd4_hidden_text_shows_a_marker_but_keeps_the_timings(
    fake: FakePipeline, vm: ViewModel
) -> None:
    fake.text_bus.publish(utt(1, 1000, 2000, decode_ms=380.0))
    fake.translation_bus.publish(tr(1, "de"))
    vm.set_hide_text(True)
    snap = vm.poll()
    cells = present.transcript_cells(snap.rows[0], snap.targets)
    assert cells[2] == "[hidden]"
    assert cells[3] == "[hidden]"
    assert cells[4] == "-"
    assert "380 ms" in cells[5]
    assert SECRET not in " ".join(cells)
    assert SECRET_DE not in " ".join(cells)


def test_wd4_columns_follow_the_targets() -> None:
    columns = present.transcript_columns(("de", "fr"))
    assert [column.heading for column in columns] == [
        "Time",
        "Pause",
        "Text",
        "de",
        "fr",
        "Processing",
    ]
    assert len({column.key for column in columns}) == len(columns)
    assert [column.key for column in present.transcript_columns(("fr",))][3] == "t:fr"


def test_wd4_summary_line_has_the_totals(fake: FakePipeline, vm: ViewModel) -> None:
    fake.text_bus.publish(utt(1, 0, 1000))
    fake.translation_bus.publish(tr(1, "de", latency_ms=800.0))
    line = present.summary_line(vm.poll())
    assert "Utterances 1" in line
    assert "translations 1" in line
    assert "mean 800 ms" in line
    assert "overflows 0" in line
    assert "underruns 0" in line
    assert SECRET not in line


def test_wd4_summary_without_samples_shows_dashes() -> None:
    line = present.summary_line(snapshot_of(FakePipeline()))
    assert "Utterances 0" in line
    assert "mean - ms" in line


def test_wd4_the_input_level_is_shown_only_when_there_is_one() -> None:
    assert present.level_percent(None) is None
    assert present.level_percent(0.0) == 0
    assert present.level_percent(0.5) == 50
    assert present.level_percent(1.0) == 100


def test_wd4_card_text_never_contains_utterance_text(fake: FakePipeline, vm: ViewModel) -> None:
    fake.text_bus.publish(utt(1, 0, 1000))
    fake.translation_bus.publish(tr(1, "de"))
    snap = vm.poll()
    assert snap.rows  # the text really is in the snapshot
    blob = " ".join(
        f"{card.title} {card.word} {card.detail} {card.metrics}"
        for card in present.stage_cards(snap)
    )
    assert SECRET not in blob
    assert SECRET_DE not in blob


# ---------------------------------------------------------------- WD14 Tk-free


@pytest.mark.parametrize(
    "module",
    ["present", "controller", "settings_form", "wiring", "model", "settings_schema"],
)
def test_wd14_the_logic_modules_import_no_toolkit(module: str) -> None:
    path = Path(present.__file__).with_name(f"{module}.py")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"tkinter", "_tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6"}
