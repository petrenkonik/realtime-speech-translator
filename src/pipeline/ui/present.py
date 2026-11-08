"""What the window shows, as plain strings and small records: no Tk, so it is tested directly.

The widgets in ``app_tk.py`` only place what this module produces. Rules kept here:

* A colour is never the only signal: every health and device state also has a word.
* Time comes from the stream sample counter (``TranscriptRow.start_s``); no wall clock.
* Text of an utterance appears only in the transcript cells, where ``hide_text`` has already
  replaced it with ``None`` (shown as ``[hidden]``). Stage cards and the summary carry numbers
  only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from pipeline.app import DeviceState
from pipeline.config import SOURCE_LANGUAGE, SUPPORTED_TARGETS, is_unselected
from pipeline.devices import WASAPI_HOST_API, DeviceInfo
from pipeline.ui.model import Health, Snapshot, TranscriptRow

StatusKind = Literal["ok", "warn", "off"]

HIDDEN_TEXT = "[hidden]"
PENDING = "-"

HEALTH_WORDS: dict[Health, str] = {
    Health.OK: "OK",
    Health.WARN: "Warning",
    Health.BAD: "Problem",
    Health.IDLE: "Idle",
}
"""The word shown next to every stage colour."""

LANGUAGE_NAMES = {"en": "English", "de": "German", "fr": "French"}


# --- devices ---------------------------------------------------------------------------------


def _names(devices: Iterable[DeviceInfo], *, inputs: bool) -> tuple[str, ...]:
    found = {
        device.name
        for device in devices
        if device.host_api == WASAPI_HOST_API
        and (device.max_input_channels if inputs else device.max_output_channels) > 0
    }
    return tuple(sorted(found, key=str.casefold))


def input_device_names(devices: Iterable[DeviceInfo]) -> tuple[str, ...]:
    """Names of the WASAPI devices that can record, each once, in alphabetical order."""
    return _names(devices, inputs=True)


def output_device_names(devices: Iterable[DeviceInfo]) -> tuple[str, ...]:
    """Names of the WASAPI devices that can play, each once, in alphabetical order."""
    return _names(devices, inputs=False)


def display_device(query: str, names: Sequence[str]) -> str:
    """What the device box shows for a configured name substring.

    The device the query finds (exact name first, then a single substring match); otherwise
    the query itself, so a configured but absent device stays visible. Nothing chosen yet
    shows an empty box.
    """
    if is_unselected(query):
        return ""
    wanted = query.strip().casefold()
    exact = [name for name in names if name.casefold() == wanted]
    if len(exact) == 1:
        return exact[0]
    partial = [name for name in names if wanted in name.casefold()]
    return partial[0] if len(partial) == 1 else query


@dataclass(frozen=True, slots=True)
class StatusView:
    """A status dot and its word.

    Attributes:
        text: The word shown next to the dot.
        kind: ``ok`` (green), ``warn`` (amber) or ``off`` (grey, hollow).
    """

    text: str
    kind: StatusKind


def device_status(state: DeviceState) -> StatusView:
    """The dot and word for a device state."""
    if state is DeviceState.CONNECTED:
        return StatusView("Connected", "ok")
    if state is DeviceState.RECONNECTING:
        return StatusView("Reconnecting", "warn")
    return StatusView("Not connected", "off")


@dataclass(frozen=True, slots=True)
class ButtonView:
    """Label and enabled flag of a button."""

    text: str
    enabled: bool


def connect_button(state: DeviceState, *, busy: bool, can_connect: bool) -> ButtonView:
    """The Connect / Disconnect button of a device.

    Args:
        state: Current state of the device.
        busy: A control call is in progress; every button waits.
        can_connect: Connecting is possible (a device is chosen and, for an output, the
            pipeline is running). Disconnecting never needs it.
    """
    if state is DeviceState.DISCONNECTED:
        return ButtonView("Connect", enabled=can_connect and not busy)
    return ButtonView("Disconnect", enabled=not busy)  # also stops retrying a lost device


# --- languages -------------------------------------------------------------------------------


def language_options(
    target: str,
    targets: Sequence[str],
    *,
    source: str = SOURCE_LANGUAGE,
    supported: Sequence[str] = SUPPORTED_TARGETS,
) -> tuple[str, ...]:
    """Languages an output may switch to: supported, not the source, not used by another lane."""
    return tuple(
        code for code in supported if code != source and (code == target or code not in targets)
    )


def language_label(code: str) -> str:
    """``de`` -> ``de - German``; an unknown code stays as it is."""
    name = LANGUAGE_NAMES.get(code)
    return code if name is None else f"{code} - {name}"


def language_code(label: str) -> str:
    """Inverse of ``language_label``."""
    return label.split(" - ", 1)[0].strip()


# --- process strip ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageCardView:
    """One box of the process strip.

    Attributes:
        key: Stage key of the snapshot (``input``, ``transcribe``, ``translate:de``, ...).
        title: Stage name.
        health: Colour class.
        word: The health in words (``OK``, ``Warning``, ``Problem``, ``Idle``).
        detail: The reason the view-model gives (numbers only).
        metrics: Latency, drops, overflows and underruns of the stage.
    """

    key: str
    title: str
    health: Health
    word: str
    detail: str
    metrics: str


def _ms(value: float | None) -> str:
    return PENDING if value is None else f"{value:.0f}"


def _dropped(snapshot: Snapshot, bus: str, name: str) -> int:
    return sum(row.dropped for row in snapshot.buses if row.bus == bus and row.name == name)


def _metrics_for(snapshot: Snapshot, key: str) -> str:
    kind, _, target = key.partition(":")
    if kind == "input":
        card = snapshot.input
        return (
            f"overflows {card.input_overflow} | lost frames {card.ring_lost_frames} "
            f"| reconnects {card.reconnects}"
        )
    if kind == "transcribe":
        stats = snapshot.transcribe
        return (
            f"decode p50/p95 {_ms(stats.decode_p50_ms)}/{_ms(stats.decode_p95_ms)} ms "
            f"| drops {_dropped(snapshot, 'chunks', 'whisper')} | errors {stats.errors}"
        )
    outputs = {card.target: card for card in snapshot.outputs}
    card_out = outputs.get(target)
    if card_out is None:
        return ""
    if kind == "translate":
        lane = card_out.lane
        return (
            f"latency p50/p95 {_ms(lane.latency_p50_ms)}/{_ms(lane.latency_p95_ms)} ms "
            f"| drops {_dropped(snapshot, 'text', f'lane-{target}')} | errors {lane.errors}"
        )
    return f"underruns {card_out.underruns} | dropped frames {card_out.dropped_frames}"


def _language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def stage_title(key: str, label: str) -> str:
    """What a stage does and in which language, for its card.

    Args:
        key: Stage key of the snapshot.
        label: The view-model's short label, used for a key this module does not know.
    """
    kind, _, target = key.partition(":")
    if kind == "input":
        return "Microphone"
    if kind == "transcribe":
        return f"Speech recognition ({_language_name(SOURCE_LANGUAGE)})"
    if kind == "translate" and target:
        name = _language_name(target)
        return f"Translate to {target}" if name == target else f"Translate to {name} ({target})"
    if kind == "output" and target:
        return f"Speak / output {target}"
    return label


def stage_cards(snapshot: Snapshot) -> tuple[StageCardView, ...]:
    """One card per stage of the signal path, in signal order."""
    return tuple(
        StageCardView(
            key=stage.key,
            title=stage_title(stage.key, stage.label),
            health=stage.health,
            word=HEALTH_WORDS[stage.health],
            detail=stage.detail,
            metrics=_metrics_for(snapshot, stage.key),
        )
        for stage in snapshot.stages
    )


@dataclass(frozen=True, slots=True)
class LaneCards:
    """The two boxes of one language lane (a missing one is ``None``).

    Attributes:
        target: Target language code of the lane.
        translate: The translation card.
        output: The speech and output-device card.
    """

    target: str
    translate: StageCardView | None
    output: StageCardView | None


@dataclass(frozen=True, slots=True)
class ProcessLayout:
    """The process strip as a fork: common stages, then lanes that run in parallel.

    Attributes:
        common: Stages every utterance passes once, in signal order (microphone, recognition).
        lanes: One entry per target, in snapshot order. After recognition every lane gets the
            utterance at the same time; no lane waits for another.
    """

    common: tuple[StageCardView, ...]
    lanes: tuple[LaneCards, ...]

    def keys(self) -> tuple[str, ...]:
        """Every card key, common stages first; the window rebuilds only when this changes."""
        keys = [card.key for card in self.common]
        for lane in self.lanes:
            keys.extend(card.key for card in (lane.translate, lane.output) if card is not None)
        return tuple(keys)


def group_stage_cards(cards: Iterable[StageCardView]) -> ProcessLayout:
    """Sort stage cards into the common part and one lane per target.

    Lanes appear in the order their first card appears; a lane with only one of its two cards
    keeps a gap rather than failing.
    """
    common: list[StageCardView] = []
    translate: dict[str, StageCardView] = {}
    output: dict[str, StageCardView] = {}
    order: list[str] = []
    for card in cards:
        kind, _, target = card.key.partition(":")
        if kind in ("translate", "output") and target:
            if target not in order:
                order.append(target)
            (translate if kind == "translate" else output)[target] = card
        else:
            common.append(card)
    lanes = tuple(LaneCards(t, translate.get(t), output.get(t)) for t in order)
    return ProcessLayout(common=tuple(common), lanes=lanes)


def process_layout(snapshot: Snapshot) -> ProcessLayout:
    """The snapshot's stage cards grouped as a fork."""
    return group_stage_cards(stage_cards(snapshot))


Segment = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ForkGeometry:
    """Line segments (x1, y1, x2, y2) of the connector from recognition to the lane rows.

    Attributes:
        stem: Enters from the left at the middle of the bar; ``None`` with no lanes.
        bar: Vertical bar joining the branches; ``None`` with fewer than two lanes.
        branches: One horizontal segment per lane, ending at the lane row.
    """

    stem: Segment | None
    bar: Segment | None
    branches: tuple[Segment, ...]


def fork_geometry(*, width_px: int, lane_ys: Sequence[int]) -> ForkGeometry:
    """Where to draw the fork connector.

    Args:
        width_px: Width of the connector area.
        lane_ys: Vertical centre of every lane row, top to bottom.
    """
    if not lane_ys:
        return ForkGeometry(None, None, ())
    bar_x = width_px // 2
    top, bottom = lane_ys[0], lane_ys[-1]
    middle = (top + bottom) // 2
    bar = (bar_x, top, bar_x, bottom) if len(lane_ys) > 1 else None
    return ForkGeometry(
        stem=(0, middle, bar_x, middle),
        bar=bar,
        branches=tuple((bar_x, y, width_px, y) for y in lane_ys),
    )


def level_percent(level: float | None) -> int | None:
    """The input level as a percentage for a meter; ``None`` when the snapshot has none."""
    return None if level is None else round(level * 100)


# --- transcript ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Column:
    """A column of the transcript table.

    Attributes:
        key: Tk column id.
        heading: Header text.
        width_px: Starting width.
        stretch: Whether the column takes extra room when the window grows.
    """

    key: str
    heading: str
    width_px: int
    stretch: bool


def transcript_columns(targets: Sequence[str]) -> tuple[Column, ...]:
    """Time, pause, text, one translation column per target, then the processing times."""
    return (
        Column("time", "Time", 80, False),
        Column("pause", "Pause", 80, False),
        Column("text", "Text", 300, True),
        *(Column(f"t:{target}", target, 300, True) for target in targets),
        Column("proc", "Processing", 230, False),
    )


def format_stream_time(seconds: float) -> str:
    """Stream time as ``mm:ss.s`` (``h:mm:ss.s`` from one hour on)."""
    tenths = round(seconds * 10)
    whole, tenth = divmod(tenths, 10)
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}.{tenth}"
    return f"{minutes:02d}:{secs:02d}.{tenth}"


def _one_line(text: str | None) -> str:
    return HIDDEN_TEXT if text is None else " ".join(text.split())


def transcript_cells(row: TranscriptRow, targets: Sequence[str]) -> tuple[str, ...]:
    """The cells of one row, in the order of ``transcript_columns(targets)``."""
    cells = [
        format_stream_time(row.start_s),
        PENDING if row.pause_ms is None else f"{row.pause_ms} ms",
        _one_line(row.text),
    ]
    times = [f"ASR {row.decode_ms:.0f} ms"]
    for target in targets:
        cell = row.cell(target)
        cells.append(_one_line(cell.text) if cell.arrived else PENDING)
        times.append(f"{target} {_ms(cell.latency_ms)}{'' if cell.latency_ms is None else ' ms'}")
    cells.append(" | ".join(times))
    return tuple(cells)


def summary_line(snapshot: Snapshot) -> str:
    """The totals under the transcript."""
    summary = snapshot.summary
    return (
        f"Utterances {summary.utterances_seen} | translations {summary.translations_received} "
        f"| latency mean {_ms(summary.latency_mean_ms)} ms, p95 {_ms(summary.latency_p95_ms)} ms "
        f"| drops {summary.bus_dropped} | overflows {summary.input_overflow} "
        f"| lost frames {summary.ring_lost_frames} | underruns {summary.underruns} "
        f"| errors {summary.errors} | stream {format_stream_time(snapshot.position_s)}"
    )
