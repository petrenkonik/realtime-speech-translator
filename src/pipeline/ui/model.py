"""View-model of the window: turns a running ``Pipeline`` into immutable snapshots.

No Tk here. A window calls ``ViewModel.poll()`` every 100-250 ms from its own thread and draws
the returned ``Snapshot``; everything is plain data, so the logic is tested without a display.

What it reads:

* its own two subscriptions (unique names, ``drop_oldest``) on the pipeline's ``TextBus`` and
  ``TranslationBus``, drained without blocking and with a cap per poll. A view-model that is
  slow, stalled or never polled loses only its own rows; ``Bus.publish`` never waits for it
  and the lanes never notice (spec P9, P14).
* the pipeline's statistics accessors, which take no lock, and ``Metrics.snapshot()``.

Rules the snapshots follow:

* Time is the stream sample counter (16 kHz samples in the utterances), never a wall clock.
  The only clock is the injected one that ages the warning colours.
* Utterance and translation text lives in RAM only. Dataclasses that hold it keep it out of
  ``repr``; with ``hide_text`` on the text fields of a snapshot are ``None`` and every timing
  stays. No text goes into stage details, metrics or exception messages.
* A translation is joined to its row by ``utterance_id``. A row is created when its utterance
  arrives; a target without a translation yet is a cell with ``arrived=False`` (the window
  shows a dash). A lane whose output is away skips before translating and publishes no
  translation, so such a cell stays pending for good; that is by design, not a lost message.
* The input level is ``None``: capture counters carry no signal level and none is invented.

Health colours (``Health``), judged per stage on every poll. "Recently" means within
``HealthThresholds.warn_hold_s`` seconds of the last increase of a counter, measured with the
injected clock; counters that are already non-zero when the view-model is created are history,
not news.

* ``IDLE``: the pipeline is not running, or the stage has had nothing to do yet.
* ``BAD``: a device the pipeline needs is ``disconnected`` (microphone or output). An output
  the user disconnected on purpose looks the same on purpose; the card says why.
* ``WARN``: a device is ``reconnecting``; or recently a new ``input_overflow``,
  ``ring_lost_frames``, output underrun, driver underflow or dropped output frames, a
  transcription error, a chunk dropped before whisper, a lane error, an utterance dropped for
  a lane (falling behind); or whisper's p95 real-time factor is above
  ``HealthThresholds.rtf_p95_warn``.
* ``OK``: none of the above.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from pipeline.app import BusStats, DeviceState
from pipeline.bus import BusClosedError, BusConfigError, Subscription, SubscriptionStats
from pipeline.capture import CaptureStats
from pipeline.config import Settings
from pipeline.devices import DeviceInfo
from pipeline.lanes import Translation, TranslationBus
from pipeline.metrics import Histogram, Metrics
from pipeline.output import OutputStats
from pipeline.transcribe import TextBus, Utterance
from pipeline.vad import SAMPLE_RATE_HZ

_SAMPLES_PER_MS = SAMPLE_RATE_HZ // 1000
_NAME_PREFIX = "view-model"
_MAX_NAME_ATTEMPTS = 1000
_LATENCY_WINDOW = 1000
_HIDDEN = "[hidden]"
_NONE_CELL = "-"


class ViewModelError(ValueError):
    """A view-model setting or a snapshot value cannot be right; the message names it."""


class PipelineView(Protocol):
    """The read-only part of ``Pipeline`` the view-model uses (tests give it a fake)."""

    @property
    def settings(self) -> Settings:
        """Current settings, including device choices made at run time."""
        ...

    @property
    def targets(self) -> tuple[str, ...]:
        """Target languages in lane order."""
        ...

    @property
    def running(self) -> bool:
        """True between ``start()`` and ``stop()``."""
        ...

    @property
    def position_samples(self) -> int:
        """Stream position in 16 kHz samples."""
        ...

    @property
    def metrics(self) -> Metrics:
        """The shared metrics registry."""
        ...

    @property
    def text_bus(self) -> TextBus:
        """Utterances."""
        ...

    @property
    def translation_bus(self) -> TranslationBus:
        """Translations."""
        ...

    def capture_stats(self) -> CaptureStats:
        """Capture counters."""
        ...

    def output_stats(self, target: str) -> OutputStats:
        """Counters of one output."""
        ...

    def bus_stats(self) -> BusStats:
        """Per-subscriber counters of the chunk and text buses."""
        ...

    def input_device(self) -> DeviceInfo | None:
        """The open microphone, if any."""
        ...

    def output_device(self, target: str) -> DeviceInfo | None:
        """The open output of ``target``, if any."""
        ...

    def input_state(self) -> DeviceState:
        """State of the microphone."""
        ...

    def output_state(self, target: str) -> DeviceState:
        """State of one output."""
        ...


class Health(StrEnum):
    """Colour of one pipeline stage (thresholds in the module docstring)."""

    OK = "ok"
    WARN = "warn"
    BAD = "bad"
    IDLE = "idle"


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    """Limits of the warning colours.

    Attributes:
        warn_hold_s: How long a stage stays ``WARN`` after the last increase of a trouble
            counter (injected-clock seconds).
        rtf_p95_warn: Whisper real-time factor (decode time / audio time) above which the
            transcription stage warns; 1.0 means slower than real time.
    """

    warn_hold_s: float = 10.0
    rtf_p95_warn: float = 1.0

    def __post_init__(self) -> None:
        """Validate the values."""
        if self.warn_hold_s <= 0.0:
            raise ViewModelError(f"warn_hold_s must be positive, got {self.warn_hold_s}")
        if self.rtf_p95_warn <= 0.0:
            raise ViewModelError(f"rtf_p95_warn must be positive, got {self.rtf_p95_warn}")


def _non_negative(owner: str, **values: float | None) -> None:
    for name, value in values.items():
        if value is not None and value < 0:
            raise ViewModelError(f"{owner}: {name} must not be negative, got {value}")


@dataclass(frozen=True, slots=True)
class StageHealth:
    """Colour and a short reason for one stage of the signal path.

    Attributes:
        key: ``input``, ``transcribe``, ``translate:<target>`` or ``output:<target>``.
        label: Text for the window.
        health: The colour.
        detail: A short reason, numbers only (never utterance text).
    """

    key: str
    label: str
    health: Health
    detail: str


@dataclass(frozen=True, slots=True)
class LaneCounters:
    """Counters of one language lane (from ``Metrics``).

    Attributes:
        utterances: Utterances played.
        errors: Utterances that failed in the lane.
        skipped_offline: Utterances skipped because the output was away.
        latency_p50_ms: Median end-to-end latency, ``None`` before the first sample.
        latency_p95_ms: 95th percentile of it, ``None`` before the first sample.
    """

    utterances: int
    errors: int
    skipped_offline: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None

    def __post_init__(self) -> None:
        """Validate the values."""
        _non_negative(
            "LaneCounters",
            utterances=self.utterances,
            errors=self.errors,
            skipped_offline=self.skipped_offline,
            latency_p50_ms=self.latency_p50_ms,
            latency_p95_ms=self.latency_p95_ms,
        )


@dataclass(frozen=True, slots=True)
class InputCard:
    """The microphone for the window.

    Attributes:
        device_name: Name of the open device, ``None`` when none is open.
        device_query: The configured name substring.
        state: Connected, reconnecting or disconnected.
        level: Signal level 0..1, always ``None`` (not derivable from the capture counters).
        callbacks: Audio callbacks so far (summed over connections).
        input_overflow: Callbacks that reported an input overflow.
        ring_lost_frames: Frames the processing thread was too slow to read.
        reconnects: Successful reconnections.
        losses: Times the stream was declared lost.
        chunks: Chunks published.
        last_error: Last failed connection attempt, or ``""``.
    """

    device_name: str | None
    device_query: str
    state: DeviceState
    level: float | None
    callbacks: int
    input_overflow: int
    ring_lost_frames: int
    reconnects: int
    losses: int
    chunks: int
    last_error: str

    def __post_init__(self) -> None:
        """Validate the values."""
        _non_negative(
            "InputCard",
            callbacks=self.callbacks,
            input_overflow=self.input_overflow,
            ring_lost_frames=self.ring_lost_frames,
            reconnects=self.reconnects,
            losses=self.losses,
            chunks=self.chunks,
        )
        if self.level is not None and not 0.0 <= self.level <= 1.0:
            raise ViewModelError(f"InputCard: level must be 0..1 or None, got {self.level}")


@dataclass(frozen=True, slots=True)
class OutputCard:
    """One output (language lane plus device) for the window.

    Attributes:
        target: Language of the lane.
        device_name: Name of the open device, ``None`` when none is open.
        device_query: The configured name substring.
        state: Connected, reconnecting or disconnected.
        lane: The lane's counters.
        underruns: Callbacks that ran dry in the middle of an utterance.
        driver_underflows: Underflows reported by PortAudio.
        dropped_frames: Frames discarded because the device was away or the ring was full.
        reconnects: Successful reconnections.
        losses: Times the stream was declared lost.
        last_error: Last failed connection attempt, or ``""``.
    """

    target: str
    device_name: str | None
    device_query: str
    state: DeviceState
    lane: LaneCounters
    underruns: int
    driver_underflows: int
    dropped_frames: int
    reconnects: int
    losses: int
    last_error: str

    def __post_init__(self) -> None:
        """Validate the values."""
        _non_negative(
            "OutputCard",
            underruns=self.underruns,
            driver_underflows=self.driver_underflows,
            dropped_frames=self.dropped_frames,
            reconnects=self.reconnects,
            losses=self.losses,
        )


@dataclass(frozen=True, slots=True)
class BusRow:
    """Counters of one bus subscriber.

    Attributes:
        bus: ``chunks``, ``text`` or ``translations``.
        name: Subscriber name.
        delivered: Items handed to the consumer.
        dropped: Items lost to the overflow policy.
        depth: Items waiting now.
        max_depth: Highest depth seen.
    """

    bus: str
    name: str
    delivered: int
    dropped: int
    depth: int
    max_depth: int

    def __post_init__(self) -> None:
        """Validate the values."""
        _non_negative(
            "BusRow",
            delivered=self.delivered,
            dropped=self.dropped,
            depth=self.depth,
            max_depth=self.max_depth,
        )


@dataclass(frozen=True, slots=True)
class TranscribeStats:
    """Whisper counters; the percentiles are ``None`` before the first decode.

    Attributes:
        utterances: Utterances published.
        errors: Decodes that raised.
        decode_p50_ms: Median decode time.
        decode_p95_ms: 95th percentile of the decode time.
        rtf_p50: Median real-time factor (decode time / audio time).
        rtf_p95: 95th percentile of it.
    """

    utterances: int
    errors: int
    decode_p50_ms: float | None
    decode_p95_ms: float | None
    rtf_p50: float | None
    rtf_p95: float | None

    def __post_init__(self) -> None:
        """Validate the values."""
        _non_negative(
            "TranscribeStats",
            utterances=self.utterances,
            errors=self.errors,
            decode_p50_ms=self.decode_p50_ms,
            decode_p95_ms=self.decode_p95_ms,
            rtf_p50=self.rtf_p50,
            rtf_p95=self.rtf_p95,
        )


@dataclass(frozen=True, slots=True)
class Summary:
    """Totals for the header of the transcript and of the exported report.

    Attributes:
        utterances_seen: Utterances this view-model received (including capped rows).
        rows_shown: Rows in the snapshot.
        rows_dropped: Rows removed by the ``max_rows`` cap.
        translations_received: Translations this view-model received.
        latency_count: Translations with an end-to-end latency in the window below.
        latency_mean_ms: Mean end-to-end latency of the most recent 1000, or ``None``.
        latency_p95_ms: 95th percentile of the same, or ``None``.
        bus_dropped: Items lost by every subscriber of all three buses.
        view_dropped: The part of ``bus_dropped`` lost by this view-model's own queues.
        input_overflow: Input overflows.
        ring_lost_frames: Frames lost in the capture ring.
        underruns: Output underruns over all outputs.
        errors: Transcription errors plus lane errors.
    """

    utterances_seen: int
    rows_shown: int
    rows_dropped: int
    translations_received: int
    latency_count: int
    latency_mean_ms: float | None
    latency_p95_ms: float | None
    bus_dropped: int
    view_dropped: int
    input_overflow: int
    ring_lost_frames: int
    underruns: int
    errors: int

    def __post_init__(self) -> None:
        """Validate the values."""
        _non_negative(
            "Summary",
            utterances_seen=self.utterances_seen,
            rows_shown=self.rows_shown,
            rows_dropped=self.rows_dropped,
            translations_received=self.translations_received,
            latency_count=self.latency_count,
            latency_mean_ms=self.latency_mean_ms,
            latency_p95_ms=self.latency_p95_ms,
            bus_dropped=self.bus_dropped,
            view_dropped=self.view_dropped,
            input_overflow=self.input_overflow,
            ring_lost_frames=self.ring_lost_frames,
            underruns=self.underruns,
            errors=self.errors,
        )


@dataclass(frozen=True, slots=True)
class TranslationCell:
    """What one target has produced for one row.

    Attributes:
        target: Language code.
        arrived: The translation was published. ``False`` shows a dash; an offline output
            publishes none, so it stays ``False``.
        text: The translation; ``None`` while not arrived or while the text is hidden.
            Out of ``repr``.
        translate_ms: Time in the translator, ``None`` while not arrived.
        tts_ms: Time in speech synthesis, ``None`` if it did not finish or not arrived.
        latency_ms: End-to-end latency to the first audio, ``None`` if there was none.
    """

    target: str
    arrived: bool
    text: str | None = field(default=None, repr=False)
    translate_ms: float | None = None
    tts_ms: float | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        """Validate the values."""
        _non_negative("TranslationCell", translate_ms=self.translate_ms, tts_ms=self.tts_ms)


@dataclass(frozen=True, slots=True)
class TranscriptRow:
    """One utterance with its timings and translations.

    Attributes:
        utterance_id: ``Utterance.id``.
        start_s: Stream time of the start of the utterance, in seconds.
        pause_ms: Silence between the end of the speech of the row before and the start of
            this one; ``None`` for the first row. Never negative.
        audio_ms: Length of the decoded audio.
        decode_ms: Whisper time.
        finalize_wait_ms: From the end of speech to the moment the chunk was cut.
        reason: Why the chunk ended.
        text: The transcript; ``None`` while hidden. Out of ``repr``.
        translations: One cell per target, in lane order.
    """

    utterance_id: int
    start_s: float
    pause_ms: int | None
    audio_ms: int
    decode_ms: float
    finalize_wait_ms: int
    reason: str
    text: str | None = field(repr=False)
    translations: tuple[TranslationCell, ...]

    def __post_init__(self) -> None:
        """Validate the values."""
        _non_negative(
            "TranscriptRow",
            utterance_id=self.utterance_id,
            start_s=self.start_s,
            pause_ms=self.pause_ms,
            audio_ms=self.audio_ms,
            decode_ms=self.decode_ms,
            finalize_wait_ms=self.finalize_wait_ms,
        )

    def cell(self, target: str) -> TranslationCell:
        """Return the cell of ``target``; a target the row never expected is pending."""
        for cell in self.translations:
            if cell.target == target:
                return cell
        return TranslationCell(target=target, arrived=False)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Everything the window draws at one moment; immutable and safe to share.

    Attributes:
        seq: Number of the poll that produced it (0, 1, 2, ...).
        running: The pipeline is running.
        position_s: Stream time in seconds.
        hide_text: Text fields are masked.
        targets: Target languages in lane order.
        input: The microphone card.
        outputs: One card per target.
        buses: Per-subscriber counters of the three buses.
        transcribe: Whisper counters.
        stages: Colour per stage of the signal path.
        summary: Totals.
        rows: The transcript, oldest first.
    """

    seq: int
    running: bool
    position_s: float
    hide_text: bool
    targets: tuple[str, ...]
    input: InputCard
    outputs: tuple[OutputCard, ...]
    buses: tuple[BusRow, ...]
    transcribe: TranscribeStats
    stages: tuple[StageHealth, ...]
    summary: Summary
    rows: tuple[TranscriptRow, ...]

    def __post_init__(self) -> None:
        """Validate the values."""
        _non_negative("Snapshot", seq=self.seq, position_s=self.position_s)

    def output(self, target: str) -> OutputCard:
        """Return the card of ``target``.

        Raises:
            ViewModelError: This snapshot has no such output.
        """
        for card in self.outputs:
            if card.target == target:
                return card
        raise ViewModelError(f"no output for target {target!r}; have {', '.join(self.targets)}")

    def stage(self, key: str) -> StageHealth:
        """Return the stage called ``key``.

        Raises:
            ViewModelError: This snapshot has no such stage.
        """
        for stage in self.stages:
            if stage.key == key:
                return stage
        raise ViewModelError(f"no stage {key!r}; have {', '.join(s.key for s in self.stages)}")


# --- helpers over Metrics ---------------------------------------------------------------


def _count(flat: Mapping[str, float], name: str) -> int:
    return int(flat.get(name, 0.0))


def _percentiles(flat: Mapping[str, float], name: str) -> tuple[float | None, float | None]:
    """Return ``(p50, p95)`` of a histogram, ``(None, None)`` while it has no samples."""
    if flat.get(f"{name}.count", 0.0) <= 0.0:
        return None, None
    return flat.get(f"{name}.p50", 0.0), flat.get(f"{name}.p95", 0.0)


class _Tripwires:
    """Remembers when each trouble counter last increased, to age the warning colours."""

    def __init__(self, hold_s: float) -> None:
        self._hold_s = hold_s
        self._last_value: dict[str, int] = {}
        self._last_increase_s: dict[str, float] = {}

    def observe(self, key: str, value: int, now_s: float) -> bool:
        """Record ``value`` and say whether ``key`` increased within the hold time.

        The first value seen for a key is a baseline, not news.
        """
        previous = self._last_value.get(key)
        self._last_value[key] = value
        if previous is not None and value > previous:
            self._last_increase_s[key] = now_s
        stamp = self._last_increase_s.get(key)
        return stamp is not None and now_s - stamp < self._hold_s


class _Record:
    """One row while it is being assembled; the frozen row is cached until it changes."""

    __slots__ = (
        "audio_ms",
        "decode_ms",
        "expected",
        "finalize_wait_ms",
        "pause_ms",
        "reason",
        "row",
        "start_s",
        "text",
        "translations",
        "utterance_id",
    )

    def __init__(
        self, utterance: Utterance, pause_ms: int | None, expected: tuple[str, ...]
    ) -> None:
        self.utterance_id = utterance.id
        self.start_s = utterance.start_sample / SAMPLE_RATE_HZ
        self.pause_ms = pause_ms
        self.audio_ms = utterance.audio_ms
        self.decode_ms = utterance.decode_ms
        self.finalize_wait_ms = max(
            0, (utterance.finalized_sample - utterance.speech_end_sample) // _SAMPLES_PER_MS
        )
        self.reason = str(utterance.reason)
        self.text = utterance.text
        self.expected = expected
        self.translations: dict[str, Translation] = {}
        self.row: TranscriptRow | None = None

    def frozen(self, hide_text: bool) -> TranscriptRow:
        if self.row is None:
            targets = (*self.expected, *(t for t in self.translations if t not in self.expected))
            self.row = TranscriptRow(
                utterance_id=self.utterance_id,
                start_s=self.start_s,
                pause_ms=self.pause_ms,
                audio_ms=self.audio_ms,
                decode_ms=self.decode_ms,
                finalize_wait_ms=self.finalize_wait_ms,
                reason=self.reason,
                text=None if hide_text else self.text,
                translations=tuple(self._cell(t, hide_text) for t in targets),
            )
        return self.row

    def _cell(self, target: str, hide_text: bool) -> TranslationCell:
        translation = self.translations.get(target)
        if translation is None:
            return TranslationCell(target=target, arrived=False)
        return TranslationCell(
            target=target,
            arrived=True,
            text=None if hide_text else translation.text,
            translate_ms=translation.translate_ms,
            tts_ms=translation.tts_ms,
            latency_ms=translation.latency_ms,
        )


class ViewModel:
    """Polls a pipeline and returns snapshots for the window; owns two bus subscriptions.

    Methods may be called from any thread (they take a private lock that no pipeline thread
    ever takes), but a window normally calls them all from its own thread.
    """

    def __init__(
        self,
        pipeline: PipelineView,
        *,
        max_rows: int = 500,
        thresholds: HealthThresholds | None = None,
        clock: Callable[[], float] = time.monotonic,
        drain_cap: int = 1000,
        text_queue: int = 256,
        translation_queue: int = 512,
    ) -> None:
        """Subscribe to the pipeline's text and translation buses.

        Args:
            pipeline: The pipeline to watch (or a fake with the same accessors).
            max_rows: Rows kept; the oldest are dropped beyond it.
            thresholds: Limits of the warning colours.
            clock: Monotonic seconds that age the warning colours.
            drain_cap: Most items taken from one subscription per ``poll()``, so a poll is
                bounded even after a long stall; the rest is read by the next poll.
            text_queue: Capacity of the utterance subscription (oldest is dropped when full).
            translation_queue: Capacity of the translation subscription.

        Raises:
            ViewModelError: A parameter is out of range, or a bus is already closed (the
                pipeline was stopped; build a new view-model for a new pipeline).
        """
        if max_rows < 1:
            raise ViewModelError(f"max_rows must be at least 1, got {max_rows}")
        if drain_cap < 1:
            raise ViewModelError(f"drain_cap must be at least 1, got {drain_cap}")
        if text_queue < 1 or translation_queue < 1:
            raise ViewModelError("text_queue and translation_queue must be at least 1")
        self._pipeline = pipeline
        self._max_rows = max_rows
        self._thresholds = thresholds or HealthThresholds()
        self._clock = clock
        self._drain_cap = drain_cap
        self._lock = threading.Lock()
        self._texts: Subscription[Utterance]
        self._translations: Subscription[Translation]
        self._texts, self._translations = self._subscribe(text_queue, translation_queue)
        self._closed = False
        self._hide_text = False
        self._seq = 0
        self._records: OrderedDict[int, _Record] = OrderedDict()
        self._orphans: OrderedDict[int, list[Translation]] = OrderedDict()
        self._last_end_sample: int | None = None
        self._utterances_seen = 0
        self._rows_dropped = 0
        self._translations_received = 0
        self._latency = Histogram(_LATENCY_WINDOW)
        self._tripwires = _Tripwires(self._thresholds.warn_hold_s)

    # --- public API ----------------------------------------------------------------

    @property
    def hide_text(self) -> bool:
        """True while text fields are masked in snapshots."""
        return self._hide_text

    def set_hide_text(self, hide: bool) -> None:
        """Mask (or show again) the text of every row from the next ``poll()`` on."""
        with self._lock:
            self._hide_text = hide
            for record in self._records.values():
                record.row = None

    def poll(self) -> Snapshot:
        """Take what arrived since the last call and return the current snapshot.

        Never blocks: both subscriptions are read with a zero timeout and at most
        ``drain_cap`` items each.
        """
        with self._lock:
            targets = tuple(self._pipeline.targets)
            if not self._closed:
                self._drain(targets)
            snapshot = self._build(targets)
            self._seq += 1
            return snapshot

    def export_text(self) -> str:
        """Return the plain-text report (summary and table) for the clipboard.

        The report is built from a fresh poll and honors ``hide_text``. Nothing is written
        anywhere.
        """
        return render_report(self.poll())

    def close(self) -> None:
        """Remove both subscriptions; idempotent. ``poll()`` still returns the rows it has."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pipeline.text_bus.unsubscribe(self._texts)
            self._pipeline.translation_bus.unsubscribe(self._translations)

    # --- intake --------------------------------------------------------------------

    def _subscribe(
        self, text_queue: int, translation_queue: int
    ) -> tuple[Subscription[Utterance], Subscription[Translation]]:
        pipeline = self._pipeline
        for attempt in range(1, _MAX_NAME_ATTEMPTS + 1):
            name = f"{_NAME_PREFIX}-{attempt}"
            try:
                texts = pipeline.text_bus.subscribe(name, maxsize=text_queue)
            except BusConfigError:
                continue  # another view-model has the name
            except BusClosedError as error:
                raise ViewModelError(
                    "the pipeline's text bus is closed (it was stopped); build a new view-model"
                ) from error
            try:
                translations = pipeline.translation_bus.subscribe(name, maxsize=translation_queue)
            except BusConfigError:
                pipeline.text_bus.unsubscribe(texts)
                continue
            except BusClosedError as error:
                pipeline.text_bus.unsubscribe(texts)
                raise ViewModelError(
                    "the pipeline's translation bus is closed (it was stopped); "
                    "build a new view-model"
                ) from error
            return texts, translations
        raise ViewModelError(f"no free subscriber name after {_MAX_NAME_ATTEMPTS} attempts")

    def _drain(self, targets: tuple[str, ...]) -> None:
        # Utterances first: a translation is published after its utterance, so any
        # translation read below belongs to a row that exists or is about to.
        for _ in range(self._drain_cap):
            utterance = self._texts.get(0.0)
            if utterance is None:
                break
            self._add_utterance(utterance, targets)
        for _ in range(self._drain_cap):
            translation = self._translations.get(0.0)
            if translation is None:
                break
            self._add_translation(translation)

    def _add_utterance(self, utterance: Utterance, targets: tuple[str, ...]) -> None:
        if utterance.id in self._records:
            return
        previous_end = self._last_end_sample
        pause_ms = (
            None
            if previous_end is None
            else max(0, (utterance.start_sample - previous_end) // _SAMPLES_PER_MS)
        )
        self._last_end_sample = utterance.speech_end_sample
        self._utterances_seen += 1
        record = _Record(utterance, pause_ms, targets)
        for waiting in self._orphans.pop(utterance.id, ()):
            record.translations[waiting.target] = waiting
        self._records[utterance.id] = record
        while len(self._records) > self._max_rows:
            self._records.popitem(last=False)
            self._rows_dropped += 1

    def _add_translation(self, translation: Translation) -> None:
        self._translations_received += 1
        if translation.latency_ms is not None:
            self._latency.record(translation.latency_ms)
        record = self._records.get(translation.utterance_id)
        if record is not None:
            record.translations[translation.target] = translation
            record.row = None
            return
        # Not seen yet (or already capped out). Keep a bounded number so a translation that
        # beat its utterance to this poll is not lost; stale ones fall off the front.
        self._orphans.setdefault(translation.utterance_id, []).append(translation)
        while len(self._orphans) > self._max_rows:
            self._orphans.popitem(last=False)

    # --- snapshot ------------------------------------------------------------------

    def _build(self, targets: tuple[str, ...]) -> Snapshot:
        pipeline = self._pipeline
        now_s = self._clock()
        running = pipeline.running
        flat = pipeline.metrics.snapshot()
        settings = pipeline.settings
        capture = pipeline.capture_stats()
        input_state = pipeline.input_state()
        input_device = pipeline.input_device()
        input_card = InputCard(
            device_name=None if input_device is None else input_device.name,
            device_query=settings.capture.device_query,
            state=input_state,
            level=None,
            callbacks=capture.callbacks,
            input_overflow=capture.input_overflow,
            ring_lost_frames=capture.ring_lost_frames,
            reconnects=capture.reconnects,
            losses=capture.losses,
            chunks=capture.chunks,
            last_error=capture.last_error,
        )
        queries = {route.target: route.output.device_query for route in settings.routes}
        outputs: list[OutputCard] = []
        for target in targets:
            card = self._output_card(target, queries.get(target, ""), flat)
            if card is not None:
                outputs.append(card)
        bus_stats = pipeline.bus_stats()
        buses = (
            *self._bus_rows("chunks", bus_stats.chunks),
            *self._bus_rows("text", bus_stats.text),
            *self._bus_rows("translations", pipeline.translation_bus.stats()),
        )
        decode_p50, decode_p95 = _percentiles(flat, "transcribe.decode_ms")
        rtf_p50, rtf_p95 = _percentiles(flat, "transcribe.rtf")
        transcribe = TranscribeStats(
            utterances=_count(flat, "transcribe.utterances"),
            errors=_count(flat, "transcribe.errors"),
            decode_p50_ms=decode_p50,
            decode_p95_ms=decode_p95,
            rtf_p50=rtf_p50,
            rtf_p95=rtf_p95,
        )
        stages = self._stages(now_s, running, input_card, tuple(outputs), transcribe, bus_stats)
        hide = self._hide_text
        rows = tuple(record.frozen(hide) for record in self._records.values())
        latency = self._latency.summary()
        view_dropped = self._texts.stats().dropped + self._translations.stats().dropped
        summary = Summary(
            utterances_seen=self._utterances_seen,
            rows_shown=len(rows),
            rows_dropped=self._rows_dropped,
            translations_received=self._translations_received,
            latency_count=min(latency.count, _LATENCY_WINDOW),
            latency_mean_ms=latency.mean if latency.count else None,
            latency_p95_ms=latency.p95 if latency.count else None,
            bus_dropped=sum(row.dropped for row in buses),
            view_dropped=view_dropped,
            input_overflow=capture.input_overflow,
            ring_lost_frames=capture.ring_lost_frames,
            underruns=sum(card.underruns for card in outputs),
            errors=transcribe.errors + sum(card.lane.errors for card in outputs),
        )
        return Snapshot(
            seq=self._seq,
            running=running,
            position_s=pipeline.position_samples / SAMPLE_RATE_HZ,
            hide_text=hide,
            targets=targets,
            input=input_card,
            outputs=tuple(outputs),
            buses=buses,
            transcribe=transcribe,
            stages=stages,
            summary=summary,
            rows=rows,
        )

    def _output_card(
        self, target: str, device_query: str, flat: Mapping[str, float]
    ) -> OutputCard | None:
        pipeline = self._pipeline
        try:
            stats = pipeline.output_stats(target)
            device = pipeline.output_device(target)
            state = pipeline.output_state(target)
        except ValueError:
            # The language of this output was changed between reading the target list and
            # asking for it (UnknownTargetError); the next poll sees the new list.
            return None
        p50, p95 = _percentiles(flat, f"lane.{target}.latency_ms")
        return OutputCard(
            target=target,
            device_name=None if device is None else device.name,
            device_query=device_query,
            state=state,
            lane=LaneCounters(
                utterances=_count(flat, f"lane.{target}.utterances"),
                errors=_count(flat, f"lane.{target}.errors"),
                skipped_offline=_count(flat, f"lane.{target}.skipped_offline"),
                latency_p50_ms=p50,
                latency_p95_ms=p95,
            ),
            underruns=stats.underruns,
            driver_underflows=stats.driver_underflows,
            dropped_frames=stats.dropped_frames,
            reconnects=stats.reconnects,
            losses=stats.losses,
            last_error=stats.last_error,
        )

    @staticmethod
    def _bus_rows(bus: str, stats: Mapping[str, SubscriptionStats]) -> tuple[BusRow, ...]:
        return tuple(
            BusRow(
                bus=bus,
                name=name,
                delivered=item.delivered,
                dropped=item.dropped,
                depth=item.depth,
                max_depth=item.max_depth,
            )
            for name, item in stats.items()
        )

    def _stages(
        self,
        now_s: float,
        running: bool,
        input_card: InputCard,
        outputs: tuple[OutputCard, ...],
        transcribe: TranscribeStats,
        bus_stats: BusStats,
    ) -> tuple[StageHealth, ...]:
        watch = self._tripwires.observe
        thresholds = self._thresholds
        stages: list[StageHealth] = []

        # Every tripwire is observed on every poll, whatever the branch below decides, so its
        # baseline never goes stale.
        overflow = watch("input.overflow", input_card.input_overflow, now_s)
        ring_lost = watch("input.ring_lost", input_card.ring_lost_frames, now_s)
        stages.append(self._input_stage(running, input_card, overflow, ring_lost))

        whisper = bus_stats.chunks.get("whisper")
        chunk_drops = watch("bus.whisper.dropped", 0 if whisper is None else whisper.dropped, now_s)
        decode_errors = watch("transcribe.errors", transcribe.errors, now_s)
        slow = transcribe.rtf_p95 is not None and transcribe.rtf_p95 > thresholds.rtf_p95_warn
        stages.append(self._transcribe_stage(running, transcribe, decode_errors, chunk_drops, slow))

        for card in outputs:
            target = card.target
            lane_errors = watch(f"lane.{target}.errors", card.lane.errors, now_s)
            queue = bus_stats.text.get(f"lane-{target}")
            behind = watch(
                f"bus.lane-{target}.dropped", 0 if queue is None else queue.dropped, now_s
            )
            underrun = watch(f"output.{target}.underruns", card.underruns, now_s)
            underflow = watch(f"output.{target}.underflows", card.driver_underflows, now_s)
            dropped = watch(f"output.{target}.dropped_frames", card.dropped_frames, now_s)
            stages.append(self._translate_stage(running, card, lane_errors, behind))
            stages.append(self._output_stage(running, card, underrun or underflow, dropped))
        return tuple(stages)

    @staticmethod
    def _input_stage(
        running: bool, card: InputCard, overflow: bool, ring_lost: bool
    ) -> StageHealth:
        label = "Microphone"
        if not running:
            return StageHealth("input", label, Health.IDLE, "stopped")
        if card.state is DeviceState.DISCONNECTED:
            return StageHealth("input", label, Health.BAD, "disconnected")
        if card.state is DeviceState.RECONNECTING:
            return StageHealth("input", label, Health.WARN, "reconnecting")
        if overflow:
            return StageHealth(
                "input", label, Health.WARN, f"input overflow ({card.input_overflow} so far)"
            )
        if ring_lost:
            return StageHealth(
                "input", label, Health.WARN, f"audio lost ({card.ring_lost_frames} frames)"
            )
        return StageHealth("input", label, Health.OK, "connected")

    def _transcribe_stage(
        self,
        running: bool,
        stats: TranscribeStats,
        errors: bool,
        chunk_drops: bool,
        slow: bool,
    ) -> StageHealth:
        label = "Speech recognition"
        if not running:
            return StageHealth("transcribe", label, Health.IDLE, "stopped")
        if errors:
            return StageHealth(
                "transcribe", label, Health.WARN, f"decode errors ({stats.errors} so far)"
            )
        if chunk_drops:
            return StageHealth("transcribe", label, Health.WARN, "chunks dropped before decode")
        if slow:
            return StageHealth(
                "transcribe",
                label,
                Health.WARN,
                f"slower than real time (p95 factor {stats.rtf_p95:.2f})",
            )
        if stats.utterances == 0 and stats.decode_p50_ms is None:
            return StageHealth("transcribe", label, Health.IDLE, "no speech yet")
        return StageHealth("transcribe", label, Health.OK, "decoding")

    @staticmethod
    def _translate_stage(
        running: bool, card: OutputCard, lane_errors: bool, behind: bool
    ) -> StageHealth:
        key = f"translate:{card.target}"
        label = f"Translation {card.target}"
        if not running:
            return StageHealth(key, label, Health.IDLE, "stopped")
        if card.state is not DeviceState.CONNECTED:
            return StageHealth(key, label, Health.IDLE, "output away, utterances skipped")
        if lane_errors:
            return StageHealth(key, label, Health.WARN, f"errors ({card.lane.errors} so far)")
        if behind:
            return StageHealth(key, label, Health.WARN, "falling behind, utterances dropped")
        if card.lane.utterances == 0 and card.lane.errors == 0:
            return StageHealth(key, label, Health.IDLE, "no speech yet")
        return StageHealth(key, label, Health.OK, f"{card.lane.utterances} played")

    @staticmethod
    def _output_stage(
        running: bool, card: OutputCard, underrun: bool, dropped: bool
    ) -> StageHealth:
        key = f"output:{card.target}"
        label = f"Output {card.target}"
        if not running:
            return StageHealth(key, label, Health.IDLE, "stopped")
        if card.state is DeviceState.DISCONNECTED:
            return StageHealth(key, label, Health.BAD, "disconnected")
        if card.state is DeviceState.RECONNECTING:
            return StageHealth(key, label, Health.WARN, "reconnecting")
        if underrun:
            return StageHealth(
                key,
                label,
                Health.WARN,
                f"underruns ({card.underruns} so far, {card.driver_underflows} driver)",
            )
        if dropped:
            return StageHealth(
                key, label, Health.WARN, f"audio dropped ({card.dropped_frames} frames)"
            )
        return StageHealth(key, label, Health.OK, "connected")


# --- plain-text report ------------------------------------------------------------------


def _clean(text: str) -> str:
    """One line, no tabs: the table is tab-separated so it pastes into a spreadsheet."""
    return " ".join(text.split())


def _number(value: float | None, digits: int = 0) -> str:
    return _NONE_CELL if value is None else f"{value:.{digits}f}"


def render_report(snapshot: Snapshot) -> str:
    """Format a snapshot as plain text: a summary header and a tab-separated table.

    Text columns show ``[hidden]`` when the snapshot was taken with ``hide_text`` on, and a
    dash for a translation that has not arrived; timings are always included.
    """
    summary = snapshot.summary
    targets = list(snapshot.targets)
    for row in snapshot.rows:
        targets.extend(c.target for c in row.translations if c.target not in targets)
    lines = [
        "Interpreter session report",
        f"utterances: {summary.utterances_seen} "
        f"(shown {summary.rows_shown}, dropped by the row cap {summary.rows_dropped})",
        f"translations: {summary.translations_received}",
        f"end-to-end latency ms: mean {_number(summary.latency_mean_ms)}, "
        f"p95 {_number(summary.latency_p95_ms)} (n={summary.latency_count})",
        f"drops and errors: bus drops {summary.bus_dropped} "
        f"(own queues {summary.view_dropped}), input overflow {summary.input_overflow}, "
        f"ring lost frames {summary.ring_lost_frames}, output underruns {summary.underruns}, "
        f"errors {summary.errors}",
        f"targets: {', '.join(snapshot.targets) or _NONE_CELL}",
        f"stream time s: {snapshot.position_s:.1f}",
        f"text: {'hidden' if snapshot.hide_text else 'shown'}",
        "",
    ]
    header = [
        "#",
        "start_s",
        "pause_ms",
        "audio_ms",
        "decode_ms",
        "finalize_wait_ms",
        "reason",
        "text",
    ]
    for target in targets:
        header.extend((target, f"{target}_latency_ms"))
    lines.append("\t".join(header))
    for row in snapshot.rows:
        cells = [
            str(row.utterance_id),
            f"{row.start_s:.2f}",
            _NONE_CELL if row.pause_ms is None else str(row.pause_ms),
            str(row.audio_ms),
            f"{row.decode_ms:.0f}",
            str(row.finalize_wait_ms),
            row.reason,
            _HIDDEN if row.text is None else _clean(row.text),
        ]
        for target in targets:
            cell = row.cell(target)
            if not cell.arrived:
                cells.extend((_NONE_CELL, _NONE_CELL))
                continue
            cells.append(_HIDDEN if cell.text is None else _clean(cell.text))
            cells.append(_number(cell.latency_ms))
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


__all__ = [
    "BusRow",
    "Health",
    "HealthThresholds",
    "InputCard",
    "LaneCounters",
    "OutputCard",
    "PipelineView",
    "Snapshot",
    "StageHealth",
    "Summary",
    "TranscribeStats",
    "TranscriptRow",
    "TranslationCell",
    "ViewModel",
    "ViewModelError",
    "render_report",
]
