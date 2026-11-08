"""The whole interpreter as one object: build, start, stop and control devices at run time.

``Pipeline`` wires the modules that already exist::

    CaptureSession --> ChunkBus --> TranscribeStage --> TextBus --> LanguageLane(de) --> output
    (one VAD stage kept across reconnects)                      +--> LanguageLane(fr) --> output

It owns no audio logic of its own and touches no hardware directly: streams, the device list,
the speech-to-text engine, the translators and the speech synthesis are injected, so tests
run it on fakes and a window or the command line decides what the real ones are.

Runtime device control (the window needs it, ``run`` does not):

* The input is a ``CaptureSession``. Disconnecting it stops the session (which publishes the
  pending chunk, ``CaptureSession.stop``) and a later connect builds a new session around
  the *same* VAD stage, so chunk ids and the stream clock keep running and the utterances
  that follow are numbered and timed as if nothing had happened.
* Each output is an ``OutputSession`` behind a stable sink handed to its lane. The lane and
  its bus subscription never change; only the session behind the sink is stopped, started
  or replaced. A lane whose output is away skips utterances before translating them
  (``skipped_offline``), and a lane is never aware of another lane's output.

Language of an output (``set_output_target``): only that lane is rebuilt (new translator and
speech synthesis from the injected factories, new ``TextBus`` subscription); the sink, the
device binding, the other lanes and the microphone are not touched. Lanes publish what they
translated on ``translation_bus`` for a window; text never goes to metrics or logs.

Statistics accessors read plain attributes without taking the control lock, so a window that
polls them is never held up by a device that is slow to open.

A ``Pipeline`` is used once: ``start()`` after ``stop()`` is refused because the buses are
closed. Applying new settings means building a new ``Pipeline``.
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from pipeline.bus import ChunkBus, Subscription, SubscriptionStats
from pipeline.capture import CaptureConfig, CaptureSession, CaptureStats, ChunkStage
from pipeline.config import SOURCE_LANGUAGE, SUPPORTED_TARGETS, ConfigError, LaneRoute, Settings
from pipeline.devices import (
    AmbiguousDeviceError,
    DeviceInfo,
    DeviceListing,
    DeviceNotFoundError,
    OutputStreamFactory,
    StreamFactory,
    StreamOpenError,
    find_device,
)
from pipeline.glossary import Glossary
from pipeline.lanes import (
    LaneConfig,
    LanguageLane,
    ResamplerFactory,
    TranslationBus,
    Translator,
    TtsEngine,
)
from pipeline.metrics import Metrics
from pipeline.output import OutputConfig, OutputSession, OutputStats
from pipeline.transcribe import SpeechToText, TextBus, TranscribeStage, Utterance
from pipeline.vad import Float32Array

_LANE_QUEUE = 2  # a lane more than two utterances behind drops the oldest (spec section 7)
_CAPTURE_LATENCY_WINDOW = 1000  # same bound as ``CaptureSession``


class DeviceState(enum.StrEnum):
    """What the window shows next to a device."""

    CONNECTED = "connected"
    RECONNECTING = "reconnecting"  # wanted, but the stream is lost and being retried
    DISCONNECTED = "disconnected"  # not wanted (or the pipeline is not running)


class PipelineStateError(RuntimeError):
    """An operation does not fit the pipeline's state (not running, already started, ...)."""


class UnknownTargetError(ValueError):
    """A target language that has no lane in this pipeline."""


@dataclass(frozen=True, slots=True)
class BusStats:
    """Per-subscriber counters of both buses.

    Attributes:
        chunks: ``ChunkBus`` subscribers (``whisper`` and whatever else subscribed).
        text: ``TextBus`` subscribers (``lane-<target>`` and whatever else subscribed).
    """

    chunks: dict[str, SubscriptionStats]
    text: dict[str, SubscriptionStats]


class _SinkProxy:
    """The sink a lane holds for good; the session behind it can be swapped at run time."""

    def __init__(self, session: OutputSession) -> None:
        self.session = session

    @property
    def sample_rate_hz(self) -> int | None:
        return self.session.sample_rate_hz

    def begin_utterance(self) -> None:
        self.session.begin_utterance()

    def end_utterance(self) -> None:
        self.session.end_utterance()

    def write_all(self, samples: Float32Array, timeout_s: float = 5.0) -> bool:
        return self.session.write_all(samples, timeout_s)


class _Slot:
    """Everything that belongs to one target language."""

    def __init__(
        self,
        route: LaneRoute,
        session: OutputSession,
        sink: _SinkProxy,
        lane: LanguageLane,
        subscription: Subscription[Utterance],
    ) -> None:
        self.route = route
        self.session = session
        self.sink = sink
        self.lane = lane
        self.subscription = subscription  # kept to remove it when the language changes
        self.enabled = False  # the user (or start) wants this output connected


def _merge_capture_stats(before: CaptureStats, now: CaptureStats | None) -> CaptureStats:
    """Add the counters of closed capture sessions to those of the current one."""
    if now is None:
        return replace(before, connected=False, last_error="")
    latencies = (*before.finalization_latency_ms, *now.finalization_latency_ms)
    return CaptureStats(
        connected=now.connected,
        callbacks=before.callbacks + now.callbacks,
        input_overflow=before.input_overflow + now.input_overflow,
        ring_lost_frames=before.ring_lost_frames + now.ring_lost_frames,
        max_callback_ms=max(before.max_callback_ms, now.max_callback_ms),
        reconnects=before.reconnects + now.reconnects,
        losses=before.losses + now.losses,
        last_loss_reason=now.last_loss_reason or before.last_loss_reason,
        last_error=now.last_error,
        chunks=before.chunks + now.chunks,
        finalization_latency_ms=latencies[-_CAPTURE_LATENCY_WINDOW:],
    )


_NO_CAPTURE_STATS = CaptureStats(
    connected=False,
    callbacks=0,
    input_overflow=0,
    ring_lost_frames=0,
    max_callback_ms=0.0,
    reconnects=0,
    losses=0,
    last_loss_reason="",
    last_error="",
    chunks=0,
    finalization_latency_ms=(),
)


class Pipeline:
    """Capture, transcription and one language lane per route, with runtime device control."""

    def __init__(
        self,
        settings: Settings,
        *,
        devices: DeviceListing,
        input_stream_factory: StreamFactory,
        output_stream_factory: OutputStreamFactory,
        stage_factory: Callable[[Settings], ChunkStage],
        speech_to_text: SpeechToText,
        translator_factory: Callable[[str], Translator],
        tts_factory: Callable[[str], TtsEngine],
        glossary: Glossary | None = None,
        resampler_factory: ResamplerFactory | None = None,
        metrics: Metrics | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Build every stage; nothing is opened or started until ``start()``.

        Args:
            settings: Validated settings (``settings.glossary_path`` is not read here; pass
                the loaded ``glossary`` instead).
            devices: Device list source (PortAudio or a fake).
            input_stream_factory: Opens the microphone stream.
            output_stream_factory: Opens an output stream.
            stage_factory: Builds the VAD stage from the settings. Called once; the stage
                outlives every input reconnect.
            speech_to_text: The loaded whisper engine (or a fake).
            translator_factory: ``target -> Translator``, called once per lane.
            tts_factory: ``target -> TtsEngine``, called once per lane.
            glossary: Optional glossary for hotwords and term protection.
            resampler_factory: Converts speech to a device rate that differs from the TTS
                rate; without it such a lane counts a ``LaneError`` per utterance.
            metrics: Shared registry; a new one is created when omitted.
            clock: Monotonic seconds for the watchdogs and retry timers of the sessions.
        """
        self._settings = settings
        self._devices = devices
        self._input_factory = input_stream_factory
        self._output_factory = output_stream_factory
        self._clock = clock
        self._translator_factory = translator_factory
        self._tts_factory = tts_factory
        self._glossary = glossary
        self._resampler_factory = resampler_factory
        self.metrics = metrics or Metrics()
        self._chunks = ChunkBus()
        self._texts: TextBus = TextBus()
        self._translations: TranslationBus = TranslationBus()
        self._stage = stage_factory(settings)
        self._transcribe = TranscribeStage(
            self._chunks.subscribe("whisper"),
            self._texts,
            speech_to_text,
            settings.transcribe,
            glossary=glossary,
            metrics=self.metrics,
        )
        self._slots: dict[str, _Slot] = {}
        for route in settings.routes:
            session = self._new_session(route.output)
            sink = _SinkProxy(session)
            lane, subscription = self._new_lane(route, sink)
            self._slots[route.target] = _Slot(route, session, sink, lane, subscription)
        self._capture: CaptureSession | None = None
        self._capture_base = _NO_CAPTURE_STATS
        self._lock = threading.RLock()  # control calls only; never taken by statistics reads
        self._started = False
        self._stopped = False

    # --- read-only accessors -------------------------------------------------------

    @property
    def settings(self) -> Settings:
        """The current settings, including device choices made at run time."""
        return self._settings

    @property
    def targets(self) -> tuple[str, ...]:
        """Target languages, in lane order."""
        return tuple(self._slots)

    @property
    def running(self) -> bool:
        """True between a successful ``start()`` and ``stop()``."""
        return self._started and not self._stopped

    @property
    def text_bus(self) -> TextBus:
        """The utterance bus; a window subscribes here with its own queue."""
        return self._texts

    @property
    def translation_bus(self) -> TranslationBus:
        """What the lanes translated; a window subscribes here with its own queue."""
        return self._translations

    @property
    def chunk_bus(self) -> ChunkBus:
        """The chunk bus; a meter or a test can subscribe."""
        return self._chunks

    @property
    def position_samples(self) -> int:
        """Stream position in 16 kHz samples; the clock of every chunk and utterance."""
        return self._stage.position_samples

    def capture_stats(self) -> CaptureStats:
        """Capture counters summed over every input connection since the pipeline was built."""
        capture = self._capture
        return _merge_capture_stats(
            self._capture_base, None if capture is None else capture.stats()
        )

    def output_stats(self, target: str) -> OutputStats:
        """Counters of one output (they survive disconnect and connect of the same device)."""
        return self._slot(target).session.stats()

    def bus_stats(self) -> BusStats:
        """Per-subscriber counters of both buses."""
        return BusStats(chunks=self._chunks.stats(), text=self._texts.stats())

    def input_device(self) -> DeviceInfo | None:
        """The microphone that is open right now, if any."""
        capture = self._capture
        return None if capture is None else capture.device

    def output_device(self, target: str) -> DeviceInfo | None:
        """The output device of ``target`` that is open right now, if any."""
        return self._slot(target).session.device

    def input_state(self) -> DeviceState:
        """Connected, reconnecting (lost, being retried) or disconnected."""
        capture = self._capture
        if capture is None or capture.fatal_error is not None:
            return DeviceState.DISCONNECTED
        return DeviceState.CONNECTED if capture.connected else DeviceState.RECONNECTING

    def output_state(self, target: str) -> DeviceState:
        """Connected, reconnecting (lost, being retried) or disconnected."""
        slot = self._slot(target)
        session = slot.session
        if not slot.enabled or session.fatal_error is not None:
            return DeviceState.DISCONNECTED
        return DeviceState.CONNECTED if session.connected else DeviceState.RECONNECTING

    # --- lifecycle -----------------------------------------------------------------

    def start(self, *, lenient: bool = False) -> tuple[str, ...]:
        """Open every device and start every thread; the microphone goes last.

        Args:
            lenient: Strict by default: a missing or unusable output device raises. With
                ``lenient=True`` such an output stays disconnected (its lane skips
                utterances, the other lanes work) and can be connected later with
                ``connect_output``. The microphone is always required.

        Returns:
            The targets whose output could not be opened (empty unless ``lenient``).

        Raises:
            PipelineStateError: Already started, or stopped before.
            DeviceNotFoundError: A configured device does not exist.
            AmbiguousDeviceError: A device name matches several devices.
            StreamOpenError: A device was found but could not be opened.

        A failure leaves the pipeline stopped (everything opened so far is closed again).
        """
        with self._lock:
            if self._stopped:
                raise PipelineStateError("the pipeline was stopped; build a new Pipeline")
            if self._started:
                raise PipelineStateError("the pipeline is already started")
            self._started = True
            unavailable: list[str] = []
            try:
                for target, slot in self._slots.items():
                    try:
                        slot.session.start()
                    except (DeviceNotFoundError, AmbiguousDeviceError, StreamOpenError):
                        if not lenient:
                            raise
                        unavailable.append(target)
                        continue
                    slot.enabled = True
                self._transcribe.start()
                for slot in self._slots.values():
                    slot.lane.start()
                self._start_capture(self._settings.capture)
            except Exception:
                self._shutdown(timeout_s=2.0)
                raise
            return tuple(unavailable)

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop everything in order; safe to call repeatedly and before ``start()``.

        The microphone stops first and publishes the pending chunk. The chunk bus is then
        closed with its backlog kept for other subscribers, whisper and the lanes finish
        the item in hand and exit, and the outputs close last. Queued speech that was not
        played yet is discarded (a late interpretation is worse than none).

        Args:
            timeout_s: How long to wait for each worker thread.
        """
        with self._lock:
            if self._stopped:
                return
            self._shutdown(timeout_s)

    def _shutdown(self, timeout_s: float) -> None:
        self._release_capture(timeout_s)
        self._chunks.close(drain=True)
        self._transcribe.stop(timeout_s)
        self._texts.close()
        for slot in self._slots.values():
            slot.lane.stop(timeout_s)
        self._translations.close()  # after the lanes: their last translation is still delivered
        for slot in self._slots.values():
            slot.session.stop(timeout_s)
            slot.enabled = False
        self._stopped = True

    # --- input control -------------------------------------------------------------

    def connect_input(self) -> None:
        """Open the microphone again after ``disconnect_input()``; no-op while connected.

        Raises:
            PipelineStateError: The pipeline is not running.
            DeviceNotFoundError: The configured microphone does not exist now.
            AmbiguousDeviceError: Its name matches several devices.
            StreamOpenError: It was found but could not be opened.
        """
        with self._lock:
            self._require_running()
            capture = self._capture
            if capture is not None and capture.fatal_error is None:
                return  # connected, or being reconnected by its own thread
            self._release_capture(timeout_s=2.0)  # a session whose thread died
            self._start_capture(self._settings.capture)

    def disconnect_input(self) -> None:
        """Close the microphone; the pending chunk is published, everything else keeps running.

        Raises:
            PipelineStateError: The pipeline is not running.
        """
        with self._lock:
            self._require_running()
            self._release_capture(timeout_s=2.0)

    def set_input_device(self, device_query: str) -> None:
        """Choose the microphone by name substring (never by index).

        While the input is connected (or being reconnected) it is switched at once: the
        name is looked up first, so a typo leaves the working microphone alone. While it is
        disconnected, or before ``start()``, the choice is only stored.

        Raises:
            CaptureConfigError: ``device_query`` is empty.
            DeviceNotFoundError: No input device matches (the old one stays connected).
            AmbiguousDeviceError: Several match (the old one stays connected).
            StreamOpenError: The new device was found but could not be opened; the input
                is then disconnected and the old choice stays in the settings.
        """
        config = replace(self._settings.capture, device_query=device_query)
        with self._lock:
            if self._capture is not None:
                find_device(self._devices.list_devices(), config.device_query, "input")
                self._release_capture(timeout_s=2.0)
                self._start_capture(config)
            self._settings = replace(self._settings, capture=config)

    # --- output control ------------------------------------------------------------

    def connect_output(self, target: str) -> None:
        """Open the output of ``target`` again after ``disconnect_output``; no-op while open.

        Raises:
            UnknownTargetError: No lane for ``target``.
            PipelineStateError: The pipeline is not running.
            DeviceNotFoundError: The configured device does not exist now.
            AmbiguousDeviceError: Its name matches several devices.
            StreamOpenError: It was found but could not be opened.
        """
        slot = self._slot(target)
        with self._lock:
            self._require_running()
            if slot.enabled and slot.session.fatal_error is None:
                return
            if slot.session.fatal_error is not None:
                slot.session.stop(timeout_s=2.0)
                self._replace_session(slot, self._new_session(slot.route.output))
            slot.session.start()
            slot.enabled = True

    def disconnect_output(self, target: str) -> None:
        """Close the output of ``target``; its lane skips utterances, other lanes go on.

        Raises:
            UnknownTargetError: No lane for ``target``.
            PipelineStateError: The pipeline is not running.
        """
        slot = self._slot(target)
        with self._lock:
            self._require_running()
            slot.session.stop(timeout_s=2.0)
            slot.enabled = False

    def set_output(self, target: str, device_query: str) -> None:
        """Choose the output device of ``target`` by name substring (never by index).

        A connected output is replaced make-before-break: the new device is looked up and
        opened first, so a typo or a busy device leaves the old output playing. Only this
        lane's session changes; its lane and the other lanes are not touched. While the
        output is disconnected, or before ``start()``, the choice is only stored.

        Raises:
            UnknownTargetError: No lane for ``target``.
            OutputConfigError: ``device_query`` is empty.
            DeviceNotFoundError: No output device matches.
            AmbiguousDeviceError: Several match.
            StreamOpenError: The device was found but could not be opened.
        """
        slot = self._slot(target)
        output = replace(slot.route.output, device_query=device_query)
        with self._lock:
            new_session = self._new_session(output)
            if slot.enabled:
                find_device(self._devices.list_devices(), output.device_query, "output")
                new_session.start()
            old_session = slot.session
            self._replace_session(slot, new_session)
            old_session.stop(timeout_s=2.0)
            slot.route = replace(slot.route, output=output)
            self._settings = replace(
                self._settings,
                routes=tuple(
                    slot.route if route.target == target else route
                    for route in self._settings.routes
                ),
            )

    def set_output_target(self, old_target: str, new_target: str) -> None:
        """Change the language of one output at run time; only that lane is rebuilt.

        The output device, its stream and the sink binding stay as they are, and so do the
        other lanes and the microphone. The new translator and speech synthesis are built
        first, so a model that cannot load leaves the old lane working. Utterances still
        waiting for the old lane are dropped. Before ``start()`` the lane is only swapped.

        Args:
            old_target: Language of an existing lane.
            new_target: Language to switch to; choosing ``old_target`` changes nothing.

        Raises:
            UnknownTargetError: No lane for ``old_target``.
            ConfigError: ``new_target`` is unsupported, the source language, or already used
                by another lane.
            PipelineStateError: The pipeline was stopped.
            Exception: Whatever the translator or TTS factory raises; nothing changed then.
        """
        with self._lock:
            slot = self._slot(old_target)
            if new_target == old_target:
                return
            if self._stopped:
                raise PipelineStateError("the pipeline was stopped; build a new Pipeline")
            if new_target == SOURCE_LANGUAGE:
                raise ConfigError(f"target language {new_target!r} is the same as the source")
            if new_target not in SUPPORTED_TARGETS:
                raise ConfigError(
                    f"unsupported target language {new_target!r}; "
                    f"choose from {', '.join(SUPPORTED_TARGETS)}"
                )
            if new_target in self._slots:
                raise ConfigError(f"target language {new_target!r} is already used by another lane")
            route = replace(slot.route, target=new_target)
            settings = replace(
                self._settings,
                targets=tuple(new_target if t == old_target else t for t in self._settings.targets),
                routes=tuple(route if r.target == old_target else r for r in self._settings.routes),
            )
            lane, subscription = self._new_lane(route, slot.sink)  # may raise; old lane intact
            self._texts.unsubscribe(slot.subscription)
            slot.lane.stop(timeout_s=5.0)  # ends after the utterance in hand
            slot.route = route
            slot.lane = lane
            slot.subscription = subscription
            self._slots = {
                (new_target if target == old_target else target): other
                for target, other in self._slots.items()
            }
            self._settings = settings
            if self.running:
                lane.start()

    # --- internals -----------------------------------------------------------------

    def _slot(self, target: str) -> _Slot:
        try:
            return self._slots[target]
        except KeyError:
            raise UnknownTargetError(
                f"no lane for target {target!r}; this pipeline has {', '.join(self._slots)}"
            ) from None

    def _require_running(self) -> None:
        if not self.running:
            raise PipelineStateError("the pipeline is not running")

    def _new_lane(
        self, route: LaneRoute, sink: _SinkProxy
    ) -> tuple[LanguageLane, Subscription[Utterance]]:
        """Build a stopped lane; its translator and TTS come from the injected factories."""
        translator = self._translator_factory(route.target)
        tts = self._tts_factory(route.target)
        subscription = self._texts.subscribe(f"lane-{route.target}", maxsize=_LANE_QUEUE)
        lane = LanguageLane(
            route.target,
            subscription,
            translator,
            tts,
            sink,
            LaneConfig(radio_offset_ms=route.radio_offset_ms),
            glossary=self._glossary,
            metrics=self.metrics,
            position_samples=lambda: self._stage.position_samples,
            resampler_factory=self._resampler_factory,
            translations=self._translations,
        )
        return lane, subscription

    def _new_session(self, config: OutputConfig) -> OutputSession:
        return OutputSession(config, self._devices, self._output_factory, self._clock)

    def _replace_session(self, slot: _Slot, session: OutputSession) -> None:
        slot.sink.session = session  # the lane keeps writing to the same sink object
        slot.session = session

    def _start_capture(self, config: CaptureConfig) -> None:
        session = CaptureSession(
            config, self._devices, self._input_factory, self._stage, self._chunks, self._clock
        )
        session.start()  # raises before anything is kept when the device is not usable
        self._capture = session

    def _release_capture(self, timeout_s: float) -> None:
        capture = self._capture
        if capture is None:
            return
        capture.stop(timeout_s)  # publishes the pending chunk and resets the VAD stage
        self._capture_base = _merge_capture_stats(self._capture_base, capture.stats())
        self._capture = None
