"""One language lane: translate, speak, play, on its own thread.

A ``LanguageLane`` reads utterances from its own ``TextBus`` subscription and turns each into
audio on its own output device. Lanes share no models, buffers or threads, so a slow or
failing lane can only hurt itself; the bus drops the oldest waiting utterance of a lane that
falls behind (the subscription is created with ``maxsize=2``), because a late interpretation is
worse than a skipped one.

The translator, the speech synthesis and the output are behind small interfaces so the lane is
tested without models or sound devices.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from pipeline.bus import Bus, Subscription
from pipeline.glossary import Glossary
from pipeline.metrics import Metrics
from pipeline.transcribe import Utterance
from pipeline.translate import split_sentences
from pipeline.vad import SAMPLE_RATE_HZ, Float32Array

_SAMPLES_PER_MS = SAMPLE_RATE_HZ // 1000
_POLL_S = 0.1
_MAX_RADIO_OFFSET_MS = 2_000


class LaneConfigError(ValueError):
    """A lane setting is out of range; the message names it."""


class LaneError(RuntimeError):
    """A lane cannot do its job with the audio it was given."""


@dataclass(frozen=True, slots=True)
class LaneConfig:
    """Settings of one lane.

    Attributes:
        radio_offset_ms: Delay of the output beyond the sound card (a wireless headset has
            a radio latency, test T6). It is added to the reported latency only; the audio
            is not delayed.
        write_timeout_s: How long to wait for room in the output ring before the rest of an
            utterance is dropped.
    """

    radio_offset_ms: int = 0
    write_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not 0 <= self.radio_offset_ms <= _MAX_RADIO_OFFSET_MS:
            raise LaneConfigError(
                f"radio_offset_ms must be 0-{_MAX_RADIO_OFFSET_MS}, got {self.radio_offset_ms}"
            )
        if self.write_timeout_s <= 0.0:
            raise LaneConfigError(f"write_timeout_s must be positive, got {self.write_timeout_s}")


class TranslationError(ValueError):
    """A ``Translation`` was built with values that cannot be right."""


@dataclass(frozen=True, slots=True)
class Translation:
    """The text a lane produced for one utterance, for a window that shows it.

    Attributes:
        utterance_id: ``Utterance.id`` it translates.
        target: Language code of the lane.
        text: The translation (after glossary restoration). RAM only: it is left out of
            ``repr`` so a stray log line or traceback cannot leak it.
        translate_ms: Time spent in the translator.
        tts_ms: Time spent in speech synthesis, ``None`` if it did not finish (it failed).
        latency_ms: Stream time from the end of speech to the first queued audio, plus the
            radio offset (the value of the ``latency_ms`` metric); ``None`` if audio never
            started (synthesis failed, nothing to play) or the lane has no stream clock.
    """

    utterance_id: int
    target: str
    text: str = field(repr=False)
    translate_ms: float
    tts_ms: float | None
    latency_ms: float | None

    def __post_init__(self) -> None:
        """Validate the values."""
        if self.utterance_id < 0:
            raise TranslationError(f"utterance_id must not be negative, got {self.utterance_id}")
        if not self.target:
            raise TranslationError("target must not be empty")
        if not math.isfinite(self.translate_ms) or self.translate_ms < 0.0:
            raise TranslationError(f"translate_ms must be finite and >= 0, got {self.translate_ms}")
        if self.tts_ms is not None and (not math.isfinite(self.tts_ms) or self.tts_ms < 0.0):
            raise TranslationError(f"tts_ms must be finite and >= 0, got {self.tts_ms}")
        if self.latency_ms is not None and not math.isfinite(self.latency_ms):
            raise TranslationError(f"latency_ms must be finite, got {self.latency_ms}")


TranslationBus = Bus[Translation]
"""Bus of finished translations (language lanes -> window)."""


@dataclass(frozen=True, slots=True)
class Speech:
    """Synthesized speech: mono float32 samples at ``rate_hz``."""

    samples: Float32Array
    rate_hz: int


class _Outcome:
    """What one utterance has produced so far; published once, whatever happens next."""

    __slots__ = ("latency_ms", "published", "text", "translate_ms", "tts_ms", "utterance_id")

    def __init__(self, utterance_id: int, text: str, translate_ms: float) -> None:
        self.utterance_id = utterance_id
        self.text = text
        self.translate_ms = translate_ms
        self.tts_ms: float | None = None
        self.latency_ms: float | None = None
        self.published = False


class Translator(Protocol):
    """Translation into the lane's language (one instance per language pair)."""

    def translate(self, text: str) -> str:
        """Return the translation of ``text``."""
        ...


class TtsEngine(Protocol):
    """Speech synthesis kept in memory."""

    def synthesize(self, text: str) -> Speech:
        """Return the speech for ``text``; never writes a file."""
        ...


class Resampler(Protocol):
    """Block resampler (``StreamingResampler`` satisfies it)."""

    def process(self, block: Float32Array) -> Float32Array:
        """Resample one block."""
        ...


class AudioSink(Protocol):
    """The part of ``OutputSession`` the lane uses."""

    @property
    def sample_rate_hz(self) -> int | None:
        """Rate the device plays at, or ``None`` while it is disconnected."""
        ...

    def begin_utterance(self) -> None:
        """Mark the start of speech."""
        ...

    def end_utterance(self) -> None:
        """Mark the end of speech."""
        ...

    def write_all(self, samples: Float32Array, timeout_s: float = ...) -> bool:
        """Queue all samples; False if the ring stayed full."""
        ...


ResamplerFactory = Callable[[int, int], Resampler]


class LanguageLane:
    """Translates and speaks the utterances of one target language."""

    def __init__(
        self,
        target: str,
        utterances: Subscription[Utterance],
        translator: Translator,
        tts: TtsEngine,
        sink: AudioSink,
        config: LaneConfig | None = None,
        *,
        glossary: Glossary | None = None,
        metrics: Metrics | None = None,
        position_samples: Callable[[], int] | None = None,
        resampler_factory: ResamplerFactory | None = None,
        translations: TranslationBus | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Create a stopped lane.

        Args:
            target: Language code; names the metrics (``lane.<target>.*``) and selects the
                glossary wording.
            utterances: This lane's subscription of the ``TextBus``.
            translator: The translator for this language.
            tts: The speech synthesis for this language.
            sink: The output device.
            config: Settings; defaults to ``LaneConfig()``.
            glossary: Optional; its terms are protected during translation.
            metrics: Optional registry.
            position_samples: Current stream position in 16 kHz samples (the clock of the
                utterance coordinates); needed for the end-to-end latency metric.
            resampler_factory: Builds ``(in_rate_hz, out_rate_hz) -> Resampler`` for speech
                whose rate differs from the device's.
            translations: Optional bus that receives one ``Translation`` per utterance that
                was translated (see ``_handle`` for when). Publishing never blocks, so a
                slow or absent reader cannot hold the lane back.
            clock: Monotonic seconds; injected so tests control the timings.
        """
        self.target = target
        self._utterances = utterances
        self._translator = translator
        self._tts = tts
        self._sink = sink
        self._config = config or LaneConfig()
        self._glossary = glossary
        self._metrics = metrics or Metrics()
        self._position_samples = position_samples
        self._resampler_factory = resampler_factory
        self._translations = translations
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: Exception | None = None

    @property
    def finished(self) -> bool:
        """True once the bus is closed and every utterance has been handled."""
        return self._utterances.finished

    @property
    def running(self) -> bool:
        """True while the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the worker thread."""
        if self.running:
            raise RuntimeError(f"the {self.target} lane is already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"lane-{self.target}", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Ask the thread to end after the utterance in hand and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            self._thread = None

    def step(self, timeout_s: float = _POLL_S) -> bool:
        """Handle the next utterance.

        Returns:
            True if an utterance was taken, False if there was nothing to do.
        """
        utterance = self._utterances.get(timeout_s)
        if utterance is None:
            return False
        try:
            self._handle(utterance)
        except Exception as exc:  # one bad utterance must not end the lane
            self._metrics.counter(self._name("errors")).inc()
            self.last_error = exc
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.step() and self.finished:
                break

    def _name(self, suffix: str) -> str:
        return f"lane.{self.target}.{suffix}"

    def _handle(self, utterance: Utterance) -> None:
        metrics = self._metrics
        text = utterance.text.strip()
        if not text:
            metrics.counter(self._name("skipped_empty")).inc()
            return
        rate_hz = self._sink.sample_rate_hz
        if rate_hz is None:
            # The device is away. Interpreting into the void wastes the CPU the other
            # lanes need, and the speech would be stale when the device comes back.
            metrics.counter(self._name("skipped_offline")).inc()
            return

        started = self._clock()
        translated = self._translate(text)
        translated_at = self._clock()
        translate_ms = (translated_at - started) * 1000.0
        metrics.histogram(self._name("translate_ms")).record(translate_ms)
        # Rule for the window: every utterance that got a translation is published exactly
        # once, as soon as audio starts or as soon as it is clear that none will (synthesis
        # failed, nothing to play, playback failed). A translator error, an empty or blank
        # utterance and an offline output (all skipped above) publish nothing.
        outcome = _Outcome(utterance.id, translated, translate_ms)
        try:
            self._speak(outcome, rate_hz, translated_at, utterance)
        finally:
            self._announce(outcome)

    def _speak(
        self, outcome: _Outcome, rate_hz: int, translated_at: float, utterance: Utterance
    ) -> None:
        """Speak the translation one sentence at a time.

        The first sentence is queued as soon as it is synthesized, so it plays while the next
        is made (the output ring holds seconds, so ``write_all`` returns at once). ``tts_ms``
        and the latency therefore describe the first audio, not the whole utterance.
        """
        metrics = self._metrics
        converter = _RateStep(self._resampler_factory, rate_hz)
        playing = False
        try:
            for sentence in split_sentences(outcome.text) or [outcome.text]:
                speech = self._tts.synthesize(sentence)
                if outcome.tts_ms is None:
                    outcome.tts_ms = (self._clock() - translated_at) * 1000.0
                    metrics.histogram(self._name("tts_ms")).record(outcome.tts_ms)
                samples = converter.convert(speech)
                if len(samples) == 0:
                    continue
                if not playing:
                    self._sink.begin_utterance()
                    playing = True
                    outcome.latency_ms = self._record_latency(utterance)
                    self._announce(outcome)  # the text appears as the audio starts, not ends
                if not self._sink.write_all(samples, self._config.write_timeout_s):
                    # A stuck device: queueing the rest behind it would only make it later.
                    metrics.counter(self._name("write_timeouts")).inc()
                    break
        finally:
            if playing:
                self._sink.end_utterance()
        metrics.counter(self._name("utterances" if playing else "skipped_empty")).inc()

    def _announce(self, outcome: _Outcome) -> None:
        if outcome.published or self._translations is None:
            return
        outcome.published = True
        self._translations.publish(
            Translation(
                utterance_id=outcome.utterance_id,
                target=self.target,
                text=outcome.text,
                translate_ms=max(0.0, outcome.translate_ms),
                tts_ms=None if outcome.tts_ms is None else max(0.0, outcome.tts_ms),
                latency_ms=outcome.latency_ms,
            )
        )

    def _translate(self, text: str) -> str:
        if self._glossary is None:
            return self._translator.translate(text)
        protected = self._glossary.protect(text, self.target)
        restored = protected.restore(self._translator.translate(protected.text))
        if restored.missing:
            self._metrics.counter(self._name("glossary_missing")).inc(restored.missing)
        return restored.text

    def _record_latency(self, utterance: Utterance) -> float | None:
        if self._position_samples is None:
            return None
        behind_samples = self._position_samples() - utterance.speech_end_sample
        latency_ms = behind_samples / _SAMPLES_PER_MS + self._config.radio_offset_ms
        self._metrics.histogram(self._name("latency_ms")).record(latency_ms)
        return latency_ms


class _RateStep:
    """Converts the sentences of one utterance to the device rate with one resampler.

    The filter history runs through the sentence boundaries, so they do not click; a new
    utterance gets a new instance and so starts from silence.
    """

    def __init__(self, factory: ResamplerFactory | None, device_rate_hz: int) -> None:
        self._factory = factory
        self._device_rate_hz = device_rate_hz
        self._resampler: Resampler | None = None
        self._in_rate_hz = 0

    def convert(self, speech: Speech) -> Float32Array:
        if speech.rate_hz == self._device_rate_hz:
            return speech.samples
        if self._factory is None:
            raise LaneError(
                f"speech at {speech.rate_hz} Hz cannot be played at {self._device_rate_hz} Hz: "
                "no resampler was given to the lane"
            )
        if self._resampler is None or self._in_rate_hz != speech.rate_hz:
            self._resampler = self._factory(speech.rate_hz, self._device_rate_hz)
            self._in_rate_hz = speech.rate_hz
        converted: Float32Array = np.asarray(
            self._resampler.process(speech.samples), dtype=np.float32
        )
        return converted
