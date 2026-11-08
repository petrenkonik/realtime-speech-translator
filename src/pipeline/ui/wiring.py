"""The composition root of the window: builds the real pipeline from the local models.

This is the only place where the window's pieces meet the real ones (Silero VAD, Smart Turn,
faster-whisper, CTranslate2, PortAudio). It has no Tk, takes every loader and factory as a
parameter (``Loaders``, stream factories, device list, guard) so tests replace them, and reads
models from ``Settings.models_dir`` only (``HF_HUB_OFFLINE=1`` is set before anything loads).

Whisper and the translators take seconds to load, and a restart (applying a setting) builds a
new ``Pipeline``. They are kept in memory between builds, keyed by what the load depends on
(whisper: model and threads; a translator: folder and ``TranslateConfig``), so a changed
silence threshold restarts in a moment. They are only ever used by one pipeline at a time:
the old one is stopped before the new one starts. The VAD is always created fresh because its
recurrent state belongs to one stream.

Known gap: there is no Windows SAPI speech synthesis yet (work-plan step 11). ``SilentTtsEngine``
stands in for it. It returns silence of a plausible length, so the lanes, resampling, outputs
and latency figures all run, but nothing audible is played; ``SILENT_TTS_NOTICE`` is shown as
a banner in the window. Remove the placeholder and the ``SilenceResampler`` with step 11.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from pipeline.app import Pipeline
from pipeline.config import VAD_MODEL, ConfigError, Settings, check_files
from pipeline.devices import (
    DeviceListing,
    OutputStreamFactory,
    SoundDeviceListing,
    StreamFactory,
    open_sounddevice_input,
    open_sounddevice_output,
)
from pipeline.glossary import Glossary
from pipeline.guards import StrictGuard
from pipeline.lanes import Resampler, Speech, Translator, TtsEngine
from pipeline.transcribe import FasterWhisperEngine, TranscribeConfig, whisper_model_dir
from pipeline.translate import CtranslateTranslator, TranslateConfig, translator_model_dir
from pipeline.turn import DEFAULT_TURN_MODEL, OnnxTurnClassifier, ThreadedTurnLink, TurnClassifier
from pipeline.ui.controller import BuiltPipeline
from pipeline.vad import SAMPLE_RATE_HZ, Float32Array, Segmenter, SileroVad, TurnLink, VadStage

SILENT_TTS_NOTICE = "TTS not installed: outputs are silent"

_SILENCE_MS_PER_CHAR = 20
_SILENCE_MIN_MS = 200
_SILENCE_MAX_MS = 4_000


# --- placeholder speech ----------------------------------------------------------------------


class SilentTtsEngine:
    """Placeholder ``TtsEngine``: silence whose length follows the text (see the module note)."""

    def synthesize(self, text: str) -> Speech:
        """Return 16 kHz silence of 20 ms per character, between 200 ms and 4 s."""
        ms = min(_SILENCE_MAX_MS, max(_SILENCE_MIN_MS, len(text.strip()) * _SILENCE_MS_PER_CHAR))
        return Speech(
            samples=np.zeros(ms * SAMPLE_RATE_HZ // 1000, dtype=np.float32), rate_hz=SAMPLE_RATE_HZ
        )


class SilenceResampler:
    """Resamples silence by length only.

    ``StreamingResampler`` cannot produce the device rates the lanes need yet (spec step 11);
    this is exact for the silent placeholder speech and wrong for anything else.
    """

    def __init__(self, in_rate_hz: int, out_rate_hz: int) -> None:
        """Remember the two rates."""
        self._ratio = out_rate_hz / in_rate_hz

    def process(self, block: Float32Array) -> Float32Array:
        """Return silence as long as ``block`` would be at the output rate."""
        return np.zeros(round(len(block) * self._ratio), dtype=np.float32)


def silence_resampler_factory(in_rate_hz: int, out_rate_hz: int) -> Resampler:
    """``ResamplerFactory`` for the placeholder speech."""
    return SilenceResampler(in_rate_hz, out_rate_hz)


# --- loaders ---------------------------------------------------------------------------------


class ClosableTurnLink(TurnLink, Protocol):
    """A turn link that owns a thread and must be closed."""

    def close(self) -> None:
        """Stop its thread."""
        ...


class Guard(Protocol):
    """The part of ``StrictGuard`` the wiring uses."""

    def arm(self) -> None:
        """Start blocking disk writes and network use."""
        ...

    def disarm(self) -> None:
        """Stop blocking."""
        ...


def _load_whisper(model_dir: Path, config: TranscribeConfig) -> FasterWhisperEngine:
    engine = FasterWhisperEngine.load(model_dir, config)
    engine.warm_up()  # the first utterance must not pay for set-up
    return engine


def _load_classifier(model_path: Path, threads: int) -> TurnClassifier:
    return OnnxTurnClassifier(model_path, threads)


@dataclass(frozen=True, slots=True)
class Loaders:
    """How each model is loaded from a local path; tests replace them with fakes.

    Attributes:
        vad: ``model path -> SileroVad``.
        turn_classifier: ``(model path, threads) -> classifier`` of Smart Turn.
        turn_link: ``classifier -> link`` that runs it on its own thread.
        whisper: ``(model dir, config) -> engine``, already warmed up.
        translator: ``(model dir, config) -> translator``.
        glossary: ``path -> Glossary``.
    """

    vad: Callable[[Path], SileroVad] = SileroVad
    turn_classifier: Callable[[Path, int], TurnClassifier] = _load_classifier
    turn_link: Callable[[TurnClassifier], ClosableTurnLink] = ThreadedTurnLink
    whisper: Callable[[Path, TranscribeConfig], FasterWhisperEngine] = _load_whisper
    translator: Callable[[Path, TranslateConfig], Translator] = CtranslateTranslator.load
    glossary: Callable[[Path], Glossary] = Glossary.from_file


def _close_all(closers: list[Callable[[], None]]) -> None:
    for close in closers:
        close()


# --- the wiring ------------------------------------------------------------------------------


class RealWiring:
    """Builds ``Pipeline`` objects on the real models (implements ``controller.Wiring``)."""

    def __init__(
        self,
        *,
        devices: DeviceListing | None = None,
        input_stream_factory: StreamFactory = open_sounddevice_input,
        output_stream_factory: OutputStreamFactory = open_sounddevice_output,
        guard: Guard | None = None,
        loaders: Loaders | None = None,
    ) -> None:
        """Create the wiring; nothing is loaded until ``build``.

        Args:
            devices: Device list source; PortAudio through ``sounddevice`` when omitted.
            input_stream_factory: Opens the microphone stream.
            output_stream_factory: Opens an output stream.
            guard: The ``--strict`` guard, armed after the models are loaded when the
                settings ask for it and released when they do not.
            loaders: Model loaders.
        """
        self._listing: DeviceListing = devices if devices is not None else SoundDeviceListing()
        self._input_factory = input_stream_factory
        self._output_factory = output_stream_factory
        self._guard: Guard = guard if guard is not None else StrictGuard()
        self._loaders = loaders if loaders is not None else Loaders()
        self._lock = threading.Lock()
        self._engines: dict[tuple[Path, str, int], FasterWhisperEngine] = {}
        self._translators: dict[tuple[Path, TranslateConfig], Translator] = {}

    @property
    def tts_notice(self) -> str | None:
        """The banner that says speech is a silent placeholder."""
        return SILENT_TTS_NOTICE

    def devices(self) -> DeviceListing:
        """The device list source."""
        return self._listing

    def problems(self, settings: Settings) -> tuple[str, ...]:
        """Missing models and files, one message each, without loading anything."""
        try:
            check_files(settings)
        except ConfigError as error:
            return tuple(str(error).split("; "))
        return ()

    def build(self, settings: Settings) -> BuiltPipeline:
        """Load what is needed and build a ``Pipeline`` (not started).

        Slow on the first call (models); the window runs it on its worker thread.

        Raises:
            Exception: Whatever a loader raises (a missing model, a bad glossary); anything
                created so far is released first.
        """
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # nothing may reach for the network
        closers: list[Callable[[], None]] = []
        try:
            glossary = (
                None
                if settings.glossary_path is None
                else self._loaders.glossary(settings.glossary_path)
            )
            engine = self._whisper(settings)
            pipeline = Pipeline(
                settings,
                devices=self._listing,
                input_stream_factory=self._input_factory,
                output_stream_factory=self._output_factory,
                stage_factory=lambda current: self._stage(current, closers),
                speech_to_text=engine,
                translator_factory=lambda target: self._translator(settings, target),
                tts_factory=_silent_tts,
                glossary=glossary,
                resampler_factory=silence_resampler_factory,
            )
            # Armed only now: every model is read and nothing below writes or connects.
            if settings.strict:
                self._guard.arm()
            else:
                self._guard.disarm()
        except Exception:
            _close_all(closers)
            raise
        return BuiltPipeline(pipeline, close=lambda: _close_all(closers))

    # --- parts ---------------------------------------------------------------------------

    def _whisper(self, settings: Settings) -> FasterWhisperEngine:
        config = settings.transcribe
        key = (settings.models_dir, config.model, config.threads)
        with self._lock:
            engine = self._engines.get(key)
            if engine is None:
                engine = self._loaders.whisper(
                    whisper_model_dir(settings.models_dir, config), config
                )
                self._engines[key] = engine
        return engine.with_config(config)

    def _translator(self, settings: Settings, target: str) -> Translator:
        folder = translator_model_dir(settings.models_dir, settings.source, target)
        key = (folder, settings.translate)
        with self._lock:
            translator = self._translators.get(key)
            if translator is None:
                translator = self._loaders.translator(folder, settings.translate)
                self._translators[key] = translator
        return translator

    def _stage(self, settings: Settings, closers: list[Callable[[], None]]) -> VadStage:
        vad = self._loaders.vad(settings.models_dir / VAD_MODEL)
        segmenter = Segmenter(settings.segmenter)
        link: ClosableTurnLink | None = None
        if settings.segmenter.turn_mode == "smart":
            classifier = self._loaders.turn_classifier(
                settings.models_dir / DEFAULT_TURN_MODEL, settings.turn_threads
            )
            link = self._loaders.turn_link(classifier)
            closers.append(link.close)
        return VadStage(vad, segmenter, link)


def _silent_tts(target: str) -> TtsEngine:
    return SilentTtsEngine()
