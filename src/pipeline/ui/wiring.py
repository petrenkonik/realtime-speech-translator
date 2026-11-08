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

Speech is the Windows SAPI voices (``tts_sapi``) and the lanes convert it to each device's
rate with ``RateConverter``. A voice is a Windows language pack, not a downloaded model, so
it is checked like one: ``problems`` reports a missing voice per target (inline in the window,
and the apply flow refuses on it), and ``tts_notice`` is therefore always ``None`` here.

COM threading. Nothing in this module touches COM on the Tk thread or on the worker thread.
``SapiTts`` makes its COM backend lazily on the lane thread that first speaks. The voice check
(``problems``, which the controller calls on the Tk thread, and the per-lane check in
``build``) runs ``check_voices`` on a short-lived helper thread that is joined at once: its
backend initialises COM there and the apartment ends with that thread. The installed voices do
not change while the program runs (the message says to restart after installing a language
pack), so each answer is cached and the helper thread is needed once per target.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

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
from pipeline.lanes import ResamplerFactory, Translator, TtsEngine
from pipeline.resample import RateConverter
from pipeline.transcribe import FasterWhisperEngine, TranscribeConfig, whisper_model_dir
from pipeline.translate import CtranslateTranslator, TranslateConfig, translator_model_dir
from pipeline.tts_sapi import (
    ComVoiceBackend,
    TtsConfig,
    VoiceBackend,
    VoiceNotFoundError,
    check_voices,
    sapi_tts_factory,
)
from pipeline.turn import DEFAULT_TURN_MODEL, OnnxTurnClassifier, ThreadedTurnLink, TurnClassifier
from pipeline.ui.controller import BuiltPipeline, describe_error
from pipeline.vad import Segmenter, SileroVad, TurnLink, VadStage

_T = TypeVar("_T")

# A COM call that never answers must not freeze the window for good.
_VOICE_CHECK_TIMEOUT_S = 30.0


# --- speech ----------------------------------------------------------------------------------


def lane_tts_config(settings: Settings, target: str) -> TtsConfig:
    """The voice settings of the lane of ``target``.

    A language that has no route yet (a lane switched at run time) speaks with the default
    voice of its language and the rate and volume of the first lane: ``--tts-rate`` is the
    same for every lane, whereas a voice name belongs to one language.
    """
    for route in settings.routes:
        if route.target == target:
            return route.tts
    first = settings.routes[0].tts if settings.routes else TtsConfig()
    return TtsConfig(rate=first.rate, volume=first.volume)


class _LaneTtsConfigs(Mapping[str, TtsConfig]):
    """``target -> TtsConfig`` for ``sapi_tts_factory``, with ``lane_tts_config``'s fallback."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def __getitem__(self, target: str) -> TtsConfig:
        return lane_tts_config(self._settings, target)

    def __iter__(self) -> Iterator[str]:
        return iter(self._settings.targets)

    def __len__(self) -> int:
        return len(self._settings.targets)


def _sapi_tts(settings: Settings) -> Callable[[str], TtsEngine]:
    return sapi_tts_factory(_LaneTtsConfigs(settings))


@dataclass(frozen=True, slots=True)
class _VoiceAnswer:
    """What the voice check found for one target: a problem text or ``None``."""

    problem: str | None
    final: bool  # False for a COM failure that may pass, so it is asked again


def _check_voice(
    backend_factory: Callable[[], VoiceBackend], target: str, config: TtsConfig
) -> _VoiceAnswer:
    try:
        check_voices({target: config}, backend_factory=backend_factory)
    except VoiceNotFoundError as error:
        return _VoiceAnswer(f"voice missing: {error}", final=True)
    except Exception as error:  # COM or comtypes failed: say so, never take the window down
        return _VoiceAnswer(
            f"speech synthesis unavailable for {target}: {describe_error(error)}", final=False
        )
    return _VoiceAnswer(None, final=True)


def _on_own_thread(job: Callable[[], _T]) -> _T:
    """Run ``job`` on a new thread and wait for it.

    COM is initialised per thread. Doing the voice check here keeps the caller's thread (the
    Tk thread) free of COM, and the apartment ends with this thread.

    Raises:
        TimeoutError: The job did not finish within ``_VOICE_CHECK_TIMEOUT_S``.
    """
    done: list[_T] = []
    failed: list[Exception] = []

    def run() -> None:
        try:
            done.append(job())
        except Exception as error:  # handed back to the caller below
            failed.append(error)

    thread = threading.Thread(target=run, name="voice-check", daemon=True)
    thread.start()
    thread.join(_VOICE_CHECK_TIMEOUT_S)
    if thread.is_alive():
        raise TimeoutError(f"speech synthesis did not answer within {_VOICE_CHECK_TIMEOUT_S:.0f} s")
    if failed:
        raise failed[0]
    return done[0]


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
        tts: ``settings -> (target -> TtsEngine)``; the SAPI engines of the lanes.
        resampler: ``(in rate, out rate) -> Resampler`` from the speech rate to a device rate.
        voice_backend: Makes the SAPI access used by the voice check (``problems`` and the
            check in ``build``); a fake in tests, so no COM is touched.
        files: Raises ``ConfigError`` when a model or the glossary file is missing.
    """

    vad: Callable[[Path], SileroVad] = SileroVad
    turn_classifier: Callable[[Path, int], TurnClassifier] = _load_classifier
    turn_link: Callable[[TurnClassifier], ClosableTurnLink] = ThreadedTurnLink
    whisper: Callable[[Path, TranscribeConfig], FasterWhisperEngine] = _load_whisper
    translator: Callable[[Path, TranslateConfig], Translator] = CtranslateTranslator.load
    glossary: Callable[[Path], Glossary] = Glossary.from_file
    tts: Callable[[Settings], Callable[[str], TtsEngine]] = _sapi_tts
    resampler: ResamplerFactory = RateConverter
    voice_backend: Callable[[], VoiceBackend] = ComVoiceBackend
    files: Callable[[Settings], None] = check_files


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
        # Its own lock: ``_lock`` is held for seconds while a model loads, and ``problems`` runs
        # on the Tk thread, which must not wait for that.
        self._voice_lock = threading.Lock()
        self._voice_answers: dict[tuple[str, str | None], str | None] = {}

    @property
    def tts_notice(self) -> str | None:
        """Always ``None``: a missing voice is a problem (``problems``), not a banner."""
        return None

    def devices(self) -> DeviceListing:
        """The device list source."""
        return self._listing

    def problems(self, settings: Settings) -> tuple[str, ...]:
        """Missing models, files and speech voices, one message each.

        Loads no model and opens no audio. The voice check asks SAPI for the installed voices on
        a helper thread (see the module note) and remembers the answer, so only the first call
        per language (and per voice name) takes a moment; it is called on the Tk thread.
        """
        found: list[str] = []
        try:
            self._loaders.files(settings)
        except ConfigError as error:
            found.extend(str(error).split("; "))
        found.extend(self._voice_problems(settings, settings.targets))
        return tuple(found)

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
            speak = self._loaders.tts(settings)
            pipeline = Pipeline(
                settings,
                devices=self._listing,
                input_stream_factory=self._input_factory,
                output_stream_factory=self._output_factory,
                stage_factory=lambda current: self._stage(current, closers),
                speech_to_text=engine,
                translator_factory=lambda target: self._translator(settings, target),
                tts_factory=lambda target: self._speaker(settings, target, speak, closers),
                glossary=glossary,
                resampler_factory=self._loaders.resampler,
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

    def _voice_problems(self, settings: Settings, targets: tuple[str, ...]) -> list[str]:
        """One message per target whose voice is missing, asking SAPI only for new questions."""
        keys = [(target, lane_tts_config(settings, target).voice) for target in targets]
        with self._voice_lock:
            answers = {key: self._voice_answers[key] for key in keys if key in self._voice_answers}
        new = [key for key in keys if key not in answers]
        if new:
            fresh = self._ask_voices(settings, [target for target, _ in new])
            with self._voice_lock:
                for key, answer in zip(new, fresh, strict=True):
                    answers[key] = answer.problem
                    if answer.final:  # a COM failure is not remembered: the next call asks again
                        self._voice_answers[key] = answer.problem
        return [problem for key in keys if (problem := answers[key]) is not None]

    def _ask_voices(self, settings: Settings, targets: list[str]) -> list[_VoiceAnswer]:
        backend_factory = self._loaders.voice_backend

        def check_all() -> list[_VoiceAnswer]:
            return [
                _check_voice(backend_factory, target, lane_tts_config(settings, target))
                for target in targets
            ]

        try:
            return _on_own_thread(check_all)
        except TimeoutError as error:
            return [_VoiceAnswer(f"speech synthesis unavailable: {error}", final=False)] * len(
                targets
            )

    def _speaker(
        self,
        settings: Settings,
        target: str,
        speak: Callable[[str], TtsEngine],
        closers: list[Callable[[], None]],
    ) -> TtsEngine:
        """The engine of one lane; a language without a voice fails here, before the lane exists.

        Called on the thread that builds the pipeline or switches a lane (the worker), so a
        language chosen at run time is refused with the old lane left working.
        """
        problems = self._voice_problems(settings, (target,))
        if problems:
            raise VoiceNotFoundError(problems[0])
        engine = speak(target)
        close = getattr(engine, "close", None)  # SapiTts releases its COM backends
        if callable(close):
            closers.append(close)
        return engine

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
