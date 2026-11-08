"""Text to speech with the Windows SAPI 5 voices, kept in memory.

SAPI is part of Windows, so nothing is bundled or downloaded: the voices are the ones the user
installed with a language pack (Settings > Time & language > Language > Speech). Speech goes
into an in-memory stream (``SAPI.SpMemoryStream``), never to a file or to the sound card; the
lane plays it through its own output device.

COM objects belong to the thread that made them, and the lane thread is the only one that
speaks, so ``SapiTts`` makes its backend lazily on the thread of the first ``synthesize`` call.
Everything COM-specific sits behind ``VoiceBackend`` so the rest is tested without Windows.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pipeline.lanes import Speech
from pipeline.vad import Float32Array

# SpeechAudioFormatType SAFT22kHz16BitMono. 22.05 kHz is the native rate of the Desktop
# voices, so SAPI does not resample.
PCM_RATE_HZ = 22_050
_SAFT_22KHZ_16BIT_MONO = 22
_PRIMARY_LANGUAGE_MASK = 0x3FF
# Primary language id and the default locale of each language the demo speaks.
_PRIMARY_LANGUAGE = {"en": 0x09, "de": 0x07, "fr": 0x0C}
_DEFAULT_LOCALE = {"en": 0x409, "de": 0x407, "fr": 0x40C}
# SAPI pads every text with about 100 ms of silence before and 800 ms after it. Sentences are
# played one after another, so the padding would be a gap of nearly a second between them and
# a late end of every utterance: keep a short lead-in and a natural sentence pause instead.
_KEEP_LEAD_MS = 20
_KEEP_PAUSE_MS = 150
_SILENCE_LEVEL = 0.003  # about -50 dBFS; the padding is digital silence or close to it


class TtsConfigError(ValueError):
    """A speech setting is out of range; the message names it."""


class VoiceNotFoundError(RuntimeError):
    """No installed SAPI voice fits; the message lists the installed ones and the fix."""


@dataclass(frozen=True, slots=True)
class TtsConfig:
    """Settings of one lane's voice.

    Attributes:
        rate: SAPI speaking rate, -10 (slowest) to 10 (fastest); 0 is the voice's natural speed.
        volume: 0-100.
        voice: Part of a voice name (``"hedda"``) to choose among several voices of the
            language; ``None`` takes the first voice of the target language.
    """

    rate: int = 0
    volume: int = 100
    voice: str | None = None

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not -10 <= self.rate <= 10:
            raise TtsConfigError(f"tts rate must be -10..10, got {self.rate}")
        if not 0 <= self.volume <= 100:
            raise TtsConfigError(f"tts volume must be 0..100, got {self.volume}")
        if self.voice is not None and not self.voice.strip():
            raise TtsConfigError("tts voice must not be blank")


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    """One installed voice.

    Attributes:
        id: SAPI token id, e.g. ``TTS_MS_DE-DE_HEDDA_11.0``.
        name: Display name, e.g. ``Microsoft Hedda Desktop``.
        language_ids: Windows language ids (LCID) the voice speaks, e.g. ``(0x407,)``.
    """

    id: str
    name: str
    language_ids: tuple[int, ...]


class VoiceBackend(Protocol):
    """The part of SAPI that is used; tests replace it."""

    def voices(self) -> list[VoiceInfo]:
        """The installed voices."""
        ...

    def speak(self, voice: VoiceInfo, text: str, rate: int, volume: int) -> bytes:
        """Synthesize ``text`` and return 16-bit little-endian mono PCM at ``PCM_RATE_HZ``."""
        ...

    def close(self) -> None:
        """Release the COM objects."""
        ...


def choose_voice(voices: list[VoiceInfo], target: str, query: str | None) -> VoiceInfo:
    """Pick the voice for ``target`` ("de", "fr", "en").

    A voice of the exact locale (de-DE) comes before a dialect (de-CH). ``query`` narrows the
    choice by part of the name, but only among voices of the right language, so a German lane
    never reads with an English voice.

    Raises:
        VoiceNotFoundError: Nothing fits; the message lists the installed voices.
    """
    primary = _PRIMARY_LANGUAGE.get(target)
    candidates = [
        voice
        for voice in voices
        if primary is not None
        and any(lcid & _PRIMARY_LANGUAGE_MASK == primary for lcid in voice.language_ids)
    ]
    if query is not None:
        wanted = query.strip().casefold()
        candidates = [voice for voice in candidates if wanted in voice.name.casefold()]
    if not candidates:
        raise VoiceNotFoundError(_missing_message(voices, target, query))
    exact = _DEFAULT_LOCALE[target]
    return sorted(candidates, key=lambda voice: exact not in voice.language_ids)[0]


def _missing_message(voices: list[VoiceInfo], target: str, query: str | None) -> str:
    installed = ", ".join(voice.name for voice in voices) or "none"
    what = f"language '{target}'" + (f" matching '{query}'" if query else "")
    return (
        f"no SAPI voice for {what}. Installed voices: {installed}. Install the language pack "
        "with text-to-speech (Settings > Time & language > Language), then restart the program."
    )


def trim_silence(samples: Float32Array, rate_hz: int) -> Float32Array:
    """Cut the silence around speech down to a short lead-in and a short pause.

    Quieter edges than that are left alone; audio that never rises above the silence level
    gives an empty array, which the lane counts as nothing to play.
    """
    loud = np.flatnonzero(np.abs(samples) > _SILENCE_LEVEL)
    if len(loud) == 0:
        return samples[:0]
    start = max(0, int(loud[0]) - rate_hz * _KEEP_LEAD_MS // 1000)
    end = min(len(samples), int(loud[-1]) + 1 + rate_hz * _KEEP_PAUSE_MS // 1000)
    return samples[start:end].copy()


def pcm16_to_float32(raw: bytes) -> Float32Array:
    """Convert 16-bit little-endian PCM to float32 in [-1, 1); an odd trailing byte is dropped."""
    usable = len(raw) - len(raw) % 2
    samples = np.frombuffer(raw, dtype="<i2", count=usable // 2)
    converted: Float32Array = (samples.astype(np.float32) / 32768.0).astype(np.float32)
    return converted


class ComVoiceBackend:
    """SAPI through ``comtypes``; create and use it on one thread only."""

    def __init__(self) -> None:
        """Start COM on this thread and create the voice object (about a second the first time)."""
        import comtypes
        import comtypes.client

        comtypes.CoInitialize()
        self._client = comtypes.client
        self._voice = self._client.CreateObject("SAPI.SpVoice")
        self._tokens: dict[str, object] = {}

    def voices(self) -> list[VoiceInfo]:
        """The installed SAPI 5 voices."""
        tokens = self._voice.GetVoices()
        found: list[VoiceInfo] = []
        for index in range(tokens.Count):
            token = tokens.Item(index)
            ids = tuple(int(part, 16) for part in str(token.GetAttribute("Language")).split(";"))
            info = VoiceInfo(
                str(token.Id).rsplit("\\", 1)[-1], str(token.GetAttribute("Name")), ids
            )
            self._tokens[info.id] = token
            found.append(info)
        return found

    def speak(self, voice: VoiceInfo, text: str, rate: int, volume: int) -> bytes:
        """Synthesize into a fresh memory stream (a reused one keeps the tail of a longer text)."""
        if voice.id not in self._tokens:
            self.voices()
        stream = self._client.CreateObject("SAPI.SpMemoryStream")
        stream.Format.Type = _SAFT_22KHZ_16BIT_MONO
        self._voice.Voice = self._tokens[voice.id]
        self._voice.Rate = rate
        self._voice.Volume = volume
        self._voice.AudioOutputStream = stream
        self._voice.Speak(text, 0)  # 0 = synchronous, default flags
        return bytes(stream.GetData())

    def close(self) -> None:
        """Drop the COM objects; the thread's COM apartment ends with the thread."""
        self._tokens.clear()
        self._voice = None


class SapiTts:
    """``TtsEngine`` of one lane: speaks the translated text of one target language."""

    def __init__(
        self,
        target: str,
        config: TtsConfig | None = None,
        *,
        backend_factory: Callable[[], VoiceBackend] = ComVoiceBackend,
    ) -> None:
        """Create an engine; no COM object exists until the first call.

        Args:
            target: Language of the text, "de", "fr" or "en".
            config: Rate, volume and voice name.
            backend_factory: Builds the SAPI access; called once per thread that speaks.
        """
        self._target = target
        self._config = config or TtsConfig()
        self._backend_factory = backend_factory
        self._lock = threading.Lock()
        self._backends: dict[int, tuple[VoiceBackend, VoiceInfo]] = {}

    def check(self) -> VoiceInfo:
        """Find the voice now, so a missing language pack stops the start, not an utterance.

        Uses a throw-away backend: the lane thread makes its own when it first speaks.

        Raises:
            VoiceNotFoundError: No installed voice fits the target language.
        """
        backend = self._backend_factory()
        try:
            return choose_voice(backend.voices(), self._target, self._config.voice)
        finally:
            backend.close()

    def synthesize(self, text: str) -> Speech:
        """Return the speech for ``text``; blank text gives empty speech.

        Raises:
            VoiceNotFoundError: No installed voice fits the target language.
        """
        if not text.strip():
            return Speech(samples=np.zeros(0, dtype=np.float32), rate_hz=PCM_RATE_HZ)
        backend, voice = self._for_this_thread()
        raw = backend.speak(voice, text, self._config.rate, self._config.volume)
        return Speech(samples=trim_silence(pcm16_to_float32(raw), PCM_RATE_HZ), rate_hz=PCM_RATE_HZ)

    def close(self) -> None:
        """Release every backend."""
        with self._lock:
            backends, self._backends = list(self._backends.values()), {}
        for backend, _ in backends:
            backend.close()

    def _for_this_thread(self) -> tuple[VoiceBackend, VoiceInfo]:
        thread_id = threading.get_ident()
        with self._lock:
            ready = self._backends.get(thread_id)
        if ready is not None:
            return ready
        backend = self._backend_factory()
        try:
            voice = choose_voice(backend.voices(), self._target, self._config.voice)
        except VoiceNotFoundError:
            backend.close()
            raise
        with self._lock:
            self._backends[thread_id] = (backend, voice)
        return backend, voice


def sapi_tts_factory(
    configs: Mapping[str, TtsConfig],
    *,
    backend_factory: Callable[[], VoiceBackend] = ComVoiceBackend,
) -> Callable[[str], SapiTts]:
    """``target -> SapiTts`` for the pipeline; a target without a config gets the defaults."""

    def make(target: str) -> SapiTts:
        return SapiTts(target, configs.get(target), backend_factory=backend_factory)

    return make


def check_voices(
    configs: Mapping[str, TtsConfig],
    *,
    backend_factory: Callable[[], VoiceBackend] = ComVoiceBackend,
) -> dict[str, VoiceInfo]:
    """Find the voice of every target before the pipeline starts.

    Returns:
        The chosen voice per target.

    Raises:
        VoiceNotFoundError: At least one target has no voice; the message lists all of them.
    """
    backend = backend_factory()
    try:
        installed = backend.voices()
    finally:
        backend.close()
    chosen: dict[str, VoiceInfo] = {}
    problems: list[str] = []
    for target, config in configs.items():
        try:
            chosen[target] = choose_voice(installed, target, config.voice)
        except VoiceNotFoundError as exc:
            problems.append(str(exc))
    if problems:
        raise VoiceNotFoundError(" | ".join(problems))
    return chosen
