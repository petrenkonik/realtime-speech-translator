"""Whisper consumer: one decode per utterance, text only in RAM.

``TranscribeStage`` is the only ``ChunkBus`` subscriber that leads to translation. It turns
finished chunks into ``Utterance`` objects on the ``TextBus``. Around the decode it handles
four of the chunk risks of the spec:

* C8  noise and hallucinations: an RMS gate before the decode, whisper's own
  no-speech / log-probability thresholds during it, and a repeated-phrase filter after it.
* C9  falling behind: chunks that piled up while whisper was busy and are close together
  are decoded in one pass instead of being dropped by the bus.
* C11 a sentence cut in the middle: the end of the previous text goes to whisper as
  ``initial_prompt`` so the next chunk continues it.
* C10 is a measurement (M5), not code.

The decode itself sits behind ``SpeechToText`` so tests need no model.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import threading
import time
import types
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import NoReturn, Protocol

import numpy as np

from pipeline.bus import Bus, Subscription
from pipeline.glossary import Glossary
from pipeline.metrics import Metrics
from pipeline.vad import SAMPLE_RATE_HZ, Chunk, ChunkReason, Float32Array, ModelNotFoundError

_SAMPLES_PER_MS = SAMPLE_RATE_HZ // 1000

# English-only Whisper models in ``models/whisper-<name>/`` (scripts/fetch_models.py). The
# default is the more accurate one; ``tiny.en`` decodes about twice as fast with about 1.8
# points more word errors on read speech (SPEC-LOW-LATENCY.md section 12.4).
WHISPER_MODELS = ("tiny.en", "base.en")
DEFAULT_WHISPER_MODEL = "base.en"
_MAX_REPEAT_PHRASE_WORDS = 8
_FORCED_CUTS: frozenset[ChunkReason] = frozenset({"max_length", "soft", "rescued"})
_SENTENCE_END = re.compile(r"[.!?…][\"')\]”]*$")
_POLL_S = 0.1


class TranscribeConfigError(ValueError):
    """A transcription setting is outside its allowed range; the message names it."""


@dataclass(frozen=True, slots=True)
class TranscribeConfig:
    """Settings of the transcription stage.

    Attributes:
        model: Which model to load, one of ``WHISPER_MODELS``.
        language: Source language code passed to whisper (the demo is English only).
        beam_size: Beam width; 1 is greedy and the fastest.
        threads: CPU threads for the decode.
        no_speech_threshold: A segment is dropped when whisper's no-speech probability is
            above this and its average log probability is below ``log_prob_threshold``.
        log_prob_threshold: See above (<= 0).
        compression_ratio_threshold: Stored for the adapter; with a single decode
            temperature whisper has nothing to re-decode, so repetition is handled by
            ``repeat_limit``.
        min_rms: Chunks quieter than this (float32 RMS) never reach whisper.
        repeat_limit: A phrase repeated this many times in a row is collapsed to one.
        merge_gap_ms: Waiting chunks separated by at most this much are decoded together.
        merge_max_ms: Longest merged audio (whisper's window is 30 s).
        merge_pad_ms: Silence inserted between merged chunks, at most the real gap.
        prompt_tail_chars: How much of the previous text is passed on as context.
        context_gap_ms: The context is used only if the next chunk starts this soon.
    """

    model: str = DEFAULT_WHISPER_MODEL
    language: str = "en"
    beam_size: int = 1
    threads: int = 4
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    min_rms: float = 0.002
    repeat_limit: int = 4
    merge_gap_ms: int = 1000
    merge_max_ms: int = 15_000
    merge_pad_ms: int = 250
    prompt_tail_chars: int = 200
    context_gap_ms: int = 2000

    def __post_init__(self) -> None:
        """Reject values that cannot work."""
        _require(
            self.model in WHISPER_MODELS,
            f"model must be one of {', '.join(WHISPER_MODELS)}, got {self.model!r}",
        )
        _require(bool(self.language.strip()), "language must not be empty")
        _require(self.beam_size >= 1, "beam_size must be at least 1")
        _require(1 <= self.threads <= 32, "threads must be between 1 and 32")
        _require(0.0 <= self.no_speech_threshold <= 1.0, "no_speech_threshold must be 0-1")
        _require(self.log_prob_threshold <= 0.0, "log_prob_threshold must not be positive")
        _require(self.compression_ratio_threshold >= 1.0, "compression_ratio_threshold >= 1")
        _require(self.min_rms >= 0.0, "min_rms must not be negative")
        _require(self.repeat_limit >= 2, "repeat_limit must be at least 2")
        _require(self.merge_gap_ms >= 0, "merge_gap_ms must not be negative")
        _require(1000 <= self.merge_max_ms <= 30_000, "merge_max_ms must be 1000-30000")
        _require(self.merge_pad_ms >= 0, "merge_pad_ms must not be negative")
        _require(self.prompt_tail_chars >= 0, "prompt_tail_chars must not be negative")
        _require(self.context_gap_ms >= 0, "context_gap_ms must not be negative")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TranscribeConfigError(message)


def whisper_model_dir(models_dir: Path, config: TranscribeConfig) -> Path:
    """The folder of the model that ``config.model`` names, e.g. ``models/whisper-base.en``."""
    return models_dir / f"whisper-{config.model}"


@dataclass(frozen=True, slots=True)
class Utterance:
    """The text of one or more chunks, on its way to the translation lanes.

    Attributes:
        id: Id of the first chunk it covers.
        chunk_ids: Every chunk that went into the decode (more than one after a merge).
        text: The transcript; lives in RAM only.
        start_sample: Stream position where the first chunk starts.
        speech_end_sample: Stream position after the last speech frame; the end-to-end
            latency metric starts here.
        finalized_sample: Stream position when the last chunk was cut.
        audio_ms: Length of the audio that was decoded.
        decode_ms: Wall-clock time of the decode.
        reason: Why the last chunk ended.
    """

    id: int
    chunk_ids: tuple[int, ...]
    text: str
    start_sample: int
    speech_end_sample: int
    finalized_sample: int
    audio_ms: int
    decode_ms: float
    reason: ChunkReason


TextBus = Bus[Utterance]
"""Bus of utterance texts (whisper -> one subscriber per language lane)."""


class SpeechToText(Protocol):
    """The decode, behind an interface so the stage can be tested without a model."""

    def transcribe(
        self, samples: Float32Array, *, hotwords: str | None, initial_prompt: str | None
    ) -> str:
        """Return the text of 16 kHz mono float32 audio, or an empty string."""
        ...


# --- faster-whisper adapter ---------------------------------------------------------------


class _MissingAv(types.ModuleType):
    """Stands in for PyAV, which only decodes audio files; this program never does."""

    def __getattr__(self, name: str) -> NoReturn:
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError(
            "PyAV is not installed on purpose (its FFmpeg build contains LGPL code), so audio "
            "files cannot be decoded. Pass float32 arrays to whisper instead."
        )


def ensure_av_stub() -> None:
    """Make ``import av`` succeed without PyAV.

    ``faster_whisper.audio`` imports ``av`` at load time but only uses it to decode files.
    The project passes arrays, so the stub is never touched; if something does touch it, it
    fails with an explanation instead of an ``AttributeError``. A real PyAV is left alone.
    """
    if "av" in sys.modules or importlib.util.find_spec("av") is not None:
        return
    sys.modules["av"] = _MissingAv("av")


class _Segment(Protocol):
    text: str


class WhisperModelLike(Protocol):
    """The part of ``faster_whisper.WhisperModel`` the engine uses."""

    def transcribe(
        self, audio: Float32Array, **options: object
    ) -> tuple[Iterable[_Segment], object]:
        """Decode ``audio``; the segments are produced lazily while they are iterated."""
        ...


class FasterWhisperEngine:
    """``SpeechToText`` on a faster-whisper model."""

    def __init__(self, model: WhisperModelLike, config: TranscribeConfig) -> None:
        """Wrap an already loaded model (``WhisperModel`` or a test double)."""
        self._model = model
        self._config = config

    @classmethod
    def load(cls, model_dir: Path, config: TranscribeConfig) -> FasterWhisperEngine:
        """Load the model from a local directory; nothing is downloaded.

        Raises:
            ModelNotFoundError: The directory is missing or incomplete.
        """
        if not (model_dir / "model.bin").is_file():
            raise ModelNotFoundError(
                f"whisper model not found in {model_dir}; run scripts/fetch_models.py"
            )
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # belt and braces next to local_files_only
        ensure_av_stub()
        from faster_whisper import WhisperModel

        model = WhisperModel(
            str(model_dir),
            device="cpu",
            compute_type="int8",
            cpu_threads=config.threads,
            num_workers=1,
            local_files_only=True,
        )
        return cls(model, config)

    def with_config(self, config: TranscribeConfig) -> FasterWhisperEngine:
        """Return an engine on the same loaded model with other decode options.

        Loading the model is the slow part and depends only on ``config.model`` and
        ``config.threads``; the window uses this to apply a changed beam size or threshold
        without loading it again.
        """
        return FasterWhisperEngine(self._model, config)

    def warm_up(self) -> None:
        """Decode a second of silence so the first real utterance does not pay for set-up."""
        self.transcribe(
            np.zeros(SAMPLE_RATE_HZ, dtype=np.float32), hotwords=None, initial_prompt=None
        )

    def transcribe(
        self, samples: Float32Array, *, hotwords: str | None, initial_prompt: str | None
    ) -> str:
        """Decode ``samples``; see ``SpeechToText``."""
        config = self._config
        segments, _ = self._model.transcribe(
            samples,
            language=config.language,
            beam_size=config.beam_size,
            # One temperature: no re-decoding with random sampling, so the time is bounded.
            temperature=0.0,
            vad_filter=False,  # our own VAD already cut the chunk
            condition_on_previous_text=False,  # context comes from ``initial_prompt``
            without_timestamps=True,
            no_speech_threshold=config.no_speech_threshold,
            log_prob_threshold=config.log_prob_threshold,
            compression_ratio_threshold=config.compression_ratio_threshold,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )
        parts = (segment.text.strip() for segment in segments)  # the generator does the decode
        return " ".join(part for part in parts if part)


# --- text helpers -------------------------------------------------------------------------


def collapse_repeats(text: str, limit: int) -> str:
    """Collapse a phrase that is repeated ``limit`` or more times in a row to one copy.

    Whisper sometimes loops on noise ("Thank you. Thank you. Thank you. ..."). Fewer
    repeats than ``limit`` are left alone because people do say "no no no".
    """
    words = text.split()
    keys = [_word_key(word) for word in words]
    result: list[str] = []
    index = 0
    while index < len(words):
        size, runs = _longest_loop(keys, index, limit)
        if runs >= limit:
            result.extend(words[index : index + size])
            index += runs * size
        else:
            result.append(words[index])
            index += 1
    return " ".join(result)


def _longest_loop(keys: list[str], index: int, limit: int) -> tuple[int, int]:
    """The shortest phrase starting at ``index`` that repeats ``limit`` times, as (size, runs)."""
    for size in range(1, _MAX_REPEAT_PHRASE_WORDS + 1):
        phrase = keys[index : index + size]
        if len(phrase) < size:
            break
        runs = 1
        while keys[index + runs * size : index + (runs + 1) * size] == phrase:
            runs += 1
        if runs >= limit:
            return size, runs
    return 0, 0


def _word_key(word: str) -> str:
    return word.strip(".,!?;:\"'()[]").casefold()


def _tail(text: str, max_chars: int) -> str:
    """The last ``max_chars`` of ``text``, starting at a word boundary."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[-max_chars:]
    if text[-max_chars - 1] != " " and " " in cut:
        cut = cut.split(" ", 1)[1]
    return cut.strip()


# --- the stage ----------------------------------------------------------------------------


class TranscribeStage:
    """Reads chunks, decodes them, publishes utterances."""

    def __init__(
        self,
        chunks: Subscription[Chunk],
        out: TextBus,
        engine: SpeechToText,
        config: TranscribeConfig | None = None,
        *,
        glossary: Glossary | None = None,
        metrics: Metrics | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Create a stopped stage.

        Args:
            chunks: The whisper subscription of the ``ChunkBus``.
            out: The ``TextBus``.
            engine: The decode.
            config: Settings; defaults to ``TranscribeConfig()``.
            glossary: Optional; its source terms are passed to whisper as hotwords.
            metrics: Optional registry for counters and histograms.
            clock: Monotonic seconds; injected so tests control the decode time.
        """
        self._chunks = chunks
        self._out = out
        self._engine = engine
        self._config = config or TranscribeConfig()
        self._hotwords = glossary.hotwords() if glossary is not None else None
        self._metrics = metrics or Metrics()
        self._clock = clock
        self._waiting: deque[Chunk] = deque()
        self._previous_text = ""
        self._previous_end_sample = 0
        self._previous_forced = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: Exception | None = None

    @property
    def finished(self) -> bool:
        """True once the bus is closed and every chunk has been handled."""
        return not self._waiting and self._chunks.finished

    @property
    def running(self) -> bool:
        """True while the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the worker thread."""
        if self.running:
            raise RuntimeError("the transcription stage is already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="transcribe", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Ask the thread to end after the decode in hand and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            self._thread = None

    def step(self, timeout_s: float = _POLL_S) -> bool:
        """Handle the next group of chunks.

        Args:
            timeout_s: How long to wait for a chunk when none is waiting.

        Returns:
            True if a group was handled, False if there was nothing to do.
        """
        self._collect(timeout_s)
        if not self._waiting:
            return False
        self._handle(self._take_group())
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.step() and self.finished:
                break

    def _collect(self, timeout_s: float) -> None:
        # Always empty the bus into our own list: a full queue would lose the oldest speech.
        if not self._waiting:
            first = self._chunks.get(timeout_s)
            if first is None:
                return
            self._waiting.append(first)
        while (chunk := self._chunks.get(0)) is not None:
            self._waiting.append(chunk)

    def _take_group(self) -> list[Chunk]:
        config = self._config
        group = [self._waiting.popleft()]
        total = len(group[0].samples)
        while self._waiting:
            nxt = self._waiting[0]
            gap_ms = max(0, (nxt.start_sample - group[-1].end_sample) // _SAMPLES_PER_MS)
            pad = min(gap_ms, config.merge_pad_ms) * _SAMPLES_PER_MS
            if gap_ms > config.merge_gap_ms:
                break
            if (total + pad + len(nxt.samples)) > config.merge_max_ms * _SAMPLES_PER_MS:
                break
            total += pad + len(nxt.samples)
            group.append(self._waiting.popleft())
        return group

    def _audio(self, group: list[Chunk]) -> Float32Array:
        if len(group) == 1:
            return group[0].samples
        pieces: list[Float32Array] = [group[0].samples]
        for before, after in pairwise(group):
            gap_ms = max(0, (after.start_sample - before.end_sample) // _SAMPLES_PER_MS)
            pad = min(gap_ms, self._config.merge_pad_ms) * _SAMPLES_PER_MS
            pieces.append(np.zeros(pad, dtype=np.float32))
            pieces.append(after.samples)
        return np.concatenate(pieces)

    def _context(self, first: Chunk) -> str | None:
        config = self._config
        if not self._previous_text or config.prompt_tail_chars == 0:
            return None
        gap_ms = (first.start_sample - self._previous_end_sample) // _SAMPLES_PER_MS
        if gap_ms > config.context_gap_ms:
            return None
        if not self._previous_forced and _SENTENCE_END.search(self._previous_text):
            return None
        return _tail(self._previous_text, config.prompt_tail_chars) or None

    def _handle(self, group: list[Chunk]) -> None:
        metrics = self._metrics
        metrics.counter("transcribe.chunks").inc(len(group))
        if len(group) > 1:
            metrics.counter("transcribe.merged_chunks").inc(len(group) - 1)
        samples = self._audio(group)
        if float(np.sqrt(np.mean(np.square(samples)))) < self._config.min_rms:
            metrics.counter("transcribe.skipped_quiet").inc()
            return
        started = self._clock()
        try:
            text = self._engine.transcribe(
                samples, hotwords=self._hotwords, initial_prompt=self._context(group[0])
            )
        except Exception as error:  # one bad chunk must not stop the interpreter
            self.last_error = error
            metrics.counter("transcribe.errors").inc()
            return
        decode_s = self._clock() - started
        audio_ms = len(samples) // _SAMPLES_PER_MS
        metrics.histogram("transcribe.decode_ms").record(decode_s * 1000.0)
        metrics.histogram("transcribe.rtf").record(decode_s * 1000.0 / max(audio_ms, 1))
        collapsed = collapse_repeats(text, self._config.repeat_limit)
        if len(collapsed.split()) < len(text.split()):
            metrics.counter("transcribe.repeats_collapsed").inc()
        if not collapsed:
            metrics.counter("transcribe.skipped_empty").inc()
            return
        last = group[-1]
        self._previous_text = collapsed
        self._previous_end_sample = last.speech_end_sample
        self._previous_forced = last.reason in _FORCED_CUTS
        metrics.counter("transcribe.utterances").inc()
        self._out.publish(
            Utterance(
                id=group[0].id,
                chunk_ids=tuple(chunk.id for chunk in group),
                text=collapsed,
                start_sample=group[0].start_sample,
                speech_end_sample=last.speech_end_sample,
                finalized_sample=last.finalized_sample,
                audio_ms=audio_ms,
                decode_ms=decode_s * 1000.0,
                reason=last.reason,
            )
        )
