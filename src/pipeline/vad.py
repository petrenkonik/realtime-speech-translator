"""Voice activity segmentation: turns per-frame speech probabilities into audio chunks.

``Segmenter`` is a pure state machine. It sees one probability and one 512-sample
frame at a time and never touches a model or a clock, so every rule is testable
with scripted probabilities. Time is the stream sample counter.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE_HZ = 16_000
FRAME_SAMPLES = 512
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE_HZ  # 32

ChunkReason = Literal["silence", "max_length", "flush", "turn", "soft", "rescued"]
TurnMode = Literal["fixed", "smart"]
Float32Array = NDArray[np.float32]

_SILENCE_MS_RANGE = (450, 600)  # client requirement
_STALL_FACTOR = 2  # frames since the last speech frame, in units of ``silence_frames``
_MAX_CHECKS_PER_SILENCE = 4  # first check plus rechecks, per silence run
_TURN_WINDOW_FRAMES = 250  # the model looks at the last 8 s = 125 000 / 512 frames
_CHECK_HEADROOM_MS = 200  # a check must start this long before the fixed fallback


class VadConfigError(ValueError):
    """A segmentation setting is outside its allowed range."""


def _frames(ms: float) -> int:
    return math.ceil(ms / FRAME_MS)


def _on_grid(value: float, step: float) -> bool:
    return abs(value / step - round(value / step)) < 1e-6


@dataclass(frozen=True, slots=True)
class SegmenterConfig:
    """Segmentation settings; every range violation raises ``VadConfigError``.

    Attributes:
        silence_ms: Pause that ends a chunk. The client requires 450-600 ms.
        min_speech_ms: Chunks with less speech than this are noise and are dropped.
        pre_roll_ms: Audio kept before the first speech frame so soft onsets survive.
        tail_ms: Silence kept after the last speech frame; the rest is trimmed.
        max_utterance_ms: Longest chunk; longer speech is cut.
        cut_search_ms: A forced cut falls on the quietest frame of this window.
        speech_threshold: Probability at or above which a frame is speech.
        exit_threshold: Probability below which a frame starts the silence clock.
        turn_mode: ``"fixed"`` ends a chunk only by silence; ``"smart"`` also asks the
            Smart Turn model during the silence and may end it earlier.
        turn_check_ms: Silence after which the first model check is requested. Clamped
            to ``silence_ms - 200`` so a check can still finish before the fallback.
        turn_recheck_ms: Cadence of repeated checks inside one silence run.
        turn_threshold: Model probability at or above which the turn is complete.
        soft_cut_ms: Chunk length after which the shorter, stricter soft check applies
            (0 turns soft cuts off).
        soft_pause_ms: Silence that triggers a soft check.
        soft_cut_threshold: Probability a soft check must reach.
    """

    silence_ms: int = 450
    min_speech_ms: int = 160
    pre_roll_ms: int = 100
    tail_ms: int = 160
    max_utterance_ms: int = 15_000
    cut_search_ms: int = 1_500
    speech_threshold: float = 0.5
    exit_threshold: float = 0.35
    turn_mode: TurnMode = "fixed"
    turn_check_ms: int = 200
    turn_recheck_ms: int = 96
    turn_threshold: float = 0.5
    soft_cut_ms: int = 6_000
    soft_pause_ms: int = 128
    soft_cut_threshold: float = 0.7

    def __post_init__(self) -> None:
        """Validate every field."""
        self._validate_turn()
        low, high = _SILENCE_MS_RANGE
        if not low <= self.silence_ms <= high:
            raise VadConfigError(
                f"silence_ms must be between {low} and {high} ms, got {self.silence_ms}"
            )
        if not 0.0 < self.exit_threshold <= self.speech_threshold <= 1.0:
            raise VadConfigError(
                "thresholds must satisfy 0 < exit_threshold <= speech_threshold <= 1, got "
                f"{self.exit_threshold} and {self.speech_threshold}"
            )
        if not FRAME_MS <= self.min_speech_ms <= 2_000:
            raise VadConfigError(f"min_speech_ms must be 32-2000 ms, got {self.min_speech_ms}")
        if not 0 <= self.pre_roll_ms <= 2_000:
            raise VadConfigError(f"pre_roll_ms must be 0-2000 ms, got {self.pre_roll_ms}")
        if not FRAME_MS <= self.tail_ms <= self.silence_ms:
            raise VadConfigError(
                f"tail_ms must be between {FRAME_MS} ms and silence_ms, got {self.tail_ms}"
            )
        if not 5_000 <= self.max_utterance_ms <= 30_000:
            raise VadConfigError(
                f"max_utterance_ms must be 5000-30000 ms, got {self.max_utterance_ms}"
            )
        if not 500 <= self.cut_search_ms <= self.max_utterance_ms // 2:
            raise VadConfigError(
                f"cut_search_ms must be 500 ms up to half of max_utterance_ms, "
                f"got {self.cut_search_ms}"
            )

    @property
    def silence_frames(self) -> int:
        """Frames of silence that end a chunk (rounded up, never shorter than configured)."""
        return _frames(self.silence_ms)

    @property
    def min_speech_frames(self) -> int:
        """Speech frames a chunk needs to be kept."""
        return _frames(self.min_speech_ms)

    @property
    def pre_roll_frames(self) -> int:
        """Frames of audio kept before the first speech frame (rounded down)."""
        return self.pre_roll_ms // FRAME_MS

    @property
    def tail_frames(self) -> int:
        """Frames of trailing silence kept in a chunk."""
        return _frames(self.tail_ms)

    @property
    def max_frames(self) -> int:
        """Longest chunk in frames (rounded down so the cap is never exceeded)."""
        return self.max_utterance_ms // FRAME_MS

    @property
    def cut_search_frames(self) -> int:
        """Frames searched for the quietest cut point."""
        return self.cut_search_ms // FRAME_MS

    def _validate_turn(self) -> None:
        if self.turn_mode not in ("fixed", "smart"):
            raise VadConfigError(f"turn_mode must be 'fixed' or 'smart', got {self.turn_mode!r}")
        if not (100 <= self.turn_check_ms <= 500 and self.turn_check_ms % 50 == 0):
            raise VadConfigError(
                f"turn_check_ms must be 100-500 in steps of 50, got {self.turn_check_ms}"
            )
        if not 64 <= self.turn_recheck_ms <= 320:
            raise VadConfigError(f"turn_recheck_ms must be 64-320 ms, got {self.turn_recheck_ms}")
        if not (0.3 <= self.turn_threshold <= 0.9 and _on_grid(self.turn_threshold, 0.05)):
            raise VadConfigError(
                f"turn_threshold must be 0.3-0.9 in steps of 0.05, got {self.turn_threshold}"
            )
        if self.soft_cut_ms != 0 and not 3_000 <= self.soft_cut_ms <= 12_000:
            raise VadConfigError(
                f"soft_cut_ms must be 0 (off) or 3000-12000, got {self.soft_cut_ms}"
            )
        if not 96 <= self.soft_pause_ms <= 192:
            raise VadConfigError(f"soft_pause_ms must be 96-192 ms, got {self.soft_pause_ms}")
        if not 0.5 <= self.soft_cut_threshold <= 0.95:
            raise VadConfigError(
                f"soft_cut_threshold must be 0.5-0.95, got {self.soft_cut_threshold}"
            )
        if self.soft_cut_threshold < self.turn_threshold:
            raise VadConfigError(
                f"soft_cut_threshold ({self.soft_cut_threshold}) must not be lower than "
                f"turn_threshold ({self.turn_threshold}): a soft cut is the stricter check"
            )

    @property
    def effective_check_ms(self) -> int:
        """``turn_check_ms`` after clamping to ``silence_ms - 200``."""
        return min(self.turn_check_ms, self.silence_ms - _CHECK_HEADROOM_MS)

    @property
    def check_frames(self) -> int:
        """Silent frames before the first model check."""
        return _frames(self.effective_check_ms)

    @property
    def recheck_frames(self) -> int:
        """Frames between repeated checks in one silence run."""
        return _frames(self.turn_recheck_ms)

    @property
    def soft_pause_frames(self) -> int:
        """Silent frames before a soft check."""
        return _frames(self.soft_pause_ms)

    @property
    def soft_cut_frames(self) -> int:
        """Chunk length in frames from which soft checks apply (0 = never)."""
        return self.soft_cut_ms // FRAME_MS


@dataclass(frozen=True, slots=True)
class Chunk:
    """One finalized stretch of speech.

    Attributes:
        id: Sequence number starting at 1; dropped noise does not consume one.
        samples: float32 16 kHz mono audio, read-only, shared by reference.
        start_sample: Stream position of the first sample.
        end_sample: Stream position after the last sample.
        speech_end_sample: Stream position after the last speech frame; the
            latency metric starts here, not at ``end_sample``.
        finalized_sample: Stream position when the decision was made.
        reason: Why the chunk ended.
    """

    id: int
    samples: Float32Array
    start_sample: int
    end_sample: int
    speech_end_sample: int
    finalized_sample: int
    reason: ChunkReason


@dataclass(slots=True)
class _Frame:
    index: int
    samples: Float32Array
    prob: float
    rms: float


@dataclass(frozen=True, slots=True)
class TurnRequest:
    """A question to the Smart Turn model: has the speaker finished?

    Attributes:
        id: Sequence number; an answer is only valid for the request it names.
        window: The last (up to) 8 s of the open chunk, 16 kHz float32, read-only.
        threshold: Probability the answer must reach to end the chunk.
        kind: ``"turn"`` for a normal check, ``"soft"`` for a check inside long speech.
    """

    id: int
    window: Float32Array
    threshold: float
    kind: Literal["turn", "soft"]


@dataclass(frozen=True, slots=True)
class TurnAnswer:
    """The model's answer to one ``TurnRequest``."""

    id: int
    probability: float


class TurnLink(Protocol):
    """How a ``VadStage`` talks to the model without blocking the audio chain."""

    def submit(self, request: TurnRequest) -> None:
        """Hand a request to the model; never blocks."""
        ...

    def poll(self) -> TurnAnswer | None:
        """Return the oldest answer that has arrived, if any; never blocks."""
        ...

    def reset(self) -> None:
        """Forget requests and answers in flight (the stream was interrupted)."""
        ...


class Segmenter:
    """Cuts a stream of frames into chunks using speech probabilities.

    Rules, in order, for every frame of an open chunk:

    * ``prob >= speech_threshold``: speech; clears the silence clock.
    * ``prob < exit_threshold``: starts the silence clock if it is not running.
      Frames between the thresholds are neutral: they neither start nor stop it.
    * The chunk ends when the clock has run for ``silence_ms``, or when no speech
      frame has been seen for ``_STALL_FACTOR`` times that long (noise hovering
      between the thresholds must not hold a chunk open).
    * A chunk reaching ``max_utterance_ms`` is cut at the quietest frame of the
      last ``cut_search_ms``; the rest stays open as the next chunk.

    In ``smart`` mode the silence clock also posts ``TurnRequest`` objects (read them
    with ``take_request``) and an answer passed to ``process`` ends the chunk early when
    it reaches the threshold. The fixed silence rule above stays active as the fallback,
    so a missing or slow model only costs the old behavior. The segmenter never calls the
    model itself and never looks at a clock.
    """

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        """Create a segmenter; ``config`` defaults to ``SegmenterConfig()``."""
        self._config = config or SegmenterConfig()
        self._ring: deque[_Frame] = deque(maxlen=self._config.pre_roll_frames)
        self._open: list[_Frame] | None = None
        self._speech_frames = 0
        self._last_speech_pos = -1
        self._clock_start: int | None = None
        self._continuation = False
        self._next_index = 0
        self._next_id = 1
        self.dropped_short = 0
        # Smart Turn state.
        self._next_request_id = 1
        self._pending_request: TurnRequest | None = None
        self._valid_request: TurnRequest | None = None  # open request of this silence run
        self._inflight = False
        self._run_requests = 0
        self._last_request_run = 0
        self._rescue = False
        self._rescue_clock_index: int | None = None
        self.turn_checks = 0
        self.turn_early_ends = 0
        self.turn_soft_cuts = 0
        self.turn_hard_cuts = 0
        self.turn_rescued = 0
        self.turn_late = 0

    @property
    def config(self) -> SegmenterConfig:
        """The active settings."""
        return self._config

    @property
    def position_samples(self) -> int:
        """Stream position in samples: the end of the last frame processed."""
        return self._next_index * FRAME_SAMPLES

    def take_request(self) -> TurnRequest | None:
        """Return the request posted by the last ``process`` call, once."""
        request, self._pending_request = self._pending_request, None
        return request

    def process(
        self, prob: float, frame: Float32Array, answer: TurnAnswer | None = None
    ) -> list[Chunk]:
        """Consume one frame; return the chunks it completed (zero or one).

        Args:
            prob: Speech probability of ``frame``.
            frame: Exactly 512 samples at 16 kHz.
            answer: The newest model answer, if one arrived since the previous frame.
        """
        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(f"frame must have exactly {FRAME_SAMPLES} samples, got {frame.shape}")
        samples = frame.astype(np.float32, copy=True)
        item = _Frame(
            self._next_index,
            samples,
            prob,
            float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))),
        )
        self._next_index += 1
        config = self._config

        if self._open is None:
            if answer is not None:
                self.turn_late += 1  # the chunk it was about is already finished
            if prob < config.speech_threshold:
                self._ring.append(item)
                return []
            self._open = [*self._ring, item]
            self._ring.clear()
            self._continuation = False
            self._rescue = self._rescue_applies(item.index)
            self._rescue_clock_index = None
            self._rescan()
            return []

        self._open.append(item)
        pos = len(self._open) - 1
        if prob >= config.speech_threshold:
            self._speech_frames += 1
            self._last_speech_pos = pos
            self._clock_start = None
            self._forget_requests()
        elif prob < config.exit_threshold and self._clock_start is None:
            self._clock_start = pos

        if answer is not None:
            early = self._apply_answer(answer, item)
            if early:
                return early

        silence_frames = config.silence_frames
        clock_done = self._clock_start is not None and pos - self._clock_start + 1 >= silence_frames
        stalled = pos - self._last_speech_pos >= _STALL_FACTOR * silence_frames
        if clock_done or stalled:
            return self._finish(
                "silence", finalized_sample=item.index * FRAME_SAMPLES + FRAME_SAMPLES
            )
        if config.turn_mode == "smart":
            self._maybe_request(pos)
        if len(self._open) >= config.max_frames:
            return self._cut_at_quietest(finalized_sample=(item.index + 1) * FRAME_SAMPLES)
        return []

    def flush(self) -> Chunk | None:
        """Finalize the open chunk (stream ended or was interrupted).

        Pre-roll is cleared too: audio from before an interruption must not be
        glued to speech after it.
        """
        chunks: list[Chunk] = []
        if self._open is not None:
            chunks = self._finish("flush", finalized_sample=self._next_index * FRAME_SAMPLES)
        self._ring.clear()
        self._rescue_clock_index = None
        self._pending_request = None
        return chunks[0] if chunks else None

    # --- Smart Turn ------------------------------------------------------------

    def _forget_requests(self) -> None:
        """Speech resumed or the chunk changed: the open request no longer applies."""
        self._valid_request = None
        self._inflight = False
        self._run_requests = 0
        self._last_request_run = 0

    def _apply_answer(self, answer: TurnAnswer, item: _Frame) -> list[Chunk]:
        """Use an answer for the open request, or count it as late. May end the chunk."""
        request = self._valid_request
        if request is None or answer.id != request.id or self._clock_start is None:
            self.turn_late += 1
            return []
        self._inflight = False
        if answer.probability < request.threshold:
            return []
        assert self._open is not None
        self._rescue_clock_index = self._open[self._clock_start].index
        if request.kind == "soft":
            self.turn_soft_cuts += 1
        else:
            self.turn_early_ends += 1
        return self._finish(request.kind, finalized_sample=(item.index + 1) * FRAME_SAMPLES)

    def _maybe_request(self, pos: int) -> None:
        """Post a check (or a recheck) when the silence run is long enough."""
        config = self._config
        if (
            self._clock_start is None
            or self._inflight
            or self._run_requests >= _MAX_CHECKS_PER_SILENCE
            or self._speech_frames < config.min_speech_frames
        ):
            return
        run = pos - self._clock_start + 1
        soft = config.soft_cut_frames > 0 and self._last_speech_pos + 1 >= config.soft_cut_frames
        if self._run_requests == 0:
            due = run >= (config.soft_pause_frames if soft else config.check_frames)
        else:
            due = run - self._last_request_run >= config.recheck_frames
        if not due:
            return
        assert self._open is not None
        window = np.concatenate([f.samples for f in self._open[-_TURN_WINDOW_FRAMES:]])
        window.setflags(write=False)
        request = TurnRequest(
            id=self._next_request_id,
            window=window,
            threshold=config.soft_cut_threshold if soft else config.turn_threshold,
            kind="soft" if soft else "turn",
        )
        self._next_request_id += 1
        self._pending_request = request
        self._valid_request = request
        self._inflight = True
        self._run_requests += 1
        self._last_request_run = run
        self.turn_checks += 1

    def _rescue_applies(self, speech_index: int) -> bool:
        """Would fixed mode still have kept this speech in the chunk that ended early?"""
        clock_index = self._rescue_clock_index
        return clock_index is not None and speech_index - clock_index < self._config.silence_frames

    # --- internals -------------------------------------------------------------

    def _rescan(self) -> None:
        """Recompute the counters of the open chunk from its frames."""
        assert self._open is not None
        config = self._config
        self._speech_frames = 0
        self._last_speech_pos = -1
        self._clock_start = None
        self._forget_requests()
        for pos, frame in enumerate(self._open):
            if frame.prob >= config.speech_threshold:
                self._speech_frames += 1
                self._last_speech_pos = pos
                self._clock_start = None
            elif frame.prob < config.exit_threshold and self._clock_start is None:
                self._clock_start = pos

    def _build(
        self, frames: list[_Frame], reason: ChunkReason, speech_end_pos: int, finalized_sample: int
    ) -> Chunk:
        samples = np.concatenate([f.samples for f in frames])
        samples.setflags(write=False)
        speech_end = frames[min(speech_end_pos, len(frames) - 1)].index + 1
        chunk = Chunk(
            id=self._next_id,
            samples=samples,
            start_sample=frames[0].index * FRAME_SAMPLES,
            end_sample=(frames[-1].index + 1) * FRAME_SAMPLES,
            speech_end_sample=speech_end * FRAME_SAMPLES,
            finalized_sample=finalized_sample,
            reason=reason,
        )
        self._next_id += 1
        return chunk

    def _finish(self, reason: ChunkReason, finalized_sample: int) -> list[Chunk]:
        """End the open chunk: trim trailing silence, apply the min-speech rule."""
        assert self._open is not None
        frames = self._open
        keep = len(frames)
        if self._last_speech_pos >= 0:
            keep = min(keep, self._last_speech_pos + 1 + self._config.tail_frames)
        short = self._speech_frames < self._config.min_speech_frames
        rescued = short and self._rescue and not self._continuation and self._speech_frames > 0
        too_short = short and not rescued and not (self._continuation and self._speech_frames > 0)
        chunks: list[Chunk] = []
        if too_short:
            self.dropped_short += 1
        else:
            if rescued:
                reason = "rescued"
                self.turn_rescued += 1
            chunks.append(
                self._build(frames[:keep], reason, max(self._last_speech_pos, 0), finalized_sample)
            )
        # Frames that were not emitted may serve as pre-roll for the next chunk.
        self._ring.extend(frames[keep:] if not too_short else frames)
        self._open = None
        self._continuation = False
        self._rescue = False
        self._forget_requests()
        return chunks

    def _cut_at_quietest(self, finalized_sample: int) -> list[Chunk]:
        """Cut an over-long chunk where the audio is quietest; keep the rest open."""
        assert self._open is not None
        frames = self._open
        window = frames[-self._config.cut_search_frames :]
        quietest = min(range(len(window)), key=lambda i: (window[i].rms, -i))
        cut = len(frames) - len(window) + quietest + 1
        head, rest = frames[:cut], frames[cut:]
        speech_end_pos = max(
            (p for p, f in enumerate(head) if f.prob >= self._config.speech_threshold),
            default=len(head) - 1,
        )
        chunk = self._build(head, "max_length", speech_end_pos, finalized_sample)
        self.turn_hard_cuts += 1
        self._open = rest
        self._continuation = True
        self._rescan()
        return [chunk]


# --- Silero VAD -------------------------------------------------------------------

_CONTEXT_SAMPLES = 64  # Silero v5+ expects the last 64 samples of the previous window in front
_STATE_SHAPE = (2, 1, 128)


class ModelNotFoundError(FileNotFoundError):
    """The Silero model file is missing."""


class OnnxSession(Protocol):
    """The part of ``onnxruntime.InferenceSession`` that ``SileroVad`` uses."""

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        """Run the model."""
        ...


def _open_session(model_path: Path) -> OnnxSession:
    if not model_path.is_file():
        raise ModelNotFoundError(
            f"Silero VAD model not found at {model_path}; run scripts/fetch_models.py first"
        )
    # Imported here: it is heavy and the Segmenter tests must not need it.
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session: OnnxSession = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return session


class SileroVad:
    """Speech probability for consecutive 512-sample frames at 16 kHz.

    The model is recurrent: its state and a 64-sample context persist across
    frames, so frames must be fed in order. Call ``reset`` after any gap.
    """

    def __init__(self, model_path: Path, session: OnnxSession | None = None) -> None:
        """Load the model from a local path (never from the network)."""
        self._session = session if session is not None else _open_session(model_path)
        self._rate = np.array(SAMPLE_RATE_HZ, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Forget the recurrent state and the context."""
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: Float32Array) -> float:
        """Return the probability that ``frame`` contains speech."""
        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(f"frame must have exactly {FRAME_SAMPLES} samples, got {frame.shape}")
        window = np.concatenate([self._context, frame.astype(np.float32, copy=False)])[np.newaxis]
        out, state = self._session.run(
            None, {"input": window, "state": self._state, "sr": self._rate}
        )
        self._state = np.asarray(state, dtype=np.float32)
        self._context = window[0, -_CONTEXT_SAMPLES:].copy()
        return float(np.asarray(out).reshape(-1)[0])


class VadStage:
    """Arbitrary-length audio in, chunks out: frame alignment + Silero + ``Segmenter``."""

    def __init__(
        self, vad: SileroVad, segmenter: Segmenter, turn_link: TurnLink | None = None
    ) -> None:
        """Wire a model and a segmenter together.

        Args:
            vad: Frame-level speech probability model.
            segmenter: The state machine.
            turn_link: Channel to the Smart Turn model; required when the segmenter
                runs in ``smart`` mode and ignored otherwise.

        Raises:
            VadConfigError: Smart mode without a ``turn_link``.
        """
        if segmenter.config.turn_mode == "smart" and turn_link is None:
            raise VadConfigError("turn_mode 'smart' needs a turn link (or use turn_mode 'fixed')")
        self._vad = vad
        self._segmenter = segmenter
        self._link = turn_link if segmenter.config.turn_mode == "smart" else None
        self._pending = np.zeros(0, dtype=np.float32)

    @property
    def segmenter(self) -> Segmenter:
        """The wrapped segmenter (for its counters)."""
        return self._segmenter

    @property
    def position_samples(self) -> int:
        """Stream position in samples including the partial frame being collected.

        Chunk coordinates (``Chunk.end_sample`` and friends) are on this clock.
        """
        return self._segmenter.position_samples + len(self._pending)

    def push(self, samples: Float32Array) -> list[Chunk]:
        """Feed any number of 16 kHz samples; return the chunks completed by them."""
        data = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        usable = len(data) // FRAME_SAMPLES * FRAME_SAMPLES
        self._pending = data[usable:].copy()
        chunks: list[Chunk] = []
        for start in range(0, usable, FRAME_SAMPLES):
            frame = data[start : start + FRAME_SAMPLES]
            answer = self._link.poll() if self._link is not None else None
            chunks += self._segmenter.process(self._vad(frame), frame, answer)
            if self._link is not None:
                request = self._segmenter.take_request()
                if request is not None:
                    self._link.submit(request)
        return chunks

    def flush(self) -> Chunk | None:
        """End of stream or interruption: finalize, drop the partial frame, reset the model."""
        chunk = self._segmenter.flush()
        self._pending = np.zeros(0, dtype=np.float32)
        self._vad.reset()
        if self._link is not None:
            self._link.reset()
        return chunk
