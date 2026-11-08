"""Labelled phrase-end evaluation on streams of human speech.

A stream is speech segments separated by silence. Every segment is labelled from its
TEXT, never from where a speaker happened to pause:

* ``end``: the sentence is finished (LibriSpeech-PC punctuation ends it with . ? !);
* ``split``: a complete sentence cut at a word gap; the head ends inside the sentence,
  so it is a thinking pause, not a phrase end;
* ``unfinished``: a sentence that stops mid-way (no sentence-final punctuation).

Scoring: a chunk boundary (``Chunk.speech_end_sample``) is correct when it lies within
``TOLERANCE_MS`` of a phrase end. One boundary matches at most one phrase end. A phrase
end without a boundary is a miss, and a boundary inside a mid-sentence pause is premature.
Precision is the share of boundaries that match a phrase end; recall is the share of phrase
ends that got a boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pipeline.vad import FRAME_SAMPLES

Float32Array = NDArray[np.float32]
UnitKind = Literal["end", "split", "unfinished"]

RATE_HZ = 16_000
RMS_FRAME = 160  # 10 ms: the grain of the labels and of the word-gap search
PAUSES_MS: tuple[int, ...] = (200, 300, 400, 500, 650, 800)  # silence after each segment
FINAL_PAUSE_MS = 1_000  # closes the stream so the last phrase end is always cut
LEAD_IN_MS = 700
NOISE_STD = 0.001  # about -60 dBFS, the level the Silero VAD treats as silence
RELATIVE_LOUD = 0.1  # speech frame: louder than this share of the 95th-percentile frame RMS
MIN_LOUD_RMS = 0.005
MIN_GAP_FRAMES = 6  # 60 ms of quiet inside a sentence counts as a word gap
# How far a boundary may sit from a phrase end and still match it. Silero releases a quiet
# tail (breath, soft consonant) up to ~0.4 s after the energy-based end label, so 300 ms
# misses real matches; 500 ms still keeps the next phrase's pause outside the window.
TOLERANCE_MS = 500
TOLERANCE_SAMPLES = TOLERANCE_MS * RATE_HZ // 1000
GROUP_END, GROUP_SPLIT = 0.40, 0.75  # group mix: plain end / split sentence / unfinished


@dataclass(frozen=True, slots=True)
class Unit:
    """One labelled segment of a stream.

    Attributes:
        kind: ``end``, ``split`` or ``unfinished`` (see the module docstring).
        end_sample: Stream position of the last speech sample of the segment.
        pause_ms: Silence that follows the segment in the stream.
    """

    kind: UnitKind
    end_sample: int
    pause_ms: int

    @property
    def is_end(self) -> bool:
        """True for a phrase end, False for a mid-sentence segment."""
        return self.kind == "end"


@dataclass(frozen=True, slots=True)
class Stream:
    """Audio plus the labels of its segments, in stream order."""

    audio: Float32Array
    units: tuple[Unit, ...]


@dataclass(frozen=True, slots=True)
class Cut:
    """A chunk boundary from the endpointer.

    Attributes:
        boundary_sample: ``Chunk.speech_end_sample``, the last speech sample of the chunk.
        finalized_sample: Stream position when the decision was made.
    """

    boundary_sample: int
    finalized_sample: int


@dataclass(frozen=True, slots=True)
class UnitResult:
    """The outcome for one labelled segment.

    Attributes:
        unit: The label.
        hit: A phrase end matched by a cut, or a mid-sentence segment that got a cut.
        latency_ms: Phrase ends only: decision time minus speech end, when matched.
        offset_ms: Phrase ends only: chunk boundary minus speech end, when matched
            (negative when the VAD released the speech before the label).
    """

    unit: Unit
    hit: bool
    latency_ms: float | None = None
    offset_ms: float | None = None


@dataclass(frozen=True, slots=True)
class StreamScore:
    """Scores of one stream: per-segment outcomes and the cut counts."""

    results: tuple[UnitResult, ...]
    cuts: int
    matched_cuts: int


@dataclass(frozen=True, slots=True)
class Summary:
    """Totals over many streams, with breakdowns by pause length and by segment kind."""

    ends: int
    ends_hit: int
    cuts: int
    cuts_matched: int
    nonends: int
    nonends_cut: int
    latencies_ms: tuple[float, ...]
    recall_by_pause: dict[int, tuple[int, int]]  # pause -> (phrase ends hit, phrase ends)
    cut_by_pause: dict[int, tuple[int, int]]  # pause -> (mid-sentence segments cut, total)
    cut_by_kind: dict[str, tuple[int, int]]  # kind -> (mid-sentence segments cut, total)
    offsets_ms: tuple[float, ...] = ()  # boundary minus speech end, matched phrase ends

    @property
    def precision(self) -> float:
        """Share of cuts that matched a phrase end (NaN without cuts)."""
        return _ratio(self.cuts_matched, self.cuts)

    @property
    def recall(self) -> float:
        """Share of phrase ends that got a cut (NaN without phrase ends)."""
        return _ratio(self.ends_hit, self.ends)

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall (NaN when either is undefined)."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r > 0 else float("nan")

    @property
    def premature_rate(self) -> float:
        """Share of mid-sentence segments that got a cut (NaN without them)."""
        return _ratio(self.nonends_cut, self.nonends)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def _frame_rms(samples: Float32Array) -> NDArray[np.float64]:
    frames = len(samples) // RMS_FRAME
    shaped = samples[: frames * RMS_FRAME].astype(np.float64).reshape(frames, RMS_FRAME)
    return np.sqrt(np.mean(shaped**2, axis=1))


def _loud_threshold(rms: NDArray[np.float64]) -> float:
    if rms.size == 0:
        return MIN_LOUD_RMS
    return max(MIN_LOUD_RMS, RELATIVE_LOUD * float(np.percentile(rms, 95)))


def trim(samples: Float32Array) -> Float32Array:
    """Cut leading and trailing quiet, so the segment starts and ends on speech.

    The threshold is relative to the utterance's own level, so quiet and loud recordings
    are treated alike. Returns an empty array when nothing is loud.
    """
    rms = _frame_rms(samples)
    loud = np.flatnonzero(rms > _loud_threshold(rms))
    if loud.size == 0:
        return np.zeros(0, dtype=np.float32)
    return samples[loud[0] * RMS_FRAME : (loud[-1] + 1) * RMS_FRAME].astype(np.float32, copy=True)


def split_at_gap(samples: Float32Array) -> tuple[Float32Array, Float32Array]:
    """Split speech at a word gap in its middle 40 %, and trim both parts.

    The gap is the longest run of quiet frames (at least ``MIN_GAP_FRAMES``) in that
    region; the cut goes to its centre. Without such a gap, the quietest frame is used.
    Empty parts mean the speech is too short to split.
    """
    rms = _frame_rms(samples)
    quiet = rms <= _loud_threshold(rms)
    low, high = int(0.3 * len(rms)), int(0.7 * len(rms))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for frame in range(low, high):
        if quiet[frame]:
            if start is None:
                start = frame
        elif start is not None:
            runs.append((start, frame - start))
            start = None
    if start is not None:
        runs.append((start, high - start))
    if runs and max(length for _, length in runs) >= MIN_GAP_FRAMES:
        first, length = max(runs, key=lambda run: run[1])
        cut_frame = first + length // 2
    elif high > low:
        cut_frame = low + int(np.argmin(rms[low:high]))
    else:
        return trim(samples), np.zeros(0, dtype=np.float32)
    cut_sample = cut_frame * RMS_FRAME
    return trim(samples[:cut_sample]), trim(samples[cut_sample:])


def _noise(rng: np.random.Generator, ms: int) -> Float32Array:
    return (rng.standard_normal(ms * RATE_HZ // 1000) * NOISE_STD).astype(np.float32)


def _pick(items: Sequence[Float32Array], rng: np.random.Generator) -> Float32Array:
    return items[int(rng.integers(len(items)))]


def build_stream(
    ends: Sequence[Float32Array],
    mids: Sequence[Float32Array],
    rng: np.random.Generator,
    groups: int = 5,
) -> Stream:
    """Join labelled utterances with silences of known length.

    Each group is one of: a plain phrase end; a sentence split at a word gap (a mid-sentence
    head, then the phrase end); or an unfinished segment followed by a phrase end. Every
    segment gets a pause from ``PAUSES_MS``; the last one gets ``FINAL_PAUSE_MS``.

    Args:
        ends: Trimmed utterances that end a sentence, 16 kHz mono float32.
        mids: Trimmed utterances that stop mid-sentence.
        rng: Source of every random choice, so streams are reproducible.
        groups: Number of groups in the stream.

    Returns:
        The stream, padded to whole VAD frames, with one label per segment.

    Raises:
        ValueError: No phrase-end utterance was given.
    """
    if not ends:
        raise ValueError("build_stream needs at least one phrase-end utterance")
    segments: list[tuple[UnitKind, Float32Array]] = []
    for _ in range(groups):
        roll = float(rng.random())
        if roll < GROUP_END or not mids:
            segments.append(("end", _pick(ends, rng)))
            continue
        if roll < GROUP_SPLIT:
            head, tail = split_at_gap(_pick(ends, rng))
            if head.size and tail.size:
                segments += [("split", head), ("end", tail)]
            else:
                segments.append(("end", _pick(ends, rng)))
            continue
        segments += [("unfinished", _pick(mids, rng)), ("end", _pick(ends, rng))]

    pauses = [int(PAUSES_MS[int(rng.integers(len(PAUSES_MS)))]) for _ in segments]
    pauses[-1] = FINAL_PAUSE_MS
    pieces = [_noise(rng, LEAD_IN_MS)]
    position = len(pieces[0])
    units: list[Unit] = []
    for (kind, audio), pause in zip(segments, pauses, strict=True):
        pieces.append(audio)
        position += len(audio)
        units.append(Unit(kind=kind, end_sample=position, pause_ms=pause))
        silence = _noise(rng, pause)
        pieces.append(silence)
        position += len(silence)
    joined = np.concatenate(pieces)
    padding = (-len(joined)) % FRAME_SAMPLES
    joined = np.concatenate([joined, np.zeros(padding, dtype=np.float32)])
    return Stream(audio=joined, units=tuple(units))


def score_stream(stream: Stream, cuts: Sequence[Cut]) -> StreamScore:
    """Match cuts to phrase ends and check mid-sentence segments for premature cuts.

    Phrase ends are matched in time order, each to the closest unused cut within
    ``TOLERANCE_MS``. A mid-sentence segment counts as cut when any boundary lies within
    the tolerance of it.
    """
    used = [False] * len(cuts)
    results: list[UnitResult | None] = [None] * len(stream.units)
    for index in sorted(range(len(stream.units)), key=lambda i: stream.units[i].end_sample):
        unit = stream.units[index]
        if not unit.is_end:
            near = any(abs(c.boundary_sample - unit.end_sample) <= TOLERANCE_SAMPLES for c in cuts)
            results[index] = UnitResult(unit=unit, hit=near)
            continue
        best: tuple[int, int] | None = None  # (distance, cut index)
        for j, cut in enumerate(cuts):
            distance = abs(cut.boundary_sample - unit.end_sample)
            if (
                not used[j]
                and distance <= TOLERANCE_SAMPLES
                and (best is None or distance < best[0])
            ):
                best = (distance, j)
        if best is None:
            results[index] = UnitResult(unit=unit, hit=False)
        else:
            used[best[1]] = True
            matched = cuts[best[1]]
            latency = (matched.finalized_sample - unit.end_sample) * 1000 / RATE_HZ
            offset = (matched.boundary_sample - unit.end_sample) * 1000 / RATE_HZ
            results[index] = UnitResult(unit=unit, hit=True, latency_ms=latency, offset_ms=offset)
    done = [result for result in results if result is not None]
    return StreamScore(results=tuple(done), cuts=len(cuts), matched_cuts=sum(used))


def summarize(scores: Sequence[StreamScore]) -> Summary:
    """Add up stream scores into precision, recall, latency and the pause breakdowns."""
    recall_by_pause: dict[int, tuple[int, int]] = {}
    cut_by_pause: dict[int, tuple[int, int]] = {}
    cut_by_kind: dict[str, tuple[int, int]] = {}
    latencies: list[float] = []
    offsets: list[float] = []
    ends = ends_hit = nonends = nonends_cut = cuts = cuts_matched = 0
    for score in scores:
        cuts += score.cuts
        cuts_matched += score.matched_cuts
        for result in score.results:
            unit = result.unit
            if unit.is_end:
                ends += 1
                ends_hit += result.hit
                hit, total = recall_by_pause.get(unit.pause_ms, (0, 0))
                recall_by_pause[unit.pause_ms] = (hit + result.hit, total + 1)
                if result.latency_ms is not None:
                    latencies.append(result.latency_ms)
                if result.offset_ms is not None:
                    offsets.append(result.offset_ms)
                continue
            nonends += 1
            nonends_cut += result.hit
            cut, total = cut_by_pause.get(unit.pause_ms, (0, 0))
            cut_by_pause[unit.pause_ms] = (cut + result.hit, total + 1)
            cut, total = cut_by_kind.get(unit.kind, (0, 0))
            cut_by_kind[unit.kind] = (cut + result.hit, total + 1)
    return Summary(
        ends=ends,
        ends_hit=ends_hit,
        cuts=cuts,
        cuts_matched=cuts_matched,
        nonends=nonends,
        nonends_cut=nonends_cut,
        latencies_ms=tuple(latencies),
        recall_by_pause=recall_by_pause,
        cut_by_pause=cut_by_pause,
        cut_by_kind=cut_by_kind,
        offsets_ms=tuple(offsets),
    )
