"""Build a self-contained HTML page that shows how pauses are marked.

Each clip is drawn as aligned tracks: waveform, Silero speech probability with the
segmenter thresholds, the pauses found from that probability (and which of them the
project's ``Segmenter`` actually cuts), what the Smart Turn model answers during those
pauses and where it would cut earlier, the reference labels (the energy-based labels of
``tests/endpointing_eval.py`` and, for synthetic streams, the inserted silence), and the
text. LibriSpeech has no word timing, so text is placed per segment, not per word.

Clips are real LibriSpeech utterances from ``data/`` and synthetic streams built like the
endpointing evaluation builds them (known silence between utterances).

Usage:
    python -B scripts/pause_viewer.py --out data/pause_viewer.html

This is a developer tool: it writes one HTML file and is not part of the runtime pipeline.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from endpointing_eval import (  # noqa: E402
    FINAL_PAUSE_MS,
    LEAD_IN_MS,
    MIN_GAP_FRAMES,
    NOISE_STD,
    PAUSES_MS,
    RATE_HZ,
    RMS_FRAME,
    _frame_rms,
    _loud_threshold,
    split_at_gap,
    trim,
)

from pipeline.turn import (  # noqa: E402
    DEFAULT_TURN_MODEL,
    OnnxTurnClassifier,
    SyncTurnLink,
    TurnClassifier,
)
from pipeline.vad import (  # noqa: E402
    FRAME_MS,
    FRAME_SAMPLES,
    Segmenter,
    SegmenterConfig,
    SileroVad,
)

Float32Array = NDArray[np.float32]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MANIFEST = DATA_DIR / "librispeech-pc" / "test-clean.json"
AUDIO_ROOT = DATA_DIR / "librispeech" / "LibriSpeech"
TEMPLATE = Path(__file__).with_name("pause_viewer.html")
SPEECH_THRESHOLD = SegmenterConfig().speech_threshold
EXIT_THRESHOLD = SegmenterConfig().exit_threshold
SILENCE_MS = SegmenterConfig().silence_ms
REAL_CLIPS = 6
SYNTHETIC_STREAMS = 4
GROUPS_PER_STREAM = 3
REAL_SECONDS = (4.0, 14.0)  # long enough to hold a pause, short enough to read
POOL_MAX_SECONDS = 10.0
GROUP_END, GROUP_SPLIT = 0.40, 0.75  # same mix as the endpointing evaluation


@dataclass(frozen=True, slots=True)
class Pause:
    """A stretch of probability below the exit threshold that follows speech.

    Attributes:
        start_frame: First frame below the exit threshold.
        end_frame: First frame at or above the speech threshold afterwards (exclusive).
        open_end: True when the clip ended before speech resumed.
    """

    start_frame: int
    end_frame: int
    open_end: bool

    @property
    def length_ms(self) -> int:
        """Length of the pause in milliseconds."""
        return (self.end_frame - self.start_frame) * FRAME_MS


def find_pauses(
    probs: Sequence[float], speech_threshold: float, exit_threshold: float
) -> list[Pause]:
    """Pauses after speech, using the segmenter's two-threshold rule.

    Speech starts at ``speech_threshold`` and a pause starts when the probability falls
    below ``exit_threshold``; values in between keep the current state. Silence before the
    first speech is not a pause.
    """
    found: list[Pause] = []
    in_speech = False
    start: int | None = None
    for index, probability in enumerate(probs):
        if not in_speech:
            if probability >= speech_threshold:
                if start is not None:
                    found.append(Pause(start, index, open_end=False))
                    start = None
                in_speech = True
        elif probability < exit_threshold:
            in_speech = False
            start = index
    if start is not None:
        found.append(Pause(start, len(probs), open_end=True))
    return found


def segmenter_cuts(probs: Sequence[float], silence_ms: int) -> list[int]:
    """Frame where each chunk cut by silence ends its speech, from the real ``Segmenter``."""
    segmenter = Segmenter(SegmenterConfig(silence_ms=silence_ms))
    frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    cuts: list[int] = []
    for probability in probs:
        for chunk in segmenter.process(float(probability), frame, None):
            if chunk.reason == "silence":
                cuts.append(chunk.speech_end_sample // FRAME_SAMPLES)
    return cuts


class _RecordingClassifier:
    """Wraps a Smart Turn classifier and remembers the last probability it returned."""

    def __init__(self, inner: TurnClassifier) -> None:
        self._inner = inner
        self.last = 0.0

    def predict(self, window: Float32Array) -> float:
        self.last = self._inner.predict(window)
        return self.last


def smart_turn_trace(
    probs: Sequence[float],
    audio: Float32Array,
    classifier: TurnClassifier,
    silence_ms: int,
) -> dict[str, list[dict[str, Any]]]:
    """Run the ``Segmenter`` in smart mode and record what the Smart Turn model said.

    The frame loop mirrors ``VadStage.push``: the newest answer goes in with the frame, and a
    request posted by the frame is submitted right after it. The answer delay is one frame, so
    the run does not depend on how fast the model is.

    Returns:
        ``probes``: every model question as ``frame`` (when it was asked), ``p`` (the answer),
        ``threshold`` (what ``p`` had to reach) and ``kind``; ``cuts``: every chunk end that
        is not a stream flush, as ``frame`` (end of speech), ``decided`` (frame of the
        decision) and ``reason``.
    """
    recording = _RecordingClassifier(classifier)
    link = SyncTurnLink(recording, delay_frames=1)
    segmenter = Segmenter(SegmenterConfig(silence_ms=silence_ms, turn_mode="smart"))
    probes: list[dict[str, Any]] = []
    cuts: list[dict[str, Any]] = []
    for index, probability in enumerate(probs):
        frame = audio[index * FRAME_SAMPLES : (index + 1) * FRAME_SAMPLES]
        for chunk in segmenter.process(float(probability), frame, link.poll()):
            if chunk.reason != "flush":
                cuts.append(
                    {
                        "frame": chunk.speech_end_sample // FRAME_SAMPLES,
                        "decided": chunk.finalized_sample // FRAME_SAMPLES,
                        "reason": chunk.reason,
                    }
                )
        request = segmenter.take_request()
        if request is not None:
            link.submit(request)
            probes.append(
                {
                    "frame": index,
                    "p": round(recording.last, 3),
                    "threshold": request.threshold,
                    "kind": request.kind,
                }
            )
    return {"probes": probes, "cuts": cuts}


def attach_cuts(
    pauses: Sequence[Pause], cuts: Sequence[int], probs: Sequence[float], speech_threshold: float
) -> dict[int, int]:
    """Map pause index -> frame of the cut the segmenter made inside that pause.

    The segmenter reports a cut at the end of the last frame at or above the speech
    threshold, while a pause starts at the first frame below the exit threshold. Frames
    in between (the hysteresis band) are neither, so a cut may sit a frame or two before its
    pause. It still belongs to that pause when nothing in the band reaches the threshold.
    """
    attached: dict[int, int] = {}
    for cut in cuts:
        for index, pause in enumerate(pauses):
            if pause.end_frame < cut:
                continue
            band = probs[cut : pause.start_frame]
            if cut >= pause.start_frame or max(band, default=0.0) < speech_threshold:
                attached[index] = cut
            break
    return attached


def energy_quiet_runs(samples: Float32Array) -> list[tuple[int, int]]:
    """Quiet runs between loud frames, in samples: the energy-based reference labels.

    Uses the labelling rule of ``tests/endpointing_eval.py`` (10 ms frames, loud when above a
    share of the clip's own level). Leading and trailing quiet is not a gap.
    """
    rms = _frame_rms(samples)
    loud = rms > _loud_threshold(rms)
    indices = np.flatnonzero(loud)
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for frame in range(int(indices[0]), int(indices[-1]) + 1):
        if not loud[frame]:
            if start is None:
                start = frame
        elif start is not None:
            if frame - start >= MIN_GAP_FRAMES:
                runs.append((start * RMS_FRAME, frame * RMS_FRAME))
            start = None
    return runs


def split_text(text: str, fraction: float) -> tuple[str, str]:
    """Split text at a word boundary near ``fraction``; neither half is ever empty."""
    words = text.split()
    if len(words) < 2:
        return text, ""
    cut = min(max(round(len(words) * fraction), 1), len(words) - 1)
    return " ".join(words[:cut]), " ".join(words[cut:])


def is_sentence_end(text_raw: str) -> bool:
    """True when the restored punctuation ends a sentence, allowing a closing quote."""
    typographic = chr(0x201D) + chr(0x2019)  # right double and right single quotation marks
    return text_raw.rstrip().rstrip("\"')]} " + typographic).endswith((".", "?", "!"))


@dataclass(frozen=True, slots=True)
class Utterance:
    """One decoded LibriSpeech utterance."""

    name: str
    audio: Float32Array  # as recorded, with its natural leading and trailing quiet
    text: str
    is_end: bool


def load_utterances() -> list[Utterance]:
    """Every manifest utterance whose audio is present under ``data/librispeech``."""
    import miniaudio

    found: list[Utterance] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        path = AUDIO_ROOT / record["audio_filepath"]
        if not path.is_file():
            continue
        decoded = miniaudio.flac_read_f32(path.read_bytes())
        if decoded.sample_rate != RATE_HZ or decoded.nchannels != 1:
            raise SystemExit(f"{path}: expected 16 kHz mono LibriSpeech audio")
        found.append(
            Utterance(
                name=Path(record["audio_filepath"]).stem,
                audio=np.asarray(decoded.samples, dtype=np.float32),
                text=record["text_raw"],
                is_end=is_sentence_end(record["text_raw"]),
            )
        )
    return found


@dataclass(frozen=True, slots=True)
class Piece:
    """One speech segment of a synthetic stream, in stream samples."""

    start_sample: int
    end_sample: int
    pause_ms: int
    text: str
    kind: str


def noise(rng: np.random.Generator, ms: int) -> Float32Array:
    """Low-level noise of known length (the "silence" of the synthetic streams)."""
    return (rng.standard_normal(ms * RATE_HZ // 1000) * NOISE_STD).astype(np.float32)


def build_synthetic(
    ends: Sequence[Utterance], mids: Sequence[Utterance], rng: np.random.Generator
) -> tuple[Float32Array, list[Piece]]:
    """Join trimmed utterances with silences of known length, keeping each segment's text."""
    segments: list[tuple[str, Float32Array, str]] = []

    def pick(items: Sequence[Utterance]) -> Utterance:
        return items[int(rng.integers(len(items)))]

    for _ in range(GROUPS_PER_STREAM):
        roll = float(rng.random())
        if roll < GROUP_END or not mids:
            chosen = pick(ends)
            segments.append(("end", trim(chosen.audio), chosen.text))
        elif roll < GROUP_SPLIT:
            chosen = pick(ends)
            head, tail = split_at_gap(chosen.audio)
            if head.size and tail.size:
                head_text, tail_text = split_text(chosen.text, head.size / (head.size + tail.size))
                segments += [("split", head, head_text), ("end", tail, tail_text)]
            else:
                segments.append(("end", trim(chosen.audio), chosen.text))
        else:
            unfinished, finished = pick(mids), pick(ends)
            segments += [
                ("unfinished", trim(unfinished.audio), unfinished.text),
                ("end", trim(finished.audio), finished.text),
            ]
    pauses = [int(PAUSES_MS[int(rng.integers(len(PAUSES_MS)))]) for _ in segments]
    pauses[-1] = FINAL_PAUSE_MS
    pieces_audio = [noise(rng, LEAD_IN_MS)]
    position = len(pieces_audio[0])
    pieces: list[Piece] = []
    for (kind, audio, text), pause in zip(segments, pauses, strict=True):
        pieces_audio.append(audio)
        pieces.append(Piece(position, position + len(audio), pause, text, kind))
        position += len(audio)
        silence = noise(rng, pause)
        pieces_audio.append(silence)
        position += len(silence)
    joined = np.concatenate(pieces_audio)
    return joined, pieces


def probabilities(vad: SileroVad, audio: Float32Array) -> list[float]:
    """Silero speech probability of every whole frame of ``audio``."""
    vad.reset()
    frames = len(audio) // FRAME_SAMPLES
    return [vad(audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]) for i in range(frames)]


def speech_span(probs: Sequence[float]) -> tuple[int, int]:
    """Sample span from the first to the last frame at or above the speech threshold."""
    loud = [i for i, p in enumerate(probs) if p >= SPEECH_THRESHOLD]
    if not loud:
        return 0, len(probs) * FRAME_SAMPLES
    return loud[0] * FRAME_SAMPLES, (loud[-1] + 1) * FRAME_SAMPLES


def encode_pcm16(audio: Float32Array) -> str:
    """Base64 of little-endian int16 samples, which the page decodes for playback."""
    pcm = np.clip(audio, -1.0, 1.0)
    return base64.b64encode((pcm * 32767.0).astype("<i2").tobytes()).decode("ascii")


def describe_clip(
    clip_id: str,
    title: str,
    kind: str,
    audio: Float32Array,
    vad: SileroVad,
    segments: list[dict[str, Any]] | None,
    known_pauses: list[dict[str, int]],
    turn: TurnClassifier | None,
) -> dict[str, Any]:
    """Everything the page needs to draw one clip; ``turn`` None leaves out Smart Turn."""
    usable = len(audio) // FRAME_SAMPLES * FRAME_SAMPLES
    audio = audio[:usable]
    probs = probabilities(vad, audio)
    pauses = find_pauses(probs, SPEECH_THRESHOLD, EXIT_THRESHOLD)
    cuts = segmenter_cuts(probs, SILENCE_MS)
    attached = attach_cuts(pauses, cuts, probs, SPEECH_THRESHOLD)
    return {
        "id": clip_id,
        "title": title,
        "kind": kind,
        "samples": usable,
        "audio": encode_pcm16(audio),
        "probs": [round(p, 3) for p in probs],
        "pauses": [
            {
                "start": p.start_frame,
                "end": p.end_frame,
                "open": p.open_end,
                "cut": attached.get(index),
            }
            for index, p in enumerate(pauses)
        ],
        "cuts": cuts,
        "energy_runs": [list(run) for run in energy_quiet_runs(audio)],
        "known_pauses": known_pauses,
        "segments": segments,
        "turn": smart_turn_trace(probs, audio, turn, SILENCE_MS) if turn is not None else None,
    }


def disagreements(clip: dict[str, Any]) -> list[str]:
    """Segmenter cuts that match no pause, and cuts in a pause shorter than ``SILENCE_MS``."""
    problems = []
    attached = {p["cut"] for p in clip["pauses"] if p["cut"] is not None}
    for cut in clip["cuts"]:
        if cut not in attached:
            problems.append(f"{clip['id']}: cut at frame {cut} is not in any pause (stall cut?)")
    for pause in clip["pauses"]:
        if pause["cut"] is not None and (pause["end"] - pause["start"]) * FRAME_MS < SILENCE_MS:
            problems.append(f"{clip['id']}: cut at frame {pause['cut']} in a pause under the limit")
    return problems


def build_clips(vad: SileroVad, turn: TurnClassifier | None, seed: int) -> list[dict[str, Any]]:
    """Real clips first, then synthetic streams."""
    utterances = load_utterances()
    if not utterances:
        raise SystemExit(f"no LibriSpeech audio found under {AUDIO_ROOT}")
    clips: list[dict[str, Any]] = []
    low, high = REAL_SECONDS
    real = [
        u
        for u in utterances
        if low <= len(u.audio) / RATE_HZ <= high and ("," in u.text or ";" in u.text)
    ]
    step = max(1, len(real) // REAL_CLIPS)
    for utterance in real[::step][:REAL_CLIPS]:
        probs = probabilities(vad, utterance.audio)
        start, end = speech_span(probs)
        segment = {"start": start, "end": end, "text": utterance.text, "kind": "real"}
        clips.append(
            describe_clip(
                f"real-{utterance.name}",
                utterance.name,
                "real",
                utterance.audio,
                vad,
                [segment],
                [],
                turn,
            )
        )
    pool = [u for u in utterances if len(u.audio) / RATE_HZ <= POOL_MAX_SECONDS]
    ends = [u for u in pool if u.is_end]
    mids = [u for u in pool if not u.is_end]
    if not ends:
        raise SystemExit("no sentence-end utterances found for the synthetic streams")
    rng = np.random.default_rng(seed)
    for number in range(1, SYNTHETIC_STREAMS + 1):
        audio, pieces = build_synthetic(ends, mids, rng)
        segments = [
            {"start": p.start_sample, "end": p.end_sample, "text": p.text, "kind": p.kind}
            for p in pieces
        ]
        known = [
            {"start": p.end_sample, "end": p.end_sample + p.pause_ms * RATE_HZ // 1000}
            for p in pieces
        ]
        clips.append(
            describe_clip(
                f"synthetic-{number}",
                f"synthetic stream {number}",
                "synthetic",
                audio,
                vad,
                segments,
                known,
                turn,
            )
        )
    return clips


def render(clips: list[dict[str, Any]]) -> str:
    """Insert the clip data into the page template."""
    payload = {
        "rate_hz": RATE_HZ,
        "frame_samples": FRAME_SAMPLES,
        "speech_threshold": SPEECH_THRESHOLD,
        "exit_threshold": EXIT_THRESHOLD,
        "silence_ms": SILENCE_MS,
        "clips": clips,
    }
    # "</" would end the script element early, so escape it inside the JSON text.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", data)


def main() -> None:
    """Build the page and print a one-line summary per clip."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "pause_viewer.html")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    vad = SileroVad(MODELS_DIR / "silero_vad.onnx")
    turn_path = MODELS_DIR / DEFAULT_TURN_MODEL
    turn: TurnClassifier | None = None
    if turn_path.is_file():
        turn = OnnxTurnClassifier(turn_path, threads=1)
    else:
        print(f"Smart Turn model not found at {turn_path}; the Smart Turn track is left out")
    clips = build_clips(vad, turn, args.seed)
    problems: list[str] = []
    for clip in clips:
        problems += disagreements(clip)
        smart = clip["turn"]
        smart_cuts = sum(c["reason"] == "turn" for c in smart["cuts"]) if smart else 0
        print(
            f"{clip['id']:<34} {clip['samples'] / RATE_HZ:6.1f} s  "
            f"pauses {len(clip['pauses']):2d}  cuts {len(clip['cuts']):2d}  "
            f"smart-turn cuts {smart_cuts:2d}"
        )
    for problem in problems:
        print(f"WARNING {problem}")
    args.out.write_text(render(clips), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
