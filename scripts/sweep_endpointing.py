"""Sweep the silence rule of the endpointer on labelled LibriSpeech speech.

Two ways to end a chunk after silence are compared on the same streams:

* ``hysteresis`` (what ``Segmenter`` ships): a clock starts at the first frame below
  ``exit_threshold`` and is cleared by any frame at or above ``speech_threshold``. The chunk
  ends when the clock has run for the silence length. Frames between the two thresholds
  change nothing.
* ``integral``: one sum S of (1 - p) over the frames since the last speech frame. The chunk
  ends when S reaches the silence length in frames. Near-certain silence adds 1 per frame,
  a frame the model is unsure about adds less, so one number carries duration and
  confidence together.
* ``drift``: S += c - p on every frame below the speech threshold, with c = exit level.
  Frames the model doubts (p between c and the speech threshold) pull S back down. The
  chunk ends when S reaches c * N, so pure silence (p = 0) still needs N frames.
* ``cusum``: the same sum, but S is never allowed below zero (a CUSUM-style test).
* ``band``: S += 1 - p only while p < c; between c and the speech threshold S is frozen.

``N`` is always the silence length of pure silence, so all rules are compared at the same
nominal silence length. Every variant uses speech threshold 0.5 for its reset, as proposed.

The shipped rule runs through the real ``Segmenter``. The other rules have no code in
``src``, so they run through ``Reference``, a frame-by-frame port of the fixed path of
``Segmenter``.
Before any result is trusted, the port must reproduce the real segmenter's cuts exactly for
every hysteresis configuration; otherwise the script stops.

Needs ``python -B scripts/fetch_models.py --with-data`` first.

    python -B scripts/sweep_endpointing.py --out data/sweep_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))  # eval_endpointing.py lives next to this file
sys.path.insert(0, str(ROOT / "tests"))

from endpointing_eval import (  # noqa: E402
    Cut,
    Stream,
    Summary,
    build_stream,
    score_stream,
    summarize,
)
from eval_endpointing import (  # noqa: E402
    DATA_DIR,
    MODELS_DIR,
    ReplayVad,
    load_utterances,
    percentile,
    vad_probabilities,
    wilson,
)

from pipeline.vad import (  # noqa: E402
    FRAME_MS,
    FRAME_SAMPLES,
    Segmenter,
    SegmenterConfig,
    SileroVad,
    VadStage,
)

SEED = 20261108  # the seed of the report in data/endpointing_report.html
STREAMS = 60
GROUPS = 5
SPEECH_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
EXIT_THRESHOLDS = (0.15, 0.25, 0.35, 0.45)  # plus "single": exit = speech, no neutral band
HYSTERESIS_SILENCE_MS = (450, 500, 550, 600)  # the client's range
INTEGRAL_SILENCE_MS = (250, 300, 350, 400, 450, 500, 550, 600)  # weighted, no range check
LEAKY_KINDS = ("drift", "cusum", "band")
LEAKY_SPEECH = 0.5  # reset threshold of the drift family, as proposed
LEAKY_EXITS = (0.15, 0.25, 0.35, 0.45)  # c: the level that frames are weighted against
_EPS = 1e-9  # float sums of 0.35 are not exact; compare with a tolerance
Float32Array = Any


@dataclass(frozen=True, slots=True)
class Rule:
    """One endpointing rule and its input parameters.

    Attributes:
        kind: ``"hysteresis"`` (the shipped clock), ``"integral"``, ``"drift"``, ``"cusum"``
            or ``"band"`` (see the module docstring).
        speech_threshold: Probability at or above which a frame is speech.
        exit_threshold: Hysteresis: probability below which the clock starts. Drift, cusum
            and band: the level c. Integral: equals ``speech_threshold``, no separate level.
        silence_ms: Silence length of pure silence that ends a chunk.
    """

    kind: str
    speech_threshold: float
    exit_threshold: float
    silence_ms: int

    @property
    def name(self) -> str:
        """Short label for tables."""
        if self.kind == "hysteresis":
            return (
                f"hysteresis {self.speech_threshold:.1f}/{self.exit_threshold:.2f}, "
                f"{self.silence_ms} ms"
            )
        if self.kind == "integral":
            return f"integral speech {self.speech_threshold:.1f}, S >= {self.silence_ms} ms"
        return f"{self.kind} c {self.exit_threshold:.2f}, N {self.silence_ms} ms"

    @property
    def silence_frames(self) -> int:
        """Silence length in frames, rounded up as ``SegmenterConfig`` does."""
        return math.ceil(self.silence_ms / FRAME_MS)


def hysteresis_rules() -> list[Rule]:
    """Every (speech, exit) pair with exit <= speech, at every silence length."""
    rules: list[Rule] = []
    for speech in SPEECH_THRESHOLDS:
        for exit_threshold in sorted({*EXIT_THRESHOLDS, speech}):
            if exit_threshold > speech:
                continue
            for silence in HYSTERESIS_SILENCE_MS:
                rules.append(Rule("hysteresis", speech, exit_threshold, silence))
    return rules


def integral_rules() -> list[Rule]:
    """Every speech threshold with every integral length."""
    return [
        Rule("integral", speech, speech, silence)
        for speech in SPEECH_THRESHOLDS
        for silence in INTEGRAL_SILENCE_MS
    ]


def leaky_rules(kind: str) -> list[Rule]:
    """Every level c with every pure-silence length, at the proposed reset threshold."""
    return [
        Rule(kind, LEAKY_SPEECH, level, silence)
        for level in LEAKY_EXITS
        for silence in INTEGRAL_SILENCE_MS
    ]


def frame_rms(audio: Float32Array) -> list[float]:
    """RMS of every full frame, computed the way ``Segmenter`` computes it."""
    frames = audio[: len(audio) // FRAME_SAMPLES * FRAME_SAMPLES].reshape(-1, FRAME_SAMPLES)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)).tolist()


def segmenter_cuts(rule: Rule, stream: Stream, probabilities: Sequence[float]) -> list[Cut]:
    """Run the real ``Segmenter`` (fixed mode) on replayed probabilities."""
    config = SegmenterConfig(
        silence_ms=rule.silence_ms,
        speech_threshold=rule.speech_threshold,
        exit_threshold=rule.exit_threshold,
        turn_mode="fixed",
    )
    stage = VadStage(ReplayVad(probabilities), Segmenter(config))  # type: ignore[arg-type]
    chunks = stage.push(stream.audio)
    last = stage.flush()
    if last is not None:
        chunks.append(last)
    return [Cut(c.speech_end_sample, c.finalized_sample) for c in chunks]


@dataclass(slots=True)
class _Open:
    """The open chunk: frame indices plus the counters the silence test reads."""

    frames: list[int]
    continuation: bool = False
    speech: int = 0
    last: int = -1  # position in ``frames`` of the last speech frame
    clock: int | None = None  # hysteresis: position of the first frame below exit
    acc: float = 0.0  # integral-family sum S, reset by every speech frame


class Reference:
    """Frame-by-frame port of the fixed path of ``Segmenter`` with the silence test of a rule.

    The port keeps the same pre-roll, minimum speech, tail trim, stall rule and hard cut at
    ``max_utterance_ms``. Only the silence test differs between the two rule kinds. For
    binary probabilities the integral and the hysteresis clock agree exactly, which the unit
    tests use as a check on the port.
    """

    def __init__(self, rule: Rule) -> None:
        """Bind the port to one rule; the base config supplies the non-silence constants."""
        self._rule = rule
        self._config = SegmenterConfig(
            silence_ms=450,  # only the constants below are read from this config
            speech_threshold=rule.speech_threshold,
            exit_threshold=rule.exit_threshold,
            turn_mode="fixed",
        )
        self._speech = rule.speech_threshold
        self._exit = rule.exit_threshold
        self._silence = rule.silence_frames
        self._probs: Sequence[float] = ()
        self._rms: Sequence[float] = ()

    def run(self, probabilities: Sequence[float], rms: Sequence[float]) -> list[tuple[int, int]]:
        """Return ``(boundary_sample, finalized_sample)`` for every chunk, in stream order."""
        self._probs = probabilities
        self._rms = rms
        cfg = self._config
        out: list[tuple[int, int]] = []
        ring: deque[int] = deque(maxlen=cfg.pre_roll_frames)
        current: _Open | None = None
        for index, prob in enumerate(probabilities):
            finalized = (index + 1) * FRAME_SAMPLES
            if current is None:
                if prob < self._speech:
                    ring.append(index)
                else:
                    current = self._scan([*ring, index], continuation=False)
                    ring.clear()
                continue
            current.frames.append(index)
            pos = len(current.frames) - 1
            self._update(current, pos, prob)
            stalled = pos - current.last >= 2 * self._silence
            if self._silence_done(current, pos) or stalled:
                self._finish(current, out, ring, finalized)
                current = None
            elif len(current.frames) >= cfg.max_frames:
                current = self._cut(current, out, finalized)
        if current is not None:
            self._finish(current, out, ring, len(probabilities) * FRAME_SAMPLES)
        return out

    def _update(self, state: _Open, pos: int, prob: float) -> None:
        kind = self._rule.kind
        if prob >= self._speech:
            state.speech += 1
            state.last = pos
            state.clock = None
            state.acc = 0.0
        elif kind == "integral":
            state.acc += 1.0 - prob
        elif kind == "drift":
            state.acc += self._exit - prob  # doubtful frames pull the sum back
        elif kind == "cusum":
            state.acc = max(0.0, state.acc + self._exit - prob)  # never below zero
        elif kind == "band":
            if prob < self._exit:  # between c and the speech threshold: frozen
                state.acc += 1.0 - prob
        elif prob < self._exit and state.clock is None:
            state.clock = pos

    def _silence_done(self, state: _Open, pos: int) -> bool:
        kind = self._rule.kind
        if kind == "hysteresis":
            return state.clock is not None and pos - state.clock + 1 >= self._silence
        # Drift and cusum add c per pure-silence frame, so they need c * N to fire at N frames.
        target = self._silence * self._exit if kind in ("drift", "cusum") else self._silence
        return state.acc >= target - _EPS

    def _scan(self, frames: list[int], continuation: bool) -> _Open:
        state = _Open(frames=frames, continuation=continuation)
        for pos, index in enumerate(frames):
            self._update(state, pos, self._probs[index])
        return state

    def _finish(
        self, state: _Open, out: list[tuple[int, int]], ring: deque[int], finalized: int
    ) -> None:
        cfg = self._config
        frames = state.frames
        keep = len(frames)
        if state.last >= 0:
            keep = min(keep, state.last + 1 + cfg.tail_frames)
        too_short = state.speech < cfg.min_speech_frames and not (
            state.continuation and state.speech > 0
        )
        if too_short:
            ring.extend(frames)
            return
        speech_end_pos = min(max(state.last, 0), keep - 1)
        out.append(((frames[speech_end_pos] + 1) * FRAME_SAMPLES, finalized))
        ring.extend(frames[keep:])

    def _cut(self, state: _Open, out: list[tuple[int, int]], finalized: int) -> _Open:
        cfg = self._config
        frames = state.frames
        window_start = max(0, len(frames) - cfg.cut_search_frames)
        window = frames[window_start:]
        quietest = min(range(len(window)), key=lambda i: (self._rms[window[i]], -i))
        head, rest = frames[: window_start + quietest + 1], frames[window_start + quietest + 1 :]
        speech_end_pos = max(
            (pos for pos, index in enumerate(head) if self._probs[index] >= self._speech),
            default=len(head) - 1,
        )
        out.append(((head[speech_end_pos] + 1) * FRAME_SAMPLES, finalized))
        return self._scan(rest, continuation=True)


def evaluate(
    rule: Rule,
    streams: Sequence[Stream],
    probabilities: Sequence[Sequence[float]],
    rms_per_stream: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Score one rule over every stream; hysteresis rules are first checked against the port."""
    reference = Reference(rule)
    scores = []
    for number, (stream, probs, rms) in enumerate(
        zip(streams, probabilities, rms_per_stream, strict=True)
    ):
        if rule.kind == "hysteresis":
            cuts = segmenter_cuts(rule, stream, probs)
            expected = reference.run(probs, rms)
            got = [(c.boundary_sample, c.finalized_sample) for c in cuts]
            if got != expected:
                raise RuntimeError(
                    f"Reference differs from Segmenter: {rule.name}, stream {number}"
                )
        else:
            cuts = [Cut(b, f) for b, f in reference.run(probs, rms)]
        scores.append(score_stream(stream, cuts))
    return result_row(rule, summarize(scores))


def result_row(rule: Rule, summary: Summary) -> dict[str, Any]:
    """Plain-data view of one rule with confidence intervals for the rates."""
    return {
        "name": rule.name,
        "kind": rule.kind,
        "speech_threshold": rule.speech_threshold,
        "exit_threshold": rule.exit_threshold,
        "silence_ms": rule.silence_ms,
        "silence_frames": rule.silence_frames,
        "ends": summary.ends,
        "ends_hit": summary.ends_hit,
        "cuts": summary.cuts,
        "cuts_matched": summary.cuts_matched,
        "nonends": summary.nonends,
        "nonends_cut": summary.nonends_cut,
        "precision": summary.precision,
        "precision_ci": wilson(summary.cuts_matched, summary.cuts),
        "recall": summary.recall,
        "recall_ci": wilson(summary.ends_hit, summary.ends),
        "f1": summary.f1,
        "premature_rate": summary.premature_rate,
        "latency_p50_ms": percentile(summary.latencies_ms, 50),
        "latency_p95_ms": percentile(summary.latencies_ms, 95),
    }


# --- reporting ----------------------------------------------------------------------


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{100 * value:.1f}%"


def _table(rows: Sequence[dict[str, Any]]) -> None:
    print(
        "| rule | cuts | precision (95% CI) | recall (95% CI) | F1 | premature | latency p50, ms |"
    )
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        p_lo, p_hi = r["precision_ci"]
        r_lo, r_hi = r["recall_ci"]
        print(
            f"| {r['name']} | {r['cuts']} | {_pct(r['precision'])} "
            f"({_pct(p_lo)}-{_pct(p_hi)}) | {_pct(r['recall'])} ({_pct(r_lo)}-{_pct(r_hi)}) | "
            f"{_pct(r['f1'])} | {_pct(r['premature_rate'])} | "
            f"{'n/a' if math.isnan(r['latency_p50_ms']) else round(r['latency_p50_ms'])} |"
        )


def _pareto(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows that no other row beats on both precision and recall."""
    front = []
    for row in rows:
        dominated = any(
            other is not row
            and other["precision"] >= row["precision"]
            and other["recall"] >= row["recall"]
            and (other["precision"] > row["precision"] or other["recall"] > row["recall"])
            for other in rows
        )
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda r: r["recall"])


def _matrix(rows: Sequence[dict[str, Any]], key: str, title: str) -> None:
    """F1 by threshold (rows) and silence length (columns)."""
    silences = sorted({r["silence_ms"] for r in rows})
    print(f"\n{title}")
    print("| threshold | " + " | ".join(f"{s} ms" for s in silences) + " |")
    print("|---|" + "---|" * len(silences))
    for label in sorted({r[key] for r in rows}):
        cells = []
        for silence in silences:
            hit = [r for r in rows if r[key] == label and r["silence_ms"] == silence]
            cells.append(_pct(hit[0]["f1"]) if hit else "-")
        print(f"| {label} | " + " | ".join(cells) + " |")


def print_report(rows: list[dict[str, Any]]) -> None:
    """Print the top rows, the Pareto front, the best rule per family and the F1 matrices."""
    kinds = [
        k for k in ("hysteresis", "integral", *LEAKY_KINDS) if any(r["kind"] == k for r in rows)
    ]
    print("\nTop 12 by F1, all rules:")
    _table(sorted(rows, key=lambda r: -r["f1"])[:12])
    print("\nPareto front (no other rule beats it on both precision and recall):")
    _table(_pareto(rows))
    print("\nBest F1 per rule family:")
    _table([max((r for r in rows if r["kind"] == k), key=lambda r: r["f1"]) for k in kinds])
    for kind in kinds:
        family = [r for r in rows if r["kind"] == kind]
        if kind == "hysteresis":
            labelled = [
                {**r, "label": f"{r['speech_threshold']:.1f}/{r['exit_threshold']:.2f}"}
                for r in family
            ]
            title = "Hysteresis, F1 by speech/exit threshold (rows) and silence (columns):"
        elif kind == "integral":
            labelled = [{**r, "label": f"speech {r['speech_threshold']:.1f}"} for r in family]
            title = "Integral, F1 by speech threshold (rows) and S length (columns):"
        else:
            labelled = [{**r, "label": f"c {r['exit_threshold']:.2f}"} for r in family]
            title = f"{kind}, F1 by level c (rows) and pure-silence length N (columns):"
        _matrix(labelled, "label", title)


def main() -> int:
    """Run every rule on the same streams and write the results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DATA_DIR / "sweep_results.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--streams", type=int, default=STREAMS)
    parser.add_argument(
        "--families",
        default="hysteresis,integral,drift,cusum,band",
        help="comma-separated subset of hysteresis, integral, drift, cusum, band",
    )
    args = parser.parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    started = time.perf_counter()
    ends, mids = load_utterances()
    rng = np.random.default_rng(args.seed)
    streams = [build_stream(ends, mids, rng, groups=GROUPS) for _ in range(args.streams)]
    labelled = [u for s in streams for u in s.units]
    print(
        f"seed {args.seed}: streams {len(streams)}, segments {len(labelled)}, "
        f"phrase ends {sum(u.is_end for u in labelled)}",
        flush=True,
    )

    vad = SileroVad(MODELS_DIR / "silero_vad.onnx")
    probabilities = [vad_probabilities(vad, s) for s in streams]
    rms_per_stream = [frame_rms(s.audio) for s in streams]

    selected: list[Rule] = []
    if "hysteresis" in families:
        selected += hysteresis_rules()
    if "integral" in families:
        selected += integral_rules()
    for kind in LEAKY_KINDS:
        if kind in families:
            selected += leaky_rules(kind)
    rows: list[dict[str, Any]] = []
    for rule in selected:
        began = time.perf_counter()
        row = evaluate(rule, streams, probabilities, rms_per_stream)
        row["seconds"] = time.perf_counter() - began
        rows.append(row)
        print(f"  {rule.name}: F1 {_pct(row['f1'])} ({row['seconds']:.1f} s)", flush=True)

    print_report(rows)
    report = {
        "generated_by": "scripts/sweep_endpointing.py",
        "seed": args.seed,
        "streams": args.streams,
        "groups_per_stream": GROUPS,
        "wall_seconds": time.perf_counter() - started,
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.out} ({report['wall_seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
