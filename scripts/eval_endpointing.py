"""Measure phrase-end detection on labelled LibriSpeech speech.

Compares the fixed silence rule (450, 525 and 600 ms) with Smart Turn (``turn_mode=smart``)
and sweeps two input parameters, on streams built by ``tests/endpointing_eval.build_stream``.
Labels come from LibriSpeech-PC punctuation: a segment that ends with . ? ! is a phrase end.

Needs ``python -B scripts/fetch_models.py --with-data`` (data/ and models/) first.
Runs the real ``Segmenter`` and ``VadStage``. Silero runs once per stream and its
probabilities are replayed, which gives the same frames and the same decisions. Smart Turn
answers come from the real ONNX model; its delay is the answer time measured here, applied
as a whole number of 32 ms frames (``SyncTurnLink``), so the result does not depend on how
busy the machine is during the run.

    python -B scripts/eval_endpointing.py --out data/endpointing_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import miniaudio
import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from endpointing_eval import (  # noqa: E402
    FINAL_PAUSE_MS,
    PAUSES_MS,
    RATE_HZ,
    TOLERANCE_MS,
    Cut,
    Stream,
    Summary,
    build_stream,
    score_stream,
    summarize,
    trim,
)

from pipeline.turn import DEFAULT_TURN_MODEL, OnnxTurnClassifier, SyncTurnLink  # noqa: E402
from pipeline.turn_features import WINDOW_SAMPLES, turn_features  # noqa: E402
from pipeline.vad import (  # noqa: E402
    FRAME_MS,
    FRAME_SAMPLES,
    Segmenter,
    SegmenterConfig,
    SileroVad,
    VadStage,
)

Float32Array = NDArray[np.float32]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MANIFEST = DATA_DIR / "librispeech-pc" / "test-clean.json"
AUDIO_ROOT = DATA_DIR / "librispeech" / "LibriSpeech"
MAX_UTTERANCE_S = 10.0  # keeps single segments under the 15 s chunk cap
PERCENTILES = (50, 95)
WORKERS = 4  # configurations run in parallel, one ONNX session each (one thread per session)


class ReplayVad:
    """Plays back Silero probabilities computed once per stream (the model is deterministic)."""

    def __init__(self, probabilities: Sequence[float]) -> None:
        """Hold the probabilities, one per 512-sample frame, in stream order."""
        self._probabilities = probabilities
        self._next = 0

    def __call__(self, frame: Float32Array) -> float:
        """Return the next recorded probability; the frame itself is not needed."""
        probability = self._probabilities[self._next]
        self._next += 1
        return probability

    def reset(self) -> None:
        """Nothing to reset: a replay starts from its first probability each time."""


class CachedClassifier:
    """Memoizes Smart Turn answers by window content; shared across configurations."""

    def __init__(self, inner: OnnxTurnClassifier, cache: dict[bytes, float]) -> None:
        """Wrap ``inner``; ``cache`` is shared by every configuration that runs."""
        self._inner = inner
        self._cache = cache

    def predict(self, window: Float32Array) -> float:
        """Return the probability for ``window``, computing it only once."""
        key = hashlib.blake2b(window.tobytes(), digest_size=16).digest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = self._inner.predict(window)
        self._cache[key] = value
        return value


@dataclass(frozen=True, slots=True)
class ConfigSpec:
    """One endpointing configuration to evaluate."""

    name: str
    smart: bool
    silence_ms: int
    threshold: float = 0.5  # Smart Turn probability that counts as "complete"
    check_ms: int = 200  # silence before the first Smart Turn check
    delay: str = "p50"  # which measured model time sets the simulated answer delay


CONFIGS = [
    ConfigSpec("fixed 450", smart=False, silence_ms=450),
    ConfigSpec("fixed 525", smart=False, silence_ms=525),
    ConfigSpec("fixed 600", smart=False, silence_ms=600),
    ConfigSpec("smart 450 (p50 delay)", smart=True, silence_ms=450),
    ConfigSpec("smart 450 (p95 delay)", smart=True, silence_ms=450, delay="p95"),
    ConfigSpec("smart 600 (p50 delay)", smart=True, silence_ms=600),
    ConfigSpec("smart 600 check 300", smart=True, silence_ms=600, check_ms=300),
    ConfigSpec("smart 450 thr 0.3", smart=True, silence_ms=450, threshold=0.3),
    ConfigSpec("smart 450 thr 0.7", smart=True, silence_ms=450, threshold=0.7),
    ConfigSpec("smart 450 thr 0.9", smart=True, silence_ms=450, threshold=0.9),
    ConfigSpec("smart 600 thr 0.3", smart=True, silence_ms=600, threshold=0.3),
    ConfigSpec("smart 600 thr 0.7", smart=True, silence_ms=600, threshold=0.7),
    ConfigSpec("smart 600 check 150", smart=True, silence_ms=600, check_ms=150),
]


def is_sentence_end(text_raw: str) -> bool:
    """True when the restored punctuation ends a sentence (. ? !), allowing a closing quote."""
    typographic = chr(0x201D) + chr(0x2019)  # right double and right single quotation marks
    closing = "\"')]} " + typographic  # closing quotes, brackets and spaces to strip
    return text_raw.rstrip().rstrip(closing).endswith((".", "?", "!"))


def load_utterances() -> tuple[list[Float32Array], list[Float32Array]]:
    """Decode test-clean utterances up to ``MAX_UTTERANCE_S``, split by their label."""
    ends: list[Float32Array] = []
    mids: list[Float32Array] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["duration"] > MAX_UTTERANCE_S:
            continue
        decoded = miniaudio.flac_read_f32((AUDIO_ROOT / record["audio_filepath"]).read_bytes())
        if decoded.sample_rate != RATE_HZ or decoded.nchannels != 1:
            raise SystemExit(f"{record['audio_filepath']}: expected 16 kHz mono LibriSpeech audio")
        audio = trim(np.asarray(decoded.samples, dtype=np.float32))
        if audio.size == 0:
            continue
        (ends if is_sentence_end(record["text_raw"]) else mids).append(audio)
    return ends, mids


def vad_probabilities(vad: SileroVad, stream: Stream) -> list[float]:
    """Silero probability of every full frame of ``stream``, in order."""
    vad.reset()
    frames = len(stream.audio) // FRAME_SAMPLES
    return [vad(stream.audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]) for i in range(frames)]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval for a proportion (NaN bounds without data)."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (centre - half, centre + half)


def percentile(values: Sequence[float], q: int) -> float:
    """Percentile of ``values``, NaN when empty."""
    return float(np.percentile(values, q)) if values else float("nan")


def run_config(
    spec: ConfigSpec,
    streams: Sequence[Stream],
    probabilities: Sequence[Sequence[float]],
    delay_frames: int,
    cache: dict[bytes, float],
) -> dict[str, Any]:
    """Run one configuration over every stream and return its summary as plain data."""
    classifier = None
    if spec.smart:
        classifier = CachedClassifier(
            OnnxTurnClassifier(MODELS_DIR / DEFAULT_TURN_MODEL, threads=1), cache
        )
    scores = []
    counters: Counter[str] = Counter()
    for stream, probs in zip(streams, probabilities, strict=True):
        config = SegmenterConfig(
            silence_ms=spec.silence_ms,
            turn_mode="smart" if spec.smart else "fixed",
            turn_check_ms=spec.check_ms,
            turn_threshold=spec.threshold,
            soft_cut_threshold=max(0.7, spec.threshold),  # must not be below the turn threshold
        )
        segmenter = Segmenter(config)
        link = SyncTurnLink(classifier, delay_frames=delay_frames) if classifier else None
        stage = VadStage(ReplayVad(probs), segmenter, link)
        chunks = stage.push(stream.audio)
        last = stage.flush()
        if last is not None:
            chunks.append(last)
        cuts = [Cut(c.speech_end_sample, c.finalized_sample) for c in chunks]
        scores.append(score_stream(stream, cuts))
        counters.update(
            turn_checks=segmenter.turn_checks,
            turn_early_ends=segmenter.turn_early_ends,
            turn_soft_cuts=segmenter.turn_soft_cuts,
            turn_hard_cuts=segmenter.turn_hard_cuts,
            turn_rescued=segmenter.turn_rescued,
            turn_late=segmenter.turn_late,
        )
    summary = summarize(scores)
    return config_result(spec, summary, delay_frames, dict(counters))


def config_result(
    spec: ConfigSpec, summary: Summary, delay_frames: int, counters: dict[str, int]
) -> dict[str, Any]:
    """Plain-data view of one configuration, with confidence intervals for the rates."""
    return {
        "name": spec.name,
        "silence_ms": spec.silence_ms,
        "smart": spec.smart,
        "threshold": spec.threshold,
        "check_ms": spec.check_ms,
        "delay_frames": delay_frames if spec.smart else None,
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
        "premature_ci": wilson(summary.nonends_cut, summary.nonends),
        "latency_p50_ms": percentile(summary.latencies_ms, 50),
        "latency_p95_ms": percentile(summary.latencies_ms, 95),
        "offset_p50_ms": percentile(summary.offsets_ms, 50),
        "offset_p95_ms": percentile(summary.offsets_ms, 95),
        "recall_by_pause": {str(k): v for k, v in sorted(summary.recall_by_pause.items())},
        "cut_by_pause": {str(k): v for k, v in sorted(summary.cut_by_pause.items())},
        "cut_by_kind": summary.cut_by_kind,
        "counters": counters,
    }


# --- speed and size -----------------------------------------------------------------

RSS_PROBE = r"""
import ctypes, json, sys
from ctypes import wintypes
from pathlib import Path
SIZE = ctypes.c_size_t
class Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", SIZE), ("WorkingSetSize", SIZE),
                ("QuotaPeakPagedPoolUsage", SIZE), ("QuotaPagedPoolUsage", SIZE),
                ("QuotaPeakNonPagedPoolUsage", SIZE), ("QuotaNonPagedPoolUsage", SIZE),
                ("PagefileUsage", SIZE), ("PeakPagefileUsage", SIZE)]
kernel32 = ctypes.WinDLL("kernel32")
kernel32.GetCurrentProcess.restype = ctypes.c_void_p  # a 64-bit pseudo-handle, not an int
psapi = ctypes.WinDLL("psapi")
psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
def working_set_mb():
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return counters.WorkingSetSize / 2**20
root = Path(sys.argv[1])
from pipeline.vad import SileroVad
from pipeline.turn import OnnxTurnClassifier
base = working_set_mb()
vad = SileroVad(root / "models" / "silero_vad.onnx")
after_vad = working_set_mb()
turn = OnnxTurnClassifier(root / "models" / sys.argv[2], threads=1)
after_turn = working_set_mb()
print(json.dumps({"base_mb": base, "silero_mb": after_vad - base,
                  "smart_turn_mb": after_turn - after_vad}))
"""


def measure_memory(turn_model: str) -> dict[str, float]:
    """Working-set growth of loading each model, measured in a fresh process."""
    result = subprocess.run(
        [sys.executable, "-B", "-c", RSS_PROBE, str(ROOT), turn_model],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def measure_speed(streams: Sequence[Stream], vad: SileroVad) -> dict[str, Any]:
    """Per-frame VAD time and per-check Smart Turn time, with one thread, as in production."""
    audio = streams[0].audio
    frames = [audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES] for i in range(600)]
    vad.reset()
    for frame in frames[:20]:  # warm-up
        vad(frame)
    vad_ms: list[float] = []
    for frame in frames[20:]:
        started = time.perf_counter()
        vad(frame)
        vad_ms.append((time.perf_counter() - started) * 1000.0)

    classifier = OnnxTurnClassifier(MODELS_DIR / DEFAULT_TURN_MODEL, threads=1)
    rng = np.random.default_rng(7)
    windows = []
    for _ in range(40):
        stream = streams[int(rng.integers(len(streams)))].audio
        start = int(rng.integers(0, max(1, len(stream) - WINDOW_SAMPLES)))
        windows.append(stream[start : start + WINDOW_SAMPLES].copy())
    model_ms: list[float] = []
    feature_ms: list[float] = []
    for window in windows:
        started = time.perf_counter()
        turn_features(window)
        feature_ms.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        classifier.predict(window)
        model_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "vad_frame_ms_p50": percentile(vad_ms, 50),
        "vad_frame_ms_p95": percentile(vad_ms, 95),
        "turn_predict_ms_p50": percentile(model_ms, 50),
        "turn_predict_ms_p95": percentile(model_ms, 95),
        "turn_predict_ms_max": max(model_ms),
        "turn_features_ms_p50": percentile(feature_ms, 50),
        "turn_predict_samples": len(model_ms),
        "turn_window_s": WINDOW_SAMPLES / RATE_HZ,
    }


def model_sizes() -> dict[str, int]:
    """File sizes of the models in bytes (what they cost on disk)."""
    names = {
        "silero_vad": "silero_vad.onnx",
        "smart_turn_fp32": "smart-turn/smart-turn-v3.2-gpu.onnx",
        "smart_turn_int8": "smart-turn/smart-turn-v3.2-cpu.onnx",
    }
    return {key: (MODELS_DIR / name).stat().st_size for key, name in names.items()}


# --- reporting ----------------------------------------------------------------------


def pct(value: float) -> str:
    """Percentage with one decimal, or n/a."""
    return "n/a" if math.isnan(value) else f"{100 * value:.1f}%"


def ms(value: float) -> str:
    """Milliseconds with no decimals, or n/a."""
    return "n/a" if math.isnan(value) else f"{value:.0f}"


def print_tables(results: list[dict[str, Any]]) -> None:
    """Print the summary, the pause breakdowns and the counters as Markdown."""
    print(
        "\n| configuration | precision (95% CI) | recall (95% CI) | F1 | premature cuts | "
        "latency p50 / p95, ms | boundary offset p50 / p95, ms | Smart Turn checks | "
        "early ends | late answers |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        c = r["counters"]
        print(
            f"| {r['name']} | {pct(r['precision'])} ({pct(r['precision_ci'][0])}-"
            f"{pct(r['precision_ci'][1])}) | {pct(r['recall'])} ({pct(r['recall_ci'][0])}-"
            f"{pct(r['recall_ci'][1])}) | {pct(r['f1'])} | {pct(r['premature_rate'])} | "
            f"{ms(r['latency_p50_ms'])} / {ms(r['latency_p95_ms'])} | "
            f"{ms(r['offset_p50_ms'])} / {ms(r['offset_p95_ms'])} | "
            f"{c.get('turn_checks', '-')} | {c.get('turn_early_ends', '-')} | "
            f"{c.get('turn_late', '-')} |"
        )
    pauses = [str(p) for p in (*PAUSES_MS, FINAL_PAUSE_MS)]
    print("\nPhrase ends hit, by the silence that follows the phrase (ms):")
    print("| configuration | " + " | ".join(pauses) + " |")
    print("|---|" + "---|" * len(pauses))
    for r in results:
        cells = [_fraction(r["recall_by_pause"].get(p)) for p in pauses]
        print(f"| {r['name']} | " + " | ".join(cells) + " |")
    print("\nMid-sentence segments cut, by the silence that follows them (ms):")
    print("| configuration | " + " | ".join(pauses) + " |")
    print("|---|" + "---|" * len(pauses))
    for r in results:
        cells = [_fraction(r["cut_by_pause"].get(p)) for p in pauses]
        print(f"| {r['name']} | " + " | ".join(cells) + " |")


def _fraction(pair: list[int] | None) -> str:
    if not pair or pair[1] == 0:
        return "-"
    return f"{100 * pair[0] / pair[1]:.0f}% ({pair[0]}/{pair[1]})"


def main() -> int:
    """Run the whole evaluation and write the results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DATA_DIR / "endpointing_results.json")
    parser.add_argument("--streams", type=int, default=60)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()

    started = time.perf_counter()
    ends, mids = load_utterances()
    print(f"utterances <= {MAX_UTTERANCE_S:.0f} s: {len(ends)} ends, {len(mids)} mid-sentence")
    rng = np.random.default_rng(args.seed)
    streams = [build_stream(ends, mids, rng, groups=args.groups) for _ in range(args.streams)]
    total_s = sum(len(s.audio) for s in streams) / RATE_HZ
    labelled = [u for s in streams for u in s.units]
    print(
        f"streams: {len(streams)}, audio {total_s / 60:.1f} min, segments {len(labelled)} "
        f"({sum(u.is_end for u in labelled)} phrase ends)"
    )

    vad = SileroVad(MODELS_DIR / "silero_vad.onnx")
    probabilities = [vad_probabilities(vad, s) for s in streams]
    speed = measure_speed(streams, vad)
    p50 = math.ceil(speed["turn_predict_ms_p50"] / (FRAME_MS))
    p95 = math.ceil(speed["turn_predict_ms_p95"] / (FRAME_MS))
    delays = {"p50": p50, "p95": p95}
    print(f"speed: {json.dumps(speed, default=float)}")
    memory = measure_memory(DEFAULT_TURN_MODEL)
    print(f"memory: {json.dumps(memory)}")

    cache: dict[bytes, float] = {}
    jobs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for spec in CONFIGS:
            delay = delays[spec.delay] if spec.smart else 0
            jobs.append((spec, pool.submit(run_config, spec, streams, probabilities, delay, cache)))
        results = [job.result() for _, job in jobs]
    print_tables(results)

    report = {
        "generated_by": "scripts/eval_endpointing.py",
        "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "data": {
            "corpus": "LibriSpeech test-clean (CC BY 4.0) with LibriSpeech-PC punctuation",
            "max_utterance_s": MAX_UTTERANCE_S,
            "sentence_end_utterances": len(ends),
            "mid_sentence_utterances": len(mids),
            "streams": args.streams,
            "groups_per_stream": args.groups,
            "seed": args.seed,
            "audio_min": total_s / 60,
            "segments": len(labelled),
            "phrase_ends": sum(u.is_end for u in labelled),
            "pauses_ms": list(PAUSES_MS),
            "final_pause_ms": FINAL_PAUSE_MS,
            "tolerance_ms": TOLERANCE_MS,
        },
        "speed": speed,
        "memory_mb": memory,
        "model_bytes": model_sizes(),
        "delay_frames": delays,
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.out} ({report['wall_seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
