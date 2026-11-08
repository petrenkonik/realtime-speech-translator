"""Milestone 1 soak run (S1, with the numbers A1 needs): looped speech through a VB-Cable.

Speech from the SAPI voices is played into ``CABLE Input`` in a loop; the real capture code
reads ``CABLE Output``, resamples, finds speech and publishes chunks, while the other cores run
a CPU load (``scripts/stress.py``). After ``--minutes`` the run is judged against S1:

* 0 input overflows and 0 ring losses,
* longest audio callback < 5 ms,
* p99 chunk finalization latency <= 50 ms,
* RSS growth < 20 MB per hour, measured after a warm-up,
* the chunk count equals the offline count for the same audio,
* plus: no output underruns or driver underflows, no reconnects,
* and, for A1, 0 write operations by this process during the measuring window (Windows I/O
  counters; ``scripts/procmon_check.ps1`` adds the Process Monitor view).

Run only on request, in a normal shell (it plays into the virtual cable, not the speakers)::

    python -B scripts/soak.py --minutes 60

Nothing is written during the measuring window: the speech is made and the models are read
first, then ``StrictGuard`` is armed (a disk write or a socket would end the run), and the
report is printed at the end. ``--progress`` prints one line a minute, which counts as
console writes and so spoils A1; use it for short trial runs only.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_CALLBACK_MS = 5.0
MAX_FINALIZATION_P99_MS = 50.0
MAX_RSS_GROWTH_MB_PER_H = 20.0
_LATENCY_WINDOW = 1000  # CaptureStats keeps this many finalization latencies
_CHUNK_TOLERANCE = 0.01
_PLAY_BLOCK = 1_600  # 100 ms at 16 kHz
_STREAM_SEEDS = tuple(range(8))


class SoakError(RuntimeError):
    """The soak run could not start or lost a device; the message says what to do."""


@dataclass(frozen=True)
class SoakOptions:
    """What to run.

    Attributes:
        minutes: Length of the measuring window.
        cable_in: Output device the speech is played into.
        cable_out: Input device that hears it.
        stress_cores: Busy cores; ``None`` is all logical cores but one.
        stress_normal_priority: Run the load at normal priority (default: below normal, which
            keeps the desktop usable but is a little kinder to the audio threads).
        strict: Arm ``StrictGuard`` for the window.
        progress: Print one line a minute.
        rss_warmup_s: Time before the RSS baseline is taken (models and caches settle).
        sample_interval_s: How often RSS and latencies are sampled.
    """

    minutes: float = 60.0
    cable_in: str = field(
        default_factory=lambda: os.environ.get("PIPELINE_CABLE_IN", "CABLE Input")
    )
    cable_out: str = field(
        default_factory=lambda: os.environ.get("PIPELINE_CABLE_OUT", "CABLE Output")
    )
    stress_cores: int | None = None
    stress_normal_priority: bool = False
    strict: bool = True
    progress: bool = False
    rss_warmup_s: float = 300.0
    sample_interval_s: float = 60.0


@dataclass(frozen=True)
class SoakReport:
    """The measurements of one run; see the module docstring for the limits."""

    minutes: float
    chunks: int
    expected_chunks: int
    input_overflow: int
    ring_lost_frames: int
    max_callback_ms: float
    p99_finalization_ms: float
    max_finalization_ms: float
    output_underruns: int
    driver_underflows: int
    reconnects: int
    rss_growth_mb_per_h: float
    write_operations: int
    write_bytes: int
    max_scheduler_stall_ms: float = 0.0


# --- pure helpers (unit tested) -------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100) with linear interpolation; 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def new_values(
    before: Sequence[float],
    after: Sequence[float],
    *,
    window: int = _LATENCY_WINDOW,
    chunks_gained: int = 0,
) -> list[float]:
    """The values a sliding window gained between two looks.

    While the window is not full its length tells exactly what was added (only chunks that
    ended on silence record a value, so the chunk count would over-count). A full window
    slides and its length stays put; then the chunk gain is the best estimate, capped at the
    window size.
    """
    if len(after) < window:
        return list(after[len(before) :])
    gained = min(chunks_gained, len(after))
    return list(after[len(after) - gained :]) if gained > 0 else []


def rss_growth_mb_per_hour(samples: Sequence[tuple[float, float]], warmup_s: float) -> float:
    """Growth of the resident memory per hour, from the first sample after the warm-up.

    Args:
        samples: ``(seconds since start, MB)`` pairs in time order.
        warmup_s: Samples before this time are start-up cost (models, caches), not growth.

    Returns:
        MB per hour; 0.0 when memory shrank or there is not enough data after the warm-up.
    """
    settled = [(t, mb) for t, mb in samples if t >= warmup_s]
    if len(settled) < 2 or settled[-1][0] <= settled[0][0]:
        return 0.0
    (t0, mb0), (t1, mb1) = settled[0], settled[-1]
    return max(0.0, (mb1 - mb0) * 3600.0 / (t1 - t0))


class JitterProbe:
    """Measures how late a sleeping thread of this process wakes up.

    A late wake-up on an idle thread means the whole process was not scheduled (a busy
    machine, another program), not that one of its stages is slow. ``observe`` takes one
    lateness in seconds; ``take`` returns the worst and the number of long ones since the last
    call and starts over, while ``worst_ms`` keeps the worst of the whole run.
    """

    def __init__(self, long_ms: float = 100.0) -> None:
        self._long_ms = long_ms
        self._lock = threading.Lock()
        self._worst_ms = 0.0
        self._long = 0
        self.worst_ms = 0.0

    def observe(self, late_s: float) -> None:
        """Record one wake-up that was ``late_s`` seconds late."""
        late_ms = late_s * 1000.0
        with self._lock:
            self._worst_ms = max(self._worst_ms, late_ms)
            self.worst_ms = max(self.worst_ms, late_ms)
            if late_ms >= self._long_ms:
                self._long += 1

    def take(self) -> tuple[float, int]:
        """(worst lateness in ms, number of long ones) since the previous call."""
        with self._lock:
            result = (self._worst_ms, self._long)
            self._worst_ms, self._long = 0.0, 0
        return result

    def run(self, stop: threading.Event, interval_s: float = 0.01) -> None:
        """Sleep and measure until ``stop`` is set (the body of the probe thread)."""
        while not stop.is_set():
            before = time.perf_counter()
            time.sleep(interval_s)
            self.observe(time.perf_counter() - before - interval_s)


def _criteria(report: SoakReport) -> list[tuple[str, str, str, bool]]:
    """(name, value, limit, ok) for every criterion."""
    tolerance = max(2, int(report.expected_chunks * _CHUNK_TOLERANCE))
    chunk_ok = (
        report.expected_chunks > 0 and abs(report.chunks - report.expected_chunks) <= tolerance
    )
    return [
        ("input overflows", str(report.input_overflow), "0", report.input_overflow == 0),
        ("ring lost frames", str(report.ring_lost_frames), "0", report.ring_lost_frames == 0),
        (
            "max callback ms",
            f"{report.max_callback_ms:.2f}",
            f"< {MAX_CALLBACK_MS:g}",
            report.max_callback_ms < MAX_CALLBACK_MS,
        ),
        (
            "p99 finalization ms",
            f"{report.p99_finalization_ms:.1f} (max {report.max_finalization_ms:.1f})",
            f"<= {MAX_FINALIZATION_P99_MS:g}",
            report.p99_finalization_ms <= MAX_FINALIZATION_P99_MS,
        ),
        (
            "RSS growth MB/h",
            f"{report.rss_growth_mb_per_h:.1f}",
            f"< {MAX_RSS_GROWTH_MB_PER_H:g}",
            report.rss_growth_mb_per_h < MAX_RSS_GROWTH_MB_PER_H,
        ),
        ("output underruns", str(report.output_underruns), "0", report.output_underruns == 0),
        ("driver underflows", str(report.driver_underflows), "0", report.driver_underflows == 0),
        ("reconnects", str(report.reconnects), "0", report.reconnects == 0),
        (
            "process write operations",
            f"{report.write_operations} ({report.write_bytes} bytes)",
            "0",
            report.write_operations == 0 and report.write_bytes == 0,
        ),
        (
            "chunks",
            f"{report.chunks} (offline count {report.expected_chunks})",
            f"within {tolerance}",
            chunk_ok,
        ),
    ]


def evaluate(report: SoakReport) -> list[str]:
    """The failed criteria, one text each; an empty list means the run passed."""
    return [
        f"{name}: {value} (limit {limit})" for name, value, limit, ok in _criteria(report) if not ok
    ]


def format_report(report: SoakReport) -> str:
    """A table of every criterion with its measured value, limit and verdict."""
    lines = [f"Soak report, {report.minutes:.1f} min"]
    for name, value, limit, ok in _criteria(report):
        lines.append(f"  {'ok  ' if ok else 'FAIL'} {name:<26} {value:<28} limit {limit}")
    lines.append(
        f"  info scheduler stall max ms     {report.max_scheduler_stall_ms:.0f}"
        "                        (not judged: a late wake-up of an idle thread)"
    )
    lines.append("PASS" if not evaluate(report) else "FAIL")
    return "\n".join(lines)


# --- Windows probes --------------------------------------------------------------------------


class _IoCounters(ctypes.Structure):
    _fields_ = (
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    )


class _MemoryCounters(ctypes.Structure):
    _fields_ = (
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    )


def _kernel() -> tuple[ctypes.WinDLL, ctypes.WinDLL, wintypes.HANDLE]:  # type: ignore[name-defined,unused-ignore]
    """kernel32, psapi and this process's handle, with 64-bit safe signatures."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
    psapi = ctypes.WinDLL("psapi", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessIoCounters.argtypes = (wintypes.HANDLE, ctypes.POINTER(_IoCounters))
    kernel32.GetProcessIoCounters.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_MemoryCounters),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    return kernel32, psapi, kernel32.GetCurrentProcess()


def io_write_counters() -> tuple[int, int]:
    """(write operations, bytes written) of this process so far, as Windows counts them."""
    kernel32, _, handle = _kernel()
    counters = _IoCounters()
    if not kernel32.GetProcessIoCounters(handle, ctypes.byref(counters)):
        raise SoakError(f"GetProcessIoCounters failed (error {ctypes.get_last_error()})")
    return int(counters.WriteOperationCount), int(counters.WriteTransferCount)


def private_megabytes() -> float:
    """Private memory of this process in MB (what a leak grows)."""
    _, psapi, handle = _kernel()
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        raise SoakError(f"GetProcessMemoryInfo failed (error {ctypes.get_last_error()})")
    return counters.PrivateUsage / (1024 * 1024)


# --- the run ----------------------------------------------------------------------------------


def run_soak(options: SoakOptions) -> SoakReport:
    """Play looped speech through the cable for ``options.minutes`` and measure.

    Raises:
        SoakError: The cable or the voices are missing, or a device is lost during the run.
    """
    sys.path.insert(0, str(ROOT / "tests"))
    from sapi_speech import build_stream, synthesize

    from pipeline.bus import ChunkBus
    from pipeline.capture import CaptureConfig, CaptureSession
    from pipeline.devices import (
        DeviceNotFoundError,
        SoundDeviceListing,
        find_device,
        open_sounddevice_input,
        open_sounddevice_output,
    )
    from pipeline.guards import StrictGuard
    from pipeline.output import OutputConfig, OutputSession
    from pipeline.resample import StreamingResampler
    from pipeline.vad import Segmenter, SileroVad, VadStage

    listing = SoundDeviceListing()
    try:
        find_device(listing.list_devices(), options.cable_in, kind="output")
        find_device(listing.list_devices(), options.cable_out, kind="input")
    except DeviceNotFoundError as error:
        raise SoakError(f"VB-Cable not found: {error}") from error
    model = ROOT / "models" / "silero_vad.onnx"
    if not model.is_file():
        raise SoakError("models/silero_vad.onnx is missing: run scripts/fetch_models.py")

    # Setup writes files (the voices go through WAV files in a temporary folder), so it is done
    # and cleaned up before the measuring window opens.
    with tempfile.TemporaryDirectory() as folder:
        utterances, error_text = synthesize(Path(folder))
    if not utterances:
        raise SoakError(f"SAPI voices unavailable: {error_text}")
    streams = [build_stream(utterances, seed) for seed in _STREAM_SEEDS]
    per_stream_chunks = []
    for stream in streams:
        offline = VadStage(SileroVad(model), Segmenter())
        count = len(offline.push(stream)) + (1 if offline.flush() is not None else 0)
        per_stream_chunks.append(count)
    if min(per_stream_chunks) < 1:
        raise SoakError("the test speech produced no chunks offline; nothing to compare with")

    bus = ChunkBus()
    subscription = bus.subscribe("soak-consumer", maxsize=64)
    consumed = threading.Event()

    def drain() -> None:
        while not consumed.is_set():
            subscription.get(0.1)

    drainer = threading.Thread(target=drain, name="soak-consumer", daemon=True)
    jitter = JitterProbe()
    jitter_thread = threading.Thread(
        target=jitter.run, args=(consumed,), name="soak-jitter", daemon=True
    )
    capture = CaptureSession(
        CaptureConfig(options.cable_out),
        listing,
        open_sounddevice_input,
        VadStage(SileroVad(model), Segmenter()),
        bus,
    )
    output = OutputSession(OutputConfig(options.cable_in), listing, open_sounddevice_output)
    guard = StrictGuard()
    workers: list[object] = []
    stress = _load_stress()
    cores = (
        options.stress_cores
        if options.stress_cores is not None
        else max(1, (os.cpu_count() or 2) - 1)
    )
    window_s = options.minutes * 60.0

    latencies: list[float] = []
    rss_samples: list[tuple[float, float]] = []
    seen_window: tuple[float, ...] = ()
    seen_chunks = 0
    expected_chunks = 0
    started_capture = False
    try:
        drainer.start()
        jitter_thread.start()
        output.start()
        capture.start()
        started_capture = True
        time.sleep(0.5)
        workers = stress.start_workers(
            cores, window_s + 120.0, below_normal=not options.stress_normal_priority
        )
        time.sleep(1.0)  # let the load reach every core before the window opens
        if options.strict:
            guard.arm()
        writes0, bytes0 = io_write_counters()
        began = time.monotonic()
        next_sample = began
        loop = 0
        while time.monotonic() - began < window_s:
            index = loop % len(streams)
            _play(output, streams[index], StreamingResampler)
            expected_chunks += per_stream_chunks[index]
            loop += 1
            now = time.monotonic()
            if now >= next_sample:
                stats = capture.stats()
                latencies += new_values(
                    seen_window,
                    stats.finalization_latency_ms,
                    chunks_gained=stats.chunks - seen_chunks,
                )
                seen_window, seen_chunks = stats.finalization_latency_ms, stats.chunks
                rss_samples.append((now - began, private_megabytes()))
                next_sample = now + options.sample_interval_s
                if capture.fatal_error is not None:
                    raise SoakError(f"capture ended: {capture.fatal_error}")
                stall_ms, long_stalls = jitter.take()
                if options.progress:
                    print(
                        f"{(now - began) / 60:5.1f} min  chunks {stats.chunks}/{expected_chunks}"
                        f"  overflow {stats.input_overflow}  ring_lost {stats.ring_lost_frames}"
                        f"  underruns {output.stats().underruns}"
                        f"  cb {stats.max_callback_ms:.2f} ms  rss {rss_samples[-1][1]:.0f} MB"
                        f"  stall {stall_ms:.0f} ms ({long_stalls} long)",
                        flush=True,
                    )
        time.sleep(1.0)  # the last stream's silence closes its final chunk
        writes1, bytes1 = io_write_counters()
        elapsed_min = (time.monotonic() - began) / 60.0
    finally:
        if started_capture:
            capture.stop()
        output.stop()
        consumed.set()
        guard.disarm()
        stress.stop_workers(workers)  # type: ignore[arg-type]

    stats = capture.stats()
    latencies += new_values(
        seen_window,
        stats.finalization_latency_ms,
        chunks_gained=stats.chunks - seen_chunks,
    )
    out_stats = output.stats()
    warmup = min(options.rss_warmup_s, window_s / 4.0)
    return SoakReport(
        minutes=elapsed_min,
        chunks=stats.chunks,
        expected_chunks=expected_chunks,
        input_overflow=stats.input_overflow,
        ring_lost_frames=stats.ring_lost_frames,
        max_callback_ms=stats.max_callback_ms,
        p99_finalization_ms=percentile(latencies, 99),
        max_finalization_ms=max(latencies, default=0.0),
        output_underruns=out_stats.underruns,
        driver_underflows=out_stats.driver_underflows,
        reconnects=stats.reconnects + out_stats.reconnects,
        rss_growth_mb_per_h=rss_growth_mb_per_hour(rss_samples, warmup),
        write_operations=writes1 - writes0,
        write_bytes=bytes1 - bytes0,
        max_scheduler_stall_ms=jitter.worst_ms,
    )


def _play(output: object, signal: object, resampler_class: type) -> None:
    """Play a 16 kHz signal in real time and wait until the ring is empty."""
    rate = output.sample_rate_hz  # type: ignore[attr-defined]
    if rate is None:
        raise SoakError("the cable output was lost during the run")
    resampler = resampler_class(16_000, rate)
    output.begin_utterance()  # type: ignore[attr-defined]
    for start in range(0, len(signal), _PLAY_BLOCK):  # type: ignore[arg-type]
        block = resampler.process(signal[start : start + _PLAY_BLOCK])  # type: ignore[index]
        if not output.write_all(block, timeout_s=30.0):  # type: ignore[attr-defined]
            raise SoakError("the output ring stayed full for 30 s")
    output.end_utterance()  # type: ignore[attr-defined]
    deadline = time.monotonic() + 30.0
    while output.stats().queued_ms > 0 and time.monotonic() < deadline:  # type: ignore[attr-defined]
        time.sleep(0.02)
    time.sleep(0.3)  # the device buffer still holds the last block


def _load_stress() -> object:
    import importlib.util

    spec = importlib.util.spec_from_file_location("stress", ROOT / "scripts" / "stress.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry; the exit code is 0 only when every criterion holds."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument(
        "--cable-in", default=SoakOptions().cable_in, help="output device name part"
    )
    parser.add_argument(
        "--cable-out", default=SoakOptions().cable_out, help="input device name part"
    )
    parser.add_argument("--stress-cores", type=int, help="default: all logical cores but one")
    parser.add_argument("--stress-normal-priority", action="store_true")
    parser.add_argument("--no-strict", action="store_true", help="do not arm StrictGuard")
    parser.add_argument("--progress", action="store_true", help="one line a minute (spoils A1)")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    options = SoakOptions(
        minutes=args.minutes,
        cable_in=args.cable_in,
        cable_out=args.cable_out,
        stress_cores=args.stress_cores,
        stress_normal_priority=args.stress_normal_priority,
        strict=not args.no_strict,
        progress=args.progress,
    )
    print(f"soak pid {os.getpid()}, {options.minutes:g} min, setup first...", flush=True)
    try:
        report = run_soak(options)
    except SoakError as error:
        print(f"soak could not run: {error}", file=sys.stderr)
        return 2
    print(format_report(report))
    return 0 if not evaluate(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
