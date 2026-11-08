"""Transcription stage on a fake whisper (spec T2, T3 and the chunk risks C8, C9, C11)."""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pipeline.bus import Bus, Subscription
from pipeline.glossary import Glossary
from pipeline.metrics import Metrics
from pipeline.transcribe import (
    WHISPER_MODELS,
    FasterWhisperEngine,
    TranscribeConfig,
    TranscribeConfigError,
    TranscribeStage,
    Utterance,
    collapse_repeats,
    ensure_av_stub,
    whisper_model_dir,
)
from pipeline.vad import SAMPLE_RATE_HZ, Chunk, ChunkReason, ModelNotFoundError

pytestmark = pytest.mark.unit

MS = SAMPLE_RATE_HZ // 1000


def make_chunk(
    chunk_id: int,
    start_ms: int,
    length_ms: int = 1000,
    amplitude: float = 0.1,
    reason: ChunkReason = "silence",
) -> Chunk:
    samples = np.full(length_ms * MS, amplitude, dtype=np.float32)
    samples.setflags(write=False)
    start = start_ms * MS
    end = start + len(samples)
    return Chunk(
        id=chunk_id,
        samples=samples,
        start_sample=start,
        end_sample=end,
        speech_end_sample=end,
        finalized_sample=end + 450 * MS,
        reason=reason,
    )


@dataclass
class Call:
    samples: int
    hotwords: str | None
    initial_prompt: str | None


class FakeEngine:
    """Returns scripted texts and records what it was asked."""

    def __init__(self, texts: list[str] | None = None) -> None:
        self.texts = deque(texts or [])
        self.calls: list[Call] = []
        self.fail_next = False

    def transcribe(
        self, samples: np.ndarray, *, hotwords: str | None, initial_prompt: str | None
    ) -> str:
        self.calls.append(Call(len(samples), hotwords, initial_prompt))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("decoder failed")
        return self.texts.popleft() if self.texts else f"text {len(self.calls)}"


class Rig:
    def __init__(
        self,
        engine: FakeEngine | None = None,
        config: TranscribeConfig | None = None,
        glossary: Glossary | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.chunks: Bus[Chunk] = Bus()
        self.texts: Bus[Utterance] = Bus()
        self.engine = engine or FakeEngine()
        self.metrics = Metrics()
        self.out: Subscription[Utterance] = self.texts.subscribe("test")
        self.stage = TranscribeStage(
            self.chunks.subscribe("whisper"),
            self.texts,
            self.engine,
            config or TranscribeConfig(),
            glossary=glossary,
            metrics=self.metrics,
            clock=clock,
        )

    def run_all(self) -> list[Utterance]:
        while self.stage.step(timeout_s=0):
            pass
        found = []
        while (utterance := self.out.get(timeout_s=0)) is not None:
            found.append(utterance)
        return found


# --- T2 and the shape of an utterance ----------------------------------------------------


def test_t2_whisper_runs_once_per_utterance_however_many_listeners_there_are() -> None:
    rig = Rig()
    lane_de = rig.texts.subscribe("de")
    lane_fr = rig.texts.subscribe("fr")
    for index in range(5):
        rig.chunks.publish(make_chunk(index + 1, start_ms=index * 5000))
        rig.stage.step(timeout_s=0)
    assert len(rig.engine.calls) == 5
    assert [u.text for u in iter(lambda: lane_de.get(timeout_s=0), None)] == [
        f"text {n}" for n in range(1, 6)
    ]
    assert lane_fr.stats().delivered == 0  # whisper did not run again for the second listener
    assert lane_fr.stats().depth == 5


def test_an_utterance_carries_the_chunk_timing_and_the_decode_time() -> None:
    ticks = iter([10.0, 10.5])  # the clock is read before and after the decode
    rig = Rig(FakeEngine(["Hello there."]), clock=lambda: next(ticks))
    chunk = make_chunk(7, start_ms=2000, length_ms=1500)
    rig.chunks.publish(chunk)
    (utterance,) = rig.run_all()
    assert utterance.id == 7
    assert utterance.chunk_ids == (7,)
    assert utterance.text == "Hello there."
    assert utterance.start_sample == chunk.start_sample
    assert utterance.speech_end_sample == chunk.speech_end_sample
    assert utterance.finalized_sample == chunk.finalized_sample
    assert utterance.audio_ms == 1500
    assert utterance.decode_ms == pytest.approx(500.0)
    assert utterance.reason == "silence"


def test_the_engine_gets_the_chunk_audio_untouched() -> None:
    rig = Rig()
    rig.chunks.publish(make_chunk(1, start_ms=0, length_ms=800))
    rig.run_all()
    assert rig.engine.calls[0].samples == 800 * MS


def test_closing_the_bus_ends_the_stage_after_the_backlog() -> None:
    rig = Rig()
    rig.chunks.publish(make_chunk(1, start_ms=0))
    rig.chunks.close(drain=True)
    assert len(rig.run_all()) == 1
    assert rig.stage.step(timeout_s=0) is False
    assert rig.stage.finished


def test_an_idle_stage_keeps_waiting() -> None:
    rig = Rig()
    assert rig.stage.step(timeout_s=0) is False  # nothing to do
    assert not rig.stage.finished  # but the bus is open, more may come


# --- T3: the glossary reaches whisper ----------------------------------------------------


def test_t3_glossary_terms_reach_whisper_as_hotwords() -> None:
    glossary = Glossary.from_toml("[[term]]\nsource='Kubernetes'\nde='Kubernetes'\n")
    rig = Rig(glossary=glossary)
    rig.chunks.publish(make_chunk(1, start_ms=0))
    rig.run_all()
    assert rig.engine.calls[0].hotwords == "Kubernetes"


def test_without_a_glossary_there_are_no_hotwords() -> None:
    rig = Rig()
    rig.chunks.publish(make_chunk(1, start_ms=0))
    rig.run_all()
    assert rig.engine.calls[0].hotwords is None


# --- C8: noise and hallucinations --------------------------------------------------------


def test_c8_a_chunk_that_is_almost_silent_never_reaches_whisper() -> None:
    rig = Rig()
    rig.chunks.publish(make_chunk(1, start_ms=0, amplitude=0.0005))
    assert rig.run_all() == []
    assert rig.engine.calls == []
    assert rig.metrics.counter("transcribe.skipped_quiet").value == 1


def test_c8_an_empty_transcript_produces_no_utterance() -> None:
    rig = Rig(FakeEngine(["   "]))
    rig.chunks.publish(make_chunk(1, start_ms=0))
    assert rig.run_all() == []
    assert rig.metrics.counter("transcribe.skipped_empty").value == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Thank you. Thank you. Thank you. Thank you. Thank you.", "Thank you."),
        ("Yes yes yes yes yes yes", "Yes"),
        ("I said no no no", "I said no no no"),  # a real repetition below the limit stays
        ("one two three one two three one two three one two three", "one two three"),
        ("Go. Then stop stop stop stop now.", "Go. Then stop now."),
        ("Nothing repeats here.", "Nothing repeats here."),
        ("", ""),
    ],
)
def test_c8_repeated_phrases_are_collapsed(text: str, expected: str) -> None:
    assert collapse_repeats(text, limit=4) == expected


def test_c8_a_collapsed_transcript_is_counted() -> None:
    rig = Rig(FakeEngine(["Thank you. " * 6]))
    rig.chunks.publish(make_chunk(1, start_ms=0))
    (utterance,) = rig.run_all()
    assert utterance.text == "Thank you."
    assert rig.metrics.counter("transcribe.repeats_collapsed").value == 1


# --- C9: falling behind ------------------------------------------------------------------


def test_c9_waiting_chunks_close_together_are_decoded_in_one_pass() -> None:
    rig = Rig(FakeEngine(["a b", "c d", "e f"]))
    for index in range(3):  # 1 s of speech each, 300 ms apart
        rig.chunks.publish(make_chunk(index + 1, start_ms=index * 1300))
    (utterance,) = rig.run_all()
    assert len(rig.engine.calls) == 1
    assert utterance.chunk_ids == (1, 2, 3)
    assert utterance.id == 1
    # three seconds of speech plus two padded gaps, never more than the real gap
    assert 3000 * MS < rig.engine.calls[0].samples <= 3600 * MS
    assert rig.metrics.counter("transcribe.merged_chunks").value == 2


def test_c9_the_merged_utterance_ends_where_the_last_chunk_ends() -> None:
    rig = Rig()
    first, second = make_chunk(1, start_ms=0), make_chunk(2, start_ms=1500)
    rig.chunks.publish(first)
    rig.chunks.publish(second)
    (utterance,) = rig.run_all()
    assert utterance.start_sample == first.start_sample
    assert utterance.speech_end_sample == second.speech_end_sample
    assert utterance.finalized_sample == second.finalized_sample


def test_c9_chunks_far_apart_are_not_merged() -> None:
    rig = Rig()
    rig.chunks.publish(make_chunk(1, start_ms=0))
    rig.chunks.publish(make_chunk(2, start_ms=6000))  # a 5 s pause between them
    assert len(rig.run_all()) == 2
    assert len(rig.engine.calls) == 2


def test_c9_a_merge_never_exceeds_the_maximum_length() -> None:
    config = TranscribeConfig(merge_max_ms=5000)
    rig = Rig(config=config)
    for index in range(4):  # 2 s each, 100 ms apart: two fit in 5 s, three do not
        rig.chunks.publish(make_chunk(index + 1, start_ms=index * 2100, length_ms=2000))
    utterances = rig.run_all()
    assert [u.chunk_ids for u in utterances] == [(1, 2), (3, 4)]
    assert all(call.samples <= 5000 * MS for call in rig.engine.calls)


def test_c9_a_single_chunk_is_never_padded_or_changed() -> None:
    rig = Rig()
    rig.chunks.publish(make_chunk(1, start_ms=0, length_ms=700))
    rig.run_all()
    assert rig.engine.calls[0].samples == 700 * MS


# --- C11: the sentence continues in the next chunk ---------------------------------------


def test_c11_an_unfinished_sentence_is_passed_on_as_context() -> None:
    rig = Rig(FakeEngine(["we can start as soon as the", "budget is approved."]))
    rig.chunks.publish(make_chunk(1, start_ms=0))
    rig.run_all()
    rig.chunks.publish(make_chunk(2, start_ms=2000))
    rig.run_all()
    assert rig.engine.calls[0].initial_prompt is None
    assert rig.engine.calls[1].initial_prompt == "we can start as soon as the"


def test_c11_a_finished_sentence_gives_no_context() -> None:
    rig = Rig(FakeEngine(["All done.", "Next topic."]))
    rig.chunks.publish(make_chunk(1, start_ms=0))
    rig.run_all()
    rig.chunks.publish(make_chunk(2, start_ms=2000))
    rig.run_all()
    assert rig.engine.calls[1].initial_prompt is None


def test_c11_a_forced_cut_gives_context_even_after_a_full_stop() -> None:
    rig = Rig(FakeEngine(["Long text here.", "and more."]))
    rig.chunks.publish(make_chunk(1, start_ms=0, reason="max_length"))
    rig.run_all()
    rig.chunks.publish(make_chunk(2, start_ms=1000))
    rig.run_all()
    assert rig.engine.calls[1].initial_prompt == "Long text here."


def test_c11_old_context_is_not_carried_across_a_long_pause() -> None:
    rig = Rig(FakeEngine(["we can start as soon as the", "Hello."]))
    rig.chunks.publish(make_chunk(1, start_ms=0))
    rig.run_all()
    rig.chunks.publish(make_chunk(2, start_ms=30_000))
    rig.run_all()
    assert rig.engine.calls[1].initial_prompt is None


def test_c11_only_the_end_of_a_long_text_is_passed_on() -> None:
    long_text = "word " * 100  # 500 characters, no final punctuation
    rig = Rig(FakeEngine([long_text, "x."]), config=TranscribeConfig(prompt_tail_chars=50))
    rig.chunks.publish(make_chunk(1, start_ms=0))
    rig.run_all()
    rig.chunks.publish(make_chunk(2, start_ms=1500))
    rig.run_all()
    prompt = rig.engine.calls[1].initial_prompt
    assert prompt is not None
    assert len(prompt) <= 50
    assert prompt.endswith("word")


# --- failures and threads ----------------------------------------------------------------


def test_a_failing_decode_drops_that_chunk_and_the_stage_goes_on() -> None:
    rig = Rig(FakeEngine(["second"]))
    rig.engine.fail_next = True
    rig.chunks.publish(make_chunk(1, start_ms=0))
    rig.chunks.publish(make_chunk(2, start_ms=10_000))
    utterances = rig.run_all()
    assert [u.text for u in utterances] == ["second"]
    assert rig.metrics.counter("transcribe.errors").value == 1
    assert isinstance(rig.stage.last_error, RuntimeError)


def test_the_decode_time_and_the_real_time_factor_are_recorded() -> None:
    ticks = iter([0.0, 2.0])
    rig = Rig(clock=lambda: next(ticks))
    rig.chunks.publish(make_chunk(1, start_ms=0, length_ms=4000))
    rig.run_all()
    assert rig.metrics.histogram("transcribe.decode_ms").summary().p50 == pytest.approx(2000.0)
    assert rig.metrics.histogram("transcribe.rtf").summary().p50 == pytest.approx(0.5)


def test_the_thread_transcribes_and_stops_promptly() -> None:
    rig = Rig()
    rig.stage.start()
    try:
        rig.chunks.publish(make_chunk(1, start_ms=0))
        utterance = rig.out.get(timeout_s=2)
        assert utterance is not None
        assert utterance.id == 1
    finally:
        started = time.monotonic()
        rig.stage.stop()
    assert time.monotonic() - started < 1.0
    assert not rig.stage.running


def test_starting_twice_is_an_error() -> None:
    rig = Rig()
    rig.stage.start()
    try:
        with pytest.raises(RuntimeError, match="already"):
            rig.stage.start()
    finally:
        rig.stage.stop()


def test_the_thread_ends_by_itself_when_the_bus_closes() -> None:
    rig = Rig()
    rig.stage.start()
    rig.chunks.close()
    deadline = time.monotonic() + 2
    while rig.stage.running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not rig.stage.running
    rig.stage.stop()


# --- configuration -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "large-v3"},
        {"model": ""},
        {"language": ""},
        {"beam_size": 0},
        {"threads": 0},
        {"threads": 33},
        {"no_speech_threshold": 1.5},
        {"log_prob_threshold": 0.5},
        {"compression_ratio_threshold": 0.5},
        {"min_rms": -0.1},
        {"repeat_limit": 1},
        {"merge_gap_ms": -1},
        {"merge_max_ms": 500},
        {"prompt_tail_chars": -1},
        {"context_gap_ms": -1},
    ],
)
def test_a_bad_setting_is_refused_with_its_name(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TranscribeConfigError, match=next(iter(kwargs))):
        TranscribeConfig(**kwargs)


def test_both_models_are_offered_and_base_is_the_default() -> None:
    assert WHISPER_MODELS == ("tiny.en", "base.en")
    assert TranscribeConfig().model == "base.en"
    for name in WHISPER_MODELS:
        assert TranscribeConfig(model=name).model == name


def test_the_model_option_picks_the_folder(tmp_path: Path) -> None:
    assert whisper_model_dir(tmp_path, TranscribeConfig(model="tiny.en")) == (
        tmp_path / "whisper-tiny.en"
    )
    assert whisper_model_dir(tmp_path, TranscribeConfig()) == tmp_path / "whisper-base.en"


# --- the faster-whisper adapter ----------------------------------------------------------


class FakeModel:
    def __init__(self, segments: list[str]) -> None:
        self.segments = segments
        self.kwargs: dict[str, Any] = {}
        self.audio: np.ndarray | None = None

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, Any]:
        self.audio, self.kwargs = audio, kwargs
        return (SimpleNamespace(text=text) for text in self.segments), SimpleNamespace()


def test_the_adapter_passes_the_decode_options_and_joins_the_segments() -> None:
    model = FakeModel([" Hello", " world. "])
    engine = FasterWhisperEngine(model, TranscribeConfig(language="en", beam_size=2))
    samples = np.zeros(1600, dtype=np.float32)
    text = engine.transcribe(samples, hotwords="Kubernetes", initial_prompt="so the")
    assert text == "Hello world."
    kwargs = model.kwargs
    assert kwargs["language"] == "en"
    assert kwargs["beam_size"] == 2
    assert kwargs["hotwords"] == "Kubernetes"
    assert kwargs["initial_prompt"] == "so the"
    assert kwargs["vad_filter"] is False  # our own VAD already cut the chunk
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["without_timestamps"] is True
    assert kwargs["temperature"] == 0.0  # no re-decoding with random sampling: bounded latency
    assert kwargs["no_speech_threshold"] == 0.6
    assert kwargs["log_prob_threshold"] == -1.0
    assert kwargs["compression_ratio_threshold"] == 2.4
    assert model.audio is samples


def test_the_adapter_returns_empty_text_when_whisper_finds_no_speech() -> None:
    engine = FasterWhisperEngine(FakeModel([]), TranscribeConfig())
    assert engine.transcribe(np.zeros(1600, np.float32), hotwords=None, initial_prompt=None) == ""


def test_a_missing_model_directory_says_how_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="fetch_models"):
        FasterWhisperEngine.load(tmp_path / "whisper-base.en", TranscribeConfig())


def test_av_is_replaced_by_a_stub_that_refuses_to_decode_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    if importlib.util.find_spec("av") is not None:
        pytest.skip("PyAV is installed here; the stub is only used without it")
    monkeypatch.delitem(sys.modules, "av", raising=False)
    ensure_av_stub()
    with pytest.raises(RuntimeError, match="files"):
        sys.modules["av"].open("x.wav")


def test_the_stub_is_left_alone_when_av_is_already_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "av", marker)
    ensure_av_stub()
    assert sys.modules["av"] is marker


def test_threads_and_locks_are_not_leaked_by_the_tests() -> None:
    names = {thread.name for thread in threading.enumerate()}
    assert "transcribe" not in names
