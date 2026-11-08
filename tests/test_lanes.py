"""Language lane: translate, speak, play; one lane never affects another (T1)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from pipeline.glossary import Glossary
from pipeline.lanes import (
    LaneConfig,
    LaneConfigError,
    LanguageLane,
    Speech,
    Translation,
    TranslationBus,
    TranslationError,
)
from pipeline.metrics import Metrics
from pipeline.transcribe import TextBus, Utterance
from pipeline.vad import Float32Array

pytestmark = pytest.mark.unit

TTS_RATE_HZ = 16_000
DEVICE_RATE_HZ = 16_000


def utterance(
    text: str = "hello there", uid: int = 1, *, speech_end_sample: int = 16_000
) -> Utterance:
    return Utterance(
        id=uid,
        chunk_ids=(uid,),
        text=text,
        start_sample=0,
        speech_end_sample=speech_end_sample,
        finalized_sample=speech_end_sample,
        audio_ms=1000,
        decode_ms=100.0,
        reason="silence",
    )


class FakeTranslator:
    def __init__(self, prefix: str = "[x] ") -> None:
        self.prefix = prefix
        self.calls: list[str] = []
        self.gate: threading.Event | None = None
        self.fail_on: str | None = None
        self.entered = threading.Event()

    def translate(self, text: str) -> str:
        self.calls.append(text)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(5.0)
        if self.fail_on is not None and self.fail_on in text:
            raise RuntimeError("translator broke")
        return self.prefix + text


class FakeTts:
    def __init__(self, rate_hz: int = TTS_RATE_HZ, samples_per_char: int = 100) -> None:
        self.rate_hz = rate_hz
        self.samples_per_char = samples_per_char
        self.spoken: list[str] = []
        self.fail = False

    def synthesize(self, text: str) -> Speech:
        self.spoken.append(text)
        if self.fail:
            raise RuntimeError("tts broke")
        count = len(text) * self.samples_per_char
        return Speech(samples=np.full(count, 0.25, dtype=np.float32), rate_hz=self.rate_hz)


class FakeSink:
    def __init__(self, rate_hz: int | None = DEVICE_RATE_HZ) -> None:
        self.rate_hz = rate_hz
        self.generation = 0
        self.events: list[str] = []
        self.written = 0
        self.accept = True

    @property
    def sample_rate_hz(self) -> int | None:
        return self.rate_hz

    def begin_utterance(self) -> None:
        self.events.append("begin")

    def end_utterance(self) -> None:
        self.events.append("end")

    def write_all(self, samples: Float32Array, timeout_s: float = 5.0) -> bool:
        self.events.append(f"write:{len(samples)}")
        self.written += len(samples)
        return self.accept


class Rig:
    """One lane wired to a bus, with a stream clock the test moves by hand."""

    def __init__(
        self,
        target: str = "de",
        *,
        glossary: Glossary | None = None,
        bus: TextBus | None = None,
        config: LaneConfig | None = None,
        sink: FakeSink | None = None,
        translator: FakeTranslator | None = None,
        tts: FakeTts | None = None,
        resampler_factory: Callable[[int, int], object] | None = None,
        metrics: Metrics | None = None,
        translations: TranslationBus | None = None,
    ) -> None:
        self.bus = bus or TextBus()
        self.metrics = metrics or Metrics()
        self.translator = translator or FakeTranslator()
        self.tts = tts or FakeTts()
        self.sink = sink or FakeSink()
        self.position = 0
        self.lane = LanguageLane(
            target,
            self.bus.subscribe(f"lane-{target}", maxsize=2),
            self.translator,
            self.tts,
            self.sink,
            config,
            glossary=glossary,
            metrics=self.metrics,
            position_samples=lambda: self.position,
            resampler_factory=resampler_factory,  # type: ignore[arg-type]
            translations=translations,
        )

    def count(self, name: str) -> int:
        return self.metrics.counter(f"lane.{self.lane.target}.{name}").value

    def latency_ms(self) -> list[float]:
        histogram = self.metrics.histogram(f"lane.{self.lane.target}.latency_ms")
        return [histogram.summary().p50] if histogram.summary().count else []


# --- the happy path -------------------------------------------------------------------------


def test_an_utterance_is_translated_spoken_and_played_in_order() -> None:
    rig = Rig()
    rig.bus.publish(utterance("hello there"))
    assert rig.lane.step(0.1)
    assert rig.translator.calls == ["hello there"]
    assert rig.tts.spoken == ["[x] hello there"]
    assert rig.sink.events[0] == "begin"
    assert rig.sink.events[-1] == "end"
    assert rig.sink.written == len("[x] hello there") * 100
    assert rig.count("utterances") == 1


def test_step_without_work_returns_false() -> None:
    rig = Rig()
    assert not rig.lane.step(0.01)


def test_the_lane_is_finished_when_the_bus_is_closed_and_drained() -> None:
    rig = Rig()
    rig.bus.publish(utterance())
    rig.bus.close(drain=True)
    assert not rig.lane.finished
    rig.lane.step(0.1)
    assert rig.lane.finished


def test_blank_text_is_skipped_without_translation() -> None:
    rig = Rig()
    rig.bus.publish(utterance("   "))
    rig.lane.step(0.1)
    assert rig.translator.calls == []
    assert rig.count("skipped_empty") == 1
    assert rig.sink.events == []


def test_timings_are_recorded() -> None:
    rig = Rig()
    rig.bus.publish(utterance())
    rig.lane.step(0.1)
    snapshot = rig.metrics.snapshot()
    assert snapshot["lane.de.translate_ms.count"] == 1
    assert snapshot["lane.de.tts_ms.count"] == 1


# --- latency --------------------------------------------------------------------------------


def test_latency_is_stream_time_from_the_end_of_speech_to_the_first_audio() -> None:
    rig = Rig()
    rig.position = 16_000 + 16 * 900  # 900 ms after the end of speech
    rig.bus.publish(utterance(speech_end_sample=16_000))
    rig.lane.step(0.1)
    assert rig.latency_ms() == [900.0]


def test_the_radio_offset_is_added_to_the_reported_latency() -> None:
    rig = Rig(config=LaneConfig(radio_offset_ms=120))
    rig.position = 16_000 + 16 * 900
    rig.bus.publish(utterance(speech_end_sample=16_000))
    rig.lane.step(0.1)
    assert rig.latency_ms() == [1020.0]


# --- glossary -------------------------------------------------------------------------------


def test_glossary_terms_are_protected_in_translation_and_restored_after() -> None:
    glossary = Glossary.from_toml('[[term]]\nsource = "Contoso Cloud"\nde = "Contoso-Cloud"\n')
    rig = Rig(glossary=glossary)
    rig.bus.publish(utterance("We use Contoso Cloud daily"))
    rig.lane.step(0.1)
    sent = rig.translator.calls[0]
    assert "Contoso" not in sent
    assert "XQ" in sent
    assert rig.tts.spoken == ["[x] We use Contoso-Cloud daily"]
    assert rig.count("glossary_missing") == 0


def test_a_placeholder_the_translator_dropped_is_counted() -> None:
    glossary = Glossary.from_toml('[[term]]\nsource = "Contoso Cloud"\nde = "Contoso-Cloud"\n')

    class Dropping(FakeTranslator):
        def translate(self, text: str) -> str:
            return "ganz anderer Satz"

    rig = Rig(glossary=glossary, translator=Dropping())
    rig.bus.publish(utterance("We use Contoso Cloud daily"))
    rig.lane.step(0.1)
    assert rig.count("glossary_missing") == 1
    assert rig.tts.spoken == ["ganz anderer Satz"]


# --- failures stay inside the lane ----------------------------------------------------------


def test_a_failing_translation_is_counted_and_the_next_utterance_still_plays() -> None:
    rig = Rig()
    rig.translator.fail_on = "bad"
    rig.bus.publish(utterance("bad one", uid=1))
    rig.lane.step(0.1)
    assert rig.count("errors") == 1
    assert isinstance(rig.lane.last_error, RuntimeError)
    assert rig.sink.events == []
    rig.bus.publish(utterance("good one", uid=2))
    rig.lane.step(0.1)
    assert rig.tts.spoken == ["[x] good one"]
    assert rig.count("utterances") == 1


def test_a_failing_tts_is_counted_and_the_output_is_not_left_open() -> None:
    rig = Rig()
    rig.tts.fail = True
    rig.bus.publish(utterance())
    rig.lane.step(0.1)
    assert rig.count("errors") == 1
    assert "begin" not in rig.sink.events


def test_the_output_is_closed_even_if_playback_fails() -> None:
    rig = Rig()

    def boom(samples: Float32Array, timeout_s: float = 5.0) -> bool:
        raise OSError("device vanished")

    rig.sink.write_all = boom  # type: ignore[method-assign]
    rig.bus.publish(utterance())
    rig.lane.step(0.1)
    assert rig.sink.events == ["begin", "end"]
    assert rig.count("errors") == 1


def test_a_ring_that_stays_full_is_counted() -> None:
    rig = Rig()
    rig.sink.accept = False
    rig.bus.publish(utterance())
    rig.lane.step(0.1)
    assert rig.count("write_timeouts") == 1


def test_a_disconnected_output_skips_the_work() -> None:
    rig = Rig(sink=FakeSink(rate_hz=None))
    rig.bus.publish(utterance())
    rig.lane.step(0.1)
    assert rig.translator.calls == []
    assert rig.count("skipped_offline") == 1


# --- sample rates ---------------------------------------------------------------------------


class DoubleResampler:
    """Repeats every sample, so the output is twice as long."""

    def __init__(self, in_rate_hz: int, out_rate_hz: int) -> None:
        self.rates = (in_rate_hz, out_rate_hz)

    def process(self, block: Float32Array) -> Float32Array:
        return np.repeat(block, 2)


def test_audio_is_converted_to_the_device_rate() -> None:
    made: list[DoubleResampler] = []

    def factory(in_rate_hz: int, out_rate_hz: int) -> DoubleResampler:
        made.append(DoubleResampler(in_rate_hz, out_rate_hz))
        return made[-1]

    rig = Rig(
        tts=FakeTts(rate_hz=24_000),
        sink=FakeSink(rate_hz=48_000),
        resampler_factory=factory,
    )
    rig.bus.publish(utterance("hi"))
    rig.lane.step(0.1)
    assert made[0].rates == (24_000, 48_000)
    assert rig.sink.written == len("[x] hi") * 100 * 2


def test_the_resampler_is_rebuilt_when_the_device_rate_changes() -> None:
    made: list[DoubleResampler] = []

    def factory(in_rate_hz: int, out_rate_hz: int) -> DoubleResampler:
        made.append(DoubleResampler(in_rate_hz, out_rate_hz))
        return made[-1]

    rig = Rig(tts=FakeTts(rate_hz=24_000), sink=FakeSink(rate_hz=48_000), resampler_factory=factory)
    rig.bus.publish(utterance("one", uid=1))
    rig.lane.step(0.1)
    rig.sink.rate_hz = 44_100
    rig.sink.generation = 1
    rig.bus.publish(utterance("two", uid=2))
    rig.lane.step(0.1)
    assert [m.rates for m in made] == [(24_000, 48_000), (24_000, 44_100)]


def test_equal_rates_need_no_resampler() -> None:
    rig = Rig()
    rig.bus.publish(utterance("hi"))
    rig.lane.step(0.1)
    assert rig.count("errors") == 0


def test_different_rates_without_a_resampler_is_an_error_not_noise() -> None:
    rig = Rig(tts=FakeTts(rate_hz=22_050))
    rig.bus.publish(utterance("hi"))
    rig.lane.step(0.1)
    assert rig.count("errors") == 1
    assert "resampl" in str(rig.lane.last_error)
    assert rig.sink.events == []


# --- T1: lane independence ------------------------------------------------------------------


def _run_en_lane(blocked_de: bool) -> tuple[list[float], Rig, Rig]:
    bus = TextBus()
    metrics = Metrics()
    en = Rig("en", bus=bus, metrics=metrics)
    de = Rig("de", bus=bus, metrics=metrics)
    if blocked_de:
        de.translator.gate = threading.Event()
        de.lane.start()
    for index in range(6):
        end = 16_000 * (index + 1)
        bus.publish(utterance(f"sentence {index}", uid=index, speech_end_sample=end))
        en.position = end + 16 * 400
        assert en.lane.step(0.5)
    latencies = [metrics.histogram("lane.en.latency_ms").summary().p50]
    return latencies, en, de


def test_t1_a_blocked_lane_does_not_change_the_latency_of_another() -> None:
    baseline, *_ = _run_en_lane(blocked_de=False)
    with_blocked, en, de = _run_en_lane(blocked_de=True)
    try:
        assert de.translator.entered.wait(2.0)
        assert abs(with_blocked[0] - baseline[0]) <= 10.0
        assert en.count("utterances") == 6
        # The blocked lane is behind by more than its 2 slots: the bus drops its oldest.
        dropped = en.bus.stats()["lane-de"].dropped
        assert dropped >= 3
        assert en.bus.stats()["lane-en"].dropped == 0
    finally:
        assert de.translator.gate is not None
        de.translator.gate.set()
        de.lane.stop()


# --- the thread -----------------------------------------------------------------------------


def test_the_thread_processes_until_the_bus_is_closed() -> None:
    rig = Rig()
    rig.lane.start()
    rig.bus.publish(utterance("one", uid=1))
    rig.bus.publish(utterance("two", uid=2))
    rig.bus.close(drain=True)
    deadline = time.monotonic() + 3.0
    while rig.lane.running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not rig.lane.running
    assert rig.count("utterances") == 2


def test_a_lane_cannot_be_started_twice() -> None:
    rig = Rig()
    rig.lane.start()
    try:
        with pytest.raises(RuntimeError, match="already"):
            rig.lane.start()
    finally:
        rig.lane.stop()


def test_stop_ends_an_idle_thread() -> None:
    rig = Rig()
    rig.lane.start()
    rig.lane.stop()
    assert not rig.lane.running


# --- configuration --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [{"radio_offset_ms": -1}, {"radio_offset_ms": 5000}, {"write_timeout_s": 0.0}],
)
def test_bad_lane_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(LaneConfigError):
        LaneConfig(**kwargs)  # type: ignore[arg-type]


# --- translations for a window (P12-P14) ----------------------------------------------------


def make_translation(**changes: object) -> Translation:
    values: dict[str, object] = {
        "utterance_id": 1,
        "target": "de",
        "text": "hallo",
        "translate_ms": 12.0,
        "tts_ms": 30.0,
        "latency_ms": 900.0,
    }
    values.update(changes)
    return Translation(**values)  # type: ignore[arg-type]


def test_p12_translation_is_immutable_and_keeps_its_text_out_of_repr() -> None:
    translation = make_translation(text="geheimer Satz")
    with pytest.raises(AttributeError):
        translation.text = "other"  # type: ignore[misc]
    assert "geheimer" not in repr(translation)
    assert "de" in repr(translation)


@pytest.mark.parametrize(
    "changes",
    [
        {"utterance_id": -1},
        {"target": ""},
        {"translate_ms": -0.1},
        {"translate_ms": float("nan")},
        {"tts_ms": -1.0},
        {"tts_ms": float("inf")},
        {"latency_ms": float("nan")},
    ],
)
def test_p12_a_translation_with_impossible_values_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(TranslationError):
        make_translation(**changes)


def test_p12_unknown_timings_may_be_none() -> None:
    translation = make_translation(tts_ms=None, latency_ms=None)
    assert translation.tts_ms is None
    assert translation.latency_ms is None


def test_p12_the_lane_publishes_one_translation_per_spoken_utterance() -> None:
    translations = TranslationBus()
    ui = translations.subscribe("ui", maxsize=8)
    rig = Rig(translations=translations)
    rig.position = 16_000 + 16 * 900
    rig.bus.publish(utterance("hello there", uid=7, speech_end_sample=16_000))
    rig.lane.step(0.1)
    translation = ui.get(timeout_s=0)
    assert translation is not None
    assert translation.utterance_id == 7
    assert translation.target == "de"
    assert translation.text == "[x] hello there"
    assert translation.translate_ms >= 0.0
    assert translation.tts_ms is not None
    assert translation.tts_ms >= 0.0
    assert translation.latency_ms == 900.0
    assert ui.get(timeout_s=0) is None  # exactly one
    assert rig.count("utterances") == 1


def test_p12_the_published_text_is_the_one_with_glossary_terms_restored() -> None:
    glossary = Glossary.from_toml('[[term]]\nsource = "Contoso Cloud"\nde = "Contoso-Cloud"\n')
    translations = TranslationBus()
    ui = translations.subscribe("ui", maxsize=8)
    rig = Rig(glossary=glossary, translations=translations)
    rig.bus.publish(utterance("We use Contoso Cloud daily"))
    rig.lane.step(0.1)
    translation = ui.get(timeout_s=0)
    assert translation is not None
    assert translation.text == "[x] We use Contoso-Cloud daily"


def test_p12_latency_includes_the_radio_offset_and_is_none_without_a_stream_clock() -> None:
    translations = TranslationBus()
    ui = translations.subscribe("ui", maxsize=8)
    rig = Rig(config=LaneConfig(radio_offset_ms=120), translations=translations)
    rig.position = 16_000 + 16 * 900
    rig.bus.publish(utterance(speech_end_sample=16_000))
    rig.lane.step(0.1)
    first = ui.get(timeout_s=0)
    assert first is not None
    assert first.latency_ms == 1020.0

    clockless = LanguageLane(
        "fr",
        rig.bus.subscribe("lane-fr", maxsize=2),
        FakeTranslator(),
        FakeTts(),
        FakeSink(),
        translations=translations,
    )
    rig.bus.publish(utterance("again", uid=2))
    assert clockless.step(0.1)
    second = ui.get(timeout_s=0)
    assert second is not None
    assert second.target == "fr"
    assert second.latency_ms is None
    assert second.tts_ms is not None


def test_p12_a_lane_without_a_translation_bus_works_as_before() -> None:
    rig = Rig()
    rig.bus.publish(utterance())
    assert rig.lane.step(0.1)
    assert rig.count("utterances") == 1


def test_p13_a_translation_is_published_even_when_speech_synthesis_fails() -> None:
    translations = TranslationBus()
    ui = translations.subscribe("ui", maxsize=8)
    rig = Rig(translations=translations)
    rig.tts.fail = True
    rig.bus.publish(utterance("hello there"))
    rig.lane.step(0.1)
    assert rig.count("errors") == 1
    translation = ui.get(timeout_s=0)
    assert translation is not None
    assert translation.text == "[x] hello there"
    assert translation.tts_ms is None
    assert translation.latency_ms is None
    assert ui.get(timeout_s=0) is None


def test_p13_a_translation_is_published_once_even_when_playback_fails() -> None:
    translations = TranslationBus()
    ui = translations.subscribe("ui", maxsize=8)
    rig = Rig(translations=translations)

    def boom(samples: Float32Array, timeout_s: float = 5.0) -> bool:
        raise OSError("device vanished")

    rig.sink.write_all = boom  # type: ignore[method-assign]
    rig.bus.publish(utterance())
    rig.lane.step(0.1)
    assert ui.get(timeout_s=0) is not None
    assert ui.get(timeout_s=0) is None
    assert rig.count("errors") == 1


def test_p13_nothing_is_published_when_nothing_was_translated() -> None:
    translations = TranslationBus()
    ui = translations.subscribe("ui", maxsize=8)

    failing = Rig(translations=translations)
    failing.translator.fail_on = "bad"
    failing.bus.publish(utterance("bad one"))
    failing.lane.step(0.1)
    assert failing.count("errors") == 1

    blank = Rig(translations=translations)
    blank.bus.publish(utterance("   "))
    blank.lane.step(0.1)

    # An output that is away skips the utterance before translating it (CPU for the other
    # lanes), so there is no text to publish. This is the documented rule.
    offline = Rig(sink=FakeSink(rate_hz=None), translations=translations)
    offline.bus.publish(utterance())
    offline.lane.step(0.1)
    assert offline.count("skipped_offline") == 1
    assert offline.translator.calls == []

    assert ui.get(timeout_s=0) is None
    assert translations.published == 0


def test_p13_translation_text_never_reaches_the_metrics() -> None:
    translations = TranslationBus()
    rig = Rig(translations=translations)
    rig.bus.publish(utterance("a very private sentence"))
    rig.lane.step(0.1)
    assert "private" not in repr(rig.metrics.snapshot())


def test_p14_a_stalled_translation_subscriber_does_not_hold_back_the_lane() -> None:
    translations = TranslationBus()
    stalled = translations.subscribe("stalled", maxsize=1)  # never read
    rig = Rig(translations=translations)
    for uid in range(1, 5):
        rig.bus.publish(utterance(f"sentence {uid}", uid=uid))
        assert rig.lane.step(0.5)
    assert rig.count("utterances") == 4
    assert stalled.stats().dropped == 3
    newest = stalled.get(timeout_s=0)
    assert newest is not None
    assert newest.utterance_id == 4  # drop-oldest: the window sees the latest text


def test_p14_a_closed_translation_bus_does_not_stop_the_lane() -> None:
    translations = TranslationBus()
    translations.close()
    rig = Rig(translations=translations)
    rig.bus.publish(utterance())
    assert rig.lane.step(0.1)
    assert rig.count("utterances") == 1
    assert rig.count("errors") == 0
