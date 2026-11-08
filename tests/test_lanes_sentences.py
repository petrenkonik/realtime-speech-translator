"""A lane speaks a translation one sentence at a time: the first plays while the next is made."""

from __future__ import annotations

import numpy as np
import pytest
from test_lanes import FakeSink, FakeTts, Rig, utterance

from pipeline.lanes import LaneConfig, Speech
from pipeline.vad import Float32Array

pytestmark = pytest.mark.unit

TWO = "First sentence. Second sentence."


class LoggingTts(FakeTts):
    """Writes into the sink's event list so the order of synthesis and playback is visible."""

    def __init__(self, sink: FakeSink, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self.sink = sink
        self.on_call: dict[int, object] = {}

    def synthesize(self, text: str) -> Speech:
        self.sink.events.append(f"synth:{text}")
        return super().synthesize(text)


def sentence_rig(**kwargs: object) -> Rig:
    sink = FakeSink()
    rig = Rig(sink=sink, tts=LoggingTts(sink), **kwargs)  # type: ignore[arg-type]
    rig.translator.prefix = ""
    return rig


def test_each_sentence_is_synthesized_on_its_own_in_order() -> None:
    rig = sentence_rig()
    rig.bus.publish(utterance(TWO))
    rig.lane.step(0.1)
    assert rig.tts.spoken == ["First sentence.", "Second sentence."]


def test_the_first_sentence_is_written_before_the_second_is_synthesized() -> None:
    rig = sentence_rig()
    rig.bus.publish(utterance(TWO))
    rig.lane.step(0.1)
    first, second = len("First sentence.") * 100, len("Second sentence.") * 100
    assert rig.sink.events == [
        "synth:First sentence.",
        "begin",
        f"write:{first}",
        "synth:Second sentence.",
        f"write:{second}",
        "end",
    ]


def test_one_utterance_is_one_begin_and_one_end() -> None:
    rig = sentence_rig()
    rig.bus.publish(utterance("One. Two. Three."))
    rig.lane.step(0.1)
    assert rig.sink.events.count("begin") == 1
    assert rig.sink.events.count("end") == 1
    assert rig.count("utterances") == 1


def test_a_text_without_a_sentence_end_is_spoken_whole() -> None:
    rig = sentence_rig()
    rig.bus.publish(utterance("no end here"))
    rig.lane.step(0.1)
    assert rig.tts.spoken == ["no end here"]


def test_latency_is_taken_at_the_first_audio_not_after_the_last_sentence() -> None:
    rig = sentence_rig()
    rig.position = 16_000 + 16 * 400
    rig.bus.publish(utterance(TWO, speech_end_sample=16_000))

    original = rig.tts.synthesize

    def slow_second(text: str) -> Speech:
        if text.startswith("Second"):
            rig.position += 16 * 5_000  # a long synthesis of the second sentence
        return original(text)

    rig.tts.synthesize = slow_second  # type: ignore[method-assign]
    rig.lane.step(0.1)
    assert rig.latency_ms() == [400.0]


def test_the_synthesis_time_is_the_time_to_the_first_audio_once_per_utterance() -> None:
    rig = sentence_rig()
    rig.bus.publish(utterance("One. Two. Three."))
    rig.lane.step(0.1)
    assert rig.metrics.snapshot()["lane.de.tts_ms.count"] == 1


def test_a_failing_second_sentence_ends_the_utterance_and_counts_an_error() -> None:
    rig = sentence_rig()
    original = rig.tts.synthesize

    def fail_on_second(text: str) -> Speech:
        if text.startswith("Second"):
            raise RuntimeError("tts broke")
        return original(text)

    rig.tts.synthesize = fail_on_second  # type: ignore[method-assign]
    rig.bus.publish(utterance(TWO))
    rig.lane.step(0.1)
    assert rig.sink.events[-1] == "end"  # the device is not left in a speaking state
    assert rig.count("errors") == 1
    assert rig.count("utterances") == 0


def test_a_failing_first_sentence_opens_nothing() -> None:
    rig = sentence_rig()
    rig.tts.fail = True
    rig.bus.publish(utterance(TWO))
    rig.lane.step(0.1)
    assert "begin" not in rig.sink.events
    assert rig.count("errors") == 1


def test_a_write_timeout_stops_the_rest_of_the_utterance() -> None:
    rig = sentence_rig()
    rig.sink.accept = False
    rig.bus.publish(utterance("One. Two. Three."))
    rig.lane.step(0.1)
    assert rig.tts.spoken == ["One."]  # late speech is dropped, not queued behind a stuck device
    assert rig.count("write_timeouts") == 1
    assert rig.sink.events[-1] == "end"


def test_silence_for_every_sentence_counts_as_skipped_and_opens_nothing() -> None:
    rig = sentence_rig()
    rig.tts.samples_per_char = 0
    rig.bus.publish(utterance(TWO))
    rig.lane.step(0.1)
    assert rig.count("skipped_empty") == 1
    assert "begin" not in rig.sink.events


def test_an_empty_sentence_in_the_middle_is_passed_over() -> None:
    rig = sentence_rig()
    original = rig.tts.synthesize

    def silent_second(text: str) -> Speech:
        speech = original(text)
        if text.startswith("Second"):
            return Speech(samples=np.zeros(0, dtype=np.float32), rate_hz=speech.rate_hz)
        return speech

    rig.tts.synthesize = silent_second  # type: ignore[method-assign]
    rig.bus.publish(utterance("First one. Second one. Third one."))
    rig.lane.step(0.1)
    assert rig.count("utterances") == 1
    assert sum(1 for e in rig.sink.events if e.startswith("write:")) == 2


# --- resampling across sentences ------------------------------------------------------------


class CountingResampler:
    """Doubles the rate by repeating samples; remembers every block it saw."""

    def __init__(self) -> None:
        self.blocks: list[int] = []

    def process(self, block: Float32Array) -> Float32Array:
        self.blocks.append(len(block))
        return np.repeat(block, 2)


def test_one_resampler_serves_all_sentences_of_an_utterance() -> None:
    made: list[CountingResampler] = []

    def factory(in_hz: int, out_hz: int) -> CountingResampler:
        made.append(CountingResampler())
        return made[-1]

    sink = FakeSink(rate_hz=32_000)
    rig = Rig(
        sink=sink,
        tts=LoggingTts(sink),
        resampler_factory=factory,
        config=LaneConfig(),
    )
    rig.translator.prefix = ""
    rig.bus.publish(utterance(TWO))
    rig.lane.step(0.1)
    # The filter history must run through the sentence boundary, or each one starts with a click.
    assert len(made) == 1
    assert made[0].blocks == [len("First sentence.") * 100, len("Second sentence.") * 100]
    assert sink.written == 2 * (len("First sentence.") + len("Second sentence.")) * 100


def test_a_new_utterance_gets_a_fresh_resampler() -> None:
    made: list[CountingResampler] = []

    def factory(in_hz: int, out_hz: int) -> CountingResampler:
        made.append(CountingResampler())
        return made[-1]

    sink = FakeSink(rate_hz=32_000)
    rig = Rig(sink=sink, resampler_factory=factory)
    rig.bus.publish(utterance("One.", uid=1))
    rig.bus.publish(utterance("Two.", uid=2))
    rig.lane.step(0.1)
    rig.lane.step(0.1)
    assert len(made) == 2
