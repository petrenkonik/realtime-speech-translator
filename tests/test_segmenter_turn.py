"""U15-U22, U29: Smart Turn rules of the Segmenter, with a scripted classifier.

No model and no clock: the test plays the part of the model link. Every answer is
handed to the segmenter a fixed number of frames after its request, so a run is exact.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from pipeline.vad import (
    FRAME_MS,
    FRAME_SAMPLES,
    Chunk,
    Segmenter,
    SegmenterConfig,
    TurnAnswer,
    TurnRequest,
    VadConfigError,
)

pytestmark = pytest.mark.unit

SPEECH = 0.9
SILENCE = 0.05
Answers = Callable[[TurnRequest], float | None]


def frames(ms: float) -> int:
    return math.ceil(ms / FRAME_MS)


def smart(**changes: object) -> SegmenterConfig:
    return SegmenterConfig(turn_mode="smart", **changes)  # type: ignore[arg-type]


def constant(probability: float) -> Answers:
    return lambda _request: probability


def scripted(*values: float | None) -> Answers:
    """Answers by request order; ``None`` = the model never answers that one."""
    queue = list(values)
    return lambda _request: queue.pop(0) if queue else None


class Run:
    """Everything a test wants to look at after a scripted run."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.requests: list[tuple[int, TurnRequest]] = []  # (frame index, request)
        self.answers: list[tuple[int, TurnAnswer]] = []  # (frame index it was applied, answer)
        self.stream = np.zeros(0, dtype=np.float32)
        self.segmenter = Segmenter()


def play(
    probs: Sequence[float],
    answers: Answers | None = None,
    config: SegmenterConfig | None = None,
    delay: int = 1,
    amplitudes: dict[int, float] | None = None,
    extra: dict[int, TurnAnswer] | None = None,
    flush: bool = False,
) -> Run:
    """Feed one probability per frame; answers arrive ``delay`` frames after the request."""
    run = Run()
    run.segmenter = Segmenter(config or smart())
    rng = np.random.default_rng(0)
    amplitudes = amplitudes or {}
    run.stream = np.concatenate(
        [
            (rng.standard_normal(FRAME_SAMPLES) * amplitudes.get(i, 0.2)).astype(np.float32)
            for i in range(len(probs))
        ]
    )
    due: dict[int, TurnAnswer] = dict(extra or {})
    for i, prob in enumerate(probs):
        answer = due.pop(i, None)
        if answer is not None:
            run.answers.append((i, answer))
        frame = run.stream[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
        run.chunks += run.segmenter.process(prob, frame, answer)
        request = run.segmenter.take_request()
        if request is not None:
            run.requests.append((i, request))
            value = answers(request) if answers is not None else None
            if value is not None:
                due[i + delay] = TurnAnswer(request.id, value)
    if flush:
        tail = run.segmenter.flush()
        if tail is not None:
            run.chunks.append(tail)
    return run


def utterance(speech_frames: int, pause_frames: int) -> list[float]:
    return [SPEECH] * speech_frames + [SILENCE] * pause_frames


# --- U15: early end ----------------------------------------------------------------


def test_u15_a_complete_phrase_ends_one_frame_after_the_answer() -> None:
    check = smart().check_frames  # 200 ms -> 7 frames
    run = play(utterance(20, 40), constant(0.9))

    assert [c.reason for c in run.chunks] == ["turn"]
    request_frame = 20 + check - 1
    assert run.requests[0][0] == request_frame
    assert run.chunks[0].finalized_sample == (request_frame + 1 + 1) * FRAME_SAMPLES


def test_u15_fixed_mode_ends_at_the_silence_threshold() -> None:
    run = play(utterance(20, 40), constant(0.9), config=SegmenterConfig())
    assert [c.reason for c in run.chunks] == ["silence"]
    assert run.chunks[0].finalized_sample == (20 + frames(450)) * FRAME_SAMPLES
    assert run.requests == []  # fixed mode never asks the model


def test_u15_smart_mode_is_earlier_than_fixed_by_the_saved_frames() -> None:
    early = play(utterance(20, 40), constant(0.9)).chunks[0]
    fixed = play(utterance(20, 40), None, config=SegmenterConfig()).chunks[0]
    assert fixed.finalized_sample - early.finalized_sample >= 6 * FRAME_SAMPLES  # >= 190 ms


def test_the_chunk_keeps_the_audio_and_the_same_tail_as_fixed_mode() -> None:
    early = play(utterance(20, 40), constant(0.9)).chunks[0]
    fixed = play(utterance(20, 40), None, config=SegmenterConfig()).chunks[0]
    np.testing.assert_array_equal(early.samples, fixed.samples)
    assert early.speech_end_sample == fixed.speech_end_sample


def test_a_probability_below_the_threshold_does_not_end_the_chunk() -> None:
    run = play(utterance(20, 40), constant(0.49))
    assert [c.reason for c in run.chunks] == ["silence"]


def test_the_threshold_is_inclusive() -> None:
    run = play(utterance(20, 40), constant(0.5))
    assert [c.reason for c in run.chunks] == ["turn"]


def test_no_check_is_made_before_enough_speech_was_heard() -> None:
    run = play(utterance(3, 40), constant(0.9))  # 96 ms of speech < min_speech
    assert run.requests == []
    assert run.chunks == []  # dropped as noise by the usual rule


# --- U16: the fallback -------------------------------------------------------------


def random_script(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    probs: list[float] = []
    for _ in range(int(rng.integers(2, 6))):
        probs += [SPEECH] * int(rng.integers(3, 40))
        probs += [float(rng.choice([0.4, 0.42]))] * int(rng.integers(0, 3))
        probs += [SILENCE] * int(rng.integers(2, 45))
    return probs


def same_chunks(a: Sequence[Chunk], b: Sequence[Chunk]) -> bool:
    return len(a) == len(b) and all(
        (x.start_sample, x.end_sample, x.speech_end_sample, x.finalized_sample, x.reason)
        == (y.start_sample, y.end_sample, y.speech_end_sample, y.finalized_sample, y.reason)
        and np.array_equal(x.samples, y.samples)
        for x, y in zip(a, b, strict=True)
    )


@pytest.mark.parametrize("silence_ms", [450, 525, 600])
@pytest.mark.parametrize(
    "answers", [constant(0.1), scripted(), None], ids=["low", "silent", "none"]
)
def test_u16_a_model_that_never_says_complete_gives_the_fixed_mode_chunks(
    silence_ms: int, answers: Answers | None
) -> None:
    for seed in range(25):
        probs = random_script(seed)
        smart_run = play(probs, answers, config=smart(silence_ms=silence_ms), flush=True)
        fixed_run = play(probs, None, config=SegmenterConfig(silence_ms=silence_ms), flush=True)
        assert same_chunks(smart_run.chunks, fixed_run.chunks), f"seed {seed}"


# --- U17: stale answers ------------------------------------------------------------


def test_u17_an_answer_that_arrives_after_speech_resumed_is_ignored() -> None:
    check = smart().check_frames
    probs = utterance(20, check) + utterance(10, 40)  # speech resumes right after the request
    run = play(probs, scripted(0.99, 0.1, 0.1, 0.1), delay=3)

    assert run.segmenter.turn_late >= 1  # the first answer; a later recheck may be late too
    assert [c.reason for c in run.chunks] == ["silence"]  # one chunk, no premature cut
    assert run.chunks[0].end_sample > (20 + check + 10) * FRAME_SAMPLES


def test_u17_an_answer_for_another_request_id_is_ignored() -> None:
    check = smart().check_frames
    wrong = {20 + check + 2: TurnAnswer(id=999, probability=0.99)}
    run = play(utterance(20, 40), constant(0.1), extra=wrong)
    assert run.segmenter.turn_late == 1
    assert [c.reason for c in run.chunks] == ["silence"]


def test_u17_an_answer_after_the_chunk_ended_counts_as_late() -> None:
    run = play(utterance(20, 40), scripted(None), extra={20 + 30: TurnAnswer(1, 0.99)})
    assert run.segmenter.turn_late == 1


def test_a_low_answer_is_not_late() -> None:
    run = play(utterance(20, 40), constant(0.1))
    assert run.segmenter.turn_late == 0
    assert run.segmenter.turn_checks >= 1


# --- U29: rechecks -----------------------------------------------------------------


def test_u29_a_phrase_rated_low_at_first_is_caught_by_a_recheck() -> None:
    check = smart().check_frames
    run = play(utterance(20, 40), scripted(0.2, 0.4, 0.8))

    assert [c.reason for c in run.chunks] == ["turn"]
    runs = [index - 20 + 1 for index, _ in run.requests]
    assert runs == [check, check + 3, check + 6]  # 96 ms = 3 frames between checks
    last_answer_frame = run.answers[-1][0]
    assert run.chunks[0].finalized_sample == (last_answer_frame + 1) * FRAME_SAMPLES


def test_u29_never_two_requests_in_flight() -> None:
    run = play(utterance(20, 40), constant(0.1), delay=5, config=smart(turn_recheck_ms=64))
    posted = [index for index, _ in run.requests]
    answered = [index for index, _ in run.answers]
    for later, earlier_answer in zip(posted[1:], answered, strict=False):
        assert later >= earlier_answer  # the next request waits for the previous answer


def test_u29_at_most_four_requests_per_silence_run() -> None:
    config = smart(silence_ms=600, turn_recheck_ms=64)
    run = play(utterance(20, 60), constant(0.1), config=config)
    assert len(run.requests) == 4
    assert [c.reason for c in run.chunks] == ["silence"]


def test_u29_a_model_that_never_answers_blocks_further_requests() -> None:
    run = play(utterance(20, 60), scripted(None, None, None))
    assert len(run.requests) == 1
    assert [c.reason for c in run.chunks] == ["silence"]  # the fallback still works


def test_u29_the_request_budget_starts_over_after_speech_resumes() -> None:
    probs = utterance(20, 12) + utterance(10, 12) + utterance(10, 40)
    run = play(probs, constant(0.1), config=smart(turn_recheck_ms=64))
    first_run = [r for r in run.requests if r[0] < 20 + 12 + 10]
    assert len(first_run) >= 2  # the second silence run was allowed its own checks


# --- U19: rescue -------------------------------------------------------------------


def test_u19_short_speech_right_after_an_early_end_is_kept_as_rescued() -> None:
    check = smart().check_frames
    probs = utterance(20, check + 1) + utterance(4, 40)  # 128 ms < min speech 160 ms
    run = play(probs, constant(0.9))

    assert [c.reason for c in run.chunks] == ["turn", "rescued"]
    assert run.segmenter.turn_rescued == 1
    assert run.segmenter.dropped_short == 0


def test_u19_the_same_short_speech_is_dropped_when_fixed_mode_would_have_split_it_too() -> None:
    probs = utterance(20, frames(450) + 2) + utterance(4, 40)
    run = play(probs, constant(0.9))
    assert [c.reason for c in run.chunks] == ["turn"]
    assert run.segmenter.dropped_short == 1


def test_u19_fixed_mode_keeps_the_short_speech_inside_the_first_chunk() -> None:
    check = smart().check_frames
    probs = utterance(20, check + 1) + utterance(4, 40)
    fixed = play(probs, None, config=SegmenterConfig())
    assert len(fixed.chunks) == 1
    smart_run = play(probs, constant(0.9))
    smart_speech = {
        s for c in smart_run.chunks for s in range(c.start_sample, c.end_sample, FRAME_SAMPLES)
    }
    for index in range(len(probs)):
        if probs[index] == SPEECH:
            assert index * FRAME_SAMPLES in smart_speech


# --- U18: nothing is lost ----------------------------------------------------------


def covered_speech(chunks: Sequence[Chunk], probs: Sequence[float]) -> set[int]:
    """Indices of speech frames inside any chunk."""
    inside: set[int] = set()
    for chunk in chunks:
        first, last = chunk.start_sample // FRAME_SAMPLES, chunk.end_sample // FRAME_SAMPLES
        inside.update(i for i in range(first, last) if probs[i] >= 0.5)
    return inside


@pytest.mark.parametrize("seed", range(60))
def test_u18_no_speech_frame_that_fixed_mode_keeps_is_lost_in_smart_mode(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    probs = random_script(seed)

    def random_answers(_request: TurnRequest) -> float | None:
        return float(rng.choice([0.1, 0.4, 0.55, 0.9, 0.99])) if rng.random() > 0.15 else None

    delay = int(rng.integers(1, 5))
    smart_run = play(probs, random_answers, delay=delay, flush=True)
    fixed_run = play(probs, None, config=SegmenterConfig(), flush=True)

    assert covered_speech(fixed_run.chunks, probs) <= covered_speech(smart_run.chunks, probs)
    previous_end = 0
    for chunk in smart_run.chunks:
        assert chunk.start_sample >= previous_end  # ordered, no overlap
        assert chunk.start_sample < chunk.end_sample
        expected = smart_run.stream[chunk.start_sample : chunk.end_sample]
        np.testing.assert_array_equal(chunk.samples, expected)  # no sample altered or shifted
        previous_end = chunk.end_sample
    ids = [c.id for c in smart_run.chunks]
    assert ids == sorted(ids) == list(range(ids[0], ids[0] + len(ids))) if ids else True


# --- U21: soft cuts ----------------------------------------------------------------


def long_speech(pause_frames: int = 6) -> list[float]:
    return (utterance(125, pause_frames)) * 5 + [SPEECH] * 40 + [SILENCE] * 40  # ~20 s


def test_u21_long_speech_is_cut_at_the_first_pause_after_six_seconds() -> None:
    run = play(long_speech(), constant(0.9), flush=True)

    reasons = [c.reason for c in run.chunks]
    assert reasons[0] == "soft"
    # The first pause (4 s) is too early; the cut is at the second one (8 s).
    first_end = run.chunks[0].speech_end_sample
    assert 8.0 <= first_end / 16_000 <= 8.3
    assert all(len(c.samples) <= 15_000 * 16 for c in run.chunks)
    assert run.segmenter.turn_soft_cuts >= 1


def test_u21_a_soft_check_must_reach_the_stricter_threshold() -> None:
    run = play(long_speech(), constant(0.5), flush=True)  # 0.5 passes turn, fails soft (0.7)

    reasons = [c.reason for c in run.chunks]
    assert "soft" not in reasons
    assert "max_length" in reasons  # only the hard cap cut the speech
    assert all(len(c.samples) <= 15_000 * 16 for c in run.chunks)


def test_u21_short_pauses_in_normal_speech_are_not_checked() -> None:
    probs = utterance(60, 4) * 2 + utterance(10, 60)  # 128 ms pauses inside a 4 s chunk
    run = play(probs, constant(0.99))
    assert [c.reason for c in run.chunks] == ["turn"]  # one chunk, ended only at the real end
    assert all(index >= 2 * 64 + 10 for index, _ in run.requests)  # no check in the short pauses


def test_u21_soft_cuts_can_be_switched_off() -> None:
    run = play(long_speech(), constant(0.99), config=smart(soft_cut_ms=0), flush=True)
    assert "soft" not in [c.reason for c in run.chunks]


# --- U22: the hard cap -------------------------------------------------------------


def test_u22_the_hard_cap_cuts_at_the_quietest_frame_and_loses_nothing() -> None:
    total = 40_000 // FRAME_MS
    quiet = 440  # inside the last 1.5 s of the first 15 s
    run = play(
        [SPEECH] * total,
        constant(0.99),
        amplitudes={quiet: 0.001},
        flush=True,
    )
    first = run.chunks[0]
    assert first.reason == "max_length"
    assert first.end_sample == (quiet + 1) * FRAME_SAMPLES
    joined = np.concatenate([c.samples for c in run.chunks])
    np.testing.assert_array_equal(joined, run.stream[: len(joined)])
    assert run.segmenter.turn_hard_cuts >= 2


# --- U20: configuration ------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("turn_mode", "auto"),
        ("turn_check_ms", 99),
        ("turn_check_ms", 501),
        ("turn_check_ms", 225),
        ("turn_recheck_ms", 63),
        ("turn_recheck_ms", 321),
        ("turn_threshold", 0.29),
        ("turn_threshold", 0.91),
        ("turn_threshold", 0.52),
        ("soft_cut_ms", 2_999),
        ("soft_cut_ms", 12_001),
        ("soft_pause_ms", 95),
        ("soft_pause_ms", 193),
        ("soft_cut_threshold", 0.49),
        ("soft_cut_threshold", 0.96),
    ],
)
def test_u20_values_just_outside_the_ranges_are_rejected(field: str, bad: object) -> None:
    with pytest.raises(VadConfigError, match=field):
        SegmenterConfig(**{field: bad})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "good"),
    [
        ("turn_mode", "smart"),
        ("turn_check_ms", 100),
        ("turn_check_ms", 500),
        ("turn_recheck_ms", 64),
        ("turn_recheck_ms", 320),
        ("turn_threshold", 0.3),
        ("turn_threshold", 0.9),
        ("soft_cut_ms", 0),
        ("soft_cut_ms", 3_000),
        ("soft_cut_ms", 12_000),
        ("soft_pause_ms", 96),
        ("soft_pause_ms", 192),
        ("soft_cut_threshold", 0.7),
        ("soft_cut_threshold", 0.95),
    ],
)
def test_u20_values_on_the_boundaries_are_accepted(field: str, good: object) -> None:
    # A high normal threshold needs an even higher soft one (the soft check is stricter).
    extra = {"soft_cut_threshold": 0.95} if field == "turn_threshold" else {}
    assert getattr(SegmenterConfig(**{field: good, **extra}), field) == good  # type: ignore[arg-type]


def test_u20_the_check_time_is_clamped_to_leave_room_for_the_model() -> None:
    assert SegmenterConfig(silence_ms=450, turn_check_ms=300).effective_check_ms == 250
    assert SegmenterConfig(silence_ms=450, turn_check_ms=200).effective_check_ms == 200
    assert SegmenterConfig(silence_ms=600, turn_check_ms=500).effective_check_ms == 400
    assert SegmenterConfig(silence_ms=450, turn_check_ms=300).check_frames == frames(250)


def test_u20_a_soft_threshold_below_the_normal_one_is_rejected() -> None:
    with pytest.raises(VadConfigError, match="soft_cut_threshold"):
        SegmenterConfig(turn_threshold=0.8, soft_cut_threshold=0.7)
    assert SegmenterConfig(turn_threshold=0.7, soft_cut_threshold=0.7)


def test_the_default_mode_of_the_segmenter_is_fixed() -> None:
    # The CLI defaults to smart (D2); the state machine itself stays literal.
    assert SegmenterConfig().turn_mode == "fixed"


def test_requests_carry_the_last_eight_seconds_as_a_read_only_window() -> None:
    probs = [SPEECH] * 400 + [SILENCE] * 40  # 12.8 s of speech, then a pause
    run = play(probs, constant(0.1), config=smart(soft_cut_ms=0), flush=True)
    window = run.requests[0][1].window
    assert len(window) == 250 * FRAME_SAMPLES  # 8 s - 0.2 s... capped at 250 frames
    assert window.flags.writeable is False
    end = run.requests[0][0] + 1
    np.testing.assert_array_equal(
        window, run.stream[(end - 250) * FRAME_SAMPLES : end * FRAME_SAMPLES]
    )


def test_a_flush_forgets_the_pending_request() -> None:
    check = smart().check_frames
    run = play(utterance(20, check), None, flush=True)
    assert run.segmenter.take_request() is None
