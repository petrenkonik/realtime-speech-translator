"""U7-U10 and chunk-processing edge cases for the Segmenter state machine."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from pipeline.vad import (
    FRAME_MS,
    FRAME_SAMPLES,
    Chunk,
    Segmenter,
    SegmenterConfig,
    VadConfigError,
)

pytestmark = pytest.mark.unit

SPEECH = 0.9
SILENCE = 0.05
AMBIGUOUS = 0.42


def _frames(ms: float) -> int:
    return math.ceil(ms / FRAME_MS)


def _run(
    probs: Sequence[float],
    config: SegmenterConfig | None = None,
    amplitudes: dict[int, float] | None = None,
    flush: bool = True,
) -> tuple[list[Chunk], NDArray[np.float32], Segmenter]:
    """Feed one probability per frame; return chunks, the full stream and the segmenter."""
    rng = np.random.default_rng(0)
    segmenter = Segmenter(config or SegmenterConfig())
    amplitudes = amplitudes or {}
    stream = np.concatenate(
        [
            (rng.standard_normal(FRAME_SAMPLES) * amplitudes.get(i, 0.2)).astype(np.float32)
            for i in range(len(probs))
        ]
    )
    chunks: list[Chunk] = []
    for i, prob in enumerate(probs):
        frame = stream[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
        chunks += segmenter.process(prob, frame)
    if flush:
        tail = segmenter.flush()
        if tail is not None:
            chunks.append(tail)
    return chunks, stream, segmenter


def _utterance(speech_frames: int, pause_frames: int) -> list[float]:
    return [SPEECH] * speech_frames + [SILENCE] * pause_frames


# --- U7: the silence threshold ---------------------------------------------------


@pytest.mark.parametrize("silence_ms", [450, 525, 600])
def test_u7_a_pause_64_ms_below_the_threshold_keeps_one_chunk(silence_ms: int) -> None:
    pause = (silence_ms - 64) // FRAME_MS
    probs = _utterance(20, pause) + _utterance(20, 60)

    chunks, _, _ = _run(probs, SegmenterConfig(silence_ms=silence_ms), flush=False)

    assert len(chunks) == 1
    assert chunks[0].reason == "silence"


@pytest.mark.parametrize("silence_ms", [450, 525, 600])
def test_u7_a_pause_64_ms_above_the_threshold_splits_into_two(silence_ms: int) -> None:
    pause = _frames(silence_ms + 64)
    probs = _utterance(20, pause) + _utterance(20, 60)

    chunks, _, _ = _run(probs, SegmenterConfig(silence_ms=silence_ms), flush=False)

    assert len(chunks) == 2


@pytest.mark.parametrize("silence_ms", [450, 525, 600])
def test_u7_the_cut_falls_exactly_at_the_configured_silence(silence_ms: int) -> None:
    config = SegmenterConfig(silence_ms=silence_ms)
    needed = _frames(silence_ms)

    just_short, _, _ = _run(_utterance(20, needed - 1) + [SPEECH] * 10, config, flush=False)
    exact, _, _ = _run(_utterance(20, needed) + [SPEECH] * 10, config, flush=False)

    assert len(just_short) == 0  # still one open chunk
    assert len(exact) == 1


def test_u7_finalization_time_is_the_frame_that_completed_the_silence() -> None:
    config = SegmenterConfig(silence_ms=450)
    chunks, _, _ = _run(_utterance(20, 40), config, flush=False)

    assert chunks[0].finalized_sample == (20 + _frames(450)) * FRAME_SAMPLES


# --- U8: validation --------------------------------------------------------------


@pytest.mark.parametrize("silence_ms", [449, 601, 0, -1])
def test_u8_silence_outside_450_to_600_is_a_config_error(silence_ms: int) -> None:
    with pytest.raises(VadConfigError, match="450"):
        SegmenterConfig(silence_ms=silence_ms)


@pytest.mark.parametrize("silence_ms", [450, 500, 600])
def test_u8_silence_inside_the_range_is_accepted(silence_ms: int) -> None:
    assert SegmenterConfig(silence_ms=silence_ms).silence_ms == silence_ms


@pytest.mark.parametrize(
    "overrides",
    [
        {"exit_threshold": 0.6},  # above the speech threshold
        {"speech_threshold": 1.5},
        {"min_speech_ms": 0},
        {"pre_roll_ms": -1},
        {"tail_ms": 0},
        {"max_utterance_ms": 1000},
        {"cut_search_ms": 100},
        {"cut_search_ms": 20_000},
    ],
)
def test_other_invalid_settings_are_rejected(overrides: dict[str, float]) -> None:
    with pytest.raises(VadConfigError):
        SegmenterConfig(**overrides)  # type: ignore[arg-type]


# --- U9: forced split ------------------------------------------------------------


def test_u9_forty_seconds_of_continuous_speech_is_split_without_losing_samples() -> None:
    probs = [SPEECH] * (40 * 1000 // FRAME_MS)

    chunks, stream, _ = _run(probs)

    assert len(chunks) >= 3
    assert all(c.end_sample - c.start_sample <= 15 * 16_000 for c in chunks)
    assert all(c.reason in ("max_length", "flush") for c in chunks)
    np.testing.assert_array_equal(np.concatenate([c.samples for c in chunks]), stream)


def test_u9_chunks_are_contiguous_with_no_gap_or_overlap() -> None:
    chunks, _, _ = _run([SPEECH] * 1100)

    for earlier, later in itertools.pairwise(chunks):
        assert later.start_sample == earlier.end_sample


def test_u9_the_cut_lands_on_the_quietest_frame_of_the_search_window() -> None:
    config = SegmenterConfig(max_utterance_ms=15_000, cut_search_ms=1_500)
    quiet_frame = 440  # inside the last 1.5 s of the first 468-frame chunk

    chunks, _, _ = _run([SPEECH] * 700, config, amplitudes={quiet_frame: 0.0005})

    assert chunks[0].end_sample == (quiet_frame + 1) * FRAME_SAMPLES
    assert chunks[0].reason == "max_length"


def test_u9_a_chunk_is_never_longer_than_the_cap_even_when_the_quietest_frame_is_last() -> None:
    chunks, _, _ = _run([SPEECH] * 600, amplitudes={467: 0.0001})

    assert chunks[0].end_sample - chunks[0].start_sample <= 15 * 16_000


def test_u9_speech_after_a_forced_cut_continues_into_the_next_chunk() -> None:
    probs = [SPEECH] * 500 + [SILENCE] * 40

    chunks, stream, _ = _run(probs)

    assert len(chunks) == 2
    assert chunks[1].reason == "silence"
    # Nothing between the cut and the speech end is lost.
    assert chunks[1].start_sample == chunks[0].end_sample
    assert chunks[1].end_sample >= 500 * FRAME_SAMPLES
    np.testing.assert_array_equal(
        np.concatenate([c.samples for c in chunks]), stream[: chunks[1].end_sample]
    )


# --- U10: pre-roll ---------------------------------------------------------------


def test_u10_the_chunk_starts_100_ms_before_the_first_speech_frame() -> None:
    probs = [SILENCE] * 30 + [SPEECH] * 20 + [SILENCE] * 30

    chunks, _, _ = _run(probs)

    speech_start = 30 * FRAME_SAMPLES
    expected = speech_start - 100 * 16
    assert abs(chunks[0].start_sample - expected) <= FRAME_SAMPLES


def test_u10_pre_roll_is_limited_by_the_start_of_the_stream() -> None:
    chunks, _, _ = _run([SILENCE] + [SPEECH] * 20 + [SILENCE] * 30)

    assert chunks[0].start_sample == 0


def test_u10_pre_roll_never_overlaps_the_previous_chunk() -> None:
    config = SegmenterConfig(pre_roll_ms=800)
    probs = _utterance(20, 15) + _utterance(20, 40)  # second speech starts right after the cut

    chunks, _, _ = _run(probs, config)

    assert len(chunks) == 2
    assert chunks[1].start_sample >= chunks[0].end_sample


def test_u10_pre_roll_can_be_widened() -> None:
    config = SegmenterConfig(pre_roll_ms=300)
    chunks, _, _ = _run([SILENCE] * 50 + [SPEECH] * 20 + [SILENCE] * 30, config)

    assert abs(chunks[0].start_sample - (50 * FRAME_SAMPLES - 300 * 16)) <= FRAME_SAMPLES


# --- chunk contents --------------------------------------------------------------


def test_samples_equal_the_original_stream_slice() -> None:
    probs = [SILENCE] * 10 + [SPEECH] * 25 + [SILENCE] * 30

    chunks, stream, _ = _run(probs)

    chunk = chunks[0]
    np.testing.assert_array_equal(chunk.samples, stream[chunk.start_sample : chunk.end_sample])
    assert chunk.samples.dtype == np.float32


def test_u14_chunk_samples_are_read_only() -> None:
    chunks, _, _ = _run(_utterance(20, 30))

    with pytest.raises(ValueError, match="read-only"):
        chunks[0].samples[0] = 1.0


def test_chunk_ids_are_sequential_from_one() -> None:
    chunks, _, _ = _run((_utterance(20, 30)) * 3)

    assert [c.id for c in chunks] == [1, 2, 3]


def test_the_chunk_does_not_alias_the_callers_frame_buffer() -> None:
    segmenter = Segmenter()
    buffer = np.full(FRAME_SAMPLES, 0.25, dtype=np.float32)
    chunks: list[Chunk] = []
    for prob in [SPEECH] * 20 + [SILENCE] * 20:
        chunks += segmenter.process(prob, buffer)
        buffer[:] = 0.0  # the caller reuses its buffer

    assert chunks[0].samples.max() == pytest.approx(0.25)


def test_a_frame_of_the_wrong_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="512"):
        Segmenter().process(SPEECH, np.zeros(100, dtype=np.float32))


# --- trailing silence ------------------------------------------------------------


def test_trailing_silence_is_trimmed_but_the_speech_end_is_recorded() -> None:
    chunks, _, _ = _run(_utterance(20, 40))

    chunk = chunks[0]
    assert chunk.speech_end_sample == 20 * FRAME_SAMPLES
    tail_samples = chunk.end_sample - chunk.speech_end_sample
    assert 0 < tail_samples <= 160 * 16 + FRAME_SAMPLES


def test_a_flushed_chunk_is_also_trimmed_and_marked() -> None:
    chunks, _, _ = _run(_utterance(20, 5))

    assert chunks[0].reason == "flush"
    assert chunks[0].end_sample == 25 * FRAME_SAMPLES  # fewer silent frames than the tail


# --- min speech ------------------------------------------------------------------


def test_a_click_shorter_than_min_speech_is_dropped_and_counted() -> None:
    chunks, _, segmenter = _run(_utterance(3, 40))

    assert chunks == []
    assert segmenter.dropped_short == 1


def test_a_one_word_answer_just_above_min_speech_is_kept() -> None:
    config = SegmenterConfig(min_speech_ms=160)

    chunks, _, _ = _run(_utterance(_frames(160), 40), config)

    assert len(chunks) == 1


def test_dropped_noise_does_not_consume_a_chunk_id() -> None:
    chunks, _, _ = _run(_utterance(2, 30) + _utterance(20, 30))

    assert [c.id for c in chunks] == [1]


# --- hysteresis and stalls -------------------------------------------------------


def test_probabilities_between_the_thresholds_do_not_start_the_silence_clock() -> None:
    probs = [SPEECH] * 20 + [AMBIGUOUS] * 5 + [SPEECH] * 20 + [SILENCE] * 30

    chunks, _, _ = _run(probs)

    assert len(chunks) == 1


def test_a_short_dip_below_the_exit_threshold_does_not_split_speech() -> None:
    probs = [SPEECH] * 20 + [SILENCE] * 5 + [SPEECH] * 20 + [SILENCE] * 30

    chunks, _, _ = _run(probs)

    assert len(chunks) == 1


def test_noise_hovering_between_the_thresholds_cannot_hold_a_chunk_open_forever() -> None:
    config = SegmenterConfig(silence_ms=450)
    probs = [SPEECH] * 20 + [AMBIGUOUS] * 200

    chunks, _, _ = _run(probs, config, flush=False)

    assert len(chunks) == 1
    assert chunks[0].reason == "silence"
    assert chunks[0].finalized_sample <= (20 + 3 * _frames(450)) * FRAME_SAMPLES


def test_silence_that_began_before_an_ambiguous_stretch_keeps_counting() -> None:
    probs = [SPEECH] * 20 + [SILENCE] * 5 + [AMBIGUOUS] * 20

    chunks, _, _ = _run(probs, flush=False)

    assert len(chunks) == 1


# --- flush -----------------------------------------------------------------------


def test_flush_with_nothing_open_returns_none() -> None:
    assert Segmenter().flush() is None


def test_flush_discards_a_too_short_pending_chunk() -> None:
    chunks, _, segmenter = _run([SPEECH] * 3)

    assert chunks == []
    assert segmenter.dropped_short == 1


def test_after_a_flush_pre_roll_does_not_reach_back_over_the_gap() -> None:
    segmenter = Segmenter()
    rng = np.random.default_rng(1)

    def frame() -> NDArray[np.float32]:
        return rng.standard_normal(FRAME_SAMPLES).astype(np.float32)

    for _ in range(10):
        segmenter.process(SILENCE, frame())
    segmenter.flush()
    chunks: list[Chunk] = []
    for prob in [SPEECH] * 20 + [SILENCE] * 30:
        chunks += segmenter.process(prob, frame())

    assert chunks[0].start_sample == 10 * FRAME_SAMPLES  # no pre-roll from before the flush


# --- invariants over random input ------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_random_probabilities_never_lose_speech_frames_or_overlap(seed: int) -> None:
    rng = np.random.default_rng(seed)
    probs: list[float] = []
    for _ in range(60):
        level = SPEECH if rng.random() < 0.5 else SILENCE
        probs += [level] * int(rng.integers(1, 40))

    chunks, stream, _ = _run(probs)

    previous_end = 0
    covered = np.zeros(len(probs), dtype=bool)
    for chunk in chunks:
        assert chunk.start_sample >= previous_end
        assert chunk.start_sample % FRAME_SAMPLES == 0
        previous_end = chunk.end_sample
        np.testing.assert_array_equal(chunk.samples, stream[chunk.start_sample : chunk.end_sample])
        covered[chunk.start_sample // FRAME_SAMPLES : chunk.end_sample // FRAME_SAMPLES] = True
    # Every speech frame that belongs to a run long enough to keep is inside some chunk.
    run_start = 0
    for i in range(1, len(probs) + 1):
        if i == len(probs) or probs[i] != probs[run_start]:
            if probs[run_start] == SPEECH and i - run_start >= 6:
                assert covered[run_start:i].all()
            run_start = i


def test_the_short_remainder_after_a_forced_cut_is_kept_not_dropped_as_noise() -> None:
    # The cut falls 3 frames before the end of speech: the remainder is far below min speech.
    probs = [SPEECH] * 471 + [SILENCE] * 40

    chunks, _, _ = _run(probs, amplitudes={467: 0.0001})

    assert len(chunks) == 2
    assert chunks[1].end_sample > chunks[0].end_sample
    assert chunks[1].speech_end_sample == 471 * FRAME_SAMPLES


def test_config_property_returns_the_active_settings() -> None:
    config = SegmenterConfig(silence_ms=500)

    assert Segmenter(config).config is config


def test_a_pre_roll_of_zero_starts_the_chunk_at_the_first_speech_frame() -> None:
    chunks, _, _ = _run(
        [SILENCE] * 10 + [SPEECH] * 20 + [SILENCE] * 30, SegmenterConfig(pre_roll_ms=0)
    )

    assert chunks[0].start_sample == 10 * FRAME_SAMPLES
