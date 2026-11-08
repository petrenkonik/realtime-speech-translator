"""Endpointing evaluation helpers: stream labels, scoring and summaries (no dataset needed).

The evaluation runs on LibriSpeech with punctuation from LibriSpeech-PC (see
``scripts/eval_endpointing.py``). These tests use synthetic tones so they run in CI.
"""

from __future__ import annotations

import numpy as np
import pytest
from endpointing_eval import (
    FINAL_PAUSE_MS,
    PAUSES_MS,
    Cut,
    Stream,
    Unit,
    build_stream,
    score_stream,
    split_at_gap,
    summarize,
)

from pipeline.vad import FRAME_SAMPLES

pytestmark = pytest.mark.unit

RATE = 16_000


def tone(seconds: float, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * RATE)) * 0.001).astype(np.float32)


def blank_stream(units: tuple[Unit, ...], length: int = 16_000) -> Stream:
    return Stream(audio=np.zeros(length, dtype=np.float32), units=units)


def test_split_cuts_inside_the_longest_quiet_run() -> None:
    audio = np.concatenate([tone(1.0), quiet(0.2, seed=1), tone(1.0)])

    head, tail = split_at_gap(audio)

    assert abs(len(head) / RATE - 1.0) < 0.02  # the first word group, gap removed
    assert abs(len(tail) / RATE - 1.0) < 0.02  # the second word group
    assert np.sqrt(np.mean(head[-160:] ** 2)) > 0.05  # head ends on speech
    assert np.sqrt(np.mean(tail[:160] ** 2)) > 0.05  # tail starts on speech


def test_a_cut_at_a_phrase_end_is_a_hit_and_its_latency_is_measured() -> None:
    stream = blank_stream((Unit("end", end_sample=1000, pause_ms=450),))

    score = score_stream(stream, [Cut(boundary_sample=1100, finalized_sample=8000)])

    result = score.results[0]
    assert result.hit
    assert result.latency_ms == pytest.approx((8000 - 1000) / 16)
    assert result.offset_ms == pytest.approx((1100 - 1000) / 16)
    assert score.cuts == 1
    assert score.matched_cuts == 1


def test_a_cut_inside_a_mid_sentence_pause_is_premature() -> None:
    stream = blank_stream(
        (Unit("unfinished", end_sample=1000, pause_ms=500), Unit("end", 20_000, 1000)),
        length=40_000,
    )

    score = score_stream(stream, [Cut(boundary_sample=1050, finalized_sample=8800)])

    assert score.results[0].hit  # the cut landed on the mid-sentence pause
    assert not score.results[1].hit  # the real end got no cut
    assert score.matched_cuts == 0  # the only cut matched no phrase end


def test_a_phrase_end_without_a_cut_is_a_miss() -> None:
    stream = blank_stream((Unit("end", 1000, 300), Unit("end", 40_000, 1000)), length=60_000)

    score = score_stream(stream, [Cut(boundary_sample=40_050, finalized_sample=47_000)])
    summary = summarize([score])

    assert not score.results[0].hit
    assert summary.recall == pytest.approx(0.5)
    assert summary.precision == pytest.approx(1.0)


def test_one_cut_can_match_only_one_phrase_end() -> None:
    stream = blank_stream((Unit("end", 1000, 300), Unit("end", 1200, 300)))

    score = score_stream(stream, [Cut(boundary_sample=1100, finalized_sample=5000)])

    assert sum(result.hit for result in score.results) == 1
    assert score.matched_cuts == 1


def test_summary_reports_recall_and_cut_rate_by_pause() -> None:
    # Positions are 2.5 s apart: the tolerance is 300 ms, so neighbours must not overlap.
    stream = blank_stream(
        (
            Unit("end", 1_000, 200),  # missed: pause too short for the fixed rule
            Unit("end", 40_000, 500),  # hit
            Unit("split", 80_000, 500),  # cut inside a mid-sentence pause
        ),
        length=100_000,
    )

    score = score_stream(stream, [Cut(40_050, 52_000), Cut(80_020, 93_000)])
    summary = summarize([score])

    assert summary.recall_by_pause[200] == (0, 1)
    assert summary.recall_by_pause[500] == (1, 1)
    assert summary.cut_by_pause[500] == (1, 1)
    assert summary.cut_by_kind["split"] == (1, 1)


def test_build_stream_puts_every_label_on_speech_and_follows_the_pause_grid() -> None:
    rng = np.random.default_rng(1)
    ends = [tone(0.8)]
    mids = [tone(0.6)]

    stream = build_stream(ends, mids, rng, groups=4)

    assert len(stream.audio) % FRAME_SAMPLES == 0
    assert stream.units[-1].pause_ms == FINAL_PAUSE_MS
    for unit in stream.units[:-1]:
        assert unit.pause_ms in PAUSES_MS
    for unit in stream.units:
        assert unit.kind in {"end", "split", "unfinished"}
        assert unit.end_sample <= len(stream.audio)
        window = stream.audio[unit.end_sample - 160 : unit.end_sample]
        assert np.sqrt(np.mean(window**2)) > 0.05  # the label sits on speech
