"""Pause extraction behind scripts/pause_viewer.py: thresholds, energy runs, cut agreement."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from pipeline.vad import FRAME_SAMPLES

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent
SPEECH, EXIT = 0.5, 0.35


def load_viewer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pause_viewer", ROOT / "scripts/pause_viewer.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pause_viewer"] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


viewer = load_viewer()


def pauses(probs: list[float]) -> list[tuple[int, int, bool]]:
    found = viewer.find_pauses(probs, SPEECH, EXIT)
    return [(p.start_frame, p.end_frame, p.open_end) for p in found]


def test_single_gap_between_speech() -> None:
    assert pauses([0.9, 0.9, 0.1, 0.1, 0.1, 0.9]) == [(2, 5, False)]


def test_dip_between_thresholds_is_not_a_pause() -> None:
    assert pauses([0.9, 0.4, 0.9]) == []


def test_rise_between_thresholds_does_not_end_a_pause() -> None:
    assert pauses([0.9, 0.1, 0.4, 0.1, 0.9]) == [(1, 4, False)]


def test_leading_silence_is_not_a_pause() -> None:
    assert pauses([0.1, 0.1, 0.9]) == []


def test_trailing_silence_is_an_open_pause() -> None:
    assert pauses([0.9, 0.1, 0.1]) == [(1, 3, True)]


def test_pause_length_in_ms() -> None:
    (pause,) = viewer.find_pauses([0.9] + [0.1] * 14 + [0.9], SPEECH, EXIT)
    assert pause.length_ms == 14 * 32


def test_cut_falls_inside_the_long_pause_only() -> None:
    probs = [0.9] * 8 + [0.1] * 10 + [0.9] * 8 + [0.1] * 20 + [0.9] * 8 + [0.1] * 20
    cuts = viewer.segmenter_cuts(probs, silence_ms=450)
    found = viewer.find_pauses(probs, SPEECH, EXIT)
    long_pauses = [p for p in found if p.length_ms >= 450]
    assert len(cuts) == len(long_pauses) == 2
    for cut_frame, pause in zip(cuts, long_pauses, strict=True):
        assert pause.start_frame <= cut_frame <= pause.end_frame


def test_cut_before_the_hysteresis_band_still_belongs_to_the_pause() -> None:
    # The last frame at or above 0.5 is frame 7; frame 8 (0.45) is in the hold band, and the
    # pause starts at frame 9 (below 0.35). The segmenter reports the cut at frame 8.
    probs = [0.9] * 8 + [0.45] + [0.1] * 20 + [0.9] * 8 + [0.1] * 20
    cuts = viewer.segmenter_cuts(probs, silence_ms=450)
    found = viewer.find_pauses(probs, SPEECH, EXIT)
    assert cuts[0] == 8
    assert found[0].start_frame == 9
    assert viewer.attach_cuts(found, cuts, probs, SPEECH) == {0: 8, 1: cuts[1]}


def test_cut_after_a_loud_frame_is_not_attached_to_a_later_pause() -> None:
    pauses = [viewer.Pause(start_frame=20, end_frame=40, open_end=False)]
    probs = [0.9] * 40
    assert viewer.attach_cuts(pauses, [10], probs, SPEECH) == {}


def test_energy_runs_find_the_quiet_gap() -> None:
    loud = (0.3 * np.sin(np.arange(8000) * 0.3)).astype(np.float32)
    quiet = np.zeros(4000, dtype=np.float32)
    runs = viewer.energy_quiet_runs(np.concatenate([quiet, loud, quiet, loud, quiet]))
    assert runs == [(12000, 16000)]


def test_energy_runs_ignore_gaps_shorter_than_the_minimum() -> None:
    loud = (0.3 * np.sin(np.arange(8000) * 0.3)).astype(np.float32)
    short_gap = np.zeros(320, dtype=np.float32)  # 20 ms
    assert viewer.energy_quiet_runs(np.concatenate([loud, short_gap, loud])) == []


def test_split_text_by_fraction_keeps_all_words() -> None:
    head, tail = viewer.split_text("one two three four", 0.5)
    assert (head, tail) == ("one two", "three four")


def test_split_text_never_returns_an_empty_half() -> None:
    head, tail = viewer.split_text("one two", 0.0)
    assert head == "one"
    assert tail == "two"


def test_frame_constant_matches_the_model_frame() -> None:
    assert viewer.FRAME_SAMPLES == FRAME_SAMPLES


class Constant:
    """A Smart Turn classifier that always answers the same probability."""

    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.windows = 0

    def predict(self, window: np.ndarray) -> float:
        self.windows += 1
        return self.probability


def two_pauses() -> tuple[list[float], np.ndarray]:
    probs = [0.9] * 8 + [0.05] * 20 + [0.9] * 8 + [0.05] * 20
    return probs, np.zeros(len(probs) * FRAME_SAMPLES, dtype=np.float32)


def test_smart_turn_cuts_before_the_silence_timer_when_it_is_sure() -> None:
    probs, audio = two_pauses()
    trace = viewer.smart_turn_trace(probs, audio, Constant(0.9), silence_ms=450)
    fixed_decided = 8 + 15  # pause start + silence_ms in frames
    turn_cuts = [c for c in trace["cuts"] if c["reason"] == "turn"]
    assert turn_cuts
    assert turn_cuts[0]["decided"] < fixed_decided
    assert all(probe["p"] == 0.9 for probe in trace["probes"])


def test_smart_turn_falls_back_to_the_silence_timer_when_it_is_unsure() -> None:
    probs, audio = two_pauses()
    trace = viewer.smart_turn_trace(probs, audio, Constant(0.1), silence_ms=450)
    assert [c["reason"] for c in trace["cuts"]] == ["silence", "silence"]
    assert trace["probes"]
    assert all(probe["p"] == 0.1 for probe in trace["probes"])


def test_probe_records_frame_threshold_and_kind() -> None:
    probs, audio = two_pauses()
    trace = viewer.smart_turn_trace(probs, audio, Constant(0.1), silence_ms=450)
    probe = trace["probes"][0]
    assert set(probe) == {"frame", "p", "threshold", "kind"}
    assert probe["frame"] >= 8
    assert probe["threshold"] == 0.5
    assert probe["kind"] == "turn"
