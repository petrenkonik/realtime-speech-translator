"""T5: the real Windows SAPI voices speak into memory, fast and without touching the disk.

The English voice is part of every Windows install, so those tests always run. The German and
French voices come with language packs: their tests skip, with the reason, when a pack is missing.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from pipeline.tts_sapi import (
    PCM_RATE_HZ,
    ComVoiceBackend,
    SapiTts,
    TtsConfig,
    VoiceNotFoundError,
    check_voices,
)

pytestmark = pytest.mark.model

TARGET_FIRST_AUDIO_MS = 300.0
SENTENCES = {
    "en": "Good morning, everyone, and welcome to the meeting.",
    "de": "Guten Morgen zusammen und willkommen zum Meeting.",
    "fr": "Bonjour à tous et bienvenue à la réunion.",
}


@pytest.fixture(scope="module", params=["en", "de", "fr"])
def engine(request: pytest.FixtureRequest) -> tuple[str, SapiTts]:
    target = str(request.param)
    try:
        check_voices({target: TtsConfig()})
    except VoiceNotFoundError as exc:
        pytest.skip(str(exc))
    tts = SapiTts(target)
    tts.synthesize("Warm up.")  # the first call pays for COM set-up, as the lane's start does
    return target, tts


def test_t5_speech_is_a_non_empty_float_array_at_the_pcm_rate(
    engine: tuple[str, SapiTts],
) -> None:
    target, tts = engine
    speech = tts.synthesize(SENTENCES[target])
    assert speech.rate_hz == PCM_RATE_HZ
    assert speech.samples.dtype == np.float32
    assert np.all(np.isfinite(speech.samples))
    assert 1.0 <= len(speech.samples) / speech.rate_hz <= 8.0  # a sentence, not a click
    assert float(np.max(np.abs(speech.samples))) > 0.05  # audible, not silence
    assert float(np.max(np.abs(speech.samples))) <= 1.0


def test_t5_a_sentence_is_ready_well_inside_the_first_audio_budget(
    engine: tuple[str, SapiTts],
) -> None:
    target, tts = engine
    times = []
    for _ in range(5):
        started = time.perf_counter()
        tts.synthesize(SENTENCES[target])
        times.append((time.perf_counter() - started) * 1000.0)
    print(f"T5 sentence ms {target}: median {statistics.median(times):.0f}, max {max(times):.0f}")
    assert statistics.median(times) <= TARGET_FIRST_AUDIO_MS


def test_t5_a_longer_text_makes_longer_speech(engine: tuple[str, SapiTts]) -> None:
    target, tts = engine
    short = tts.synthesize(SENTENCES[target])
    long = tts.synthesize(f"{SENTENCES[target]} {SENTENCES[target]}")
    assert len(long.samples) > 1.6 * len(short.samples)


def test_t5_a_second_call_does_not_replay_the_tail_of_the_first(
    engine: tuple[str, SapiTts],
) -> None:
    # A reused memory stream keeps the tail of a longer text; each call must be self-contained.
    target, tts = engine
    long = tts.synthesize(f"{SENTENCES[target]} {SENTENCES[target]}")
    short = tts.synthesize(SENTENCES[target])
    assert len(short.samples) < 0.7 * len(long.samples)


def test_t5_speaking_touches_neither_disk_nor_network(
    engine: tuple[str, SapiTts], no_disk_writes: list[object]
) -> None:
    target, tts = engine
    tts.synthesize(SENTENCES[target])
    assert no_disk_writes == []


def test_t5_a_second_thread_gets_its_own_voice_object(engine: tuple[str, SapiTts]) -> None:
    import threading

    target, tts = engine
    results: list[int] = []
    worker = threading.Thread(
        target=lambda: results.append(len(tts.synthesize(SENTENCES[target]).samples))
    )
    worker.start()
    worker.join(30)
    assert results
    assert results[0] > PCM_RATE_HZ // 2


def test_t5_the_installed_voices_are_listed_with_their_languages() -> None:
    backend = ComVoiceBackend()
    try:
        voices = backend.voices()
    finally:
        backend.close()
    assert voices
    assert all(voice.language_ids for voice in voices)
