"""SAPI text to speech: voice choice, PCM conversion and threading (no COM needed)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from pipeline.tts_sapi import (
    PCM_RATE_HZ,
    SapiTts,
    TtsConfig,
    TtsConfigError,
    VoiceInfo,
    VoiceNotFoundError,
    check_voices,
    choose_voice,
    pcm16_to_float32,
    sapi_tts_factory,
)

pytestmark = pytest.mark.unit

DAVID = VoiceInfo("TTS_MS_EN-US_DAVID_11.0", "Microsoft David Desktop", (0x409,))
HEDDA = VoiceInfo("TTS_MS_DE-DE_HEDDA_11.0", "Microsoft Hedda Desktop", (0x407,))
HORTENSE = VoiceInfo("TTS_MS_FR-FR_HORTENSE_11.0", "Microsoft Hortense Desktop", (0x40C,))
SWISS = VoiceInfo("TTS_MS_DE-CH_X", "Microsoft Swiss Desktop", (0x807,))
ZIRA = VoiceInfo("TTS_MS_EN-US_ZIRA_11.0", "Microsoft Zira Desktop", (0x409,))
INSTALLED = [DAVID, HEDDA, ZIRA, HORTENSE]


class FakeBackend:
    def __init__(self, voices: list[VoiceInfo] | None = None, pcm: bytes | None = None) -> None:
        self.installed = INSTALLED if voices is None else voices
        self.pcm = np.array([0, 16384, -16384, 32767, -32768], dtype="<i2").tobytes()
        if pcm is not None:
            self.pcm = pcm
        self.calls: list[tuple[str, str, int, int]] = []
        self.threads: list[int] = []
        self.closed = False

    def voices(self) -> list[VoiceInfo]:
        return self.installed

    def speak(self, voice: VoiceInfo, text: str, rate: int, volume: int) -> bytes:
        self.calls.append((voice.name, text, rate, volume))
        self.threads.append(threading.get_ident())
        return self.pcm

    def close(self) -> None:
        self.closed = True


# --- config -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [{"rate": -11}, {"rate": 11}, {"volume": -1}, {"volume": 101}, {"voice": "  "}],
)
def test_bad_tts_settings_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(TtsConfigError):
        TtsConfig(**kwargs)  # type: ignore[arg-type]


def test_the_defaults_are_the_natural_speed_and_full_volume() -> None:
    config = TtsConfig()
    assert (config.rate, config.volume, config.voice) == (0, 100, None)


# --- voice choice -----------------------------------------------------------------------------


@pytest.mark.parametrize(("target", "expected"), [("de", HEDDA), ("fr", HORTENSE), ("en", DAVID)])
def test_the_voice_matches_the_language_of_the_target(target: str, expected: VoiceInfo) -> None:
    assert choose_voice(INSTALLED, target, None) == expected


def test_a_voice_name_picks_among_several_of_one_language() -> None:
    assert choose_voice(INSTALLED, "en", "zira") == ZIRA


def test_a_query_is_matched_by_part_of_the_name_without_case() -> None:
    assert choose_voice(INSTALLED, "de", "HEDDA DESK") == HEDDA


def test_a_dialect_of_the_language_is_accepted_when_nothing_closer_exists() -> None:
    assert choose_voice([DAVID, SWISS], "de", None) == SWISS


def test_the_exact_locale_beats_a_dialect() -> None:
    assert choose_voice([SWISS, HEDDA], "de", None) == HEDDA


def test_a_missing_language_names_the_installed_voices_and_the_fix() -> None:
    with pytest.raises(VoiceNotFoundError) as caught:
        choose_voice([DAVID, ZIRA], "fr", None)
    message = str(caught.value)
    assert "fr" in message
    assert "Microsoft David Desktop" in message
    assert "language pack" in message


def test_a_query_that_matches_nothing_names_the_query() -> None:
    with pytest.raises(VoiceNotFoundError, match="Katja"):
        choose_voice(INSTALLED, "de", "Katja")


def test_a_query_for_another_language_is_not_used() -> None:
    # "zira" is an English voice: asking for it on the German lane must fail, not read German
    # text with an English voice.
    with pytest.raises(VoiceNotFoundError):
        choose_voice(INSTALLED, "de", "zira")


def test_an_unknown_target_language_is_an_error() -> None:
    with pytest.raises(VoiceNotFoundError, match="xx"):
        choose_voice(INSTALLED, "xx", None)


# --- PCM --------------------------------------------------------------------------------------


def test_pcm16_becomes_float32_in_minus_one_to_one() -> None:
    raw = np.array([0, 16384, -16384, 32767, -32768], dtype="<i2").tobytes()
    out = pcm16_to_float32(raw)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 0.5, -0.5, 32767 / 32768, -1.0])


def test_an_odd_trailing_byte_is_dropped() -> None:
    assert len(pcm16_to_float32(b"\x01\x00\x02")) == 1


def test_no_bytes_give_no_samples() -> None:
    assert len(pcm16_to_float32(b"")) == 0


# --- the engine -------------------------------------------------------------------------------


def make(backend: FakeBackend, target: str = "de", config: TtsConfig | None = None) -> SapiTts:
    return SapiTts(target, config or TtsConfig(), backend_factory=lambda: backend)


def test_synthesize_returns_speech_at_the_pcm_rate() -> None:
    speech = make(FakeBackend()).synthesize("Guten Morgen.")
    assert speech.rate_hz == PCM_RATE_HZ == 22_050
    assert speech.samples.dtype == np.float32
    assert len(speech.samples) == 5


def test_the_voice_of_the_target_language_speaks() -> None:
    backend = FakeBackend()
    make(backend, "fr").synthesize("Bonjour.")
    assert backend.calls == [("Microsoft Hortense Desktop", "Bonjour.", 0, 100)]


def test_rate_volume_and_voice_come_from_the_config() -> None:
    backend = FakeBackend()
    make(backend, "en", TtsConfig(rate=3, volume=60, voice="zira")).synthesize("Hi.")
    assert backend.calls == [("Microsoft Zira Desktop", "Hi.", 3, 60)]


def test_blank_text_gives_empty_speech_without_calling_the_engine() -> None:
    backend = FakeBackend()
    speech = make(backend).synthesize("   ")
    assert len(speech.samples) == 0
    assert backend.calls == []


def test_each_call_returns_its_own_array() -> None:
    # Each call returns its own array: a lane may keep or resample it while the next call runs.
    engine = make(FakeBackend())
    first, second = engine.synthesize("a"), engine.synthesize("b")
    assert first.samples is not second.samples


def test_the_backend_is_created_once_per_thread_that_speaks() -> None:
    made: list[FakeBackend] = []

    def factory() -> FakeBackend:
        made.append(FakeBackend())
        return made[-1]

    engine = SapiTts("de", TtsConfig(), backend_factory=factory)
    engine.synthesize("one")
    engine.synthesize("two")
    assert len(made) == 1

    worker = threading.Thread(target=lambda: engine.synthesize("three"))
    worker.start()
    worker.join()
    # COM objects belong to the thread that made them: a second thread gets its own.
    assert len(made) == 2
    assert made[0].threads == [threading.get_ident()] * 2
    assert made[1].threads != made[0].threads


def test_a_missing_voice_fails_on_the_first_utterance_with_the_voice_list() -> None:
    engine = SapiTts("fr", TtsConfig(), backend_factory=lambda: FakeBackend([DAVID]))
    with pytest.raises(VoiceNotFoundError, match="Microsoft David Desktop"):
        engine.synthesize("Bonjour.")


def test_check_reports_a_missing_voice_before_any_utterance() -> None:
    engine = SapiTts("fr", TtsConfig(), backend_factory=lambda: FakeBackend([DAVID]))
    with pytest.raises(VoiceNotFoundError):
        engine.check()
    assert SapiTts("de", TtsConfig(), backend_factory=lambda: FakeBackend()).check() == HEDDA


def test_close_releases_the_backend() -> None:
    backend = FakeBackend()
    engine = make(backend)
    engine.synthesize("x")
    engine.close()
    assert backend.closed


# --- factory and start-up check ---------------------------------------------------------------


def test_the_factory_builds_an_engine_per_target_with_its_own_config() -> None:
    backend = FakeBackend()
    factory = sapi_tts_factory({"de": TtsConfig(rate=2)}, backend_factory=lambda: backend)
    factory("de").synthesize("Hallo.")
    factory("fr").synthesize("Bonjour.")  # not in the mapping: defaults
    assert backend.calls == [
        ("Microsoft Hedda Desktop", "Hallo.", 2, 100),
        ("Microsoft Hortense Desktop", "Bonjour.", 0, 100),
    ]


def test_check_voices_returns_the_voice_of_each_target() -> None:
    chosen = check_voices(
        {"de": TtsConfig(), "fr": TtsConfig(voice="hortense")},
        backend_factory=lambda: FakeBackend(),
    )
    assert chosen == {"de": HEDDA, "fr": HORTENSE}


def test_check_voices_reports_every_missing_voice_at_once() -> None:
    with pytest.raises(VoiceNotFoundError) as caught:
        check_voices(
            {"de": TtsConfig(), "fr": TtsConfig()}, backend_factory=lambda: FakeBackend([DAVID])
        )
    assert "'de'" in str(caught.value)
    assert "'fr'" in str(caught.value)


def test_check_voices_closes_its_backend() -> None:
    backend = FakeBackend()
    check_voices({"de": TtsConfig()}, backend_factory=lambda: backend)
    assert backend.closed


# --- trimming the silence SAPI pads around a sentence -----------------------------------------


def padded_pcm(lead_ms: int, tone_ms: int, tail_ms: int, noise: float = 0.0) -> bytes:
    def ms(n: int) -> int:
        return PCM_RATE_HZ * n // 1000

    quiet = lambda n: np.full(ms(n), int(noise * 32768), dtype="<i2")  # noqa: E731
    tone = np.full(ms(tone_ms), 16384, dtype="<i2")
    return np.concatenate([quiet(lead_ms), tone, quiet(tail_ms)]).tobytes()


def length_ms(samples: np.ndarray) -> float:
    return len(samples) * 1000 / PCM_RATE_HZ


def test_long_silence_around_a_sentence_is_cut_to_a_short_lead_and_a_short_pause() -> None:
    # SAPI adds about 100 ms before and 800 ms after each sentence; played one sentence after
    # another that is a gap of nearly a second, and the end of the utterance comes late.
    backend = FakeBackend(pcm=padded_pcm(100, 500, 800))
    speech = make(backend).synthesize("Guten Morgen.")
    assert length_ms(speech.samples) == pytest.approx(20 + 500 + 150, abs=2)


def test_silence_shorter_than_the_kept_lead_and_pause_is_left_alone() -> None:
    backend = FakeBackend(pcm=padded_pcm(10, 500, 60))
    speech = make(backend).synthesize("Hi.")
    assert length_ms(speech.samples) == pytest.approx(10 + 500 + 60, abs=2)


def test_very_quiet_noise_counts_as_silence() -> None:
    backend = FakeBackend(pcm=padded_pcm(100, 500, 800, noise=0.001))
    speech = make(backend).synthesize("Hi.")
    assert length_ms(speech.samples) == pytest.approx(20 + 500 + 150, abs=2)


def test_speech_made_only_of_silence_is_empty() -> None:
    backend = FakeBackend(pcm=padded_pcm(300, 0, 300))
    assert len(make(backend).synthesize("...").samples) == 0
