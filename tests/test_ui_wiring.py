"""The window's composition root: loaders, caches, speech, voices, strict mode (WD10, WD15, WD16)"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fakes import (
    EnergyVad,
    FakeDeviceList,
    FakeOutputStream,
    FakeStreamFactory,
    make_device,
    silence,
    speech,
    stream_of,
)
from test_app import HUGE_WATCHDOG_MS
from ui_fakes import DE_VOICE, EN_VOICE, FakeVoices, ToneTts

from pipeline.config import VAD_MODEL, Settings, check_files, parse_settings
from pipeline.lanes import TtsEngine
from pipeline.resample import RateConverter
from pipeline.transcribe import FasterWhisperEngine, TranscribeConfig
from pipeline.translate import TranslateConfig, translator_model_dir
from pipeline.tts_sapi import ComVoiceBackend, SapiTts, TtsConfig, VoiceNotFoundError
from pipeline.turn import DEFAULT_TURN_MODEL
from pipeline.ui.controller import WindowController
from pipeline.ui.model import ViewModel
from pipeline.ui.wiring import Loaders, RealWiring, lane_tts_config

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parent.parent / "models"


def settings_for(*extra: str) -> Settings:
    return parse_settings(
        [
            "--input",
            "V380",
            "--route",
            "de=Realtek,fr=HS100W",
            "--models-dir",
            "fake-models",
            "--whisper-threads",
            "2",
            *extra,
        ]
    )


class FakeModel:
    """Stands in for ``faster_whisper.WhisperModel``; records the options of each decode."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio: Any, **options: object) -> tuple[list[Any], object]:
        self.calls.append(options)

        class Segment:
            text = "hello world"

        return [Segment()], None


class FakeLink:
    """A turn link that records being closed."""

    def __init__(self, classifier: Any) -> None:
        self.classifier = classifier
        self.closed = 0

    def submit(self, request: Any) -> None:
        pass

    def poll(self) -> None:
        return None

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.closed += 1


class FakeTranslator:
    def __init__(self, target: str) -> None:
        self.target = target

    def translate(self, text: str) -> str:
        return f"[{self.target}] {text}"


class FakeGuard:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def arm(self) -> None:
        self.events.append("arm")

    def disarm(self) -> None:
        self.events.append("disarm")


def _no_missing_files(settings: Settings) -> None:
    """Stands in for ``check_files`` where the test is about something other than models."""


class Rig:
    """Wiring with every loader replaced by a recording fake (no model, no COM, no audio)."""

    def __init__(self, *, models_present: bool = False) -> None:
        self.events: list[str] = []
        self.voices = FakeVoices()
        self.engines: list[ToneTts] = []
        self.tts_targets: list[str] = []
        self.tts_loads: list[Settings] = []
        self.models: list[FakeModel] = []
        self.links: list[FakeLink] = []
        self.classifier_args: list[tuple[Path, int]] = []
        self.vad_paths: list[Path] = []
        self.translator_loads: list[tuple[Path, TranslateConfig]] = []
        self.whisper_loads: list[tuple[Path, TranscribeConfig]] = []
        self.glossary_paths: list[Path] = []
        self.fail_translator_for: str | None = None
        self.devices = FakeDeviceList(
            [
                make_device("Microphone (V380 USB)", index=1, inputs=2),
                make_device("Speakers (Realtek)", index=2, inputs=0, outputs=2),
                make_device("Headset (HS100W)", index=3, inputs=0, outputs=2),
            ]
        )
        self.input_factory = FakeStreamFactory()
        self.output_factory = FakeStreamFactory(FakeOutputStream)
        self.guard = FakeGuard(self.events)
        self.wiring = RealWiring(
            devices=self.devices,
            input_stream_factory=self.input_factory,
            output_stream_factory=self.output_factory,
            guard=self.guard,
            loaders=Loaders(
                vad=self._vad,
                turn_classifier=self._classifier,
                turn_link=self._link,
                whisper=self._whisper,
                translator=self._translator,
                glossary=self._glossary,
                tts=self._tts,
                voice_backend=self.voices,
                files=_no_missing_files if models_present else check_files,
            ),
        )

    def _tts(self, settings: Settings) -> Callable[[str], TtsEngine]:
        self.tts_loads.append(settings)

        def make(target: str) -> TtsEngine:
            engine = ToneTts()
            self.engines.append(engine)
            self.tts_targets.append(target)
            return engine

        return make

    def _vad(self, path: Path) -> Any:
        self.events.append("vad")
        self.vad_paths.append(path)
        return EnergyVad()

    def _classifier(self, path: Path, threads: int) -> Any:
        self.events.append("classifier")
        self.classifier_args.append((path, threads))
        return object()

    def _link(self, classifier: Any) -> FakeLink:
        link = FakeLink(classifier)
        self.links.append(link)
        return link

    def _whisper(self, path: Path, config: TranscribeConfig) -> FasterWhisperEngine:
        self.events.append("whisper")
        self.whisper_loads.append((path, config))
        model = FakeModel()
        self.models.append(model)
        return FasterWhisperEngine(model, config)

    def _translator(self, path: Path, config: TranslateConfig) -> FakeTranslator:
        self.events.append("translator")
        self.translator_loads.append((path, config))
        target = path.name.rsplit("-", 1)[1]
        if target == self.fail_translator_for:
            raise FileNotFoundError(f"translation model missing in {path}")
        return FakeTranslator(target)

    def _glossary(self, path: Path) -> Any:
        self.glossary_paths.append(path)
        raise AssertionError("not used")


def wait_for(predicate: Callable[[], bool], what: str, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        time.sleep(0.002)


# ---------------------------------------------------------------- WD10 speech


def test_wd10_the_real_loaders_are_sapi_voices_and_the_rate_converter() -> None:
    loaders = Loaders()
    assert loaders.resampler is RateConverter
    assert loaders.voice_backend is ComVoiceBackend
    settings = settings_for("--tts-rate", "3", "--tts-voice", "de=Hedda")
    engine = loaders.tts(settings)("de")  # no COM object exists before the first sentence
    assert isinstance(engine, SapiTts)
    assert engine._config == TtsConfig(rate=3, volume=100, voice="Hedda")


def test_wd10_a_lane_without_a_route_speaks_with_the_default_voice_at_the_shared_rate() -> None:
    settings = settings_for("--tts-rate", "3", "--tts-voice", "de=Hedda")
    assert lane_tts_config(settings, "de") == TtsConfig(rate=3, voice="Hedda")
    assert lane_tts_config(settings, "es") == TtsConfig(rate=3, voice=None)
    engine = Loaders().tts(settings)("es")
    assert isinstance(engine, SapiTts)
    assert engine._config == TtsConfig(rate=3, volume=100, voice=None)


def test_wd10_the_banner_is_gone_because_a_missing_voice_is_a_problem() -> None:
    assert RealWiring().tts_notice is None


def test_wd10_each_lane_gets_a_tts_engine_that_the_build_closes() -> None:
    rig = Rig(models_present=True)
    settings = settings_for()
    built = rig.wiring.build(settings)
    try:
        assert rig.tts_loads == [settings]
        assert rig.tts_targets == ["de", "fr"]
        assert [engine.closed for engine in rig.engines] == [0, 0]
    finally:
        built.pipeline.stop()
        built.close()
    assert [engine.closed for engine in rig.engines] == [1, 1]


def test_wd10_an_engine_made_before_a_failed_build_is_closed() -> None:
    rig = Rig(models_present=True)
    rig.fail_translator_for = "fr"  # de's lane exists by then
    with pytest.raises(FileNotFoundError):
        rig.wiring.build(settings_for())
    assert [engine.closed for engine in rig.engines] == [1]


# ---------------------------------------------------------------- WD10 problems


def test_wd10_missing_models_are_listed_without_loading_anything() -> None:
    rig = Rig()
    problems = rig.wiring.problems(settings_for())
    assert any("silero_vad.onnx" in problem for problem in problems)
    assert any("whisper-base.en" in problem for problem in problems)
    assert any("opus-mt-en-de" in problem for problem in problems)
    assert any("fetch_models.py" in problem for problem in problems)
    assert rig.events == []


def test_wd10_smart_mode_also_needs_the_turn_model() -> None:
    rig = Rig()
    problems = rig.wiring.problems(settings_for("--turn-mode", "smart"))
    assert any("smart-turn" in problem for problem in problems)


# ---------------------------------------------------------------- WD10 build


def test_wd10_build_loads_each_model_from_the_models_dir() -> None:
    rig = Rig()
    settings = settings_for()
    built = rig.wiring.build(settings)
    try:
        assert rig.vad_paths == [Path("fake-models") / VAD_MODEL]
        assert [path for path, _ in rig.whisper_loads] == [Path("fake-models/whisper-base.en")]
        assert [path for path, _ in rig.translator_loads] == [
            translator_model_dir(Path("fake-models"), "en", "de"),
            translator_model_dir(Path("fake-models"), "en", "fr"),
        ]
        assert built.pipeline.targets == ("de", "fr")
        assert not built.pipeline.running  # built, not started
        assert rig.links == []  # fixed mode has no turn model
        assert rig.classifier_args == []
    finally:
        built.pipeline.stop()
        built.close()


def test_wd10_translators_get_the_translation_settings() -> None:
    rig = Rig()
    built = rig.wiring.build(settings_for("--mt-beam", "3", "--mt-threads", "4"))
    built.pipeline.stop()
    built.close()
    assert {config.beam_size for _, config in rig.translator_loads} == {3}
    assert {config.threads for _, config in rig.translator_loads} == {4}


def test_wd10_smart_mode_builds_a_turn_link_that_close_releases() -> None:
    rig = Rig()
    built = rig.wiring.build(settings_for("--turn-mode", "smart", "--turn-threads", "2"))
    assert rig.classifier_args == [(Path("fake-models") / DEFAULT_TURN_MODEL, 2)]
    assert len(rig.links) == 1
    assert rig.links[0].closed == 0
    built.pipeline.stop()
    built.close()
    assert rig.links[0].closed == 1


def test_wd10_a_failure_while_building_releases_the_turn_link() -> None:
    rig = Rig()
    rig.fail_translator_for = "fr"  # fails inside the Pipeline constructor, after the stage
    with pytest.raises(FileNotFoundError, match="translation model missing"):
        rig.wiring.build(settings_for("--turn-mode", "smart"))
    assert [link.closed for link in rig.links] == [1]


def test_wd10_a_second_build_reuses_whisper_and_the_translators() -> None:
    rig = Rig()
    first = rig.wiring.build(settings_for())
    first.pipeline.stop()
    first.close()
    second = rig.wiring.build(settings_for("--silence-ms", "500", "--pre-roll-ms", "200"))
    second.pipeline.stop()
    second.close()
    assert len(rig.whisper_loads) == 1
    assert len(rig.translator_loads) == 2
    assert len(rig.vad_paths) == 2  # the VAD state belongs to one stream: always fresh


def test_wd10_changed_model_settings_reload_only_what_they_change() -> None:
    rig = Rig()
    for extra in (
        (),
        ("--whisper-threads", "3"),  # a new whisper model instance
        ("--whisper-model", "tiny.en"),
        ("--mt-beam", "4"),  # new translators, same whisper
    ):
        built = rig.wiring.build(settings_for(*extra))
        built.pipeline.stop()
        built.close()
    assert len(rig.whisper_loads) == 3
    assert len(rig.translator_loads) == 4


def test_wd10_a_cached_whisper_gets_the_new_decode_options() -> None:
    model = FakeModel()
    first = FasterWhisperEngine(model, TranscribeConfig(beam_size=1))
    second = first.with_config(TranscribeConfig(beam_size=3))
    audio = np.zeros(16_000, dtype=np.float32)
    first.transcribe(audio, hotwords=None, initial_prompt=None)
    second.transcribe(audio, hotwords=None, initial_prompt=None)
    assert [call["beam_size"] for call in model.calls] == [1, 3]


def test_wd10_strict_arms_the_guard_after_every_model_is_loaded() -> None:
    rig = Rig()
    built = rig.wiring.build(settings_for("--strict"))
    built.pipeline.stop()
    built.close()
    assert rig.events[-1] == "arm"
    assert rig.events.count("arm") == 1
    assert set(rig.events[:-1]) == {"vad", "whisper", "translator"}


def test_wd10_without_strict_the_guard_is_released() -> None:
    rig = Rig()
    built = rig.wiring.build(settings_for())
    built.pipeline.stop()
    built.close()
    assert rig.events[-1] == "disarm"


def test_wd10_the_glossary_is_loaded_from_its_path() -> None:
    rig = Rig()
    with pytest.raises(AssertionError, match="not used"):
        rig.wiring.build(settings_for("--glossary", "terms.toml"))
    assert rig.glossary_paths == [Path("terms.toml")]


def _fast_settings() -> Settings:
    base = settings_for()
    from pipeline.capture import CaptureConfig
    from pipeline.output import OutputConfig

    return replace(
        base,
        capture=CaptureConfig("V380", watchdog_ms=HUGE_WATCHDOG_MS),
        routes=tuple(
            replace(
                route, output=OutputConfig(route.output.device_query, watchdog_ms=HUGE_WATCHDOG_MS)
            )
            for route in base.routes
        ),
    )


def _dominant_hz(audio: Any, rate_hz: int) -> float:
    spectrum = np.abs(np.fft.rfft(audio[:, 0] * np.hanning(len(audio))))
    return float(np.argmax(spectrum) * rate_hz / len(audio))


def test_wd10_the_built_pipeline_speaks_at_the_device_rate_end_to_end_on_fakes() -> None:
    rig = Rig(models_present=True)
    built = rig.wiring.build(_fast_settings())
    pipeline = built.pipeline
    vm = ViewModel(pipeline)
    try:
        assert pipeline.start(lenient=True) == ()
        rig.input_factory.current.feed(
            stream_of(speech(600, 48_000), silence(700, 48_000)),
        )
        wait_for(lambda: len(vm.poll().rows) == 1, "the utterance row")
        wait_for(
            lambda: all(cell.arrived for cell in vm.poll().rows[0].translations),
            "both translations",
        )
        row = vm.poll().rows[0]
        assert row.text == "hello world"
        assert [cell.text for cell in row.translations] == ["[de] hello world", "[fr] hello world"]
        # The tone is 440 Hz at 22.05 kHz; the real RateConverter must bring it to the 48 kHz
        # device unchanged in pitch (the fake sound card only reads what the lane wrote).
        stream = rig.output_factory.streams[0]
        heard: list[Any] = []

        def something_played() -> bool:
            block = stream.pull(blocks=4)
            if np.any(block):
                heard.append(block)
            return len(heard) >= 4

        wait_for(something_played, "audio at the output")
        assert abs(_dominant_hz(np.concatenate(heard), 48_000) - 440.0) < 20.0
        assert pipeline.metrics.counter("lane.de.errors").value == 0
        assert pipeline.metrics.counter("lane.de.utterances").value == 1
    finally:
        vm.close()
        pipeline.stop()
        built.close()


def test_wd10_the_controller_connects_and_applies_through_the_real_wiring() -> None:
    rig = Rig(models_present=True)
    controller = WindowController(_fast_settings(), rig.wiring, auto_connect=True)
    try:
        controller.start()
        deadline = time.monotonic() + 10.0
        while controller.tick().busy is not None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        state = controller.tick()
        assert state.session_active
        assert state.tts_notice is None
        assert controller.apply_settings({"segmenter.silence_ms": 500}).restarting
        while controller.tick().busy is not None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        state = controller.tick()
        assert state.session_active
        assert state.settings.segmenter.silence_ms == 500
        assert state.generation == 2
        assert len(rig.whisper_loads) == 1  # the restart did not reload whisper
    finally:
        controller.shutdown()


# ---------------------------------------------------------------- WD15 voices in problems


def only_voice_problems(rig: Rig, settings: Settings | None = None) -> tuple[str, ...]:
    return rig.wiring.problems(settings or settings_for())


def test_wd15_all_voices_installed_is_no_problem() -> None:
    rig = Rig(models_present=True)
    assert only_voice_problems(rig) == ()


def test_wd15_a_missing_voice_is_one_message_for_its_target_only() -> None:
    rig = Rig(models_present=True)
    rig.voices.installed = [EN_VOICE, DE_VOICE]  # no French pack
    problems = only_voice_problems(rig)
    assert len(problems) == 1
    assert problems[0].startswith("voice missing: ")
    assert "language 'fr'" in problems[0]
    assert "Microsoft Hedda Desktop" in problems[0]  # what is installed
    assert "language pack" in problems[0]  # what to do
    assert "; " not in problems[0]  # one entry stays one entry when the controller joins them


def test_wd15_every_target_without_a_voice_gets_its_own_message() -> None:
    rig = Rig(models_present=True)
    rig.voices.installed = [EN_VOICE]
    problems = only_voice_problems(rig)
    assert len(problems) == 2
    assert "language 'de'" in problems[0]
    assert "language 'fr'" in problems[1]


def test_wd15_a_voice_name_that_matches_nothing_is_a_problem() -> None:
    rig = Rig(models_present=True)
    problems = only_voice_problems(rig, settings_for("--tts-voice", "de=Nobody"))
    assert len(problems) == 1
    assert "matching 'Nobody'" in problems[0]


def test_wd15_voice_and_model_problems_are_reported_together() -> None:
    rig = Rig()  # no models in fake-models
    rig.voices.installed = [EN_VOICE, DE_VOICE]
    problems = rig.wiring.problems(settings_for())
    assert any(problem.startswith("model missing: ") for problem in problems)
    assert any("fetch_models.py" in problem for problem in problems)
    assert any(problem.startswith("voice missing: ") for problem in problems)


def test_wd15_the_check_loads_no_model_and_opens_no_audio() -> None:
    rig = Rig(models_present=True)
    rig.voices.installed = []
    only_voice_problems(rig)
    assert rig.events == []  # no model load, no guard change
    assert rig.tts_loads == []  # no engine either
    assert rig.input_factory.streams == []
    assert rig.output_factory.streams == []


def test_wd15_the_check_runs_on_its_own_thread_and_closes_its_backend_there() -> None:
    rig = Rig(models_present=True)
    only_voice_problems(rig)
    me = threading.get_ident()
    threads = set(rig.voices.created) | set(rig.voices.listed) | set(rig.voices.closed)
    assert len(threads) == 1  # made, used and released on one thread (COM belongs to it)
    assert me not in threads  # never the caller's: the Tk thread stays free of COM
    assert len(rig.voices.created) == len(rig.voices.closed) == 2  # one per target


def test_wd15_answers_are_remembered_so_the_tk_thread_asks_once() -> None:
    rig = Rig(models_present=True)
    rig.voices.installed = [EN_VOICE, DE_VOICE]
    first = only_voice_problems(rig)
    asked = len(rig.voices.created)
    assert asked == 2
    assert only_voice_problems(rig) == first
    assert len(rig.voices.created) == asked  # nothing asked again
    only_voice_problems(rig, settings_for("--tts-voice", "de=Hedda"))
    assert len(rig.voices.created) == asked + 1  # a new voice name is a new question
    only_voice_problems(rig, settings_for("--tts-rate", "4"))
    assert len(rig.voices.created) == asked + 1  # the rate does not change the voice


def test_wd15_a_broken_speech_system_is_a_problem_and_is_asked_again() -> None:
    rig = Rig(models_present=True)
    rig.voices.fail = OSError("COM is not available")
    problems = only_voice_problems(rig)
    assert len(problems) == 2  # reported, not raised
    assert all(problem.startswith("speech synthesis unavailable for ") for problem in problems)
    assert "COM is not available" in problems[0]
    rig.voices.fail = None
    assert only_voice_problems(rig) == ()  # a failure that passed is not remembered


def test_wd15_the_check_is_not_blocked_by_a_model_load_in_progress() -> None:
    # problems() runs on the Tk thread while the worker may hold the model lock for seconds.
    rig = Rig(models_present=True)
    held = threading.Event()
    release = threading.Event()

    def slow_whisper(path: Path, config: TranscribeConfig) -> FasterWhisperEngine:
        held.set()
        assert release.wait(10.0)
        return FasterWhisperEngine(FakeModel(), config)

    wiring = RealWiring(
        devices=rig.devices,
        input_stream_factory=rig.input_factory,
        output_stream_factory=rig.output_factory,
        guard=rig.guard,
        loaders=Loaders(
            vad=rig._vad,
            whisper=slow_whisper,
            translator=rig._translator,
            tts=rig._tts,
            voice_backend=rig.voices,
            files=_no_missing_files,
        ),
    )
    builder = threading.Thread(target=lambda: wiring.build(settings_for()).close(), daemon=True)
    builder.start()
    try:
        assert held.wait(10.0)
        started = time.monotonic()
        assert wiring.problems(settings_for()) == ()
        assert time.monotonic() - started < 2.0
    finally:
        release.set()
        builder.join(10.0)


# ---------------------------------------------------------------- WD15 build and lane switch


def test_wd15_build_refuses_a_missing_voice_and_releases_what_it_made() -> None:
    rig = Rig(models_present=True)
    rig.voices.installed = [EN_VOICE, DE_VOICE]
    with pytest.raises(VoiceNotFoundError, match="language 'fr'"):
        rig.wiring.build(settings_for("--turn-mode", "smart"))
    assert [engine.closed for engine in rig.engines] == [1]  # de's engine, made before fr failed
    assert [link.closed for link in rig.links] == [1]


def one_lane_settings() -> Settings:
    return parse_settings(
        [
            "--input",
            "V380",
            "--targets",
            "de",
            "--route",
            "de=Realtek",
            "--models-dir",
            "fake-models",
        ]
    )


def test_wd15_a_lane_switched_to_a_language_without_a_voice_keeps_the_old_lane() -> None:
    rig = Rig(models_present=True)
    rig.voices.installed = [EN_VOICE, DE_VOICE]
    built = rig.wiring.build(one_lane_settings())
    try:
        with pytest.raises(VoiceNotFoundError, match="language 'fr'"):
            built.pipeline.set_output_target("de", "fr")
        assert built.pipeline.targets == ("de",)
    finally:
        built.pipeline.stop()
        built.close()


def test_wd15_a_lane_switched_to_a_language_with_a_voice_gets_it() -> None:
    rig = Rig(models_present=True)
    built = rig.wiring.build(one_lane_settings())
    try:
        built.pipeline.set_output_target("de", "fr")
        assert built.pipeline.targets == ("fr",)
        assert rig.tts_targets == ["de", "fr"]
    finally:
        built.pipeline.stop()
        built.close()
    assert [engine.closed for engine in rig.engines] == [1, 1]  # the replaced lane's too


# ---------------------------------------------------------------- WD16 window, missing voice


def settle(controller: WindowController) -> None:
    deadline = time.monotonic() + 10.0
    while controller.tick().busy is not None:
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_wd16_a_missing_voice_acts_like_a_missing_model_in_the_controller() -> None:
    rig = Rig(models_present=True)
    rig.voices.installed = [EN_VOICE, DE_VOICE]
    controller = WindowController(_fast_settings(), rig.wiring, auto_connect=True)
    try:
        controller.start()
        settle(controller)
        state = controller.tick()
        assert not state.session_active  # auto-connect did not start anything
        assert len(state.problems) == 1
        assert "language 'fr'" in state.problems[0]
        assert state.tts_notice is None
        assert rig.whisper_loads == []  # nothing was loaded
        assert not controller.connect_input()  # refused, like a missing model
        notice = controller.tick().notice
        assert notice is not None
        assert notice.level == "error"
        assert notice.text.startswith("Cannot start: voice missing: ")
        # The window keeps working: other settings are stored, the devices are listed.
        assert controller.apply_settings({"segmenter.silence_ms": 500}).errors == {}
        state = controller.tick()
        assert state.settings.segmenter.silence_ms == 500
        assert len(state.problems) == 1
        assert state.input_devices
    finally:
        controller.shutdown()


def test_wd16_the_voice_of_a_new_language_is_checked_in_standby_and_recovers() -> None:
    rig = Rig(models_present=True)
    rig.voices.installed = [EN_VOICE, DE_VOICE]
    base = _fast_settings()
    one_lane = replace(
        base, targets=("de",), routes=(replace(base.routes[0], tts=TtsConfig(voice="Hedda")),)
    )
    controller = WindowController(one_lane, rig.wiring)
    try:
        assert controller.tick().problems == ()
        assert controller.set_language("de", "fr")  # stored; the French pack is missing
        state = controller.tick()
        assert len(state.problems) == 1
        assert "language 'fr'" in state.problems[0]
        assert "matching" not in state.problems[0]  # the German voice name did not follow
        assert controller.set_language("fr", "de")
        assert controller.tick().problems == ()
    finally:
        controller.shutdown()


def test_wd16_a_running_lane_refused_for_a_missing_voice_leaves_the_session_up() -> None:
    rig = Rig(models_present=True)
    rig.voices.installed = [EN_VOICE, DE_VOICE]
    base = _fast_settings()
    one_lane = replace(base, targets=("de",), routes=base.routes[:1])
    controller = WindowController(one_lane, rig.wiring, auto_connect=True)
    try:
        controller.start()
        settle(controller)
        assert controller.tick().session_active
        assert controller.set_language("de", "fr")
        settle(controller)
        state = controller.tick()
        assert state.session_active  # the old lane is still there
        assert state.settings.targets == ("de",)
        assert state.notice is not None
        assert state.notice.level == "error"
        assert "language 'fr'" in state.notice.text
        assert state.generation == 1  # no restart happened
    finally:
        controller.shutdown()
