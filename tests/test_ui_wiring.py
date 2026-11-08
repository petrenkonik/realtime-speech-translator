"""The composition root of the window: loaders, caches, silent speech, strict mode (WD10, WD11)."""

from __future__ import annotations

import time
from collections.abc import Callable
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

from pipeline.config import VAD_MODEL, Settings, parse_settings
from pipeline.lanes import Speech
from pipeline.transcribe import FasterWhisperEngine, TranscribeConfig
from pipeline.translate import TranslateConfig, translator_model_dir
from pipeline.turn import DEFAULT_TURN_MODEL
from pipeline.ui.controller import WindowController
from pipeline.ui.model import ViewModel
from pipeline.ui.wiring import (
    SILENT_TTS_NOTICE,
    Loaders,
    RealWiring,
    SilenceResampler,
    SilentTtsEngine,
    silence_resampler_factory,
)
from pipeline.vad import SAMPLE_RATE_HZ

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


class Rig:
    """Wiring with every loader replaced by a recording fake."""

    def __init__(self) -> None:
        self.events: list[str] = []
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
            ),
        )

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


# ---------------------------------------------------------------- WD10 silent speech


def test_wd10_silent_speech_is_silent_non_empty_and_at_16_khz() -> None:
    engine = SilentTtsEngine()
    result = engine.synthesize("Guten Tag, wie geht es Ihnen heute?")
    assert isinstance(result, Speech)
    assert result.rate_hz == SAMPLE_RATE_HZ
    assert result.samples.dtype == np.float32
    assert len(result.samples) > 0
    assert not np.any(result.samples)


def test_wd10_silent_speech_length_follows_the_text_within_bounds() -> None:
    engine = SilentTtsEngine()
    short = len(engine.synthesize("a").samples)
    longer = len(engine.synthesize("a" * 80).samples)
    huge = len(engine.synthesize("a" * 100_000).samples)
    assert short == 200 * SAMPLE_RATE_HZ // 1000  # never shorter than 200 ms
    assert short < longer < huge
    assert huge == 4 * SAMPLE_RATE_HZ  # never longer than 4 s


def test_wd10_the_silence_resampler_keeps_the_duration() -> None:
    resampler = silence_resampler_factory(16_000, 48_000)
    assert isinstance(resampler, SilenceResampler)
    out = resampler.process(np.zeros(1600, dtype=np.float32))
    assert len(out) == 4800
    assert out.dtype == np.float32
    assert not np.any(out)
    assert (
        len(silence_resampler_factory(16_000, 44_100).process(np.zeros(1600, np.float32))) == 4410
    )


def test_wd10_the_banner_text_is_the_agreed_one() -> None:
    assert SILENT_TTS_NOTICE == "TTS not installed: outputs are silent"
    assert RealWiring().tts_notice == SILENT_TTS_NOTICE


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
    from dataclasses import replace

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


def test_wd10_the_built_pipeline_runs_end_to_end_on_fakes_with_silent_speech() -> None:
    rig = Rig()
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
        # The lanes played something (silence) at the device rate, so the outputs saw audio
        # frames and no lane error was counted.
        assert pipeline.metrics.counter("lane.de.errors").value == 0
        assert pipeline.metrics.counter("lane.de.utterances").value == 1
        audio = rig.output_factory.streams[0].pull(blocks=20)
        assert not np.any(audio)
    finally:
        vm.close()
        pipeline.stop()
        built.close()


def test_wd10_the_controller_connects_and_applies_through_the_real_wiring() -> None:
    rig = Rig()
    rig.wiring.problems = lambda settings: ()  # type: ignore[method-assign]  # fake folders
    controller = WindowController(_fast_settings(), rig.wiring, auto_connect=True)
    try:
        controller.start()
        deadline = time.monotonic() + 10.0
        while controller.tick().busy is not None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        state = controller.tick()
        assert state.session_active
        assert state.tts_notice == SILENT_TTS_NOTICE
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
