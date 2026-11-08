"""Command-line settings: parsing, validation and the check for missing model files."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.config import (
    ConfigError,
    Settings,
    check_files,
    parse_settings,
)
from pipeline.transcribe import WHISPER_MODELS

pytestmark = pytest.mark.unit

BASE = ["--input", "V380", "--route", "de=Realtek,fr=HS100W"]


def parse(*extra: str) -> Settings:
    return parse_settings([*BASE, *extra])


# --- defaults -------------------------------------------------------------------------------


def test_the_minimal_command_line_gives_the_documented_defaults() -> None:
    settings = parse()
    assert settings.source == "en"
    assert settings.targets == ("de", "fr")
    assert settings.capture.device_query == "V380"
    assert settings.segmenter.silence_ms == 450
    assert settings.segmenter.turn_mode == "fixed"
    assert settings.transcribe.model == "base.en"
    assert settings.transcribe.threads == 4
    assert settings.turn_threads == 1
    assert settings.glossary_path is None
    assert not settings.strict
    assert not settings.hide_text
    assert settings.dashboard_mode == "live"


def test_the_silence_threshold_reaches_the_segmenter_and_the_capture_report() -> None:
    settings = parse("--silence-ms", "600")
    assert settings.segmenter.silence_ms == 600
    assert settings.capture.silence_ms == 600


# --- routes ---------------------------------------------------------------------------------


def test_every_target_gets_its_own_output() -> None:
    settings = parse()
    assert [(r.target, r.output.device_query) for r in settings.routes] == [
        ("de", "Realtek"),
        ("fr", "HS100W"),
    ]
    assert [r.radio_offset_ms for r in settings.routes] == [0, 0]


def test_the_route_order_follows_the_targets_not_the_route_flag() -> None:
    settings = parse_settings(
        ["--input", "V380", "--targets", "fr,de", "--route", "de=Realtek,fr=HS100W"]
    )
    assert [r.target for r in settings.routes] == ["fr", "de"]


def test_the_radio_offset_and_prebuffer_go_to_the_right_lane() -> None:
    settings = parse("--radio-offset", "fr=120", "--out-prebuffer-ms", "40")
    by_target = {r.target: r for r in settings.routes}
    assert by_target["fr"].radio_offset_ms == 120
    assert by_target["de"].radio_offset_ms == 0
    assert by_target["de"].output.prebuffer_ms == 40


def test_a_single_target_needs_only_its_own_route() -> None:
    settings = parse_settings(
        ["--input", "V380", "--targets", "de", "--route", "de=Realtek"],
    )
    assert settings.targets == ("de",)
    assert len(settings.routes) == 1


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["--input", "V380"], "--route"),
        (["--input", "V380", "--route", "de=Realtek"], "fr"),
        (["--input", "V380", "--route", "de=Realtek,fr=HS100W,es=X"], "es"),
        (["--input", "V380", "--route", "de"], "de"),
        (["--input", "V380", "--route", "de=,fr=HS100W"], "de"),
        (["--input", "V380", "--route", "de=A,de=B,fr=C"], "twice"),
        ([*BASE, "--radio-offset", "es=10"], "es"),
        ([*BASE, "--radio-offset", "de=-5"], "radio"),
        ([*BASE, "--radio-offset", "de=abc"], "de"),
        ([*BASE, "--radio-offset", "de=5000"], "radio"),
    ],
)
def test_bad_routes_and_offsets_are_actionable_errors(argv: list[str], fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        parse_settings(argv)


# --- languages ------------------------------------------------------------------------------


def test_only_english_is_a_source_language() -> None:
    with pytest.raises(ConfigError, match="source"):
        parse("--source", "ru")


@pytest.mark.parametrize("targets", ["", "de,de", "en", "de,xx", "ru", "de,,fr"], ids=repr)
def test_bad_target_lists_are_rejected(targets: str) -> None:
    with pytest.raises(ConfigError, match="target"):
        parse_settings(["--input", "V380", "--targets", targets, "--route", "de=A,fr=B"])


# --- the options of the other modules -------------------------------------------------------


@pytest.mark.parametrize("name", WHISPER_MODELS)
def test_both_whisper_models_can_be_chosen(name: str) -> None:
    assert parse("--whisper-model", name).transcribe.model == name


def test_an_unknown_whisper_model_names_the_choices() -> None:
    with pytest.raises(ConfigError, match=r"tiny.en"):
        parse("--whisper-model", "large-v3")


def test_the_turn_options_reach_the_segmenter() -> None:
    settings = parse(
        "--turn-mode", "smart",
        "--turn-check-ms", "150",
        "--turn-recheck-ms", "128",
        "--turn-threshold", "0.6",
        "--soft-cut-ms", "0",
        "--soft-pause-ms", "160",
        "--soft-cut-threshold", "0.8",
        "--turn-threads", "2",
        "--max-utterance-ms", "12000",
        "--cut-search-ms", "1000",
        "--pre-roll-ms", "300",
    )  # fmt: skip
    seg = settings.segmenter
    assert (seg.turn_mode, seg.turn_check_ms, seg.turn_recheck_ms) == ("smart", 150, 128)
    assert (seg.turn_threshold, seg.soft_cut_ms, seg.soft_pause_ms) == (0.6, 0, 160)
    assert (seg.soft_cut_threshold, seg.max_utterance_ms) == (0.8, 12_000)
    assert (seg.cut_search_ms, seg.pre_roll_ms) == (1000, 300)
    assert settings.turn_threads == 2


def test_whisper_threads_reach_the_transcriber() -> None:
    assert parse("--whisper-threads", "6").transcribe.threads == 6


@pytest.mark.parametrize(
    "flag_and_value",
    [
        ("--silence-ms", "300"),
        ("--silence-ms", "900"),
        ("--turn-mode", "sometimes"),
        ("--turn-check-ms", "125"),
        ("--turn-threshold", "0.95"),
        ("--soft-cut-ms", "1000"),
        ("--turn-threads", "3"),
        ("--turn-threads", "0"),
        ("--whisper-threads", "0"),
        ("--whisper-threads", "9"),
        ("--max-utterance-ms", "1000"),
        ("--out-prebuffer-ms", "500"),
    ],
    ids=lambda p: " ".join(p) if isinstance(p, tuple) else str(p),
)
def test_values_outside_the_range_are_config_errors(flag_and_value: tuple[str, str]) -> None:
    with pytest.raises(ConfigError):
        parse(*flag_and_value)


def test_the_error_names_the_flag_for_a_module_range_violation() -> None:
    with pytest.raises(ConfigError, match="silence"):
        parse("--silence-ms", "300")


def test_a_soft_cut_threshold_below_the_turn_threshold_is_refused() -> None:
    with pytest.raises(ConfigError, match="soft_cut_threshold"):
        parse("--turn-threshold", "0.8", "--soft-cut-threshold", "0.6")


# --- switches -------------------------------------------------------------------------------


def test_the_switches() -> None:
    settings = parse("--strict", "--hide-text", "--no-ui")
    assert settings.strict
    assert settings.hide_text
    assert settings.dashboard_mode == "plain"


def test_paths() -> None:
    settings = parse("--glossary", "terms.toml", "--models-dir", "somewhere")
    assert settings.glossary_path == Path("terms.toml")
    assert settings.models_dir == Path("somewhere")


# --- argparse itself never exits the process ------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--input"],
        [*BASE, "--no-such-flag"],
        [*BASE, "--silence-ms", "fast"],
    ],
    ids=lambda a: " ".join(a) or "empty",
)
def test_a_syntax_error_is_a_config_error_not_a_system_exit(argv: list[str]) -> None:
    with pytest.raises(ConfigError):
        parse_settings(argv)


def test_the_input_is_required_and_must_not_be_blank() -> None:
    with pytest.raises(ConfigError, match="input"):
        parse_settings(["--input", "  ", "--route", "de=A,fr=B"])


def test_a_blank_device_in_a_route_is_refused() -> None:
    with pytest.raises(ConfigError, match="de"):
        parse_settings(["--input", "V380", "--route", "de= ,fr=B"])


# --- the model files ------------------------------------------------------------------------


def _stock_models(
    root: Path,
    *,
    whisper: str = "base.en",
    smart: bool = False,
    targets: tuple[str, ...] = ("de", "fr"),
) -> None:
    (root / "silero_vad.onnx").write_bytes(b"x")
    (root / f"whisper-{whisper}").mkdir()
    for target in targets:
        (root / f"opus-mt-en-{target}").mkdir()
        (root / f"opus-mt-en-{target}" / "model.bin").write_bytes(b"x")
    if smart:
        (root / "smart-turn").mkdir()
        (root / "smart-turn" / "smart-turn-v3.2-gpu.onnx").write_bytes(b"x")


def test_check_files_passes_when_everything_is_there(tmp_path: Path) -> None:
    _stock_models(tmp_path)
    check_files(parse("--models-dir", str(tmp_path)))


def test_a_missing_whisper_model_points_at_the_fetch_script(tmp_path: Path) -> None:
    _stock_models(tmp_path, whisper="tiny.en")
    with pytest.raises(ConfigError, match=r"whisper-base\.en.*fetch_models"):
        check_files(parse("--models-dir", str(tmp_path)))


def test_the_chosen_whisper_model_is_the_one_that_is_checked(tmp_path: Path) -> None:
    _stock_models(tmp_path, whisper="tiny.en")
    check_files(parse("--models-dir", str(tmp_path), "--whisper-model", "tiny.en"))


def test_the_smart_turn_model_is_needed_only_in_smart_mode(tmp_path: Path) -> None:
    _stock_models(tmp_path)
    check_files(parse("--models-dir", str(tmp_path)))
    with pytest.raises(ConfigError, match=r"smart-turn.*fetch_models"):
        check_files(parse("--models-dir", str(tmp_path), "--turn-mode", "smart"))
    _stock_models_smart = tmp_path / "smart-turn"
    _stock_models_smart.mkdir()
    (_stock_models_smart / "smart-turn-v3.2-gpu.onnx").write_bytes(b"x")
    check_files(parse("--models-dir", str(tmp_path), "--turn-mode", "smart"))


def test_a_missing_vad_model_is_reported(tmp_path: Path) -> None:
    (tmp_path / "whisper-base.en").mkdir()
    with pytest.raises(ConfigError, match=r"silero_vad\.onnx"):
        check_files(parse("--models-dir", str(tmp_path)))


def test_a_missing_glossary_file_is_reported(tmp_path: Path) -> None:
    _stock_models(tmp_path)
    with pytest.raises(ConfigError, match=r"nope\.toml"):
        check_files(parse("--models-dir", str(tmp_path), "--glossary", str(tmp_path / "nope.toml")))


def test_every_missing_file_is_listed_at_once(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        check_files(parse("--models-dir", str(tmp_path), "--turn-mode", "smart"))
    text = str(info.value)
    assert "silero_vad.onnx" in text
    assert "whisper-base.en" in text
    assert "smart-turn" in text


def test_a_missing_translation_model_is_reported_for_each_target(tmp_path: Path) -> None:
    _stock_models(tmp_path, targets=("de",))
    with pytest.raises(ConfigError, match=r"opus-mt-en-fr.*fetch_models"):
        check_files(parse("--models-dir", str(tmp_path)))


def test_only_the_chosen_targets_need_their_translation_model(tmp_path: Path) -> None:
    _stock_models(tmp_path, targets=("de",))
    settings = parse_settings(
        [
            "--input",
            "V380",
            "--targets",
            "de",
            "--route",
            "de=Realtek",
            "--models-dir",
            str(tmp_path),
        ]
    )
    check_files(settings)


def test_the_translation_options_reach_the_settings() -> None:
    settings = parse("--mt-threads", "3", "--mt-beam", "4")
    assert (settings.translate.threads, settings.translate.beam_size) == (3, 4)
    assert (parse().translate.threads, parse().translate.beam_size) == (2, 2)


@pytest.mark.parametrize("flag_and_value", [("--mt-threads", "0"), ("--mt-beam", "9")])
def test_translation_options_out_of_range_are_config_errors(
    flag_and_value: tuple[str, str],
) -> None:
    with pytest.raises(ConfigError):
        parse(*flag_and_value)


# --- speech ---------------------------------------------------------------------------------


def test_the_speech_defaults_are_natural_speed_and_the_first_voice_of_the_language() -> None:
    for route in parse().routes:
        assert (route.tts.rate, route.tts.volume, route.tts.voice) == (0, 100, None)


def test_the_tts_rate_applies_to_every_lane() -> None:
    assert [route.tts.rate for route in parse("--tts-rate", "-2").routes] == [-2, -2]


def test_a_voice_is_chosen_per_target() -> None:
    by_target = {r.target: r for r in parse("--tts-voice", "de=Hedda,fr=Hortense").routes}
    assert by_target["de"].tts.voice == "Hedda"
    assert by_target["fr"].tts.voice == "Hortense"


def test_a_voice_for_one_target_leaves_the_other_on_its_default() -> None:
    by_target = {r.target: r for r in parse("--tts-voice", "fr=Hortense").routes}
    assert by_target["de"].tts.voice is None


@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        (["--tts-rate", "11"], "rate"),
        (["--tts-rate", "-11"], "rate"),
        (["--tts-voice", "xx=Hedda"], "--tts-voice"),
        (["--tts-voice", "de"], "--tts-voice"),
        (["--tts-voice", "de=Hedda,de=Katja"], "twice"),
        (["--tts-voice", "de= "], "blank"),
    ],
)
def test_bad_speech_flags_are_config_errors(extra: list[str], fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        parse(*extra)
