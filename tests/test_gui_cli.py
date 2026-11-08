"""The ``gui`` and ``list-devices`` commands and the settings the window starts from (WD1)."""

from __future__ import annotations

import argparse

import pytest
from fakes import FakeDeviceList, make_device

from pipeline.config import (
    SUPPORTED_TARGETS,
    UNSELECTED_DEVICE,
    ConfigError,
    add_run_arguments,
    is_unselected,
    parse_settings,
)

pytestmark = pytest.mark.unit


def gui_settings(*argv: str):  # type: ignore[no-untyped-def]
    return parse_settings(list(argv), gui=True)


def test_wd1_the_window_may_start_with_no_device_named() -> None:
    settings = gui_settings()
    assert is_unselected(settings.capture.device_query)
    assert [is_unselected(route.output.device_query) for route in settings.routes] == [True, True]
    assert settings.targets == SUPPORTED_TARGETS


def test_wd1_a_device_named_on_the_command_line_is_kept() -> None:
    settings = gui_settings("--input", "V380", "--route", "de=Realtek")
    assert settings.capture.device_query == "V380"
    by_target = {route.target: route.output.device_query for route in settings.routes}
    assert by_target["de"] == "Realtek"
    assert is_unselected(by_target["fr"])  # only de was routed


def test_wd1_the_run_command_still_insists_on_its_devices() -> None:
    with pytest.raises(ConfigError, match="--input"):
        parse_settings(["--route", "de=Realtek,fr=HS100W"])
    with pytest.raises(ConfigError, match="--route"):
        parse_settings(["--input", "V380"])


def test_wd1_the_other_run_flags_still_validate_for_the_window() -> None:
    settings = gui_settings("--silence-ms", "500", "--hide-text", "--strict", "--targets", "de")
    assert settings.segmenter.silence_ms == 500
    assert settings.capture.silence_ms == 500
    assert settings.hide_text
    assert settings.strict
    assert settings.targets == ("de",)
    with pytest.raises(ConfigError, match="silence"):
        gui_settings("--silence-ms", "100")
    with pytest.raises(ConfigError, match="not in --targets"):
        gui_settings("--route", "fr=HS100W", "--targets", "de")


def test_wd1_the_window_has_no_use_for_the_plain_output_flag() -> None:
    with pytest.raises(ConfigError):
        gui_settings("--no-ui")
    parser = argparse.ArgumentParser(add_help=False)
    add_run_arguments(parser)
    assert parser.parse_args(["--input", "x", "--no-ui"]).no_ui


def test_wd1_an_unselected_device_can_never_match_a_real_one() -> None:
    assert is_unselected(UNSELECTED_DEVICE)
    assert not is_unselected("Microphone (V380 USB)")
    assert not is_unselected("")  # a blank query is a different mistake, refused elsewhere


def test_wd1_main_refuses_a_bad_gui_option_with_a_message_not_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pipeline.__main__ import main

    assert main(["gui", "--silence-ms", "100"]) == 2
    assert "silence" in capsys.readouterr().err


def test_wd1_main_needs_a_command(capsys: pytest.CaptureFixture[str]) -> None:
    from pipeline.__main__ import main

    assert main([]) == 2
    assert "gui" in capsys.readouterr().err


def test_wd1_list_devices_prints_the_wasapi_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pipeline.__main__ import main

    listing = FakeDeviceList(
        [
            make_device("Microphone (V380 USB)", inputs=2),
            make_device("Speakers (Realtek)", inputs=0, outputs=2),
            make_device("Some MME device", host_api="MME"),
        ]
    )
    assert main(["list-devices"], listing=listing) == 0
    out = capsys.readouterr().out
    assert "V380" in out
    assert "Realtek" in out
    assert "MME device" not in out


def test_wd1_the_gui_command_hands_settings_to_the_window_runner() -> None:
    from pipeline.__main__ import main

    seen = []

    def runner(settings):  # type: ignore[no-untyped-def]
        seen.append(settings)
        return 0

    assert main(["gui", "--input", "V380", "--strict"], gui_runner=runner) == 0
    assert seen[0].capture.device_query == "V380"
    assert seen[0].strict
