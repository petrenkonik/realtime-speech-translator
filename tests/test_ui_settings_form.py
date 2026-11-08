"""Settings tab logic without Tk: texts, parsing, hints, notes and error placement (WD5)."""

from __future__ import annotations

import pytest

from pipeline.config import Settings, parse_settings
from pipeline.ui import settings_form as form_module
from pipeline.ui.settings_form import SettingsForm, field_hint, page_note, restart_note
from pipeline.ui.settings_schema import (
    GENERAL_ERROR_KEY,
    SCHEMA,
    SettingsValidationError,
    build_settings,
    schema_for,
    values_from_settings,
)

pytestmark = pytest.mark.unit


def two_lanes() -> Settings:
    return parse_settings(["--input", "V380", "--route", "de=Realtek,fr=HS100W"])


def field(key: str):  # type: ignore[no-untyped-def]
    return next(item for item in SCHEMA if item.key == key)


def test_wd5_a_form_starts_with_the_texts_of_the_current_settings() -> None:
    settings = two_lanes()
    form = SettingsForm(settings)
    assert form.value("segmenter.silence_ms") == "450"
    assert form.value("segmenter.speech_threshold") == "0.5"
    assert form.value("segmenter.turn_mode") == "fixed"
    assert form.value("hide_text") is False
    assert form.value("routes.de.radio_offset_ms") == "0"
    assert [item.key for item in form.fields] == [item.key for item in schema_for(settings)]


def test_wd5_collecting_untouched_texts_gives_back_the_same_settings() -> None:
    settings = two_lanes()
    values, errors = SettingsForm(settings).collect()
    assert errors == {}
    assert values == values_from_settings(settings)
    assert build_settings(settings, values) == settings


def test_wd5_edited_texts_become_typed_values() -> None:
    settings = two_lanes()
    form = SettingsForm(settings)
    form.set("segmenter.silence_ms", " 500 ")
    form.set("segmenter.exit_threshold", "0.3")
    form.set("hide_text", True)
    form.set("segmenter.turn_mode", "smart")
    values, errors = form.collect()
    assert errors == {}
    assert values["segmenter.silence_ms"] == 500
    assert isinstance(values["segmenter.silence_ms"], int)
    assert values["segmenter.exit_threshold"] == 0.3
    assert values["hide_text"] is True
    assert values["segmenter.turn_mode"] == "smart"
    new = build_settings(settings, values)
    assert new.segmenter.silence_ms == 500
    assert new.capture.silence_ms == 500
    assert new.hide_text


@pytest.mark.parametrize(
    ("key", "text", "fragment"),
    [
        ("segmenter.silence_ms", "abc", "whole number"),
        ("segmenter.silence_ms", "450.5", "whole number"),
        ("segmenter.silence_ms", "", "required"),
        ("segmenter.speech_threshold", "high", "number"),
        ("segmenter.speech_threshold", "", "required"),
    ],
)
def test_wd5_text_that_is_not_a_number_is_an_error_on_its_field(
    key: str, text: str, fragment: str
) -> None:
    form = SettingsForm(two_lanes())
    form.set(key, text)
    values, errors = form.collect()
    assert fragment in errors[key]
    assert key not in values  # never guessed
    assert set(errors) == {key}


def test_wd5_a_float_field_accepts_an_integer_text() -> None:
    form = SettingsForm(two_lanes())
    form.set("segmenter.speech_threshold", "1")
    values, errors = form.collect()
    assert errors == {}
    assert values["segmenter.speech_threshold"] == 1.0


def test_wd5_discarding_edits_goes_back_to_the_applied_settings() -> None:
    settings = two_lanes()
    form = SettingsForm(settings)
    form.set("segmenter.silence_ms", "999")
    assert form.edited_keys() == ("segmenter.silence_ms",)
    form.load(settings)
    assert form.value("segmenter.silence_ms") == "450"
    assert form.edited_keys() == ()


def test_wd5_reset_to_defaults_uses_the_schema_defaults() -> None:
    settings = build_settings(two_lanes(), {"segmenter.silence_ms": 600, "hide_text": True})
    form = SettingsForm(settings)
    assert form.value("segmenter.silence_ms") == "600"
    form.reset_to_defaults()
    assert form.value("segmenter.silence_ms") == "450"
    assert form.value("hide_text") is False
    values, errors = form.collect()
    assert errors == {}
    assert build_settings(settings, values).segmenter.silence_ms == 450


def test_wd5_the_form_follows_the_lanes_of_the_settings() -> None:
    one = SettingsForm(
        parse_settings(["--input", "V380", "--route", "de=Realtek", "--targets", "de"])
    )
    keys = {item.key for item in one.fields}
    assert "routes.de.radio_offset_ms" in keys
    assert "routes.fr.radio_offset_ms" not in keys
    assert one.groups() == (
        "Segmentation",
        "Turn detection",
        "Whisper",
        "Capture",
        "Output",
        "Advanced",
    )


def test_wd5_hints_state_the_range_and_unit() -> None:
    assert field_hint(field("segmenter.silence_ms")) == "450-600 ms"
    assert field_hint(field("segmenter.soft_cut_ms")) == "0 (off) or 3000-12000 ms"
    assert field_hint(field("segmenter.speech_threshold")) == "0.05-1"
    assert field_hint(field("capture.ring_seconds")) == "0.5-30 s"
    assert field_hint(field("transcribe.model")) == ""  # a drop-down shows its own options
    assert field_hint(field("hide_text")) == ""
    route = next(item for item in schema_for(two_lanes()) if item.key.startswith("routes.de"))
    assert field_hint(route) == "0-2000 ms"


def test_wd5_every_field_says_whether_it_restarts_the_pipeline() -> None:
    assert "restart" in restart_note(field("segmenter.silence_ms")).lower()
    assert "immediately" in restart_note(field("hide_text")).lower()
    assert all(restart_note(item) for item in SCHEMA)


def test_wd5_a_page_note_tells_which_settings_restart() -> None:
    form = SettingsForm(two_lanes())
    segmentation = [item for item in form.fields if item.group == "Segmentation"]
    output = [item for item in form.fields if item.group == "Output"]
    assert "restart" in page_note("Segmentation", segmentation).lower()
    assert "clear" in page_note("Segmentation", segmentation).lower()
    mixed = page_note("Output", output).lower()
    assert "immediately" in mixed
    assert "restart" in mixed
    only_live = [item for item in output if item.key == "hide_text"]
    assert "restart" not in page_note("Output", only_live).lower()


def test_wd5_errors_are_placed_next_to_their_field_and_the_rest_is_general() -> None:
    form = SettingsForm(two_lanes())
    error = SettingsValidationError(
        {
            "segmenter.silence_ms": "must be 450-600 ms, got 100",
            "segmenter.tail_ms": "tail too long",
            GENERAL_ERROR_KEY: "something about two modules",
            "unknown.key": "stray",
        }
    )
    placed = form.place_errors(error.errors)
    assert placed.by_field == {
        "segmenter.silence_ms": "must be 450-600 ms, got 100",
        "segmenter.tail_ms": "tail too long",
    }
    assert placed.general == ("something about two modules", "unknown.key: stray")
    assert placed.advanced_fields == ()


def test_wd5_an_error_in_an_advanced_field_is_reported_so_the_tab_can_open_it() -> None:
    form = SettingsForm(two_lanes())
    placed = form.place_errors({"segmenter.speech_threshold": "bad", "segmenter.silence_ms": "x"})
    assert placed.advanced_fields == ("segmenter.speech_threshold",)


def test_wd5_real_validation_errors_reach_their_fields() -> None:
    settings = two_lanes()
    form = SettingsForm(settings)
    form.set("segmenter.silence_ms", "100")
    form.set("segmenter.exit_threshold", "0.9")
    values, errors = form.collect()
    assert errors == {}
    with pytest.raises(SettingsValidationError) as caught:
        build_settings(settings, values)
    placed = form.place_errors(caught.value.errors)
    assert "segmenter.silence_ms" in placed.by_field
    assert "segmenter.exit_threshold" in placed.by_field
    assert placed.general == ()


def test_wd5_the_module_is_importable_without_a_display() -> None:
    assert form_module.SettingsForm is SettingsForm


def test_wd5_rebasing_keeps_the_users_edits_and_updates_the_rest() -> None:
    settings = two_lanes()
    form = SettingsForm(settings)
    form.set("segmenter.silence_ms", "520")  # an edit in progress
    moved = build_settings(settings, {"capture.block_ms": 20, "segmenter.silence_ms": 600})
    form.rebase(moved)
    assert form.value("segmenter.silence_ms") == "520"  # kept
    assert form.value("capture.block_ms") == "20"  # followed the applied settings
    assert form.edited_keys() == ("segmenter.silence_ms",)


def test_wd5_rebasing_to_other_lanes_drops_edits_of_fields_that_no_longer_exist() -> None:
    settings = two_lanes()
    form = SettingsForm(settings)
    form.set("routes.fr.radio_offset_ms", "50")
    form.set("segmenter.silence_ms", "520")
    form.rebase(parse_settings(["--input", "V380", "--route", "de=Realtek", "--targets", "de"]))
    assert "routes.fr.radio_offset_ms" not in {item.key for item in form.fields}
    assert form.edited_keys() == ("segmenter.silence_ms",)


def test_wd5_the_field_keys_change_only_with_the_lanes() -> None:
    settings = two_lanes()
    form = SettingsForm(settings)
    same = build_settings(settings, {"capture.block_ms": 20})
    assert form.same_fields_as(same)
    assert not form.same_fields_as(
        parse_settings(["--input", "V380", "--route", "de=Realtek", "--targets", "de"])
    )
    swapped = parse_settings(["--input", "V380", "--route", "fr=Realtek", "--targets", "fr"])
    assert not form.same_fields_as(swapped)
