"""The Tk-free description of every user-tunable setting, and its validation (SC1-SC10)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

import pipeline.ui.settings_schema as schema_module
from pipeline.capture import CaptureConfig
from pipeline.config import LaneRoute, Settings
from pipeline.output import OutputConfig
from pipeline.transcribe import TranscribeConfig
from pipeline.translate import TranslateConfig
from pipeline.ui.settings_schema import (
    GENERAL_ERROR_KEY,
    GROUPS,
    NOT_EXPOSED,
    SCHEMA,
    ApplyMode,
    Field,
    SettingsSchemaError,
    SettingsValidationError,
    build_settings,
    construct_settings,
    diff,
    fields_by_group,
    keys_named_in,
    needs_restart,
    schema_for,
    values_from_settings,
)
from pipeline.vad import SegmenterConfig

pytestmark = pytest.mark.unit


def base_settings() -> Settings:
    return Settings(
        capture=CaptureConfig(device_query="mic"),
        routes=(
            LaneRoute("de", OutputConfig(device_query="speakers")),
            LaneRoute("fr", OutputConfig(device_query="headset"), radio_offset_ms=120),
        ),
    )


def single_route_settings() -> Settings:
    return Settings(
        capture=CaptureConfig(device_query="mic"),
        routes=(LaneRoute("de", OutputConfig(device_query="speakers")),),
        targets=("de",),
    )


ALL_FIELDS = schema_for(base_settings())
BY_KEY = {item.key: item for item in ALL_FIELDS}
NUMERIC = [item for item in ALL_FIELDS if item.kind in ("int", "float")]

# Fields whose legal range depends on another field: the bounds test sets the other one first.
CONTEXT: dict[str, dict[str, object]] = {
    "segmenter.tail_ms": {"segmenter.silence_ms": 600},
    "segmenter.cut_search_ms": {"segmenter.max_utterance_ms": 30_000},
    "segmenter.speech_threshold": {"segmenter.exit_threshold": 0.05},
    "segmenter.exit_threshold": {"segmenter.speech_threshold": 1.0},
    "segmenter.turn_threshold": {"segmenter.soft_cut_threshold": 0.95},
}


def outside(item: Field, *, below: bool) -> float:
    """One unit beyond a bound: one for whole numbers, one step for fractions."""
    unit = 1 if item.kind == "int" else (item.step or 0.0)
    bound = item.minimum if below else item.maximum
    assert bound is not None
    return bound - unit if below else bound + unit


def with_context(item: Field, value: object) -> dict[str, object]:
    return {**CONTEXT.get(item.key, {}), item.key: value}


def errors_of(values: Mapping[str, object], base: Settings | None = None) -> dict[str, str]:
    with pytest.raises(SettingsValidationError) as caught:
        build_settings(base or base_settings(), values)
    return caught.value.errors


def default_of(key: str) -> object:
    """The value the real dataclass gives a field that is not set."""
    section, _, attr = key.partition(".")
    if section == "routes":
        return LaneRoute("de", OutputConfig(device_query="x")).radio_offset_ms
    owners: dict[str, object] = {
        "segmenter": SegmenterConfig(),
        "transcribe": TranscribeConfig(),
        "capture": CaptureConfig(device_query="x"),
        "output": OutputConfig(device_query="x"),
        "translate": TranslateConfig(),
    }
    if attr:
        return getattr(owners[section], attr)
    return getattr(base_settings(), section)


# --- SC1 round trip -------------------------------------------------------------------------


def test_sc1_values_round_trip_to_the_same_settings() -> None:
    base = base_settings()
    assert build_settings(base, values_from_settings(base)) == base


def test_sc1_round_trip_with_one_route_and_odd_values() -> None:
    base = single_route_settings()
    values = values_from_settings(base)
    assert not any(key.startswith("routes.fr") for key in values)
    assert build_settings(base, values) == base


def test_sc1_values_cover_every_schema_field() -> None:
    base = base_settings()
    assert set(values_from_settings(base)) == {item.key for item in schema_for(base)}


def test_sc1_changed_values_reach_the_right_dataclass() -> None:
    base = base_settings()
    new = build_settings(
        base,
        {
            "segmenter.max_utterance_ms": 20_000,
            "transcribe.model": "tiny.en",
            "capture.block_ms": 20,
            "output.prebuffer_ms": 40,
            "routes.fr.radio_offset_ms": 200,
            "translate.beam_size": 4,
            "turn_threads": 2,
            "strict": True,
            "hide_text": True,
        },
    )
    assert new.segmenter.max_utterance_ms == 20_000
    assert new.transcribe.model == "tiny.en"
    assert new.capture.block_ms == 20
    assert [route.output.prebuffer_ms for route in new.routes] == [40, 40]
    assert [route.radio_offset_ms for route in new.routes] == [0, 200]
    assert new.translate.beam_size == 4
    assert (new.turn_threads, new.strict, new.hide_text) == (2, True, True)
    assert new.capture.device_query == "mic"
    assert [route.output.device_query for route in new.routes] == ["speakers", "headset"]


def test_sc1_a_partial_mapping_leaves_the_rest_alone() -> None:
    base = base_settings()
    assert build_settings(base, {}) == base


# --- SC2 defaults ---------------------------------------------------------------------------


@pytest.mark.parametrize("item", ALL_FIELDS, ids=lambda item: item.key)
def test_sc2_defaults_equal_the_dataclass_defaults(item: Field) -> None:
    assert item.default == default_of(item.key)


def test_sc2_the_silence_default_is_the_default_of_both_configs() -> None:
    assert BY_KEY["segmenter.silence_ms"].default == SegmenterConfig().silence_ms
    assert BY_KEY["segmenter.silence_ms"].default == CaptureConfig(device_query="x").silence_ms


# --- SC3 bounds against the real constructors -----------------------------------------------


@pytest.mark.parametrize("item", NUMERIC, ids=lambda item: item.key)
def test_sc3_bounds_are_accepted_by_the_real_constructors(item: Field) -> None:
    base = base_settings()
    for bound in (item.minimum, item.maximum, item.default):
        construct_settings(base, with_context(item, bound))
        build_settings(base, with_context(item, bound))


@pytest.mark.parametrize("item", NUMERIC, ids=lambda item: item.key)
def test_sc3_the_real_constructors_reject_what_is_beyond_a_hard_bound(item: Field) -> None:
    base = base_settings()
    for below, hard in ((True, item.hard_minimum), (False, item.hard_maximum)):
        if not hard:
            continue
        with pytest.raises(SettingsValidationError):
            construct_settings(base, with_context(item, outside(item, below=below)))


@pytest.mark.parametrize("item", NUMERIC, ids=lambda item: item.key)
def test_sc3_the_schema_rejects_what_is_beyond_any_bound(item: Field) -> None:
    for below in (True, False):
        errors = errors_of(with_context(item, outside(item, below=below)))
        assert item.key in errors


def test_sc3_a_soft_limit_is_not_a_real_one() -> None:
    # The decode accepts up to 32 threads; the command line stops at 8 and so does the window.
    item = BY_KEY["transcribe.threads"]
    assert not item.hard_maximum
    assert item.maximum == 8
    construct_settings(base_settings(), {item.key: 9})
    assert item.key in errors_of({item.key: 9})


def test_sc3_soft_cut_can_be_switched_off_with_zero() -> None:
    new = build_settings(base_settings(), {"segmenter.soft_cut_ms": 0})
    assert new.segmenter.soft_cut_ms == 0
    assert "segmenter.soft_cut_ms" in errors_of({"segmenter.soft_cut_ms": 1_000})
    assert "segmenter.soft_cut_ms" in errors_of({"segmenter.soft_cut_ms": 2_999})


@pytest.mark.parametrize(
    "item", [item for item in ALL_FIELDS if item.kind == "choice"], ids=lambda item: item.key
)
def test_sc3_every_choice_is_accepted_and_others_are_not(item: Field) -> None:
    for choice in item.choices:
        build_settings(base_settings(), {item.key: choice})
    assert item.key in errors_of({item.key: "nonsense"})


def test_sc3_choices_come_from_the_real_modules() -> None:
    assert set(BY_KEY["transcribe.model"].choices) == {"tiny.en", "base.en"}
    assert set(BY_KEY["segmenter.turn_mode"].choices) == {"fixed", "smart"}
    assert set(BY_KEY["capture.downmix"].choices) == {"auto", "mean"}


# --- SC4 cross-field rules ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "keys"),
    [
        (
            {"segmenter.exit_threshold": 0.8, "segmenter.speech_threshold": 0.6},
            {"segmenter.exit_threshold", "segmenter.speech_threshold"},
        ),
        (
            {"segmenter.soft_cut_threshold": 0.6, "segmenter.turn_threshold": 0.8},
            {"segmenter.soft_cut_threshold", "segmenter.turn_threshold"},
        ),
        (
            {"segmenter.tail_ms": 500, "segmenter.silence_ms": 450},
            {"segmenter.tail_ms", "segmenter.silence_ms"},
        ),
        (
            {"segmenter.cut_search_ms": 4_000, "segmenter.max_utterance_ms": 5_000},
            {"segmenter.cut_search_ms", "segmenter.max_utterance_ms"},
        ),
    ],
)
def test_sc4_cross_field_errors_point_at_the_fields_involved(
    values: dict[str, object], keys: set[str]
) -> None:
    errors = errors_of(values)
    assert set(errors) == keys
    assert all(message for message in errors.values())


def test_sc4_a_change_of_one_field_can_break_a_rule_with_an_untouched_one() -> None:
    # Only the threshold is edited; the speech threshold is still the default 0.5.
    errors = errors_of({"segmenter.exit_threshold": 0.9})
    assert set(errors) == {"segmenter.exit_threshold", "segmenter.speech_threshold"}


def test_sc4_the_rules_hold_at_their_edges() -> None:
    build_settings(base_settings(), {"segmenter.tail_ms": 450})
    build_settings(base_settings(), {"segmenter.exit_threshold": 0.5})
    build_settings(base_settings(), {"segmenter.soft_cut_threshold": 0.5})
    build_settings(
        base_settings(), {"segmenter.cut_search_ms": 2_500, "segmenter.max_utterance_ms": 5_000}
    )


def test_sc4_several_problems_are_reported_together() -> None:
    errors = errors_of(
        {
            "segmenter.silence_ms": 100,
            "capture.block_ms": 1,
            "transcribe.merge_max_ms": 5,
            "turn_threads": 9,
        }
    )
    assert set(errors) == {
        "segmenter.silence_ms",
        "capture.block_ms",
        "transcribe.merge_max_ms",
        "turn_threads",
    }


def test_sc4_the_message_of_the_real_validator_is_mapped_back_to_its_fields() -> None:
    # The real constructors are the authority: reached directly, they name the same fields.
    with pytest.raises(SettingsValidationError) as caught:
        construct_settings(base_settings(), {"segmenter.tail_ms": 500, "segmenter.silence_ms": 450})
    assert set(caught.value.errors) == {"segmenter.tail_ms", "segmenter.silence_ms"}
    with pytest.raises(SettingsValidationError) as caught:
        construct_settings(base_settings(), {"transcribe.merge_max_ms": 5})
    assert set(caught.value.errors) == {"transcribe.merge_max_ms"}
    with pytest.raises(SettingsValidationError) as caught:
        construct_settings(base_settings(), {"output.block_ms": 1})
    assert set(caught.value.errors) == {"output.block_ms"}
    with pytest.raises(SettingsValidationError) as caught:
        construct_settings(base_settings(), {"routes.de.radio_offset_ms": 9_999})
    assert set(caught.value.errors) == {"routes.de.radio_offset_ms"}
    with pytest.raises(SettingsValidationError) as caught:
        construct_settings(base_settings(), {"turn_threads": 3})
    assert set(caught.value.errors) == {"turn_threads"}


def test_sc4_an_error_that_names_no_field_is_general() -> None:
    assert keys_named_in("something odd happened", "segmenter") == {
        GENERAL_ERROR_KEY: "something odd happened"
    }


def test_sc4_a_capture_silence_error_belongs_to_the_one_silence_field() -> None:
    with pytest.raises(SettingsValidationError) as caught:
        construct_settings(base_settings(), {"capture.silence_ms": 0})
    assert set(caught.value.errors) == {"segmenter.silence_ms"}
    with pytest.raises(SettingsValidationError) as caught:
        construct_settings(base_settings(), {"capture.silence_ms": 500})
    assert set(caught.value.errors) == {"segmenter.silence_ms"}


def test_sc4_the_exception_message_lists_every_key() -> None:
    with pytest.raises(SettingsValidationError) as caught:
        build_settings(base_settings(), {"capture.block_ms": 1, "turn_threads": 9})
    text = str(caught.value)
    assert "capture.block_ms" in text
    assert "turn_threads" in text
    assert isinstance(caught.value, ValueError)


# --- SC5 one silence_ms ---------------------------------------------------------------------


def test_sc5_silence_ms_sets_the_segmenter_and_the_capture_report() -> None:
    new = build_settings(base_settings(), {"segmenter.silence_ms": 550})
    assert new.segmenter.silence_ms == 550
    assert new.capture.silence_ms == 550


def test_sc5_there_is_no_separate_capture_silence_field() -> None:
    assert "capture.silence_ms" not in BY_KEY
    assert "capture.silence_ms" in NOT_EXPOSED
    errors = errors_of({"capture.silence_ms": 500})
    assert "capture.silence_ms" in errors


def test_sc5_the_value_read_back_is_the_segmenter_one() -> None:
    new = build_settings(base_settings(), {"segmenter.silence_ms": 600})
    assert values_from_settings(new)["segmenter.silence_ms"] == 600


# --- SC6 schema shape -----------------------------------------------------------------------


def test_sc6_every_key_is_unique() -> None:
    keys = [item.key for item in ALL_FIELDS]
    assert len(keys) == len(set(keys))


def test_sc6_every_group_has_fields_and_every_field_has_a_known_group() -> None:
    grouped = fields_by_group()
    assert tuple(grouped) == GROUPS
    assert GROUPS == (
        "Segmentation",
        "Turn detection",
        "Whisper",
        "Capture",
        "Output",
        "Advanced",
    )
    assert all(grouped[group] for group in GROUPS)
    assert {item.group for item in SCHEMA} <= set(GROUPS)
    assert sum(len(items) for items in grouped.values()) == len(SCHEMA)


def test_sc6_fields_by_group_adds_the_routes_of_a_settings_object() -> None:
    plain = fields_by_group()
    with_routes = fields_by_group(base_settings())
    keys = [item.key for item in with_routes["Output"]]
    assert "routes.de.radio_offset_ms" in keys
    assert "routes.fr.radio_offset_ms" in keys
    assert len(with_routes["Output"]) == len(plain["Output"]) + 2
    single = [item.key for item in fields_by_group(single_route_settings())["Output"]]
    assert "routes.de.radio_offset_ms" in single
    assert "routes.fr.radio_offset_ms" not in single


def test_sc6_every_field_is_documented() -> None:
    for item in ALL_FIELDS:
        assert item.label.strip()
        assert item.help.strip().endswith(".")
        assert item.help.count(". ") == 0, f"{item.key}: help is more than one sentence"
        assert len(item.label) <= 32


def test_sc6_the_advanced_group_is_advanced_and_the_flags_are_set_deliberately() -> None:
    for item in ALL_FIELDS:
        if item.group == "Advanced":
            assert item.advanced
    assert not BY_KEY["segmenter.silence_ms"].advanced
    assert BY_KEY["segmenter.speech_threshold"].advanced


def test_sc6_every_tunable_field_of_the_configs_is_in_the_schema_or_listed_as_not_exposed() -> None:
    covered = set(BY_KEY)
    for prefix, config in (
        ("segmenter", SegmenterConfig),
        ("transcribe", TranscribeConfig),
        ("capture", CaptureConfig),
        ("output", OutputConfig),
        ("translate", TranslateConfig),
    ):
        for item in dataclass_fields(config):
            key = f"{prefix}.{item.name}"
            assert key in covered or key in NOT_EXPOSED, f"{key} is neither exposed nor excluded"
    for name in ("turn_threads", "strict", "hide_text"):
        assert name in covered
    # Hardcoded in transcribe.py and owned by the device panel: never in the schema.
    for key in ("transcribe.language", "capture.device_query", "output.device_query"):
        assert key in NOT_EXPOSED
        assert key not in covered
    assert not covered & NOT_EXPOSED


def test_sc6_a_field_that_contradicts_itself_is_refused() -> None:
    good = BY_KEY["capture.block_ms"]
    with pytest.raises(SettingsSchemaError):
        Field(
            key="x.y",
            label="X",
            help="Does a thing.",
            kind="int",
            default=1,
            group="Capture",
            minimum=5,
            maximum=2,
            step=1,
        )
    with pytest.raises(SettingsSchemaError):
        Field(
            key="x.y",
            label="X",
            help="Does a thing.",
            kind="int",
            default=1000,
            group="Capture",
            minimum=1,
            maximum=2,
            step=1,
        )
    with pytest.raises(SettingsSchemaError):
        Field(
            key="x.y",
            label="X",
            help="Does a thing.",
            kind="choice",
            default="c",
            group="Capture",
            choices=("a", "b"),
        )
    with pytest.raises(SettingsSchemaError):
        Field(
            key="x.y",
            label="X",
            help="Does a thing.",
            kind="int",
            default=1,
            group="Nowhere",  # type: ignore[arg-type]
            minimum=0,
            maximum=2,
            step=1,
        )
    assert good.apply is ApplyMode.RESTART


def test_sc6_fields_are_frozen() -> None:
    with pytest.raises(AttributeError):
        BY_KEY["capture.block_ms"].label = "x"  # type: ignore[misc]


# --- SC7 apply modes, diff, needs_restart ---------------------------------------------------


def test_sc7_only_hide_text_applies_live() -> None:
    live = {item.key for item in ALL_FIELDS if item.apply is ApplyMode.LIVE}
    assert live == {"hide_text"}


def test_sc7_diff_lists_changed_fields_only() -> None:
    old = base_settings()
    new = build_settings(old, {"segmenter.pre_roll_ms": 200, "strict": True})
    assert diff(old, new) == {
        "segmenter.pre_roll_ms": (100, 200),
        "strict": (False, True),
    }
    assert diff(old, old) == {}


def test_sc7_diff_includes_the_silence_sync_once() -> None:
    old = base_settings()
    new = build_settings(old, {"segmenter.silence_ms": 500})
    assert diff(old, new) == {"segmenter.silence_ms": (450, 500)}


def test_sc7_a_live_change_needs_no_restart() -> None:
    old = base_settings()
    new = build_settings(old, {"hide_text": True})
    assert diff(old, new) == {"hide_text": (False, True)}
    assert not needs_restart(old, new)
    assert not needs_restart(old, old)


def test_sc7_any_restart_field_needs_a_restart_even_next_to_a_live_one() -> None:
    old = base_settings()
    assert needs_restart(old, build_settings(old, {"transcribe.beam_size": 2}))
    assert needs_restart(old, build_settings(old, {"hide_text": True, "capture.block_ms": 20}))
    assert needs_restart(old, build_settings(old, {"routes.de.radio_offset_ms": 50}))


def test_sc7_a_device_name_change_is_not_a_schema_change() -> None:
    old = base_settings()
    new = Settings(
        capture=CaptureConfig(device_query="other mic"),
        routes=old.routes,
    )
    assert diff(old, new) == {}
    assert not needs_restart(old, new)


# --- SC8 types ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("segmenter.silence_ms", "500"),
        ("segmenter.silence_ms", 500.5),
        ("segmenter.silence_ms", True),
        ("segmenter.silence_ms", None),
        ("segmenter.turn_threshold", "0.5"),
        ("segmenter.turn_threshold", True),
        ("segmenter.turn_threshold", float("nan")),
        ("segmenter.turn_threshold", float("inf")),
        ("strict", 1),
        ("hide_text", "yes"),
        ("transcribe.model", 3),
    ],
)
def test_sc8_values_of_the_wrong_type_are_rejected(key: str, value: object) -> None:
    assert key in errors_of({key: value})


def test_sc8_an_integer_is_fine_for_a_fractional_field() -> None:
    new = build_settings(base_settings(), {"segmenter.speech_threshold": 1})
    assert new.segmenter.speech_threshold == 1.0


def test_sc8_an_unknown_key_is_an_error_not_ignored() -> None:
    assert errors_of({"segmenter.nonsense": 1}) == {"segmenter.nonsense": "unknown setting"}
    assert "transcribe.language" in errors_of({"transcribe.language": "de"})
    assert "routes.xx.radio_offset_ms" in errors_of({"routes.xx.radio_offset_ms": 0})


def test_sc8_the_error_message_names_the_range() -> None:
    message = errors_of({"segmenter.silence_ms": 100})["segmenter.silence_ms"]
    assert "450" in message
    assert "600" in message


# --- SC9 routes -----------------------------------------------------------------------------


def test_sc9_radio_offset_is_per_route() -> None:
    new = build_settings(base_settings(), {"routes.de.radio_offset_ms": 80})
    assert [route.radio_offset_ms for route in new.routes] == [80, 120]
    assert values_from_settings(new)["routes.fr.radio_offset_ms"] == 120


def test_sc9_a_route_that_does_not_exist_has_no_field() -> None:
    assert "routes.fr.radio_offset_ms" in errors_of(
        {"routes.fr.radio_offset_ms": 10}, single_route_settings()
    )


def test_sc9_output_settings_apply_to_every_route() -> None:
    new = build_settings(base_settings(), {"output.ring_seconds": 5.0})
    assert [route.output.ring_seconds for route in new.routes] == [5.0, 5.0]


# --- SC10 purity ----------------------------------------------------------------------------


def test_sc10_the_module_does_not_import_tk() -> None:
    source = Path(schema_module.__file__ or "").read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert imports
    assert not any("tkinter" in line or "tk" in line.split() for line in imports)


def test_sc10_building_does_not_mutate_the_base() -> None:
    base = base_settings()
    before = values_from_settings(base)
    build_settings(base, {"segmenter.pre_roll_ms": 500})
    assert values_from_settings(base) == before
