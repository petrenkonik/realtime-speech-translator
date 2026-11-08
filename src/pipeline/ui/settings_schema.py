"""Every user-tunable setting as data, plus the validation behind the Apply button.

This module is free of Tk (and of any other toolkit) so that the settings tab can be
generated from it and every rule can be tested without a display. ``SCHEMA`` lists one
``Field`` per setting: its label, help text, kind, range, step, group and whether a change
needs the pipeline to restart (the configs are frozen, so almost everything does).

``build_settings`` applies a mapping of ``key -> value`` to a base ``Settings`` through the
real dataclass constructors, so their validators stay the authority. Before that it checks
types and ranges and the rules that involve two fields, so the window can show an error next
to every field involved instead of the first message of the first failing constructor.

Ranges come from the validation in the config classes. Where a class exports a constant
(the silence range, the thread ranges, the radio offset limit, the model names) the schema
reads it; the other limits are written out here and ``tests/test_settings_schema.py`` proves
them against the real constructors. A field whose limit exists only in this window (an
upper bound that the class does not enforce) says so with ``hard_minimum`` / ``hard_maximum``.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, TypeVar, get_args

from pipeline.capture import CaptureConfig, CaptureConfigError
from pipeline.config import (
    _MAX_RADIO_OFFSET_MS,
    _TURN_THREADS_RANGE,
    _WHISPER_THREADS_RANGE,
    ConfigError,
    LaneRoute,
    Settings,
)
from pipeline.lanes import LaneConfig, LaneConfigError
from pipeline.output import OutputConfig, OutputConfigError
from pipeline.pcm import DownmixMode
from pipeline.transcribe import WHISPER_MODELS, TranscribeConfig, TranscribeConfigError
from pipeline.translate import TranslateConfig, TranslateConfigError
from pipeline.vad import _SILENCE_MS_RANGE, FRAME_MS, SegmenterConfig, TurnMode, VadConfigError

FieldKind = Literal["int", "float", "choice", "bool"]
Unit = Literal["ms", "s", "none"]
Group = Literal["Segmentation", "Turn detection", "Whisper", "Capture", "Output", "Advanced"]

GROUPS: tuple[Group, ...] = (
    "Segmentation",
    "Turn detection",
    "Whisper",
    "Capture",
    "Output",
    "Advanced",
)

GENERAL_ERROR_KEY = "*"
"""Key of an error that names no single field."""

NOT_EXPOSED: frozenset[str] = frozenset(
    {
        # Devices have their own panel; the query is a name substring, not a number to tune.
        "capture.device_query",
        "output.device_query",
        # One input, one source language, hardcoded in the whisper adapter.
        "transcribe.language",
        # Set by ``silence_ms`` (config.py demands that the two are equal).
        "capture.silence_ms",
        # Internal polling cadence and safety limits; nothing a demo operator should touch.
        "capture.poll_ms",
        "output.poll_ms",
        "translate.max_decoding_length",
    }
)
"""Config fields that deliberately have no ``Field`` (a consistency test checks the rest)."""

_SILENCE_MIN_MS, _SILENCE_MAX_MS = _SILENCE_MS_RANGE
_TURN_THREADS_MIN, _TURN_THREADS_MAX = _TURN_THREADS_RANGE
_WHISPER_THREADS_MIN, _WHISPER_THREADS_MAX = _WHISPER_THREADS_RANGE
_MAX_UTTERANCE_MAX_MS = 30_000


class SettingsSchemaError(ValueError):
    """A ``Field`` contradicts itself; raised when the schema is built, never at run time."""


class SettingsValidationError(ValueError):
    """Some values cannot be applied.

    Attributes:
        errors: One message per field key, ready to show next to the field. The key
            ``GENERAL_ERROR_KEY`` carries a message that names no single field.
    """

    def __init__(self, errors: Mapping[str, str]) -> None:
        """Keep the messages and build one readable summary line."""
        self.errors: dict[str, str] = dict(errors)
        super().__init__("; ".join(f"{key}: {message}" for key, message in self.errors.items()))


class ApplyMode(Enum):
    """What it takes for a changed setting to come into effect."""

    LIVE = "live"
    RESTART = "restart"


@dataclass(frozen=True, slots=True)
class Field:
    """One setting of the settings tab.

    Attributes:
        key: Dotted path such as ``segmenter.silence_ms``; ``output.*`` fields apply to every
            route and ``routes.<target>.radio_offset_ms`` to one.
        label: Short English label of the control.
        help: One sentence: what it does and what it costs.
        kind: ``int``, ``float``, ``choice`` or ``bool``.
        default: The value of the real dataclass default.
        group: Tab section the control belongs to.
        apply: Whether a change takes effect live or needs a pipeline restart.
        advanced: Hidden unless the user asks for advanced settings.
        minimum: Lowest value of a number (inclusive).
        maximum: Highest value of a number (inclusive).
        step: Increment of the spin box; also the unit of the "just outside" tests.
        choices: Allowed values of a ``choice``.
        unit: ``ms``, ``s`` or ``none``.
        hard_minimum: The real constructor rejects values below ``minimum``.
        hard_maximum: The real constructor rejects values above ``maximum``; ``False`` when
            the bound is a limit of the window only.
        off_value: A number outside the range that switches the feature off (0 for soft cuts).
    """

    key: str
    label: str
    help: str
    kind: FieldKind
    default: object
    group: Group
    apply: ApplyMode = ApplyMode.RESTART
    advanced: bool = False
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    unit: Unit = "none"
    hard_minimum: bool = True
    hard_maximum: bool = True
    off_value: int | None = None

    def __post_init__(self) -> None:
        """Refuse a field that contradicts itself."""
        _need(bool(self.key.strip()), "a field needs a key")
        where = self.key
        _need(bool(self.label.strip()) and bool(self.help.strip()), f"{where}: label and help")
        _need(self.group in GROUPS, f"{where}: unknown group {self.group!r}")
        if self.kind in ("int", "float"):
            self._check_number()
        elif self.kind == "choice":
            _need(len(self.choices) >= 2, f"{where}: a choice needs at least two options")
            _need(self.default in self.choices, f"{where}: default is not one of the choices")
        else:
            _need(isinstance(self.default, bool), f"{where}: default of a bool must be a bool")
        if self.kind != "choice":
            _need(not self.choices, f"{where}: only a choice has choices")

    def _check_number(self) -> None:
        where = self.key
        low, high, step = self.minimum, self.maximum, self.step
        _need(low is not None and high is not None, f"{where}: a number needs minimum, maximum")
        _need(step is not None and step > 0, f"{where}: a number needs a positive step")
        assert low is not None
        assert high is not None
        _need(low < high, f"{where}: minimum must be below maximum")
        problem = _number_problem(self.kind, self.default)
        _need(problem is None, f"{where}: default {problem}")
        _need(_in_range(self, self.default), f"{where}: default is outside the range")


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise SettingsSchemaError(message)


def _fmt(number: float) -> str:
    return f"{number:g}"


def _unit_suffix(item: Field) -> str:
    return "" if item.unit == "none" else f" {item.unit}"


def _number_problem(kind: FieldKind, value: object) -> str | None:
    if isinstance(value, bool):
        return "must be a number, not a yes/no value"
    if kind == "int":
        if not isinstance(value, int):
            return "must be a whole number"
        return None
    if not isinstance(value, (int, float)):
        return "must be a number"
    if not math.isfinite(value):
        return "must be a finite number"
    return None


def _in_range(item: Field, value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if item.off_value is not None and value == item.off_value:
        return True
    assert item.minimum is not None
    assert item.maximum is not None
    return item.minimum <= value <= item.maximum


def _range_text(item: Field) -> str:
    assert item.minimum is not None
    assert item.maximum is not None
    span = f"{_fmt(item.minimum)}-{_fmt(item.maximum)}{_unit_suffix(item)}"
    if item.off_value is None:
        return span
    return f"{item.off_value} (off) or {span}"


# --- building the schema ----------------------------------------------------------------------


def _owner(key: str) -> type:
    section = key.partition(".")[0]
    owners: dict[str, type] = {
        "segmenter": SegmenterConfig,
        "transcribe": TranscribeConfig,
        "capture": CaptureConfig,
        "output": OutputConfig,
        "translate": TranslateConfig,
        "routes": LaneRoute,
    }
    return owners.get(section, Settings)


def _default_of(key: str) -> object:
    """The dataclass default behind ``key``, so no default is typed in twice."""
    parts = key.split(".")
    attr = parts[-1]
    for item in dataclasses.fields(_owner(key)):
        if item.name == attr:
            return item.default
    raise SettingsSchemaError(f"{key}: the config class has no field {attr!r}")


def _number(
    kind: FieldKind,
    key: str,
    label: str,
    help_: str,
    group: Group,
    bounds: tuple[float, float, float],
    unit: Unit,
    advanced: bool,
    soft_limits: tuple[bool, bool],
    off_value: int | None,
) -> Field:
    minimum, maximum, step = bounds
    return Field(
        key=key,
        label=label,
        help=help_,
        kind=kind,
        default=_default_of(key),
        group=group,
        advanced=advanced or group == "Advanced",
        minimum=minimum,
        maximum=maximum,
        step=step,
        unit=unit,
        hard_minimum=not soft_limits[0],
        hard_maximum=not soft_limits[1],
        off_value=off_value,
    )


def _int(
    key: str,
    label: str,
    help_: str,
    group: Group,
    bounds: tuple[int, int, int],
    *,
    unit: Unit = "none",
    advanced: bool = False,
    soft_minimum: bool = False,
    soft_maximum: bool = False,
    off_value: int | None = None,
) -> Field:
    """A whole-number field; ``soft_*`` marks a bound that only this window enforces."""
    return _number(
        "int",
        key,
        label,
        help_,
        group,
        bounds,
        unit,
        advanced,
        (soft_minimum, soft_maximum),
        off_value,
    )


def _float(
    key: str,
    label: str,
    help_: str,
    group: Group,
    bounds: tuple[float, float, float],
    *,
    unit: Unit = "none",
    advanced: bool = False,
    soft_minimum: bool = False,
    soft_maximum: bool = False,
) -> Field:
    """A fractional field; ``soft_*`` marks a bound that only this window enforces."""
    return _number(
        "float",
        key,
        label,
        help_,
        group,
        bounds,
        unit,
        advanced,
        (soft_minimum, soft_maximum),
        None,
    )


def _choice(
    key: str,
    label: str,
    help_: str,
    group: Group,
    choices: Iterable[str],
    *,
    advanced: bool = False,
) -> Field:
    return Field(
        key=key,
        label=label,
        help=help_,
        kind="choice",
        default=_default_of(key),
        group=group,
        advanced=advanced or group == "Advanced",
        choices=tuple(choices),
    )


def _flag(
    key: str,
    label: str,
    help_: str,
    group: Group,
    *,
    apply: ApplyMode = ApplyMode.RESTART,
    advanced: bool = False,
) -> Field:
    return Field(
        key=key,
        label=label,
        help=help_,
        kind="bool",
        default=_default_of(key),
        group=group,
        apply=apply,
        advanced=advanced or group == "Advanced",
    )


_SEG: Group = "Segmentation"
_TURN: Group = "Turn detection"
_WHISPER: Group = "Whisper"
_CAPTURE: Group = "Capture"
_OUTPUT: Group = "Output"
_ADVANCED: Group = "Advanced"

SCHEMA: tuple[Field, ...] = (
    # --- Segmentation ---
    _int(
        "segmenter.silence_ms",
        "Pause that ends a phrase",
        "How long a pause must last before the phrase is sent on; shorter is faster but "
        "cuts sentences at breaths.",
        _SEG,
        (_SILENCE_MIN_MS, _SILENCE_MAX_MS, 10),
        unit="ms",
    ),
    _int(
        "segmenter.min_speech_ms",
        "Shortest phrase",
        "Phrases with less speech than this are treated as noise and dropped; higher "
        "ignores clicks but loses one-word answers.",
        _SEG,
        (FRAME_MS, 2_000, 16),
        unit="ms",
    ),
    _int(
        "segmenter.pre_roll_ms",
        "Audio kept before speech",
        "Audio kept before the first speech frame so soft word starts survive; more "
        "protects onsets but adds a little to every decode.",
        _SEG,
        (0, 2_000, 10),
        unit="ms",
    ),
    _int(
        "segmenter.tail_ms",
        "Silence kept after speech",
        "Trailing silence kept in a phrase (at most the pause setting); more helps whisper "
        "finish the last word but adds audio to decode.",
        _SEG,
        (FRAME_MS, _SILENCE_MAX_MS, 8),
        unit="ms",
    ),
    _int(
        "segmenter.max_utterance_ms",
        "Longest phrase",
        "Longest chunk before speech is cut by force; longer keeps whole sentences "
        "together but delays the translation of a monologue.",
        _SEG,
        (5_000, _MAX_UTTERANCE_MAX_MS, 500),
        unit="ms",
    ),
    _int(
        "segmenter.cut_search_ms",
        "Search window for a forced cut",
        "A forced cut falls on the quietest moment of this window (at most half the "
        "longest phrase); wider finds a better spot but cuts earlier in the phrase.",
        _SEG,
        (500, _MAX_UTTERANCE_MAX_MS // 2, 100),
        unit="ms",
    ),
    _float(
        "segmenter.speech_threshold",
        "Speech probability",
        "Voice probability from which a frame counts as speech; higher ignores noise but "
        "can miss quiet speakers.",
        _SEG,
        (0.05, 1.0, 0.05),
        advanced=True,
    ),
    _float(
        "segmenter.exit_threshold",
        "Silence probability",
        "Voice probability below which the pause clock starts (not above the speech "
        "probability); higher ends phrases sooner but can split slow speech.",
        _SEG,
        (0.05, 1.0, 0.05),
        advanced=True,
    ),
    # --- Turn detection ---
    _choice(
        "segmenter.turn_mode",
        "Turn detection",
        "Fixed ends a phrase by silence only; smart also asks a model whether the speaker "
        "has finished, which can end it earlier but costs CPU.",
        _TURN,
        get_args(TurnMode),
    ),
    _int(
        "segmenter.turn_check_ms",
        "First turn check after",
        "Silence after which the smart model is asked first; earlier answers sooner but "
        "asks more often.",
        _TURN,
        (100, 500, 50),
        unit="ms",
    ),
    _int(
        "segmenter.turn_recheck_ms",
        "Turn recheck every",
        "How often the smart model is asked again during one pause; shorter reacts "
        "faster but runs the model more.",
        _TURN,
        (64, 320, 16),
        unit="ms",
    ),
    _float(
        "segmenter.turn_threshold",
        "Turn-complete probability",
        "Model probability at which the speaker counts as finished (not above the soft "
        "cut probability); lower ends phrases earlier but risks cutting them short.",
        _TURN,
        (0.3, 0.9, 0.05),
    ),
    _int(
        "segmenter.soft_cut_ms",
        "Soft cut after",
        "Phrase length from which a shorter, stricter pause check applies (0 turns it "
        "off); lower splits long speech sooner at natural breaks.",
        _TURN,
        (3_000, 12_000, 500),
        unit="ms",
        off_value=0,
    ),
    _int(
        "segmenter.soft_pause_ms",
        "Soft cut pause",
        "Pause that triggers a soft check inside long speech; shorter cuts more often.",
        _TURN,
        (96, 192, 16),
        unit="ms",
    ),
    _float(
        "segmenter.soft_cut_threshold",
        "Soft cut probability",
        "Probability a soft check must reach (not below the turn probability); higher "
        "cuts long speech less eagerly.",
        _TURN,
        (0.5, 0.95, 0.05),
    ),
    _int(
        "turn_threads",
        "Turn model threads",
        "CPU threads of the smart turn model; two are quicker but take a core from whisper.",
        _TURN,
        (_TURN_THREADS_MIN, _TURN_THREADS_MAX, 1),
    ),
    # --- Whisper ---
    _choice(
        "transcribe.model",
        "Whisper model",
        "tiny.en decodes about twice as fast, base.en makes fewer mistakes.",
        _WHISPER,
        WHISPER_MODELS,
    ),
    _int(
        "transcribe.beam_size",
        "Beam width",
        "Candidates whisper keeps while decoding; 1 is greedy and fastest, more is a "
        "little more accurate and slower.",
        _WHISPER,
        (1, 5, 1),
        soft_maximum=True,
    ),
    _int(
        "transcribe.threads",
        "Decode threads",
        "CPU threads of the whisper decode; more is faster until the cores are shared "
        "with the translators.",
        _WHISPER,
        (_WHISPER_THREADS_MIN, _WHISPER_THREADS_MAX, 1),
        soft_maximum=True,
    ),
    _float(
        "transcribe.min_rms",
        "Quiet chunk gate",
        "Chunks quieter than this level never reach whisper; higher saves work in a noisy "
        "room but can drop whispers.",
        _WHISPER,
        (0.0, 0.05, 0.001),
        soft_maximum=True,
    ),
    _int(
        "transcribe.merge_gap_ms",
        "Merge chunks closer than",
        "Chunks that waited for whisper and are this close are decoded together; larger "
        "recovers from a backlog but joins separate phrases.",
        _WHISPER,
        (0, 5_000, 100),
        unit="ms",
        soft_maximum=True,
    ),
    _int(
        "transcribe.merge_max_ms",
        "Longest merged audio",
        "Cap on merged audio (whisper's window is 30 s); higher clears a backlog in one "
        "pass but delays the first words.",
        _WHISPER,
        (1_000, 30_000, 1_000),
        unit="ms",
    ),
    _int(
        "transcribe.prompt_tail_chars",
        "Context passed on",
        "How much of the previous text is given to whisper so a cut sentence continues; "
        "more helps spelling and continuity but can repeat words.",
        _WHISPER,
        (0, 1_000, 50),
        unit="none",
        soft_maximum=True,
    ),
    _int(
        "transcribe.context_gap_ms",
        "Use context if next chunk within",
        "The context is used only if the next chunk starts this soon; longer keeps "
        "context across pauses but can carry over a finished topic.",
        _WHISPER,
        (0, 10_000, 100),
        unit="ms",
        soft_maximum=True,
    ),
    _float(
        "transcribe.no_speech_threshold",
        "No-speech threshold",
        "Whisper drops a segment it thinks is not speech above this probability; lower "
        "drops more noise but can lose quiet speech.",
        _WHISPER,
        (0.0, 1.0, 0.05),
        advanced=True,
    ),
    _float(
        "transcribe.log_prob_threshold",
        "Log-probability floor",
        "Segments below this average log probability count as doubtful; higher is "
        "stricter and drops more.",
        _WHISPER,
        (-5.0, 0.0, 0.1),
        advanced=True,
        soft_minimum=True,
    ),
    _float(
        "transcribe.compression_ratio_threshold",
        "Compression ratio limit",
        "Text compressing better than this is treated as a repetition loop; lower is "
        "stricter against loops but can cut real repeats.",
        _WHISPER,
        (1.0, 5.0, 0.1),
        advanced=True,
        soft_maximum=True,
    ),
    _int(
        "transcribe.repeat_limit",
        "Repeated phrase limit",
        "A phrase repeated this many times in a row is collapsed to one; lower removes "
        "hallucinated loops sooner but can shorten a real chant.",
        _WHISPER,
        (2, 10, 1),
        advanced=True,
        soft_maximum=True,
    ),
    _int(
        "transcribe.merge_pad_ms",
        "Silence between merged chunks",
        "Silence inserted between merged chunks (never more than the real gap); more "
        "separates words cleanly but lengthens the audio.",
        _WHISPER,
        (0, 2_000, 50),
        unit="ms",
        advanced=True,
        soft_maximum=True,
    ),
    # --- Capture ---
    _int(
        "capture.block_ms",
        "Block size",
        "Length of one audio callback block; smaller lowers latency but wakes the "
        "driver more often.",
        _CAPTURE,
        (5, 100, 5),
        unit="ms",
    ),
    _float(
        "capture.ring_seconds",
        "Ring buffer",
        "Audio the input can hold while processing is busy; larger survives longer "
        "stalls but uses more memory.",
        _CAPTURE,
        (0.5, 30.0, 0.5),
        unit="s",
    ),
    _int(
        "capture.watchdog_ms",
        "Lost-stream timeout",
        "No audio callback for this long means the microphone is lost; shorter "
        "reconnects sooner but can misfire on a busy machine.",
        _CAPTURE,
        (100, 5_000, 100),
        unit="ms",
        soft_maximum=True,
    ),
    _int(
        "capture.retry_ms",
        "Reconnect pause",
        "Pause between attempts to reopen a lost microphone; shorter recovers sooner but "
        "polls the driver harder.",
        _CAPTURE,
        (100, 10_000, 100),
        unit="ms",
        soft_minimum=True,
        soft_maximum=True,
    ),
    _choice(
        "capture.downmix",
        "Downmix",
        "How a multi-channel microphone becomes mono; auto picks the live channels, "
        "mean averages all and can thin out a one-sided signal.",
        _CAPTURE,
        get_args(DownmixMode),
    ),
    # --- Output ---
    _int(
        "output.prebuffer_ms",
        "Prebuffer",
        "Audio collected at the start of a phrase before playback begins; more avoids "
        "stutter but delays every phrase.",
        _OUTPUT,
        (0, 100, 10),
        unit="ms",
    ),
    _int(
        "output.block_ms",
        "Block size",
        "Length of one audio callback block of each output; smaller lowers latency but "
        "wakes the driver more often.",
        _OUTPUT,
        (2, 100, 2),
        unit="ms",
    ),
    _float(
        "output.ring_seconds",
        "Ring buffer",
        "Speech each output can queue; larger holds long translations but uses more memory.",
        _OUTPUT,
        (0.5, 60.0, 0.5),
        unit="s",
    ),
    _int(
        "output.watchdog_ms",
        "Lost-stream timeout",
        "No callback for this long means an output is lost; shorter reconnects sooner "
        "but can misfire on a busy machine.",
        _OUTPUT,
        (100, 5_000, 100),
        unit="ms",
        soft_maximum=True,
    ),
    _int(
        "output.retry_ms",
        "Reconnect pause",
        "Pause between attempts to reopen a lost output; shorter recovers sooner but "
        "polls the driver harder.",
        _OUTPUT,
        (100, 10_000, 100),
        unit="ms",
        soft_minimum=True,
        soft_maximum=True,
    ),
    _flag(
        "hide_text",
        "Hide utterance text",
        "Keeps the transcript and translations off the screen; text is never logged either way.",
        _OUTPUT,
        apply=ApplyMode.LIVE,
    ),
    # --- Advanced ---
    _flag(
        "strict",
        "Strict mode",
        "Blocks disk writes and network access once the models are loaded and ends the "
        "program on a violation.",
        _ADVANCED,
    ),
    _int(
        "translate.beam_size",
        "Translation beam width",
        "Candidates each translator keeps; 1 is greedy and fastest, 2 is a good trade "
        "for short sentences.",
        _ADVANCED,
        (1, 8, 1),
    ),
    _int(
        "translate.threads",
        "Translation threads",
        "CPU threads of each translator; more is faster per sentence but competes with whisper.",
        _ADVANCED,
        (1, 8, 1),
    ),
)


def route_fields(targets: Iterable[str]) -> tuple[Field, ...]:
    """The radio offset field of each target, which exist only while the lane does."""
    return tuple(
        Field(
            key=f"routes.{target}.radio_offset_ms",
            label=f"Radio offset {target}",
            help="Extra delay of a wireless headset, added to the reported latency so the "
            "outputs line up; the audio itself is not delayed.",
            kind="int",
            default=_default_of("routes.radio_offset_ms"),
            group="Output",
            minimum=0,
            maximum=_MAX_RADIO_OFFSET_MS,
            step=10,
            unit="ms",
        )
        for target in targets
    )


def schema_for(settings: Settings) -> tuple[Field, ...]:
    """``SCHEMA`` plus one radio offset field per route of ``settings``."""
    return SCHEMA + route_fields(route.target for route in settings.routes)


def fields_by_group(settings: Settings | None = None) -> dict[str, tuple[Field, ...]]:
    """The fields of each group, in display order; ``settings`` adds its routes."""
    fields = SCHEMA if settings is None else schema_for(settings)
    return {group: tuple(item for item in fields if item.group == group) for group in GROUPS}


# --- reading values ---------------------------------------------------------------------------


def _read(settings: Settings, key: str) -> object:
    section, _, attr = key.partition(".")
    if not attr:
        return getattr(settings, section)
    if section == "routes":
        target, _, name = attr.partition(".")
        for route in settings.routes:
            if route.target == target:
                return getattr(route, name)
        raise SettingsSchemaError(f"{key}: no route for {target!r}")
    if section == "output":
        return getattr(settings.routes[0].output, attr)
    return getattr(getattr(settings, section), attr)


def values_from_settings(settings: Settings) -> dict[str, object]:
    """The current value of every field (``output.*`` is read from the first route)."""
    return {item.key: _read(settings, item.key) for item in schema_for(settings)}


# --- checking values --------------------------------------------------------------------------


def _check_value(item: Field, raw: object) -> tuple[object, str | None]:
    """Return the value in its final type, or the message that explains the refusal."""
    if item.kind == "bool":
        if isinstance(raw, bool):
            return raw, None
        return raw, "must be on or off"
    if item.kind == "choice":
        if isinstance(raw, str) and raw in item.choices:
            return raw, None
        return raw, f"must be one of {', '.join(item.choices)}"
    problem = _number_problem(item.kind, raw)
    if problem is not None:
        return raw, problem
    assert isinstance(raw, (int, float))
    if not _in_range(item, raw):
        return raw, f"must be {_range_text(item)}, got {_fmt(raw)}"
    return (float(raw) if item.kind == "float" else raw), None


@dataclass(frozen=True, slots=True)
class _Rule:
    """A rule over two fields; a break is reported on both."""

    keys: tuple[str, str]
    holds: Callable[[Mapping[str, object]], bool]
    message: str


def _le(low: str, high: str) -> Callable[[Mapping[str, object]], bool]:
    def holds(values: Mapping[str, object]) -> bool:
        a, b = values[low], values[high]
        assert isinstance(a, (int, float))
        assert isinstance(b, (int, float))
        return a <= b

    return holds


def _cut_search_fits(values: Mapping[str, object]) -> bool:
    cut, longest = values["segmenter.cut_search_ms"], values["segmenter.max_utterance_ms"]
    assert isinstance(cut, int)
    assert isinstance(longest, int)
    return cut <= longest // 2


_RULES: tuple[_Rule, ...] = (
    _Rule(
        ("segmenter.exit_threshold", "segmenter.speech_threshold"),
        _le("segmenter.exit_threshold", "segmenter.speech_threshold"),
        "the silence probability must not be above the speech probability",
    ),
    _Rule(
        ("segmenter.turn_threshold", "segmenter.soft_cut_threshold"),
        _le("segmenter.turn_threshold", "segmenter.soft_cut_threshold"),
        "the soft cut probability must not be lower than the turn-complete probability",
    ),
    _Rule(
        ("segmenter.tail_ms", "segmenter.silence_ms"),
        _le("segmenter.tail_ms", "segmenter.silence_ms"),
        "the silence kept after speech must not be longer than the pause that ends a phrase",
    ),
    _Rule(
        ("segmenter.cut_search_ms", "segmenter.max_utterance_ms"),
        _cut_search_fits,
        "the forced-cut search window must be at most half of the longest phrase",
    ),
)


# --- applying values --------------------------------------------------------------------------

_T = TypeVar("_T")

_SECTION_ERRORS: dict[str, type[ValueError]] = {
    "segmenter": VadConfigError,
    "transcribe": TranscribeConfigError,
    "translate": TranslateConfigError,
    "capture": CaptureConfigError,
    "output": OutputConfigError,
}


def keys_named_in(message: str, section: str) -> dict[str, str]:
    """Map a module's message back to the fields of ``section`` that it names."""
    text = message.lower().replace("-", "_")
    found = [
        item.key
        for item in SCHEMA
        if item.key.startswith(f"{section}.")
        and re.search(rf"\b{re.escape(item.key.partition('.')[2])}\b", text)
    ]
    if section == "capture" and not found and re.search(r"\bsilence_ms\b", text):
        found = ["segmenter.silence_ms"]
    return dict.fromkeys(found, message) if found else {GENERAL_ERROR_KEY: message}


def _settings_error_keys(message: str) -> dict[str, str]:
    text = message.lower().replace("-", "_")
    if "turn_threads" in text:
        return {"turn_threads": message}
    if "silence_ms" in text:
        return {"segmenter.silence_ms": message}
    return {GENERAL_ERROR_KEY: message}


def construct_settings(base: Settings, values: Mapping[str, object]) -> Settings:
    """Apply ``values`` through the real constructors only, with no range checks of its own.

    ``segmenter.silence_ms`` also sets the capture config (the two must be equal). The values
    must already have the right types.

    Raises:
        SettingsValidationError: A constructor refused; the errors are keyed by field.
    """
    changes: dict[str, dict[str, Any]] = {
        section: {} for section in ("segmenter", "transcribe", "capture", "output", "translate")
    }
    offsets: dict[str, object] = {}
    top: dict[str, object] = {}
    for key, value in values.items():
        section, _, attr = key.partition(".")
        if not attr:
            top[section] = value
        elif section == "routes":
            target, _, _name = attr.partition(".")
            offsets[target] = value
        else:
            changes[section][attr] = value
        if key == "segmenter.silence_ms":
            changes["capture"]["silence_ms"] = value

    errors: dict[str, str] = {}

    def attempt(section: str, build: Callable[[], _T]) -> _T | None:
        try:
            return build()
        except _SECTION_ERRORS[section] as exc:
            errors.update(keys_named_in(str(exc), section))
            return None

    segmenter = attempt("segmenter", lambda: replace(base.segmenter, **changes["segmenter"]))
    transcribe = attempt("transcribe", lambda: replace(base.transcribe, **changes["transcribe"]))
    translate = attempt("translate", lambda: replace(base.translate, **changes["translate"]))
    capture = attempt("capture", lambda: replace(base.capture, **changes["capture"]))
    routes = tuple(
        _route_with(route, changes["output"], offsets.get(route.target), errors)
        for route in base.routes
    )
    if errors or None in (segmenter, transcribe, translate, capture) or None in routes:
        raise SettingsValidationError(errors)
    assert segmenter is not None
    assert transcribe is not None
    assert translate is not None
    assert capture is not None
    try:
        return replace(
            base,
            segmenter=segmenter,
            transcribe=transcribe,
            translate=translate,
            capture=capture,
            routes=tuple(route for route in routes if route is not None),
            **top,  # type: ignore[arg-type]
        )
    except ConfigError as exc:
        raise SettingsValidationError(_settings_error_keys(str(exc))) from exc


def _route_with(
    route: LaneRoute,
    output_changes: Mapping[str, Any],
    offset: object,
    errors: dict[str, str],
) -> LaneRoute | None:
    key = f"routes.{route.target}.radio_offset_ms"
    try:
        output = replace(route.output, **output_changes)
    except OutputConfigError as exc:
        errors.update(keys_named_in(str(exc), "output"))
        return None
    if offset is None:
        return replace(route, output=output)
    if not isinstance(offset, int) or isinstance(offset, bool):
        errors[key] = "must be a whole number"
        return None
    try:
        # LaneConfig owns the radio offset limit; LaneRoute itself checks nothing.
        LaneConfig(radio_offset_ms=offset)
    except LaneConfigError as exc:
        errors[key] = str(exc)
        return None
    return replace(route, output=output, radio_offset_ms=offset)


def build_settings(base: Settings, values: Mapping[str, object]) -> Settings:
    """Apply ``values`` to ``base`` and return the new, validated ``Settings``.

    Keys that are missing keep the value of ``base``. Types, ranges and the rules between
    two fields are checked first and all problems are reported together; the real
    constructors then have the last word.

    Raises:
        SettingsValidationError: One or more values are refused; ``errors`` maps a field key
            (or ``GENERAL_ERROR_KEY``) to a message.
    """
    known = {item.key: item for item in schema_for(base)}
    errors: dict[str, str] = {}
    accepted: dict[str, object] = {}
    for key, raw in values.items():
        item = known.get(key)
        if item is None:
            errors[key] = "unknown setting"
            continue
        value, problem = _check_value(item, raw)
        if problem is None:
            accepted[key] = value
        else:
            errors[key] = problem
    merged = {**values_from_settings(base), **accepted}
    for rule in _RULES:
        if any(key in errors for key in rule.keys):
            continue
        if not rule.holds(merged):
            errors.update(dict.fromkeys(rule.keys, rule.message))
    if errors:
        raise SettingsValidationError(errors)
    return construct_settings(base, accepted)


# --- comparing --------------------------------------------------------------------------------


def diff(old: Settings, new: Settings) -> dict[str, tuple[object, object]]:
    """Fields whose value differs, as ``key -> (old, new)``; a missing route reads ``None``."""
    before = values_from_settings(old)
    after = values_from_settings(new)
    return {
        key: (before.get(key), after.get(key))
        for key in {**before, **after}
        if before.get(key) != after.get(key)
    }


def needs_restart(old: Settings, new: Settings) -> bool:
    """Whether any changed field needs the pipeline to restart (``hide_text`` does not)."""
    known = {item.key: item for item in schema_for(old) + schema_for(new)}
    return any(known[key].apply is ApplyMode.RESTART for key in diff(old, new))
