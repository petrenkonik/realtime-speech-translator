"""Command-line settings: one validated ``Settings`` object built from the flags.

The flags map one to one to the fields of the per-module configs (``SegmenterConfig``,
``TranscribeConfig``, ``CaptureConfig``, ``OutputConfig``); those classes keep their own
range checks, and this module only turns their errors into a single ``ConfigError`` with a
message that names the setting. Nothing here touches hardware or the network; the one
function that reads the disk is ``check_files`` and it only looks, never writes.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

from pipeline.capture import CaptureConfig, CaptureConfigError
from pipeline.dashboard import DashboardMode
from pipeline.output import OutputConfig, OutputConfigError
from pipeline.transcribe import (
    DEFAULT_WHISPER_MODEL,
    WHISPER_MODELS,
    TranscribeConfig,
    TranscribeConfigError,
    whisper_model_dir,
)
from pipeline.translate import TranslateConfig, TranslateConfigError, translator_model_dir
from pipeline.tts_sapi import TtsConfig, TtsConfigError
from pipeline.turn import DEFAULT_TURN_MODEL
from pipeline.vad import SegmenterConfig, VadConfigError

SOURCE_LANGUAGE = "en"
SUPPORTED_TARGETS = ("de", "fr")
VAD_MODEL = "silero_vad.onnx"
_WHISPER_THREADS_RANGE = (1, 8)
_TURN_THREADS_RANGE = (1, 2)
_MAX_RADIO_OFFSET_MS = 2_000

UNSELECTED_DEVICE = "(no device selected)"
"""Device query of a window that starts before the user chose a device.

The configs refuse an empty query, and a window must be able to start with nothing chosen, so
this placeholder stands in. It is not a name any real device carries, so a lookup with it ends
in a normal "not found"; the window checks ``is_unselected`` first and says "select a device".
"""


def is_unselected(device_query: str) -> bool:
    """True for the placeholder that means "no device chosen yet"."""
    return device_query == UNSELECTED_DEVICE


class ConfigError(ValueError):
    """A setting is missing, malformed or outside its range; the message says which."""


@dataclass(frozen=True, slots=True)
class LaneRoute:
    """Where the speech of one target language goes.

    Attributes:
        target: Language code of the lane.
        output: Settings of the output device that plays it.
        radio_offset_ms: Extra delay that lines this output up with the others
            (a wireless headset has a radio latency; see test T6).
        tts: Voice, speaking rate and volume of the lane.
    """

    target: str
    output: OutputConfig
    radio_offset_ms: int = 0
    tts: TtsConfig = field(default_factory=TtsConfig)


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything ``run`` needs, validated.

    Attributes:
        source: Spoken language (English only).
        targets: Languages to translate into, in lane order.
        capture: Input device settings.
        segmenter: Utterance segmentation settings.
        turn_threads: onnxruntime threads for the Smart Turn model.
        transcribe: Whisper settings.
        translate: Machine translation settings (shared by the lanes).
        routes: One entry per target, in the order of ``targets``.
        glossary_path: Optional glossary file.
        models_dir: Where ``scripts/fetch_models.py`` put the models.
        strict: Arm the offline / no-disk guard after the models are loaded.
        hide_text: Keep utterance text off the dashboard.
        dashboard_mode: ``"live"`` or ``"plain"``.
    """

    capture: CaptureConfig
    routes: tuple[LaneRoute, ...]
    source: str = SOURCE_LANGUAGE
    targets: tuple[str, ...] = SUPPORTED_TARGETS
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    turn_threads: int = 1
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    glossary_path: Path | None = None
    models_dir: Path = Path("models")
    strict: bool = False
    hide_text: bool = False
    dashboard_mode: DashboardMode = "live"

    def __post_init__(self) -> None:
        """Check the settings that involve more than one module."""
        if self.source != SOURCE_LANGUAGE:
            raise ConfigError(
                f"source language must be {SOURCE_LANGUAGE!r} (English only), got {self.source!r}"
            )
        _check_targets(self.targets, self.source)
        if [route.target for route in self.routes] != list(self.targets):
            raise ConfigError("routes must be one per target, in the order of the targets")
        low, high = _TURN_THREADS_RANGE
        if not low <= self.turn_threads <= high:
            raise ConfigError(f"--turn-threads must be {low}-{high}, got {self.turn_threads}")
        if self.capture.silence_ms != self.segmenter.silence_ms:
            raise ConfigError(
                "the capture report and the segmenter must use the same silence_ms, got "
                f"{self.capture.silence_ms} and {self.segmenter.silence_ms}"
            )


def _check_targets(targets: Sequence[str], source: str) -> None:
    if not targets:
        raise ConfigError("--targets needs at least one language")
    for code in targets:
        if code not in SUPPORTED_TARGETS:
            raise ConfigError(
                f"unsupported target language {code!r}; choose from {', '.join(SUPPORTED_TARGETS)}"
            )
        if code == source:
            raise ConfigError(f"target language {code!r} is the same as the source")
    if len(set(targets)) != len(targets):
        raise ConfigError(f"--targets lists a language twice: {','.join(targets)}")


# --- parsing ---------------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """An ``ArgumentParser`` that raises instead of printing usage and exiting."""

    def error(self, message: str) -> NoReturn:
        raise ConfigError(message)


def add_run_arguments(parser: argparse.ArgumentParser, *, gui: bool = False) -> None:
    """Add every ``run`` flag to ``parser`` (the CLI entry point reuses this).

    Args:
        parser: The parser to fill.
        gui: For the window: ``--input`` and ``--route`` become optional (the window starts
            with no device chosen) and ``--no-ui`` is left out (it has no meaning there).
    """
    parser.add_argument(
        "--input",
        required=not gui,
        default="",
        help="substring of the input device name" + (" (optional in the window)" if gui else ""),
    )
    parser.add_argument("--source", default=SOURCE_LANGUAGE, help="spoken language (en)")
    parser.add_argument(
        "--targets",
        default=",".join(SUPPORTED_TARGETS),
        help="comma separated target languages (default: %(default)s)",
    )
    parser.add_argument(
        "--route",
        default="",
        help="output device per target, e.g. de=Realtek,fr=HS100W (a name substring)",
    )
    parser.add_argument(
        "--radio-offset",
        default="",
        help="extra delay in ms per target, e.g. fr=120 (wireless headset, test T6)",
    )
    parser.add_argument(
        "--tts-voice",
        default="",
        help="voice per target by part of its name, e.g. de=Hedda (default: the first voice "
        "installed for the language)",
    )
    parser.add_argument(
        "--tts-rate", type=int, default=0, help="speaking rate, -10 (slow) to 10 (fast)"
    )
    parser.add_argument("--glossary", type=Path, help="glossary file (TOML)")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--whisper-model",
        choices=WHISPER_MODELS,
        default=DEFAULT_WHISPER_MODEL,
        help="tiny.en decodes about twice as fast, base.en is more accurate",
    )
    parser.add_argument("--whisper-threads", type=int, default=4, help="1-8")
    parser.add_argument("--mt-threads", type=int, default=2, help="threads per translator, 1-8")
    parser.add_argument("--mt-beam", type=int, default=2, help="translation beam width, 1-8")
    parser.add_argument("--silence-ms", type=int, default=450, help="450-600")
    parser.add_argument("--turn-mode", choices=("fixed", "smart"), default="fixed")
    parser.add_argument("--turn-check-ms", type=int, default=200, help="100-500, step 50")
    parser.add_argument("--turn-recheck-ms", type=int, default=96, help="64-320")
    parser.add_argument("--turn-threshold", type=float, default=0.5, help="0.3-0.9, step 0.05")
    parser.add_argument("--soft-cut-ms", type=int, default=6000, help="0 (off) or 3000-12000")
    parser.add_argument("--soft-pause-ms", type=int, default=128, help="96-192")
    parser.add_argument("--soft-cut-threshold", type=float, default=0.7, help="0.5-0.95")
    parser.add_argument("--turn-threads", type=int, default=1, help="1-2")
    parser.add_argument("--max-utterance-ms", type=int, default=15_000, help="5000-30000")
    parser.add_argument("--cut-search-ms", type=int, default=1500, help="500-3000")
    parser.add_argument("--pre-roll-ms", type=int, default=100, help="0-2000")
    parser.add_argument("--out-prebuffer-ms", type=int, default=20, help="0-100")
    parser.add_argument("--strict", action="store_true", help="block disk writes and the network")
    parser.add_argument("--hide-text", action="store_true", help="no utterance text on screen")
    if not gui:
        parser.add_argument("--no-ui", action="store_true", help="plain one-line status output")


def parse_settings(argv: Sequence[str], *, gui: bool = False) -> Settings:
    """Parse the ``run`` flags (or, with ``gui``, the window's flags) into validated settings.

    Raises:
        ConfigError: Any syntax error, missing flag or out-of-range value.
    """
    parser = _Parser(prog="pipeline gui" if gui else "pipeline run", add_help=False)
    add_run_arguments(parser, gui=gui)
    return settings_from_args(parser.parse_args(list(argv)), allow_unselected=gui)


def settings_from_args(args: argparse.Namespace, *, allow_unselected: bool = False) -> Settings:
    """Build ``Settings`` from a parsed namespace; module range errors become ``ConfigError``.

    Args:
        args: Namespace filled by ``add_run_arguments``.
        allow_unselected: A missing ``--input`` or output device becomes ``UNSELECTED_DEVICE``
            instead of an error (the window starts that way).
    """
    try:
        return _build(args, allow_unselected)
    except (
        VadConfigError,
        TranscribeConfigError,
        TranslateConfigError,
        TtsConfigError,
        CaptureConfigError,
        OutputConfigError,
    ) as exc:
        raise ConfigError(str(exc)) from exc


def _build(args: argparse.Namespace, allow_unselected: bool) -> Settings:
    input_query = args.input.strip()
    if not input_query:
        if not allow_unselected:
            raise ConfigError("--input must name an input device")
        input_query = UNSELECTED_DEVICE
    targets = _split_languages(args.targets)
    _check_targets(targets, args.source)
    devices = _pairs(args.route, "--route")
    offsets = _pairs(args.radio_offset, "--radio-offset")
    _only_known(devices, targets, "--route")
    _only_known(offsets, targets, "--radio-offset")
    voices = _pairs(args.tts_voice, "--tts-voice")
    _only_known(voices, targets, "--tts-voice")
    if not devices and not allow_unselected:
        raise ConfigError(
            "--route is required: give an output device for each target, e.g. "
            + ",".join(f"{code}=DEVICE" for code in targets)
        )
    for code in targets:
        if not devices.get(code, "").strip():
            if not allow_unselected:
                raise ConfigError(f"--route has no output device for target {code!r}")
            devices[code] = UNSELECTED_DEVICE

    low, high = _WHISPER_THREADS_RANGE
    if not low <= args.whisper_threads <= high:
        raise ConfigError(f"--whisper-threads must be {low}-{high}, got {args.whisper_threads}")

    segmenter = SegmenterConfig(
        silence_ms=args.silence_ms,
        pre_roll_ms=args.pre_roll_ms,
        max_utterance_ms=args.max_utterance_ms,
        cut_search_ms=args.cut_search_ms,
        turn_mode=args.turn_mode,
        turn_check_ms=args.turn_check_ms,
        turn_recheck_ms=args.turn_recheck_ms,
        turn_threshold=args.turn_threshold,
        soft_cut_ms=args.soft_cut_ms,
        soft_pause_ms=args.soft_pause_ms,
        soft_cut_threshold=args.soft_cut_threshold,
    )
    routes = tuple(
        LaneRoute(
            target=code,
            output=OutputConfig(device_query=devices[code], prebuffer_ms=args.out_prebuffer_ms),
            radio_offset_ms=_offset_ms(code, offsets.get(code, "0")),
            tts=TtsConfig(rate=args.tts_rate, voice=voices.get(code)),
        )
        for code in targets
    )
    return Settings(
        capture=CaptureConfig(device_query=input_query, silence_ms=segmenter.silence_ms),
        routes=routes,
        source=args.source,
        targets=targets,
        segmenter=segmenter,
        turn_threads=args.turn_threads,
        transcribe=TranscribeConfig(model=args.whisper_model, threads=args.whisper_threads),
        translate=TranslateConfig(beam_size=args.mt_beam, threads=args.mt_threads),
        glossary_path=args.glossary,
        models_dir=args.models_dir,
        strict=args.strict,
        hide_text=args.hide_text,
        dashboard_mode="plain" if getattr(args, "no_ui", False) else "live",
    )


def _split_languages(text: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in text.split(","))
    if not text.strip() or "" in parts:
        raise ConfigError(f"--targets must be a list like de,fr, got {text!r}")
    return parts


def _pairs(text: str, flag: str) -> dict[str, str]:
    """Parse ``a=1,b=2``; a repeated key is an error rather than a silent override."""
    result: dict[str, str] = {}
    if not text.strip():
        return result
    for entry in text.split(","):
        key, separator, value = entry.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ConfigError(f"{flag} entry {entry!r} must look like LANGUAGE=VALUE")
        if key in result:
            raise ConfigError(f"{flag} gives {key!r} twice")
        result[key] = value.strip()
    return result


def _only_known(values: dict[str, str], targets: Sequence[str], flag: str) -> None:
    for key in values:
        if key not in targets:
            raise ConfigError(f"{flag} names {key!r}, which is not in --targets")


def _offset_ms(target: str, text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise ConfigError(
            f"--radio-offset for {target!r} must be a whole number of ms, got {text!r}"
        ) from None
    if not 0 <= value <= _MAX_RADIO_OFFSET_MS:
        raise ConfigError(
            f"--radio-offset for {target!r} must be 0-{_MAX_RADIO_OFFSET_MS} ms, got {value}"
        )
    return value


# --- files -----------------------------------------------------------------------------------


def check_files(settings: Settings) -> None:
    """Fail early, with all problems at once, if a model or the glossary is missing.

    Raises:
        ConfigError: Lists every missing file and says how to get the models.
    """
    models = settings.models_dir
    missing: list[str] = []
    if not (models / VAD_MODEL).is_file():
        missing.append(str(models / VAD_MODEL))
    whisper = whisper_model_dir(models, settings.transcribe)
    if not whisper.is_dir():
        missing.append(str(whisper))
    for target in settings.targets:
        folder = translator_model_dir(models, settings.source, target)
        if not (folder / "model.bin").is_file():
            missing.append(str(folder))
    if settings.segmenter.turn_mode == "smart" and not (models / DEFAULT_TURN_MODEL).is_file():
        missing.append(str(models / DEFAULT_TURN_MODEL))
    problems = [f"model missing: {path}" for path in missing]
    if missing:
        problems.append("run scripts/fetch_models.py to download the models")
    if settings.glossary_path is not None and not settings.glossary_path.is_file():
        problems.append(f"glossary file not found: {settings.glossary_path}")
    if problems:
        raise ConfigError("; ".join(problems))
