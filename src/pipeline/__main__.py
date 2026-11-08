"""Command line: ``python -B -m pipeline gui`` and ``python -B -m pipeline list-devices``.

``run`` and ``soak`` belong to later work-plan steps and are not here yet. Everything that
touches hardware or a toolkit is imported inside the command that needs it, so ``--help`` and
``list-devices`` never load Tk, onnxruntime or the models.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from pipeline.config import ConfigError, Settings, add_run_arguments, settings_from_args
from pipeline.devices import DeviceListing, SoundDeviceListing, format_device_table

GuiRunner = Callable[[Settings], int]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline", description="Offline real-time interpreter demo."
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    gui = commands.add_parser(
        "gui",
        help="open the control window",
        description="Open the control window. Devices can be chosen in the window; the options "
        "below only set where it starts.",
    )
    add_run_arguments(gui, gui=True)
    commands.add_parser("list-devices", help="print the WASAPI input and output devices")
    return parser


def _run_gui(settings: Settings) -> int:
    from pipeline.ui.app_tk import run_gui  # Tk is loaded only for this command

    return run_gui(settings)


def main(
    argv: Sequence[str] | None = None,
    *,
    listing: DeviceListing | None = None,
    gui_runner: GuiRunner | None = None,
) -> int:
    """Run one command and return the process exit code.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` when omitted.
        listing: Device source for ``list-devices`` (PortAudio when omitted; tests inject one).
        gui_runner: Opens the window for ``gui`` (the real Tk window when omitted).

    Returns:
        0 on success, 2 for a usage or configuration error.
    """
    parser = _parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "gui":
        try:
            settings = settings_from_args(args, allow_unselected=True)
        except ConfigError as error:
            sys.stderr.write(f"pipeline gui: error: {error}\n")
            return 2
        return (gui_runner or _run_gui)(settings)
    if args.command == "list-devices":
        devices = (listing or SoundDeviceListing()).list_devices()
        sys.stdout.write(format_device_table(devices) + "\n")
        return 0
    parser.print_usage(sys.stderr)
    sys.stderr.write("pipeline: error: choose a command: gui or list-devices\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
