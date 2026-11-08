"""``--strict``: an audit hook that stops the program on disk writes and network use.

The hard constraints are zero disk writes and zero network at runtime. Tests check them
for our own code; this guard checks them for everything Python-level in the running
process, including third-party libraries. It cannot see native code (PortAudio,
onnxruntime, CTranslate2): that is covered at OS level by Process Monitor and by
Airplane Mode (spec A1, A2).

Audit hooks cannot be removed once added. One process-wide hook is therefore installed
on first use, and it asks the currently armed guards; a disarmed guard costs nothing.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

ViolationKind = Literal["disk", "network"]

EXIT_CODE_STRICT_VIOLATION = 70

_DISK_EVENTS = frozenset(
    {
        "open",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.mkdir",
        "os.truncate",
        "os.symlink",
        "os.link",
        "tempfile.mkstemp",
        "tempfile.mkdtemp",
    }
)
_NETWORK_EVENTS = frozenset(
    {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto"}
)
_WATCHED_EVENTS = _DISK_EVENTS | _NETWORK_EVENTS

_WRITE_MODE_CHARS = frozenset("wax+")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

# Console and null devices are opened "for writing" by the runtime itself but touch no disk.
_HARMLESS_TARGETS = frozenset({"nul", "con", "conout$", "conin$", os.devnull.casefold()})


class StrictViolationError(RuntimeError):
    """Raised by a guard handler to block an operation that strict mode forbids."""


@dataclass(frozen=True, slots=True)
class Violation:
    """One forbidden operation.

    Attributes:
        kind: ``"disk"`` or ``"network"``.
        event: The audit event name, for example ``"open"`` or ``"socket.connect"``.
        detail: Path or address involved (never audio or text content).
    """

    kind: ViolationKind
    event: str
    detail: str

    def message(self) -> str:
        """Human-readable explanation for the abort message."""
        return (
            f"strict mode: {self.kind} access blocked ({self.event}: {self.detail}). "
            "This program must not write to disk or use the network at runtime."
        )


def abort_process(violation: Violation) -> None:
    """Default handler: print the reason and end the process immediately.

    ``os._exit`` is deliberate: an exception raised inside a worker thread could be
    swallowed there, which would defeat the point of strict mode.
    """
    sys.stderr.write(violation.message() + "\n")
    sys.stderr.flush()
    os._exit(EXIT_CODE_STRICT_VIOLATION)


_lock = threading.Lock()
_hook_installed = False
_armed: tuple[StrictGuard, ...] = ()


def _dispatch(event: str, args: tuple[Any, ...]) -> None:
    if event not in _WATCHED_EVENTS:
        return
    for guard in _armed:
        guard._check(event, args)


class StrictGuard:
    """Watches the process for disk writes and network access while armed."""

    def __init__(self, on_violation: Callable[[Violation], None] = abort_process) -> None:
        """Create a disarmed guard.

        Args:
            on_violation: Called for every forbidden operation. If it raises, the
                operation is blocked with that exception. The default ends the process.
        """
        self._on_violation = on_violation
        self.violations: list[Violation] = []

    def arm(self) -> None:
        """Start watching. Safe to call repeatedly."""
        global _hook_installed, _armed
        with _lock:
            if not _hook_installed:
                sys.addaudithook(_dispatch)
                _hook_installed = True
            if self not in _armed:
                _armed = (*_armed, self)

    def disarm(self) -> None:
        """Stop watching."""
        global _armed
        with _lock:
            _armed = tuple(guard for guard in _armed if guard is not self)

    def __enter__(self) -> StrictGuard:
        """Arm for the duration of a ``with`` block."""
        self.arm()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Disarm."""
        self.disarm()

    def _check(self, event: str, args: tuple[Any, ...]) -> None:
        violation = classify(event, args)
        if violation is None:
            return
        self.violations.append(violation)
        self._on_violation(violation)


def classify(event: str, args: tuple[Any, ...]) -> Violation | None:
    """Decide whether an audit event is forbidden; ``None`` means allowed."""
    if event in _NETWORK_EVENTS:
        return Violation("network", event, _describe_address(args))
    if event == "open":
        return classify_open(args)
    return Violation("disk", event, _describe_path(args[0] if args else ""))


def classify_open(args: tuple[Any, ...]) -> Violation | None:
    """Judge an ``open`` event by its mode string or, for ``os.open``, its flags."""
    target, mode, flags = (*args, None, None, None)[:3]
    if isinstance(target, int):  # an already open descriptor: nothing new is created
        return None
    writes = (isinstance(mode, str) and bool(_WRITE_MODE_CHARS & set(mode))) or (
        isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
    )
    if not writes:
        return None
    detail = _describe_path(target)
    if os.path.basename(detail).casefold() in _HARMLESS_TARGETS:
        return None
    return Violation("disk", "open", detail)


def _describe_path(value: object) -> str:
    if isinstance(value, str | bytes | os.PathLike):
        return os.fsdecode(value)
    return repr(value)


def _describe_address(args: tuple[Any, ...]) -> str:
    # Some events carry (socket, address), others (host, port): show what is there.
    return " ".join(repr(arg) for arg in args if not hasattr(arg, "fileno"))
