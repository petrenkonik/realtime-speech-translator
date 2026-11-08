"""Strict mode and the zero-disk / zero-network guarantees (spec G1, G2, G3)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from fakes import silence, speech, stream_of
from pytest_socket import SocketBlockedError
from test_capture import build, collect, play

from pipeline.guards import (
    EXIT_CODE_STRICT_VIOLATION,
    StrictGuard,
    StrictViolationError,
    Violation,
    abort_process,
    classify,
)

pytestmark = pytest.mark.unit


def raising(violation: Violation) -> None:
    raise StrictViolationError(violation.message())


@pytest.fixture
def guard() -> StrictGuard:
    return StrictGuard(on_violation=raising)


# ---------------------------------------------------------------- G3: disk


@pytest.mark.parametrize("mode", ["w", "wb", "a", "x", "w+"])
def test_g3_opening_a_file_for_writing_is_blocked(
    guard: StrictGuard, tmp_path: Path, mode: str
) -> None:
    target = tmp_path / "out.bin"
    with guard, pytest.raises(StrictViolationError, match="disk"):
        open(target, mode)  # noqa: SIM115
    assert not target.exists()
    assert guard.violations[0].event == "open"
    assert guard.violations[0].detail == str(target)


def test_read_write_mode_on_an_existing_file_is_blocked_and_leaves_it_alone(
    guard: StrictGuard, tmp_path: Path
) -> None:
    target = tmp_path / "data.bin"
    target.write_bytes(b"x")
    with guard, pytest.raises(StrictViolationError):
        open(target, "r+b")  # noqa: SIM115
    assert target.read_bytes() == b"x"


def test_reading_files_is_allowed(guard: StrictGuard, tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"weights")
    with guard:
        assert source.read_bytes() == b"weights"
        with open(source, "rb") as handle:
            assert handle.read() == b"weights"
    assert guard.violations == []


def test_low_level_open_with_create_flags_is_blocked(guard: StrictGuard, tmp_path: Path) -> None:
    with guard, pytest.raises(StrictViolationError):
        os.open(tmp_path / "raw", os.O_CREAT | os.O_WRONLY)
    assert not (tmp_path / "raw").exists()


def test_removing_renaming_and_creating_directories_are_blocked(
    guard: StrictGuard, tmp_path: Path
) -> None:
    victim = tmp_path / "keep.txt"
    victim.write_text("data")
    with guard:
        with pytest.raises(StrictViolationError):
            os.remove(victim)
        with pytest.raises(StrictViolationError):
            os.rename(victim, tmp_path / "moved.txt")
        with pytest.raises(StrictViolationError):
            os.mkdir(tmp_path / "newdir")
    assert victim.read_text() == "data"
    assert not (tmp_path / "newdir").exists()
    assert [v.event for v in guard.violations] == ["os.remove", "os.rename", "os.mkdir"]


def test_temporary_files_are_blocked(guard: StrictGuard) -> None:
    with guard:
        with pytest.raises(StrictViolationError):
            tempfile.mkstemp()
        with pytest.raises(StrictViolationError):
            tempfile.mkdtemp()
        with pytest.raises(StrictViolationError):
            tempfile.TemporaryFile()  # noqa: SIM115


def test_the_null_device_and_descriptors_are_not_disk_writes(guard: StrictGuard) -> None:
    with guard:
        with open(os.devnull, "w") as null:
            null.write("discarded")
        fd = os.dup(1)  # duplicating an existing descriptor creates nothing
        os.close(fd)
    assert guard.violations == []


def test_a_disarmed_guard_lets_everything_through(guard: StrictGuard, tmp_path: Path) -> None:
    with guard:
        pass
    (tmp_path / "ok.txt").write_text("fine")
    assert guard.violations == []


# ---------------------------------------------------------------- G3: network


@pytest.mark.parametrize(
    "event", ["socket.getaddrinfo", "socket.gethostbyname", "socket.connect", "socket.sendto"]
)
def test_g3_every_network_event_is_blocked(guard: StrictGuard, event: str) -> None:
    # Raised through sys.audit so that no real lookup or connection is ever attempted.
    with guard, pytest.raises(StrictViolationError, match="network"):
        sys.audit(event, "example.invalid", 80)
    assert guard.violations[0].kind == "network"
    assert guard.violations[0].event == event


def _run_strict(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=cwd, check=False
    )


def test_g3_the_default_handler_ends_the_process_with_a_clear_message(tmp_path: Path) -> None:
    result = _run_strict(
        "from pipeline.guards import StrictGuard\n"
        "StrictGuard().arm()\n"
        "import socket\n"
        "s = socket.socket()\n"
        "s.connect(('127.0.0.1', 9))\n"
        "print('not reached')\n",
        tmp_path,
    )
    assert result.returncode == EXIT_CODE_STRICT_VIOLATION
    assert "strict mode: network access blocked" in result.stderr
    assert "not reached" not in result.stdout


def test_g3_a_write_open_ends_the_process_and_creates_no_file(tmp_path: Path) -> None:
    result = _run_strict(
        "from pipeline.guards import StrictGuard\n"
        "StrictGuard().arm()\n"
        "open('leak.txt', 'w').write('secret')\n",
        tmp_path,
    )
    assert result.returncode == EXIT_CODE_STRICT_VIOLATION
    assert "strict mode: disk access blocked" in result.stderr
    assert "leak.txt" in result.stderr
    assert not (tmp_path / "leak.txt").exists()


def test_abort_process_is_the_default_handler() -> None:
    assert StrictGuard.__init__.__defaults__ == (abort_process,)


def test_violation_message_names_the_operation() -> None:
    message = Violation("disk", "os.remove", "C:/x").message()
    assert "os.remove" in message
    assert "C:/x" in message


# ---------------------------------------------------------------- G2


@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_g2_sockets_cannot_be_created_in_the_test_suite() -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket()


# ---------------------------------------------------------------- G1


def test_g1_the_whole_capture_run_writes_nothing_and_uses_no_network(
    no_disk_writes: list[Violation],
) -> None:
    rig = build()
    play(
        rig,
        stream_of(
            silence(500, 48_000),
            speech(900, 48_000),
            silence(1200, 48_000),
            speech(700, 48_000),
            silence(1200, 48_000),
        ),
    )
    rig.factory.current.fail()
    rig.session.tick()  # a reconnect is part of the run
    rig.session.tick()
    rig.session.stop()
    assert len(collect(rig.sub)) >= 2
    assert no_disk_writes == []


# ---------------------------------------------------------------- classification (no hook)


@pytest.mark.parametrize(
    ("args", "blocked"),
    [
        (("a.txt", "w", 0), True),
        (("a.txt", "wb", 0), True),
        (("a.txt", "a", 0), True),
        (("a.txt", "x", 0), True),
        (("a.txt", "r+", 0), True),
        (("a.txt", "r", 0), False),
        (("a.txt", "rb", 0), False),
        (("a.txt", None, os.O_RDONLY), False),
        (("a.txt", None, os.O_WRONLY), True),
        (("a.txt", None, os.O_RDWR), True),
        (("a.txt", None, os.O_CREAT), True),
        (("a.txt", None, os.O_APPEND), True),
        (("a.txt", None, os.O_TRUNC), True),
        ((3, "w", 0), False),  # an existing descriptor
        ((os.devnull, "w", 0), False),
        (("nul", "wb", 0), False),
        (("CONOUT$", "w", 0), False),
        ((b"bytes.txt", "w", 0), True),
        ((Path("p.txt"), "w", 0), True),
    ],
)
def test_classification_of_open_events(args: tuple[object, ...], blocked: bool) -> None:
    violation = classify("open", args)
    assert (violation is not None) is blocked
    if violation is not None:
        assert violation.kind == "disk"


@pytest.mark.parametrize(
    "event", ["os.remove", "os.rename", "os.rmdir", "os.mkdir", "tempfile.mkstemp"]
)
def test_classification_of_other_disk_events(event: str) -> None:
    violation = classify(event, ("some/path",))
    assert violation == Violation("disk", event, "some/path")


def test_classification_of_network_events_hides_socket_objects() -> None:
    class FakeSocket:
        def fileno(self) -> int:
            return 1

    violation = classify("socket.connect", (FakeSocket(), ("10.0.0.1", 80)))
    assert violation is not None
    assert violation.kind == "network"
    assert "10.0.0.1" in violation.detail
    assert "FakeSocket" not in violation.detail


def test_a_violation_without_path_arguments_is_still_described() -> None:
    violation = classify("os.remove", ())
    assert violation is not None
    assert violation.detail == ""


def test_abort_process_prints_the_reason_and_exits_with_the_strict_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", exits.append)
    abort_process(Violation("disk", "open", "C:/leak.txt"))
    assert exits == [EXIT_CODE_STRICT_VIOLATION]
    assert "C:/leak.txt" in capsys.readouterr().err
