"""Window controller: everything the window does that is not drawing, and no Tk.

The window runs in the main thread and polls ``WindowController.tick()`` from its timer. Every
call that can take long (loading models, opening or closing a device, stopping and rebuilding
the pipeline, loading a translator) runs on one worker thread (``JobRunner``) and reports back
through a queue that only ``tick()`` drains, so the pipeline's threads and the worker never
touch a widget and the window never freezes. While a call is pending ``tick()`` reports it as
``busy`` and the controller refuses further actions (the window disables its buttons).

Sessions. The window needs something to show before any pipeline exists, and a pipeline
cannot start without its microphone, so the controller always holds one session:

* *standby*: a ``StandbyPipeline`` that is not running. Choices (devices, language, settings)
  are only stored in its settings. "Connect microphone" builds the real pipeline.
* *active*: a built and started ``Pipeline``. Device buttons and language changes go to it on
  the worker thread.

A ``Pipeline`` is single use, so applying a setting that needs a restart is: stop it, build a
new one through ``Wiring.build`` and start it, with a new ``ViewModel`` (``generation`` goes up
and the window clears its transcript). When that fails the controller falls back to standby
with the new settings, keeps the message, and the user can connect again.

Device choices made at run time live in ``Pipeline.settings``; the controller reads them from
there, so a restart starts from what the user last chose, not from the command line.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from pipeline.app import BusStats, DeviceState, UnknownTargetError
from pipeline.capture import CaptureStats
from pipeline.config import ConfigError, Settings, is_unselected
from pipeline.devices import DeviceInfo, DeviceListing
from pipeline.lanes import TranslationBus
from pipeline.metrics import Metrics
from pipeline.output import OutputStats
from pipeline.transcribe import TextBus
from pipeline.ui.model import PipelineView, Snapshot, ViewModel
from pipeline.ui.present import input_device_names, output_device_names
from pipeline.ui.settings_schema import (
    GENERAL_ERROR_KEY,
    SettingsValidationError,
    build_settings,
    needs_restart,
)

NoticeLevel = Literal["info", "warn", "error"]

_JOIN_TIMEOUT_S = 15.0  # a restart in flight (model load) may take a few seconds to finish


def describe_error(error: BaseException) -> str:
    """One line for the window: the message of a known error, else its type and message.

    The messages of this program's own errors already say what to do. Utterance text never
    appears in them: control calls and builds do not handle it.
    """
    message = str(error)
    name = type(error).__name__
    if type(error).__module__.startswith("pipeline."):
        return message or name
    return f"{name}: {message}" if message else name


# --- worker ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobResult:
    """What a job returned, or the exception that ended it."""

    value: object
    error: Exception | None


class JobRunner:
    """One worker thread that runs jobs in order and queues their results for ``drain()``.

    Nothing here blocks the caller except ``close()``. The thread is a daemon: a job stuck in
    a driver call cannot keep the process alive after the window is gone.
    """

    def __init__(self, name: str = "window-worker") -> None:
        """Start the thread."""
        self._jobs: queue.Queue[Callable[[], object] | None] = queue.Queue()
        self._results: queue.Queue[JobResult] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, job: Callable[[], object]) -> None:
        """Queue a job.

        Raises:
            RuntimeError: The runner is closed.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("the job runner is closed")
            self._jobs.put(job)

    def drain(self) -> list[JobResult]:
        """Return the results that are ready, oldest first, without waiting."""
        results: list[JobResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def close(self, timeout_s: float) -> bool:
        """Finish the queued jobs and stop the thread; safe to call again.

        Returns:
            True if the thread ended within ``timeout_s``.
        """
        with self._lock:
            if not self._closed:
                self._closed = True
                self._jobs.put(None)
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                result = JobResult(job(), None)
            except Exception as error:  # reported to the window; the worker must survive
                result = JobResult(None, error)
            self._results.put(result)


# --- state for the window --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Notice:
    """A message for the status bar.

    Attributes:
        level: ``info``, ``warn`` or ``error``.
        text: The message (never utterance text).
        seq: Grows with every new notice, so the window can tell a repeated message from an
            old one.
    """

    level: NoticeLevel
    text: str
    seq: int


@dataclass(frozen=True, slots=True)
class WindowState:
    """Everything the window draws on one tick.

    Attributes:
        snapshot: The view-model's snapshot (all stages idle while in standby).
        settings: Current settings, including device and language choices made at run time.
        generation: Grows when a new pipeline took over; the window clears its transcript.
        busy: What the worker is doing, or ``None`` when idle.
        notice: The latest message, if any.
        problems: Missing models or files that stop the pipeline from starting.
        input_devices: WASAPI input device names (empty until the first scan finished).
        output_devices: WASAPI output device names.
        session_active: A pipeline is built and started (outputs can be connected).
        tts_notice: A banner about speech synthesis, or ``None``.
    """

    snapshot: Snapshot
    settings: Settings
    generation: int
    busy: str | None
    notice: Notice | None
    problems: tuple[str, ...]
    input_devices: tuple[str, ...]
    output_devices: tuple[str, ...]
    session_active: bool
    tts_notice: str | None


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """The answer to ``apply_settings``.

    Attributes:
        accepted: The request was taken up (False while another call is in progress).
        errors: Messages by field key (``GENERAL_ERROR_KEY`` for a general one).
        restarting: A restart was queued; the window shows its busy state.
        message: Why a request was not accepted.
    """

    accepted: bool = True
    errors: dict[str, str] = field(default_factory=dict)
    restarting: bool = False
    message: str = ""


# --- what the controller needs from the outside -----------------------------------------------


class ControlledPipeline(PipelineView, Protocol):
    """The part of ``Pipeline`` the controller drives (tests give it a fake)."""

    def start(self, *, lenient: bool = False) -> tuple[str, ...]:
        """Open the devices and start the threads."""
        ...

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop everything."""
        ...

    def connect_input(self) -> None:
        """Open the microphone again."""
        ...

    def disconnect_input(self) -> None:
        """Close the microphone."""
        ...

    def set_input_device(self, device_query: str) -> None:
        """Choose the microphone by name."""
        ...

    def connect_output(self, target: str) -> None:
        """Open an output again."""
        ...

    def disconnect_output(self, target: str) -> None:
        """Close an output."""
        ...

    def set_output(self, target: str, device_query: str) -> None:
        """Choose the device of an output by name."""
        ...

    def set_output_target(self, old_target: str, new_target: str) -> None:
        """Change the language of one output."""
        ...


@dataclass(frozen=True, slots=True)
class BuiltPipeline:
    """A pipeline that is built but not started, and how to release what it does not own.

    Attributes:
        pipeline: The pipeline.
        close: Releases resources the pipeline does not stop itself (the Smart Turn thread);
            called once after ``pipeline.stop()``.
    """

    pipeline: ControlledPipeline
    close: Callable[[], None]


class Wiring(Protocol):
    """Builds the real parts; the window never imports a model or a device library itself."""

    @property
    def tts_notice(self) -> str | None:
        """A banner for the window when speech synthesis is a placeholder, else ``None``."""
        ...

    def devices(self) -> DeviceListing:
        """The device list source."""
        ...

    def problems(self, settings: Settings) -> tuple[str, ...]:
        """Missing models or files for ``settings``; empty when it can start."""
        ...

    def build(self, settings: Settings) -> BuiltPipeline:
        """Build (but do not start) a pipeline; slow, so it runs on the worker thread."""
        ...


# --- standby ---------------------------------------------------------------------------------

_IDLE_CAPTURE = CaptureStats(
    connected=False,
    callbacks=0,
    input_overflow=0,
    ring_lost_frames=0,
    max_callback_ms=0.0,
    reconnects=0,
    losses=0,
    last_loss_reason="",
    last_error="",
    chunks=0,
    finalization_latency_ms=(),
)
_IDLE_OUTPUT = OutputStats(
    connected=False,
    callbacks=0,
    underruns=0,
    underrun_frames=0,
    driver_underflows=0,
    frames_played=0,
    dropped_frames=0,
    queued_ms=0.0,
    max_callback_ms=0.0,
    reconnects=0,
    losses=0,
    last_loss_reason="",
    last_error="",
)


class StandbyPipeline:
    """A ``PipelineView`` of a pipeline that does not exist yet: stopped, nothing connected.

    It lets the window draw its panels and keep the user's choices before the first connect.
    """

    def __init__(self, settings: Settings) -> None:
        """Create it with the settings the window starts from."""
        self._settings = settings
        self.metrics = Metrics()
        self.text_bus: TextBus = TextBus()
        self.translation_bus: TranslationBus = TranslationBus()

    @property
    def settings(self) -> Settings:
        """The settings the next pipeline will be built from."""
        return self._settings

    def set_settings(self, settings: Settings) -> None:
        """Replace the stored settings."""
        self._settings = settings

    @property
    def targets(self) -> tuple[str, ...]:
        """Target languages in lane order."""
        return self._settings.targets

    @property
    def running(self) -> bool:
        """Always False."""
        return False

    @property
    def position_samples(self) -> int:
        """Always 0."""
        return 0

    def capture_stats(self) -> CaptureStats:
        """Zero counters."""
        return _IDLE_CAPTURE

    def output_stats(self, target: str) -> OutputStats:
        """Zero counters."""
        if target not in self._settings.targets:
            raise UnknownTargetError(target)
        return _IDLE_OUTPUT

    def bus_stats(self) -> BusStats:
        """No subscribers."""
        return BusStats(chunks={}, text={})

    def input_device(self) -> DeviceInfo | None:
        """No device is open."""
        return None

    def output_device(self, target: str) -> DeviceInfo | None:
        """No device is open."""
        return None

    def input_state(self) -> DeviceState:
        """Always disconnected."""
        return DeviceState.DISCONNECTED

    def output_state(self, target: str) -> DeviceState:
        """Always disconnected."""
        return DeviceState.DISCONNECTED

    def close(self) -> None:
        """Close its buses."""
        self.text_bus.close()
        self.translation_bus.close()


# --- the controller --------------------------------------------------------------------------


@dataclass(slots=True)
class _Session:
    """The pipeline the window currently shows, with its view-model."""

    pipeline: PipelineView
    vm: ViewModel
    built: BuiltPipeline | None = None  # None: standby
    standby: StandbyPipeline | None = None

    @property
    def active(self) -> bool:
        return self.built is not None


@dataclass(slots=True)
class _Outcome:
    """What a worker job that builds a session hands back."""

    session: _Session
    notice: tuple[NoticeLevel, str] | None = None


@dataclass(slots=True)
class _Pending:
    label: str
    done: Callable[[JobResult], None]


def _with_language(settings: Settings, old: str, new: str) -> Settings:
    """``settings`` with the lane of ``old`` switched to ``new`` (the device stays).

    Raises:
        ConfigError: ``old`` has no lane, or ``new`` is unsupported, the source or taken.
    """
    if old not in settings.targets:
        raise ConfigError(f"no output for language {old!r}")
    return replace(
        settings,
        targets=tuple(new if target == old else target for target in settings.targets),
        routes=tuple(
            replace(route, target=new) if route.target == old else route
            for route in settings.routes
        ),
    )


class WindowController:
    """Drives the pipeline for the window; every public method is called from the UI thread."""

    def __init__(
        self,
        settings: Settings,
        wiring: Wiring,
        *,
        auto_connect: bool = False,
        clock: Callable[[], float] = time.monotonic,
        view_model_factory: Callable[[PipelineView], ViewModel] | None = None,
        join_timeout_s: float = _JOIN_TIMEOUT_S,
    ) -> None:
        """Create the controller in standby; nothing is loaded or opened until asked.

        Args:
            settings: Where the window starts (devices may be ``UNSELECTED_DEVICE``).
            wiring: Builds pipelines (real or fake).
            auto_connect: ``start()`` also connects the microphone when one is chosen and no
                model is missing.
            clock: Monotonic seconds for the view-model's warning colours.
            view_model_factory: Builds the view-model of a pipeline; ``ViewModel`` by default.
            join_timeout_s: How long ``shutdown()`` waits for a call in progress.
        """
        self._wiring = wiring
        self._auto_connect = auto_connect
        self._join_timeout_s = join_timeout_s
        self._vm_factory = view_model_factory or (lambda view: ViewModel(view, clock=clock))
        self._hide_text = settings.hide_text
        self._runner = JobRunner()
        self._pending: deque[_Pending] = deque()
        self._notice: Notice | None = None
        self._notice_seq = 0
        self._generation = 0
        self._devices: tuple[DeviceInfo, ...] = ()
        self._closed = False
        self._problems = wiring.problems(settings)
        self._session = self._new_standby(settings)
        self._session.vm.set_hide_text(self._hide_text)

    # --- reading -------------------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        """Current settings, with the live ``hide_text`` switch.

        The pipeline's settings already include device and language choices made at run time.
        """
        return replace(self._session.pipeline.settings, hide_text=self._hide_text)

    def tick(self) -> WindowState:
        """Take finished worker results, poll the view-model and return what to draw.

        Never blocks: the view-model reads queues with a zero timeout and the pipeline's
        statistics take no lock.
        """
        self._drain()
        session = self._session
        return WindowState(
            snapshot=session.vm.poll(),
            settings=self.settings,
            generation=self._generation,
            busy=self._pending[0].label if self._pending else None,
            notice=self._notice,
            problems=self._problems,
            input_devices=input_device_names(self._devices),
            output_devices=output_device_names(self._devices),
            session_active=session.active,
            tts_notice=self._wiring.tts_notice,
        )

    # --- lifecycle -----------------------------------------------------------------------

    def start(self) -> None:
        """Scan the devices on the worker and, with ``auto_connect``, connect the microphone."""
        self._submit(
            "Scanning audio devices...", lambda: self._scan(reinitialize=False), self._on_scan
        )
        settings = self.settings
        if (
            self._auto_connect
            and not is_unselected(settings.capture.device_query)
            and not self._problems
        ):
            self._begin_session(settings)

    def shutdown(self) -> None:
        """Stop the pipeline and the worker; safe to call again.

        Waits for a call in progress (a session that finishes meanwhile is stopped too).
        """
        if self._closed:
            return
        self._closed = True
        if self._runner.close(self._join_timeout_s):
            self._drain()
        session = self._session
        if session.built is not None:
            self._release(session.built)
        self._close_session(session)

    # --- devices -------------------------------------------------------------------------

    def refresh_devices(self) -> bool:
        """Scan the devices again; PortAudio is restarted only while no pipeline is running."""
        if not self._idle():
            return False
        reinitialize = not self._session.active
        self._submit(
            "Scanning audio devices...",
            lambda: self._scan(reinitialize=reinitialize),
            self._on_scan,
        )
        return True

    def connect_input(self) -> bool:
        """Start the pipeline (standby) or reopen the microphone (running)."""
        if not self._idle():
            return False
        settings = self.settings
        if is_unselected(settings.capture.device_query):
            self._say("error", "Select a microphone first.")
            return False
        if not self._session.active:
            return self._begin_session(settings)
        pipeline = self._controlled()
        self._call("Connecting the microphone...", pipeline.connect_input)
        return True

    def disconnect_input(self) -> bool:
        """Close the microphone; the pipeline and the outputs keep running."""
        if not self._idle() or not self._session.active:
            return False
        self._call("Disconnecting the microphone...", self._controlled().disconnect_input)
        return True

    def select_input_device(self, name: str) -> bool:
        """Choose the microphone (switches at once while connected)."""
        if not self._idle():
            return False
        if not self._session.active:
            return self._edit_standby(
                lambda s: replace(s, capture=replace(s.capture, device_query=name))
            )
        pipeline = self._controlled()
        self._call("Switching the microphone...", lambda: pipeline.set_input_device(name))
        return True

    def connect_output(self, target: str) -> bool:
        """Reopen the output of ``target``."""
        if not self._idle():
            return False
        if not self._session.active:
            self._say("error", "Connect the microphone first: the outputs start with it.")
            return False
        query = next(
            (r.output.device_query for r in self.settings.routes if r.target == target), ""
        )
        if is_unselected(query):
            self._say("error", f"Select an output device for {target} first.")
            return False
        pipeline = self._controlled()
        self._call(f"Connecting output {target}...", lambda: pipeline.connect_output(target))
        return True

    def disconnect_output(self, target: str) -> bool:
        """Close the output of ``target``; the other outputs keep playing."""
        if not self._idle() or not self._session.active:
            return False
        pipeline = self._controlled()
        self._call(f"Disconnecting output {target}...", lambda: pipeline.disconnect_output(target))
        return True

    def select_output_device(self, target: str, name: str) -> bool:
        """Choose the device of one output (switches at once while connected)."""
        if not self._idle():
            return False
        if not self._session.active:
            return self._edit_standby(lambda s: _with_output(s, target, name))
        pipeline = self._controlled()
        self._call(f"Switching output {target}...", lambda: pipeline.set_output(target, name))
        return True

    def set_language(self, old: str, new: str) -> bool:
        """Change the language of one output; only that lane is rebuilt while running."""
        if not self._idle():
            return False
        if new == old:
            return True
        if not self._session.active:
            return self._edit_standby(lambda s: _with_language(s, old, new))
        pipeline = self._controlled()
        self._call(f"Loading the {new} translator...", lambda: pipeline.set_output_target(old, new))
        return True

    def export_report(self) -> str:
        """The plain-text report for the clipboard (honors ``hide_text``); nothing is written."""
        return self._session.vm.export_text()

    def notify(self, level: NoticeLevel, text: str) -> int:
        """Show a message from the window itself (for example a failed clipboard copy).

        Returns:
            The ``seq`` of the new notice.
        """
        self._say(level, text)
        return self._notice_seq

    def set_hide_text(self, hide: bool) -> None:
        """Mask or show the transcript at once; allowed while a call is in progress."""
        self._hide_text = hide
        self._session.vm.set_hide_text(hide)

    # --- settings ------------------------------------------------------------------------

    def apply_settings(self, values: Mapping[str, object]) -> ApplyResult:
        """Validate ``values`` and apply them: live when possible, else by a restart.

        Validation errors come back at once with their field keys. A restart runs on the
        worker; the window sees it as ``busy`` and, when it ends, as a new ``generation``.
        """
        if self._closed or self._pending:
            label = self._pending[0].label if self._pending else "the window is closing"
            return ApplyResult(accepted=False, message=f"Busy: {label}")
        current = self.settings
        try:
            new = build_settings(current, values)
        except SettingsValidationError as error:
            self._say("error", "Some settings are not valid; see the marked fields.")
            return ApplyResult(errors=dict(error.errors))
        if new == current:
            self._say("info", "No changes to apply.")
            return ApplyResult()
        if new.hide_text != current.hide_text:
            self.set_hide_text(new.hide_text)
        if not needs_restart(current, new):
            self._say("info", "Applied.")
            return ApplyResult()
        if not self._session.active:
            # Nothing is running, so nothing can be lost: store the choice even if a model for
            # it is missing (the banner shows it) and let the user fix that before connecting.
            self._standby().set_settings(new)
            self._problems = self._wiring.problems(new)
            self._say("info", "Settings saved; they take effect when the microphone is connected.")
            return ApplyResult()
        problems = self._wiring.problems(new)
        if problems:  # refuse before the working pipeline is stopped
            message = "Cannot apply: " + "; ".join(problems)
            self._say("error", message)
            return ApplyResult(errors={GENERAL_ERROR_KEY: message})
        self._restart(new)
        return ApplyResult(restarting=True)

    # --- worker plumbing -----------------------------------------------------------------

    def _idle(self) -> bool:
        return not self._closed and not self._pending

    def _controlled(self) -> ControlledPipeline:
        built = self._session.built
        assert built is not None
        return built.pipeline

    def _standby(self) -> StandbyPipeline:
        standby = self._session.standby
        assert standby is not None
        return standby

    def _say(self, level: NoticeLevel, text: str) -> None:
        self._notice_seq += 1
        self._notice = Notice(level, text, self._notice_seq)

    def _submit(
        self, label: str, job: Callable[[], object], done: Callable[[JobResult], None]
    ) -> None:
        self._pending.append(_Pending(label, done))
        self._runner.submit(job)

    def _drain(self) -> None:
        for result in self._runner.drain():
            self._pending.popleft().done(result)

    def _call(self, label: str, action: Callable[[], None]) -> None:
        """Run a control call of the active pipeline on the worker; show its error, if any."""
        self._notice = None
        self._submit(label, action, self._on_call)

    def _on_call(self, result: JobResult) -> None:
        if result.error is not None:
            self._say("error", describe_error(result.error))

    def _scan(self, *, reinitialize: bool) -> tuple[DeviceInfo, ...]:
        listing = self._wiring.devices()
        if reinitialize:
            listing.reinitialize()
        return tuple(listing.list_devices())

    def _on_scan(self, result: JobResult) -> None:
        if result.error is not None:
            self._say("error", f"Cannot list the audio devices: {describe_error(result.error)}")
        else:
            assert isinstance(result.value, tuple)
            self._devices = result.value

    # --- sessions ------------------------------------------------------------------------

    def _new_standby(self, settings: Settings) -> _Session:
        standby = StandbyPipeline(settings)
        return _Session(standby, self._vm_factory(standby), None, standby)

    def _edit_standby(self, change: Callable[[Settings], Settings]) -> bool:
        """Store a choice in the standby settings; a refused one becomes a notice."""
        self._notice = None
        try:
            new = change(self._standby().settings)  # the config classes validate on creation
        except ValueError as error:  # ConfigError, CaptureConfigError, OutputConfigError
            self._say("error", describe_error(error))
            return False
        self._standby().set_settings(new)
        self._problems = self._wiring.problems(new)
        return True

    def _begin_session(self, settings: Settings) -> bool:
        if self._problems:
            self._say("error", "Cannot start: " + "; ".join(self._problems))
            return False
        self._notice = None
        self._submit(
            "Loading models and connecting the microphone...",
            lambda: self._open(settings, False, (), "start"),
            self._on_session,
        )
        return True

    def _restart(self, new: Settings) -> None:
        old = self._session
        assert old.built is not None
        pipeline = old.built.pipeline
        off_input = pipeline.input_state() is DeviceState.DISCONNECTED
        off_outputs = tuple(
            target
            for target in pipeline.targets
            if pipeline.output_state(target) is DeviceState.DISCONNECTED
        )
        built = old.built
        self._notice = None

        def job() -> _Outcome:
            self._release(built)
            return self._open(new, off_input, off_outputs, "restart")

        self._submit("Applying settings: restarting the pipeline...", job, self._on_session)

    def _open(
        self, settings: Settings, off_input: bool, off_outputs: tuple[str, ...], action: str
    ) -> _Outcome:
        """Worker: build and start a pipeline; on any failure fall back to standby."""
        try:
            built = self._wiring.build(settings)
        except Exception as error:  # a model that cannot load must not take the window down
            return self._fallback(settings, action, error)
        pipeline = built.pipeline
        try:
            unavailable = pipeline.start(lenient=True)
            if off_input:
                pipeline.disconnect_input()
            for target in off_outputs:
                pipeline.disconnect_output(target)
            vm = self._vm_factory(pipeline)
        except Exception as error:  # start() already closed what it opened
            self._release(built)
            return self._fallback(settings, action, error)
        notice: tuple[NoticeLevel, str] | None = None
        if unavailable:
            notice = (
                "warn",
                f"Output {', '.join(unavailable)} could not be opened and is left disconnected; "
                "choose a device and connect it.",
            )
        elif action == "restart":
            notice = ("info", "Settings applied; the pipeline was restarted.")
        return _Outcome(_Session(pipeline, vm, built, None), notice)

    def _fallback(self, settings: Settings, action: str, error: Exception) -> _Outcome:
        verb = "restart" if action == "restart" else "connect"
        text = f"Could not {verb}: {describe_error(error)}"
        return _Outcome(self._new_standby(settings), ("error", text))

    @staticmethod
    def _release(built: BuiltPipeline) -> None:
        """Stop a pipeline and release what the wiring made for it."""
        try:
            built.pipeline.stop()
        except Exception:  # already tearing down; the caller reports the cause
            pass
        finally:
            built.close()

    def _on_session(self, result: JobResult) -> None:
        if result.error is not None:
            self._say("error", describe_error(result.error))
            return
        outcome = result.value
        assert isinstance(outcome, _Outcome)
        if outcome.notice is not None:
            self._say(*outcome.notice)
        old = self._session
        self._session = outcome.session
        self._session.vm.set_hide_text(self._hide_text)
        self._generation += 1
        self._problems = self._wiring.problems(self.settings)
        self._close_session(old)

    @staticmethod
    def _close_session(session: _Session) -> None:
        session.vm.close()
        if session.standby is not None:
            session.standby.close()


def _with_output(settings: Settings, target: str, name: str) -> Settings:
    if target not in settings.targets:
        raise ConfigError(f"no output for language {target!r}")
    return replace(
        settings,
        routes=tuple(
            replace(route, output=replace(route.output, device_query=name))
            if route.target == target
            else route
            for route in settings.routes
        ),
    )


__all__ = [
    "ApplyResult",
    "BuiltPipeline",
    "ControlledPipeline",
    "JobResult",
    "JobRunner",
    "Notice",
    "StandbyPipeline",
    "WindowController",
    "WindowState",
    "Wiring",
    "describe_error",
]
