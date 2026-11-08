"""Doubles for the window controller: a scriptable pipeline and a wiring that builds it."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace

from fakes import FakeDeviceList, make_device
from test_ui_model import capture_stats, device, output_stats

from pipeline.app import BusStats, DeviceState, PipelineStateError, UnknownTargetError
from pipeline.bus import Bus, ChunkBus
from pipeline.capture import CaptureStats
from pipeline.config import SOURCE_LANGUAGE, SUPPORTED_TARGETS, ConfigError, Settings
from pipeline.devices import DeviceInfo
from pipeline.lanes import TranslationBus
from pipeline.metrics import Metrics
from pipeline.output import OutputStats
from pipeline.transcribe import TextBus
from pipeline.ui.controller import BuiltPipeline

MIC = "Microphone (V380 USB)"
MIC_B = "Microphone (Backup Mic)"
REALTEK = "Speakers (Realtek Audio)"
HS100W = "Headset (HS100W)"

UI_DEVICES = (
    make_device(MIC, index=1, inputs=2),
    make_device(MIC_B, index=2, inputs=1),
    make_device(REALTEK, index=3, inputs=0, outputs=2),
    make_device(HS100W, index=4, inputs=0, outputs=2),
    make_device("Speakers (MME only)", index=5, inputs=0, outputs=2, host_api="MME"),
)


class ControlPipeline:
    """What the controller calls on a ``Pipeline``, recorded and scriptable.

    ``log`` receives ``(method, args, thread_name)``; ``fail`` maps a method name to an
    exception raised by it; ``gates`` maps a method name to an event the call waits for, so a
    test can hold the worker thread in the middle of a call.
    """

    def __init__(self, settings: Settings, log: list[tuple[str, tuple[object, ...], str]]) -> None:
        self._settings = settings
        self.log = log
        self.fail: dict[str, Exception] = {}
        self.gates: dict[str, threading.Event] = {}
        self.absent_outputs: set[str] = set()
        self.metrics = Metrics()
        self.text_bus: TextBus = TextBus()
        self.translation_bus: TranslationBus = TranslationBus()
        self.chunk_bus: ChunkBus = Bus()
        self.running = False
        self.stopped = False
        self.position_samples = 0
        self.input_st = DeviceState.DISCONNECTED
        self.output_st = {t: DeviceState.DISCONNECTED for t in settings.targets}
        self.chunk_bus.subscribe("whisper", maxsize=4)
        for target in settings.targets:
            self.text_bus.subscribe(f"lane-{target}", maxsize=2)

    # --- scripted behaviour ---------------------------------------------------------

    def _call(self, name: str, *args: object) -> None:
        self.log.append((name, args, threading.current_thread().name))
        gate = self.gates.get(name)
        if gate is not None:
            assert gate.wait(10.0), f"gate of {name} was never opened"
        error = self.fail.get(name)
        if error is not None:
            raise error

    def _need_running(self) -> None:
        if not self.running:
            raise PipelineStateError("the pipeline is not running")

    # --- PipelineView -------------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def targets(self) -> tuple[str, ...]:
        return self._settings.targets

    def capture_stats(self) -> CaptureStats:
        return capture_stats(connected=self.input_st is DeviceState.CONNECTED)

    def output_stats(self, target: str) -> OutputStats:
        if target not in self.output_st:
            raise UnknownTargetError(target)
        return output_stats()

    def bus_stats(self) -> BusStats:
        return BusStats(chunks=self.chunk_bus.stats(), text=self.text_bus.stats())

    def input_device(self) -> DeviceInfo | None:
        return device("Mic One") if self.input_st is DeviceState.CONNECTED else None

    def output_device(self, target: str) -> DeviceInfo | None:
        return device(f"Out-{target}") if self.output_st[target] is DeviceState.CONNECTED else None

    def input_state(self) -> DeviceState:
        return self.input_st

    def output_state(self, target: str) -> DeviceState:
        return self.output_st[target]

    # --- control ------------------------------------------------------------------------

    def start(self, *, lenient: bool = False) -> tuple[str, ...]:
        self._call("start", lenient)
        self.running = True
        self.input_st = DeviceState.CONNECTED
        for target in self.output_st:
            away = target in self.absent_outputs
            self.output_st[target] = DeviceState.DISCONNECTED if away else DeviceState.CONNECTED
        return tuple(t for t in self._settings.targets if t in self.absent_outputs)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._call("stop")
        self.running = False
        self.stopped = True
        self.input_st = DeviceState.DISCONNECTED
        self.output_st = dict.fromkeys(self.output_st, DeviceState.DISCONNECTED)
        self.text_bus.close()
        self.translation_bus.close()

    def connect_input(self) -> None:
        self._need_running()
        self._call("connect_input")
        self.input_st = DeviceState.CONNECTED

    def disconnect_input(self) -> None:
        self._need_running()
        self._call("disconnect_input")
        self.input_st = DeviceState.DISCONNECTED

    def set_input_device(self, device_query: str) -> None:
        self._call("set_input_device", device_query)
        self._settings = replace(
            self._settings, capture=replace(self._settings.capture, device_query=device_query)
        )

    def connect_output(self, target: str) -> None:
        self._need_running()
        self._call("connect_output", target)
        self.output_st[target] = DeviceState.CONNECTED

    def disconnect_output(self, target: str) -> None:
        self._need_running()
        self._call("disconnect_output", target)
        self.output_st[target] = DeviceState.DISCONNECTED

    def set_output(self, target: str, device_query: str) -> None:
        self._call("set_output", target, device_query)
        routes = tuple(
            replace(route, output=replace(route.output, device_query=device_query))
            if route.target == target
            else route
            for route in self._settings.routes
        )
        self._settings = replace(self._settings, routes=routes)

    def set_output_target(self, old_target: str, new_target: str) -> None:
        self._call("set_output_target", old_target, new_target)
        if new_target == old_target:
            return
        if new_target == SOURCE_LANGUAGE or new_target not in SUPPORTED_TARGETS:
            raise ConfigError(f"unsupported target language {new_target!r}")
        if new_target in self._settings.targets:
            raise ConfigError(f"target language {new_target!r} is already used by another lane")
        self._settings = replace(
            self._settings,
            targets=tuple(new_target if t == old_target else t for t in self._settings.targets),
            routes=tuple(
                replace(route, target=new_target) if route.target == old_target else route
                for route in self._settings.routes
            ),
        )
        self.output_st = {
            (new_target if t == old_target else t): state for t, state in self.output_st.items()
        }


class FakeWiring:
    """Builds ``ControlPipeline`` objects; records what the controller did with them."""

    tts_notice: str | None = "TTS not installed: outputs are silent"

    def __init__(self) -> None:
        self.listing = FakeDeviceList(UI_DEVICES)
        self.log: list[tuple[str, tuple[object, ...], str]] = []
        self.built: list[ControlPipeline] = []
        self.closed: list[ControlPipeline] = []
        self.build_error: Exception | None = None
        self.build_gate: threading.Event | None = None
        self.problem_list: tuple[str, ...] = ()
        self.configure: Callable[[ControlPipeline], None] = lambda pipeline: None

    def devices(self) -> FakeDeviceList:
        return self.listing

    def problems(self, settings: Settings) -> tuple[str, ...]:
        return self.problem_list

    def build(self, settings: Settings) -> BuiltPipeline:
        self.log.append(("build", (settings,), threading.current_thread().name))
        if self.build_gate is not None:
            assert self.build_gate.wait(10.0), "the build gate was never opened"
        if self.build_error is not None:
            raise self.build_error
        pipeline = ControlPipeline(settings, self.log)
        self.configure(pipeline)
        self.built.append(pipeline)
        return BuiltPipeline(pipeline, close=lambda: self.closed.append(pipeline))

    def calls(self, name: str) -> list[tuple[object, ...]]:
        return [args for method, args, _ in self.log if method == name]

    def threads(self, name: str) -> set[str]:
        return {thread for method, _, thread in self.log if method == name}
