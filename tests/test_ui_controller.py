"""Window controller without Tk: standby, connecting, apply and restart, worker thread (WD6-WD9)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from ui_fakes import HS100W, MIC, MIC_B, REALTEK, FakeWiring

from pipeline.app import DeviceState, PipelineStateError
from pipeline.config import (
    UNSELECTED_DEVICE,
    ConfigError,
    Settings,
    is_unselected,
    parse_settings,
)
from pipeline.devices import DeviceNotFoundError, StreamOpenError
from pipeline.ui.controller import (
    JobRunner,
    WindowController,
    WindowState,
    describe_error,
)
from pipeline.ui.model import Health
from pipeline.ui.settings_schema import GENERAL_ERROR_KEY

pytestmark = pytest.mark.unit


def chosen() -> Settings:
    return parse_settings(["--input", "V380", "--route", "de=Realtek,fr=HS100W"])


def empty() -> Settings:
    return parse_settings([], gui=True)


def settle(controller: WindowController, timeout_s: float = 5.0) -> WindowState:
    """Tick until no control call is pending, like the window's timer does."""
    deadline = time.monotonic() + timeout_s
    while True:
        state = controller.tick()
        if state.busy is None:
            return state
        if time.monotonic() > deadline:
            pytest.fail(f"still busy: {state.busy}")
        time.sleep(0.002)


@pytest.fixture
def wiring() -> FakeWiring:
    return FakeWiring()


def make(
    wiring: FakeWiring, settings: Settings | None = None, *, auto_connect: bool = False
) -> WindowController:
    return WindowController(settings or chosen(), wiring, auto_connect=auto_connect)


@pytest.fixture
def controller(wiring: FakeWiring) -> Iterator[WindowController]:
    made = make(wiring)
    yield made
    made.shutdown()


@pytest.fixture
def running(wiring: FakeWiring, controller: WindowController) -> WindowController:
    """A controller with a started pipeline."""
    controller.start()
    settle(controller)
    assert controller.connect_input()
    state = settle(controller)
    assert state.session_active
    return controller


# ---------------------------------------------------------------- WD6 standby


def test_wd6_the_window_starts_with_no_pipeline_and_everything_idle(
    wiring: FakeWiring, controller: WindowController
) -> None:
    state = controller.tick()
    assert not state.session_active
    assert not state.snapshot.running
    assert state.snapshot.targets == ("de", "fr")
    assert {stage.health for stage in state.snapshot.stages} == {Health.IDLE}
    assert state.snapshot.input.state is DeviceState.DISCONNECTED
    assert state.busy is None
    assert state.generation == 0
    assert wiring.built == []  # nothing is built until the user connects


def test_wd6_the_window_starts_even_with_no_device_chosen(wiring: FakeWiring) -> None:
    controller = make(wiring, empty())
    try:
        state = controller.tick()
        assert is_unselected(state.settings.capture.device_query)
        assert not state.session_active
        assert state.snapshot.targets == ("de", "fr")
    finally:
        controller.shutdown()


def test_wd6_start_lists_the_wasapi_devices_on_the_worker(
    wiring: FakeWiring, controller: WindowController
) -> None:
    controller.start()
    state = settle(controller)
    assert state.input_devices == (MIC_B, MIC)  # alphabetical; the MME speakers are not offered
    assert state.output_devices == (HS100W, REALTEK)
    assert wiring.listing.list_calls == 1
    assert wiring.built == []


def test_wd6_a_failing_device_scan_is_an_error_not_a_crash(
    wiring: FakeWiring, controller: WindowController
) -> None:
    def boom() -> Any:
        raise OSError("PortAudio is not available")

    wiring.listing.list_devices = boom  # type: ignore[method-assign]
    controller.start()
    state = settle(controller)
    assert state.notice is not None
    assert state.notice.level == "error"
    assert "PortAudio is not available" in state.notice.text
    assert state.input_devices == ()


def test_wd6_refresh_rescans_and_reinitializes_only_without_a_pipeline(
    wiring: FakeWiring, running: WindowController
) -> None:
    before = wiring.listing.reinitialize_calls
    assert running.refresh_devices()
    settle(running)
    assert wiring.listing.reinitialize_calls == before  # PortAudio must not restart under streams
    running.shutdown()
    standby = make(wiring)
    try:
        assert standby.refresh_devices()
        settle(standby)
        assert wiring.listing.reinitialize_calls == before + 1
    finally:
        standby.shutdown()


def test_wd6_missing_models_are_shown_and_connecting_is_refused(wiring: FakeWiring) -> None:
    wiring.problem_list = ("model missing: models/silero_vad.onnx",)
    controller = make(wiring)
    try:
        assert controller.tick().problems == wiring.problem_list
        assert not controller.connect_input()
        state = controller.tick()
        assert state.notice is not None
        assert state.notice.level == "error"
        assert "models" in state.notice.text.lower()
        assert wiring.built == []
        assert state.busy is None
    finally:
        controller.shutdown()


def test_wd6_connecting_without_a_chosen_microphone_asks_for_one(wiring: FakeWiring) -> None:
    controller = make(wiring, empty())
    try:
        assert not controller.connect_input()
        state = controller.tick()
        assert state.notice is not None
        assert "microphone" in state.notice.text.lower()
        assert wiring.built == []
    finally:
        controller.shutdown()


def test_wd6_choices_made_in_standby_are_only_stored(
    wiring: FakeWiring, controller: WindowController
) -> None:
    assert controller.select_input_device(MIC_B)
    assert controller.select_output_device("fr", REALTEK)
    state = controller.tick()
    assert state.settings.capture.device_query == MIC_B
    by_target = {r.target: r.output.device_query for r in state.settings.routes}
    assert by_target == {"de": "Realtek", "fr": REALTEK}
    assert state.busy is None  # nothing to wait for
    assert wiring.built == []


def test_wd6_a_language_change_in_standby_is_validated_like_the_pipeline_does(
    wiring: FakeWiring, controller: WindowController
) -> None:
    assert not controller.set_language("de", "fr")  # fr is used by the other lane
    notice = controller.tick().notice
    assert notice is not None
    assert notice.level == "error"
    assert "fr" in notice.text
    assert controller.tick().snapshot.targets == ("de", "fr")


def test_wd6_a_single_lane_can_change_language_in_standby(wiring: FakeWiring) -> None:
    controller = make(
        wiring, parse_settings(["--input", "V380", "--route", "de=Realtek", "--targets", "de"])
    )
    try:
        assert controller.set_language("de", "fr")
        state = controller.tick()
        assert state.snapshot.targets == ("fr",)
        assert state.settings.routes[0].target == "fr"
        assert state.settings.routes[0].output.device_query == "Realtek"  # the device stays
    finally:
        controller.shutdown()


def test_wd6_output_buttons_need_a_running_pipeline(
    wiring: FakeWiring, controller: WindowController
) -> None:
    assert not controller.connect_output("de")
    notice = controller.tick().notice
    assert notice is not None
    assert "microphone" in notice.text.lower()


def test_wd6_hide_text_is_live_with_or_without_a_pipeline(
    wiring: FakeWiring, running: WindowController
) -> None:
    running.set_hide_text(True)
    state = running.tick()
    assert state.snapshot.hide_text
    assert state.settings.hide_text
    assert wiring.built[0].settings.hide_text is False  # the pipeline never heard of it


def test_wd6_the_tts_banner_comes_from_the_wiring(
    wiring: FakeWiring, controller: WindowController
) -> None:
    # The real wiring has no banner any more (missing voices are problems); the path stays
    # for any wiring that returns a notice.
    assert controller.tick().tts_notice is None
    wiring.tts_notice = "Speech is a placeholder"
    other = make(wiring)
    try:
        assert other.tick().tts_notice == "Speech is a placeholder"
    finally:
        other.shutdown()


def test_wd16_a_language_change_in_standby_drops_the_voice_name_but_keeps_rate(
    wiring: FakeWiring,
) -> None:
    one_lane = parse_settings(
        [
            "--input",
            "V380",
            "--targets",
            "de",
            "--route",
            "de=Realtek",
            "--tts-voice",
            "de=Hedda",
            "--tts-rate",
            "2",
        ]
    )
    controller = make(wiring, one_lane)
    try:
        assert controller.set_language("de", "fr")
        tts = controller.tick().settings.routes[0].tts
        assert (tts.rate, tts.voice) == (2, None)  # a German voice name never fits French
    finally:
        controller.shutdown()


# ---------------------------------------------------------------- WD7 connecting and device control


def test_wd7_connect_builds_and_starts_a_pipeline_on_the_worker(
    wiring: FakeWiring, controller: WindowController
) -> None:
    assert controller.connect_input()
    assert controller.tick().busy is not None  # at once, before the worker is done
    state = settle(controller)
    assert state.session_active
    assert state.snapshot.running
    assert state.snapshot.input.state is DeviceState.CONNECTED
    assert state.generation == 1
    assert len(wiring.built) == 1
    assert wiring.calls("start") == [(True,)]  # lenient: an absent output does not stop it
    assert wiring.threads("build") == wiring.threads("start")
    assert threading.current_thread().name not in wiring.threads("build")


def test_wd7_auto_connect_starts_the_session_after_the_scan(wiring: FakeWiring) -> None:
    controller = make(wiring, auto_connect=True)
    try:
        controller.start()
        state = settle(controller)
        assert state.session_active
        assert state.input_devices == (MIC_B, MIC)
    finally:
        controller.shutdown()


def test_wd7_auto_connect_does_nothing_without_a_chosen_microphone(wiring: FakeWiring) -> None:
    controller = make(wiring, empty(), auto_connect=True)
    try:
        controller.start()
        state = settle(controller)
        assert not state.session_active
        assert wiring.built == []
    finally:
        controller.shutdown()


def test_wd7_a_microphone_that_cannot_be_opened_leaves_a_usable_window(
    wiring: FakeWiring, controller: WindowController
) -> None:
    wiring.configure = lambda pipeline: pipeline.fail.update(
        start=DeviceNotFoundError("no input device matching 'V380' on Windows WASAPI")
    )
    controller.connect_input()
    state = settle(controller)
    assert not state.session_active
    assert state.notice is not None
    assert state.notice.level == "error"
    assert "no input device matching 'V380'" in state.notice.text
    assert wiring.closed == wiring.built  # what the wiring created is released
    assert state.snapshot.targets == ("de", "fr")
    wiring.configure = lambda pipeline: None  # the user plugs it in and tries again
    controller.connect_input()
    assert settle(controller).session_active


def test_wd7_a_failed_build_is_reported_and_the_window_stays(
    wiring: FakeWiring, controller: WindowController
) -> None:
    wiring.build_error = FileNotFoundError("whisper model not found in models/whisper-base.en")
    controller.connect_input()
    state = settle(controller)
    assert not state.session_active
    assert state.notice is not None
    assert "whisper model not found" in state.notice.text
    assert state.busy is None


def test_wd7_outputs_that_could_not_open_are_named_in_a_warning(
    wiring: FakeWiring, controller: WindowController
) -> None:
    wiring.configure = lambda pipeline: pipeline.absent_outputs.add("fr")
    controller.connect_input()
    state = settle(controller)
    assert state.session_active
    assert state.notice is not None
    assert state.notice.level == "warn"
    assert "fr" in state.notice.text
    assert state.snapshot.output("fr").state is DeviceState.DISCONNECTED
    assert state.snapshot.output("de").state is DeviceState.CONNECTED


def test_wd7_disconnect_and_connect_the_microphone_run_off_the_ui_thread(
    wiring: FakeWiring, running: WindowController
) -> None:
    assert running.disconnect_input()
    state = settle(running)
    assert state.snapshot.input.state is DeviceState.DISCONNECTED
    assert state.session_active  # the pipeline keeps running
    assert running.connect_input()
    state = settle(running)
    assert state.snapshot.input.state is DeviceState.CONNECTED
    assert len(wiring.built) == 1  # no rebuild for a plain reconnect
    assert threading.current_thread().name not in wiring.threads("disconnect_input")
    assert threading.current_thread().name not in wiring.threads("connect_input")


def test_wd7_output_connect_and_disconnect(wiring: FakeWiring, running: WindowController) -> None:
    assert running.disconnect_output("fr")
    state = settle(running)
    assert state.snapshot.output("fr").state is DeviceState.DISCONNECTED
    assert state.snapshot.output("de").state is DeviceState.CONNECTED
    assert running.connect_output("fr")
    state = settle(running)
    assert state.snapshot.output("fr").state is DeviceState.CONNECTED
    assert wiring.calls("disconnect_output") == [("fr",)]
    assert wiring.calls("connect_output") == [("fr",)]


def test_wd7_choosing_devices_while_running_goes_through_the_pipeline(
    wiring: FakeWiring, running: WindowController
) -> None:
    assert running.select_input_device(MIC_B)
    settle(running)
    assert running.select_output_device("de", HS100W)
    state = settle(running)
    assert wiring.calls("set_input_device") == [(MIC_B,)]
    assert wiring.calls("set_output") == [("de", HS100W)]
    assert state.settings.capture.device_query == MIC_B
    assert state.settings.routes[0].output.device_query == HS100W


@pytest.mark.parametrize(
    ("method", "error", "fragment"),
    [
        ("set_input_device", DeviceNotFoundError("no input device matching 'x'"), "matching 'x'"),
        ("set_output", StreamOpenError("cannot open 'Headset': busy"), "cannot open"),
        ("connect_output", PipelineStateError("the pipeline is not running"), "not running"),
        ("disconnect_input", RuntimeError("boom"), "RuntimeError: boom"),
    ],
)
def test_wd7_control_errors_become_a_notice_and_the_window_goes_on(
    wiring: FakeWiring,
    running: WindowController,
    method: str,
    error: Exception,
    fragment: str,
) -> None:
    wiring.built[0].fail[method] = error
    actions = {
        "set_input_device": lambda: running.select_input_device(MIC_B),
        "set_output": lambda: running.select_output_device("de", HS100W),
        "connect_output": lambda: running.connect_output("de"),
        "disconnect_input": running.disconnect_input,
    }
    assert actions[method]()
    state = settle(running)
    assert state.notice is not None
    assert state.notice.level == "error"
    assert fragment in state.notice.text
    assert state.busy is None
    assert state.session_active


def test_wd7_a_language_change_while_running_rebuilds_only_that_lane(
    wiring: FakeWiring,
) -> None:
    controller = make(
        wiring, parse_settings(["--input", "V380", "--route", "de=Realtek", "--targets", "de"])
    )
    try:
        controller.connect_input()
        settle(controller)
        assert controller.set_language("de", "fr")
        state = settle(controller)
        assert wiring.calls("set_output_target") == [("de", "fr")]
        assert state.snapshot.targets == ("fr",)
        assert state.generation == 1  # the same session: the transcript is kept
        assert len(wiring.built) == 1
        assert threading.current_thread().name not in wiring.threads("set_output_target")
    finally:
        controller.shutdown()


def test_wd7_a_language_that_cannot_load_leaves_the_old_one(
    wiring: FakeWiring,
) -> None:
    controller = make(
        wiring, parse_settings(["--input", "V380", "--route", "de=Realtek", "--targets", "de"])
    )
    try:
        controller.connect_input()
        settle(controller)
        wiring.built[0].fail["set_output_target"] = FileNotFoundError("translation model missing")
        controller.set_language("de", "fr")
        state = settle(controller)
        assert state.snapshot.targets == ("de",)
        assert state.notice is not None
        assert "translation model missing" in state.notice.text
    finally:
        controller.shutdown()


def test_wd7_the_same_language_is_not_a_change(
    wiring: FakeWiring, running: WindowController
) -> None:
    assert running.set_language("de", "de")
    settle(running)
    assert wiring.calls("set_output_target") == []


# ---------------------------------------------------------------- WD8 apply and restart


def test_wd8_invalid_values_come_back_at_once_with_their_field_keys(
    wiring: FakeWiring, running: WindowController
) -> None:
    result = running.apply_settings({"segmenter.silence_ms": 100, "segmenter.tail_ms": 160})
    assert not result.restarting
    assert "segmenter.silence_ms" in result.errors
    assert running.tick().busy is None  # nothing was queued
    assert len(wiring.built) == 1
    assert not wiring.built[0].stopped


def test_wd8_applying_the_same_values_does_nothing(
    wiring: FakeWiring, running: WindowController
) -> None:
    result = running.apply_settings({"segmenter.silence_ms": 450})
    assert result.errors == {}
    assert not result.restarting
    notice = running.tick().notice
    assert notice is not None
    assert notice.level == "info"
    assert "no changes" in notice.text.lower()
    assert len(wiring.built) == 1


def test_wd8_hide_text_applies_live_without_a_restart(
    wiring: FakeWiring, running: WindowController
) -> None:
    result = running.apply_settings({"hide_text": True})
    assert result.errors == {}
    assert not result.restarting
    state = running.tick()
    assert state.snapshot.hide_text
    assert len(wiring.built) == 1
    assert not wiring.built[0].stopped
    assert state.busy is None


def test_wd8_a_restart_setting_builds_a_new_pipeline_and_a_new_view_model(
    wiring: FakeWiring, running: WindowController
) -> None:
    old = wiring.built[0]
    result = running.apply_settings({"segmenter.silence_ms": 500})
    assert result.errors == {}
    assert result.restarting
    busy = running.tick().busy
    assert busy is not None
    assert "apply" in busy.lower()
    state = settle(running)

    assert old.stopped
    assert old in wiring.closed  # the wiring is told to release what it created for it
    new = wiring.built[1]
    assert new is not old
    assert new.settings.segmenter.silence_ms == 500
    assert new.settings.capture.silence_ms == 500
    assert new.running
    assert state.generation == 2  # a new session: the window clears its transcript
    assert state.snapshot.running
    assert state.settings.segmenter.silence_ms == 500
    order = [method for method, _, _ in wiring.log if method in ("stop", "build", "start")]
    assert order == ["build", "start", "stop", "build", "start"]  # stop before the new build
    assert "view-model-1" in new.text_bus.stats()  # a view-model on the new pipeline
    assert "view-model-1" in new.translation_bus.stats()
    assert old.text_bus.closed


def test_wd8_the_restart_keeps_device_and_language_choices_made_at_run_time(
    wiring: FakeWiring, running: WindowController
) -> None:
    running.select_output_device("de", HS100W)
    settle(running)
    running.select_input_device(MIC_B)
    settle(running)
    running.apply_settings({"segmenter.silence_ms": 500})
    settle(running)
    new = wiring.built[-1].settings
    assert new.capture.device_query == MIC_B
    assert new.routes[0].output.device_query == HS100W


def test_wd8_devices_the_user_disconnected_stay_disconnected_after_a_restart(
    wiring: FakeWiring, running: WindowController
) -> None:
    running.disconnect_output("fr")
    settle(running)
    running.disconnect_input()
    settle(running)
    running.apply_settings({"segmenter.silence_ms": 500})
    state = settle(running)
    new = wiring.built[-1]
    assert new.input_st is DeviceState.DISCONNECTED
    assert new.output_st["fr"] is DeviceState.DISCONNECTED
    assert new.output_st["de"] is DeviceState.CONNECTED
    assert state.snapshot.input.state is DeviceState.DISCONNECTED


def test_wd8_a_restart_that_fails_leaves_a_standby_window_with_the_new_settings(
    wiring: FakeWiring, running: WindowController
) -> None:
    wiring.configure = lambda pipeline: pipeline.fail.update(
        start=StreamOpenError("cannot open 'Microphone (V380 USB)': in use")
    )
    running.apply_settings({"segmenter.silence_ms": 500})
    state = settle(running)
    assert not state.session_active
    assert state.notice is not None
    assert state.notice.level == "error"
    assert "restart" in state.notice.text.lower()
    assert "in use" in state.notice.text
    assert state.settings.segmenter.silence_ms == 500  # the user's choice is not lost
    assert state.generation == 2
    assert wiring.built[0].stopped
    assert wiring.built[1] in wiring.closed
    wiring.configure = lambda pipeline: None
    running.connect_input()  # the user fixes the cause and connects again
    state = settle(running)
    assert state.session_active
    assert wiring.built[-1].settings.segmenter.silence_ms == 500


def test_wd8_missing_models_refuse_the_apply_before_anything_is_stopped(
    wiring: FakeWiring, running: WindowController
) -> None:
    wiring.problem_list = ("model missing: models/whisper-tiny.en",)
    result = running.apply_settings({"transcribe.model": "tiny.en"})
    assert not result.restarting
    assert GENERAL_ERROR_KEY in result.errors
    assert "whisper-tiny.en" in result.errors[GENERAL_ERROR_KEY]
    assert not wiring.built[0].stopped
    assert running.tick().busy is None
    assert running.tick().settings.transcribe.model == "base.en"


def test_wd8_in_standby_apply_only_stores_the_settings(
    wiring: FakeWiring, controller: WindowController
) -> None:
    result = controller.apply_settings({"segmenter.silence_ms": 600})
    assert result.errors == {}
    assert not result.restarting
    state = controller.tick()
    assert state.settings.segmenter.silence_ms == 600
    assert state.busy is None
    assert wiring.built == []
    assert state.notice is not None
    assert "connected" in state.notice.text.lower()
    controller.connect_input()
    settle(controller)
    assert wiring.built[0].settings.segmenter.silence_ms == 600


def test_wd8_nothing_is_accepted_while_a_call_is_in_progress(
    wiring: FakeWiring, controller: WindowController
) -> None:
    gate = threading.Event()
    wiring.build_gate = gate
    controller.connect_input()
    try:
        assert controller.tick().busy is not None
        assert not controller.connect_input()
        assert not controller.select_input_device(MIC_B)
        assert not controller.set_language("de", "fr")
        assert not controller.refresh_devices()
        rejected = controller.apply_settings({"segmenter.silence_ms": 500})
        assert not rejected.accepted
        assert "busy" in rejected.message.lower()
    finally:
        gate.set()
    state = settle(controller)
    assert state.session_active
    assert state.settings.capture.device_query == "V380"  # the rejected choice changed nothing


def test_wd8_hide_text_can_be_toggled_while_a_call_runs(
    wiring: FakeWiring, controller: WindowController
) -> None:
    gate = threading.Event()
    wiring.build_gate = gate
    controller.connect_input()
    try:
        controller.set_hide_text(True)
        assert controller.tick().snapshot.hide_text
    finally:
        gate.set()
    state = settle(controller)
    assert state.snapshot.hide_text  # the new session starts hidden too


# ---------------------------------------------------------------- WD9 worker and shutdown


def test_wd9_the_tick_never_waits_for_the_worker(
    wiring: FakeWiring, controller: WindowController
) -> None:
    gate = threading.Event()
    wiring.build_gate = gate
    controller.connect_input()
    try:
        started = time.perf_counter()
        for _ in range(50):
            state = controller.tick()
            assert state.busy is not None
        assert time.perf_counter() - started < 2.0
    finally:
        gate.set()
    settle(controller)


def test_wd9_the_busy_label_says_what_is_going_on(
    wiring: FakeWiring, controller: WindowController
) -> None:
    gate = threading.Event()
    wiring.build_gate = gate
    controller.connect_input()
    try:
        busy = controller.tick().busy
        assert busy is not None
        assert "microphone" in busy.lower() or "connect" in busy.lower()
    finally:
        gate.set()
    settle(controller)


def test_wd9_job_runner_runs_jobs_in_order_on_one_thread() -> None:
    runner = JobRunner()
    seen: list[tuple[int, str]] = []
    for index in range(5):
        runner.submit(lambda index=index: seen.append((index, threading.current_thread().name)))
    runner.close(timeout_s=5.0)
    assert [index for index, _ in seen] == [0, 1, 2, 3, 4]
    assert len({name for _, name in seen}) == 1
    assert threading.current_thread().name not in {name for _, name in seen}


def test_wd9_job_runner_drain_returns_finished_jobs_without_blocking() -> None:
    runner = JobRunner()
    release = threading.Event()
    runner.submit(lambda: release.wait(5.0))
    assert runner.drain() == []  # nothing finished yet, and drain did not wait
    release.set()
    deadline = time.monotonic() + 5.0
    done: list[Any] = []
    while not done and time.monotonic() < deadline:
        done = runner.drain()
        time.sleep(0.002)
    assert len(done) == 1
    assert done[0].error is None
    assert done[0].value is True
    runner.close(timeout_s=5.0)


def test_wd9_job_runner_keeps_the_error_of_a_job_that_raised() -> None:
    runner = JobRunner()

    def fail() -> None:
        raise ValueError("bad job")

    runner.submit(fail)
    deadline = time.monotonic() + 5.0
    done: list[Any] = []
    while not done and time.monotonic() < deadline:
        done = runner.drain()
        time.sleep(0.002)
    runner.close(timeout_s=5.0)
    assert isinstance(done[0].error, ValueError)
    assert done[0].value is None


def test_wd9_job_runner_refuses_work_after_close() -> None:
    runner = JobRunner()
    runner.close(timeout_s=5.0)
    with pytest.raises(RuntimeError, match="closed"):
        runner.submit(lambda: None)
    runner.close(timeout_s=5.0)  # idempotent


def test_wd9_shutdown_stops_the_pipeline_and_releases_what_the_wiring_built(
    wiring: FakeWiring, running: WindowController
) -> None:
    pipeline = wiring.built[0]
    running.shutdown()
    assert pipeline.stopped
    assert pipeline in wiring.closed
    running.shutdown()  # idempotent
    assert wiring.closed.count(pipeline) == 1
    assert running.tick().snapshot is not None  # a late tick does not raise


def test_wd9_shutdown_waits_for_a_call_that_is_in_progress(
    wiring: FakeWiring, controller: WindowController
) -> None:
    gate = threading.Event()
    wiring.build_gate = gate
    controller.connect_input()
    threading.Timer(0.1, gate.set).start()
    controller.shutdown()
    assert wiring.built
    assert wiring.built[0].stopped  # the session that finished during shutdown is stopped too
    assert wiring.built[0] in wiring.closed


def test_wd9_errors_are_described_with_their_message() -> None:
    assert describe_error(DeviceNotFoundError("no input device matching 'x'")) == (
        "no input device matching 'x'"
    )
    assert describe_error(ConfigError("target language 'fr' is already used")).startswith("target")
    assert describe_error(RuntimeError("boom")) == "RuntimeError: boom"
    assert describe_error(RuntimeError()) == "RuntimeError"


def test_wd9_the_unselected_placeholder_is_never_sent_to_a_pipeline(
    wiring: FakeWiring,
) -> None:
    controller = make(wiring, empty())
    try:
        controller.select_input_device(MIC)
        controller.select_output_device("de", REALTEK)
        controller.select_output_device("fr", HS100W)
        controller.connect_input()
        state = settle(controller)
        assert state.session_active
        queries = [state.settings.capture.device_query] + [
            route.output.device_query for route in state.settings.routes
        ]
        assert UNSELECTED_DEVICE not in queries
    finally:
        controller.shutdown()


# ---------------------------------------------------------------- WD13 no disk writes


def test_wd13_the_controller_flows_touch_no_disk_and_no_network(
    wiring: FakeWiring, no_disk_writes: list[Any]
) -> None:
    controller = make(wiring)
    try:
        controller.start()
        settle(controller)
        controller.connect_input()
        settle(controller)
        controller.set_hide_text(True)
        controller.select_output_device("de", HS100W)
        settle(controller)
        controller.apply_settings({"segmenter.silence_ms": 500})
        settle(controller)
        controller.disconnect_input()
        settle(controller)
    finally:
        controller.shutdown()
    assert no_disk_writes == []


# ---------------------------------------------------------------- WD9 report and notices


def test_wd9_the_report_is_text_for_the_clipboard_and_honors_hide_text(
    wiring: FakeWiring, running: WindowController
) -> None:
    from test_ui_model import SECRET, utt

    pipeline = wiring.built[0]
    pipeline.text_bus.publish(utt(1, 1000, 2000))
    running.tick()
    report = running.export_report()
    assert SECRET in report
    assert "Interpreter session report" in report
    running.set_hide_text(True)
    hidden = running.export_report()
    assert SECRET not in hidden
    assert "[hidden]" in hidden


def test_wd9_the_window_can_post_its_own_notice(
    wiring: FakeWiring, controller: WindowController
) -> None:
    first = controller.notify("error", "The clipboard is not available.")
    state = controller.tick()
    assert state.notice is not None
    assert (state.notice.level, state.notice.text) == ("error", "The clipboard is not available.")
    assert state.notice.seq == first
    assert controller.notify("error", "The clipboard is not available.") > first
