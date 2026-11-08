"""The Tk window on a fake pipeline: one smoke flow, with no disk or network use (WD12, WD13).

Skipped cleanly when Tk cannot open a window (no display). The root is withdrawn, so nothing
flashes on screen while the tests run.
"""

from __future__ import annotations

import contextlib
import threading
import time
import tkinter as tk
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from test_ui_controller import chosen
from test_ui_model import SECRET, SECRET_DE, tr, utt
from ui_fakes import MIC, FakeWiring

from pipeline.config import parse_settings
from pipeline.ui.app_tk import MainWindow
from pipeline.ui.controller import WindowController

pytestmark = pytest.mark.unit


@pytest.fixture
def root() -> Iterator[tk.Tk]:
    # Tk sometimes fails to read its own script files when a root is created right after
    # another was destroyed (seen on Windows, about once in thirty roots); a second try works.
    # With no display at all every try fails and the test is skipped.
    failure: tk.TclError | None = None
    for _attempt in range(3):
        try:
            window = tk.Tk()
            break
        except tk.TclError as error:
            failure = error
            time.sleep(0.3)
    else:
        pytest.skip(f"Tk cannot open a window here: {failure}")
    window.withdraw()
    yield window
    with contextlib.suppress(tk.TclError):
        window.destroy()  # the test may have closed it already


def pump(
    root: tk.Tk,
    window: MainWindow,
    until: Callable[[], bool],
    what: str,
    timeout_s: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        root.update()
        window.tick()
        if until():
            return
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        time.sleep(0.005)


def idle(window: MainWindow, controller: WindowController) -> Callable[[], bool]:
    return lambda: controller.tick().busy is None


def test_wd12_the_window_runs_a_whole_session_on_a_fake_pipeline(
    root: tk.Tk, no_disk_writes: list[Any]
) -> None:
    wiring = FakeWiring()
    controller = WindowController(chosen(), wiring)
    window = MainWindow(root, controller, poll_ms=20)
    controller.start()

    # Standby: the window is usable with nothing running and says so in words.
    assert window.input_status_text == "Not connected"
    assert set(window.stage_words().values()) == {"Idle"}
    assert window.tts_banner_text == "TTS not installed: outputs are silent"
    assert window.problem_banner_text == ""
    assert [row.language.get() for row in window.output_rows] == ["de - German", "fr - French"]
    pump(root, window, lambda: MIC in window.input_names, "the device scan")
    assert window.input_button_text == "Connect"
    assert window.input_button_enabled

    # Connect: the worker builds and starts, the status dot and the process cards follow.
    window.press_input_button()
    assert window.busy_text  # a busy label while the worker is at it
    pump(root, window, lambda: window.input_status_text == "Connected", "the connection")
    pump(root, window, idle(window, controller), "the end of the busy state")
    assert window.busy_text == ""
    assert window.input_dot.kind == "ok"
    assert window.input_button_text == "Disconnect"
    assert [row.dot.text for row in window.output_rows] == ["Connected", "Connected"]
    assert window.stage_words()["input"] == "OK"
    assert window.stage_words()["transcribe"] == "Idle"  # no speech yet

    # An utterance arrives; its translation follows: the row is filled in place.
    pipeline = wiring.built[0]
    pipeline.text_bus.publish(utt(1, 1000, 2000, decode_ms=380.0))
    pump(root, window, lambda: window.transcript.row_ids() == ("u1",), "the transcript row")
    values = window.transcript.row_values("u1")
    assert values[0] == "00:01.0"
    assert values[2] == SECRET
    assert values[3:5] == ("-", "-")  # both translations are pending: a dash
    pipeline.translation_bus.publish(tr(1, "de", latency_ms=910.0))
    pump(
        root,
        window,
        lambda: window.transcript.row_values("u1")[3] == SECRET_DE,
        "the German translation",
    )
    assert window.transcript.row_values("u1")[4] == "-"
    assert "Utterances 1" in window.transcript.summary_text

    # Hide text is live and masks the table at once.
    window.transcript.hide_button.invoke()
    pump(
        root,
        window,
        lambda: window.transcript.row_values("u1")[2] == "[hidden]",
        "the masked row",
    )
    assert window.transcript.row_values("u1")[3] == "[hidden]"
    window.transcript.hide_button.invoke()
    pump(root, window, lambda: window.transcript.row_values("u1")[2] == SECRET, "the unmasked row")

    # Copy report: clipboard only.
    window.copy_report()
    assert "Interpreter session report" in root.clipboard_get()
    pump(root, window, lambda: "clipboard" in window.status_text, "the copy notice")

    # A device button: disconnect output 2 only.
    window.output_rows[1].button.invoke()
    pump(root, window, lambda: window.output_rows[1].dot.text == "Not connected", "output 2 away")
    assert window.output_rows[0].dot.text == "Connected"
    assert window.output_rows[1].button.cget("text") == "Connect"

    # Settings: a wrong value is refused next to its field and nothing restarts.
    view = window.settings_view
    view.edit("segmenter.silence_ms", "100")
    view.apply()
    assert "450-600" in view.field_error("segmenter.silence_ms")
    assert len(wiring.built) == 1
    view.edit("segmenter.silence_ms", "abc")
    view.apply()
    assert "whole number" in view.field_error("segmenter.silence_ms")

    # A valid one restarts the pipeline: busy state, then a cleared transcript.
    view.edit("segmenter.silence_ms", "500")
    view.apply()
    assert view.field_error("segmenter.silence_ms") == ""
    assert "restart" in window.busy_text.lower() or controller.tick().busy is not None
    pump(root, window, lambda: len(wiring.built) == 2, "the new pipeline")
    pump(root, window, idle(window, controller), "the end of the restart")
    assert window.transcript.row_ids() == ()
    assert window.input_status_text == "Connected"
    assert wiring.built[0].stopped
    assert wiring.built[1].settings.segmenter.silence_ms == 500

    # Closing stops the pipeline and destroys the window.
    window.close()
    assert wiring.built[1].stopped
    with pytest.raises(tk.TclError):
        root.title()  # a destroyed root refuses every call
    assert no_disk_writes == []  # WD13: Tk, the clipboard and the controller wrote nothing


def test_wd12_advanced_settings_are_hidden_until_asked_for(root: tk.Tk) -> None:
    wiring = FakeWiring()
    controller = WindowController(chosen(), wiring)
    window = MainWindow(root, controller, poll_ms=20)
    try:
        root.update()
        view = window.settings_view
        assert view.field_visible("segmenter.silence_ms")
        assert not view.field_visible("segmenter.speech_threshold")  # advanced
        assert not view.field_visible("strict")
        # A refusal in an advanced field opens the advanced settings so the error is seen.
        view.edit("segmenter.speech_threshold", "0.2")
        view.edit("segmenter.exit_threshold", "0.9")  # above the speech probability
        view.apply()
        assert view.field_visible("segmenter.speech_threshold")
        assert "silence probability" in view.field_error("segmenter.exit_threshold")
        assert "silence probability" in view.field_error("segmenter.speech_threshold")
    finally:
        window.close()


def test_wd12_a_missing_model_is_shown_inline_and_the_window_stays_usable(root: tk.Tk) -> None:
    wiring = FakeWiring()
    wiring.problem_list = ("model missing: models/silero_vad.onnx",)
    controller = WindowController(chosen(), wiring)
    window = MainWindow(root, controller, poll_ms=20)
    try:
        root.update()
        assert "models/silero_vad.onnx" in window.problem_banner_text
        assert not window.input_button_enabled  # connecting would only fail
        assert window.input_status_text == "Not connected"
        window.settings_view.edit("segmenter.silence_ms", "500")
        window.settings_view.apply()  # saving settings still works in standby
        pump(root, window, lambda: "saved" in window.status_text, "the settings notice")
    finally:
        window.close()


def test_wd12_the_window_starts_with_no_device_selected(root: tk.Tk) -> None:
    wiring = FakeWiring()
    controller = WindowController(parse_settings([], gui=True), wiring)
    window = MainWindow(root, controller, poll_ms=20)
    try:
        root.update()
        assert window.input_status_text == "Not connected"
        assert not window.input_button_enabled
        assert [row.device.get() for row in window.output_rows] == ["", ""]
        assert all(str(row.button.cget("state")) == "disabled" for row in window.output_rows)
        controller.start()
        pump(root, window, lambda: MIC in window.input_names, "the device scan")
        window._input_box.set(MIC)
        window._input_box.event_generate("<<ComboboxSelected>>")
        pump(root, window, lambda: window.input_button_enabled, "the microphone choice")
    finally:
        window.close()


def test_wd12_a_language_chosen_in_the_box_reaches_the_pipeline(root: tk.Tk) -> None:
    wiring = FakeWiring()
    settings = parse_settings(["--input", "V380", "--route", "de=Realtek", "--targets", "de"])
    controller = WindowController(settings, wiring)
    window = MainWindow(root, controller, poll_ms=20)
    try:
        controller.start()
        pump(root, window, idle(window, controller), "the device scan")
        window.press_input_button()
        pump(root, window, lambda: window.input_status_text == "Connected", "the connection")
        pump(root, window, idle(window, controller), "the end of the busy state")
        row = window.output_rows[0]
        assert tuple(row.language.cget("values")) == ("de - German", "fr - French")
        row.language.set("fr - French")
        row.language.event_generate("<<ComboboxSelected>>")
        pump(root, window, lambda: wiring.calls("set_output_target") == [("de", "fr")], "the call")
        pump(root, window, idle(window, controller), "the end of the language change")
        assert controller.tick().snapshot.targets == ("fr",)
        assert window.output_rows[0].language.get() == "fr - French"
        assert len(wiring.built) == 1
    finally:
        window.close()


def test_wd12_control_errors_show_in_the_status_bar(root: tk.Tk) -> None:
    wiring = FakeWiring()
    controller = WindowController(chosen(), wiring)
    window = MainWindow(root, controller, poll_ms=20)
    try:
        controller.start()
        pump(root, window, idle(window, controller), "the device scan")
        window.press_input_button()
        pump(root, window, lambda: window.input_status_text == "Connected", "the connection")
        pump(root, window, idle(window, controller), "the end of the busy state")
        from pipeline.devices import DeviceNotFoundError

        wiring.built[0].fail["set_input_device"] = DeviceNotFoundError(
            "no input device matching 'Gone'"
        )
        window._input_box.set(MIC)
        window._input_box.event_generate("<<ComboboxSelected>>")
        pump(root, window, lambda: "Gone" in window.status_text, "the error message")
        assert window.status_text.startswith("Error: ")
        pump(root, window, idle(window, controller), "the end of the call")
    finally:
        window.close()


def test_wd12_the_window_never_calls_tk_from_the_worker_thread(root: tk.Tk) -> None:
    """A Tk call from another thread raises ``RuntimeError`` or hangs; the flow must not."""
    wiring = FakeWiring()
    controller = WindowController(chosen(), wiring)
    window = MainWindow(root, controller, poll_ms=20)
    try:
        controller.start()
        pump(root, window, idle(window, controller), "the device scan")
        window.press_input_button()
        pump(root, window, lambda: window.input_status_text == "Connected", "the connection")
        pump(root, window, idle(window, controller), "the end of the busy state")
        window.settings_view.edit("segmenter.silence_ms", "500")
        window.settings_view.apply()
        pump(root, window, lambda: len(wiring.built) == 2, "the restart")
        pump(root, window, idle(window, controller), "the end of the restart")
        threads = wiring.threads("build") | wiring.threads("start") | wiring.threads("stop")
        assert threading.current_thread().name not in threads
    finally:
        window.close()


def test_wd12_the_level_meter_exists_only_when_a_snapshot_has_a_level(root: tk.Tk) -> None:
    wiring = FakeWiring()
    controller = WindowController(chosen(), wiring)
    window = MainWindow(root, controller, poll_ms=20)
    try:
        root.update()
        assert window._level is None  # InputCard.level is None today: nothing is invented
        window._render_level(0.5)
        assert window._level is not None
        assert float(window._level["value"]) == 50.0
        window._render_level(None)
        assert window._level.winfo_manager() == ""
    finally:
        window.close()


def test_wd12_the_transcript_follows_the_newest_row_unless_the_user_scrolled_up(
    root: tk.Tk,
) -> None:
    wiring = FakeWiring()
    controller = WindowController(chosen(), wiring)
    window = MainWindow(root, controller, poll_ms=20)
    window.notebook.select(0)
    root.attributes("-alpha", 0.0)  # mapped (scrolling needs real geometry) but invisible
    root.deiconify()
    try:
        controller.start()
        pump(root, window, idle(window, controller), "the device scan")
        window.press_input_button()
        pump(root, window, lambda: window.input_status_text == "Connected", "the connection")
        pump(root, window, idle(window, controller), "the end of the busy state")
        pipeline = wiring.built[0]
        tree = window.transcript.tree

        def publish(first: int, last: int) -> None:
            for number in range(first, last):
                pipeline.text_bus.publish(utt(number, number * 3000, number * 3000 + 1000))

        publish(1, 41)
        pump(root, window, lambda: len(window.transcript.row_ids()) == 40, "40 rows")
        root.update()
        assert tree.yview()[1] >= 0.999  # at the end: following

        tree.yview_moveto(0.0)  # the user scrolls up to read
        root.update()
        assert tree.yview()[1] < 0.999
        publish(41, 61)
        pump(root, window, lambda: len(window.transcript.row_ids()) == 60, "60 rows")
        root.update()
        assert tree.yview()[0] == 0.0  # still where the user left it
        assert tree.yview()[1] < 0.999

        tree.yview_moveto(1.0)  # back at the end: following again
        root.update()
        publish(61, 71)
        pump(root, window, lambda: len(window.transcript.row_ids()) == 70, "70 rows")
        root.update()
        assert tree.yview()[1] >= 0.999
    finally:
        window.close()
