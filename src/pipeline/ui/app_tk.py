"""The control window (Tk) and the ``gui`` command's runner.

Threading rule: Tk lives in the main thread only. The window polls
``WindowController.tick()`` from ``root.after`` every ``poll_ms`` and draws the returned state.
Anything slow (opening a device, loading a model, a restart) runs on the controller's worker
thread; pipeline threads never see a widget. The window never opens a dialog, never writes a
file and never uses the network: the report goes to the clipboard and nothing else.

Layout, top to bottom: banners (a speech notice, missing models and voices), the *Interpreter*
page (input, outputs, process strip, transcript with its summary) and the *Settings* page in a
notebook, and a status bar with the busy state and the latest message. The process strip is a
fork: microphone and recognition in one row, then one row per language in an "In parallel" box
(every lane gets each utterance at the same time).
"""

from __future__ import annotations

import sys
import tkinter as tk
from collections.abc import Mapping
from tkinter import font as tkfont
from tkinter import ttk

from pipeline.app import DeviceState
from pipeline.config import Settings, is_unselected
from pipeline.ui import present
from pipeline.ui.controller import ApplyResult, WindowController, WindowState, Wiring
from pipeline.ui.model import Snapshot
from pipeline.ui.settings_view import SettingsView
from pipeline.ui.tk_widgets import (
    BANNER_ERROR,
    BANNER_WARN,
    HEALTH_COLORS,
    NOTICE_COLORS,
    StatusDot,
)
from pipeline.ui.transcript_view import TranscriptView
from pipeline.ui.wiring import RealWiring

_POLL_MS = 200  # the view-model is meant to be polled every 100-250 ms
_FONT_SIZE = 10
_CARD_WRAP_PX = 170
_ERROR_PREVIEW_CHARS = 90
_FORK_WIDTH_PX = 36
_FORK_COLOR = "#555555"
_STRIP_MAX_COLUMNS = 12


def _shorten(text: str) -> str:
    return text if len(text) <= _ERROR_PREVIEW_CHARS else text[: _ERROR_PREVIEW_CHARS - 3] + "..."


class _StageCard:
    """One box of the process strip."""

    def __init__(self, parent: tk.Misc) -> None:
        self.frame = tk.Frame(parent, padx=8, pady=6, highlightthickness=1)
        self.title = tk.Label(self.frame, anchor="w", justify="left", wraplength=_CARD_WRAP_PX)
        self.status = tk.Label(self.frame, anchor="w", justify="left", wraplength=_CARD_WRAP_PX)
        self.metrics = tk.Label(self.frame, anchor="w", justify="left", wraplength=_CARD_WRAP_PX)
        bold = tkfont.nametofont("TkDefaultFont").copy()
        bold.configure(weight="bold")
        self._bold = bold  # keep a reference, or Tk drops the font
        self.title.configure(font=bold)
        for label in (self.title, self.status, self.metrics):
            label.pack(fill="x", anchor="w")
        self.shown: present.StageCardView | None = None

    def update(self, card: present.StageCardView) -> None:
        if card == self.shown:
            return
        background, foreground = HEALTH_COLORS[card.health]
        self.frame.configure(bg=background, highlightbackground=foreground)
        for label in (self.title, self.status, self.metrics):
            label.configure(bg=background, fg=foreground)
        self.title.configure(text=card.title)
        self.status.configure(text=f"{card.word}: {card.detail}")
        self.metrics.configure(text=card.metrics)
        self.shown = card


class _LaneRow:
    """One language lane of the fork: label, translate card, arrow, speak / output card."""

    def __init__(self, parent: tk.Misc, row: int, lane: present.LaneCards) -> None:
        self.target = lane.target
        self.frame = ttk.Frame(parent)
        self.frame.grid(row=row, column=0, sticky="ew", pady=3)
        self.frame.columnconfigure(1, weight=1, uniform="lane")
        self.frame.columnconfigure(3, weight=1, uniform="lane")
        self.label = ttk.Label(self.frame, text=present.language_label(lane.target), width=12)
        self.label.grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.translate = _StageCard(self.frame)
        self.translate.frame.grid(row=0, column=1, sticky="ew")
        # The only arrow of a lane: translation feeds speech inside this lane, never another one.
        ttk.Label(self.frame, text="→").grid(row=0, column=2, padx=4)
        self.output = _StageCard(self.frame)
        self.output.frame.grid(row=0, column=3, sticky="ew")


class _OutputRow:
    """Language box, device box, status dot and button of one output."""

    def __init__(self, parent: tk.Misc, row: int, window: MainWindow) -> None:
        self.target = ""
        self.state = DeviceState.DISCONNECTED
        self._window = window
        self.label = ttk.Label(parent, text=f"Output {row + 1}:")
        self.label.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        self.language = ttk.Combobox(parent, state="readonly", width=14)
        self.language.grid(row=row, column=1, sticky="w", padx=(0, 8))
        self.language.bind("<<ComboboxSelected>>", self._language_chosen)
        self.device = ttk.Combobox(parent, state="readonly", width=44)
        self.device.grid(row=row, column=2, sticky="w", padx=(0, 8))
        self.device.bind("<<ComboboxSelected>>", self._device_chosen)
        self.dot = StatusDot(parent)
        self.dot.grid(row=row, column=3, sticky="w", padx=(0, 8))
        self.button = ttk.Button(parent, width=12, command=self._pressed)
        self.button.grid(row=row, column=4, sticky="w")
        self.error = tk.StringVar(master=parent, value="")
        ttk.Label(parent, textvariable=self.error, foreground=NOTICE_COLORS["error"]).grid(
            row=row, column=5, sticky="w", padx=(8, 0)
        )
        self.shown_language = ""
        self.shown_device = ""
        self.enabled: bool | None = None

    def _language_chosen(self, event: tk.Event) -> None:
        self._window.choose_language(self.target, present.language_code(self.language.get()))

    def _device_chosen(self, event: tk.Event) -> None:
        self._window.choose_output_device(self.target, self.device.get())

    def _pressed(self) -> None:
        self._window.press_output_button(self.target, self.state)


class MainWindow:
    """The whole window; ``tick()`` draws one state and is what the timer calls."""

    def __init__(
        self, root: tk.Tk, controller: WindowController, *, poll_ms: int = _POLL_MS
    ) -> None:
        """Build the widgets and start the timer.

        Args:
            root: The Tk root (created by the caller, so tests can skip without a display).
            controller: The controller whose ``tick()`` drives the window.
            poll_ms: Timer period; 100-250 ms keeps the view-model's poll cheap.
        """
        self._root = root
        self._controller = controller
        self._poll_ms = poll_ms
        self._closed = False
        self._after_id: str | None = None
        self._generation = 0
        self._state: WindowState | None = None
        self._busy_shown: str | None = None
        self._notice_seq_shown = -1
        self._card_keys: tuple[str, ...] = ()
        self._cards: dict[str, _StageCard] = {}
        self._common_cards: list[_StageCard] = []
        self._lane_rows: list[_LaneRow] = []
        self._lane_group: ttk.LabelFrame | None = None
        self._fork: tk.Canvas | None = None
        self._fork_drawn: tuple[int, tuple[int, ...]] | None = None
        self._outputs: list[_OutputRow] = []
        self._input_names: tuple[str, ...] = ()
        self._input_enabled: bool | None = None

        self._configure_style()
        root.title("Offline interpreter")
        root.geometry("1280x860")
        root.minsize(980, 640)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self._banners = ttk.Frame(root)
        self._banners.pack(fill="x", padx=8, pady=(8, 0))
        self._tts_banner = tk.Label(
            self._banners, bg=BANNER_WARN[0], fg=BANNER_WARN[1], anchor="w", padx=8, pady=4
        )
        self._problem_banner = tk.Label(
            self._banners,
            bg=BANNER_ERROR[0],
            fg=BANNER_ERROR[1],
            anchor="w",
            justify="left",
            padx=8,
            pady=4,
        )

        self._bar = ttk.Frame(root)
        self._bar.pack(side="bottom", fill="x", padx=8, pady=6)
        self._build_status_bar()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        live = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(live, text="Interpreter")
        self._build_input(live)
        self._build_outputs(live)
        self._build_process(live)
        self.transcript = TranscriptView(
            live, on_hide_text=controller.set_hide_text, on_copy=self.copy_report
        )
        self.transcript.pack(fill="both", expand=True, pady=(8, 0))
        settings_page = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(settings_page, text="Settings")
        self.settings_view = SettingsView(
            settings_page, controller.settings, on_apply=self._apply_settings
        )
        self.settings_view.pack(fill="both", expand=True)

        self.tick()
        self._after_id = root.after(self._poll_ms, self._on_timer)

    # --- style --------------------------------------------------------------------------

    def _configure_style(self) -> None:
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkFixedFont"):
            tkfont.nametofont(name).configure(size=_FONT_SIZE)
        style = ttk.Style(self._root)
        style.configure("Treeview", rowheight=int(_FONT_SIZE * 2.6))
        heading = tkfont.nametofont("TkDefaultFont").copy()
        heading.configure(weight="bold")
        self._heading_font = heading
        style.configure("Treeview.Heading", font=heading)
        # A stray wheel turn over a device box must not switch a device while presenting.
        self._root.bind_class("TCombobox", "<MouseWheel>", lambda event: "break")

    # --- building -----------------------------------------------------------------------

    def _build_input(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Input", padding=8)
        frame.pack(fill="x")
        ttk.Label(frame, text="Microphone:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._input_box = ttk.Combobox(frame, state="readonly", width=52)
        self._input_box.grid(row=0, column=1, sticky="w", padx=(0, 8))
        self._input_box.bind("<<ComboboxSelected>>", self._input_chosen)
        self.input_dot = StatusDot(frame)
        self.input_dot.grid(row=0, column=2, sticky="w", padx=(0, 8))
        self._input_button = ttk.Button(frame, width=12, command=self.press_input_button)
        self._input_button.grid(row=0, column=3, sticky="w", padx=(0, 8))
        self._refresh_button = ttk.Button(
            frame, text="Refresh devices", command=self._controller.refresh_devices
        )
        self._refresh_button.grid(row=0, column=4, sticky="w", padx=(0, 8))
        self._level: ttk.Progressbar | None = None
        self._level_frame = frame
        self._input_error = tk.StringVar(master=frame, value="")
        ttk.Label(frame, textvariable=self._input_error, foreground=NOTICE_COLORS["error"]).grid(
            row=1, column=1, columnspan=4, sticky="w"
        )

    def _build_outputs(self, parent: ttk.Frame) -> None:
        self._outputs_frame = ttk.LabelFrame(parent, text="Outputs", padding=8)
        self._outputs_frame.pack(fill="x", pady=(8, 0))
        self._outputs_hint = ttk.Label(self._outputs_frame)  # replaced with the first rows

    def _build_process(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Process", padding=8)
        frame.pack(fill="x", pady=(8, 0))
        self._strip = ttk.Frame(frame)
        self._strip.pack(fill="x")

    def _build_status_bar(self) -> None:
        self._progress = ttk.Progressbar(self._bar, mode="indeterminate", length=120)
        self._busy_label = ttk.Label(self._bar, text="")
        self._notice_label = ttk.Label(self._bar, text="", anchor="w")
        self._notice_label.pack(side="left", fill="x", expand=True)
        self._busy_label.pack(side="right", padx=(8, 0))
        self._progress.pack(side="right")
        self._progress.pack_forget()

    # --- inspection (used by tests) -----------------------------------------------------

    @property
    def input_status_text(self) -> str:
        """The word next to the microphone's dot."""
        return self.input_dot.text

    @property
    def input_button_text(self) -> str:
        """Label of the microphone's Connect / Disconnect button."""
        return str(self._input_button.cget("text"))

    @property
    def input_button_enabled(self) -> bool:
        """Whether the microphone's button can be pressed."""
        return str(self._input_button.cget("state")) != "disabled"

    @property
    def input_names(self) -> tuple[str, ...]:
        """The devices offered in the microphone box."""
        return tuple(str(name) for name in self._input_box.cget("values"))

    @property
    def output_rows(self) -> tuple[_OutputRow, ...]:
        """The output rows, in lane order."""
        return tuple(self._outputs)

    @property
    def status_text(self) -> str:
        """The latest message in the status bar."""
        return str(self._notice_label.cget("text"))

    @property
    def busy_text(self) -> str:
        """What the status bar says is in progress."""
        return str(self._busy_label.cget("text"))

    @property
    def tts_banner_text(self) -> str:
        """The speech banner (a wiring's notice), empty while hidden."""
        return str(self._tts_banner.cget("text")) if self._tts_banner.winfo_manager() else ""

    @property
    def problem_banner_text(self) -> str:
        """The missing-models banner, empty while hidden."""
        return (
            str(self._problem_banner.cget("text")) if self._problem_banner.winfo_manager() else ""
        )

    @property
    def common_cards(self) -> tuple[_StageCard, ...]:
        """The cards of the left part (microphone, recognition), in signal order."""
        return tuple(self._common_cards)

    @property
    def lane_group(self) -> ttk.LabelFrame | None:
        """The "In parallel" box holding the lane rows; ``None`` with no lanes."""
        return self._lane_group

    @property
    def lane_rows(self) -> tuple[_LaneRow, ...]:
        """The lane rows, top to bottom, in the snapshot's order."""
        return tuple(self._lane_rows)

    @property
    def fork(self) -> tk.Canvas | None:
        """The connector from recognition to the lane rows; ``None`` with no lanes."""
        return self._fork

    def stage_words(self) -> dict[str, str]:
        """The health word shown on each process card."""
        return {key: card.shown.word for key, card in self._cards.items() if card.shown}

    # --- timer --------------------------------------------------------------------------

    def _on_timer(self) -> None:
        if self._closed:
            return
        # Scheduled first: a bug in drawing must not stop the updates.
        self._after_id = self._root.after(self._poll_ms, self._on_timer)
        self.tick()

    def tick(self) -> None:
        """Take the controller's state and draw it."""
        state = self._controller.tick()
        self._state = state
        self._render(state)

    # --- drawing ------------------------------------------------------------------------

    def _render(self, state: WindowState) -> None:
        if state.generation != self._generation:
            self._generation = state.generation
            self.transcript.clear()
        busy = state.busy is not None
        self._render_banners(state)
        self._render_input(state, busy)
        self._render_outputs(state, busy)
        self._render_process(state.snapshot)
        self.transcript.render(state.snapshot)
        self.settings_view.sync(state.settings)
        self.settings_view.set_busy(busy)
        self._render_status(state)

    def _render_banners(self, state: WindowState) -> None:
        if state.tts_notice:
            self._tts_banner.configure(text=state.tts_notice)
            self._tts_banner.pack(fill="x", pady=(0, 4))
        else:
            self._tts_banner.pack_forget()
        if state.problems:
            lines = ["The pipeline cannot start: models or voices are missing."]
            lines.extend(f"  {problem}" for problem in state.problems)
            self._problem_banner.configure(text="\n".join(lines))
            self._problem_banner.pack(fill="x", pady=(0, 4))
        else:
            self._problem_banner.pack_forget()

    def _render_input(self, state: WindowState, busy: bool) -> None:
        card = state.snapshot.input
        if state.input_devices != self._input_names:
            self._input_names = state.input_devices
            self._input_box.configure(values=state.input_devices)
        if not busy:
            shown = present.display_device(state.settings.capture.device_query, state.input_devices)
            if self._input_box.get() != shown:
                self._input_box.set(shown)
        enabled = not busy
        if enabled != self._input_enabled:
            self._input_enabled = enabled
            self._input_box.configure(state="readonly" if enabled else "disabled")
            self._refresh_button.configure(state="normal" if enabled else "disabled")
        self.input_dot.set(present.device_status(card.state))
        can_connect = not is_unselected(state.settings.capture.device_query) and not (
            state.problems and not state.session_active
        )
        button = present.connect_button(card.state, busy=busy, can_connect=can_connect)
        self._input_button.configure(
            text=button.text, state="normal" if button.enabled else "disabled"
        )
        self._input_error.set(f"Last error: {_shorten(card.last_error)}" if card.last_error else "")
        self._render_level(card.level)

    def _render_level(self, level: float | None) -> None:
        percent = present.level_percent(level)
        if percent is None:
            if self._level is not None:
                self._level.grid_remove()
            return
        if self._level is None:
            self._level = ttk.Progressbar(self._level_frame, maximum=100, length=120)
            self._level.grid(row=0, column=5, sticky="w")
        self._level.grid()
        self._level["value"] = percent

    def _render_outputs(self, state: WindowState, busy: bool) -> None:
        targets = state.snapshot.targets
        if len(self._outputs) != len(targets):
            for child in self._outputs_frame.winfo_children():
                child.destroy()
            self._outputs = [_OutputRow(self._outputs_frame, i, self) for i in range(len(targets))]
            self._outputs_hint = ttk.Label(
                self._outputs_frame,
                text="Outputs connect together with the microphone: connect it first.",
                foreground=NOTICE_COLORS["warn"],
            )
        queries = {route.target: route.output.device_query for route in state.settings.routes}
        for row, target in zip(self._outputs, targets, strict=True):
            card = state.snapshot.output(target)
            row.target = target
            row.state = card.state
            options = tuple(
                present.language_label(code) for code in present.language_options(target, targets)
            )
            if tuple(row.language.cget("values")) != options:
                row.language.configure(values=options)
            if tuple(row.device.cget("values")) != state.output_devices:
                row.device.configure(values=state.output_devices)
            if not busy:
                language = present.language_label(target)
                if language != row.shown_language:
                    row.language.set(language)
                    row.shown_language = language
                device = present.display_device(queries.get(target, ""), state.output_devices)
                if row.device.get() != device:
                    row.device.set(device)
            enabled = not busy
            if enabled != row.enabled:
                row.enabled = enabled
                combo_state = "readonly" if enabled else "disabled"
                row.language.configure(state=combo_state)
                row.device.configure(state=combo_state)
            row.dot.set(present.device_status(card.state))
            can_connect = state.session_active and not is_unselected(queries.get(target, ""))
            button = present.connect_button(card.state, busy=busy, can_connect=can_connect)
            row.button.configure(text=button.text, state="normal" if button.enabled else "disabled")
            row.error.set(_shorten(card.last_error) if card.last_error else "")
        if state.session_active:
            self._outputs_hint.grid_forget()
        else:
            self._outputs_hint.grid(
                row=len(targets), column=0, columnspan=6, sticky="w", pady=(6, 0)
            )

    def _render_process(self, snapshot: Snapshot) -> None:
        layout = present.process_layout(snapshot)
        keys = layout.keys()
        if keys != self._card_keys:
            self._build_strip(layout)
            self._card_keys = keys
        for card, widget in zip(layout.common, self._common_cards, strict=True):
            widget.update(card)
        for lane, row in zip(layout.lanes, self._lane_rows, strict=True):
            if lane.translate is not None:
                row.translate.update(lane.translate)
            if lane.output is not None:
                row.output.update(lane.output)

    def _build_strip(self, layout: present.ProcessLayout) -> None:
        """Common cards in one row, then the fork into a group with one row per language."""
        for child in self._strip.winfo_children():
            child.destroy()
        for column in range(_STRIP_MAX_COLUMNS):
            self._strip.columnconfigure(column, weight=0, uniform="")
        self._cards = {}
        self._common_cards = []
        self._lane_rows = []
        self._lane_group = None
        self._fork = None
        self._fork_drawn = None
        column = 0
        for index, card in enumerate(layout.common):
            if index:
                ttk.Label(self._strip, text="→").grid(row=0, column=column)
                column += 1
            widget = _StageCard(self._strip)
            widget.frame.grid(row=0, column=column, sticky="ew")
            self._strip.columnconfigure(column, weight=1, uniform="stage")
            self._common_cards.append(widget)
            self._cards[card.key] = widget
            column += 1
        if not layout.lanes:
            return
        background = str(ttk.Style(self._root).lookup("TFrame", "background") or "SystemButtonFace")
        self._fork = tk.Canvas(
            self._strip, width=_FORK_WIDTH_PX, height=1, highlightthickness=0, bg=background
        )
        self._fork.grid(row=0, column=column, sticky="ns", padx=2)
        self._fork.bind("<Configure>", self._draw_fork)
        group = ttk.LabelFrame(self._strip, text="In parallel", padding=(8, 2, 8, 6))
        group.grid(row=0, column=column + 1, sticky="nsew")
        group.columnconfigure(0, weight=1)
        group.bind("<Configure>", self._draw_fork)
        self._strip.columnconfigure(column + 1, weight=3)
        self._lane_group = group
        for index, lane in enumerate(layout.lanes):
            row = _LaneRow(group, index, lane)
            self._lane_rows.append(row)
            if lane.translate is not None:
                self._cards[lane.translate.key] = row.translate
            if lane.output is not None:
                self._cards[lane.output.key] = row.output
        self._draw_fork()

    def _draw_fork(self, event: tk.Event | None = None) -> None:
        """Draw the stem, bar and one branch per lane; redrawn only when the rows moved."""
        canvas = self._fork
        if canvas is None or not self._lane_rows:
            return
        count = len(self._lane_rows)
        if canvas.winfo_ismapped():
            top = canvas.winfo_rooty()
            ys = tuple(
                row.frame.winfo_rooty() - top + row.frame.winfo_height() // 2
                for row in self._lane_rows
            )
        else:  # not laid out (hidden window): the rows share the group's height evenly
            height = max(1, self._lane_group_height())
            ys = tuple(int((index + 0.5) * height / count) for index in range(count))
        if self._fork_drawn == (_FORK_WIDTH_PX, ys):
            return
        self._fork_drawn = (_FORK_WIDTH_PX, ys)
        canvas.delete("all")
        geometry = present.fork_geometry(width_px=_FORK_WIDTH_PX, lane_ys=ys)
        if geometry.stem is not None:
            canvas.create_line(*geometry.stem, tags="stem", fill=_FORK_COLOR, width=2)
        if geometry.bar is not None:
            canvas.create_line(*geometry.bar, tags="bar", fill=_FORK_COLOR, width=2)
        for branch in geometry.branches:
            canvas.create_line(*branch, tags="branch", fill=_FORK_COLOR, width=2, arrow="last")

    def _lane_group_height(self) -> int:
        group = self._lane_group
        return 0 if group is None else group.winfo_reqheight()

    def _render_status(self, state: WindowState) -> None:
        if state.busy != self._busy_shown:
            self._busy_shown = state.busy
            if state.busy is None:
                self._progress.stop()
                self._progress.pack_forget()
                self._busy_label.configure(text="")
            else:
                self._progress.pack(side="right")
                self._progress.start(12)
                self._busy_label.configure(text=state.busy)
        notice = state.notice
        if notice is None:
            if self._notice_seq_shown != -1:
                self._notice_label.configure(text="")
                self._notice_seq_shown = -1
            return
        if notice.seq != self._notice_seq_shown:
            self._notice_seq_shown = notice.seq
            prefix = {"error": "Error: ", "warn": "Warning: ", "info": ""}[notice.level]
            self._notice_label.configure(
                text=prefix + notice.text, foreground=NOTICE_COLORS[notice.level]
            )

    # --- actions ------------------------------------------------------------------------

    def _input_chosen(self, event: tk.Event) -> None:
        self._controller.select_input_device(self._input_box.get())
        self.tick()  # show the busy state (and any refusal) at once, not at the next timer tick

    def press_input_button(self) -> None:
        """Connect or disconnect the microphone, depending on its state."""
        state = self._state
        if state is not None and state.snapshot.input.state is not DeviceState.DISCONNECTED:
            self._controller.disconnect_input()
        else:
            self._controller.connect_input()
        self.tick()

    def choose_language(self, target: str, new_target: str) -> None:
        """The language box of an output changed."""
        self._controller.set_language(target, new_target)
        self.tick()

    def choose_output_device(self, target: str, name: str) -> None:
        """The device box of an output changed."""
        self._controller.select_output_device(target, name)
        self.tick()

    def press_output_button(self, target: str, state: DeviceState) -> None:
        """Connect or disconnect one output, depending on its state."""
        if state is DeviceState.DISCONNECTED:
            self._controller.connect_output(target)
        else:
            self._controller.disconnect_output(target)
        self.tick()

    def _apply_settings(self, values: Mapping[str, object]) -> ApplyResult:
        result = self._controller.apply_settings(values)
        self.tick()
        return result

    def copy_report(self) -> None:
        """Put the plain-text report on the clipboard (nothing is saved anywhere)."""
        text = self._controller.export_report()
        try:
            self._root.clipboard_clear()
            self._root.clipboard_append(text)
            self._root.update()  # hand the data to Windows now, not at the next idle time
        except tk.TclError as error:
            self._controller.notify("error", f"Cannot copy to the clipboard: {error}")
            return
        lines = max(0, text.count("\n") - 1)
        self._controller.notify("info", f"Report copied to the clipboard ({lines} lines).")

    def close(self) -> None:
        """Hide the window, stop the pipeline and end the program."""
        if self._closed:
            return
        self._closed = True
        if self._after_id is not None:
            self._root.after_cancel(self._after_id)
        self._root.withdraw()  # gone at once; stopping the pipeline can take a few seconds
        self._controller.shutdown()
        self._root.destroy()


def _enable_dpi_awareness() -> None:
    """Ask Windows not to scale (blur) the window on high-DPI screens."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass  # an older Windows: the window is merely scaled


def run_gui(settings: Settings, *, wiring: Wiring | None = None) -> int:
    """Open the window and run until it is closed.

    Args:
        settings: Where the window starts (devices may be unselected).
        wiring: Builds the pipeline; the real models and devices when omitted.

    Returns:
        0 after a normal close, 1 if no window could be opened.
    """
    controller = WindowController(
        settings,
        wiring if wiring is not None else RealWiring(),
        auto_connect=not is_unselected(settings.capture.device_query),
    )
    _enable_dpi_awareness()
    try:
        root = tk.Tk()
    except tk.TclError as error:
        controller.shutdown()
        sys.stderr.write(f"pipeline gui: cannot open a window: {error}\n")
        return 1
    MainWindow(root, controller)  # the Tk callbacks keep it alive
    controller.start()
    root.mainloop()
    return 0
