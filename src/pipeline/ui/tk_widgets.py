"""Small Tk building blocks shared by the window: colours, a status dot, a scrolling page.

Plain ``ttk`` with the system theme. Every colour pair here is chosen for contrast (at least
4.5:1 for text on its background), and a colour is never the only signal: the widgets that use
them always show a word as well.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from pipeline.ui.model import Health
from pipeline.ui.present import StatusKind, StatusView

HEALTH_COLORS: dict[Health, tuple[str, str]] = {
    Health.OK: ("#1b5e20", "#ffffff"),  # background, text
    Health.WARN: ("#ffc107", "#000000"),
    Health.BAD: ("#b71c1c", "#ffffff"),
    Health.IDLE: ("#bdbdbd", "#000000"),
}

DOT_COLORS: dict[StatusKind, str] = {"ok": "#2e7d32", "warn": "#f9a825", "off": "#616161"}

NOTICE_COLORS = {
    "info": "#0d47a1",
    "warn": "#8a5a00",
    "error": "#b71c1c",
}

BANNER_WARN = ("#fff3cd", "#4d3b00")  # background, text
BANNER_ERROR = ("#f8d7da", "#6a1b1b")
MUTED_TEXT = "#4a4a4a"
ERROR_TEXT = "#b71c1c"


class StatusDot(ttk.Frame):
    """A round dot and a word: filled for ok and warning, hollow for not connected."""

    _SIZE = 16

    def __init__(self, master: tk.Misc) -> None:
        """Create the dot in the "not connected" look."""
        super().__init__(master)
        background = ttk.Style(master).lookup("TFrame", "background") or "#f0f0f0"
        self._canvas = tk.Canvas(
            self,
            width=self._SIZE,
            height=self._SIZE,
            highlightthickness=0,
            borderwidth=0,
            background=background,
        )
        self._dot = self._canvas.create_oval(2, 2, self._SIZE - 2, self._SIZE - 2, width=2)
        self._canvas.pack(side="left", padx=(0, 6))
        self._text = tk.StringVar(master=self)
        ttk.Label(self, textvariable=self._text, width=16).pack(side="left")
        self.kind: StatusKind = "off"
        self.set(StatusView("Not connected", "off"))

    @property
    def text(self) -> str:
        """The word next to the dot."""
        return self._text.get()

    def set(self, status: StatusView) -> None:
        """Show ``status``."""
        color = DOT_COLORS[status.kind]
        fill = "" if status.kind == "off" else color
        self._canvas.itemconfigure(self._dot, outline=color, fill=fill)
        self._text.set(status.text)
        self.kind = status.kind


class ScrollFrame(ttk.Frame):
    """A vertically scrolling area; put widgets into ``body``."""

    def __init__(self, master: tk.Misc) -> None:
        """Create the canvas, the scroll bar and the body frame."""
        super().__init__(master)
        background = ttk.Style(master).lookup("TFrame", "background") or "#f0f0f0"
        self._canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=background)
        bar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=bar.set)
        self.body = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._on_body_resized)
        self._canvas.bind("<Configure>", self._on_canvas_resized)
        self._canvas.bind("<Enter>", self._grab_wheel)
        self._canvas.bind("<Leave>", self._release_wheel)
        self.body.bind("<Enter>", self._grab_wheel)
        bar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

    def _on_body_resized(self, event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_resized(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)

    def _grab_wheel(self, event: tk.Event) -> None:
        self._canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _release_wheel(self, event: tk.Event) -> None:
        # Leaving the canvas for its own body (a child) also fires Leave; keep scrolling then.
        pointer = self.winfo_containing(event.x_root, event.y_root)
        if pointer is not None and str(pointer).startswith(str(self)):
            return
        self._canvas.unbind_all("<MouseWheel>")

    def _on_wheel(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
