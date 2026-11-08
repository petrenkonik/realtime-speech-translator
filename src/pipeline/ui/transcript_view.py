"""The transcript table: one row per utterance, translations as they arrive, auto-scroll.

Only draws what ``present`` produced. A row keeps its ``utterance_id`` as the Tk item id, so a
translation that arrives later updates its row in place; rows the view-model dropped (its row
cap) are removed. The table follows the newest row unless the user has scrolled up.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from pipeline.ui import present
from pipeline.ui.model import Snapshot, TranscriptRow
from pipeline.ui.tk_widgets import MUTED_TEXT

_FOLLOW_EDGE = 0.999  # the bottom of the scroll range counts as "at the end" from here


def _column_targets(snapshot: Snapshot) -> tuple[str, ...]:
    """Current targets first, then languages that older rows still carry (after a change)."""
    older = (cell.target for row in snapshot.rows for cell in row.translations)
    return tuple(dict.fromkeys([*snapshot.targets, *older]))


class TranscriptView(ttk.Frame):
    """Header (hide text, copy), the table, a detail line for the selected row, the summary."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_hide_text: Callable[[bool], None],
        on_copy: Callable[[], None],
    ) -> None:
        """Build the widgets.

        Args:
            master: Parent widget.
            on_hide_text: Called with the new state when the user toggles "Hide text".
            on_copy: Called when the user presses "Copy report".
        """
        super().__init__(master)
        self._on_hide_text = on_hide_text
        self._targets: tuple[str, ...] | None = None
        self._cells: dict[str, tuple[str, ...]] = {}
        self._rows: dict[str, TranscriptRow] = {}
        self._order: list[str] = []

        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 4))
        ttk.Label(header, text="Transcript").pack(side="left")
        ttk.Button(header, text="Copy report", command=on_copy).pack(side="right")
        self._hide = tk.BooleanVar(master=self, value=False)
        self.hide_button = ttk.Checkbutton(
            header, text="Hide text", variable=self._hide, command=self._hide_toggled
        )
        self.hide_button.pack(side="right", padx=8)

        table = ttk.Frame(self)
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table, show="headings", selectmode="browse", height=8)
        bar = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda event: self._show_detail())

        self._detail = tk.StringVar(master=self, value="")
        detail = ttk.Label(self, textvariable=self._detail, wraplength=900, foreground=MUTED_TEXT)
        detail.pack(fill="x", pady=(4, 0))
        detail.bind("<Configure>", lambda event: detail.configure(wraplength=max(200, event.width)))
        self._summary = tk.StringVar(master=self, value="")
        ttk.Label(self, textvariable=self._summary).pack(fill="x", pady=(4, 0))

    # --- inspection (used by tests) -----------------------------------------------------

    def row_ids(self) -> tuple[str, ...]:
        """Item ids of the rows in the table, oldest first."""
        return tuple(self.tree.get_children())

    def row_values(self, item_id: str) -> tuple[str, ...]:
        """The cell texts of one row."""
        return tuple(str(value) for value in self.tree.item(item_id, "values"))

    @property
    def summary_text(self) -> str:
        """The summary line."""
        return self._summary.get()

    # --- drawing ------------------------------------------------------------------------

    def clear(self) -> None:
        """Remove every row (a new pipeline took over)."""
        self.tree.delete(*self.tree.get_children())
        self._cells.clear()
        self._rows.clear()
        self._order.clear()
        self._detail.set("")

    def render(self, snapshot: Snapshot) -> None:
        """Show the rows and the summary of ``snapshot``."""
        if self._hide.get() != snapshot.hide_text:
            self._hide.set(snapshot.hide_text)
        targets = _column_targets(snapshot)
        if targets != self._targets:
            self._set_columns(targets)
        follow = self._at_end()
        wanted = {f"u{row.utterance_id}": row for row in snapshot.rows}
        for item_id in [item for item in self._order if item not in wanted]:
            self.tree.delete(item_id)
            del self._cells[item_id]
            del self._rows[item_id]
            self._order.remove(item_id)
        newest: str | None = None
        for item_id, row in wanted.items():
            cells = present.transcript_cells(row, targets)
            if item_id not in self._cells:
                self.tree.insert("", "end", iid=item_id, values=cells)
                self._order.append(item_id)
            elif self._cells[item_id] != cells:
                self.tree.item(item_id, values=cells)
            self._cells[item_id] = cells
            self._rows[item_id] = row
            newest = item_id
        if follow and newest is not None:
            # Rows inserted in this very call are not laid out yet; without this, scrolling to
            # the newest one can stop a row short when many arrive at once.
            self.tree.update_idletasks()
            self.tree.see(newest)
        self._summary.set(present.summary_line(snapshot))
        self._show_detail()

    def _set_columns(self, targets: tuple[str, ...]) -> None:
        columns = present.transcript_columns(targets)
        self.clear()
        self.tree.configure(columns=[column.key for column in columns])
        for column in columns:
            self.tree.heading(column.key, text=column.heading)
            self.tree.column(
                column.key, width=column.width_px, minwidth=60, stretch=column.stretch, anchor="w"
            )
        self._targets = targets

    def _at_end(self) -> bool:
        _first, last = self.tree.yview()
        return last >= _FOLLOW_EDGE

    def _hide_toggled(self) -> None:
        self._on_hide_text(bool(self._hide.get()))

    def _show_detail(self) -> None:
        """Show the full text of the selected row (a table cell cuts long sentences)."""
        selected = self.tree.selection()
        if not selected or selected[0] not in self._cells:
            self._detail.set("")
            return
        columns = present.transcript_columns(self._targets or ())
        cells = self._cells[selected[0]]
        parts = [
            f"{column.heading}: {cell}"
            for column, cell in zip(columns, cells, strict=True)
            if column.key == "text" or column.key.startswith("t:")
        ]
        self._detail.set("   ".join(parts))
