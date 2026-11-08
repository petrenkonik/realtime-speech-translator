"""The settings tab: one page per group, one widget per field, Apply and Reset.

Everything about the fields (labels, ranges, units, help, restart notes, error placement) comes
from ``settings_schema`` through ``SettingsForm``; this module only places widgets. Advanced
fields stay hidden until the user ticks "Show advanced settings" (or an error sits in one).
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Mapping
from tkinter import ttk

from pipeline.config import Settings
from pipeline.ui.controller import ApplyResult
from pipeline.ui.settings_form import (
    PlacedErrors,
    SettingsForm,
    field_hint,
    page_note,
    restart_note,
)
from pipeline.ui.settings_schema import ApplyMode, Field
from pipeline.ui.tk_widgets import ERROR_TEXT, MUTED_TEXT, ScrollFrame

_HIDDEN_ADVANCED = (
    "The advanced settings of this page are hidden. "
    "Tick 'Show advanced settings' above to edit them."
)


def _decimals(step: float) -> int:
    text = f"{step:g}"
    return len(text.split(".")[1]) if "." in text else 0


class _FieldRow:
    """The widgets of one field: label, input, range, restart note, help and error line."""

    def __init__(
        self, parent: ttk.Frame, item: Field, form: SettingsForm, on_edit: Callable[[], None]
    ) -> None:
        self.item = item
        self.frame = ttk.Frame(parent, padding=(0, 6, 0, 2))
        self.frame.columnconfigure(1, weight=1)
        ttk.Label(self.frame, text=item.label).grid(row=0, column=0, sticky="w", padx=(0, 12))
        holder = ttk.Frame(self.frame)
        holder.grid(row=0, column=1, sticky="w")
        self._var: tk.Variable
        initial = form.value(item.key)

        def changed(*_args: object) -> None:
            form.set(item.key, self._var.get())  # type: ignore[no-untyped-call]
            on_edit()

        if item.kind == "bool":
            self._var = tk.BooleanVar(master=parent, value=bool(initial))
            ttk.Checkbutton(holder, variable=self._var, text="On").pack(side="left")
        elif item.kind == "choice":
            self._var = tk.StringVar(master=parent, value=str(initial))
            ttk.Combobox(
                holder,
                textvariable=self._var,
                values=item.choices,
                state="readonly",
                width=14,
            ).pack(side="left")
        else:
            assert item.minimum is not None
            assert item.maximum is not None
            assert item.step is not None
            self._var = tk.StringVar(master=parent, value=str(initial))
            digits = 0 if item.kind == "int" else _decimals(item.step)
            ttk.Spinbox(
                holder,
                textvariable=self._var,
                from_=item.minimum,
                to=item.maximum,
                increment=item.step,
                format=f"%.{digits}f",
                width=10,
            ).pack(side="left")
            hint = field_hint(item)
            ttk.Label(holder, text=hint, foreground=MUTED_TEXT).pack(side="left", padx=(8, 0))
        self._var.trace_add("write", changed)
        note = ttk.Label(
            self.frame,
            text=restart_note(item),
            foreground=MUTED_TEXT if item.apply is ApplyMode.RESTART else "#1b5e20",
        )
        note.grid(row=0, column=2, sticky="e", padx=(12, 0))
        ttk.Label(
            self.frame, text=item.help, foreground=MUTED_TEXT, wraplength=620, justify="left"
        ).grid(row=1, column=0, columnspan=3, sticky="w")
        self._error = tk.StringVar(master=parent, value="")
        self._error_label = ttk.Label(
            self.frame, textvariable=self._error, foreground=ERROR_TEXT, wraplength=620
        )
        self._error_label.grid(row=2, column=0, columnspan=3, sticky="w")
        self._error_label.grid_remove()

    def set_value(self, value: str | bool) -> None:
        """Show ``value`` without treating it as an edit of the form."""
        self._var.set(value)

    def set_error(self, message: str) -> None:
        """Show a message under the field, or hide the line for an empty one."""
        self._error.set(f"Error: {message}" if message else "")
        if message:
            self._error_label.grid()
        else:
            self._error_label.grid_remove()

    @property
    def error(self) -> str:
        """The message currently shown."""
        return self._error.get()


class SettingsView(ttk.Frame):
    """The notebook of settings pages with the Apply, Discard and Reset buttons."""

    def __init__(
        self,
        master: tk.Misc,
        settings: Settings,
        *,
        on_apply: Callable[[Mapping[str, object]], ApplyResult],
    ) -> None:
        """Build the tab from ``settings``.

        Args:
            master: Parent widget.
            settings: The settings the fields start from.
            on_apply: Sends typed values to the controller and returns its answer.
        """
        super().__init__(master)
        self._on_apply = on_apply
        self._settings = settings
        self._form = SettingsForm(settings)
        self._rows: dict[str, _FieldRow] = {}
        self._pages: dict[str, ttk.Frame] = {}
        self._empty_notes: dict[str, ttk.Label] = {}
        self._built: set[str] = set()  # pages whose widgets exist (built when first shown)
        self._errors: dict[str, str] = {}
        self._busy = False

        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, 6))
        self._advanced = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(
            top,
            text="Show advanced settings",
            variable=self._advanced,
            command=self._apply_advanced_visibility,
        ).pack(side="left")
        self.apply_button = ttk.Button(top, text="Apply", command=self.apply)
        self.apply_button.pack(side="right")
        self.defaults_button = ttk.Button(top, text="Reset to defaults", command=self.reset)
        self.defaults_button.pack(side="right", padx=6)
        self.discard_button = ttk.Button(top, text="Discard edits", command=self.discard)
        self.discard_button.pack(side="right")
        self._general = tk.StringVar(master=self, value="")
        ttk.Label(
            self, textvariable=self._general, foreground=ERROR_TEXT, wraplength=900, justify="left"
        ).pack(fill="x")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._tab_changed)
        self._build_pages()

    # --- inspection (used by tests) -----------------------------------------------------

    @property
    def form(self) -> SettingsForm:
        """The state behind the widgets."""
        return self._form

    def edit(self, key: str, value: str | bool) -> None:
        """Type ``value`` into the widget of a field, as the user would."""
        self._row(key).set_value(value)

    def field_error(self, key: str) -> str:
        """The message shown under a field."""
        return self._row(key).error

    def field_visible(self, key: str) -> bool:
        """Whether the row of a field is on screen (advanced rows may be hidden)."""
        return bool(self._row(key).frame.winfo_manager())

    def _row(self, key: str) -> _FieldRow:
        group = next(item.group for item in self._form.fields if item.key == key)
        self._ensure_page(group)
        return self._rows[key]

    @property
    def general_error(self) -> str:
        """The message above the pages."""
        return self._general.get()

    # --- building -----------------------------------------------------------------------

    def _build_pages(self) -> None:
        """Create the pages (empty); a page gets its widgets when it is first shown.

        Building every field of every page up front takes over a second, and most of them are
        never looked at.
        """
        tabs = self._tabs()
        selected = self.notebook.index("current") if tabs else 0  # type: ignore[no-untyped-call]
        for tab in tabs:
            page = self.nametowidget(tab)
            self.notebook.forget(tab)
            page.destroy()
        self._rows.clear()
        self._pages.clear()
        self._empty_notes.clear()
        self._built.clear()
        for group in self._form.groups():
            holder = ScrollFrame(self.notebook)
            self.notebook.add(holder, text=group)
            self._pages[group] = holder.body
        tabs = self._tabs()
        if tabs:
            self.notebook.select(min(selected, len(tabs) - 1))  # type: ignore[no-untyped-call]
        self._ensure_page(self._current_group())

    def _tabs(self) -> tuple[str, ...]:
        return tuple(str(tab) for tab in self.notebook.tabs())  # type: ignore[no-untyped-call]

    def _current_group(self) -> str:
        index = self.notebook.index("current")  # type: ignore[no-untyped-call]
        return self._form.groups()[int(index)]

    def _tab_changed(self, event: tk.Event) -> None:
        if self._pages:
            self._ensure_page(self._current_group())

    def _ensure_page(self, group: str) -> None:
        if group in self._built or group not in self._pages:
            return
        self._built.add(group)
        fields = [item for item in self._form.fields if item.group == group]
        body = self._pages[group]
        body.columnconfigure(0, weight=1)
        ttk.Label(body, text=page_note(group, fields), foreground=MUTED_TEXT, wraplength=760).pack(
            anchor="w", pady=(6, 4)
        )
        self._empty_notes[group] = ttk.Label(
            body, text=_HIDDEN_ADVANCED, foreground=MUTED_TEXT, wraplength=760
        )
        for item in fields:
            row = _FieldRow(body, item, self._form, self._edited)
            row.set_error(self._errors.get(item.key, ""))
            self._rows[item.key] = row
        self._apply_advanced_visibility()

    def _apply_advanced_visibility(self) -> None:
        show = bool(self._advanced.get())
        for group in self._built:
            visible = 0
            for item in self._form.fields:
                if item.group != group:
                    continue
                row = self._rows[item.key]
                if item.advanced and not show:
                    row.frame.pack_forget()
                else:
                    row.frame.pack(fill="x", anchor="w")
                    visible += 1
            note = self._empty_notes[group]
            if visible == 0:
                note.pack(anchor="w", pady=8)
            else:
                note.pack_forget()

    def _edited(self) -> None:
        """A widget changed the form; nothing else to do until Apply."""

    # --- actions ------------------------------------------------------------------------

    def apply(self) -> None:
        """Parse the widgets, hand the values to the controller and show what it answers."""
        if self._busy:
            return
        values, parse_errors = self._form.collect()
        if parse_errors:
            self._show_errors(PlacedErrors(parse_errors, (), self._advanced_keys(parse_errors)))
            return
        result = self._on_apply(values)
        if not result.accepted:
            self._show_errors(PlacedErrors({}, (result.message,), ()))
            return
        self._show_errors(self._form.place_errors(result.errors))

    def discard(self) -> None:
        """Drop the edits and show the applied values again."""
        self._form.load(self._settings)
        self._push_form_to_widgets()
        self._show_errors(PlacedErrors({}, (), ()))

    def reset(self) -> None:
        """Fill every field with its default; nothing is applied until Apply is pressed."""
        self._form.reset_to_defaults()
        self._push_form_to_widgets()
        self._show_errors(PlacedErrors({}, (), ()))

    def set_busy(self, busy: bool) -> None:
        """Disable the buttons while a restart or a device call is in progress."""
        if busy == self._busy:
            return
        self._busy = busy
        state = "disabled" if busy else "normal"
        for button in (self.apply_button, self.defaults_button, self.discard_button):
            button.configure(state=state)

    def sync(self, settings: Settings) -> None:
        """Follow settings that changed under the tab, keeping what the user is typing."""
        if settings == self._settings:
            return
        self._settings = settings
        if self._form.same_fields_as(settings):
            self._form.rebase(settings)
            self._push_form_to_widgets()
            return
        self._form.rebase(settings)
        self._build_pages()

    # --- helpers ------------------------------------------------------------------------

    def _advanced_keys(self, errors: Mapping[str, str]) -> tuple[str, ...]:
        return tuple(item.key for item in self._form.fields if item.advanced and item.key in errors)

    def _push_form_to_widgets(self) -> None:
        for key, row in self._rows.items():
            row.set_value(self._form.value(key))

    def _show_errors(self, placed: PlacedErrors) -> None:
        self._errors = dict(placed.by_field)
        for key, row in self._rows.items():
            row.set_error(placed.by_field.get(key, ""))
        self._general.set("\n".join(placed.general))
        if placed.advanced_fields and not self._advanced.get():
            self._advanced.set(True)
            self._apply_advanced_visibility()
