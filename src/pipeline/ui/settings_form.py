"""State behind the settings tab, without any widget: texts, parsing, hints and error placement.

The tab holds one editable value per ``Field`` of ``settings_schema``. Numbers are edited as
text (a spin box can hold anything the user types), so this module turns the texts into typed
values and says what is wrong with a text that is not a number. Whether the numbers are in
range, and how fields relate, stays with ``settings_schema.build_settings``; the errors it
raises are mapped back to the fields here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from pipeline.config import Settings
from pipeline.ui.settings_schema import (
    GENERAL_ERROR_KEY,
    GROUPS,
    ApplyMode,
    Field,
    schema_for,
    values_from_settings,
)

FormValue = str | bool

RESTART_NOTE = "Applying restarts the pipeline"
LIVE_NOTE = "Applies immediately"
_PAGE_RESTART = (
    "Changes on this page restart the pipeline: devices reconnect and the transcript is cleared."
)
_PAGE_LIVE = "Changes on this page apply immediately."
_PAGE_MIXED = (
    "Settings marked 'applies immediately' take effect at once; the others restart the "
    "pipeline (devices reconnect and the transcript is cleared)."
)


def _fmt(number: float) -> str:
    return f"{number:g}"


def field_hint(item: Field) -> str:
    """The allowed range of a number, e.g. ``450-600 ms``; empty for choices and switches."""
    if item.kind not in ("int", "float"):
        return ""
    assert item.minimum is not None
    assert item.maximum is not None
    unit = "" if item.unit == "none" else f" {item.unit}"
    span = f"{_fmt(item.minimum)}-{_fmt(item.maximum)}{unit}"
    return span if item.off_value is None else f"{item.off_value} (off) or {span}"


def restart_note(item: Field) -> str:
    """Per-field note: does a change need the pipeline to restart."""
    return LIVE_NOTE if item.apply is ApplyMode.LIVE else RESTART_NOTE


def page_note(group: str, fields: Iterable[Field]) -> str:
    """One line for the top of a page: what applying changes on it does."""
    modes = {item.apply for item in fields}
    if modes == {ApplyMode.LIVE}:
        return _PAGE_LIVE
    return _PAGE_MIXED if ApplyMode.LIVE in modes else _PAGE_RESTART


def _text_of(item: Field, value: object) -> FormValue:
    if item.kind == "bool":
        return bool(value)
    if isinstance(value, float):
        return _fmt(value)
    return str(value)


def _parse(item: Field, raw: FormValue) -> tuple[object, str | None]:
    """Turn the edited value into the type the schema expects, or explain why it cannot."""
    if item.kind == "bool":
        return bool(raw), None
    if item.kind == "choice":
        return raw, None
    text = str(raw).strip()
    if not text:
        return None, "required: enter a number"
    try:
        if item.kind == "int":
            return int(text), None
        return float(text), None
    except ValueError:
        return None, "must be a whole number" if item.kind == "int" else "must be a number"


@dataclass(frozen=True, slots=True)
class PlacedErrors:
    """Validation messages sorted by where the window shows them.

    Attributes:
        by_field: Message per field key, shown next to the field.
        general: Messages that name no field of this form, shown above the buttons.
        advanced_fields: Keys in ``by_field`` that sit in the hidden advanced section, so the
            tab can open it.
    """

    by_field: dict[str, str]
    general: tuple[str, ...]
    advanced_fields: tuple[str, ...]


class SettingsForm:
    """Editable copy of the settings values; one entry per field of the schema."""

    def __init__(self, settings: Settings) -> None:
        """Load the form from ``settings`` (its routes decide which offset fields exist)."""
        self._fields = schema_for(settings)
        self._applied: dict[str, FormValue] = {}
        self._values: dict[str, FormValue] = {}
        self.load(settings)

    @property
    def fields(self) -> tuple[Field, ...]:
        """The fields in display order."""
        return self._fields

    @staticmethod
    def groups() -> tuple[str, ...]:
        """The pages, in display order."""
        return GROUPS

    def value(self, key: str) -> FormValue:
        """The current text (or switch) of a field."""
        return self._values[key]

    def set(self, key: str, value: FormValue) -> None:
        """Record an edit.

        Raises:
            KeyError: ``key`` is not a field of this form.
        """
        if key not in self._values:
            raise KeyError(key)
        self._values[key] = value

    def load(self, settings: Settings) -> None:
        """Replace every value with that of ``settings`` (discards edits)."""
        current = values_from_settings(settings)
        self._applied = {item.key: _text_of(item, current[item.key]) for item in self._fields}
        self._values = dict(self._applied)

    def rebase(self, settings: Settings) -> None:
        """Follow ``settings`` that changed under the form without losing the user's typing.

        Typical causes are a device or language choice and a restart. Edits of fields that no
        longer exist (the offset of a lane that was replaced) are dropped; every other edit is
        kept on the new baseline.
        """
        edits = {key: self._values[key] for key in self.edited_keys()}
        self._fields = schema_for(settings)
        self.load(settings)
        for key, value in edits.items():
            if key in self._values:
                self._values[key] = value

    def same_fields_as(self, settings: Settings) -> bool:
        """Whether ``settings`` yields exactly the fields this form has (same lanes)."""
        return tuple(item.key for item in schema_for(settings)) == tuple(
            item.key for item in self._fields
        )

    def reset_to_defaults(self) -> None:
        """Set every field to its schema default (nothing is applied until the user does so)."""
        self._values = {item.key: _text_of(item, item.default) for item in self._fields}

    def edited_keys(self) -> tuple[str, ...]:
        """Fields whose value differs from the last loaded settings."""
        return tuple(key for key, value in self._values.items() if value != self._applied[key])

    def collect(self) -> tuple[dict[str, object], dict[str, str]]:
        """Typed values of every field, and a message for each text that cannot be parsed.

        A field with a message is left out of the values: nothing is guessed.
        """
        values: dict[str, object] = {}
        errors: dict[str, str] = {}
        for item in self._fields:
            parsed, problem = _parse(item, self._values[item.key])
            if problem is None:
                values[item.key] = parsed
            else:
                errors[item.key] = problem
        return values, errors

    def place_errors(self, errors: Mapping[str, str]) -> PlacedErrors:
        """Sort ``SettingsValidationError.errors`` into per-field and general messages."""
        known = {item.key: item for item in self._fields}
        by_field: dict[str, str] = {}
        general: list[str] = []
        for key, message in errors.items():
            if key in known:
                by_field[key] = message
            elif key == GENERAL_ERROR_KEY:
                general.append(message)
            else:
                general.append(f"{key}: {message}")
        advanced = tuple(key for key in by_field if known[key].advanced)
        return PlacedErrors(by_field, tuple(general), advanced)
