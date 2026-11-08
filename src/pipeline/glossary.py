"""Company glossary: terms for whisper and exact wording in the translation.

One TOML file is read once at startup::

    [[term]]
    source = "stand-up meeting"     # as the speaker says it (English)
    de = "Stand-up-Meeting"         # how it must appear in German
    fr = "reunion debout"           # ... and in French

The source terms go to whisper as ``hotwords`` so it spells them correctly. Before an
utterance is translated, each term is swapped for a placeholder so the translator cannot
rewrite it; afterwards the placeholder is replaced with the wording from the file. This is
the "domain-specific prompt injection" of the demo.

Placeholders are plain letters and digits (``XQ0Z``), which sentencepiece-based translators
usually copy through unchanged. Whether the real models do is checked with the translators
(spec T4); a placeholder that does get lost is counted in ``Restored.missing`` instead of
being hidden.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

MAX_TERM_CHARS = 80
MAX_HOTWORDS_CHARS = 400  # whisper keeps only about half of its 448-token context for a prompt

_PLACEHOLDER = "XQ{index}Z"
_PLACEHOLDER_RE = re.compile(r"XQ(\d+)Z", re.IGNORECASE)


class GlossaryError(ValueError):
    """The glossary file is missing or cannot be used; the message says what to fix."""


@dataclass(frozen=True, slots=True)
class Term:
    """One glossary entry.

    Attributes:
        source: The term as spoken in the source language.
        targets: Language code -> wording to use in that language.
    """

    source: str
    targets: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Restored:
    """A translation after the placeholders were replaced.

    Attributes:
        text: The final text.
        missing: Placeholders the translator dropped; their terms are absent from ``text``.
    """

    text: str
    missing: int


@dataclass(frozen=True, slots=True)
class Protected:
    """A source text with its terms swapped for placeholders.

    Attributes:
        text: What goes to the translator.
    """

    text: str
    _wording: tuple[str, ...]

    def restore(self, translated: str) -> Restored:
        """Put the glossary wording where the placeholders survived.

        Args:
            translated: The translator's output.

        Returns:
            The final text and the number of placeholders that were not found.
        """
        found: set[int] = set()

        def replace(match: re.Match[str]) -> str:
            index = int(match.group(1))
            if index >= len(self._wording):
                return match.group(0)
            found.add(index)
            return self._wording[index]

        text = _PLACEHOLDER_RE.sub(replace, translated)
        return Restored(text=text, missing=len(self._wording) - len(found))


class Glossary:
    """A set of terms, loaded once and read-only afterwards."""

    def __init__(self, terms: tuple[Term, ...]) -> None:
        """Build the matcher; ``from_toml`` / ``from_file`` validate the input first."""
        self._terms = terms
        ordered = sorted(terms, key=lambda term: len(term.source), reverse=True)
        self._by_source = {term.source.casefold(): term for term in terms}
        self._pattern = (
            re.compile(
                r"(?<!\w)(?:" + "|".join(_term_regex(term.source) for term in ordered) + r")(?!\w)",
                re.IGNORECASE,
            )
            if ordered
            else None
        )

    @property
    def terms(self) -> tuple[Term, ...]:
        """The entries in file order."""
        return self._terms

    @property
    def languages(self) -> frozenset[str]:
        """Every language code that has at least one translation."""
        return frozenset(language for term in self._terms for language in term.targets)

    @classmethod
    def from_file(cls, path: Path) -> Glossary:
        """Read and validate a glossary file (UTF-8 TOML)."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise GlossaryError(f"cannot read the glossary file {path}: {error}") from error
        return cls.from_toml(text)

    @classmethod
    def from_toml(cls, text: str) -> Glossary:
        """Parse and validate glossary text; an empty text is an empty glossary."""
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            raise GlossaryError(f"the glossary is not valid TOML: {error}") from error
        entries = data.get("term", [])
        if not isinstance(entries, list):
            raise GlossaryError("the glossary must use [[term]] tables, one per term")
        terms: list[Term] = []
        seen: set[str] = set()
        for position, entry in enumerate(entries, start=1):
            term = _parse_term(entry, position)
            key = term.source.casefold()
            if key in seen:
                raise GlossaryError(f"the term {term.source!r} is listed twice")
            seen.add(key)
            terms.append(term)
        glossary = cls(tuple(terms))
        hotwords = glossary.hotwords()
        if hotwords is not None and len(hotwords) > MAX_HOTWORDS_CHARS:
            raise GlossaryError(
                f"the source terms are {len(hotwords)} characters long, too long for a whisper "
                f"prompt (limit {MAX_HOTWORDS_CHARS}); remove or shorten some"
            )
        return glossary

    def hotwords(self) -> str | None:
        """The source terms for whisper's ``hotwords``, or ``None`` for an empty glossary."""
        if not self._terms:
            return None
        return ", ".join(term.source for term in self._terms)

    def protect(self, text: str, language: str) -> Protected:
        """Swap every term in ``text`` for a placeholder.

        Args:
            text: Source-language text.
            language: Target language code; picks the wording used by ``restore``. A term
                with no wording for it is restored as written in the source.

        Returns:
            The protected text and the means to restore it.
        """
        if self._pattern is None:
            return Protected(text=text, _wording=())
        wording: list[str] = []

        def replace(match: re.Match[str]) -> str:
            term = self._by_source[re.sub(r"\s+", " ", match.group(0)).casefold()]
            wording.append(term.targets.get(language, term.source))
            return _PLACEHOLDER.format(index=len(wording) - 1)

        protected = self._pattern.sub(replace, text)
        return Protected(text=protected, _wording=tuple(wording))


def _term_regex(source: str) -> str:
    # Spaces in a term match any run of whitespace; everything else is literal.
    return r"\s+".join(re.escape(part) for part in source.split())


def _parse_term(entry: object, position: int) -> Term:
    if not isinstance(entry, dict):
        raise GlossaryError(f"term {position}: expected a table")
    source = entry.get("source")
    if not isinstance(source, str):
        raise GlossaryError(f"term {position}: 'source' is required and must be text")
    source = " ".join(source.split())
    if not source:
        raise GlossaryError(f"term {position}: 'source' is empty")
    if len(source) > MAX_TERM_CHARS:
        raise GlossaryError(
            f"term {position}: the source is too long ({len(source)} characters, "
            f"limit {MAX_TERM_CHARS})"
        )
    targets: dict[str, str] = {}
    for key, value in entry.items():
        if key == "source":
            continue
        if key != key.lower():
            raise GlossaryError(f"term {source!r}: language code {key!r} must be lower-case")
        if not isinstance(value, str) or not value.strip():
            raise GlossaryError(f"term {source!r}: the {key!r} wording must be non-empty text")
        targets[key] = value.strip()
    return Term(source=source, targets=targets)
