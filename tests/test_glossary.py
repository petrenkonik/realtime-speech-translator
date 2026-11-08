"""Glossary loading, whisper hotwords and term protection in translation (spec T3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.glossary import Glossary, GlossaryError

pytestmark = pytest.mark.unit

TERMS = """
[[term]]
source = "Kubernetes"
de = "Kubernetes"
fr = "Kubernetes"

[[term]]
source = "stand-up meeting"
de = "Stand-up-Meeting"
fr = "réunion debout"

[[term]]
source = "Acme Cloud"
de = "Acme-Cloud"
"""


@pytest.fixture
def glossary() -> Glossary:
    return Glossary.from_toml(TERMS)


def fake_translate(text: str) -> str:
    """A stand-in for machine translation: changes the words, keeps the placeholders."""
    return " ".join(word if word.upper().startswith("XQ") else f"<{word}>" for word in text.split())


# --- loading ---------------------------------------------------------------------------


def test_t3_terms_are_loaded_with_their_translations(glossary: Glossary) -> None:
    assert [term.source for term in glossary.terms] == [
        "Kubernetes",
        "stand-up meeting",
        "Acme Cloud",
    ]
    assert glossary.languages == frozenset({"de", "fr"})


def test_t3_an_empty_glossary_is_valid_and_does_nothing() -> None:
    glossary = Glossary.from_toml("")
    assert glossary.terms == ()
    assert glossary.hotwords() is None
    protected = glossary.protect("nothing to protect", "de")
    assert protected.text == "nothing to protect"
    assert protected.restore("nichts").text == "nichts"


def test_t3_the_file_is_read_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "terms.toml"
    path.write_text(TERMS, encoding="utf-8")
    assert len(Glossary.from_file(path).terms) == 3


def test_t3_a_missing_file_says_which_one(tmp_path: Path) -> None:
    with pytest.raises(GlossaryError, match=r"terms\.toml"):
        Glossary.from_file(tmp_path / "terms.toml")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[[term]]\nde = 'x'\n", "source"),
        ("[[term]]\nsource = ''\n", "empty"),
        ("[[term]]\nsource = 'a'\n[[term]]\nsource = 'A'\n", "twice"),
        ("[[term]]\nsource = 'a'\nde = 5\n", "text"),
        ("[[term]]\nsource = 'a'\nDE = 'x'\n", "lower-case"),
        ("term = 'x'\n", r"\[\[term\]\]"),
        ("this is = not toml [", "TOML"),
        ("[[term]]\nsource = '" + "x" * 500 + "'\n", "too long"),
    ],
)
def test_t3_a_broken_glossary_fails_with_a_message_naming_the_problem(
    text: str, message: str
) -> None:
    with pytest.raises(GlossaryError, match=message):
        Glossary.from_toml(text)


# --- whisper side ----------------------------------------------------------------------


def test_t3_hotwords_are_the_source_terms(glossary: Glossary) -> None:
    assert glossary.hotwords() == "Kubernetes, stand-up meeting, Acme Cloud"


# --- translation side ------------------------------------------------------------------


def test_t3_a_term_comes_out_exactly_as_the_glossary_says(glossary: Glossary) -> None:
    protected = glossary.protect("Please start the stand-up meeting now.", "de")
    assert "stand-up" not in protected.text  # the translator never sees the term
    restored = protected.restore(fake_translate(protected.text))
    assert "Stand-up-Meeting" in restored.text
    assert restored.missing == 0


def test_t3_each_language_gets_its_own_wording(glossary: Glossary) -> None:
    text = "Open the stand-up meeting."
    for language, wanted in (("de", "Stand-up-Meeting"), ("fr", "réunion debout")):
        protected = glossary.protect(text, language)
        assert wanted in protected.restore(fake_translate(protected.text)).text


def test_t3_matching_ignores_case_and_whole_words_only(glossary: Glossary) -> None:
    protected = glossary.protect("KUBERNETES and kubernetesish", "de")
    restored = protected.restore(fake_translate(protected.text))
    assert restored.text.count("Kubernetes") == 1  # the first word, written as in the glossary
    assert "<kubernetesish>" in restored.text  # a longer word is not a match


def test_t3_the_longest_term_wins(glossary: Glossary) -> None:
    glossary = Glossary.from_toml(
        "[[term]]\nsource='cloud'\nde='Wolke'\n[[term]]\nsource='Acme Cloud'\nde='Acme-Cloud'\n"
    )
    protected = glossary.protect("Use Acme Cloud today", "de")
    assert "Acme-Cloud" in protected.restore(protected.text).text
    assert "Wolke" not in protected.restore(protected.text).text


def test_t3_a_term_without_a_translation_for_the_language_is_kept_as_written() -> None:
    glossary = Glossary.from_toml("[[term]]\nsource='Acme Cloud'\nde='Acme-Cloud'\n")
    protected = glossary.protect("Use Acme Cloud", "fr")
    assert protected.restore(fake_translate(protected.text)).text.endswith("Acme Cloud")


def test_t3_a_repeated_term_is_restored_every_time(glossary: Glossary) -> None:
    protected = glossary.protect("Kubernetes, then Kubernetes again", "de")
    assert protected.restore(fake_translate(protected.text)).text.count("Kubernetes") == 2


def test_t3_a_placeholder_the_translator_dropped_is_counted_not_hidden(
    glossary: Glossary,
) -> None:
    protected = glossary.protect("Start Kubernetes", "de")
    restored = protected.restore("Starte")  # the translator lost the placeholder
    assert restored.text == "Starte"
    assert restored.missing == 1


def test_t3_a_placeholder_survives_changed_capitalization(glossary: Glossary) -> None:
    protected = glossary.protect("Start Kubernetes", "de")
    mangled = protected.text.lower()
    assert "Kubernetes" in protected.restore(mangled).text


def test_t3_text_without_terms_is_returned_unchanged(glossary: Glossary) -> None:
    protected = glossary.protect("Nothing special here.", "de")
    assert protected.text == "Nothing special here."
    assert protected.restore("Nichts Besonderes.").text == "Nichts Besonderes."
