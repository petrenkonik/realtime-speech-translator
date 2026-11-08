"""Translation: sentence splitting, batching and config (no model files needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.translate import (
    CtranslateTranslator,
    TranslateConfig,
    TranslateConfigError,
    split_sentences,
    translator_model_dir,
)
from pipeline.vad import ModelNotFoundError

pytestmark = pytest.mark.unit


class FakeTokenizer:
    def encode(self, text: str, out_type: type[str]) -> list[str]:
        return text.split()

    def decode(self, pieces: list[str]) -> str:
        return " ".join(pieces)


class FakeResult:
    def __init__(self, hypothesis: list[str]) -> None:
        self.hypotheses = [hypothesis]


class FakeModel:
    def __init__(self) -> None:
        self.batches: list[list[list[str]]] = []
        self.options: list[tuple[int, int]] = []

    def translate_batch(
        self, source: list[list[str]], *, beam_size: int, max_decoding_length: int
    ) -> list[FakeResult]:
        self.batches.append(source)
        self.options.append((beam_size, max_decoding_length))
        return [FakeResult([piece.upper() for piece in tokens[:-1]]) for tokens in source]


def make(config: TranslateConfig | None = None) -> tuple[CtranslateTranslator, FakeModel]:
    model = FakeModel()
    return CtranslateTranslator(model, FakeTokenizer(), FakeTokenizer(), config), model


# --- sentences ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello there.", ["Hello there."]),
        ("It works. Really? Yes! Good.", ["It works.", "Really?", "Yes!", "Good."]),
        ("  padded   ", ["padded"]),
        ("", []),
        ("   ", []),
        ("Dr. smith came.", ["Dr. smith came."]),
        ("It costs 3.50 dollars.", ["It costs 3.50 dollars."]),
        ("Wait... Then go.", ["Wait...", "Then go."]),
        ('He said "stop." Then he left.', ['He said "stop."', "Then he left."]),
        ("Room 5. 10 people came.", ["Room 5.", "10 people came."]),
        ("no capital. after the dot", ["no capital. after the dot"]),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


# --- translation ----------------------------------------------------------------------------


def test_each_sentence_goes_through_the_model_in_one_batch() -> None:
    translator, model = make()
    assert translator.translate("One two. Three four.") == "ONE TWO. THREE FOUR."
    assert len(model.batches) == 1
    assert model.batches[0] == [["One", "two.", "</s>"], ["Three", "four.", "</s>"]]


def test_blank_text_gives_an_empty_string_without_calling_the_model() -> None:
    translator, model = make()
    assert translator.translate("   ") == ""
    assert model.batches == []


def test_the_decoding_options_come_from_the_config() -> None:
    translator, model = make(TranslateConfig(beam_size=4, max_decoding_length=64))
    translator.translate("Hi.")
    assert model.options == [(4, 64)]


def test_an_empty_translation_of_one_sentence_is_left_out() -> None:
    class Blank(FakeModel):
        def translate_batch(
            self, source: list[list[str]], *, beam_size: int, max_decoding_length: int
        ) -> list[FakeResult]:
            return [FakeResult([]), FakeResult(["ok"])]

    translator = CtranslateTranslator(Blank(), FakeTokenizer(), FakeTokenizer())
    assert translator.translate("First. Second.") == "ok"


# --- config and files -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beam_size": 0},
        {"beam_size": 9},
        {"threads": 0},
        {"threads": 9},
        {"max_decoding_length": 8},
        {"max_decoding_length": 5000},
    ],
)
def test_bad_translate_settings_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(TranslateConfigError):
        TranslateConfig(**kwargs)


def test_the_model_folder_is_named_after_the_pair(tmp_path: Path) -> None:
    assert translator_model_dir(tmp_path, "en", "de") == tmp_path / "opus-mt-en-de"


def test_a_missing_model_names_the_missing_files_and_the_fetch_script(tmp_path: Path) -> None:
    (tmp_path / "model.bin").write_bytes(b"x")
    with pytest.raises(ModelNotFoundError, match=r"source\.spm.*fetch_models"):
        CtranslateTranslator.load(tmp_path)
