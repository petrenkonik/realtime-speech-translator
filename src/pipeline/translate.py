"""Machine translation with CTranslate2 and the OPUS-MT (Marian) models.

One ``CtranslateTranslator`` per language pair. The model folders come from
``scripts/fetch_models.py`` (``models/opus-mt-en-<target>/``); nothing is downloaded here.
Marian models translate one sentence at a time best, so a text is cut into sentences, the
sentences go through the model as one batch, and the results are joined again.

Everything that needs a model sits behind two small Protocols, so the splitting and batching
are tested without any model file.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pipeline.vad import ModelNotFoundError

_MODEL_FILES = ("model.bin", "shared_vocabulary.txt", "source.spm", "target.spm")
# A sentence ends at . ! ? (or an ellipsis) followed by a space and a capital letter or digit.
# "Dr. smith" or "e.g. this" stay together; the cost of a wrong cut is a slightly worse
# translation, the cost of a missed cut is a model that merges two sentences.
_SENTENCE_BREAK = re.compile(r"([.!?…][\"')\]”]*)\s+(?=[\"'(\[“]?[A-Z0-9])")
_END_OF_SENTENCE_TOKEN = "</s>"


class TranslateConfigError(ValueError):
    """A translation setting is out of range; the message names it."""


@dataclass(frozen=True, slots=True)
class TranslateConfig:
    """Settings of one translator.

    Attributes:
        beam_size: Beam width; 1 is greedy and fastest, 2 is a good trade for short sentences.
        threads: CPU threads of the decode.
        max_decoding_length: Longest output per sentence, in tokens.
    """

    beam_size: int = 2
    threads: int = 2
    max_decoding_length: int = 256

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not 1 <= self.beam_size <= 8:
            raise TranslateConfigError(f"beam_size must be 1-8, got {self.beam_size}")
        if not 1 <= self.threads <= 8:
            raise TranslateConfigError(f"mt threads must be 1-8, got {self.threads}")
        if not 16 <= self.max_decoding_length <= 1024:
            raise TranslateConfigError(
                f"max_decoding_length must be 16-1024, got {self.max_decoding_length}"
            )


def translator_model_dir(models_dir: Path, source: str, target: str) -> Path:
    """The folder of the model for one pair, e.g. ``models/opus-mt-en-de``."""
    return models_dir / f"opus-mt-{source}-{target}"


def split_sentences(text: str) -> list[str]:
    """Cut ``text`` into sentences; blank input gives an empty list."""
    # With a capturing group ``re.split`` returns text, closer, text, closer, ... so the
    # closing quote or bracket stays with the sentence it ends.
    parts = _SENTENCE_BREAK.split(text.strip())
    sentences = [
        parts[i] + (parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)
    ]
    return [sentence.strip() for sentence in sentences if sentence.strip()]


class SentenceTokenizer(Protocol):
    """The part of ``sentencepiece.SentencePieceProcessor`` that is used."""

    def encode(self, text: str, out_type: type[str]) -> list[str]:
        """Split text into pieces."""
        ...

    def decode(self, pieces: list[str]) -> str:
        """Join pieces into text."""
        ...


class BatchTranslator(Protocol):
    """The part of ``ctranslate2.Translator`` that is used."""

    def translate_batch(
        self,
        source: list[list[str]],
        *,
        beam_size: int,
        max_decoding_length: int,
    ) -> Sequence[_Result]:
        """Translate token lists."""
        ...


class _Result(Protocol):
    @property
    def hypotheses(self) -> list[list[str]]: ...


class CtranslateTranslator:
    """Translates English text into one target language."""

    def __init__(
        self,
        model: BatchTranslator,
        source_tokenizer: SentenceTokenizer,
        target_tokenizer: SentenceTokenizer,
        config: TranslateConfig | None = None,
    ) -> None:
        """Wrap a loaded model; ``load`` builds the real one."""
        self._model = model
        self._source = source_tokenizer
        self._target = target_tokenizer
        self._config = config or TranslateConfig()

    @classmethod
    def load(cls, model_dir: Path, config: TranslateConfig | None = None) -> CtranslateTranslator:
        """Load a model folder from disk; nothing is downloaded.

        Raises:
            ModelNotFoundError: The folder or one of its files is missing.
        """
        config = config or TranslateConfig()
        missing = [name for name in _MODEL_FILES if not (model_dir / name).is_file()]
        if missing:
            raise ModelNotFoundError(
                f"translation model incomplete in {model_dir} (missing {', '.join(missing)}); "
                "run scripts/fetch_models.py"
            )
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import ctranslate2
        import sentencepiece

        model = ctranslate2.Translator(
            str(model_dir),
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=config.threads,
        )
        source = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / "source.spm"))
        target = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / "target.spm"))
        translator = cls(model, source, target, config)
        translator.translate("Hello.")  # first call pays for set-up, the first utterance must not
        return translator

    def translate(self, text: str) -> str:
        """Translate ``text``; blank text gives an empty string."""
        sentences = split_sentences(text)
        if not sentences:
            return ""
        batch = [
            [*self._source.encode(sentence, out_type=str), _END_OF_SENTENCE_TOKEN]
            for sentence in sentences
        ]
        results = self._model.translate_batch(
            batch,
            beam_size=self._config.beam_size,
            max_decoding_length=self._config.max_decoding_length,
        )
        return " ".join(
            translated
            for result in results
            if (translated := self._target.decode(result.hypotheses[0]).strip())
        )
