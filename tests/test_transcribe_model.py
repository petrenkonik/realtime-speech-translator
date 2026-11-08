"""M2 and the glossary effect on the real whisper model.

The audio is Windows SAPI speech, so the numbers are optimistic compared with a person on a
microphone: they show that the wiring works and that hotwords help, not how good whisper is
in a room. Each utterance gets the 100 ms of low noise in front and the 160 ms tail that
the Segmenter puts around real speech.
"""

from __future__ import annotations

import re
import socket
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pytest_socket import SocketBlockedError
from sapi_speech import SENTENCES, synthesize

from pipeline.bus import Bus
from pipeline.glossary import Glossary
from pipeline.transcribe import (
    WHISPER_MODELS,
    FasterWhisperEngine,
    TranscribeConfig,
    TranscribeStage,
    Utterance,
    whisper_model_dir,
)
from pipeline.vad import SAMPLE_RATE_HZ, Chunk

pytestmark = pytest.mark.model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MS = SAMPLE_RATE_HZ // 1000

# Words that a general model has never seen, spoken twice each by two voices.
TERM_SENTENCES = [
    "We deploy the service on Kubernetes every Friday.",
    "The Zorvex gateway talks to the Quantiq database.",
    "Please ask Doctor Szymanski about the Veltrix rollout.",
    "Our Kubernetes cluster runs the Zorvex gateway.",
]
TERMS = ["Kubernetes", "Zorvex", "Quantiq", "Szymanski", "Veltrix"]


@pytest.fixture(scope="module", params=WHISPER_MODELS)
def engine(request: pytest.FixtureRequest) -> FasterWhisperEngine:
    config = TranscribeConfig(model=request.param)
    model_dir = whisper_model_dir(MODELS_DIR, config)
    if not (model_dir / "model.bin").exists():
        pytest.skip("run scripts/fetch_models.py first")
    loaded = FasterWhisperEngine.load(model_dir, config)
    loaded.warm_up()
    return loaded


@pytest.fixture(scope="module")
def speech(tmp_path_factory: pytest.TempPathFactory) -> list[NDArray[np.float32]]:
    out, error = synthesize(tmp_path_factory.mktemp("sapi_whisper"))
    if not out:
        pytest.skip(f"SAPI voices unavailable: {error}")
    return out


@pytest.fixture(scope="module")
def term_speech(tmp_path_factory: pytest.TempPathFactory) -> list[NDArray[np.float32]]:
    out, error = synthesize(tmp_path_factory.mktemp("sapi_terms"), TERM_SENTENCES)
    if not out:
        pytest.skip(f"SAPI voices unavailable: {error}")
    return out


def around(speech: NDArray[np.float32], seed: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)

    def noise(ms: int) -> NDArray[np.float32]:
        return (rng.standard_normal(ms * MS) * 0.001).astype(np.float32)

    return np.concatenate([noise(100), speech, noise(160)])


_NUMBERS = {"twenty four": "24", "nine": "9"}  # whisper writes digits, the references words


def words(text: str) -> list[str]:
    text = text.lower()
    for spoken, digits in _NUMBERS.items():
        text = text.replace(spoken, digits)
    return re.sub(r"[^a-z0-9 ]", " ", text).split()


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Word-level edit distance and the number of reference words."""
    ref, hyp = words(reference), words(hypothesis)
    table = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    table[:, 0] = range(len(ref) + 1)
    table[0, :] = range(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            table[i, j] = min(
                table[i - 1, j] + 1,
                table[i, j - 1] + 1,
                table[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    return int(table[len(ref), len(hyp)]), len(ref)


def chunk_of(chunk_id: int, samples: NDArray[np.float32]) -> Chunk:
    samples.setflags(write=False)
    start = chunk_id * 20 * SAMPLE_RATE_HZ  # far apart: never merged, no context carried over
    return Chunk(
        id=chunk_id,
        samples=samples,
        start_sample=start,
        end_sample=start + len(samples),
        speech_end_sample=start + len(samples) - 160 * MS,
        finalized_sample=start + len(samples) + 300 * MS,
        reason="silence",
    )


def transcribe_all(
    engine: FasterWhisperEngine,
    audio: list[NDArray[np.float32]],
    glossary: Glossary | None = None,
) -> list[Utterance]:
    chunks: Bus[Chunk] = Bus()
    texts: Bus[Utterance] = Bus()
    out = texts.subscribe("test")
    stage = TranscribeStage(chunks.subscribe("whisper"), texts, engine, glossary=glossary)
    found: list[Utterance] = []
    for index, samples in enumerate(audio, start=1):
        chunks.publish(chunk_of(index, samples))
        stage.step(timeout_s=0)
        if (utterance := out.get(timeout_s=0)) is not None:
            found.append(utterance)
    return found


@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_m2_the_model_loads_and_runs_with_the_network_blocked(
    engine: FasterWhisperEngine,
) -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket()
    # `engine` was built while pytest-socket is active: it needed nothing from the network.
    silence = np.zeros(SAMPLE_RATE_HZ, np.float32)
    assert engine.transcribe(silence, hotwords=None, initial_prompt=None) == ""


def test_m2_word_error_rate_is_at_most_15_percent_on_the_synthesized_sentences(
    engine: FasterWhisperEngine, speech: list[NDArray[np.float32]]
) -> None:
    utterances = transcribe_all(engine, [around(s, i) for i, s in enumerate(speech)])
    assert len(utterances) == len(speech)  # one utterance per chunk, none lost
    errors = total = 0
    for index, utterance in enumerate(utterances):
        wrong, count = word_errors(SENTENCES[index % len(SENTENCES)], utterance.text)
        errors += wrong
        total += count
    assert errors / total <= 0.15, f"WER {errors / total:.1%} ({errors}/{total} words)"


def test_m2_the_decode_time_is_recorded_on_every_utterance(
    engine: FasterWhisperEngine, speech: list[NDArray[np.float32]]
) -> None:
    utterances = transcribe_all(engine, [around(s, i) for i, s in enumerate(speech[:3])])
    assert all(u.decode_ms > 0 for u in utterances)
    assert all(u.audio_ms > 1000 for u in utterances)


def correct_terms(utterances: list[Utterance]) -> int:
    joined = " ".join(u.text.lower() for u in utterances)
    expected = " ".join(s.lower() for s in TERM_SENTENCES * 2)
    return sum(min(joined.count(t.lower()), expected.count(t.lower())) for t in TERMS)


def test_t3_hotwords_make_whisper_spell_company_terms_correctly(
    engine: FasterWhisperEngine, term_speech: list[NDArray[np.float32]]
) -> None:
    audio = [around(s, i) for i, s in enumerate(term_speech)]
    glossary = Glossary.from_toml("".join(f"[[term]]\nsource = '{t}'\n" for t in TERMS))
    without = correct_terms(transcribe_all(engine, audio))
    with_terms = correct_terms(transcribe_all(engine, audio, glossary))
    # Measured on this machine: 5 of 14 term occurrences without, 11 of 14 with hotwords.
    assert with_terms >= without + 3, f"without hotwords {without}, with {with_terms} of 14"
