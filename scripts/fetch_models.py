"""Download the models into ``models/`` and, with ``--with-data``, the evaluation data into
``data/``. The only code allowed to use the network.

Every file is pinned by URL and verified by sha256, so a changed upstream file is an error
instead of a silent swap.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class ModelFile:
    """One downloadable file."""

    name: str
    url: str
    sha256: str
    license: str


# Whisper in CTranslate2 format (Systran conversions, MIT). The source language is English
# only, so the English-only models are used: smaller and more accurate than multilingual ones.
# ``base.en`` is the default, ``tiny.en`` the faster option (SPEC-LOW-LATENCY.md section 12.4).
# The tokenizer and vocabulary files are identical in both. A directory with these four files
# is what ``faster_whisper.WhisperModel`` loads.
_WHISPER_SHARED = {
    "tokenizer.json": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
    "vocabulary.txt": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf",
}


def _whisper(size: str, revision: str, config_sha256: str, model_sha256: str) -> list[ModelFile]:
    """The four files of one pinned ``Systran/faster-whisper-<size>`` revision."""
    hashes = {"config.json": config_sha256, "model.bin": model_sha256, **_WHISPER_SHARED}
    return [
        ModelFile(
            name=f"whisper-{size}/{name}",
            url=f"https://huggingface.co/Systran/faster-whisper-{size}/resolve/{revision}/{name}",
            sha256=sha256,
            license="MIT",
        )
        for name, sha256 in sorted(hashes.items())
    ]


# OPUS-MT English -> German / French (Helsinki-NLP, converted to CTranslate2 float16 by
# michaelfeil; the loader quantizes to int8 on the CPU). Licenses are those of the original
# models: en-de CC-BY-4.0 (attribution, see THIRD_PARTY_NOTICES.md), en-fr Apache-2.0. A
# folder with these five files is what ``pipeline.translate`` loads.
def _opus_mt(
    target: str, revision: str, license_name: str, hashes: dict[str, str]
) -> list[ModelFile]:
    """The five files of one pinned ``michaelfeil/ct2fast-opus-mt-en-<target>`` revision."""
    return [
        ModelFile(
            name=f"opus-mt-en-{target}/{name}",
            url=(
                f"https://huggingface.co/michaelfeil/ct2fast-opus-mt-en-{target}"
                f"/resolve/{revision}/{name}"
            ),
            sha256=sha256,
            license=license_name,
        )
        for name, sha256 in sorted(hashes.items())
    ]


MODELS = [
    ModelFile(
        name="silero_vad.onnx",
        url="https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad.onnx",
        sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        license="MIT",
    ),
    ModelFile(
        name="smart-turn/smart-turn-v3.2-cpu.onnx",
        url=(
            "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/"
            "f766f81d3cfdf7737ac64aad813d91bbfd56bf93/smart-turn-v3.2-cpu.onnx"
        ),
        sha256="2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
        license="BSD-2-Clause",
    ),
    # The unquantized weights (the vendor names the file "-gpu", it runs on any CPU too).
    # This is the default turn model: on CPUs without VNNI the int8 file answers very
    # differently from fp32 and is no faster (SPEC-LOW-LATENCY.md section 4).
    ModelFile(
        name="smart-turn/smart-turn-v3.2-gpu.onnx",
        url=(
            "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/"
            "f766f81d3cfdf7737ac64aad813d91bbfd56bf93/smart-turn-v3.2-gpu.onnx"
        ),
        sha256="ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be",
        license="BSD-2-Clause",
    ),
    *_whisper(
        "base.en",
        "3d3d5dee26484f91867d81cb899cfcf72b96be6c",
        "f3bc3821e9fc76a27bae538e11ae5b677dcdd352b4600429ce7951d398569aeb",
        "2a166925539a16005f14ff328359f9b9adb9dc4fb631bb3b227526862e93e2ef",
    ),
    *_whisper(
        "tiny.en",
        "0d3d19a32d3338f10357c0889762bd8d64bbdeba",
        "14b1b421a90349bc551b881461426b561a874049cb9e4c4864f2ca384f6a7cc5",
        "1a5afae06a4db91c975c9a9d78be5cc110ee4ea022ad57d55492e4550e936b2a",
    ),
    *_opus_mt(
        "de",
        "8ccfcb72178a843301fab5a288dee687e0aa1afc",
        "CC-BY-4.0",
        {
            "config.json": "0c2f6fa2057c7264d052fb4a62ba3476eeae70487acddfa8e779a53a00cbf44c",
            "model.bin": "e19b3ee9697c56978f1fb8f001c5bfb31285547322be9897e8727e2c58308e96",
            "shared_vocabulary.txt": "743c10261f7ea38f18422062aaf2f5cb6ad34f39526cfa279148f95b6efd1632",  # noqa: E501
            "source.spm": "678f2a1177d8389f67b66299762dcc4fc567e89b07e212ba91b0c56daecf47ce",
            "target.spm": "bbd1f495eea99c8e21ae086d9146e0fa7b096c3dfdd9ba07ab8b631889df5c9b",
        },
    ),
    *_opus_mt(
        "fr",
        "55bcef18761227161433d86b81bbbcd9d813b429",
        "Apache-2.0",
        {
            "config.json": "0c2f6fa2057c7264d052fb4a62ba3476eeae70487acddfa8e779a53a00cbf44c",
            "model.bin": "636d926880a6e6e4b773403a5991b50d4dfdfabe49a230e148676a05e99f2bec",
            "shared_vocabulary.txt": "e87446c025bbe57cc4f9ef90fa6e28ab303a2e31479b7355c2a2d3a6a3a43a0a",  # noqa: E501
            "source.spm": "173e9f493a668fe396d599e28d414a201193094e6ffd7a4678e5aab0f6d3d838",
            "target.spm": "78d0e717c77053f1c4b856d8661d9cb87c64f083a35418c087b9146300e4f585",
        },
    ),
]

# Evaluation data for scripts/eval_endpointing.py only: never imported by src/, never shipped.
# LibriSpeech is CC BY 4.0 (not in the permissive list that governs models and dependencies);
# that is acceptable because it is a test input, see SPEC-LOW-LATENCY.md section 10 (M3, T10).
# The test-clean md5 (32fa31d27d2e1cad72775fee3f4849a9) is published in
# openslr.org/resources/12/md5sum.txt and was checked by hand; the sha256 pins the same file.
# OpenSLR publishes no checksum for the manifests, so their sha256 was taken from the file
# downloaded from the official OpenSLR mirror on 2026-11-08.
DATASETS = [
    ModelFile(
        name="librispeech/test-clean.tar.gz",
        url="https://www.openslr.org/resources/12/test-clean.tar.gz",
        sha256="39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
        license="CC-BY-4.0",
    ),
    ModelFile(
        name="librispeech-pc/manifests.tar.gz",
        url="https://openslr.trmal.net/resources/145/manifests.tar.gz",
        sha256="96d4eae2222b29b66437a21959252419bcd4762e5042e71e023790171054d1c0",
        license="CC-BY-4.0",
    ),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(model: ModelFile, models_dir: Path = MODELS_DIR) -> None:
    """Download ``model`` unless a verified copy exists; a wrong hash is never saved."""
    target = models_dir / model.name
    if target.is_file() and _sha256(target) == model.sha256:
        print(f"ok      {model.name} ({model.license})")
        return
    print(f"fetch   {model.name} ...")
    with urllib.request.urlopen(model.url, timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != model.sha256:
        raise SystemExit(f"{model.name}: sha256 mismatch (got {digest}); refusing to save it")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f"saved   {model.name} ({len(data)} bytes, {model.license})")


def main(argv: list[str] | None = None) -> int:
    """Fetch every model; with ``--with-data`` also the evaluation data."""
    args = sys.argv[1:] if argv is None else argv
    for model in MODELS:
        fetch(model)
    if "--with-data" in args:
        for dataset in DATASETS:
            fetch(dataset, DATA_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
