"""G5: model provenance. Pinned revision, pinned hash, corrupted files are never kept."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent
SMART_TURN_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
SMART_TURN_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
SMART_TURN_FP32_SHA256 = "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_models", ROOT / "scripts/fetch_models.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["fetch_models"] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


fetch_models = load_script()


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[str]:
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: float = 0) -> FakeResponse:
        calls.append(url)
        return FakeResponse(payload)

    monkeypatch.setattr(fetch_models.urllib.request, "urlopen", fake_urlopen)
    return calls


def smart_turn(precision: str = "cpu") -> Any:
    """The int8 file is named ``-cpu``, the unquantized one ``-gpu`` (the vendor's names)."""
    return next(m for m in fetch_models.MODELS if f"smart-turn-v3.2-{precision}" in m.name)


def test_g5_smart_turn_is_pinned_to_a_revision_and_a_hash() -> None:
    model = smart_turn()
    assert SMART_TURN_REVISION in model.url
    assert "/main/" not in model.url  # a branch name would let upstream swap the file
    assert model.sha256 == SMART_TURN_SHA256
    assert model.license == "BSD-2-Clause"
    assert model.name == "smart-turn/smart-turn-v3.2-cpu.onnx"


def test_g5_smart_turn_fp32_is_pinned_to_the_same_revision_and_a_hash() -> None:
    model = smart_turn("gpu")
    assert SMART_TURN_REVISION in model.url
    assert "/main/" not in model.url
    assert model.sha256 == SMART_TURN_FP32_SHA256
    assert model.license == "BSD-2-Clause"
    assert model.name == "smart-turn/smart-turn-v3.2-gpu.onnx"


def test_g5_the_default_turn_model_is_a_file_the_fetch_script_downloads() -> None:
    from pipeline.turn import DEFAULT_TURN_MODEL

    assert DEFAULT_TURN_MODEL in {m.name for m in fetch_models.MODELS}
    assert DEFAULT_TURN_MODEL.endswith("-gpu.onnx")  # unquantized: see SPEC-LOW-LATENCY section 4


WHISPER_REVISIONS = {
    "base.en": "3d3d5dee26484f91867d81cb899cfcf72b96be6c",
    "tiny.en": "0d3d19a32d3338f10357c0889762bd8d64bbdeba",
}
WHISPER_REVISION = WHISPER_REVISIONS["base.en"]
OPUS_REVISIONS = {
    "de": "8ccfcb72178a843301fab5a288dee687e0aa1afc",
    "fr": "55bcef18761227161433d86b81bbbcd9d813b429",
}


@pytest.mark.parametrize("size", sorted(WHISPER_REVISIONS))
def test_g5_each_whisper_model_is_four_files_of_one_pinned_revision(size: str) -> None:
    files = [m for m in fetch_models.MODELS if m.name.startswith(f"whisper-{size}/")]
    assert sorted(m.name.split("/")[1] for m in files) == [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.txt",
    ]
    for model in files:
        assert f"Systran/faster-whisper-{size}/resolve/{WHISPER_REVISIONS[size]}/" in model.url
        assert len(model.sha256) == 64
        assert model.license == "MIT"


def test_g5_the_whisper_weights_are_pinned() -> None:
    weights = {m.name: m.sha256 for m in fetch_models.MODELS if m.name.endswith("/model.bin")}
    assert weights["whisper-base.en/model.bin"] == (
        "2a166925539a16005f14ff328359f9b9adb9dc4fb631bb3b227526862e93e2ef"
    )
    assert weights["whisper-tiny.en/model.bin"] == (
        "1a5afae06a4db91c975c9a9d78be5cc110ee4ea022ad57d55492e4550e936b2a"
    )


def test_g5_every_whisper_model_the_code_offers_is_downloaded_by_the_script() -> None:
    from pipeline.transcribe import WHISPER_MODELS

    assert set(WHISPER_MODELS) == set(WHISPER_REVISIONS)
    names = {m.name for m in fetch_models.MODELS}
    for size in WHISPER_MODELS:
        assert f"whisper-{size}/model.bin" in names


def test_g5_every_model_has_a_pinned_hash_and_a_permissive_license() -> None:
    for model in fetch_models.MODELS:
        assert len(model.sha256) == 64
        # CC-BY-4.0 only for opus-mt-en-de: it allows commercial use and asks for attribution
        # (THIRD_PARTY_NOTICES.md); the non-commercial and share-alike variants are refused.
        assert model.license in {"MIT", "BSD-2-Clause", "Apache-2.0", "BSD-3-Clause", "CC-BY-4.0"}
        if model.license == "CC-BY-4.0":
            assert model.name.startswith("opus-mt-en-de/")


def test_g5_a_corrupted_download_is_rejected_and_not_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(monkeypatch, b"not the model")
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        fetch_models.fetch(smart_turn(), tmp_path)
    assert not list(tmp_path.rglob("*.onnx"))


def test_g5_a_good_download_is_saved_inside_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"model bytes"
    model = replace(smart_turn(), sha256=hashlib.sha256(payload).hexdigest())
    serve(monkeypatch, payload)
    fetch_models.fetch(model, tmp_path)
    assert (tmp_path / model.name).read_bytes() == payload
    assert (tmp_path / "smart-turn").is_dir()


def test_g5_a_verified_copy_is_not_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"model bytes"
    model = replace(smart_turn(), sha256=hashlib.sha256(payload).hexdigest())
    target = tmp_path / model.name
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    calls = serve(monkeypatch, b"something else")
    fetch_models.fetch(model, tmp_path)
    assert calls == []
    assert target.read_bytes() == payload


def test_g5_a_corrupted_local_copy_is_replaced_by_a_verified_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"model bytes"
    model = replace(smart_turn(), sha256=hashlib.sha256(payload).hexdigest())
    target = tmp_path / model.name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"damaged")
    serve(monkeypatch, payload)
    fetch_models.fetch(model, tmp_path)
    assert target.read_bytes() == payload


def test_g5_evaluation_data_is_pinned_to_openslr_and_kept_apart_from_the_models() -> None:
    # Test inputs only (CC BY 4.0, see THIRD_PARTY_NOTICES.md): never shipped, never mixed
    # into MODELS, so the permissive-license check above keeps covering what ships.
    assert fetch_models.DATASETS
    assert not {d.name for d in fetch_models.DATASETS} & {m.name for m in fetch_models.MODELS}
    for dataset in fetch_models.DATASETS:
        assert len(dataset.sha256) == 64
        assert dataset.url.startswith("https://")
        assert "openslr" in dataset.url
        assert dataset.license == "CC-BY-4.0"
    assert fetch_models.DATA_DIR.name == "data"


def test_g5_the_notices_file_carries_both_model_licenses() -> None:
    text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "BSD 2-Clause License" in text
    assert "Copyright (c) 2024-2025, Daily" in text
    assert SMART_TURN_REVISION in text
    assert "MIT License" in text
    assert "Silero" in text
    assert all(revision in text for revision in WHISPER_REVISIONS.values())
    assert "Copyright (c) 2022 OpenAI" in text
    assert "tqdm" in text  # the two remarks about runtime libraries
    assert "PyAV" in text
    assert "Helsinki-NLP" in text
    assert "CC BY 4.0" in text
    assert "Apache License" in text
    assert OPUS_REVISIONS["de"] in text
    assert OPUS_REVISIONS["fr"] in text


@pytest.mark.parametrize("target", sorted(OPUS_REVISIONS))
def test_g5_each_translation_model_is_five_files_of_one_pinned_revision(target: str) -> None:
    files = [m for m in fetch_models.MODELS if m.name.startswith(f"opus-mt-en-{target}/")]
    assert sorted(m.name.split("/")[1] for m in files) == [
        "config.json",
        "model.bin",
        "shared_vocabulary.txt",
        "source.spm",
        "target.spm",
    ]
    for model in files:
        assert (
            f"michaelfeil/ct2fast-opus-mt-en-{target}/resolve/{OPUS_REVISIONS[target]}/"
            in model.url
        )
        assert len(model.sha256) == 64


def test_g5_the_translation_licenses_follow_the_original_models() -> None:
    licenses = {
        m.name.split("/")[0]: m.license for m in fetch_models.MODELS if m.name.startswith("opus-")
    }
    assert licenses == {"opus-mt-en-de": "CC-BY-4.0", "opus-mt-en-fr": "Apache-2.0"}
