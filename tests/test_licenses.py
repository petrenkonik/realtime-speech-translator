"""License policy (spec G4): the classifier used by scripts/check_licenses.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SPEC = importlib.util.spec_from_file_location(
    "check_licenses", Path(__file__).parent.parent / "scripts" / "check_licenses.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
check_licenses = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_licenses)


@pytest.mark.parametrize(
    "text",
    [
        "MIT",
        "MIT License",
        "MIT-0",
        "BSD License",
        "BSD-3-Clause",
        "3-Clause BSD License",
        "Apache-2.0",
        "Apache Software License",
        "PSF-2.0",
        "Python Software Foundation License",
        "ISC License (ISCL)",
        "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",  # numpy
        "Apache-2.0 OR BSD-2-Clause",  # packaging
        "GPL-3.0-only OR MIT",  # an alternative we may choose
    ],
)
def test_permissive_licenses_pass(text: str) -> None:
    assert check_licenses.is_permissive(text)


@pytest.mark.parametrize(
    "text",
    [
        "GPL-3.0-only",
        "GNU General Public License v3 (GPLv3)",
        "LGPL-2.1-or-later",
        "GNU Lesser General Public License v3 (LGPLv3)",
        "AGPL-3.0",
        "CC-BY-NC-4.0",
        "Creative Commons Non-Commercial",
        "MIT AND GPL-2.0-only",  # one copyleft part spoils the whole expression
        "Mozilla Public License 2.0 (MPL 2.0)",  # not on the runtime allowlist
        "Proprietary",
        "UNKNOWN",
    ],
)
def test_other_licenses_fail_for_runtime_dependencies(text: str) -> None:
    assert not check_licenses.is_permissive(text)


@pytest.mark.parametrize(
    ("text", "forbidden"),
    [
        ("MIT", False),
        ("Mozilla Public License 2.0 (MPL 2.0)", False),
        ("GPL-3.0-only", True),
        ("GNU Lesser General Public License v3 (LGPLv3)", True),
        ("CC-BY-NC-4.0", True),
        ("GPL-3.0-only OR MIT", False),
        ("MIT AND GPL-2.0-only", True),
    ],
)
def test_forbidden_licenses_are_fatal_even_for_development_tools(
    text: str, forbidden: bool
) -> None:
    assert check_licenses.is_forbidden(text) is forbidden


def test_runtime_closure_contains_our_dependencies_and_not_dev_tools() -> None:
    closure = check_licenses.runtime_closure()
    assert {"numpy", "scipy", "onnxruntime", "sounddevice", "rich"} <= closure
    assert "pytest" not in closure
    assert "mypy" not in closure


def test_whisper_is_a_runtime_dependency_and_pyav_is_not_installed() -> None:
    assert {"faster-whisper", "ctranslate2"} <= check_licenses.runtime_closure()
    assert importlib.util.find_spec("av") is None  # its wheels bundle LGPL FFmpeg libraries


def test_pyav_is_banned_by_name_whatever_license_it_declares() -> None:
    assert "av" in check_licenses.BANNED_PACKAGES


def test_sentencepiece_is_a_runtime_dependency_with_a_license_decided_by_name() -> None:
    assert "sentencepiece" in check_licenses.runtime_closure()
    assert check_licenses.LICENSE_BY_NAME["sentencepiece"] == "Apache-2.0"


def test_the_installed_environment_passes_the_policy(capsys: pytest.CaptureFixture[str]) -> None:
    assert check_licenses.main() == 0
    assert "OK" in capsys.readouterr().out
