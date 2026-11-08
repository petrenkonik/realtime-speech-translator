"""Fail if a dependency has a license this project must not ship (spec G4).

Runtime dependencies (the closure of ``[project].dependencies``) must be permissive:
MIT, BSD, Apache-2.0, PSF, ISC and the equally permissive Zlib, 0BSD, CC0, MIT-0.
Development tools are not shipped, so for them only copyleft and non-commercial licenses
(GPL, LGPL, AGPL, NC) are fatal; weak copyleft such as MPL is reported as a warning.

Usage: ``python -B scripts/check_licenses.py``. Needs the dev extras (pip-licenses).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from importlib import metadata

PERMISSIVE = re.compile(
    r"\b(MIT|BSD|Apache|PSF|Python Software Foundation|ISC|Zlib|0BSD|CC0|Unlicense)",
    re.IGNORECASE,
)
FORBIDDEN = re.compile(
    r"(\bA?L?GPL\b|General Public License|Non[- ]?Commercial|\bCC[- ]BY[- ]NC)",
    re.IGNORECASE,
)
OWN_PACKAGE = "pipeline"

# Declared licenses do not tell the whole story for these two cases, so they are decided by
# name, each with its reason (also in THIRD_PARTY_NOTICES.md).
#
# tqdm is MPL-2.0 AND MIT. MPL-2.0 is copyleft per file: it binds anyone who modifies
# tqdm's own files, not code that imports it. We import it unmodified (faster-whisper and
# huggingface-hub do), so it is accepted and reported as a warning.
ACCEPTED_WEAK_COPYLEFT = {"tqdm"}
# PyAV declares BSD-3-Clause, but its wheels ship an FFmpeg build with LGPL code
# (libmp3lame, opencore-amr). faster-whisper needs it only to decode files, which this
# project never does, so it is removed from the resolution (``[tool.uv]
# override-dependencies``) and replaced by a stub (``pipeline.transcribe.ensure_av_stub``).
BANNED_PACKAGES = {"av": "its wheels bundle LGPL FFmpeg libraries"}
# sentencepiece publishes no license field in its wheel metadata (pip-licenses says UNKNOWN);
# its repository is Apache-2.0 (https://github.com/google/sentencepiece/blob/v0.2.1/LICENSE).
LICENSE_BY_NAME = {"sentencepiece": "Apache-2.0"}


def is_permissive(expression: str) -> bool:
    """True if a license string (maybe an SPDX-style AND/OR expression) is permissive.

    ``A AND B`` needs both permissive; ``A OR B`` needs one.
    """
    if FORBIDDEN.search(expression) and " OR " not in expression:
        return False
    for conjunct in re.split(r"\s+AND\s+", expression):
        alternatives = re.split(r"\s+OR\s+", conjunct)
        usable = [alt for alt in alternatives if not FORBIDDEN.search(alt)]
        if not any(PERMISSIVE.search(alt) for alt in usable):
            return False
    return True


def is_forbidden(expression: str) -> bool:
    """True if no alternative of the license is free of GPL/LGPL/AGPL/NC terms."""
    return any(
        all(FORBIDDEN.search(alt) for alt in re.split(r"\s+OR\s+", conjunct))
        for conjunct in re.split(r"\s+AND\s+", expression)
    )


def runtime_closure(root: str = OWN_PACKAGE) -> set[str]:
    """Normalized names of the packages ``root`` needs at runtime (extras excluded)."""
    seen: set[str] = set()
    pending = [root]
    while pending:
        name = _normalize(pending.pop())
        if name in seen:
            continue
        seen.add(name)
        try:
            requirements = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        for requirement in requirements:
            if "extra ==" in requirement:
                continue
            pending.append(re.split(r"[ ;<>=!~\[(]", requirement, maxsplit=1)[0])
    return seen


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    """Check all installed packages; return the process exit code."""
    output = subprocess.run(
        [sys.executable, "-B", "-m", "piplicenses", "--format=json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    runtime = runtime_closure()
    failures: list[str] = []
    warnings: list[str] = []
    for entry in json.loads(output):
        name, license_text = entry["Name"], entry["License"]
        if license_text in ("UNKNOWN", "") and _normalize(name) in LICENSE_BY_NAME:
            license_text = LICENSE_BY_NAME[_normalize(name)]
        if _normalize(name) == OWN_PACKAGE:
            continue
        if _normalize(name) in BANNED_PACKAGES:
            failures.append(f"{name} is installed but banned: {BANNED_PACKAGES[_normalize(name)]}")
        elif _normalize(name) in ACCEPTED_WEAK_COPYLEFT and not is_forbidden(license_text):
            warnings.append(f"runtime dependency {name}: {license_text} (accepted, see script)")
        elif _normalize(name) in runtime:
            if not is_permissive(license_text):
                failures.append(f"runtime dependency {name}: {license_text}")
        elif is_forbidden(license_text):
            failures.append(f"development tool {name}: {license_text}")
        elif not is_permissive(license_text):
            warnings.append(f"development tool {name}: {license_text} (not shipped)")
    for line in warnings:
        print(f"warning: {line}")
    for line in failures:
        print(f"FAIL: {line}")
    if failures:
        return 1
    print(f"OK: licenses acceptable ({len(runtime) - 1} runtime packages checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
