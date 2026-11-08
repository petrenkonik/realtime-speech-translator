"""Fail if any repository text file contains Cyrillic letters.

Enforces the English-only rule from CLAUDE.md. Uses `git ls-files` when the
repository has a git index, otherwise walks the tree. Standard library only.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Built from code points so this file itself stays free of Cyrillic letters.
CYRILLIC = re.compile(f"[{chr(0x0400)}-{chr(0x052F)}]")
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "models",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}
BINARY_SUFFIXES = {".onnx", ".bin", ".flac", ".wav", ".png", ".jpg", ".pml", ".zip"}


def candidate_files() -> list[Path]:
    """Return tracked files, or all files outside skipped dirs if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return [ROOT / line for line in result.stdout.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        return [
            path
            for path in ROOT.rglob("*")
            if path.is_file() and not SKIP_DIRS.intersection(path.relative_to(ROOT).parts)
        ]


def main() -> int:
    """Print every offending line and return a non-zero exit code if any exist."""
    violations = 0
    for path in candidate_files():
        if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if CYRILLIC.search(line):
                violations += 1
                print(f"{path.relative_to(ROOT)}:{number}: Cyrillic text is not allowed")
    if violations:
        print(f"{violations} line(s) violate the English-only rule.")
        return 1
    print("OK: no Cyrillic text found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
