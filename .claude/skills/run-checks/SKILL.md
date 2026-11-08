---
name: run-checks
description: Run this repo's quality gates (ruff format, ruff check, mypy --strict, pytest with the right markers and flags, English-only check). Use after any code change and before reporting a plan step as done.
---

# Run checks

Always run Python with `-B` (no `__pycache__` writes, which would break the
zero-disk checks) and pytest with `--disable-socket` (no network).

Run from the repo root, in this order, and stop at the first failure:

```powershell
ruff format --check .
ruff check .
mypy --strict src
python scripts/check_no_cyrillic.py
python -B -m pytest -m "unit" --disable-socket -q
```

When models are present in `models/` (after `python scripts/fetch_models.py`):

```powershell
python -B -m pytest -m "unit or model" --disable-socket -q
```

Coverage for hardware-free modules (target >= 85%):

```powershell
python -B -m pytest -m unit --disable-socket --cov=pipeline --cov-report=term-missing
```

Hardware levels are never run by default:

| Marker | Run it when | Command |
| --- | --- | --- |
| `virtual` | VB-Cable A+B installed (see skill `audio-devices`) | `python -B -m pytest -m virtual --disable-socket` |
| `hw` | V380, Realtek and HS100W connected; user present | `python -B -m pytest -m hw --disable-socket -s` |
| `soak` | User asked for the 60-minute run | `python -B -m pipeline soak --minutes 60 --input CABLE-A --stress auto --no-ui` |

## Reporting

- Quote the failing output; do not summarize a failure as a pass.
- List which markers ran and which were skipped and why.
- `ruff format` may be applied (`ruff format .`) if only formatting failed;
  say that you did it.
