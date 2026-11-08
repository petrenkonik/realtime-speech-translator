# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Offline real-time interpreter demo

Spec (Russian): https://claude.ai/artifact/92tDd7zcmYfYBxFfjgYNH5
English source of truth for implementation: `docs/SPEC.md` (requirements,
test catalog with IDs, work plan in section 11). If the two disagree, ask.

One microphone, one source language → faster-whisper (once per utterance) →
parallel per-language lanes (CTranslate2 → Windows SAPI TTS → own output device).
Windows only, CPU only, fully offline.

> Status: in progress. Steps 1-7, 2b, 9, 10 and 11 of the work plan are done (7: V3 still to run in an elevated shell;
> 2b: Smart Turn is implemented but measured slow here, `turn_mode` defaults to `fixed`
> in `SegmenterConfig`) (`pcm`, `resample`,
> `vad`, `bus`, `devices`, `capture`, `supervisor`, `output`, `turn_features`, `turn`, `metrics`, `dashboard`, `guards`, `transcribe`, `glossary`, `config`, `lanes`, `translate`, `tts_sapi`, `app`),
> plus the operator window: `ui/model`, `ui/settings_schema`, `ui/settings_form`, `ui/present`,
> `ui/controller`, `ui/wiring` (no Tk) and `ui/app_tk`, `ui/transcript_view`, `ui/settings_view`,
> `ui/tk_widgets` (Tk), started with `python -B -m pipeline gui`. `__main__.py` has only `gui` and
> `list-devices`; `run` and `soak` are still to be written. The operator window
> (`ui/wiring.py`) speaks with `tts_sapi.sapi_tts_factory` and `RateConverter`; a missing SAPI voice is
> reported by `RealWiring.problems` like a missing model (audible playback through the window is not yet
> verified by hand). The rest of `src/` is created step by step per `docs/SPEC.md` section 11.

## Commands

Always run Python with `-B` and pytest with `--disable-socket` (a
`__pycache__` write or a socket would break the zero-disk / zero-network checks).

```powershell
pip install -e ".[dev]"                                   # after a venv is created
python scripts/fetch_models.py                            # only step allowed to use the network
ruff format --check . ; ruff check . ; mypy --strict src
python scripts/check_no_cyrillic.py                       # English-only rule
python -B scripts/check_licenses.py                       # permissive licenses only
python -B -m pytest -m unit --disable-socket -q           # fast, no hardware, no models
python -B -m pytest -m "unit or model" --disable-socket   # needs models/ from fetch_models.py
python -B -m pytest -m unit --disable-socket --cov=pipeline --cov-report=term-missing
python -B -m pytest tests/test_resample.py::test_u4 --disable-socket   # single test (name is illustrative)
python -B -m pipeline list-devices
python -B -m pipeline gui                                  # control window; devices can be chosen in it
python -B -m pipeline gui --input V380 --route de=Realtek,fr=HS100W --strict
python -B -m pipeline run --input V380 --source en --targets de,fr --route de=Realtek,fr=HS100W --silence-ms 450 --strict
```

Pytest markers: `unit` (CI), `model` (CI, needs `models/`), `virtual` (needs
VB-Cable A+B), `hw` (V380 + Realtek + HS100W, user present), `soak` (60 min).
Never run `virtual`, `hw` or `soak` unless asked. Use the `run-checks` skill
for the full gate sequence.

## Architecture (big picture)

```text
capture -> resample -> VAD/Segmenter -> ChunkBus --> transcribe --> TextBus --> LanguageLane(de) -> output A
                                               \--> meter                \--> LanguageLane(fr) -> output B
                                               \--> slow consumer (demo only)
metrics / dashboard / guards observe everything
```

- Audio callback only copies into a preallocated ring buffer; whisper, MT, TTS
  and playback run in their own threads behind non-blocking buses, so heavy
  work can never cause an input overflow. Threads, not processes (NumPy,
  onnxruntime, CTranslate2 release the GIL).
- `ChunkBus` / `TextBus` are the same implementation: one bounded queue per
  subscriber, `publish()` never blocks, drop-oldest policy, per-subscriber stats.
  Chunks are immutable (`samples` has `writeable=False`) and shared by reference.
- Time is the stream sample counter, not wall clock, so tests are independent
  of machine speed. Hardware is injected through stream factories; tests use
  `FakeInputStream` / `FakeOutputStream` / `FakeDeviceList` (`tests/fakes.py`).
- Whisper runs once per utterance and is the only ChunkBus consumer that feeds
  translation. Each `LanguageLane` owns its translator, TTS, resampler,
  output stream and buffer; lanes share nothing and one failing never affects
  another. A lane more than 2 utterances behind drops the oldest.
- Devices are resolved by name substring + WASAPI host API, never by index.
  Input and outputs auto-reconnect (watchdog: no callback for 500 ms).
- Utterance text lives only in RAM, never in logs (`--hide-text` hides it
  on the dashboard).
- The window (Tk, standard library; Qt is LGPL and a local web server would break the zero-network
  proof) runs in the main thread only. `WindowController.tick()` is polled from `root.after`; every
  slow control call (connect, switch, apply/restart) runs on one worker thread and reports back
  through a queue. Applying a setting that needs a restart builds a new `Pipeline` (single use) and
  a new `ViewModel`; the report goes to the clipboard only. Details: `docs/SPEC.md` section 7.
- Milestone 1 ends at ChunkBus (spec steps 1-8); everything after previews
  Milestone 2 (steps 9-12).
- Planned module layout is in `docs/SPEC.md` section 10 (`src/pipeline/…`).

## Project agents and skills

- Agents (`.claude/agents/`): `pipeline-implementer` (builds one plan step
  test-first), `realtime-reviewer` (read-only review before committing).
- Skills (`.claude/skills/`): `run-checks`, `audio-devices`, `privacy-audit`.

## Language rule (mandatory)

Everything in this repository is written in **English**: code, identifiers,
comments, docstrings, log and error messages, CLI help, dashboard text, README,
docs, test names, and commit messages. No Russian prose in the codebase:
ruff `RUF001`–`RUF003` flag look-alike Unicode characters, and a CI check
fails on any Cyrillic letter in tracked files. Conversation with the user may
be in Russian; the repository is not.

## Hard constraints

- Zero network activity at runtime. Models load from local paths only
  (also set `HF_HUB_OFFLINE=1`); `scripts/fetch_models.py` is the only code
  allowed to touch the network.
- Zero disk writes at runtime: audio and text live only in RAM. No temp files,
  no WAV encoding, no file logging. Run Python with `-B`.
- Permissive licenses only (MIT, Apache-2.0, BSD, PSF, ISC). No GPL/LGPL,
  no non-commercial model licenses (e.g. NLLB-200). Piper is not a dependency.

## Code standards

- Python 3.11+, `src/` layout, `pyproject.toml` with pinned versions.
- `ruff format`, `ruff check`, `mypy --strict` must pass.
- Type hints everywhere; Google-style docstrings on public API; units in names
  (`silence_ms`, `rate_hz`).
- Comments explain *why*, not *what*.
- Frozen slotted dataclasses for `Chunk`, `Utterance`, config; validate in
  `__post_init__`.
- Dependencies injected via constructors (stream factories, clocks);
  interfaces via `typing.Protocol`; no global state.
- No allocation, logging or blocking calls inside audio callbacks.
- No bare `except:`; custom exceptions with actionable messages.
- Tests first (TDD); test IDs (U1, F3, T7, …) match the spec.
  Coverage ≥ 85% for hardware-free modules.
