# Offline Real-Time Interpreter (demo)

A strictly offline, privacy-first speech interpretation pipeline for Windows.
One speaker talks into one microphone in one source language; listeners hear
the translation in their own language on their own output device. Nothing is
written to disk and nothing leaves the machine.

```text
microphone -> capture (WASAPI) -> resample 16 kHz mono -> Silero VAD -> ChunkBus
  -> faster-whisper (once per utterance) -> TextBus
       -> lane EN: CTranslate2 -> SAPI TTS -> output A (e.g. Realtek Speakers)
       -> lane DE: CTranslate2 -> SAPI TTS -> output B (e.g. HS100W USB headset)
```

> Status: scaffolding. See [docs/SPEC.md](docs/SPEC.md) for the full
> specification, test catalog and work plan.

## Guarantees

| Guarantee | How it is enforced |
| --- | --- |
| No network at runtime | `pytest --disable-socket`; `--strict` audit hook blocks `socket.connect`; OS check with `Get-NetTCPConnection` in Airplane Mode |
| No disk writes at runtime | Audit-hook fixture in tests; `--strict` mode in the app; Process Monitor capture during the soak run |
| Permissive licenses only | `pip-licenses --allow-only` in CI (MIT, Apache-2.0, BSD, PSF, ISC) |
| Zero input overflows | 60-minute soak run under CPU load, report in this README |
| Low latency | Per-stage and per-lane latency on the dashboard; target <= 1.5 s from end of speech to start of translation |

## Requirements

- Windows 10/11 x64, Python 3.11 or 3.12, CPU only.
- A USB microphone (any WASAPI capture device).
- One output device per target language.
- Windows SAPI voices installed for each target language
  (Settings -> Time & Language -> Speech).
- For tests only: [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (A+B).

## Quick start (planned CLI)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python scripts/fetch_models.py            # the only step that uses the network
python -B -m pipeline list-devices
python -B -m pipeline gui                 # control window: devices, languages, settings, live report
python -B -m pipeline run --input V380 --source en --targets de,fr `
    --route de=Realtek,fr=HS100W --silence-ms 450 --strict
```

The window (`gui`) starts even with no device chosen. Until the SAPI voices (work-plan step 11)
are in, it plays silence and shows the banner "TTS not installed: outputs are silent".

## Development

```powershell
python -B -m pytest -m "unit or model" --disable-socket   # fast, no hardware
python -B -m pytest -m virtual --disable-socket           # needs VB-Cable
ruff format --check . ; ruff check . ; mypy --strict src
python scripts/check_no_cyrillic.py                       # English-only rule
```

Rules for contributors and AI agents live in [CLAUDE.md](CLAUDE.md).
Project agents are in `.claude/agents/`, project skills in `.claude/skills/`.

## Pause marking: Silero vs Silero + Smart Turn

An offline viewer shows, for real LibriSpeech clips and for synthetic streams with known
silence, the waveform, the Silero speech probability, the pauses found from it, where the
segmenter cuts, what the Smart Turn model answers during each pause, the reference labels and
the text. The table at the top of the page compares the two ways of ending a chunk (the
fixed `silence_ms` timer, and the timer plus Smart Turn) over all clips.

- Open the comparison: [data/pause_viewer.html](data/pause_viewer.html). The page is generated
  on your machine and not committed: `data/` is ignored and the page embeds LibriSpeech audio.
- Build it: `python -B scripts/pause_viewer.py` (needs `models/` and `data/` from
  `scripts/fetch_models.py --with-data`). Source: [scripts/pause_viewer.py](scripts/pause_viewer.py).

## License

MIT (to be added with the first code commit).
