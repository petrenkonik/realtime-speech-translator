---
name: privacy-audit
description: Verify the hard constraints of this repo - zero network, zero disk writes, permissive licenses only - at test level and at OS level (Process Monitor, Airplane Mode, open connections). Use before a release, after adding a dependency or model, and for checks G1-G4, A1, A2.
---

# Privacy and license audit

## 1. Test level (automatic)

```powershell
python -B -m pytest -m unit --disable-socket -q -k "g1 or g2 or g3 or t5"
```

- G1: `no_disk_writes` fixture (audit hook on `open` write modes, `os.remove`,
  `os.rename`, `tempfile.*`) over a full fake pipeline run.
- G2: `--disable-socket` for every test.
- G3: the app's `--strict` guard aborts on a write or `socket.connect`.

Audit hooks do not see native code (onnxruntime, PortAudio, CTranslate2,
SAPI) - that is what the OS-level checks are for.

## 2. Licenses (G4)

```powershell
pip-licenses --from=mixed --allow-only="MIT License;MIT;Apache Software License;Apache-2.0;BSD License;BSD-3-Clause;Python Software Foundation License;ISC License (ISCL)"
pip-licenses --from=mixed --format=markdown --order=license
```

Models are not pip packages - check their cards by hand and keep a table in
the README:

| Model | License | Allowed |
| --- | --- | --- |
| Silero VAD | MIT | yes |
| Whisper weights (faster-whisper conversions) | MIT | yes |
| OPUS-MT (Helsinki-NLP) | CC BY 4.0 | yes, with attribution |
| M2M100 418M | MIT | yes |
| NLLB-200 | CC BY-NC 4.0 | **no** |
| Piper / espeak-ng | GPL-3 | **no** |

Also check that `faster-whisper` does not pull `av` (LGPL FFmpeg) into the
runtime import path.

## 3. OS level (A1, A2) - during the soak run

Network (A2), with Airplane Mode on:

```powershell
$p = (Get-Process python | Sort-Object StartTime | Select-Object -Last 1).Id
Get-NetTCPConnection -OwningProcess $p -ErrorAction SilentlyContinue
Get-NetUDPEndpoint  -OwningProcess $p -ErrorAction SilentlyContinue
```

Both must print nothing for the whole run.

Disk (A1), Sysinternals Process Monitor:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/procmon_check.ps1 -Minutes 60
```

The script starts `Procmon.exe /AcceptEula /Quiet /Minimized /BackingFile`,
stops it with `/Terminate`, exports CSV and filters by the demo PID for
`WriteFile` and write-mode `CreateFile`. Expected result: 0 events.
The capture file is written by Process Monitor, not the demo, and is ignored
by git (`*.pml`).

## Reporting

State each check as passed, failed (with output) or not run (with reason).
Never report an OS-level check as passed without having run it.
