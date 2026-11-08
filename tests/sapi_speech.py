"""Offline speech for tests: Windows SAPI voices, joined with pauses of known length.

Used by the model test (M1) and the virtual-cable tests (V1, V2). Nothing is downloaded;
the WAV files live in a pytest temporary folder, never in the pipeline.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

RATE_HZ = 16_000
SENTENCES = [
    "The meeting starts at nine tomorrow morning.",
    "Please send me the final report by Friday.",
    "We need to review the budget before the end of the month.",
    "Can you hear me clearly at the back of the room?",
    "The new software update fixes several security problems.",
    "Our customers expect a reply within twenty four hours.",
    "I would like to introduce our new head of engineering.",
    "Thank you all for coming and have a safe trip home.",
]
# Unfinished on purpose: each ends on a connective, a filler or an open article, which is
# what the Smart Turn vendor labels "incomplete". The label comes from the text, so no
# guess about where a speaker paused is needed.
UNFINISHED = [
    "I would like to order a pizza with mushrooms and",
    "Please send me the final report because",
    "The new software update fixes the problem but",
    "Our customers expect a reply within twenty four hours, so",
    "Can you tell me where the, um,",
    "I think that the budget for the",
    "We can start the meeting as soon as",
    "Thank you all for coming, and then",
]
VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop"]

_SYNTH_SCRIPT = """
Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, 'Sixteen', 'Mono')
$i = 0
foreach ($voice in $args[1].Split('|')) {
  foreach ($text in $args[2].Split('|')) {
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SelectVoice($voice)
    $s.SetOutputToWaveFile((Join-Path $args[0] ("u" + $i + ".wav")), $fmt)
    $s.Speak($text)
    $s.Dispose()
    $i++
  }
}
"""


def trim(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    """Cut leading and trailing quiet audio so the speech span is tight."""
    frame = 160
    n = len(samples) // frame
    level = np.sqrt(np.mean(samples[: n * frame].reshape(n, frame) ** 2, axis=1))
    loud = np.flatnonzero(level > 0.02)
    return samples[loud[0] * frame : (loud[-1] + 1) * frame]


def synthesize(
    folder: Path, sentences: list[str] | None = None
) -> tuple[list[NDArray[np.float32]], str]:
    """Synthesize every sentence with every voice into 16 kHz mono float32 arrays.

    Args:
        folder: Empty folder for the intermediate WAV files.
        sentences: Texts to speak; ``SENTENCES`` when omitted.

    Returns:
        The trimmed utterances, and an error text that is empty on success.
    """
    texts = SENTENCES if sentences is None else sentences
    script = folder / "synth.ps1"
    script.write_text(_SYNTH_SCRIPT, encoding="utf-8")
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-File",
            str(script),
            str(folder),
            "|".join(VOICES),
            "|".join(texts),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    files = sorted(folder.glob("u*.wav"), key=lambda p: int(p.stem[1:]))
    if result.returncode != 0 or len(files) != len(VOICES) * len(texts):
        return [], result.stderr[:200] or "no files produced"
    out = []
    for path in files:
        with wave.open(str(path), "rb") as wav:
            raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        out.append(trim(raw.astype(np.float32) / 32768.0))
    return out, ""


def build_stream(utterances: list[NDArray[np.float32]], seed: int) -> NDArray[np.float32]:
    """Three utterances separated by random pauses, some well below the threshold."""
    rng = np.random.default_rng(seed)

    def noise(ms: int) -> NDArray[np.float32]:
        return (rng.standard_normal(ms * 16) * 0.001).astype(np.float32)

    pieces = [noise(700)]
    for index in range(3):
        pieces.append(utterances[int(rng.integers(len(utterances)))])
        if index < 2:
            pieces.append(noise(250 if rng.random() < 0.4 else int(rng.choice([900, 1100, 1400]))))
    pieces.append(noise(1500))
    return np.concatenate(pieces)
