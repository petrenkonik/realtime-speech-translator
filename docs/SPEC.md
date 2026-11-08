# Specification: Offline Real-Time Interpreter Demo

Source of truth for implementation. Mirrors the Russian design doc
(https://claude.ai/artifact/92tDd7zcmYfYBxFfjgYNH5); if they disagree, ask.

## 1. Goal

An offline simultaneous-interpretation pipeline for a company: **one
microphone, one source language**, translation into several languages, each
language played on its own output device. No audio or text is stored or sent
anywhere. The key metric is **latency from end of speech to start of the
translated audio**.

The demo (a public MIT repo + a 2-minute video) must prove:

1. The pipeline works live end to end: mic -> VAD -> faster-whisper (once)
   -> parallel translation into 2 languages (CTranslate2) -> TTS -> one output
   device per language.
2. Latency is low and measured, per stage and per language lane.
3. Privacy constraints are verified automatically: zero network, zero disk
   writes (audio and text), permissive licenses only, zero buffer overflows.

Demo simplifications: one speaker, source language fixed up front
(`--source en`), two targets (`--targets de,fr`). Decided 2026-11-08: English
source; German and French targets (an English target would be a pass-through
lane and is not needed).

The demo is generic (no client name) and owned by the author.

## 2. Client requirements mapping

| # | Feature | Client requirement |
| --- | --- | --- |
| 1 | WASAPI capture with native format auto-detection (24-bit/48 kHz capable) | M1.1 |
| 2 | Streaming resampling to 16 kHz mono in RAM | M1.2 |
| 3 | Silero VAD (ONNX), silence threshold 450-600 ms | M1.3 |
| 4 | ChunkBus: thread-safe multi-consumer chunk queue | M1.4 |
| 5 | Auto-reconnect of input and every output on USB loss | M1.5 |
| 6 | Latency log per stage and lane (RAM + console only) | M1.5 |
| 7 | 60-minute soak run under load with report | M1.5 (zero overruns) |
| 8 | Transcription once per utterance (faster-whisper) | M2 single-source transcription |
| 9 | Company glossary -> whisper hotwords/prompt + term protection in MT | M2 domain-specific prompt injection |
| 10 | Parallel translation into 2 languages (CTranslate2) | M2 multi-target translation |
| 11 | TTS per lane | M2 multi-stream TTS |
| 12 | One physical output device per language | M2 routing |
| 13 | Proof suite: disk/network guards in tests and `--strict`, Process Monitor, Airplane Mode | Constraints |
| 14 | License report + CI check | Constraints |
| 15 | Terminal dashboard (`rich`) | - |
| 16 | Smart Turn end-of-utterance detection as an extra finalization trigger (default `--turn-mode smart`; `fixed` = silence threshold only) and soft cuts inside long speech | M1.3: the 450-600 ms silence threshold stays configurable and remains the fallback trigger; see [SPEC-LOW-LATENCY.md](SPEC-LOW-LATENCY.md) |
| 17 | Sentence-level translation and TTS pipelining inside one utterance | M2 extension; see [SPEC-LOW-LATENCY.md](SPEC-LOW-LATENCY.md) |
| 18 | Control window (Tk): devices, output languages, settings, live process and transcript, report to the clipboard | - (operator aid; see "control window" in section 7) |

Out of scope: standalone exe packaging, multiple mics/speakers, source-language
auto-detection, translating partial ASR hypotheses (whisper still runs once
per finished chunk; splitting its *final* text into sentences is in scope),
non-Windows OS.

A control window (feature 18) is an operator aid and in scope; it is not a client requirement.
**Toolkit decision: Tk (`tkinter`, standard library).** Qt (PyQt, PySide) is LGPL/GPL and fails
the license check (G4); a local web server would open a listening socket and break the
zero-network proof (A2). Tk is part of Python (PSF License; Tcl/Tk has a BSD-style license), adds
no dependency, and uses neither sockets nor files. The window never opens a file dialog: the
report goes to the clipboard only.

## 3. Hardware

| Role | Device | Notes |
| --- | --- | --- |
| Live input | V380 FHD Camera (USB, WASAPI) | Format auto-detected |
| 24-bit/48 kHz input | VB-Cable configured to 24-bit/48 kHz | Stands in for the DJI Mic 2 |
| Output A (EN) | Realtek Speakers | |
| Output B (DE) | HS100W wireless headset, USB dongle | Separate USB device with its own clock and radio latency |
| Fallback output C | HDMI monitor Q2713S | Used if HS100W is unavailable |
| Test loopback | VB-Cable A+B | Test bench only, never shipped |

Devices are resolved by **name substring + WASAPI host API**, never by index.
Edge case not shown on video: HS100W powered off while the dongle stays
plugged in -> no WASAPI error, audio goes nowhere (documented limitation).

## 4. Architecture

```text
capture -> resample -> VAD/Segmenter -> ChunkBus --> transcribe --> TextBus --> LanguageLane(de) -> output A
                                               \--> meter                \--> LanguageLane(fr) -> output B
                                               \--> slow consumer (--slow-consumer, demo only)
metrics / dashboard / guards observe everything
```

- The capture callback only copies into a ring buffer. All heavy work
  (whisper, MT, TTS, playback) runs in its own threads behind non-blocking
  buses, so it can never cause an input overflow.
- Milestone 1 ends at ChunkBus. Everything after it previews Milestone 2.
- Threads, not processes: NumPy, onnxruntime and CTranslate2 release the GIL;
  arrays are shared without copying.

## 5. Latency budget (targets, to be replaced by measurements)

| Stage | Target (ms) | How |
| --- | --- | --- |
| VAD silence wait | 450-600 | Required by client; default 450. With `--turn-mode smart` the wait ends after `turn-check-ms` + model time (~270-320 ms) whenever Smart Turn says the phrase is complete; 450-600 stays the fallback (SPEC-LOW-LATENCY.md) |
| Chunk finalization | <= 50 | 10 ms capture blocks, 32 ms VAD windows |
| Transcription of a ~3 s utterance | <= 500 | int8, fixed language, `beam_size=1`, model size by measurement |
| Translation per language | <= 150 | Lanes in parallel threads, small per-pair int8 models |
| TTS to first audio | <= 300 | Sentence-by-sentence synthesis |
| Output buffer | <= 50 | `latency='low'`, short ring buffer |
| **Total** | **<= 1500** at 450 ms silence | HS100W radio latency added on top, measured acoustically (T6) |

Load and warm up all models at startup; set thread counts per engine
(`cpu_threads`, `intra_threads`) explicitly so engines do not fight for cores.

## 6. Library choices

| Task | Choice | License | Why | Rejected |
| --- | --- | --- | --- | --- |
| Capture/playback | `sounddevice` (bundled PortAudio) | MIT | NumPy callbacks, explicit WASAPI host API, `WasapiSettings`, `int24` via `RawInputStream` | PyAudio (bytes API) |
| Resampling | Own streaming polyphase resampler, filter from `scipy.signal.firwin` | BSD-3 | `resample_poly` is stateless (clicks at block edges) | `python-soxr` (LGPL), `librosa` (heavy deps) |
| VAD | Silero VAD ONNX + `onnxruntime`, own wrapper | MIT/MIT | Local file, sub-ms per 32 ms window | `silero-vad` pkg / `torch.hub` (PyTorch, may hit network); `webrtcvad` |
| End-of-turn | Smart Turn v3.2 `smart-turn-v3.2-gpu.onnx` (fp32, 32 MB; the int8 `-cpu` file is also fetched but is inaccurate on CPUs without VNNI) + `onnxruntime`, own wrapper and log-mel features | BSD-2-Clause (model card `pipecat-ai/smart-turn-v3`, verified 2026-11-07) | Audio-only semantic endpointing; 23 languages incl. ru, en, de; no network, no extra runtime | `transformers` / `torch` (heavy) |
| Queues | stdlib `queue.Queue`, one bounded queue per subscriber | PSF | Zero-copy sharing of read-only arrays | `multiprocessing` |
| STT | `faster-whisper` (CTranslate2), int8 CPU | MIT | Client-mandated, NumPy input | `openai-whisper` |
| MT engine | CTranslate2 | MIT | Client-mandated, int8, releases GIL | - |
| MT models | OPUS-MT per pair (CC BY 4.0, attribution); fallback M2M100 418M (MIT) | | Small and fast | **NLLB-200 (CC BY-NC) is forbidden** |
| TTS (demo) | Windows SAPI voices via `comtypes` | OS component / MIT | Offline, synthesizes into a memory stream, no license risk | Piper (GPL-3 + espeak-ng GPL-3), Kokoro (check espeak-ng need) |
| Dashboard | `rich` | MIT | Live console panel | `textual` |
| Control window | Tk (`tkinter`, standard library) | PSF (Python) / Tcl-Tk BSD-style | No new dependency, no sockets, no files | PyQt / PySide (LGPL / GPL), local web server (opens a socket, breaks A2) |
| Dev tools | `pytest`, `pytest-socket`, `pytest-cov`, `pip-licenses`, `ruff`, `mypy`, `psutil` | MIT/BSD | | |

TTS is behind `TtsEngine` (`synthesize(text) -> NDArray[np.float32]`),
selected with `--tts sapi|kokoro|piper`; the demo ships only `sapi`.

Known license risks: `faster-whisper` depends on `av` (PyAV bundles LGPL
FFmpeg) — verify whether it is imported at module load; if so, call
`ctranslate2.models.Whisper` directly. `huggingface_hub` (Apache-2.0) is
pulled in; always pass local model paths and set `HF_HUB_OFFLINE=1`.
NumPy/SciPy are BSD-3: permissive but not literally MIT/Apache — confirm
with the client.

`io.BytesIO` (mentioned by the client) is not needed: audio moves as NumPy
arrays between all stages and is never encoded to WAV.

## 7. Functional requirements

### capture (features 1, 5) - implemented, step 4

- `--input <name>`: case-insensitive substring of the device name on the
  **WASAPI** host API, input channels only. An exact name wins; two substring
  matches are an error (`AmbiguousDeviceError`), never a guess. `list-devices`
  prints the WASAPI devices (`format_device_table`). Devices are never addressed
  by index: Windows renumbers them.
- Open at the device's native rate and channel count (`default_samplerate`),
  10 ms blocks, `latency='low'`, float32 (PortAudio converts int24 losslessly).
  We resample, not Windows. Measured on this machine: the V380 camera microphone
  is 16 kHz mono (resampler passes it through), HS100W and the Realtek devices
  are 48 kHz.
- **Callback** (`_Link.on_block`): copy the block into a preallocated ring
  (`ring_seconds`, 2 s), count callbacks and `input_overflow`, record its own
  duration (`max_callback_ms`). Nothing else; no clock calls, locks or logging.
- `RingBuffer` is single-producer / single-consumer without locks: the producer
  stores data, then publishes the write position. If the processing thread falls
  more than a ring behind, the oldest frames are lost and counted
  (`ring_lost_frames`), so a slow stage can never make the driver drop input. A
  consumer lapped during a copy drops the possibly overwritten frames instead of
  passing noise on. A block larger than the ring keeps its tail.
- **Processing** (`CaptureSession.tick()`, run by one thread every `poll_ms`, 5):
  read the ring -> `ChannelMixer` -> `StreamingResampler` -> `VadStage` ->
  `ChunkBus`. Polling instead of waking the thread from the callback keeps the
  callback free of synchronization; it costs at most `poll_ms` of delay.
- **Watchdog** (`supervisor.py`, shared with the outputs): the stream is lost when
  `stream.active` is false, or no callback happened for `watchdog_ms` (500; also
  counted from the moment of connecting). The watchdog compares callback counters
  against an injected clock, so tests use a fake clock.
- **Reconnect**: on loss the ring is drained, the VAD stage is flushed (the pending
  chunk is published with reason `flush`, pre-roll is cleared), the stream is
  closed, then every `retry_ms` (1000; the first retry is immediate) PortAudio is
  re-initialized (`DeviceListing.reinitialize`, it caches the device list), the
  device is looked up by name again and reopened. Counters (`callbacks`,
  `input_overflow`, `ring_lost_frames`) are totals across reconnects;
  `reconnects` counts successful reconnections after the first connection.
- **Startup is strict**: `start()` raises `DeviceNotFoundError`,
  `AmbiguousDeviceError` or `StreamOpenError` when the device is missing, instead of
  waiting silently. Only later losses are retried. A device whose native rate the
  resampler does not support (not 16/32/44.1/48 kHz) ends the thread with
  `fatal_error` set (shown on the dashboard) instead of retrying forever.
- `stop()` joins the thread, closes the stream and publishes the pending chunk.
- **Finalization latency metric** (`CaptureStats.finalization_latency_ms`): for each
  chunk that ended on silence, stream position when it was published - end of
  speech - silence threshold. It includes the segmenter's rounding of the threshold
  up to whole 32 ms frames (up to 31 ms), so the F5 budget of 50 ms leaves about
  20 ms for blocks, polling and the resampler.
- Not implemented: `--raw-int24` (`RawInputStream`); float32 already holds 24-bit
  audio exactly, so it is only worth adding if a driver mishandles float32.
- Adapters: `devices.py` (`SoundDeviceListing`, `open_sounddevice_input`) is the only
  code that imports `sounddevice`, lazily. Unit tests cover it with a fake module;
  `tests/test_capture_hw.py` (marker `hw`, run only on request) opens the real V380.

### resample and audio conversion (feature 2) - implemented, step 1

Three small components in `pcm.py` and `resample.py`; all pure NumPy/SciPy,
all run outside the audio callback.

**`StreamingResampler(in_rate_hz, out_rate_hz=16000).process(block)`**
- Inputs 16/32/44.1/48 kHz; others raise `UnsupportedSampleRateError` with the
  rate in the message. 16 kHz passes through (copied, never filtered).
- Polyphase linear-phase FIR designed with a Kaiser window
  (`scipy.signal.kaiserord` + `firwin`): passband edge 7 kHz, stopband edge
  8 kHz, **65 dB** stopband (5 dB margin over the required 60 dB). Edges scale
  with the lower of the two rates.
- Streaming: the last `taps - 1` input samples are carried between calls, so
  the output does not depend on block boundaries (no clicks, unlike the
  stateless `resample_poly`). Output length over a whole stream is exactly
  `ceil(N * out / in)`; empty blocks are legal.
- Cost per output sample is one dot product of 129-193 taps; group delay
  **2.0 ms** at every supported rate (budget: 3 ms).
- Stereo input is accepted and averaged first (`to_mono`).

**`ChannelMixer(rate_hz, mode="auto"|"mean", tau_ms=300).process(block)`**
(capture uses it; flag `--downmix auto|mean`, default `auto`)
- Problem: the plain channel mean loses **6 dB** when one channel is dead (a
  mono mic on a stereo device) and gives **silence** when the channels are in
  antiphase (miswired cable).
- `mean` = `to_mono`: the plain average, kept for reference and U2.
- `auto`: per-channel weights proportional to the smoothed RMS (DC removed, so
  a stuck-offset dead channel stays dead), summing to 1. Identical channels
  give exactly the input; a dead or hissing channel gets weight ~0.
- For two channels, a smoothed correlation below -0.5 flips the polarity of
  channel 2 (back above 0.0, hysteresis). `polarity_inverted` is exposed for
  the dashboard.
- Weights are ramped linearly across each block, so state changes never click.
- Not handled: time-offset channels (comb filtering between two separate
  mics). More than two channels are weighted by level only.

**`float_to_int16(block, dither=True, rng=None)`**
- Used wherever the program itself reduces float audio to 16 bit (output
  device opened as int16 through `RawOutputStream`). It is not on the capture
  path: capture is float32 (24-bit audio fits float32's 24-bit mantissa), and
  with a float32 stream PortAudio quantizes itself.
- Scale 32767, TPDF dither (two uniform draws subtracted, +-1 LSB), round,
  saturate (no wrap-around). Block of exact zeros stays exactly zero, so
  pauses carry no hiss. Pass a seeded `rng` for reproducible tests.
- Effect: the quantization error stops being a function of the signal. Without
  dither a tone below 1 LSB disappears and a larger tone gets harmonic lines;
  with dither the tone survives and the error is a flat floor (variance 1/4
  LSB^2: 1/12 rounding + 1/6 dither, about -98 dBFS).

**Measured with these tests (2026-11-08, this machine)**

| Property | Result | Limit |
| --- | --- | --- |
| Rejection of 8.1-15 kHz tones (32/44.1/48 kHz in) | -70 dB worst case (8.1 kHz), -80 to -89 dB above 9 kHz | <= -60 dB |
| Passband gain 100 Hz - 7 kHz | within 0.004 dB | +-0.1 dB |
| Distortion + noise of a 1 kHz tone | -99 to -107 dB | <= -80 dB |
| -90 dBFS tone through the resampler | gain error 0.0 dB | +-0.1 dB |
| -80 dBFS tone next to a 0.5 amplitude tone | level kept, error < 0.5 dB | +-0.5 dB |
| Group delay | 2.0 ms | <= 3 ms |
| Dead channel, `mean` vs `auto` | -6.02 dB vs < 0.05 dB | < 0.05 dB |
| Antiphase, `mean` vs `auto` | 0 (silence) vs level kept within 0.1 dB | 0.1 dB |

Known limits, accepted: a sine that starts abruptly overshoots by up to ~0.8%
during the first milliseconds (filter start-up), so every conversion to
integers saturates; one-LSB-level tones through `float_to_int16` need the
default dither to survive.

### vad (feature 3) - implemented, step 2

- `SileroVad(model_path)`: 512-sample frames (32 ms) at 16 kHz. The model is
  recurrent, so it keeps its state and a 64-sample context (the last 64 samples
  of the previous frame, prepended to every call; without it accuracy drops).
  `reset()` after any gap. Measured 0.48 ms per frame on the target CPU, about
  1.5% of one core. Model: Silero VAD v6.2.3, MIT, 2.3 MB, pinned by URL tag and
  sha256 in `scripts/fetch_models.py`.
- `VadStage(vad, segmenter).push(samples)`: accepts blocks of any length, keeps
  the partial frame, returns finished chunks. The same chunks come out for any
  block size. `flush()` finalizes the open chunk, drops the partial frame and
  resets the model (used on reconnect).
- `Segmenter`: pure state machine over probabilities (testable without the
  model). Rules for every frame of an open chunk:
  - `p >= 0.5` is speech and stops the silence clock.
  - `p < 0.35` starts the silence clock if it is not running. Frames in between
    are neutral: they neither start nor stop it.
  - The chunk ends when the clock has run for `silence_ms` (450-600, else
    `VadConfigError`; rounded up to whole frames, so 450 ms means 15 frames =
    480 ms), **or** when no speech frame was seen for twice that long
    (noise hovering between the thresholds must not hold a chunk open until the
    15 s cap).
  - A chunk reaching `max_utterance_ms` (15 s) is cut at the quietest frame of
    the last `cut_search_ms` (1.5 s); the rest stays open as the next chunk and
    is never dropped as "too short" (it is a continuation of speech).
  - Trailing silence is trimmed to `tail_ms` (160 ms). `Chunk.speech_end_sample`
    keeps the real end of speech for the latency metric.
  - Chunks with less than `min_speech_ms` of speech are noise and are dropped
    (counted in `dropped_short`; they do not consume a chunk id). Dropped frames
    stay available as pre-roll for the next chunk.
  - `pre_roll_ms` (100) of audio before the first speech frame is kept, but
    never audio that already belongs to the previous chunk.
- `Chunk`: frozen; `id` (from 1), `samples` (float32 16 kHz, `writeable=False`),
  `start_sample`, `end_sample`, `speech_end_sample`, `finalized_sample`, `reason`
  (`silence`, `max_length`, `flush`).
- Differences from the first draft of this spec, decided in step 2:
  `min_speech_ms` 250 -> **160** (a one-word answer such as "yes" or "no" is
  shorter than 250 ms and was being thrown away; the whisper-side filters in
  the risk table below handle the extra noise); ambiguous probabilities no
  longer reset the silence clock; trailing silence is trimmed.
- Ultra-low-latency extension (Smart Turn end-of-utterance detection, soft
  cuts in long speech, sentence pipelining, per-stage tuning):
  [SPEC-LOW-LATENCY.md](SPEC-LOW-LATENCY.md). Its `--turn-mode smart` is the
  default there; `--turn-mode fixed` gives exactly the silence-threshold
  behavior described in this file.

#### Chunk processing risks (found while building step 2)

Where the chunk-per-utterance design can go wrong, what is done about it, and in
which step. "Open" means a decision or measurement is still needed.

| # | Risk | Effect | Handling | Step |
| --- | --- | --- | --- | --- |
| C1 | Short answers ("yes", "no", "stop") below `min_speech_ms` | Dropped, the listener hears nothing | Default lowered to 160 ms; tested | 2, done |
| C2 | Probability hovering at 0.35-0.5 (fan, breath, quiet end of a word) | Chunk stays open to the 15 s cap, up to 15 s of delay | Neutral band + stall rule (twice the silence) | 2, done |
| C3 | A forced cut leaves a short remainder | Remainder dropped as "too short", end of the sentence lost | Continuations are exempt from the min-speech rule | 2, done |
| C4 | Trailing silence inside the chunk | Whisper hallucinates "Thank you." on silence, latency metric starts too late | Tail trimmed to 160 ms, `speech_end_sample` recorded | 2, done |
| C5 | Block alignment: the resampler returns arbitrary lengths, Silero needs exactly 512 | Lost or doubled audio at block edges | `VadStage` keeps the partial frame; same chunks for block sizes 160/512/777 | 2, done |
| C6 | Reconnect or stall: the sample counter has a gap | Pre-roll glues audio from before the gap to speech after it | `flush()` clears pre-roll and resets the model | 2, done |
| C7 | Soft onsets ("s", "f", "th") before the VAD fires | The first sound of a word is cut | Pre-roll is configurable; 100 ms gave onset error within 40 ms on synthetic speech; 300 ms still proposed for real microphones | Open (needs H1) |
| C8 | Whisper hallucinations on noise that the VAD let through | Invented sentences are translated and spoken | `no_speech_threshold`, `log_prob_threshold`, `compression_ratio_threshold`, chunk RMS gate, repeated-phrase filter | 9, done |
| C9 | Whisper falls behind (decode time > speech time) | `drop_oldest` silently deletes speech | Before dropping, merge queued chunks (gap <= 1 s, total <= 15 s) into one decode; drop only past that | 9, done |
| C10 | Whisper always encodes a 30 s window, so a 1 s chunk may cost almost as much as a 15 s one | Many short chunks are slow in total | Measure encoder time against chunk length in M5 before choosing the merge rule | Open (M5) |
| C11 | Cut in the middle of a sentence (soft cut, 15 s cap, premature Smart Turn) | Two translations of half-sentences, bad grammar, broken TTS prosody | Pass the tail of the previous text as `initial_prompt`; hold a fragment without final punctuation until the next chunk or 700 ms | 9 done (context), 9b open (hold) |
| C12 | A short burst of speech-like noise passes the VAD and `min_speech_ms` | Garbage reaches whisper | Whisper filters in C8 | 9, done |
| C13 | A pause inside a sentence longer than the silence threshold (thinking pause) | The sentence is split into two chunks | Accepted for `fixed`; Smart Turn and the hold mode (low-latency spec) address it | 2b |

### bus (feature 4) - implemented, step 3

- One generic class `Bus[T]`; `ChunkBus = Bus[Chunk]`. `TextBus` will be
  `Bus[Utterance]`.
- `subscribe(name, maxsize=32, policy="drop_oldest")` returns a `Subscription`.
  Names are unique. A subscriber sees every item published after it subscribed.
  `policy` is `drop_oldest` (default: fresh speech matters more) or `drop_newest`.
- `publish()` never blocks on a consumer: it takes each subscriber's lock for a
  few instructions and moves on. The subscriber list is replaced as a whole on
  `subscribe`, so publishing takes no bus-wide lock. The same object goes to all
  subscribers (shared by reference; `Chunk` is immutable).
- Consumer side: `Subscription.get(timeout_s=None)` blocks; it returns `None`
  when the bus is closed and drained, or when the timeout passed (`finished`
  tells the two apart). Iterating a subscription ends at close.
- Stats per subscriber: `delivered`, `dropped`, `depth`, `max_depth`; `Bus.stats()`
  returns them by name, `Bus.published` counts accepted items.
- `close(drain=False)`: by default the backlog is **discarded**, so every consumer
  stops as soon as it finishes the item in hand. A consumer sleeping 2 s per chunk
  with 32 queued items would otherwise take over a minute to exit, which breaks the
  1 s shutdown rule. `drain=True` lets consumers read the backlog first (used by
  tests that need every item). `close()` is idempotent.
- `publish()` after `close()` is silently ignored (the audio thread can race with
  shutdown); `subscribe()` after `close()` raises `BusClosedError`.
- Demo subscribers: transcribe, meter, `--slow-consumer` (sleeps 2 s/chunk).

### transcribe and glossary (features 8, 9) - implemented, step 9

- `transcribe.py`: `TranscribeStage` is the only `ChunkBus` subscriber that leads to
  translation. It runs in one daemon thread (`step()` is the same code for tests) and
  publishes frozen `Utterance` objects (first chunk id, all chunk ids, text, start /
  speech-end / finalized sample, audio length, decode time, reason) on the `TextBus`. The
  text lives only in RAM and is never logged.
- The decode sits behind `SpeechToText`; `FasterWhisperEngine` is the adapter:
  `WhisperModel("models/whisper-base.en", device="cpu", compute_type="int8")`, local path
  only (`local_files_only=True`, `HF_HUB_OFFLINE=1`), `language=<source>`, `beam_size=1`,
  `temperature=0.0` (a single temperature: no random re-decoding, bounded time),
  `vad_filter=False`, `condition_on_previous_text=False`, `without_timestamps=True`.
  `warm_up()` decodes a second of silence at startup.
- **Model choice: an option, `TranscribeConfig.model`** (CLI flag `--whisper-model`, see
  `config.py`). The source language is English only, so the English-only Systran
  CTranslate2 conversions of OpenAI Whisper are used (MIT):
  - `base.en` (145 MB), **the default**: 5.4% WER on read speech, about 2.2 s per decode;
  - `tiny.en` (75 MB), the fast option: 7.2% WER, about 1.2 s per decode.
  Both are pinned in `fetch_models.py`; the code refuses any other name. `small.en` is out
  (about 7 s per decode here). Numbers: SPEC-LOW-LATENCY.md section 12.4.
- **`av` (PyAV) is not installed.** `faster_whisper` imports it at load time (this answers
  the open question) but uses it only to decode files. Its wheels bundle LGPL FFmpeg
  libraries, so `[tool.uv] override-dependencies` removes it and
  `transcribe.ensure_av_stub()` installs a stub module that fails with an explanation if
  anything tries to decode a file. `check_licenses.py` fails if `av` is ever installed.
  With plain pip use `pip install --no-deps faster-whisper==1.2.1` after the rest.
- Chunk risks handled here:
  - **C8** RMS gate (`min_rms` 0.002) before the decode; whisper's `no_speech_threshold` /
    `log_prob_threshold` / `compression_ratio_threshold` during it; `collapse_repeats` after it
    (a phrase repeated 4 or more times in a row becomes one; "no no no" is kept).
  - **C9** the stage empties the bus into its own list on every step, so a full queue never
    drops speech while whisper is busy. Waiting chunks at most 1 s apart are decoded in one
    pass up to 15 s, with up to 250 ms of silence between them (`transcribe.merged_chunks`).
  - **C11** the end of the previous text (at most 200 characters, from a word boundary)
    goes to whisper as `initial_prompt` when the previous chunk was cut by force
    (`max_length`, `soft`, `rescued`) or its text has no final punctuation, and the next chunk
    starts within 2 s. Holding a fragment back for the next chunk is step 9b.
- Counters and histograms (`Metrics`): `transcribe.chunks`, `.utterances`, `.merged_chunks`,
  `.skipped_quiet`, `.skipped_empty`, `.repeats_collapsed`, `.errors`, `.decode_ms`, `.rtf`.
  A failing decode drops that chunk, counts the error and keeps `last_error`; the stage goes on.
- `glossary.py`: `--glossary terms.toml`, read once. `[[term]]` tables with `source` and one
  key per target language (`de = "..."`). The source terms (at most 400 characters in
  total) go to whisper as `hotwords`. `Glossary.protect(text, language)` swaps each term
  (case-insensitive, whole words, longest first) for a placeholder `XQ<n>Z` before
  translation; `Protected.restore(translated)` puts the wording from the file in its place
  and reports placeholders the translator dropped in `Restored.missing`. A term without a
  wording for the language is restored as written. Whether the real translators copy the
  placeholders through is checked with them in step 10 (T4).
- **Measured (this machine, `base.en`, SAPI speech, 4 threads):** WER 0.0% on 16 sentences
  with the 100 ms / 160 ms padding a chunk has, 1.3% without it (two small word errors on
  two utterances; numerals are normalized). Hotwords raise the correct spelling of 5 made-up
  company terms from 5 to 11 of 14 occurrences with `base.en` and from 3 to 13 with `tiny.en`. Decode time **1.7 s median** for 2-3 s of
  speech (the latency consequence is in SPEC-LOW-LATENCY.md section 12.4). Hotwords also made
  two of eight outputs lose a leading word and the final full stop; step 9b must not treat a
  missing full stop as proof that a sentence is unfinished.

### supervisor and output (feature 5) - implemented, step 6

- `supervisor.py`: `Supervisor` keeps one stream alive for capture and for every
  output: lookup by name (`kind` input or output), watchdog, close, rescan, reopen
  every `retry_ms`. The owner passes `open_link` (builds the stream and its state),
  `before_close(link, reason)` (flush what is pending) and `after_close(link)` (carry
  the counters over). `TickThread` runs a `tick()` in a daemon thread and keeps an
  exception in `fatal_error` for the dashboard instead of dying silently.
- `output.py`: one `OutputSession` per device; sessions share nothing.
  - `OutputRing`: mono float32 ring, lock-free for one producer (the lane) and one
    consumer (the audio callback). `write()` accepts only what fits and never
    overwrites unplayed audio; the producer decides whether to wait (`write_all`, with
    a timeout) or drop. `discard_queued()` marks everything written so far as stale.
  - The audio at the ring is mono at the device's native rate (`sample_rate_hz`); the
    lane resamples TTS output with `StreamingResampler` and rebuilds that when
    `generation` changes (a reconnect may land on a device with another rate). The
    callback copies the mono signal to every opened channel; at most 2 channels are
    opened. float32 goes to PortAudio, which converts to the device format.
  - `begin_utterance()` / `end_utterance()` bracket speech. **Underrun** = the ring
    ran dry while an utterance is open and playback had already started. Idle silence
    between utterances, waiting for the prebuffer and the tail after `end_utterance()`
    are not underruns (S2: "0 underruns during speech"). `driver_underflows` counts the
    flag PortAudio reports on its own.
  - **Prebuffer** (`prebuffer_ms`, 20, spec O1): at the start of an utterance, and after
    a ring that ran dry, playback waits until that much audio is queued so the first
    callbacks do not starve; a short utterance below the prebuffer plays as soon as it
    ends.
  - While the device is disconnected `write()` discards the audio and counts it
    (`dropped_frames`) instead of queueing it: late interpretation is worse than skipped.
    Queued audio of a lost stream is counted as dropped too.
  - Same reconnect rules as capture (500 ms watchdog, first retry immediate, then 1 s,
    PortAudio rescan). Startup is strict: a missing device raises.
  - The real stream is `open_sounddevice_output` (WASAPI shared, float32, `latency='low'`).
    The hw test for it plays sound, so it is written only when the lane exists (step 10).

### TextBus and language lanes (features 10-12)
- `TextBus` = same implementation as ChunkBus, for `Utterance`.
  One `LanguageLane` per target (`--targets de,fr`,
  `--route de=Realtek,fr=HS100W`).
- `LanguageLane` (`lanes.py`, implemented): own thread; `Translator` (CTranslate2, one model per
  pair) -> `TtsEngine` -> its device's `OutputSession`. Lanes share no models or buffers. All
  three are `Protocol`s (`Translator.translate(text)`, `TtsEngine.synthesize(text) -> Speech`,
  `AudioSink`), so the lane is tested without models or sound. The lane subscribes with
  `maxsize=2`; `step()` handles one utterance, an exception is counted and kept in `last_error`
  and the lane goes on. While the output is disconnected the utterance is skipped before any
  translation (`skipped_offline`). Metrics `lane.<target>.`: `utterances`, `errors`,
  `skipped_empty`, `skipped_offline`, `write_timeouts`, `glossary_missing`, `translate_ms`,
  `tts_ms`, `latency_ms` (stream position at the first queued audio - end of speech, plus the
  radio offset; needs the `position_samples` clock of the capture).
- Translations for a window: the lane gets an optional `TranslationBus` (`Bus[Translation]`) and
  publishes one frozen `Translation(utterance_id, target, text, translate_ms, tts_ms,
  latency_ms)` per utterance that got a translation, as soon as the audio starts or as soon as
  it is clear that none will (synthesis failed, nothing to play, playback failed; then `tts_ms` /
  `latency_ms` may be `None`). A translator error, a blank utterance and an offline output
  (skipped before translating) publish nothing. `publish()` is non-blocking with drop-oldest, so
  a stalled reader cannot hold a lane back. `text` is RAM only and left out of `repr`; it never
  reaches metrics or logs.
- Glossary terms are replaced by placeholders before MT and restored with the
  configured translation afterwards.
- TTS per sentence (implemented in `LanguageLane`): the translation is cut with `split_sentences`;
  the first sentence is queued as soon as it is synthesized and plays while the next is made (the
  output ring holds seconds, so `write_all` returns at once). One `begin_utterance` /
  `end_utterance` pair brackets the whole utterance, one resampler serves all its sentences so the
  filter history runs through the boundaries, and `tts_ms` and the latency describe the first
  audio. A failing later sentence ends the utterance (counted as an error, the device is not left
  speaking); a write timeout drops the rest. TTS audio is converted to the device rate with
  `RateConverter` (`resample.py`: any rate from 8 to 192 kHz, same polyphase filter as
  `StreamingResampler`, which stays strict for the microphone). (Sentence-level translation
  pipelining: SPEC-LOW-LATENCY.md.)
- `tts_sapi.py`: `SapiTts(target, TtsConfig, backend_factory)` speaks with the SAPI 5 voices the user
  installed (`comtypes`, `SAPI.SpVoice` into a fresh `SAPI.SpMemoryStream`, 22.05 kHz 16-bit mono,
  converted to float32). The voice is chosen by language (de-DE before de-CH) and, optionally, by
  part of its name (`--tts-voice de=Hedda`); a voice of another language is never used, and a
  missing one raises `VoiceNotFoundError` naming the installed voices and the language-pack fix.
  COM objects live on the thread that made them: the backend is created lazily on the thread of
  the first `synthesize` (the lane thread) and once per thread. SAPI pads every text with about
  100 ms of silence before and 800 ms after it; `trim_silence` keeps 20 ms of lead-in and a 150 ms
  pause. `check_voices(configs)` and `SapiTts.check()` fail before the start; `sapi_tts_factory`
  builds the `tts_factory` of `Pipeline`.
- Each output: own callback and ring buffer; empty buffer -> silence +
  underrun counter. Reconnect like the input; one output failing never affects
  others.
- A lane more than 2 utterances behind drops the oldest (late interpretation
  is worse than skipped); drops shown on the dashboard.
- Lane latency = start of translated audio - end of speech, plus
  `stream.latency`; wireless outputs add the measured radio offset
  (`--radio-offset de=<ms>`, see T6).

### pipeline orchestrator (`app.py`) - implemented, GUI plan step 1

- `Pipeline(settings, *, devices, input_stream_factory, output_stream_factory, stage_factory,
  speech_to_text, translator_factory, tts_factory, glossary=None, resampler_factory=None,
  metrics=None, clock=time.monotonic)` builds `ChunkBus`, the VAD stage (once, via
  `stage_factory(settings)`), `TranscribeStage`, `TextBus` and, per route, an `OutputSession`
  and a `LanguageLane`. The real translator and TTS are not part of it: they come through the
  factories (`target -> Translator`, `target -> TtsEngine`). A `Pipeline` is used once: after
  `stop()` the buses are closed, so new settings mean a new `Pipeline`.
  `Pipeline` is used once: after
  `stop()` the buses are closed, so new settings mean a new `Pipeline`.
- `start(lenient=False)`: outputs (strict by default: a missing device raises; with
  `lenient=True` an output that cannot be opened stays disconnected and its target is returned,
  so a window can start with a device absent and `connect_output` it later; the microphone is
  always required), whisper, lanes, microphone last; a
  failure closes everything opened so far. `stop()` (idempotent): microphone (publishes the
  pending chunk), chunk bus closed with its backlog kept, whisper, text bus, lanes, outputs.
  Queued speech not yet played is discarded; there is no drain mode.
- Runtime control: `connect_input` / `disconnect_input`, `connect_output(target)` /
  `disconnect_output(target)` (idempotent, need a running pipeline), `set_input_device(query)`
  and `set_output(target, query)` (name substring, never an index; the new name is looked up
  first and a miss leaves the old device alone; while the device is disconnected only the
  choice is stored). The input is re-created as a new `CaptureSession` around the same stage,
  so chunk ids and the stream clock continue. Each lane writes to a stable sink whose
  `OutputSession` is stopped, started or replaced; the lane, its subscription and the other
  lanes are never restarted.
- `set_output_target(old, new)` changes the language of one output: validated against
  `SUPPORTED_TARGETS` (not the source, not used by another lane; `old == new` is a no-op),
  the new translator and TTS are built first (a failing factory leaves the old lane working),
  then the old lane's `TextBus` subscription is removed (`Bus.unsubscribe`) and the lane
  stopped, and a new lane with a new subscription takes over the same sink and device
  binding. Other lanes, their counters and the microphone are untouched; `settings.targets`
  and `routes` are updated in place. Utterances queued for the old lane are dropped.
- `translation_bus`: the lanes' `Translation`s for a window; closed at `stop()` after the lanes.
- State: `input_state()` / `output_state(target)` return `DeviceState` `connected`,
  `reconnecting` (wanted but the stream is lost and the supervisor is retrying; this does not
  depend on `last_error`, which is only set once a retry has failed) or `disconnected` (not
  wanted, never started, stopped, or the supervising thread died).
- Read-only accessors for a window: `text_bus`, `chunk_bus`, `metrics`, `settings` (reflects
  device changes), `targets`, `running`, `position_samples`, `capture_stats()` (summed over
  connections), `output_stats(target)`, `bus_stats()`, `input_device()`, `output_device(target)`.
  They take no lock, so a polling window is never held up by a device that is slow to open.

### metrics and dashboard (features 6, 15) - implemented, step 5

- `metrics.py`: `Counter`, `Histogram` (last N values, nearest-rank p50/p95, max,
  lifetime count) and the `Metrics` registry (`counter(name)`, `histogram(name)`,
  `snapshot()` flattened to `name -> number`). In memory only, one lock per metric,
  used from worker threads and never from the audio callback. Capture and the buses
  already keep their own counters; `Metrics` is for what has no other home
  (per-stage and per-lane latency, whisper real-time factor).
- Finalization latency = stream position when the chunk is published - end of speech
  - silence threshold (computed by `CaptureSession`, see capture).
- `dashboard.py`: a dashboard is a set of *sections*, each a function returning
  `(label, value)` rows (`capture_rows`, `bus_rows`, `metrics_rows`; later stages add
  their own). `rich.Live` at 4 Hz in `live` mode; `plain` mode (`--no-ui`) prints one
  line per second for soak runs. A section that raises shows its error and the rest
  keeps drawing. No file logs. Rows carry numbers and device names; utterance text
  will come from its own section that honors `--hide-text`.

### guards (feature 13) - implemented, step 5

- `StrictGuard` (`guards.py`) wraps `sys.addaudithook`. Audit hooks cannot be removed,
  so one process-wide hook is added on first use and asks the guards that are armed
  (`arm()` / `disarm()` / `with guard:`). `--strict` arms it after the models load.
- Blocked: `open` for writing (mode `w a x +`, or `os.open` flags write/create/append/
  truncate), `os.remove`, `os.rename`, `os.rmdir`, `os.mkdir`, `os.truncate`,
  `os.symlink`, `os.link`, `tempfile.mkstemp`, `tempfile.mkdtemp` (this covers
  `TemporaryFile`), and `socket.connect`, `getaddrinfo`, `gethostbyname`, `sendto`.
  Reading files is fine. The null device, console handles and already-open
  descriptors are not disk writes.
- The default handler prints `strict mode: <disk|network> access blocked (...)` to
  stderr and ends the process with `os._exit(70)`. An exception would be swallowed by
  a worker thread; tests pass a handler that raises or records instead.
- Native code (PortAudio, onnxruntime, CTranslate2) is not covered -> Process Monitor
  (A1) and Airplane Mode (A2).
- Coverage note: code running inside the audit hook is not traced by coverage, so
  `classify()` is also tested directly, without a hook.

### settings schema (`ui/settings_schema.py`) - implemented, GUI plan step 4

- Tk-free description of every user-tunable setting; the settings tab is generated from it.
  `Field` (frozen, slotted): `key` (dotted path such as `segmenter.silence_ms`), `label`, `help`
  (one sentence: what it does, what it costs), `kind` (`int`, `float`, `choice`, `bool`),
  `default`, `minimum` / `maximum` / `step` or `choices`, `unit` (`ms`, `s`, `none`), `group`
  (Segmentation, Turn detection, Whisper, Capture, Output, Advanced), `apply` (`ApplyMode.LIVE`
  or `RESTART`; the configs are frozen so everything restarts except `hide_text`), `advanced`,
  `hard_minimum` / `hard_maximum` (the real constructor enforces the bound; `False` for a limit
  of the window only) and `off_value` (0 for `soft_cut_ms`).
- Covered: all of `SegmenterConfig`, `TranscribeConfig` (not `language`), `CaptureConfig` and
  `OutputConfig` (not the device query or `poll_ms`; `output.*` applies to every route), the
  radio offset of each route (`routes.<target>.radio_offset_ms`), `turn_threads`, `strict`,
  `hide_text`, and the translator's beam and threads. `compute_type` and `device` stay
  hardcoded in `transcribe.py`. `NOT_EXPOSED` lists what is left out on purpose.
- Defaults are read from the dataclasses. Ranges are read from the constants the modules
  export (silence range, thread ranges, radio offset limit, model names, downmix and turn
  modes); the rest are written out and proved against the real constructors by SC3.
- API: `SCHEMA`, `schema_for(settings)` (adds one radio offset field per route),
  `fields_by_group(settings=None)`, `values_from_settings(settings)`,
  `build_settings(base, values)`, `diff(old, new)`, `needs_restart(old, new)`.
  `segmenter.silence_ms` is the one silence field and sets the capture config too.
- `build_settings` checks types, ranges and four cross-field rules (a break is reported on both
  keys), then runs the real constructors through `construct_settings`, which are the final
  authority. All problems are collected into `SettingsValidationError.errors` (`key ->
  message`); a constructor message is mapped back to the fields it names, otherwise it goes to
  `GENERAL_ERROR_KEY`. Nothing is written to disk and nothing is global.

### window view-model (`ui/model.py`) - implemented, GUI plan step 3

- `ViewModel(pipeline, *, max_rows=500, thresholds=None, clock=time.monotonic, drain_cap=1000,
  text_queue=256, translation_queue=512)`: no Tk. It reads the `Pipeline` only through the
  `PipelineView` protocol (the accessors of `app.py` plus `settings`), so tests can give it a
  fake. It owns two subscriptions named `view-model-<n>` (the first free `n`; `drop_oldest`) on
  `text_bus` and `translation_bus`. `poll() -> Snapshot` drains each with a zero timeout and at
  most `drain_cap` items, builds the snapshot and returns; it is meant for a UI tick every
  100-250 ms. `close()` unsubscribes both (idempotent; `poll()` after it returns the rows
  already held). A stalled or never polled view-model loses only its own items
  (`Summary.view_dropped`); a closed bus at construction raises `ViewModelError`.
- `Snapshot` (frozen, slotted, validated, tuples only, so the UI never races the poller):
  `input` (`InputCard`: device name and configured query, `DeviceState`, capture counters;
  `level` is always `None` because the capture counters carry no signal level), `outputs`
  (`OutputCard` per target: device, state, `LaneCounters` from `Metrics` - utterances, errors,
  `skipped_offline`, latency p50 / p95 of `lane.<t>.latency_ms`, `None` before the first sample
  - plus underruns, driver underflows, dropped frames, reconnects, losses), `buses`
  (`BusRow` per subscriber of the chunk, text and translation buses), `transcribe`
  (`TranscribeStats`: decode ms and real-time factor p50 / p95, errors), `stages`, `summary`
  and `rows`.
- Health (`Health` = `ok | warn | bad | idle`), per stage `input`, `transcribe`,
  `translate:<t>`, `output:<t>`, judged on every poll. `idle`: the pipeline is not running or
  the stage has had nothing to do yet (a lane whose output is away is `idle`: it only skips).
  `bad`: the microphone or an output is `disconnected` (an output disconnected on purpose looks
  the same; the card says why). `warn`: `reconnecting`; or, within `warn_hold_s` (10 s,
  injected clock) of the last increase, a new `input_overflow`, `ring_lost_frames`, output
  underrun, driver underflow or dropped output frames, transcription error, chunk dropped
  before whisper, lane error, or utterance dropped for a lane; or whisper p95 real-time factor
  above `rtf_p95_warn` (1.0). Counters that are already non-zero when the view-model starts
  are history, not news. Details are numbers only.
- Transcript: one row per `Utterance`, in arrival order. `pause_ms` = (`start_sample` - the
  previous row's `speech_end_sample`) / 16, never negative, `None` for the first row and kept
  when older rows are capped; `start_s` from stream samples; `audio_ms`, `decode_ms`,
  `finalize_wait_ms` (`finalized_sample` - `speech_end_sample`), `reason`. Translations join
  by `utterance_id` into one `TranslationCell` per target: the lane order at the time the row
  was created, plus any other target that later delivers. A late translation updates its row
  (older snapshots stay as they were); a translation read before its utterance waits (bounded);
  a translation for a capped row is ignored. `arrived=False` means a dash; an offline output
  publishes no translation, so its cell stays a dash by design. After
  `set_output_target` old rows keep the old language, new rows expect the new one. Beyond
  `max_rows` the oldest rows go (`Summary.rows_dropped`).
- `set_hide_text(bool)` takes effect at the next poll: text fields (`TranscriptRow.text`,
  `TranslationCell.text`) are `None`, every timing stays, `arrived` still says whether a
  translation exists. Text fields are out of `repr`; text never reaches stage details,
  metrics or exception messages.
- `Summary`: rows seen / shown / dropped, translations received, mean and p95 end-to-end
  latency of the last 1000 translations that had audio, `bus_dropped` (all subscribers of the
  three buses), `view_dropped`, input overflow, ring-lost frames, underruns, errors
  (transcription plus lanes).
- `export_text()` / `render_report(snapshot)`: a plain-text report for the clipboard: a summary
  header, then a tab-separated table (`#`, `start_s`, `pause_ms`, `audio_ms`, `decode_ms`,
  `finalize_wait_ms`, `reason`, `text`, and `<t>`, `<t>_latency_ms` per target); newlines and
  tabs in text become spaces; `[hidden]` when hidden, `-` for a pending translation. It returns
  a string; nothing is written to disk.

### control window (`gui` command, `ui/`) - implemented, GUI plan step 5

- **Start.** `python -B -m pipeline gui [run options]`. The options are those of `run`
  (`config.add_run_arguments(parser, gui=True)`), except that `--input` and `--route` are optional
  (a device that is not named becomes the placeholder `config.UNSELECTED_DEVICE`, shown as an
  empty box and refused by every connect) and `--no-ui` does not exist. With `--input` named and
  no model missing the window connects at start; otherwise it opens idle. The window opens even
  with no device selected or available. `python -B -m pipeline list-devices` prints the WASAPI
  table; `run` and `soak` belong to later steps. A usage or range error prints a message and
  exits with 2.
- **Layers.** Everything except the last four modules is Tk-free and tested without a display
  (WD14 scans the imports): `present.py` (strings, options and cell texts), `settings_form.py`
  (texts, parsing, hints, notes, error placement), `controller.py` (`WindowController`),
  `wiring.py` (composition root); `app_tk.py` (`MainWindow`, `run_gui`), `transcript_view.py`,
  `settings_view.py` and `tk_widgets.py` only place widgets.
- **Threading.** Tk lives in the main thread only. `MainWindow` calls
  `WindowController.tick()` from `root.after` every 200 ms (and right after each user action) and
  draws the returned `WindowState` (snapshot, settings, generation, busy label, notice, missing
  models, device names, session flag, TTS banner). `tick()` never blocks: it drains the finished
  jobs of the `JobRunner` (one daemon worker thread, FIFO) and polls the `ViewModel`
  (zero-timeout queues, lock-free pipeline statistics). Everything slow runs on the worker:
  building and starting a pipeline, connect / disconnect of the microphone or an output,
  switching a device, loading a translator for a new language, the device scan, and the stop and
  rebuild of a restart. While one job is pending the controller reports it as `busy` and refuses
  further actions (the window disables its buttons and boxes and shows a progress bar and the
  label); `hide_text` stays available. Pipeline threads and the worker never touch a widget.
- **Sessions.** The controller always holds one session. *Standby* is a `StandbyPipeline` (a
  `PipelineView` that is stopped and has nothing connected) so the same panels draw before
  anything exists; choices made there (devices, language, settings) are only stored, in the same
  `Settings` the next pipeline is built from. *Active* is a built and started `Pipeline`.
  A `Pipeline` cannot start without its microphone, so "Connect" of the microphone in standby is
  "build and start the pipeline" (`start(lenient=True)`: an output that cannot open stays
  disconnected and is named in a warning). Outputs connect together with it; their Connect
  buttons are disabled in standby. Disconnecting the microphone keeps the pipeline running.
- **Apply and restart.** `apply_settings(values)` runs `settings_schema.build_settings` on the
  current settings (the running pipeline's, which include device and language choices made at
  run time, plus the live `hide_text`) in the UI thread, so errors come back at once with their
  field keys. No change: a notice. Only `hide_text` changed (`needs_restart` is false): applied
  live, no restart. In standby: stored. Otherwise: models are checked first (`Wiring.problems`;
  a missing model or voice refuses the apply before anything is stopped), then one worker job does
  `old.stop()`, `old.close()` (releases what the wiring made for it), `Wiring.build(new)`,
  `start(lenient=True)`, restores the devices the user had disconnected on purpose, and creates
  a new `ViewModel`. The UI thread then swaps the session, closes the old view-model and
  increases `generation`; the window clears its transcript (a new pipeline starts a new stream
  clock). If the restart fails (microphone busy, model cannot load) the controller falls back
  to standby with the new settings and an error notice, and the user can connect again.
  A language change while running is `Pipeline.set_output_target` on the worker (only that
  lane); the transcript and the session are kept.
- **Panels.** *Input*: microphone box (WASAPI input names, alphabetical, from
  `SoundDeviceListing`), status dot with a word (`Connected`, `Reconnecting`, `Not connected`),
  Connect / Disconnect, Refresh devices (PortAudio is re-initialized only in standby), the last
  error, and a level meter that appears only if a snapshot ever carries a level (today
  `InputCard.level` is `None`). *Outputs*: per lane a language box (`SUPPORTED_TARGETS` without
  the source and without languages used by other lanes), a device box (WASAPI outputs), a dot
  and Connect / Disconnect. *Process*: a fork diagram, not one flat row. Left, one row: *Microphone*
  then *Speech recognition (English)*. From recognition a connector (stem, vertical bar, one
  arrowed branch per lane, drawn on a canvas) leads into a titled *In parallel* box with one row
  per target, each row labelled `de - German` and holding *Translate to German (de)* then
  *Speak / output de*; the only arrow inside a lane joins its own two boxes, and no arrow joins
  one lane to another (after recognition every lane gets the utterance at once and runs on its
  own thread). With one lane there is one row, with none only the left part. `present.py`
  groups the snapshot's stage keys (`input`, `transcribe`, `translate:<t>`, `output:<t>`) into
  `ProcessLayout(common, lanes)` and computes the connector's segments (`fork_geometry`); the
  window rebuilds only when the set of keys changes. One box per stage key, coloured ok green,
  warning amber, problem red, idle grey and always carrying the word, the reason, and numbers
  (decode or lane latency p50 / p95, drops, overflows, lost frames, underruns, errors).
  *Transcript*: `ttk.Treeview` with time (stream clock), pause, text, one column per target
  (languages that older rows still carry stay as columns), and processing times (ASR and
  per-language latency); a row per utterance, updated in place when a translation arrives (`-`
  while pending), following the newest row unless the user scrolled up; the full text of the
  selected row under the table; Hide text and Copy report (clipboard only) above it; the summary
  line below. *Settings*: one notebook page per group, built when first shown, a widget per
  `Field` kind (spin box with min / max / step, drop-down, switch), range and unit, help text,
  a restart note per field and per page, an inline error under the field, Advanced fields hidden
  until "Show advanced settings" (an error in one opens them), Apply, Discard edits, Reset to
  defaults. Banners: a speech notice (only for a wiring that returns one; `RealWiring` never
  does) and, inline instead of a dialog, the list of what is missing (models, files, speech
  voices). A status bar shows the busy state and the latest notice. Fonts are 10 pt and
  every colour pair has at least 4.5:1 contrast.
- **Wiring (`ui/wiring.py`).** `RealWiring.build(settings)` loads Silero VAD (fresh per build),
  Smart Turn plus `ThreadedTurnLink` when `turn_mode` is `smart` (the link is closed by
  `BuiltPipeline.close`), faster-whisper (loaded once per model and thread count and reused
  with the new decode options through `FasterWhisperEngine.with_config`), one CTranslate2
  translator per target (reused per folder and `TranslateConfig`), the glossary if given, and
  returns a `Pipeline` that is not started. Everything is read from `settings.models_dir`
  (`HF_HUB_OFFLINE=1`); every loader, the stream factories, the device list and the strict guard
  are injected. With `strict` it arms the `StrictGuard` after the loads, otherwise it releases it.
  A failure while building closes what was created.
- **Speech (WD10, WD15-WD17).** `RealWiring.build` passes `tts_factory` from `Loaders.tts`
  (default `tts_sapi.sapi_tts_factory` over the routes' `LaneRoute.tts`) and `resampler_factory`
  from `Loaders.resampler` (default `RateConverter`, so a 22.05 kHz voice reaches any device
  rate). `Loaders.voice_backend` (default `ComVoiceBackend`) and `Loaders.files` (default
  `check_files`) are injectable, so tests use fakes and never touch COM. The engines a build
  creates are closed by `BuiltPipeline.close`. A lane switched at run time has no route yet: it
  speaks with the default voice of its language and the rate and volume of the first lane
  (`lane_tts_config`); `WindowController` and `Pipeline.set_output_target` drop a voice name on a
  language change (it belongs to the old language) and keep rate and volume.
  *Missing voices are problems, not a banner.* `Wiring.problems` also asks SAPI (`check_voices`,
  one question per target and voice name) and returns one message per target, for example
  `voice missing: no SAPI voice for language 'fr'. Installed voices: ... Install the language pack
  with text-to-speech (...), then restart the program.`; a COM failure is
  `speech synthesis unavailable for <target>: ...`. The controller treats them like missing models:
  inline list, Connect refused, apply refused before anything stops, a language change in standby
  re-checks, and the window and the other settings keep working. A running lane that is switched
  to a language without a voice fails in `set_output_target` before the old lane is touched (the
  `build` check is the same cached question). `tts_notice` stays in the `Wiring` protocol and the
  banner code in `app_tk.py` for any wiring that wants a notice; `RealWiring.tts_notice` is `None`.
  *COM threading.* The Tk and worker threads never hold a COM object. `SapiTts` makes its backend
  on the lane thread of the first sentence. `problems` (called by the controller on the Tk thread,
  and by `build` on the worker) runs `check_voices` on a short-lived helper thread that is
  joined (30 s limit): `ComVoiceBackend` initialises COM there, closes there, and the apartment
  ends with the thread. Answers are cached under a separate lock (the model lock can be held
  for seconds by a load), so only the first call per language blocks the Tk thread, about 0.6 s
  for de and fr on the development machine; a COM failure is not cached and is asked again.
  Installing a language pack needs a restart of the program, as the message says.
  *Strict mode.* SAPI speaks into `SpMemoryStream` and the check only lists voices, so neither
  writes or connects; WD17 runs the real check and a real sentence under the `no_disk_writes`
  fixture (the real `StrictGuard`, English voice, no audio device). Audible playback through the
  window on a speaker is not covered by any test (it needs the operator at the machine).
- **Strict mode.** Tk, the clipboard and the controller use no sockets and write no files (WD13
  runs the whole window flow under `no_disk_writes` and `--disable-socket`); `--strict` therefore
  needs no exception for the window. Tk reads its own Tcl files, which the guard allows.

### licenses (G4)

- `scripts/check_licenses.py` (CI step): runtime dependencies, taken as the closure of
  `[project].dependencies`, must be permissive (MIT, BSD, Apache, PSF, ISC, Zlib, 0BSD,
  CC0). `A AND B` needs both, `A OR B` one. Development tools are not shipped: GPL,
  LGPL, AGPL and non-commercial licenses fail, weak copyleft (`pathspec`, MPL-2.0, a
  `mypy` dependency) only warns.

## 8. Code standards

See [CLAUDE.md](../CLAUDE.md). Everything in the repo is English.

## 9. Tests

Hardware is injected: streams come from factories; tests use
`FakeInputStream` / `FakeOutputStream` (random block sizes, faster than real
time, scripted `input_overflow` / `PortAudioError`). Time is the stream sample
counter, so results do not depend on machine speed. All runs use
`python -B` and `--disable-socket`.

| Marker | Needs | Where | Time |
| --- | --- | --- | --- |
| `unit` | nothing | CI (`windows-latest`), every push | < 30 s |
| `model` | models in `models/` (cached in CI) | CI | 1-2 min |
| `virtual` | VB-Cable A+B | local or Hyper-V VM | 2-3 min |
| `hw` | V380 + Realtek + HS100W | manual checklist before the video | 10 min |
| `soak` | VB-Cable / real outputs + CPU load | once, report in README | 60 min |

| ID | Check | Level | Criterion |
| --- | --- | --- | --- |
| U1 | int24 unpack | unit | `0x7FFFFF -> +1.0`, `0x800000 -> -1.0`, `0 -> 0.0`, little-endian, exact |
| U2 | Stereo -> mono | unit | Channel mean; antiphase gives zero |
| U3 | Resampled length | unit | 48->16: exactly N/3 per stream; 44.1->16: +-1 sample |
| U4 | Block-size invariance | unit | Random blocks vs one-shot: max abs diff < 1e-5 |
| U5 | Frequency preserved | unit | 1 kHz sine -> FFT peak at 1 kHz +-1 bin |
| U6 | Anti-aliasing | unit | 10 kHz tone at 48 kHz in -> <= -60 dB out |
| U7 | VAD state machine on fake probabilities | unit | Pause = threshold - 64 ms -> 1 chunk; threshold + 64 ms -> 2; for 450, 525, 600 |
| U8 | Silence threshold validation | unit | 449 and 601 -> config error |
| U9 | Forced split | unit | 40 s continuous speech -> chunks <= 15 s, no lost samples |
| U10 | Pre-roll | unit | Chunk start = speech start - 100 ms +- 32 ms; never overlaps the previous chunk; limited by stream start |
| U10b | Segmenter edge cases | unit | Short click dropped and counted, one-word answer kept, ambiguous-band noise cannot hold a chunk, trailing silence trimmed, forced-cut remainder kept, random-input invariants (no overlap, speech covered) |
| U10c | Silero wrapper and `VadStage` | unit | 64-sample context, state carried, `reset()`, block-size invariance (160/512/777/whole), `flush()` |
| U11 | Fan-out | unit | 8 subscribers x 10 000 chunks: all received, ordered, no duplicates |
| U12 | Slow subscriber | unit | `publish` < 1 ms (99th percentile, so one OS hiccup does not fail CI); drops counted; others got everything; drop_oldest keeps the newest, drop_newest keeps the oldest |
| U13 | Shutdown | unit | All threads exit within 1 s of `close()`, also with a slow consumer and a full backlog; `drain=True` delivers the backlog; publish/subscribe after close |
| U14 | Chunk immutability | unit | Writing `chunk.samples` -> `ValueError` |
| U27 | Resampler delay | unit | Impulse through `StreamingResampler` at 32/44.1/48 kHz: peak delay <= 3 ms (measured 2.0 ms). `test_resample.py` |
| Q1 | Downmix, one dead channel | unit | `mean` loses 6.02 dB (documented); `auto` keeps the level within 0.05 dB for a silent left or right channel, a -90 dBFS hissing channel, and a channel stuck at a DC offset. `test_downmix.py` |
| Q2 | Downmix, both channels active | unit | Identical channels pass through exactly (1e-6); two independent full-scale tones each land at -6 dB +-1 dB with no clipping; a channel 20 dB quieter does not pull the level down by more than 1 dB. 3 identical channels pass through |
| Q3 | Downmix, antiphase | unit | `mean` gives exact zero; `auto` recovers the level within 0.1 dB and reports `polarity_inverted`; uncorrelated noise is never flipped |
| Q4 | Downmix, no clicks | unit | A channel that switches on mid-stream, or a polarity swap mid-stream: largest sample step <= 1.5x / 2x that of the plain sine. Digital silence stays exactly silent; empty block, mono input, channel-count change and bad arguments are handled |
| Q5 | Aliasing | unit | Control: naive decimation of a 10 kHz tone stays at 0 dB. Tones 8.1/9/10/12/15 kHz at 32/44.1/48 kHz and band-limited noise above 8.5 kHz are rejected by >= 60 dB. `test_resample_quality.py` |
| Q6 | Passband | unit | 100 Hz, 1/3/5/7 kHz tones at 32/44.1/48 kHz keep their level within 0.1 dB |
| Q7 | Dynamics and detail | unit | 1 kHz tone: distortion + noise <= -80 dB; a -90 dBFS tone keeps its level within 0.1 dB; a -80 dBFS tone next to a 0.5 amplitude tone keeps its level within 0.5 dB; a settled full-scale 7 kHz tone peaks at <= 1.001 |
| Q8 | Quantization, small signals | unit | `float_to_int16`: 0/+1/-1 exact, saturation instead of wrap-around; without dither a 0.3 LSB DC level and a 0.4 LSB tone vanish; with dither the DC mean is 0.3 +-0.01 and the tone is recovered (amplitude +-15%, >= 15 dB over the floor). `test_dither.py` |
| Q9 | Quantization, distortion | unit | Error spectrum of a 10.3 LSB sine: harmonic lines (> 30 dB peak-to-median) without dither, flat (< 15 dB) with dither |
| Q10 | Dither statistics | unit | Error variance 1/12 LSB^2 without dither and 1/4 with (+-5%); dithered error has mean < 0.01 LSB and correlation with the signal < 0.01; a seeded generator reproduces the output; digital silence gets no noise |
| Q11 | 24-bit precision | unit | int24 -> float32 -> int24 differs by at most 1 step over random data, so 24-bit input needs no dither |
| F1 | Whole pipeline on fakes | unit | Synthetic utterances with known pauses -> expected chunks and boundaries, for 48 kHz stereo, 44.1 kHz mono, 16 kHz mono, 32 kHz stereo, random block sizes |
| F2 | Overflow accounting | unit | Fake raises the flag 3 times -> counter = 3 |
| F3 | Input reconnect | unit | Stream dies mid-utterance -> pending chunk finalized (`flush`), factory called again, chunks resume, reconnects = 1; watchdog at 500 ms; retries 1 s apart; open and start failures retried; counters survive; startup errors raise |
| F4 | Device lookup by name | unit | Same device found after the device list is reordered |
| F5 | Finalization latency | unit | <= 50 ms beyond the silence threshold, by stream clock, for 450, 525 and 600 ms |
| F6 | Output independence | unit | Output A fails mid-speech and stays away for 1 s -> B plays every sample once, in order, 0 losses and 0 underruns; A drops the audio meant for the outage, reconnects (reconnects = 1) and plays again; a failing open on A does not touch B |
| F6b | Output behavior | unit | Ring order across the wrap point, partial writes never overwrite unplayed audio, discard keeps later audio; underrun only during an open utterance; prebuffer waits and then starts at the first sample; short tail plays; driver underflows counted; write_all waits or times out; watchdog 500 ms; counters survive a reconnect; mono -> every channel; devices with more than 2 channels open 2 |
| G1 | No disk writes | unit | `no_disk_writes` fixture (`conftest.py`, an armed `StrictGuard` that records) over a whole capture run with a reconnect: 0 violations |
| G2 | No network | all | `pytest-socket` is on by default (`addopts`): `socket.socket()` raises `SocketBlockedError` |
| G3 | `--strict` works | unit | Every write mode, `os.open` flags, remove/rename/mkdir, temp files and network events are blocked; reads are not; in a subprocess `open('x', 'w')` and `socket.connect` end the process with code 70, the message, and no file created |
| G4 | Licenses | CI | `scripts/check_licenses.py` (uses `pip-licenses`): runtime dependencies permissive, any GPL/LGPL/AGPL/NC fails; classifier unit-tested |
| M1 | Silero on labeled audio | model | 10 streams of three SAPI-synthesized sentences with known pauses (900-1400 ms and 250 ms): chunk count matches in >= 9/10, median boundary error <= 100 ms. Measured: 10/10, median 41 ms (start within 40 ms, speech end 40-110 ms late). The LibriSpeech variant was dropped: it needs a 300 MB download |
| M2 | Whisper on chunks | model | WER <= 15% on SAPI-synthesized sentences, one utterance per chunk; loads and decodes with the network blocked. Measured: 0.0% (`base.en`) and 1.3% (`tiny.en`) on 16 sentences; the test runs for both models. Synthetic speech is easy: the number shows the wiring works, not how whisper does in a room |
| T1 | Lane independence | unit | DE blocked by a fake translator -> EN latency unchanged (+-10 ms), DE drops counted |
| T2 | One transcription per utterance | unit | 2 lanes x N utterances -> fake whisper called exactly N times. Also: RMS gate, repeated phrases, merge of waiting chunks, context for cut sentences, failing decode, thread start / stop |
| T3 | Glossary | unit + model | Terms reach `hotwords`; a term in the utterance appears in the translation exactly as in the glossary (fake translator). Model part: hotwords raise correct spelling of unknown terms from 5 to 11 of 14 |
| T4 | Translation quality | model | 20 reference sentences en->de, en->fr: chrF not below the first-run baseline; <= 150 ms per sentence |
| T5 | TTS in memory | model | Non-empty array; first audio <= 300 ms; G1 fixture sees no disk writes |
| T6 | HS100W radio latency | hw | Click played on A and B at once, V380 mic between speaker and earcup -> arrival difference by cross-correlation (median of 10) -> `--radio-offset`; dashboard shows per-output latency |
| T7 | End-to-end latency | virtual | English utterance into CABLE-A -> start of translation on CABLE-B: median <= 1.5 s, p95 <= 2.5 s over 20 utterances (450 ms silence) |
| T8 | Dongle without headset | hw, not on video | Headset off, dongle plugged -> app does not crash, other lanes fine; documented limitation |
| V1 | VB-Cable loopback, int24/48 kHz | virtual | Reference audio into CABLE-A -> same chunk count as M1; 0 overflows. Played by `OutputSession`, captured by `CaptureSession` with the real Silero; the chunk count is compared with an offline run of the same signal; also 0 ring loss and 0 output underruns. The cable's bit depth is a Windows setting and is not checked |
| V2 | Output reaches device | virtual | Output B -> CABLE-B; recording correlates with the chunk; 0 underruns during speech. Normalized cross-correlation of the recording with the reference >= 0.8; 0 underruns and 0 driver underflows |
| V3 | Device restart | virtual | `pnputil /restart-device` on CABLE-A (admin) -> stream restored <= 3 s |
| P1 | Pipeline lifecycle | unit | `start()` opens every output, then the microphone last; `stop()` closes every stream, ends every thread and closes both buses; `stop()` is idempotent and safe before `start()`; `start()` twice or after `stop()` raises `PipelineStateError`. `test_app.py` |
| P2 | Pipeline end to end on fakes | unit | One spoken utterance -> one whisper call, one translation per lane, audio on every output, the same utterance on an extra `TextBus` subscription; accessors report capture, output, bus and device state |
| P3 | Stop flushes | unit | Speech without trailing silence, then `stop()` -> the pending chunk (reason `flush`) is published and a subscriber of the chunk bus still reads it |
| P4 | Input disconnect / reconnect | unit | `disconnect_input()` closes the stream and publishes the pending chunk; `connect_input()` builds a new `CaptureSession` around the same VAD stage: utterance ids and the stream clock continue, one stage for the whole life; both calls idempotent; capture counters are summed across connections; control calls before `start()` raise; a missing device raises and leaves the input disconnected |
| P5 | Reconnecting state | unit | A lost input or output is `reconnecting` (device gone, stream failed) until the device returns, then `connected`; the other lane is `connected` and keeps playing while the lost one counts `skipped_offline` |
| P6 | Output disconnect / reconnect | unit | `disconnect_output(de)` -> its lane skips utterances without translating (`skipped_offline`), the other lane plays every one and its stream is never reopened; `connect_output(de)` plays again; idempotent; missing device raises; unknown target raises `UnknownTargetError` |
| P7 | Change a device by name | unit | `set_output` opens the new device before closing the old one (make-before-break), changes only that lane's session, updates `settings.routes`; unknown name keeps the old output; while disconnected the choice is stored only; empty name rejected. `set_input_device` does the same for the microphone (looked up before the old one is closed) |
| P8 | Failed start | unit | A missing output or microphone at `start()` raises, closes everything opened so far and leaves the pipeline stopped |
| P9 | Stalled UI subscriber | unit | A never-read `TextBus` subscriber of size 1 drops its own items (counted) and holds back no lane |
| P10 | No disk writes | unit | `no_disk_writes` fixture over start, utterance, input and output disconnect / reconnect, `set_output` and `stop`: 0 violations |
| P11 | Single route | unit | One route builds one lane and opens one output |
| U15 | Unsubscribe | unit | `Bus.unsubscribe(sub)` closes that subscriber (a blocked `get` wakes up, backlog discarded), removes it from the stats and from later publishes, leaves the others' items and counters alone, frees the name, and is idempotent (returns False for an unknown subscription) |
| P12 | `Translation` | unit | Frozen; `text` out of `repr`; impossible values rejected (`TranslationError`); a lane publishes exactly one per spoken utterance with id, target, restored text, timings, latency (radio offset included, `None` without a stream clock); a lane without a bus works as before. `test_lanes.py` |
| P13 | When a translation is published | unit | Published once even if TTS or playback fails (`tts_ms` / `latency_ms` `None`); not published after a translator error, a blank utterance or while the output is offline (`skipped_offline`, unchanged semantics); text never in metrics. `test_lanes.py` |
| P14 | Stalled translation reader | unit | A never-read `TranslationBus` subscriber of size 1 drops its own items (the newest is kept) and holds back no lane; a closed bus does not stop a lane. `test_lanes.py` |
| P15 | `Pipeline.translation_bus` | unit | Every lane's translation arrives with its target and text; closed by `stop()`; a stalled subscriber holds back no lane; a disconnected output publishes none |
| P16 | `set_output_target` | unit | Running lane changes language on the same stream (not closed or reopened), mic untouched, old thread gone and new one running, text-bus stats show the new lane only; with two lanes only the changed one is rebuilt (the other keeps translator, counters, stream); `set_output` still works afterwards; the radio offset stays |
| P17 | `set_output_target` validation | unit | Unsupported language, the source language, a language used by another lane -> `ConfigError`; unknown old target -> `UnknownTargetError`; nothing changes; `old == new` is a no-op |
| P18 | `set_output_target` situations | unit | A translator factory that raises leaves the old lane working; stopped pipeline -> `PipelineStateError`; before `start()` the lane is swapped; a disconnected output stays disconnected and the new lane skips, then plays after `connect_output`; threads end with `stop()` |
| P19 | Lenient start | unit | `start(lenient=True)` with an output absent returns its target, leaves it `disconnected` (lane skips, others play), `connect_output` works later; default stays strict and returns `()`; a missing microphone still raises and closes the outputs |
| P20 | No disk writes | unit | `no_disk_writes` fixture over translation and `set_output_target`: 0 violations |
| SC1 | Settings round trip | unit | `build_settings(base, values_from_settings(base)) == base` (two routes and one route); values cover every schema field; changed values reach the right dataclass (`output.*` on every route, `routes.<target>.radio_offset_ms` on one); a partial mapping leaves the rest alone. `test_settings_schema.py` (SC1-SC10; the ID `S1` is taken by the soak test) |
| SC2 | Schema defaults | unit | Every `Field.default` equals the default of the real dataclass; the silence default is the default of both `SegmenterConfig` and `CaptureConfig` |
| SC3 | Schema bounds vs the real constructors | unit | For every numeric field: minimum, maximum and default are accepted by the real constructors and by `build_settings`; one unit beyond a hard bound is rejected by the real constructors alone (`construct_settings`); one unit beyond any bound is rejected by the schema (with the other field of a cross-field rule set first). A soft bound (a limit of the window, such as 8 whisper threads where the class allows 32) is not a hard one; `soft_cut_ms` accepts 0 (off); every choice is accepted and others are not |
| SC4 | Cross-field rules | unit | `exit_threshold <= speech_threshold`, `turn_threshold <= soft_cut_threshold`, `tail_ms <= silence_ms`, `cut_search_ms <= max_utterance_ms / 2`: a break is reported on both keys (also when only one is edited), the edges are accepted, several problems are reported together; the real constructors' messages map back to field keys, a message that names none goes to `GENERAL_ERROR_KEY`; `SettingsValidationError.errors` and its message |
| SC5 | One silence field | unit | `segmenter.silence_ms` sets the segmenter and the capture config; there is no `capture.silence_ms` field (it is in `NOT_EXPOSED`; as a key it is unknown) |
| SC6 | Schema shape | unit | Keys unique; six groups in order, each non-empty; label and one-sentence help on every field; Advanced group implies `advanced`; every config field is in the schema or in `NOT_EXPOSED` (devices, `language`, polling, `max_decoding_length`); a self-contradicting `Field` raises `SettingsSchemaError`; fields are frozen |
| SC7 | Apply modes and diff | unit | Only `hide_text` is `live`; `diff` lists changed fields as `key -> (old, new)`; `needs_restart` is false for live-only changes and true as soon as one restart field changes; device names are not part of the schema |
| SC8 | Value types | unit | Wrong types (string, float for an int, bool for a number, `None`, NaN, inf) and unknown keys are rejected; an int is fine for a float field; the message names the range |
| SC9 | Routes | unit | Radio offset is per route and only exists for routes that exist; `output.*` applies to every route |
| SC10 | Tk-free | unit | `settings_schema.py` imports no Tk; `build_settings` does not mutate its base |
| VM1 | Snapshot shape | unit | Fresh snapshot: cards per target, input card with `level=None`, no rows; snapshots are frozen tuples and an old one never changes; impossible values and bad settings raise `ViewModelError`; `model.py` imports no Tk (static scan). `test_ui_model.py` (VM1-VM16; the ID `V1` is taken by the VB-Cable test) |
| VM2 | Row timings | unit | `pause_ms` from stream samples (`None` for the first row, never negative, survives polls), `start_s`, `audio_ms`, `decode_ms`, `finalize_wait_ms`, `reason` |
| VM3 | Translations join | unit | Per-target cells by `utterance_id`; pending cell is `arrived=False`; a late translation updates its row without changing older snapshots; one that beat its utterance waits; one for a capped row is ignored; `None` latency and TTS kept |
| VM4 | Row cap | unit | `max_rows` drops the oldest rows, keeps the first remaining row's pause, counts `rows_dropped` |
| VM5 | Hide text | unit | Text fields `None` (also for rows that arrive while hidden), timings and `arrived` stay, text absent from `repr`, metrics and stage details; showing again restores the text |
| VM6 | Export | unit | Summary header and tab-separated table, one line per row, no stray tabs or newlines, `-` for pending, `[hidden]` when hidden (latencies stay); `render_report` is pure |
| VM7 | Cards and counters | unit | Device name, state and lane counters / latency p50 / p95 from `Metrics`; capture, whisper, output and bus numbers reach the snapshot |
| VM8 | Health colours | unit | Stage keys in signal order; stopped is idle; disconnected bad, reconnecting warn; a new overflow, ring loss, underrun, decode error, slow decode or lane falling behind warns for the hold time and then clears; pre-existing counts are history; details carry no text |
| VM9 | Summary | unit | Seen / received counts, mean and p95 latency (none for a translation without audio), bus drops, view drops, overflows, underruns, errors |
| VM10 | Slow view | unit | A never polled view-model with tiny queues never delays publishers, keeps its newest items and counts its own drops; one poll reads at most `drain_cap` items per bus; polling 500 rows is far below a UI tick |
| VM11 | Close | unit | `close()` removes both subscriptions and is idempotent; two view-models get distinct names and a freed name can return; `poll()` after `close()` keeps the rows; a closed bus raises `ViewModelError` and leaves nothing subscribed |
| VM12 | Real pipeline | unit | On a real `Pipeline` with fakes: a spoken utterance becomes a row with both translations, timings, cards, all stages `ok`; a second utterance has a pause |
| VM13 | Device states | unit | Input disconnect then connect, a lost input `reconnecting` then `connected`, an output that is away (`bad`, lane `skipped_offline`, dash cell) |
| VM14 | Language change | unit | After `set_output_target` cards, stages and buses show the new target; old rows keep the old language, new rows expect the new one |
| VM15 | Real close and stall | unit | `close()` frees the subscriptions of a real pipeline, which keeps working; a never polled view-model on a real pipeline slows nobody and counts its drops |
| VM16 | No disk writes | unit | `no_disk_writes` fixture over polls, hide text, export, disconnect and close: 0 violations |
| WD1 | `gui` and `list-devices` commands | unit | `parse_settings(gui=True)` accepts no `--input` / `--route` (placeholder device, partial routes keep what was given), `run` still insists on both, other flags still validate, `--no-ui` is refused for the window, `main` prints a message and returns 2 for bad options, `list-devices` prints only WASAPI devices, the `gui` command hands settings to the runner. `test_gui_cli.py` (WD1-WD14 are the window IDs; `W1`-`W6` belong to SPEC-LOW-LATENCY) |
| WD2 | Device options and status | unit | WASAPI input / output names only, once each, alphabetical; the displayed device is what the query finds (empty for "none selected"); every device state has a word and a kind; the Connect button follows state and busy flag. `test_ui_present.py` |
| WD3 | Language options | unit | Supported, not the source, not used by another lane (own language always offered); labels round-trip |
| WD4 | Snapshot formatting | unit | A word on every stage card, latency / drops / overflows / underruns on the right card, stream time format, transcript cells (time, pause, text, translations or `-`, processing), `[hidden]` keeps timings, columns follow the targets, summary line, level only if present, no utterance text outside the transcript cells |
| WD5 | Settings form | unit | Texts from settings, typed values back (an untouched form rebuilds the same `Settings`), non-numbers refused per field, discard / reset / rebase (edits survive a baseline change), hints with range and unit, restart note per field and page, errors placed on fields (general ones apart, advanced ones flagged), real validation errors reach their fields. `test_ui_settings_form.py` |
| WD6 | Controller standby | unit | Starts with no pipeline and everything idle, also with no device chosen; device scan on the worker; a failed scan is a notice; refresh re-initializes PortAudio only without a pipeline; missing models shown and connect refused; choices stored in standby; language rules as the pipeline's; output buttons need a running pipeline; hide text live; the speech banner is whatever the wiring says (none for `RealWiring`); a language change in standby drops the voice name. `test_ui_controller.py` |
| WD7 | Connect and device control | unit | Build and start on the worker (`lenient`), auto-connect, a microphone or build failure leaves a usable window and releases what the wiring made, absent outputs named in a warning, disconnect / connect of input and outputs and device switches off the UI thread, control errors become a notice, a language change rebuilds only that lane (session kept) and a failing one keeps the old lane |
| WD8 | Apply and restart | unit | Invalid values return field keys at once; no change is a notice; `hide_text` is live; a restart setting stops, releases, builds, starts and swaps (new view-model, new generation, stop before build); run-time device and language choices and deliberate disconnects survive; a failed restart falls back to standby with the new settings; missing models refuse the apply before anything stops; standby only stores; nothing is accepted while busy |
| WD9 | Worker, shutdown, report | unit | The tick never waits for the worker; busy label; `JobRunner` order, results, errors, close; shutdown stops and releases the pipeline, waits for a call in progress, is idempotent; errors described with their message; the placeholder device never reaches a pipeline; the report text for the clipboard honors hide text; the window can post a notice |
| WD10 | Wiring | unit | The real loaders are SAPI and `RateConverter` (a lane without a route gets the default voice and the shared rate), no banner, a lane's engine is closed with the build and after a failed one, missing models listed without loading, loaders get the right paths and settings, smart mode creates a link that `close` releases, a failing build releases it, whisper and translators are reused between builds and reloaded only when what they depend on changes, `with_config` passes new decode options, strict arms the guard after every load and otherwise releases it, glossary path, a built pipeline runs end to end on fakes (row with both translations, a 440 Hz test tone at 22.05 kHz arrives at the 48 kHz device as 440 Hz through the real `RateConverter`, no lane error), controller connect and restart through the real wiring. `test_ui_wiring.py` |
| WD11 | Real wiring on local models | model | `RealWiring` builds and starts on `models/` (tiny.en, fixed and smart mode) with fake streams, fake voices and fake speech. `test_ui_wiring_model.py` |
| WD12 | Tk window smoke | unit | On a fake pipeline: standby look in words, connect, status dots and process cards, a transcript row filled in place, hide text, copy to the clipboard, a device button, inline settings errors, a restart clears the transcript, close stops the pipeline; advanced fields hidden until asked or until an error sits in one; missing model shown inline; no device selected; the language box reaches the pipeline; a control error in the status bar; no Tk call from the worker thread. Skips when Tk cannot open a window. `test_ui_tk.py` |
| WD13 | Window and controller touch no disk or network | unit | The whole WD12 flow and the controller flows run under `no_disk_writes` and `--disable-socket`: 0 violations |
| WD14 | Tk-free logic | unit | `present`, `controller`, `settings_form`, `wiring`, `model` and `settings_schema` import no toolkit (static scan) |
| WD17 | Process layout | unit | Stage cards grouped into `ProcessLayout(common, lanes)`: two lanes after the common stages, one lane, no lanes, a missing translate or output card leaves a gap, lane order follows the snapshot, `keys()` lists every card key in signal order. `test_ui_present.py` |
| WD18 | Process titles and fork | unit | Card titles say what the stage does and the language (`Microphone`, `Speech recognition (English)`, `Translate to German (de)`, `Speak / output de`; unknown code falls back to the code); `fork_geometry` gives one stem, one bar and one branch per lane, a straight line for one lane, nothing for none |
| WD19 | Fork in the window | unit | Tk smoke: with two targets both lane rows are children of the one "In parallel" group and nothing else is, each row has exactly one arrow (translate to speak), no arrow links lanes, the connector has one stem, one bar and two branches, every card keeps its health word, a new tick keeps the same widgets; one target gives one row and a bar-less connector. `test_ui_tk.py` |
| WD15 | Voices in `problems` and lane building | unit | All voices installed: no problem; one message per target without a voice (`voice missing: ...`, installed voices and the fix named, no `; ` inside); a voice name that matches nothing; voice and model problems together; the check loads no model, makes no engine, opens no audio; the backend is made, used and closed on one helper thread that is not the caller's; answers are cached per target and voice name (a rate change asks nothing) and a COM failure is a problem that is asked again; the check is not blocked by a model load in progress; `build` refuses a missing voice and releases the engines and the turn link it made; a lane switched to a language without a voice raises before the old lane is touched, one with a voice gets an engine that the build also closes; a language change drops the voice name and keeps rate and volume (`Pipeline.set_output_target` in `test_app.py`, standby in `test_ui_controller.py`). `test_ui_wiring.py` |
| WD16 | The window with a missing voice | unit | Through the controller on the real wiring with fake voices: auto-connect and connect are refused with `Cannot start: voice missing: ...`, nothing is loaded, other settings are stored, devices are listed, no banner; a language change in standby re-checks (the old language's voice name does not follow) and recovers; a running session whose lane switch is refused stays up with an error notice and the same generation. In Tk: a missing voice shows inline in the problems banner, not in the speech banner, Connect is disabled and settings can still be applied; a wiring notice still shows the speech banner. `test_ui_wiring.py`, `test_ui_tk.py`, `test_ui_controller.py` |
| WD17 | Real SAPI through the wiring | model | The real voice check (English) answers "no problem" and runs off the calling thread; a language without a voice is reported by the real check; the real check plus a real sentence under the `no_disk_writes` fixture: 0 violations. Memory streams only, no audio device. `test_tts_sapi_model.py` |
| H1 | Live microphone | hw | V380 format detected; utterances chunked, transcribed and translated |
| H2 | Pull the HS100W dongle | hw | Dongle out and in -> output B restored <= 3 s; input and output A uninterrupted; no crash |
| H3 | Two physical outputs | hw | English on Realtek, German on HS100W at the same time; without HS100W the second language goes to output C (HDMI Q2713S) |
| S1 | Soak | soak | 60 min looped speech via VB-Cable, N-1 cores busy: 0 overflows; max callback < 5 ms; p99 finalization latency <= 50 ms; RSS growth < 20 MB/h |
| S2 | Soak on real outputs | soak | 60 min with Realtek + HS100W: 0 underruns during speech on both; HS100W buffer fill does not grow (clock drift) |
| A1 | Disk at OS level | soak | Process Monitor during S1: 0 `WriteFile` / write `CreateFile` events from the demo process |
| A2 | Network at OS level | soak | Airplane Mode; `Get-NetTCPConnection` / `Get-NetUDPEndpoint -OwningProcess` empty during the run |

Test data: synthetic signals generated in memory; 10-15 short LibriSpeech
utterances (~1 MB) in `tests/data/` with attribution and reference texts;
20 reference sentences for T4. Tests may read files; only writes are banned.

## 10. Repository layout

```text
pyproject.toml  LICENSE  README.md  CLAUDE.md
src/pipeline/
  __main__.py     CLI: gui, list-devices (run and soak: later steps)
  config.py       parameters + validation
  devices.py      WASAPI lookup by name, PortAudio re-init
  capture.py      input stream, ring buffer, processing chain
  turn_features.py  log-mel features for the Smart Turn model (NumPy)
  turn.py         Smart Turn classifier and the links to the VAD chain
  supervisor.py   watchdog and reconnect shared by capture and outputs
  output.py       OutputRing, OutputSession
  pcm.py          int24 -> float32, to_mono, ChannelMixer, float_to_int16 (dither)
  resample.py     StreamingResampler
  vad.py          SileroVad, Segmenter, Chunk
  bus.py          ChunkBus / TextBus, Subscription
  transcribe.py   faster-whisper consumer, Utterance
  glossary.py     glossary loading, hotwords, term protection
  lanes.py        LanguageLane, Translator, TtsEngine, output
  app.py          Pipeline: wires everything, start/stop, runtime device control
  tts_sapi.py     SAPI TtsEngine via comtypes, memory stream
  metrics.py      in-memory counters and histograms
  dashboard.py    rich.Live and --no-ui
  guards.py       audit hook for --strict
  ui/model.py             view-model: Snapshot, Health, transcript rows, report text (no Tk)
  ui/settings_schema.py   every setting as data, build_settings, needs_restart (no Tk)
  ui/settings_form.py     settings tab state: texts, parsing, hints, error placement (no Tk)
  ui/present.py           strings, options and cell texts the window shows (no Tk)
  ui/controller.py        WindowController, JobRunner worker, standby / active sessions (no Tk)
  ui/wiring.py            composition root: real loaders and caches, SAPI speech and voice check (no Tk)
  ui/app_tk.py            MainWindow, run_gui (Tk)
  ui/transcript_view.py   transcript table (Tk)
  ui/settings_view.py     settings notebook (Tk)
  ui/tk_widgets.py        colours, status dot, scrolling page (Tk)
tests/
  conftest.py     no_disk_writes fixture, fake streams
  fakes.py        FakeInputStream, FakeOutputStream, FakeDeviceList
  ui_fakes.py     ControlPipeline and FakeWiring for the window controller tests
  test_*.py       one file per module; markers unit/model/virtual/hw/soak
  data/           LibriSpeech clips + ATTRIBUTION.md, reference sentences
scripts/
  fetch_models.py      the only code that uses the network
  check_no_cyrillic.py English-only rule
  procmon_check.ps1    A1
  stress.py            CPU load for soak
.github/workflows/ci.yml   unit + model + licenses + lint + English check
```

## 11. Work plan (~26 h). Each step: tests first, done when green.

Steps 1-8 are Milestone 1; 9-12 preview Milestone 2. The optional low-latency
extension adds steps 1b, 2b, 9b, 10b, 11b and 12b (about 12.5 h); see
[SPEC-LOW-LATENCY.md](SPEC-LOW-LATENCY.md) section 11.

1. Scaffold, CI, ruff, mypy, `pcm`, `resample` — U1-U6, U27, Q1-Q11 (2.5 h).
   **Done 2026-11-08**: 110 unit tests green, ruff and mypy clean, 100%
   coverage of `pcm.py` and `resample.py`.
2. `Segmenter`, Silero wrapper, model fetch script — U7-U10, M1 (2.5 h).
   **Done 2026-11-08**: 181 tests green (unit + model), ruff and mypy clean,
   `vad.py` coverage 99%.
3. `ChunkBus` — U11-U14 (1 h).
   **Done 2026-11-08**: 29 bus tests, 210 tests in total green, ruff and mypy clean,
   `bus.py` coverage 99%.
4. `capture`, fakes, reconnect — F1-F5 (3 h).
   **Done 2026-11-08**: 269 unit tests green (+3 model, +1 hw not run), ruff and
   mypy clean; coverage `capture.py` 97%, `devices.py` 97%.
5. `metrics`, `dashboard`, `guards`, license check — G1-G4 (2 h).
   **Done 2026-11-08**: 374 unit tests green, ruff, mypy, licenses and English check clean.
6. Output with reconnect — F6 (1.5 h).
   **Done 2026-11-08**: 422 unit + model tests green, ruff, mypy, licenses and English
   check clean. The capture reconnect logic moved into the shared `Supervisor`
   (existing capture tests unchanged and green).
2b. Smart Turn (see SPEC-LOW-LATENCY.md section 11). **Done 2026-11-08**; the model's
   measured speed and accuracy on this machine are far below the vendor figures, see
   its "Measured results".
7. VB-Cable tests — V1-V3 (1 h).
   **Done 2026-11-08 (V1, V2)**: both pass on the real cable (`tests/test_virtual.py`, shared
   speech helpers in `tests/sapi_speech.py`). This machine has one cable pair, so the tests
   take device names from `PIPELINE_CABLE_IN` / `PIPELINE_CABLE_OUT` and run V1 and V2 one
   after another. **V3 is written but not yet run**: it needs an elevated shell
   (`pnputil /restart-device`) and skips otherwise. T7 needs a second cable (A+B).
8. Milestone 1 soak — S1, A1, A2 (1 h + 1 h run)
   **Run 2026-11-08, S1 NOT passed.** `scripts/soak.py` (looped SAPI speech through the VB-Cable,
   7 busy worker processes from `scripts/stress.py` at below-normal priority, `StrictGuard` armed,
   silent until the report) and `scripts/netcheck.ps1` (A2) are written; the verdict logic has 33
   unit tests in `tests/test_soak.py`. **60 min:** 0 input overflows, longest callback 3.88 ms,
   RSS growth 0 MB/h, 698 chunks = the offline count, 0 reconnects, **A1: 0 write operations and
   0 bytes** (Windows I/O counters of the process, not Process Monitor, which is not installed here),
   **A2: 59 samples, 0 sockets**. **Failed:** 452 640 frames (about 9.4 s) lost from the capture
   ring, one output underrun, p99 finalization latency 96 ms (max 1726 ms). A processing thread
   that is not scheduled for over 2 s overflows the 2 s ring; the cause is not known (the run was
   silent and other programs, including the parallel GUI work, shared the machine). **A 10 min
   rerun with a per-minute timeline did not reproduce it:** 0 ring losses, 0 underruns, longest
   callback 2.26 ms, but an idle thread of the process woke up to 278 ms late under the load.
   **Finding, independent of the stall:** p99 finalization is 70-96 ms with real speech even on an
   idle machine (30-48 ms for most chunks, 68-90 ms for those with a fading word end). The silence
   clock starts at the first frame below `exit_threshold` while the metric counts from the last
   frame above `speech_threshold`, and the frames between the two thresholds (1-2 frames of 32 ms)
   are not silence; F5 uses a synthetic signal without that zone and passes. Either the criterion
   is measured from the start of the silence clock, or the clock starts at the last speech frame
   (shorter latency, but it changes segmentation and needs M1 and the endpointing evals again).
   Open until decided; the 60 min run has to be repeated with `--progress` to catch the stall.
9. `transcribe`, glossary — M2, T2, T3 (2 h)
   **Done 2026-11-08**: 692 unit tests green (+4 model tests for M2 / T3), ruff, mypy, licenses
   and English check clean; `transcribe.py` 93%, `glossary.py` 96% covered. Dependencies added:
   `faster-whisper` 1.2.1 and `ctranslate2` 4.8.5, without PyAV (see the transcribe section).
   **Whisper takes about 1.7 s per utterance here (`base.en`, 4 threads), roughly three times
   the 500 ms the budget assumed**; see SPEC-LOW-LATENCY.md section 12.4.
   `config.py` (parameters and validation, section 10) is in place: `parse_settings(argv)` turns the
   `run` flags into one `Settings` object (capture, segmenter, transcribe, one `LaneRoute` per target,
   `--radio-offset`, `--strict`, `--hide-text`, `--no-ui`) and `check_files()` lists every missing
   model or glossary at once. Flags of modules that do not exist yet (`--speculative`, `--sentences-per-job`, ...) are added with those modules. `__main__.py` has
   `gui` and `list-devices`; `run` and `soak` are not written yet. `--turn-mode` defaults to `fixed`
   (see D2).
10. `TextBus`, `LanguageLane`, `Translator` — T1, T4 (3 h)
   **10a done 2026-11-08** (`lanes.py`, `TextBus` in `transcribe.py`, 25 + 57 unit tests): T1 passes
   with a translator blocked on one lane (the other lane's latency is unchanged, the bus drops the
   blocked lane's oldest and counts it).
   **10b done 2026-11-08** (`translate.py`, 23 unit + 8 model tests): `CtranslateTranslator` runs
   OPUS-MT on CTranslate2 (int8 on the CPU) with SentencePiece; text is cut into sentences, the
   sentences go through the model as one batch (beam 2) and are joined again. Models:
   `michaelfeil/ct2fast-opus-mt-en-de` (**CC-BY-4.0**, attribution in THIRD_PARTY_NOTICES.md) and
   `-en-fr` (Apache-2.0), both CTranslate2 float16 conversions of Helsinki-NLP, pinned by revision
   and sha256 in `fetch_models.py`; new runtime dependency `sentencepiece` 0.2.1 (Apache-2.0, no
   license field in its metadata, so `check_licenses.py` decides it by name). `config.py` got
   `--mt-threads` / `--mt-beam` and `check_files` checks each target's folder. **T4 measured:**
   chrF 82.8 for en->de and en->fr on the 20 hand-written reference sentences
   (`tests/translation_reference.py`; the test fails below baseline - 3); a sentence takes
   **136-189 ms** median with 4 threads (best of three passes, shared CPU) against the **150 ms**
   target, so the target holds for de on a quiet CPU and is missed for fr; the test guards at
   300 ms. Glossary placeholders (`XQ<n>Z`) survived the real translator in 14 of 14 cases
   (7 per language). The SAPI `TtsEngine` is step 11. The lane needs a resampler
   from the TTS rate to the device rate: `StreamingResampler` only supports inputs of 16 / 32 /
   44.1 / 48 kHz and a 16 kHz output, so step 11 must extend it (without it a mismatching rate is a
   counted `LaneError`, never noise).
11. SAPI `TtsEngine`, sentence-by-sentence synthesis — T5 (2 h)
   **Done 2026-11-08** (`tts_sapi.py`, `RateConverter` in `resample.py`, sentence loop in
   `lanes.py`; 37 + 32 + 13 + 10 unit and 19 model tests). **T5 measured** (Ryzen 5 2400G, Windows 10
   voices Microsoft David / Hedda / Hortense Desktop): a sentence is ready in a median of
   **23-40 ms** against the 300 ms target; the German and French voices needed the Windows
   language packs (no pack: the tests skip with the reason). New runtime dependency `comtypes`
   1.4.17 (MIT). Flags `--tts-rate` (-10..10, every lane) and `--tts-voice de=Hedda,fr=Hortense`
   fill `LaneRoute.tts`. The operator window (`ui/wiring.py`) passes `sapi_tts_factory` and
   `RateConverter` and reports missing voices through `problems` (section 7, WD10, WD15-WD17);
   audible playback through the window was not verified by hand. The G1 fixture watches Python-level opens only; that SAPI
   writes nothing is by construction (`SpMemoryStream`).
12. Latency and hardware — T6-T8, H1-H3, S2 (2 h + 1 h run)
13. Video and README (1.5 h)

GUI plan (operator window, outside the numbered steps): the orchestrator `app.py`, the view-model,
the settings schema and the Tk window with the `gui` command are implemented (section 7, WD1-WD14).
The window speaks with the SAPI voices through the same engine the CLI will use (section 7).

## 12. Video script (~2 min)

1. 0:00 Airplane Mode on. Start with `--strict --source en --targets de,fr`;
   dashboard shows V380, Realtek (DE), HS100W (FR).
2. 0:15 A few English sentences, one with a glossary term: transcript and both
   translations on screen.
3. 0:35 English from the speakers, German in the headset (held near the
   camera mic). Per-stage and per-lane latency on the dashboard, HS100W
   including radio offset.
4. 0:55 Pull the HS100W dongle: DE lane "device lost", English keeps going.
   Plug back: "reconnected", German resumes.
5. 1:15 Counters: overflow 0, underrun 0, reconnects 1.
6. 1:25 `pytest` all green, disk and network guard tests visible.
7. 1:40 Process Monitor report (0 writes), soak report and license table.

## 13. Open questions

- Language pairs: en -> de, fr, OPUS-MT (done, step 10b). **The en-de model is CC-BY-4.0**, not
  one of the licenses named in the hard constraints; it allows commercial use and needs only
  attribution, which THIRD_PARTY_NOTICES.md gives. Replace it if CC-BY is not acceptable.
  **Windows SAPI
  voices: this machine has only en-US and ru-RU installed**, so German and French
  voices (for example Hedda and Hortense) must be added as Windows language
  features before step 10.
- Whisper size: both `base.en` (default) and `tiny.en` are selectable; `base.en` already needs
  about 1.7 s per utterance on this CPU. Which one is the default for the demo is open. Measured on 80 LibriSpeech utterances
  (SPEC-LOW-LATENCY.md section 12.4): `tiny.en` WER 7.2% and 1.2 s median decode, `base.en`
  5.4% and 2.2 s. `small.en` takes about 7 s and is out. Decide by M5 and T7.
- ~~Does `faster-whisper` import `av` at load time?~~ Yes; PyAV is replaced by a stub (step 9).
- Discuss Kokoro or a Piper exception with the client for Milestone 2.
- Public repo or private with link access.
- Ultra-low latency: target number and how it is measured, and whether the
  client accepts ending an utterance before 450 ms of silence. See the open
  questions in [SPEC-LOW-LATENCY.md](SPEC-LOW-LATENCY.md).
