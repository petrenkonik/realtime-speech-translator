# Specification: Ultra-Low-Latency Extension

Companion to [SPEC.md](SPEC.md) (the base spec). Goal: **the lowest latency
the CPU allows at every stage**, measured, not assumed. Where this file and
`SPEC.md` disagree, this file wins for the flags it defines.
`--turn-mode fixed` gives back the pure silence-threshold behavior of the base
spec. Test IDs continue the numbering of `SPEC.md` section 9.

### Decisions (from the user, with the author's reading)

- **D1 - Goal.** Minimize latency at all stages, not only at the VAD. Section 12
  lists every lever per stage with expected gain, cost and test.
- **D2 - Client text.** The requirement reads: "Silence threshold must be
  configurable between 450 ms and 600 ms to trigger chunk finalization." The
  user reads 450-600 ms as the intended silence setting. Adopted reading (the
  author's interpretation): the silence threshold stays configurable in
  450-600 ms (U8 unchanged) and remains the guaranteed trigger; Smart Turn is
  an **additional** trigger that can finalize earlier. Hence `--turn-mode`
  defaults to `smart`. It is disclosed in the README and on the demo
  dashboard, and `--turn-mode fixed` restores the literal behavior. If the
  client objects, flip the default; nothing else changes.
- **D3 - Languages.** Russian is out of scope for this extension. The source
  language is a parameter (`--source`); no Russian test audio, abbreviation
  lists or accuracy claims are part of this document. Test audio is English
  (LibriSpeech) unless the user names another source language.

## 1. Why this document exists

The base pipeline waits for a fixed silence (450-600 ms, client requirement),
runs whisper once on the finished chunk, then translates and speaks the whole
text. The review of `SPEC.md` found these latency costs:

| # | Cost | Where |
| --- | --- | --- |
| L1 | The silence wait is mandatory and dominates the budget | VAD |
| L2 | A fixed silence cuts sentences at commas or waits needlessly after a clear sentence end | VAD |
| L3 | Long speech without 450 ms pauses is cut only at the 15 s cap, so the first words are translated up to 15 s late | Segmenter |
| L4 | Translation and TTS start only after the whole utterance is transcribed and covers all its sentences | Lanes |
| L5 | The metric "end of speech to first audio" hides utterance length | Metrics |

This extension addresses L1-L4 with two mechanisms and fixes L5 with a second
metric. It does **not** add partial ASR hypotheses (still out of scope in
`SPEC.md`): whisper runs once per finished chunk.

1. **Smart Turn** (sections 4-6): an audio-only model decides whether the
   speaker has finished, so a chunk can end after ~200 ms of silence when the
   phrase is complete, and long speech is cut at sentence ends (soft cuts)
   instead of at 15 s.
2. **Sentence pipelining** (section 8): the final text is split into
   sentences; sentence k+1 is translated and synthesized while sentence k plays.

## 2. What was learned from Sokuji

Sokuji (https://github.com/kizuna-ai-lab/sokuji, shallow clone read on
2026-11-07, commit as of that date) is an open-source live-translation app.
Facts below come from its source, not its marketing.

**License: AGPL-3.0. No Sokuji code is copied or translated.** Only ideas,
algorithm shapes and public numbers are reused; the implementation here is
independent. Reviewers should not paste Sokuji code into this repository.

What its local (offline) pipeline does:

- VAD cuts utterances -> ASR -> translation -> TTS, one job queue per session,
  **serial** (`src/providers/localInference/adapter.ts`): a job is translated
  and spoken before the next starts; nothing is dropped, the queue just grows.
- Per-job failures are isolated; punctuation and translation helpers have hard
  time budgets (3 s and 1 s) and fall back to the raw text.
- **Smart Turn** gates the VAD (`src/lib/turn/`,
  `workers/_shared/turn-gate.ts`): after a short silence it sends the last 8 s
  to the model, applies the answer at the next frame, and ends the segment if
  the probability exceeds a threshold. Requests carry an id; an answer for an
  old silence run is ignored. If speech resumes inside the window the plain VAD
  would still have waited, the "misfire" segment is rescued instead of dropped.
- **Sentence jobs inside an utterance** (`sentenceCut.ts`, `SentenceStream`):
  a translation job every N sentences (1-5, default 3) as text is sealed.
- **Silence timer deferral** (`silenceDeferral.ts`): a pause timer does not
  close a bubble while the unsealed tail is mid-sentence and still growing.
- Streaming ASR (Voxtral Realtime) needs GPU; partials trail speech by ~480 ms.
  Not usable here (CPU only).

Sokuji's own numbers: translation (HY-MT, ~30 chars) 420 ms warm / 980 ms first
call; Supertonic TTS 0.5-2 s on WebGPU; cloud Soniox ~645 ms first word, ~1.1 s
final. Its default pause is **1.4 s**, which shows how conservative the
sentence-level approach is when a model does not decide the cut.

### Parameter comparison

| Parameter | Sokuji default (range) | This project | Decision |
| --- | --- | --- | --- |
| speech threshold | 0.3 | 0.5 | Keep 0.5. 0.3 is tuned for clean headset mics; the V380 camera mic is noisy |
| exit threshold | derived | 0.35 | Keep |
| min speech | 400 ms | 250 ms | Keep 250 ms, add rescue (section 5.4) |
| min silence (pause) | 1400 ms | 450-600 ms | Client requirement; Smart Turn shortens it |
| max speech | 30 s | 15 s | Keep 15 s (whisper time); add soft cuts |
| pre-roll | 800 ms | 100 ms | **Proposal: 300 ms** (range 100-800). Needs U10 changed; see section 11 |
| Smart Turn check after | 300 ms (100-500) | 200 ms (100-500) | Lower default because the fallback is 450 ms, not 1400 ms |
| Smart Turn threshold | 0.5 (0.3-0.9) | 0.5 (0.3-0.9) | Take as is |
| sentences per job | 3 (1-5) | 1 (1-5) | Lower: first audio after one sentence |
| TTS speed | 1.0 | 1.0 | Static only. Sokuji source has no adaptive speed (README claim not confirmed in code) |
| job queue | serial, never drops | pipelined, drops oldest whole utterance | Latency over completeness (base spec) |

## 3. Terminology and units

- Time is the stream sample counter (16 kHz). One VAD frame = 512 samples =
  32 ms. "ms" parameters are converted with `frames = ceil(ms / 32)`, so
  200 ms = 7 frames = 224 ms in practice.
- `silence run`: consecutive frames with Silero probability < exit threshold
  (0.35) while a chunk is open. A frame with probability in [0.35, 0.5) is
  neither speech nor silence and **resets** the run.
- `check`: one Smart Turn inference request on the audio of the open chunk.

## 4. Smart Turn model

| Item | Value |
| --- | --- |
| Model | **default: `smart-turn-v3.2-gpu.onnx`, fp32 (unquantized; the vendor's name, it runs on any CPU), 32 411 198 bytes.** Alternative: `smart-turn-v3.2-cpu.onnx`, int8, 8 679 182 bytes (see "Which file" below) |
| Source | Hugging Face `pipecat-ai/smart-turn-v3`, revision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93` |
| sha256 | fp32: `ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be`; int8: `2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f` (both equal the HF `X-Linked-ETag` at the pinned revision; the fetch script re-checks them) |
| License | **BSD-2-Clause** (model card metadata, verified 2026-11-07); allowed by the project rule. Copyright notice goes to `THIRD_PARTY_NOTICES.md` |
| Languages | 23 (full list in the model card), including English and German |
| Published accuracy (v3.2 cpu) | overall 92.6%; en 94.3%; de 96.4%. Check the card for the actual source language |
| Speed | vendor: ~12 ms on fast CPUs, ~65 ms on a cloud vCPU. **Measured here (2026-11-08, Ryzen 5 2400G, browsers running): 1 thread min 320 / median 350-470 / p95 500-730 ms; 2 threads median 210-280 ms; 4 threads median 150-200 ms.** Features cost 14 ms. Re-measured on a quiet machine: fp32 1 thread median 264 ms (min 219), int8 median 350 ms (min 320), int8 with `x64quantprecision=1` 395 ms. Roughly 5-20x slower than the vendor figure; see "Which file" and "Measured results" below |
| Input | `input_features` float32 `[1, 80, 800]`: log-mel of exactly 8 s at 16 kHz |
| Output | `logits[0]` = probability the turn is complete, in [0, 1]. Verify range at load |

### Which file, and why the numbers differ from the vendor's (follow-up to step 2b)

Investigated 2026-11-08 after the first measurements looked far worse than published.

- **Speed is a hardware limit, not a bug.** The vendor's 12.6 ms is a `c7a.2xlarge`
  (8 vCPU, AVX-512, onnxruntime's default thread count). One core of the Ryzen 5 2400G
  (Zen1, AVX2 only) does about 27 GFLOPS on a large fp32 matmul; the model needs roughly
  7 GFLOP (estimate), so ~260 ms is the floor for one thread. In the int8 file
  `QLinearMatMul` takes 48% and `QLinearSoftmax` 38% of the time; without VNNI int8 is
  not faster than fp32.
- **The int8 file answers differently on this CPU.** On 73 LibriSpeech clips the int8
  probabilities correlate 0.71 with fp32 (max difference 0.87; "complete" 68% vs 82%),
  and clear sentence ends got p = 0.01. With `session.x64quantprecision=1` the
  correlation is 0.88 and "complete" 81%, which is consistent with int8 saturation on
  AVX2 (the mechanism itself was not proved). The vendor measured on AVX-512 machines.
  On the synthesized M3 probes the files agree on 97% of decisions, so the divergence
  shows on real speech only; full comparison in `docs/SMART-TURN-MODEL-COMPARISON.md`.
- **Hence the default is the fp32 file**: the vendor's accuracy reference, and faster
  than int8 at one thread here (264 vs 350 ms).
- **The first test was unfair to the check time.** The LibriSpeech clips end with a
  natural silence of 535 ms median, longer than the 480 ms fixed rule, so "as recorded"
  does not describe a check at 224 ms. With the same speech trimmed and a synthetic
  tail (fp32): +0 ms 19% complete, +100 ms 36%, +224 ms 40%, +320 ms 59%. The model
  partly measures the silence length and is reliable from about 300 ms, which leaves a
  window of ~160 ms before the fixed rule fires, shorter than the model's own latency.
  The tail was built from each clip's noise floor, so a real microphone may differ.
- **With labels taken from the text the model discriminates well** (M3, fp32, 16 probes each,
  SAPI, 224 ms of silence after the speech): finished sentences 94% "complete"
  (100% at 320 ms, 75% at 0 ms); sentences that stop on a connective or filler ("... and",
  "... because", "..., um,") 12% "complete" (spec: <= 15%). So the earlier "weak
  discrimination" finding came from the labels, the int8 file and the tail, not from the
  model. SAPI is synthetic and the vendor trains largely on TTS, so this is the best case.
- **False ends at real pauses in read speech stay at 44-48% on fp32**, but the labels are weak (read
  audiobooks, clips cut at pauses, no punctuation) and the vendor's "incomplete" class is
  fillers, connectives and thinking prosody, not any pause. This does not show how the
  model behaves on conversation; T10 on a real microphone is the test that matters.
- The vendor's 94% and 12 ms come from its v3 blog post, not re-run for v3.2, and the
  blog does not say how the test audio ends (trailing silence or not).

### Measured results (step 2b, 2026-11-08; int8 file, before the follow-up above)

Model input verified, quality measured with `OnnxTurnClassifier` on this machine.

| Check | Result | Spec / hope |
| --- | --- | --- |
| Features vs vendor extractor | max diff 3e-6 | < 1e-3 |
| Output range | within [0, 1] on all clips | [0, 1] |
| Sentence-final, SAPI speech, 224 ms pause | 88% say "complete" (n=16) | >= 80% |
| Sentence-final, LibriSpeech clips as recorded (natural tail), n=73 | 68% (median p 0.85) | published: 94% (en) |
| Sentence-final, LibriSpeech cut to the last loud frame + 224 ms synthetic noise tail | 30-38% | - (shows the tail matters) |
| Mid-sentence, SAPI cuts at word gaps + 224 ms | 81% say "complete" | <= 15% false ends |
| Mid-utterance word gaps of read speech (LibriSpeech, n=275) + 224 ms | 37% | <= 15% |
| Mid-utterance real pauses >= 224 ms (n=90) / >= 320 ms (n=63) | 50% / 62% | <= 15% |
| Model time, 1 thread | 320 ms best, 350-730 ms typical | p95 <= 100 ms |

What this means:

- **Latency.** A check at 224 ms of silence answered ~350-470 ms later arrives *after*
  the fixed 480 ms fallback, so with one thread smart mode gains nothing, and every
  check burns about a third of a core-second. With 2 threads the answer lands at
  ~430-500 ms (no gain), with 4 threads at ~370-430 ms (gain 50-110 ms, but those cores
  are needed by whisper). The estimates in sections 5.3, 7 and 12.3 assumed 12-65 ms
  and are **not valid on this CPU**. Only a much earlier first check (`turn_check_ms`
  100, 128 ms of silence) with 2-4 threads would give a clear gain, at the price of
  more premature cuts.
- **Quality.** *Superseded by "Which file" above:* this table was measured with the int8
  file and with unfair probes. On read speech the int8 file separated sentence ends from
  word pauses only weakly (68% vs 37-50%); the fp32 file with text-labelled probes does
  well (94% vs 12%). The 73-clip LibriSpeech sample (CC-BY-4.0) is kept outside the
  repository.
- **Safety.** The design absorbs a bad model: a wrong "complete" costs two shorter
  chunks (rescue keeps the audio, I1 holds), a missing answer costs nothing (fixed
  fallback). So smart mode cannot lose speech, but on this evidence it can split
  sentences without saving time. `SegmenterConfig.turn_mode` therefore defaults to
  `fixed` in the state machine; the CLI default (`smart`, D2) is applied in
  `config.py` and should be revisited after T10 on real microphone speech (question 10).

FPR = the model says "complete" while the speaker would continue (a premature
cut). FNR = it says "not complete" at a real end (falls back to the silence
timer, no harm beyond the old behavior).

Pre-processing (own implementation, NumPy only; no `transformers`/`torch`):

1. Take the last 8 s of the open chunk's audio; if shorter, **zero-pad at the
   front** so the audio sits at the end of the 128 000-sample vector.
2. **Normalize the whole 8 s vector to zero mean and unit variance**
   (`(x - mean) / sqrt(var + 1e-7)`, padding included). The vendor passes
   `do_normalize=True`; the first version of this spec left the step out, which
   would have fed the model a wrong input. Found by reading the vendor's
   `inference.py` at step 2b.
3. Whisper-style log-mel: periodic Hann window, n_fft 400, hop 160, 80 Slaney
   mel bins, `log10(max(x, 1e-10))`, clamp to `max - 8`, then `(x + 4) / 4`;
   drop the last STFT frame -> 800 frames (reflect padding, as the vendor's
   `transformers` extractor). `faster_whisper.feature_extractor`
   may be reused if it matches (it pads at the end, so pad first and bypass its
   padding); otherwise ~40 lines of NumPy. Parity with the vendor reference
   is verified by M4 before anything is built on it. **Result (step 2b):** max abs
difference to the vendor extractor 3e-6 on five clips (spec limit 1e-3); the
reference output is stored in `tests/data/smart_turn_reference.npz`, produced by
`scripts/make_turn_reference.py` in a throwaway environment with `transformers`.

Runtime: one `onnxruntime.InferenceSession`, `intra_op_num_threads=1` so it
does not take cores from whisper/CTranslate2, loaded and warmed at startup
from the local path (`models/smart-turn/`, `pipeline.turn.DEFAULT_TURN_MODEL`).
`scripts/fetch_models.py` downloads both files at the pinned revision and checks the
sha256 (the only network use). The session sets `session.x64quantprecision=1`: it makes
int8 matmuls on AVX2 CPUs without VNNI non-saturating and is ignored by the fp32 file.

## 5. Turn detection algorithm

### 5.1 Components

```text
capture -> resample -> SileroVad -> Segmenter -(TurnRequest)-> TurnClassifier thread
                          ^                                          |
                          +-------------(TurnAnswer, next frame)-----+
```

- `Segmenter` stays a **pure state machine**: input per frame is
  `(probability, frame_samples, optional TurnAnswer)`, output is a list of
  events (`TurnRequest`, `ChunkReady`). It never calls the model.
- `TurnClassifier` is a `typing.Protocol`: `predict(window) -> float`. The real
  one runs on its own thread with a 1-slot mailbox (newest request wins);
  tests inject a scripted fake with a sample-counted delay. A glue object
  (`TurnLink`) posts requests and hands the latest answer to the Segmenter once
  per frame, without blocking the VAD thread.
- The capture callback never touches any of this.

### 5.2 Per-frame rules (mode `smart`)

Let `utt_ms` = duration of the open chunk so far, `S` = `silence_frames`
(fixed fallback), `C` = `check_frames`, `P` = `soft_pause_frames`.

1. **Speech frame** (probability >= 0.5): `speech_frames += 1`; `silence_run = 0`;
   forget any open request.
2. **Ambiguous frame** (0.35 <= probability < 0.5): neutral. It does not start the silence clock and does not stop a running one. (Changed in step 2: resetting let noise at 0.4 hold a chunk open to the 15 s cap; see SPEC.md, risk C2.)
3. **Silent frame** (probability < 0.35): starts the clock if idle; `silence_run` is the number of frames since the clock started (ambiguous frames inside it count), then:
   1. Trigger frames `T = C` if `utt_ms < soft_cut_after_ms`, else `T = P`
      (soft mode: shorter pause, stricter threshold; section 5.5).
   2. If `silence_run == T` and speech so far >= `min_speech_ms`: emit
      `TurnRequest(id = ++last_id, window = last 8 s of chunk)`, remember `id`
      as the open request and the active threshold (`turn_threshold` or
      `soft_cut_threshold`). **Rechecks:** while `silence_run < S`, request again
      every `R` frames (`--turn-recheck-ms`, default 96 ms = 3 frames) if no
      request is in flight, at most 4 requests per silence run. A true end
      scores higher as the silence grows, so a phrase the first check rated
      0.4 is often caught at the second or third check, long before `S`.
   3. If `silence_run == S`: finalize with reason `silence` (the fixed-mode
      behavior, always available as the fallback).
4. **Answer** (applied at the frame after it arrives): if `answer.id` equals the
   open request and the Segmenter is still in the same silence run and
   `answer.probability >= active threshold`, finalize now with reason `turn`
   (or `soft`). Otherwise ignore it; a stale answer increments `turn_late`.
5. **Hard cap**: if `utt_ms >= max_utterance_ms` (15 s) finalize with reason
   `hard`, cutting at the quietest frame of the last `cut_search_ms` (1500 ms)
   so the cut falls in a breath rather than inside a word. No sample is lost or
   duplicated at the cut (U9 still holds).

Fixed mode is rules 1-3 with only step 3.3 and rule 5 active.

### 5.3 Worked timeline (smart mode, defaults, complete sentence)

```text
t = 0 ms       last speech frame
t = 224 ms     silence_run == C (7 frames): TurnRequest posted
t = ~240-290   model answers (12-65 ms, to be measured); p = 0.93 >= 0.5
t = ~270-320   answer applied at the next frame; chunk finalized (reason turn)
fixed mode:    chunk would be finalized at t = 480 ms (15 frames >= 450 ms)
saving:        ~160-210 ms, only when the model says "complete"
```

Mid-sentence pause (e.g. 300 ms at a comma): p = 0.12 -> ignored; speech
resumes, one chunk, no premature cut (this is where fixed 450 ms also works,
but a 300 ms pause at 450 ms would never be cut either way).

### 5.4 Premature cut handling (rescue, no lost audio)

When smart mode ends a chunk early and speech resumes **before `S` frames of
silence would have elapsed** (the fixed mode would have kept it in the same
chunk):

- The new speech opens a new chunk as usual.
- If that chunk turns out shorter than `min_speech_ms` (a misfire in normal
  rules), it is **kept** and emitted with reason `rescued` instead of being
  dropped. Rationale: fixed mode would have transcribed those samples.
- **Invariant I1**: every sample that fixed mode would put into some chunk is
  inside some chunk in smart mode (checked by a property test, U18).
- A premature cut costs two shorter translations instead of one; it does not
  lose audio. Merging two adjacent chunks before translation is not specified
  (open question 9).

### 5.5 Soft cuts (long speech, L3)

When `utt_ms >= soft_cut_after_ms` (default 6000), the check trigger drops from
`C` frames to `P` frames (default 128 ms = 4 frames) and the threshold rises to
`soft_cut_threshold` (default 0.7). The effect: in fast speech with pauses
shorter than `check_after`, the speaker's real sentence ends are still found,
but a breath mid-sentence is much less likely to cut because the bar is higher.
Before `soft_cut_after_ms`, pauses shorter than `check_after` are ignored as
usual. The 15 s cap remains the last resort. Soft cuts exist only in smart mode.

### 5.6 Optional hold (off by default)

`--turn-hold-ms H` (0 = off, range 0-1000). If at `silence_run == S` the last
answer was below threshold, finalization is postponed instead of happening: checks continue at the
rechecking cadence of rule 3.2 (unbounded count while holding) and the chunk is
finalized when a check passes, speech resumes (no cut), or
`silence_run >= S + H/32` (reason `silence`). It protects against cutting a long
thinking pause, at the cost of up to `H` ms extra wait in exactly those cases.
It lengthens the silence beyond the client's 450-600 ms, so it needs separate
sign-off and is not part of the first implementation (step 2b ships without it;
the hook point is the `S` branch of rule 3.3).

## 6. Parameters and validation

CLI flags map 1:1 to `config.py` fields. Every range violation is a config
error with an actionable message (like `--silence-ms` today).

| Flag | Default | Range | Source / note |
| --- | --- | --- | --- |
| `--turn-mode` | `smart` | `fixed`, `smart` | `fixed` = pure silence threshold (D2) |
| `--turn-recheck-ms` | 96 | 64-320 | Cadence of repeated checks in a silence run (rule 3.2) |
| `--speculative` | off | on/off | Section 12, W1: start whisper at the first check, discard if not an end |
| `--whisper-threads` | 4 | 1-8 | Set by `bench_stages.py` (M5); machine has 4 cores / 8 threads |
| `--tts-rate` | 0 | -2..4 | SAPI rate; positive = faster speech |
| `--tts-early-start` | on | on/off | Start playing from the first synthesized bytes (S2) |
| `--out-prebuffer-ms` | 20 | 0-100 | Audio buffered before the output stream starts consuming (O1) |
| `--exclusive-output` | off | `NAME=...` list | WASAPI exclusive mode for chosen wired outputs only (O2) |
| `--silence-ms` | 450 | 450-600 | Client requirement (U8); fallback in smart mode |
| `--turn-check-ms` | 200 | 100-500, step 50 | Sokuji 300; must satisfy `check_ms <= silence_ms - 200`, else clamped to that value (leaves room for model time) |
| `--turn-threshold` | 0.5 | 0.3-0.9, step 0.05 | Sokuji 0.5 |
| `--soft-cut-ms` | 6000 | 0 (off) or 3000-12000 | Utterance length after which soft cuts apply |
| `--soft-pause-ms` | 128 | 96-192 | Pause that triggers a soft check |
| `--soft-cut-threshold` | 0.7 | 0.5-0.95 | Stricter than `--turn-threshold` (config error if lower) |
| `--turn-hold-ms` | 0 | 0-1000 | Section 5.6, off |
| `--turn-threads` | 1 | 1-2 | onnxruntime intra-op threads |
| `--max-utterance-ms` | 15000 | 5000-20000 | Base spec 15 s; whisper window is 30 s |
| `--cut-search-ms` | 1500 | 500-3000 | Where the hard cap looks for the quietest frame |
| `--pre-roll-ms` | 100 | 100-800 | Base default stays 100 until the client accepts 300 (section 11) |
| `--sentences-per-job` | 1 | 1-5 | Section 8 |
| `--min-sentence-chars` | 12 | 4-60 | Section 8.2 |
| `--max-job-chars` | 300 | 100-600 | Section 8.2 |

Other validation: `--turn-mode smart` requires the model file (clear error
naming `scripts/fetch_models.py`); with `fixed` the model is not loaded.

## 7. Latency model

Estimates to be replaced by measurements (T7, T9, T10). Utterance of ~3 s,
`--silence-ms 450`. Totals are superseded by the all-stage budget in 12.3;
this section keeps the VAD-level comparison and the long-speech case.

| Component | Fixed mode | Smart mode (complete phrase) |
| --- | --- | --- |
| End of speech -> chunk finalized | ~480 ms | ~270-320 ms |
| Transcription | <= 500 | <= 500 |
| Translation (1 sentence) | <= 150 | <= 150 |
| TTS to first audio | <= 300 | <= 300 |
| Output buffer | <= 50 | <= 50 |
| **Total** | **<= 1500** | **<= ~1290-1340** |

Long speech (12 s, pauses of 150-250 ms between sentences):

| | Fixed, no pauses >= 450 ms | Smart + soft cuts + sentence jobs |
| --- | --- | --- |
| First translated audio after the speaker started | ~15 s cap + processing | ~6-8 s + processing (first soft cut) |
| Audio for the 2nd sentence | after the whole utterance is spoken | contiguous, synthesized while sentence 1 plays |

New metric (fixes L5): **lag behind speaker** = start of translated audio for a
chunk minus the stream time of that chunk's *first* word. Reported next to the
existing "end of speech to audio". For short phrases the two differ by the
phrase length; for long speech the second one shows what the listener feels.

## 8. Sentence pipelining

### 8.1 Flow

```text
Utterance(text) -> SentenceSplitter -> Job(utterance_id, index, last, text)
   -> lane: translator thread -> [queue 2] -> tts thread -> [queue 2] -> playback
```

- Each lane splits independently of the others (the splitter is pure and
  shared as a function, no shared state). Jobs of one utterance stay in order
  and contiguous; no interleaving with the next utterance.
- While job k plays, job k+1 is translated and synthesized. The two queues are
  bounded (2 jobs) so a slow stage back-pressures instead of growing memory.
- First audio of an utterance = translate(job 1) + TTS(job 1), not the whole
  text.
- Glossary placeholders (base spec) are applied **per job after the split**, so
  a placeholder never spans a job boundary.
- Failure of one job (translator/TTS error) is logged without text, counted,
  and the lane continues with the next job.

### 8.2 SentenceSplitter rules

Pure function `split(text, sentences_per_job, min_sentence_chars,
max_job_chars) -> list[str]`:

1. Sentence ends: `.` `!` `?` `...` and the Unicode ellipsis, followed by space
   or end of text, **unless**:
   - the dot sits between digits (`3.5`) or inside a known abbreviation
     (one list per supported source language in `sentences.py`; English
     ships first; entries with non-ASCII letters, if a language needs them,
     are written as `\uXXXX` escapes because the repo is English-only),
   - a lowercase letter follows (whisper often puts a dot before a lowercase
     continuation; Sokuji applies the same rule),
   - the previous token is a single capital letter (initials).
2. Fragments shorter than `min_sentence_chars` are merged into the next one
   (translation quality: OPUS-MT is poor on 1-2 word fragments); a short last
   fragment merges into the previous one.
3. `sentences_per_job` consecutive sentences form one job.
4. A sentence or job longer than `max_job_chars` with no terminal punctuation is
   cut at the last comma/semicolon/space before the limit (Sokuji seals "by
   length" the same way).
5. Text with no terminal punctuation at all is one job (cut by rule 4 if long).

### 8.3 Drop policy

The lane lag is counted in **utterances**, as in the base spec. When a lane is
more than 2 utterances behind, the oldest utterance **that has not started
playing** is dropped as a whole. An utterance with a job already playing is
never cut in the middle (a missing middle sentence changes the meaning).
Dropped utterances and jobs are counted per lane.

## 9. Metrics (RAM only)

Per run, shown on the dashboard and in the soak report: `turn_checks`,
`turn_early_ends`, `turn_soft_cuts`, `turn_hard_cuts`, `turn_rescued`,
`turn_late`, `turn_holds`, model time avg/p95/max, chunks by reason,
`lag_behind_speaker` per lane (median, p95), `jobs_per_utterance`,
`jobs_dropped`. Text is never part of any metric.

## 10. Tests

All unit tests use the sample-counted clock and a scripted `TurnClassifier`
fake; no model, no hardware.

| ID | Check | Level | Criterion |
| --- | --- | --- | --- |
| U15 | Early end | unit | Smart mode, silence >= 7 frames, fake p = 0.9 -> chunk ends at the frame after the answer, reason `turn`; fixed mode ends at 15 frames |
| U16 | Fallback | unit | Fake p = 0.1 everywhere -> same chunks as fixed mode for 450, 525, 600 (U7 holds) |
| U17 | Stale answer | unit | Answer for an old request id, or after speech resumed -> ignored, `turn_late` incremented |
| U18 | No lost audio | unit | Property test over random speech/pause scripts and random answers: every sample fixed mode keeps is inside some smart-mode chunk (I1); chunk order and non-overlap hold |
| U19 | Rescue | unit | Early end, then 150 ms of speech (< min speech) before `S` frames -> emitted as `rescued`, not dropped |
| U20 | Config validation | unit | Each range in section 6 rejects value just outside; `check_ms` clamped to `silence_ms - 200`; `soft_cut_threshold < turn_threshold` rejected |
| U21 | Soft cut | unit | 20 s speech with 130 ms pauses every 4 s, fake p = 0.9 -> first cut at the first pause after 6 s, all chunks <= 15 s; fake p = 0.5 (below 0.7) -> no soft cut, hard cap used |
| U22 | Hard cap placement | unit | 40 s continuous speech -> cuts at the quietest frame within the last 1.5 s, no lost or duplicated samples (extends U9) |
| U23 | Feature shape | unit | `turn_features(audio)` -> `(80, 800)` float32; 3 s input is front-padded (zeros first); 12 s input keeps the last 8 s |
| U24 | Sentence splitter | unit | Table-driven: decimals, abbreviations (escaped), initials, lowercase continuation, merge of short fragments, length cut, no-punctuation text, `sentences_per_job` 1 and 3 |
| U25 | Lane pipelining | unit | Fake translator/TTS with scripted costs on a fake clock: job 2 is translated while job 1 plays; first audio after job 1 only; order preserved; queues never exceed 2 |
| U26 | Whole-utterance drops | unit | Lane 3 utterances behind -> oldest unstarted utterance dropped as a whole; a playing utterance is never cut |
| M3 | Smart Turn on labeled audio | model (**time test xfail on this CPU**) | Labels from the text, spoken by offline SAPI voices (LibriSpeech clips are cut at pauses, so their "sentence-final" label is a guess): after the first-check silence (224 ms) finished sentences give early end >= 80% and sentences stopping on a connective or filler give false end <= 15%; more silence never lowers the early-end rate; p95 model time (1 thread) fits the window between the first check and the fixed rule (256 ms with the defaults), the question the vendor's 12 ms figure stands in for; the model loads with the network disabled |
| M4 | Feature parity | unit + model | Log-mel of 5 fixed clips matches the vendor reference to max abs diff < 1e-3 (unit, `test_turn_features.py`, stored reference); model outputs within [0, 1] (model, `test_turn_model.py`) |
| G5 | Model provenance | unit | `fetch_models.py` pins revision `f766f81d...` and sha256; a corrupted file is rejected; `THIRD_PARTY_NOTICES.md` contains the BSD-2-Clause notice |
| T9 | Sustained speech lag | virtual | 30 s continuous source-language speech with natural 150-250 ms pauses into CABLE-A -> first translated audio <= 9 s after speech start (fixed mode: reported for comparison); no chunk > 15 s |
| T10 | Smart vs fixed | virtual | T7 utterance set run in both modes: median end-to-end improves by >= 120 ms in smart mode; premature-cut rate (chunk count above the labeled count) <= 10% |

G1 (no disk writes) and G2 (no network) cover all new code paths. Text
fixtures with non-ASCII letters, if any, are stored as `\uXXXX`-escaped data so
the English-only check passes.

Added by section 12:

| ID | Check | Level | Criterion |
| --- | --- | --- | --- |
| U28 | TTS leading-silence trim | unit | 120 ms of zeros + tone -> output starts <= 10 ms before the tone; samples after the cut are bit-exact; all-silence input does not crash and yields <= 10 ms |
| U29 | Rechecks | unit | Scripted answers 0.2 at C, 0.4 at C+3, 0.8 at C+6 frames -> chunk ends after the third answer; at most 4 requests per silence run; never 2 in flight |
| U30 | Speculative whisper | unit | Fake whisper starts at the first check; negative check -> result discarded, accepted transcriptions per utterance = 1; positive -> speculative result is used, whisper not run twice |
| U31 | Output prebuffer | unit | Fake output clock: first TTS block reaches the device callback within one block + `--out-prebuffer-ms` of the hand-off |
| U32 | Early TTS start | unit | Fake SAPI stream that grows over time: playback starts once >= 100 ms is available; if measured RTF >= 0.5 it waits for the whole sentence |
| M5 | Stage benchmark | model | `scripts/bench_stages.py`: whisper tiny/base/small at 2/4/6/8 threads, MT greedy vs beam 2, SAPI first-audio, Smart Turn, Silero; prints a table, stored in the README. Used to set defaults; asserts only the section 12.3 budget |

## 11. Repository layout and work plan

New files:

```text
src/pipeline/turn_features.py   log-mel for Smart Turn (NumPy)
src/pipeline/turn.py            TurnClassifier Protocol, OnnxTurnClassifier, TurnLink
src/pipeline/sentences.py       SentenceSplitter, Job
tests/test_turn_features.py  tests/test_segmenter_turn.py
tests/test_sentences.py      tests/test_lane_pipeline.py
src/pipeline/runtime_tuning.py  timer resolution, GIL switch interval, priorities, GC freeze
scripts/bench_stages.py         M5 stage benchmark
```

Changed: `vad.py` (Segmenter: turn requests/answers, rechecks, soft cuts,
rescue, `Chunk.reason`), `config.py`, `lanes.py`, `transcribe.py` (decode
options, speculative start), `tts_sapi.py` (trim, early start), `resample.py`
(delay budget), `capture.py` (output prebuffer), `metrics.py`, `dashboard.py`,
`scripts/fetch_models.py`, `THIRD_PARTY_NOTICES.md`.

Work plan (each step test first), added to `SPEC.md` section 11:

| Step | Content | Tests | Hours |
| --- | --- | --- | --- |
| 1b (after 1) | `runtime_tuning.py` (the resampler delay test U27 is already done in step 1) | F5 | 0.3 |
| 2b (after 2) | `turn_features`, `OnnxTurnClassifier`, Segmenter extension with rechecks (no hold), fetch + sha256 | U15-U23, U29, M3, M4, G5 | 4 |

**Step 2b status (done 2026-11-08):** `turn_features.py`, `turn.py`, Segmenter extension
(requests, answers, rechecks, soft cuts, rescue), `VadStage` link, fetch + sha256,
`THIRD_PARTY_NOTICES.md`. U15-U23, U29, G5, M4 pass; M3 passes for sentence-final
speech and its timing criterion is `xfail` (measured, see section 4; the false-end
criterion passes since the follow-up that moved to the fp32 file and text-labelled probes).
Differences from the text above, all deliberate:

- `TurnRequest`, `TurnAnswer` and the `TurnLink` Protocol live in `vad.py` (the state
  machine's own input and output); `turn.py` holds the classifier and the two links
  (`ThreadedTurnLink` for real use, `SyncTurnLink` answering on a frame count for
  tests and offline runs). The Segmenter exposes requests through `take_request()`
  and takes the answer as an argument of `process()`.
- The hard-cap reason keeps its name `max_length` (not `hard`); new reasons are
  `turn`, `soft`, `rescued`. Counters on the Segmenter: `turn_checks`,
  `turn_early_ends`, `turn_soft_cuts`, `turn_hard_cuts`, `turn_rescued`, `turn_late`.
- Invariant I1 is checked on *speech frames* (probability >= 0.5 inside a chunk fixed
  mode would keep): the silent gap between two chunks that fixed mode would have glued
  together is not part of any smart-mode chunk, by design. Verified on 1500 random
  scripts with random answers and delays.
- `soft_cut_threshold` may not be lower than `turn_threshold`, so raising the normal
  threshold above 0.7 requires raising the soft one too.
- A soft or normal check needs the pause to outlast the check plus one frame of answer
  delay; a 4-frame pause with a 1-frame answer cannot cut (speech resumes first). U21
  uses 6-frame (192 ms) pauses for that reason.
- The `ThreadedTurnLink` keeps a one-slot mailbox and drops a prediction that was
  running when `reset()` is called; a failing model counts an error and gives no answer.
- Dashboard rows and the metrics of section 9 are added with the lanes (steps 9b/10b).
| 9b (after 9) | `sentences.py`; whisper decode options; speculative start | U24, U30 | 2.5 |
| 10b (after 10) | Lane pipeline, whole-utterance drops, new metrics | U25, U26 | 1.5 |
| 11b (after 11) | TTS leading-silence trim, early start, output prebuffer | U28, U31, U32 | 2 |
| 12b (after 12) | `bench_stages.py`; virtual measurements; tuning of thresholds and threads | M5, T9, T10 | 2 |

Total added: ~12.5 h (base plan ~26 h -> ~38.5 h). Pre-roll: if the client accepts
300 ms, change base default and U10 (`start - 300 ms +- 32 ms`) in the same
commit; until then it stays 100 ms.

## 12. Stage-by-stage latency minimization

Machine measured on 2026-11-07: AMD Ryzen 5 2400G, 4 cores / 8 threads, 31 GB
RAM, Windows "High performance" power plan. Four physical cores are the
tightest resource: every lever that burns CPU (speculation, extra threads,
beam search) competes with whisper. All gains below are **estimates** until
M5/T7/T10 measure them; each lever has a flag so it can be turned off.

### 12.1 Levers per stage

| ID | Stage | Lever | Expected gain | Cost / risk | Flag / test |
| --- | --- | --- | --- | --- | --- |
| X1 | all | No polling: every hand-off blocks on a queue or event; no `sleep` on the hot path | removes up to one poll interval per hop | none | code review, F5 |
| X2 | all | `timeBeginPeriod(1)` (1 ms timer) and `sys.setswitchinterval(0.001)` (GIL hand-off 5 ms -> 1 ms) | lower wake-up jitter, a few ms per hop | slightly more context switches | `runtime_tuning.py`; F5, S1 |
| X3 | all | Audio/VAD threads `ABOVE_NORMAL` priority, process `HIGH` via `psutil`; never `REALTIME` | stable callback/VAD timing under load | can starve the desktop if misused | S1 |
| X4 | all | `gc.freeze()` after warm-up and a high GC threshold during the run | no GC pause mid-utterance | memory bound must hold | S1 (RSS < 20 MB/h) |
| X5 | all | Warm-up of every model at startup with a dummy full-size input: Silero, Smart Turn, whisper (30 s window), each translator, each TTS voice | first utterance 0.5-2 s faster | longer startup | covered by T7 first-utterance check |
| C1 | capture | 10 ms blocks, shared mode, `latency='low'` (base) | baseline | - | F5 |
| C2 | capture | Optional exclusive-mode input | <= 10 ms | locks the device | not planned |
| R1 | resample | Filter delay budget <= 3 ms (a 60 dB / 1 kHz transition at 48 kHz needs ~174 taps = ~1.8 ms) | keeps resampling out of the budget | none | U27 |
| V1 | VAD | 32 ms frames are fixed by Silero; quantization costs ~16 ms on average | unavoidable | - | - |
| V2 | VAD | Smart Turn early end with rechecks (sections 5.2-5.3) | 160-260 ms vs 480 ms | ~5% premature cuts (published FPR) | U15, U29, T10 |
| V3 | VAD | Lower `--turn-check-ms` to 150 or 100 and tune by T10 | up to 100 ms more | more premature cuts; clamp `<= silence_ms - 200` | T10 grid {100, 150, 200} |
| V4 | VAD | Soft cuts for long speech (5.5) | first audio after ~6-8 s instead of 15 s | mid-sentence cut risk, mitigated by 0.7 threshold | U21, T9 |
| W1 | whisper | Speculative start: whisper begins at the first check; discarded if the check says "not finished" | 50-100 ms (model time + one frame) | wasted CPU on every pause >= `check_ms`; with 4 cores it can slow other lanes. Off by default; decide by T10 and S1 | `--speculative`; U30 |
| W2 | whisper | Decode only what is needed: `beam_size=1`, `temperature=0` only (no fallback re-decode), `without_timestamps=True`, `condition_on_previous_text=False`, no word timestamps, `no_speech_threshold`, `log_prob_threshold`, `compression_ratio_threshold` set to `None`, `max_new_tokens` = tokens/s x chunk seconds + 8, built-in VAD off | avoids fallback passes (each can double the time); ~10-20% fewer tokens | hallucination loop on noise (bounded by `max_new_tokens`). Parameter names verified against the installed `faster-whisper` at step 9b | M2 WER, M5 |
| W3 | whisper | Smallest model that meets WER <= 15%; int8; `--whisper-threads` from M5. faster-whisper pads every input to a 30 s window, so encoder time is roughly constant and does not shrink with short chunks | the largest single lever | accuracy | M2, M5 |
| W4 | whisper | Trim trailing silence of the chunk to ~100 ms before decoding | fewer decoder tokens on silence, fewer hallucinations | none | F1 |
| W5 | whisper | Incremental log-mel while the speaker talks, so finalization only runs the encoder (needs direct `ctranslate2.models.Whisper`) | ~20-40 ms | tokenizer/prompt code of our own | only if profiling shows features > 30 ms |
| W6 | whisper | Shorter encoder context than 30 s | possibly large | unverified whether CTranslate2 accepts it, and quality loss | research item, no commitment |
| M1 | MT | Greedy `beam_size=1`, `max_decoding_length` ~ 2 x source tokens + 10, `max_batch_size=1`, `return_scores=False`, int8, one sentence per job (section 8) | 60-120 ms per sentence | small quality loss vs beam | T4 (<= 150 ms, chrF), M5 |
| M2 | MT | One preloaded model per lane, `intra_threads=1`, `inter_threads=1`; lanes run in parallel | lanes do not wait for each other | uses 2 more threads | T1 |
| S1 | TTS | Trim leading silence of the synthesized audio to <= 10 ms (energy gate about -45 dBFS) | 0-150 ms (SAPI voices often start with silence; measure) | clipping a soft onset if the gate is too high | U28 |
| S2 | TTS | Early start: speak asynchronously into the memory stream and start playback when >= 100 ms is available; fall back to whole-sentence mode if measured RTF >= 0.5 | up to hundreds of ms on long sentences | COM stream handling; underrun risk | `--tts-early-start`; U32 |
| S3 | TTS | Voice object created and warmed at startup, one per lane, own COM thread | removes voice-init time from the first utterance | none | X5 |
| S4 | TTS | `--tts-rate` +1..+2 | shorter playback, smaller lag behind the speaker | less natural | listening test |
| O1 | output | 128-256 frame blocks (3-6 ms) and a 20 ms prebuffer instead of a 50 ms ring | ~30 ms | underruns if too small; counted | U31, S2 soak |
| O2 | output | Exclusive mode for wired outputs only (not the HS100W dongle) | 10-30 ms | device lock, format must match | `--exclusive-output` |
| O3 | output | HS100W radio latency | not reducible in software | - | T6 measures it; reported separately |

### 12.2 CPU plan (4 cores / 8 threads)

Stages are mostly sequential in time (whisper finishes before translation
starts), so the real contention is: whisper of utterance n+1 against
translation/TTS of utterance n, and speculation. Starting point, to be set by
M5: whisper `cpu_threads=4`; each MT lane 1 thread; Silero 1; Smart Turn 1;
SAPI synthesis on its own COM thread per lane. Beyond 4 threads the SMT
siblings add little for int8 matrix work, so more than 4 whisper threads is
expected to be a loss. The soak S1 runs with the real thread plan.

### 12.3 Latency budget (projection, replaces the table in section 7 for totals)

| Stage | Base spec | Defaults (smart, W2, S1) | Floor (tiny/base, `check` 100, early TTS) |
| --- | --- | --- | --- |
| Capture block + resample | included in 50 | ~10 | ~8 |
| VAD quantization | included in 50 | ~16 | ~16 |
| Wait for end (silence or Smart Turn) | 450-600 | 270-320 | 170-220 |
| Transcription | <= 500 | 300-500 | 200-300 |
| Translation | <= 150 | 60-150 | 50-80 |
| TTS to first audio | <= 300 | 100-250 | 80-150 |
| Output buffer | <= 50 | 10-30 | 5-20 |
| **Total (end of speech to audio)** | **<= 1500** | **~0.8-1.3 s, typical ~1.05 s** | **~0.55-0.85 s** |

HS100W radio latency is added on top (T6). Acceptance gates proposed for T7 in
smart mode: median <= 1.2 s and p95 <= 1.8 s; stretch median <= 1.0 s. The
floor column needs a small whisper model, so it is a quality/latency trade-off
that M2 (WER) decides, not a free win. A sub-second median is not credible on
this CPU with one whisper pass per chunk.

### 12.4 Measured whisper time (step 9)

The "Transcription" row of 12.3 assumed 200-500 ms. Measured on this machine (Ryzen 5 2400G,
Windows, a browser and Docker running, so absolute numbers are noisy), faster-whisper 1.2.1,
int8, greedy, 8 SAPI utterances of 2.0-3.2 s:

| Model | Threads | Median | Min - max |
| --- | --- | --- | --- |
| `base.en` | 2 | 2.28 s | 2.18 - 2.58 s |
| `base.en` | 4 | 1.67 s | 1.61 - 2.00 s |
| `base.en` | 8 | 1.67 s | 1.54 - 2.03 s |
| `base.en` (second run, busier machine) | 4 | 2.04 s | 1.80 - 2.38 s |
| `tiny.en` | 4 | 1.14 s | 0.94 - 2.32 s |
| `small.en` | 4 | 7.22 s | - |

Batched decoding (`BatchedInferencePipeline`, `batch_size=8`, int8, 4 threads, same machine,
median per audio; `small.en` ran from a scratch folder and is not in `models/`):

| Model | Audio | Plain | Batched, own clips | Batched, VAD |
| --- | --- | --- | --- | --- |
| `base.en` | 8 utterances of 2.5 s | 2.00 s | 1.94 s | 2.36 s |
| `base.en` | 14.5 s | 2.51 s | 2.56 s | 2.74 s |
| `base.en` | 48 s | 6.37 s | 5.28 s | 5.08 s |
| `small.en` | 8 utterances of 2.5 s | 7.22 s | 8.56 s | 8.33 s |
| `small.en` | 14.5 s | 9.09 s | 9.20 s | 8.82 s |
| `small.en` | 48 s | 21.39 s | 15.91 s | 15.33 s |

Batching only pays when the audio spans more than one 30 s window (48 s: 17-26% faster).
Our chunks are at most 15 s, so each fits one window, batching gives nothing (2.5 s and 14.5 s
rows are equal within noise, or slower) and is not used. `small.en` is 3.6 times slower than
`base.en` on a short utterance, about three times real time, so it cannot run live on this CPU.

**Accuracy of `tiny.en` against `base.en` on human speech.** 80 LibriSpeech `test-clean`
utterances (read speech, 15 speakers, 1.5-15 s each, 495 s in total, every 33rd file of the
sorted list that fits that length), through the project's `FasterWhisperEngine` with its
default settings, 4 threads. Text normalized (lower case, no punctuation, hyphen as space,
Mr/Mrs/Dr/St expanded); references are LibriSpeech's own transcripts. Both models make the
same kinds of mistakes (names, rare words), the numbers below are for the whole sample:

| Model | WER | Utterances with any error | Decode median | Decode p95 |
| --- | --- | --- | --- | --- |
| `tiny.en` | 7.20% (99 of 1375 words) | 46 of 80 | 1.17 s | 1.80 s |
| `base.en` | 5.38% (74 of 1375 words) | 40 of 80 | 2.19 s | 2.83 s |

`tiny.en` is worse than `base.en` on 23 utterances, better on 6 and equal on 51, so the gap is
real and not noise (a sign test on 29 differing utterances gives two-sided p = 0.002), but it is
1.8 points of WER for a decode that is 1.9 times faster. Typical `tiny.en` slips are
sound-alike words ("peak" for "beak", "rolled" for "ruled", "Mark a lot" for "Margolotte").
Read speech is cleaner than a microphone in a room; the SAPI test (M2) says 0.0% for `base.en`
and 1.3% for `tiny.en` and is the easy end. With glossary hotwords (5 made-up company terms,
14 occurrences) `tiny.en` goes from 3 to 13 correct and `base.en` from 5 to 11, so the glossary
helps the small model more. Real decode times here are longer than for the 2.5 s SAPI phrases because
these clips average 6 s.

Feature extraction is 2-4 ms and model load 0.7 s, so the time is the decode. Eight threads
do not beat four (as 12.2 expected). The time barely depends on the length of the speech,
which fits the 30 s encoder window of W3. `tiny.en` and `base.en` are both in `models/` and
selectable (`TranscribeConfig.model`); `small.en` is not offered.

Consequences:

* Even `tiny.en` is about twice the 500 ms of the budget; `base.en` is about three to four times.
* With the silence wait (`fixed`, 450-600 ms) plus about 1.1-2 s of whisper, the end-to-end
  time is far above the T7 gates (median 1.2 s, p95 1.8 s) before translation and TTS are
  counted. The gates cannot be met on this CPU with one whisper pass per chunk, whatever
  Smart Turn does. The levers that attack the decode are W5 (incremental features, small) and
  W6 (shorter encoder context, unverified), or a faster machine.
* Open question 1 (acceptance numbers) therefore needs the user's answer before step 12.

## 13. Open questions (need the user / client)

1. **Acceptance numbers**: confirm the gates in 12.3 (median <= 1.2 s, p95 <=
   1.8 s) or name others, and whether the metric is "end of speech to audio"
   (primary) plus "lag behind speaker" (reported).
2. **Disclosure of D2**: the smart trigger is an additional finalization
   trigger next to the configurable 450-600 ms silence threshold. Confirm the
   client is told in the README and demo, or flip the default to `fixed`.
3. Hold mode (5.6) lengthens silence beyond 600 ms; include or not.
4. Pre-roll 300 ms instead of 100 ms (section 2): accept.
5. Smart Turn threshold tuning: ship 0.5 and tune in T10, or pick a safer 0.6
   for the video.
6. **Source language.** The base spec uses `--source` with an example value
   that the user has now ruled out for this extension. Which source language
   (and targets) should the demo, M3 and the video use? Until named, tests use
   English audio.
7. CPU budget: Smart Turn adds one thread; confirm S1 soak still has 0
   overflows with whisper, two MT lanes and TTS running.
8. Catch-up speed (speed up TTS when a lane falls behind): not specified;
   Sokuji's source has only a static speed.
9. Merging two chunks after a premature cut before translation: not specified;
   revisit if T10 shows more than 10% premature cuts.
10. **Smart mode default (new, from the step 2b measurements, revised after the
    follow-up).** Quality is not the problem: with the fp32 file and text-labelled
    probes the model separates finished from unfinished speech (94% vs 12%). Speed is:
    one Zen1 core needs 220-570 ms per check, while the window between a reliable check
    (about 224-320 ms of silence) and the fixed rule (480 ms) is 160-256 ms. Options:
    (a) keep `fixed` as the working default and ship smart as an opt-in flag until T10
    shows a gain on real microphone speech; (b) keep `smart` as the default per D2 and
    accept that on this CPU it gains nothing and may add cuts; (c) give the model 2-4
    threads (int8 measured: 210-280 ms with 2, 150-200 ms with 4; fp32 not re-measured)
    at the cost of whisper's cores. Recommendation: (a).
