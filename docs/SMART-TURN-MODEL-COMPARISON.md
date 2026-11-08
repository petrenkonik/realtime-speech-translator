# Smart Turn v3.2: int8 file vs fp32 file

Measured 2026-11-08. This note compares the two model files the vendor publishes for
end-of-turn detection ("has the speaker finished the sentence?"): the int8 file
(`smart-turn-v3.2-cpu.onnx`) and the unquantized fp32 file (`smart-turn-v3.2-gpu.onnx`,
the vendor's name; it runs on any CPU). A third variant, the int8 file with the
onnxruntime option `session.x64quantprecision=1`, is included because it changes the
int8 kernels.

> **All quality numbers in sections 3 and 4 were measured on synthesized speech**
> (Windows SAPI text-to-speech, two English voices), not on a human voice. Section 5
> uses real read speech, but only to compare the files with each other: it has no
> reliable "finished / unfinished" labels. How the models behave on a live microphone is
> still unmeasured (test T10).

## 1. Summary

- **Same network, two precisions.** Both files have the same input and output; the int8
  one is the quantized copy of the fp32 one. Nothing else differs.
- **On synthesized speech the two files are practically the same.** At the first check
  (224 ms of silence) they take the same decision on 97% of the 32 probes (int8) and on
  100% (int8 with `x64quantprecision=1`). Balanced accuracy is 88% (int8) and 91%
  (the other two); with 16 probes per class these differences are far inside the noise.
- **On real read speech the files disagree much more**: the correlation of their
  probabilities is 0.71 for plain int8 and 0.88 for int8 with the precision option. The
  sample has no ground truth, so this shows that int8 moves away from the vendor's
  reference, not which file is right.
- **fp32 is faster on this CPU at every thread count** (by 13-30% in the median): 292-300 ms
  against 369-386 ms on one thread. The 8 MB int8 file saves disk, not time, on a CPU
  without the VNNI instructions.
- **Choice:** fp32 is the default (`pipeline.turn.DEFAULT_TURN_MODEL`): the vendor's
  reference, faster, and the only variant whose answers do not depend on the CPU's
  int8 kernels.

## 2. The compared files

| | int8 | int8 + precision option | fp32 |
| --- | --- | --- | --- |
| File | `smart-turn-v3.2-cpu.onnx` | the same file | `smart-turn-v3.2-gpu.onnx` |
| Size | 8 679 182 bytes | same | 32 411 198 bytes |
| sha256 | `2bb02631...a967e4f` | same | `ab8dc64b...c539be` |
| Graph | quantized operators (`QLinearMatMul`, `QLinearSoftmax`, `QLinearConv`) | same | float operators |
| onnxruntime option | none | `session.x64quantprecision=1` (avoids int16 saturation of int8 matmuls on AVX2 CPUs) | none (no effect) |
| Input / output | `input_features` float32 `[1, 80, 800]` / `logits` `[1, 1]`, a probability in [0, 1] | same | same |
| License | BSD-2-Clause | same | same |
| Source | Hugging Face `pipecat-ai/smart-turn-v3`, revision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93` | same | same |

The vendor's README says the fp32 file has "slightly improved accuracy by around 1%"
over the quantized one; that figure is the vendor's, not measured here.

Profile of the int8 file on one thread (this machine): `QLinearMatMul` 48% of the time,
`QLinearSoftmax` 38%, convolution 3%. The fp32 file was not profiled.

## 3. Quality on synthesized speech

**What is synthesized.** English speech from the offline Windows voices
`Microsoft David Desktop` and `Microsoft Zira Desktop`. Two sets of 8 sentences, each
spoken by both voices, so 16 probes per class:

- *Finished* (mean 2.51 s of speech), e.g. "The meeting starts at nine tomorrow morning."
  The full list is `SENTENCES` in `tests/sapi_speech.py`.
- *Unfinished* (mean 2.23 s), sentences that stop on a connective, a filler or an open
  article, e.g. "I would like to order a pizza with mushrooms and", "Can you tell me
  where the, um,". The list is `UNFINISHED` in the same file. The label comes from the
  text, which is the vendor's definition of an incomplete turn.

**How a probe is built.** 96 ms of low white noise, the speech, then a tail of silence
(white noise, standard deviation 0.001) of 0, 100, 224, 320 or 500 ms. 224 ms is what the
Segmenter waits before its first check with the default settings. The model answers with
a probability; "complete" means p >= 0.5. Brackets are 95% Wilson intervals.

**"Complete" after the speech plus the tail (share of probes).**

Finished sentences (higher is better):

| Tail | int8 | int8 + precision | fp32 |
| --- | --- | --- | --- |
| 0 ms | 14/16 = 88% [64-97] | 13/16 = 81% [57-93] | 12/16 = 75% [51-90] |
| 100 ms | 13/16 = 81% [57-93] | 15/16 = 94% [72-99] | 14/16 = 88% [64-97] |
| **224 ms** | **15/16 = 94% [72-99]** | **15/16 = 94% [72-99]** | **15/16 = 94% [72-99]** |
| 320 ms | 15/16 = 94% [72-99] | 16/16 = 100% [81-100] | 16/16 = 100% [81-100] |
| 500 ms | 15/16 = 94% [72-99] | 16/16 = 100% [81-100] | 16/16 = 100% [81-100] |

Unfinished sentences (lower is better; this is the false end rate):

| Tail | int8 | int8 + precision | fp32 |
| --- | --- | --- | --- |
| 0 ms | 2/16 = 12% [3-36] | 1/16 = 6% [1-28] | 1/16 = 6% [1-28] |
| 100 ms | 3/16 = 19% [7-43] | 1/16 = 6% [1-28] | 1/16 = 6% [1-28] |
| **224 ms** | **3/16 = 19% [7-43]** | **2/16 = 12% [3-36]** | **2/16 = 12% [3-36]** |
| 320 ms | 3/16 = 19% [7-43] | 3/16 = 19% [7-43] | 1/16 = 6% [1-28] |
| 500 ms | 3/16 = 19% [7-43] | 3/16 = 19% [7-43] | 3/16 = 19% [7-43] |

Balanced accuracy (mean of the finished hit rate and the unfinished rejection rate):

| Tail | int8 | int8 + precision | fp32 |
| --- | --- | --- | --- |
| 0 ms | 88% | 88% | 84% |
| 100 ms | 81% | 94% | 91% |
| **224 ms** | **88%** | **91%** | **91%** |
| 320 ms | 88% | 91% | 97% |
| 500 ms | 88% | 91% | 91% |

Reading the tables:

- Every difference between the files is one or two probes out of 16, and the intervals
  overlap almost completely. **On this data the three variants cannot be told apart
  statistically.** The visible trends (fp32 and the precision option slightly better than
  plain int8; the hit rate rising with more silence) are consistent with the vendor's
  statement, but they are not proof.
- **The silence length matters for all three**: for finished sentences "complete" rises
  from 75-88% with no tail to 94-100% with 224-320 ms.
- Median probability: 0.91-0.99 for finished and 0.01 for unfinished sentences in every
  cell, so wrong answers are single probes, not a general blur.

**Agreement between the files at 224 ms (all 32 probes).**

| Variant vs fp32 | Same decision | Correlation of p | Mean abs. difference | Max abs. difference |
| --- | --- | --- | --- | --- |
| int8 | 97% | 0.967 | 0.049 | 0.477 |
| int8 + precision | 100% | 0.988 | 0.029 | 0.318 |

## 4. Speed

Model only (features are extra: median 22 ms, p95 32 ms), milliseconds as
min / median / p95 over 40 runs after 3 warm-ups, two passes. Input: one 8 s window.
Machine: AMD Ryzen 5 2400G (Zen1, AVX2, no AVX-512, no VNNI), Windows 10 Pro,
onnxruntime 1.31.1, `ORT_SEQUENTIAL`, `ORT_ENABLE_ALL`. Docker, a browser and other
applications were running, so absolute numbers are noisy (compare the two passes).

| Threads | Pass | int8 | int8 + precision | fp32 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 337 / 386 / 476 | 365 / 409 / 505 | **254 / 292 / 358** |
| 1 | 2 | 330 / 369 / 471 | 378 / 470 / 632 | **238 / 300 / 379** |
| 2 | 1 | 214 / 239 / 301 | 216 / 254 / 345 | **161 / 187 / 232** |
| 2 | 2 | 202 / 239 / 319 | 202 / 243 / 356 | **128 / 175 / 259** |
| 4 | 1 | 139 / 170 / 256 | 132 / 147 / 533 | **102 / 119 / 160** |
| 4 | 2 | 128 / 149 / 211 | 124 / 141 / 232 | **96 / 130 / 213** |

- fp32 has the lowest median at every thread count, by 13-30% over plain int8 (the
  smallest gap is 130 vs 149 ms on four threads in the second pass).
- The precision option costs 6-27% on one thread and is within the noise on 2-4 threads.
- Scaling is far from linear: four threads give about 2.3-2.5x, not 4x.
- The vendor's 12.6 ms (8 vCPU AVX-512 server, default threads) is at least 7x below
  any number here (best: 96 ms, fp32, four threads). One Zen1 core does about 27 GFLOPS on a large fp32 matmul and the model needs
  roughly 7 GFLOP (estimate), so ~260 ms for one thread is close to this core's limit.

## 5. Real speech, agreement only

73 short English read-speech clips (LibriSpeech, `hf-internal-testing/librispeech_asr_dummy`,
CC-BY-4.0, kept outside the repository), each fed as recorded. They end with 105-755 ms of
natural silence (median 535 ms). The clips are cut at pauses, not at sentence ends, so
**there are no trustworthy labels and no accuracy is claimed**; the table only compares
the files with each other.

| Variant | Correlation with fp32 | Mean abs. difference | Max abs. difference | "Complete" |
| --- | --- | --- | --- | --- |
| fp32 | - | - | - | 82% |
| int8 | 0.713 | 0.164 | 0.868 | 68% |
| int8 + precision | 0.881 | 0.083 | 0.825 | 81% |

For plain int8 several clearly finished sentences got p = 0.01 (for example "your friends
are asking for you", with a 505 ms tail). The precision option removes most of the gap,
which is consistent with int8 saturation on AVX2 CPUs without VNNI; the mechanism itself
was not proved. Synthesized speech is close to the vendor's training data (largely TTS),
which is a likely reason why int8 looks fine in section 3 and not here.

## 6. Limits of this comparison

- 16 probes per class, two voices, eight sentences each: the intervals are wide.
- The speech is synthesized and flat-voiced; the tail is clean low-level white noise, not
  a room.
- Only one CPU was tested. On a CPU with VNNI or AVX-512 the int8 file is expected to be
  closer to fp32 and faster (the vendor's timings come from such machines); this was not
  measured.
- No live microphone and no human conversation. The vendor's 94% English accuracy is
  from its v3 blog post, on a different test set, and was not reproduced.
- The probes, the per-probe probabilities and the benchmark script are in the session
  scratch area, not in the repository. The repository's model tests (`tests/test_turn_model.py`,
  marker `model`) rerun the finished / unfinished checks on the fp32 file.

## 7. Related

- `docs/SPEC-LOW-LATENCY.md` section 4 (model, features, why fp32 is the default) and
  open question 10 (whether smart mode is the default).
- `THIRD_PARTY_NOTICES.md` (model licenses).
