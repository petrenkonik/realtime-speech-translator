# Endpointing comparison: measured results

Measured 2026-11-08 on the project machine (AMD Ryzen 5 2400G, 4 cores /
8 threads, Windows 10, other applications running — absolute numbers are
pessimistic, comparisons between rows are still valid). Data: real LibriSpeech
read-speech clips from `data/` (CC BY-4.0, fetched by
`scripts/fetch_models.py --with-data`), labelled by the energy-based labels of
`tests/endpointing_eval.py`. Two experiments:

- **A. Fixed vs Smart Turn** — the real `Segmenter` (`src/pipeline/vad.py`),
  13 configurations, one seed.
- **B. Silence-rule sweep** — the shipped hysteresis rule against the
  alternative "integral" rule, 128 configurations, two seeds.

Raw data: `data/endpointing_results.json` (experiment A),
`data/sweep_results.json` and `data/sweep_results_seed2.json` (experiment B).
All three files are generated, not committed.

## 1. Metrics

- **Precision** — share of cuts that fall on a reference phrase end.
- **Recall** — share of reference phrase ends that got a cut. Brackets are
  95% Wilson intervals.
- **Premature cuts** — share of reference mid-speech segments that got a cut,
  i.e. the segmenter ended a chunk while the speaker was still going. This is
  the harmful error: audio is split mid-sentence (Smart Turn's rescue path
  returns it to the open chunk at runtime, but the benchmark scores the raw
  cut decision).
- **Latency** — decision time after the end of speech, matched phrase ends
  only, p50 / p95.

300 phrase ends and 183 mid-speech segments per experiment.

## 2. Experiment A: fixed vs Smart Turn

`fixed N` ends a chunk after N ms of silence. `smart` additionally asks the
pipecat-ai smart-turn v3.2 model (fp32) during the pause; `check N` is the
first-check time, `thr X` the turn threshold, `(p50/p95 delay)` means the
model check runs on a worker and its answer applies after its measured p50 or
p95 delay.

| Configuration | Precision (95% CI) | Recall (95% CI) | Premature cuts | Latency p50 / p95 |
| --- | --- | --- | --- | --- |
| fixed 450 | 52% (46-57) | 49% (44-55) | 40% | 554 / 646 ms |
| fixed 525 | 55% (49-61) | 46% (41-52) | 33% | 618 / 712 ms |
| fixed 600 | 58% (51-64) | 42% (37-48) | 22% | 675 / 774 ms |
| smart 450 (p50 delay) | 53% (47-58) | 52% (46-57) | 40% | 540 / 626 ms |
| smart 450 (p95 delay) | 52% (46-57) | 49% (44-55) | 40% | 554 / 646 ms |
| smart 600 (p50 delay) | 56% (50-62) | 49% (43-55) | 30% | 544 / 715 ms |
| smart 600 check 300 | 60% (54-66) | 49% (44-55) | 27% | 636 / 738 ms |
| smart 450 thr 0.3 | 53% (47-58) | 52% (46-57) | 40% | 540 / 626 ms |
| smart 450 thr 0.7 | 53% (47-58) | 52% (46-57) | 40% | 540 / 626 ms |
| smart 450 thr 0.9 | 52% (46-58) | 51% (45-56) | 40% | 542 / 636 ms |
| smart 600 thr 0.3 | 56% (50-62) | 50% (44-55) | 31% | 544 / 714 ms |
| smart 600 thr 0.7 | 57% (50-62) | 49% (43-54) | 30% | 553 / 726 ms |
| smart 600 check 150 | 56% (50-61) | 51% (46-57) | 31% | 499 / 720 ms |

Reading the table:

- **The threshold barely matters** (rows `smart 450 thr 0.3/0.7/0.9` are
  equal within noise): the model's answers are near 0 or 1, so any threshold
  between 0.3 and 0.9 separates the same pauses.
- **More silence buys fewer premature cuts at the price of recall** — the
  fixed 450 -> 600 row gradient. Smart mode shifts the frontier slightly:
  at the same `silence_ms` it recovers some 500 ms pauses (recall 52% vs 49%
  at 450) and cuts a bit earlier.
- Every smart gain is one to seven phrase ends out of 300 — inside the
  intervals. **Smart mode is not yet worth its cost on this CPU** (~120-300 ms
  per check, see [SMART-TURN-MODEL-COMPARISON.md](SMART-TURN-MODEL-COMPARISON.md)),
  which is why `turn_mode` defaults to `fixed`.

### Recall by reference pause length (fixed 450 vs smart 450)

| Reference pause | fixed 450 | smart 450 (p50 delay) |
| --- | --- | --- |
| 200 ms | 3 / 46 | 2 / 46 |
| 300 ms | 4 / 47 | 4 / 47 |
| 400 ms | 3 / 27 | 3 / 27 |
| 500 ms | 7 / 49 | 15 / 49 |
| 650 ms | 36 / 36 | 36 / 36 |
| 800 ms | 35 / 35 | 35 / 35 |
| 1000 ms | 60 / 60 | 60 / 60 |

The shipped rule never cuts pauses shorter than itself (by design — a cut
there would be premature) and cuts every pause of 650 ms and longer. The
whole game is the 500 ms row: Smart Turn turns 7 of 49 into 15 of 49.

## 3. Experiment B: silence-rule sweep

Two ways to turn frame probabilities into an "end of speech" decision, both
calibrated so that pure silence of the nominal length ends the chunk:

- **hysteresis** (what ships): the silence clock starts at the first frame
  below `exit_threshold` and is cleared by any frame at or above
  `speech_threshold`; the chunk ends when the clock has run the full
  `silence_ms`.
- **integral**: one sum S of `(1 - p)` per frame since the last speech frame;
  the chunk ends when S reaches `silence_ms` worth of frames. Unsure frames
  add less, so duration and confidence share one number. Not shipped — the
  sweep ran it through a reference port, not `src/`.

128 configurations per seed (88 hysteresis: speech thresholds 0.3-0.7,
exit thresholds 0.15-0.7, silence 450-600 ms; 40 integral: speech thresholds
0.3-0.7, nominal silence 250-600 ms), 60 streams x 5 phrase groups, two seeds.

**Best rows per family.**

| Seed | Rule | Configuration | F1 | Precision | Recall | Premature | Latency p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hysteresis | 0.7/0.70, 450 ms | 0.53 | 52% | 54% | 43% | 536 ms |
| 1 | integral | speech 0.4, S >= 300 ms | 0.58 | 51% | 67% | 52% | 422 ms |
| 2 | hysteresis | 0.6/0.60, 450 ms | 0.57 | 57% | 57% | 40% | 544 ms |
| 2 | integral | speech 0.7, S >= 250 ms | 0.62 | 52% | 79% | 63% | 358 ms |

**The shipped thresholds across both seeds.**

| Seed | Configuration | F1 | Precision | Recall | Premature | Latency p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | hysteresis 0.5/0.35, 450 ms | 0.50 | 52% | 49% | 40% | 554 / 646 ms |
| 1 | hysteresis 0.5/0.35, 500 ms | 0.51 | 54% | 48% | 37% | 586 / 672 ms |
| 1 | hysteresis 0.5/0.35, 550 ms | 0.50 | 56% | 45% | 30% | 648 / 744 ms |
| 1 | hysteresis 0.5/0.35, 600 ms | 0.49 | 58% | 42% | 22% | 675 / 774 ms |
| 2 | hysteresis 0.5/0.35, 450 ms | 0.56 | 57% | 54% | 38% | 558 / 650 ms |
| 2 | hysteresis 0.5/0.35, 500 ms | 0.55 | 57% | 52% | 38% | 590 / 680 ms |
| 2 | hysteresis 0.5/0.35, 550 ms | 0.53 | 57% | 49% | 34% | 650 / 746 ms |
| 2 | hysteresis 0.5/0.35, 600 ms | 0.50 | 57% | 45% | 29% | 681 / 778 ms |

Reading the sweep:

- **Integral gets its higher F1 entirely from recall bought with premature
  cuts** (67-79% recall at 52-63% premature). Hysteresis at the same
  premature rate keeps more precision and loses less text mid-sentence — the
  trade-off a live interpreter should prefer.
- The exact thresholds matter less than the silence length; exit thresholds
  between 0.25 and 0.45 at speech 0.5 are equivalent within noise.
- The two seeds agree on every conclusion; row-to-row differences of 1-4
  phrase ends are noise.

## 4. Conclusions

- Ship **hysteresis, speech 0.5, exit 0.35, silence 450 ms** (the defaults of
  `SegmenterConfig`) — it sits on the safe side of the premature-cut
  trade-off, and the sweep shows no configuration that dominates it.
- Ship **`turn_mode: fixed`** and expose smart mode as an opt-in setting; on
  this CPU the model check costs more than it saves (experiment A).
- Revisit after the model runs on live microphone audio (open question 10 of
  [SPEC-LOW-LATENCY.md](SPEC-LOW-LATENCY.md)): read speech has clean pauses;
  a real room may not.

## 5. Reproduce

```powershell
python -B scripts/fetch_models.py --with-data
python -B scripts/eval_endpointing.py --out data/endpointing_results.json
python -B scripts/sweep_endpointing.py --out data/sweep_results.json
python -B scripts/pause_viewer.py --out data/pause_viewer.html
```

See also: the interactive per-clip viewer (`data/pause_viewer.html`, built by
[scripts/pause_viewer.py](../scripts/pause_viewer.py)) and the model-file
comparison in [SMART-TURN-MODEL-COMPARISON.md](SMART-TURN-MODEL-COMPARISON.md).
