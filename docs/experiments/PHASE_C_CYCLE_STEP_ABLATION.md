# Phase C Cycle-Step Ablation

Date: 2026-05-24

## Completed Scope

Phase C compares four balanced-debug Stage 2 runs where the only ablation variable is the cycle step setting. The monitor interval is 10 for all runs, and `best_model: raw` is also fixed for all runs. Those two fields define this round of diagnostics; they are not ablation variables.

No code logic was changed for this document. No new training was started. The tables below were generated from each checkpoint directory's `manifest.json` and `last_metrics.json`.

## Run Artifacts

| Run | Cycle setting | Config | Checkpoint dir | Log | Best | Last | best_model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| monitor10_rawbest_fixed8 | fixed 8 | `configs/stability/train_g_balanced_debug_monitor10_rawbest_fixed8.yaml` | `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_fixed8` | `artifacts/logs/train_g_balanced_debug_monitor10_rawbest_fixed8_gpu3_6.log` | `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_fixed8/best.pt` | `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_fixed8/last.pt` | `raw` |
| monitor10_rawbest_fixed16 | fixed 16 | `configs/stability/train_g_balanced_debug_monitor10_rawbest_fixed16.yaml` | `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_fixed16` | `artifacts/logs/train_g_balanced_debug_monitor10_rawbest_fixed16_gpu3_6.log` | `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_fixed16/best.pt` | `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_fixed16/last.pt` | `raw` |
| monitor10_rawbest_schedule_4_8_16 | schedule [4, 8, 16] | `configs/stability/train_g_balanced_debug_monitor10_rawbest_schedule_4_8_16.yaml` | `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_schedule_4_8_16` | `artifacts/logs/train_g_balanced_debug_monitor10_rawbest_schedule_4_8_16_gpu3_6.log` | `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_schedule_4_8_16/best.pt` | `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_schedule_4_8_16/last.pt` | `raw` |
| monitor10_rawbest_schedule_4_8_16_32 | schedule [4, 8, 16, 32] | `configs/stability/train_g_balanced_debug_monitor10_rawbest_schedule_4_8_16_32.yaml` | `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_schedule_4_8_16_32` | `artifacts/logs/train_g_balanced_debug_monitor10_rawbest_schedule_4_8_16_32_gpu3_6.log` | `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_schedule_4_8_16_32/best.pt` | `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_schedule_4_8_16_32/last.pt` | `raw` |

## Shared Setup

These fields are held fixed across the four runs:

- Seed and sampling seed: `1337`
- Train index: `data/index/train_balanced_debug.jsonl`
- Train feature cache: `artifacts/e0_features/train_balanced_debug`
- E0 checkpoint: `artifacts/checkpoints/e0/best.pt`
- Resume checkpoint: `artifacts/checkpoints/g_v2_best_stage1/best.pt`
- Stage 1 epochs: `0`; Stage 2 epochs: `5`
- Generator: conditional flow matching, `base_channels: 32`, `channel_multipliers: [1, 2, 4, 4]`, Heun sampling with 32 sample steps
- Batch size: `16`; learning rate: `0.0003`; gradient clip norm: `1.0`
- Cycle weight: `lambda_initial: 0.01`, `lambda_max: 0.01`, `lambda_growth: 0`
- EMA monitoring: enabled, `decay: 0.999`, raw validation enabled, EMA validation enabled
- Gradient conflict monitoring: enabled with `interval: 10`
- Validation: `data/index/val.jsonl`, `artifacts/e0_features/val`, `max_samples: 512`, InsightFace `buffalo_l`

## Final Epoch Metrics

All final rows are `stage: stage2`, `stage_epoch: 4`.

| Run | Cycle setting | Raw cosine | EMA cosine | Raw eq1 | EMA eq1 | Conflict | Raw cycle/FM norm | Weighted cycle/FM | Weighted cycle norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| monitor10_rawbest_fixed8 | fixed 8 | 0.768856 | 0.650133 | 0.988281 | 0.998047 | 0.250000 | 18.517310 | 0.185173 | 0.047821 |
| monitor10_rawbest_fixed16 | fixed 16 | 0.885797 | 0.674191 | 1.000000 | 1.000000 | 0.153846 | 18.247272 | 0.182473 | 0.047944 |
| monitor10_rawbest_schedule_4_8_16 | schedule [4, 8, 16] | 0.885008 | 0.669990 | 1.000000 | 1.000000 | 0.307692 | 16.659872 | 0.166599 | 0.040728 |
| monitor10_rawbest_schedule_4_8_16_32 | schedule [4, 8, 16, 32] | 0.879287 | 0.676597 | 0.998047 | 1.000000 | 0.384615 | 16.064982 | 0.160650 | 0.035583 |

## Epoch History

### monitor10_rawbest_fixed8

| Epoch | Raw cosine | EMA cosine | Raw eq1 | EMA eq1 | Conflict | Raw cycle/FM norm | Weighted cycle/FM | Weighted cycle norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.754125 | 0.543660 | 0.921875 | 0.996094 | 0.403846 | 32.673145 | 0.326731 | 0.087606 |
| 1 | 0.812015 | 0.578451 | 1.000000 | 0.996094 | 0.288462 | 25.405290 | 0.254053 | 0.059044 |
| 2 | 0.691010 | 0.607023 | 0.974609 | 0.998047 | 0.403846 | 16.545885 | 0.165459 | 0.047468 |
| 3 | 0.738246 | 0.628775 | 0.998047 | 0.998047 | 0.211538 | 18.361333 | 0.183613 | 0.045044 |
| 4 | 0.768856 | 0.650133 | 0.988281 | 0.998047 | 0.250000 | 18.517310 | 0.185173 | 0.047821 |

### monitor10_rawbest_fixed16

| Epoch | Raw cosine | EMA cosine | Raw eq1 | EMA eq1 | Conflict | Raw cycle/FM norm | Weighted cycle/FM | Weighted cycle norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.815891 | 0.550213 | 0.992188 | 0.996094 | 0.288462 | 34.999394 | 0.349994 | 0.088153 |
| 1 | 0.872974 | 0.590083 | 1.000000 | 1.000000 | 0.346154 | 29.598034 | 0.295980 | 0.065400 |
| 2 | 0.859661 | 0.622698 | 0.990234 | 1.000000 | 0.307692 | 18.714368 | 0.187144 | 0.053815 |
| 3 | 0.873867 | 0.650067 | 1.000000 | 1.000000 | 0.288462 | 20.136686 | 0.201367 | 0.053225 |
| 4 | 0.885797 | 0.674191 | 1.000000 | 1.000000 | 0.153846 | 18.247272 | 0.182473 | 0.047944 |

### monitor10_rawbest_schedule_4_8_16

| Epoch | Raw cosine | EMA cosine | Raw eq1 | EMA eq1 | Conflict | Raw cycle/FM norm | Weighted cycle/FM | Weighted cycle norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.844404 | 0.548120 | 0.998047 | 0.996094 | 0.384615 | 31.984062 | 0.319841 | 0.087859 |
| 1 | 0.853088 | 0.586458 | 1.000000 | 0.996094 | 0.250000 | 27.275664 | 0.272757 | 0.058003 |
| 2 | 0.759812 | 0.618028 | 0.962891 | 0.998047 | 0.384615 | 19.397371 | 0.193974 | 0.055005 |
| 3 | 0.849306 | 0.646088 | 0.998047 | 0.998047 | 0.192308 | 16.132511 | 0.161325 | 0.043641 |
| 4 | 0.885008 | 0.669990 | 1.000000 | 1.000000 | 0.307692 | 16.659872 | 0.166599 | 0.040728 |

### monitor10_rawbest_schedule_4_8_16_32

| Epoch | Raw cosine | EMA cosine | Raw eq1 | EMA eq1 | Conflict | Raw cycle/FM norm | Weighted cycle/FM | Weighted cycle norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.826947 | 0.551862 | 0.996094 | 0.996094 | 0.250000 | 30.816041 | 0.308160 | 0.081611 |
| 1 | 0.875840 | 0.592528 | 1.000000 | 0.998047 | 0.346154 | 23.164346 | 0.231643 | 0.053299 |
| 2 | 0.843974 | 0.623873 | 0.976562 | 1.000000 | 0.326923 | 17.404552 | 0.174046 | 0.048716 |
| 3 | 0.875866 | 0.651530 | 1.000000 | 1.000000 | 0.365385 | 15.160463 | 0.151605 | 0.041218 |
| 4 | 0.879287 | 0.676597 | 0.998047 | 1.000000 | 0.384615 | 16.064982 | 0.160650 | 0.035583 |

## Answers To The 10 Phase C Questions

### 1. What changed in this ablation?

Only the cycle step setting changed: fixed 8, fixed 16, schedule `[4, 8, 16]`, or schedule `[4, 8, 16, 32]`. The monitor interval is 10 for every run, and raw-best checkpoint selection is used for every run. They are part of this diagnostic protocol, not ablation variables.

### 2. What stayed fixed?

Data, feature caches, E0 checkpoint, Stage 1 resume checkpoint, model scale, seed, Stage 2 length, `lambda_cycle=0.01`, validation set, EMA decay, raw/EMA validation, and gradient-conflict monitor interval all stayed fixed.

### 3. Which checkpoint family should be used for short-run judgment?

Raw is the better short-run signal. EMA is still useful to monitor, but with `ema.decay: 0.999` it lags in these 5-epoch debug runs. Every final raw cosine is higher than its paired EMA cosine by about 0.12 to 0.21.

### 4. Which cycle setting is best by final raw cosine?

Fixed16 and schedule `[4, 8, 16]` are effectively tied on final raw cosine: `0.885797` versus `0.885008`. Fixed8 is lower at `0.768856`. Schedule `[4, 8, 16, 32]` is also below fixed16 at `0.879287`.

### 5. How should fixed16 and schedule [4, 8, 16] be compared?

They reach nearly the same final raw cosine, but fixed16 has the cleaner final conflict signal: `0.153846` for fixed16 versus `0.307692` for schedule `[4, 8, 16]`. With no image-quality or privacy eval, fixed16 is the more conservative Phase C pick.

### 6. Did schedule [4, 8, 16, 32] help?

No. Its final raw cosine is below fixed16, and its final conflict is the highest of the four runs at `0.384615`. It also includes a 32-step cycle phase, so it is slower and more expensive than the settings capped at 16 cycle steps. There is no measured benefit here.

### 7. Did cycle dominate flow matching after lambda weighting?

No. The raw cycle/FM norm ratio is large, roughly 16x to 35x across these histories, but that is not the actual update ratio because `lambda_cycle=0.01` scales the cycle term. The weighted cycle/FM ratio is about 0.16 to 0.35 across the recorded histories, with final values around 0.16 to 0.19. After lambda weighting, cycle does not overpower flow matching.

### 8. What does the direction-conflict signal say?

Direction conflict remains present. Fixed16 has the lowest final conflict at `0.153846`, while the schedule runs end higher. This supports keeping conflict diagnostics in Phase D planning, but it does not prove that PCGrad or another gradient surgery method is required now.

### 9. What should be done next before Phase D?

The next step should stay close to fixed16: either reproduce fixed16 once, or run a small EMA-decay check with fixed16. After that, decide whether Phase D needs PCGrad or another gradient-balancing method. The current evidence is not strong enough to jump straight to PCGrad.

### 10. What can this ablation not prove?

It cannot prove final image quality, because FID, KID, and IQA were not run. It cannot prove privacy, because no privacy evaluation was run. It also cannot prove seed-level reproducibility or behavior on longer training runs.

## Current Decision

Use fixed16 as the Phase C cycle-step choice for the next short-run decision point. It matches schedule `[4, 8, 16]` on final raw cosine, has lower final conflict, avoids the extra 32-step cost, and keeps the interpretation simple.
