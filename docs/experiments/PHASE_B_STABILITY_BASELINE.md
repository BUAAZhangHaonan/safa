# Phase B Stability Baseline

Date: 2026-05-24

## Completed Scope

Phase B ran one balanced-debug Stage 2 stability baseline:
`stability_balanced_debug_fixed16`.

This document records the run outcome, the raw-vs-EMA comparison, and the
gradient-conflict signal. It does not claim any extra training, privacy
evaluation, or generation-quality evaluation beyond the artifacts listed here.

## Run Artifacts

- Config: `configs/stability/train_g_balanced_debug_ema_monitor_fixed16.yaml`
- Checkpoint directory: `artifacts/checkpoints/stability_balanced_debug_fixed16`
- Last metrics: `artifacts/checkpoints/stability_balanced_debug_fixed16/last_metrics.json`
- Log: `artifacts/logs/train_g_balanced_debug_ema_monitor_fixed16_gpu3_6.log`
- Completion time: around 2026-05-24 17:45 CST
- Training resources: GPUs 3, 4, 5, and 6 with 4-card DDP
- Known machine note: GPUs 0 and 7 were abnormal, but they were not experiment
  variables for this run.

The run completed 5 Stage 2 epochs. The log check found no `Traceback`, `OOM`,
or `RuntimeError`.

## What Changed

This experiment only ran the balanced-debug Stage 2 stability baseline with
fixed 16 cycle training steps and EMA monitoring:

- `generator.train_cycle_steps: 16`
- raw model validation enabled
- EMA model validation enabled with `ema.decay: 0.999`
- gradient conflict monitoring enabled every 50 steps

No model-scale change was made. No PCGrad, CAGrad, UW/uncertainty weighting,
FAMO, or other gradient-balancing method was added.

## Variables Held Fixed

- Seed and sampling seed: 1337
- Train index: `data/index/train_balanced_debug.jsonl`
- Train feature cache: `artifacts/e0_features/train_balanced_debug`
- E0 checkpoint: `artifacts/checkpoints/e0/best.pt`
- Resume checkpoint: `artifacts/checkpoints/g_v2_best_stage1/best.pt`
- Generator type: conditional flow matching
- Generator scale: `base_channels: 32`, `channel_multipliers: [1, 2, 4, 4]`
- Sampling: 32 Heun steps
- Stage 2 length: 5 epochs
- Cycle weight: `lambda_cycle: 0.01`
- Batch size: 16
- Validation index: `data/index/val.jsonl`
- Validation feature cache: `artifacts/e0_features/val`
- Validation sample count: 512
- Face detector: InsightFace `buffalo_l`

## Final Metrics

Final `last_metrics.json` records `stage: stage2` and `stage_epoch: 4`.

| Metric | Raw model | EMA model |
| --- | ---: | ---: |
| Latent cosine mean | 0.8760175500065088 | 0.6736126346513629 |
| Single-face eq1 rate | 0.994140625 | 1.0 |

Gradient diagnostics at `stage_epoch=4`:

- `gradient_conflict_fraction`: 0.3333333333333333
- `gradient_norm_ratio_cycle_to_fm_mean`: 21.025898896541765

## Epoch Metrics

| Stage epoch | Raw cosine | EMA cosine | Raw single-face eq1 | EMA single-face eq1 | Conflict fraction | Cycle/FM norm ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8208568263798952 | 0.5480066854506731 | 0.9921875 | 0.99609375 | 0.3333333333333333 | 37.1076249014568 |
| 1 | 0.8758511245250702 | 0.5866657737642527 | 1.0 | 0.998046875 | 0.6666666666666666 | 26.982893327323485 |
| 2 | 0.8283501639962196 | 0.6184868467971683 | 0.9765625 | 1.0 | 0.3333333333333333 | 16.855739922724474 |
| 3 | 0.8966683838516474 | 0.6478839032351971 | 1.0 | 1.0 | 0.5833333333333334 | 31.751255386686196 |
| 4 | 0.8760175500065088 | 0.6736126346513629 | 0.994140625 | 1.0 | 0.3333333333333333 | 21.025898896541765 |

## Answers To The Phase B Questions

### What did this experiment change?

It changed only the Phase B stability run setup: fixed 16 cycle training steps,
raw/EMA validation, and gradient-conflict monitoring on the balanced-debug Stage
2 run. It did not change model scale, data, privacy evaluation, or the
optimization algorithm.

### Which variables stayed unchanged?

The balanced-debug data, feature caches, E0 checkpoint, Stage 1 resume
checkpoint, model scale, seed, Stage 2 cycle weight, validation set, and
validation detector stayed unchanged.

### Which is better, raw or EMA?

Raw is better in this 5-epoch debug run on latent cosine:

- raw final cosine: 0.8760175500065088
- EMA final cosine: 0.6736126346513629

EMA has a slightly higher final single-face eq1 rate, 1.0 versus raw
0.994140625, but the cosine gap is large. The cautious read is that raw is the
better current checkpoint for this run.

This does not prove EMA is permanently ineffective. With `ema.decay: 0.999`,
EMA clearly lags in a short 5-epoch debug run.

### Is latent cosine stable?

Raw latent cosine is reasonably stable for a debug run. It stays between
0.8208568263798952 and 0.8966683838516474, and it ends at
0.8760175500065088.

EMA latent cosine rises monotonically from 0.5480066854506731 to
0.6736126346513629, but it remains far behind raw. That pattern is consistent
with EMA lag, not with proof that EMA cannot catch up in a longer run.

### Is single_face_eq1_rate stable?

Yes for this debug run. Raw stays between 0.9765625 and 1.0. EMA stays between
0.99609375 and 1.0 and reaches 1.0 from epoch 2 onward.

### Did FID, KID, or IQA improve?

Unknown. This experiment did not run FID, KID, or IQA. It also did not run a
privacy evaluation.

### What is the gradient conflict fraction?

The final `gradient_conflict_fraction` is 0.3333333333333333. Across epochs it
ranges from 0.3333333333333333 to 0.6666666666666666.

### What is the cycle-to-FM gradient norm ratio?

The final `gradient_norm_ratio_cycle_to_fm_mean` is 21.025898896541765. Across
epochs it ranges from 16.855739922724474 to 37.1076249014568.

### Is this enough to decide the next optimization method?

Not by itself. The conflict fraction is above 0.3, and the norm ratio is very
large, so the run shows both direction conflict and magnitude imbalance between
cycle and flow-matching gradients.

However, this is one fixed16 debug experiment. It is enough to justify a tighter
follow-up test, but not enough to directly implement PCGrad, GradNorm, CAGrad,
UW/uncertainty weighting, FAMO, or another optimizer change as the next
committed method. The next decision should be based on a cycle-step ablation or
at least a reproduction run.

### What can this experiment not prove?

- It cannot prove EMA is permanently ineffective.
- It cannot prove FID, KID, or IQA improved.
- It cannot prove privacy improved.
- It cannot prove the best gradient-balancing method.
- It cannot prove fixed16 is better than other cycle-step settings.
- It cannot prove the result is reproducible across seeds or repeated runs.
- It cannot prove behavior on larger splits or longer training.

## Current Decision

Use the raw checkpoint as the current Phase B baseline for this run. Treat EMA as
lagging under `ema.decay: 0.999` in a short 5-epoch debug run, not as ruled out.

Treat the gradient diagnostics as a warning signal. The run supports doing a
cycle-step ablation or reproducing the fixed16 run before adding a gradient
balancing method.
