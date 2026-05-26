# MEDIUM V1 Stage1 Experiments

Date: 2026-05-26

## Current Decision

The completed Stage1 long run to use as the medium_v1 Stage2 starting point is
`g_medium_v1_stage1_long200_v4`.

Use this run with a condition: the Stage1 single-face gate is stable and the
image distribution metrics improved a lot, but the final FID is still above the
15-35 target band. Stage2 can start from the best-FID/latest Stage1 checkpoint
as M0, but the generated images should not be treated as final quality.

## Run Scope

This experiment changed only the Stage1 training horizon and quality monitoring
around the existing medium_v1 setup:

- Config: `configs/medium_v1/train_g_medium_v1_stage1_long200_v4.yaml`
- Output checkpoint dir: `artifacts/checkpoints/g_medium_v1_stage1_long200_v4`
- Quality output dir: `artifacts/eval/g_medium_v1_stage1_long200_v4/quality`
- Plot JSON: `artifacts/plots/medium_v1/stage1_long200_v4_metrics_timeseries.json`
- Training logs:
  `artifacts/logs/train_g_medium_v1_stage1_long200_v4_gpu3_6.log`,
  `artifacts/logs/train_g_medium_v1_stage1_long200_v4_resume52_gpu3_6.log`,
  `artifacts/logs/train_g_medium_v1_stage1_long200_v4_resume172_to200_gpu3_6.log`
- Plot log: `artifacts/logs/plot_stage1_long200_v4_curves.log`
- GPU range used by the run scripts/logs: GPU3-6
- Stage1 length: 200 epochs
- NIQE cadence: every epoch
- FID/KID cadence: every 20 epochs

The run was resumed after interruptions. The visible recovery path is the initial
long200_v4 run, a resume after the quality DataLoader file-descriptor leak around
epoch 53, and a final resume from epoch 172 to epoch 200. The failed quality
artifact kept for that incident is
`artifacts/eval/g_medium_v1_stage1_long200_v4/quality/epoch_0053.failed_fd_leak_20260526_024901`.
The final metrics record `stage_epoch_1based: 200` in
`artifacts/checkpoints/g_medium_v1_stage1_long200_v4/last_metrics.json`.

## What Stayed Fixed

- E0-medium stayed fixed at `artifacts/checkpoints/e0_medium_v1/best.pt`.
- The balanced-medium training split stayed fixed at `data/index/train_balanced_medium.jsonl`.
- The validation split stayed fixed at `data/index/val_single_face.jsonl`.
- The seed and sampling seed stayed fixed at `1337`; the manifest records
  `sampling.base_seed: 1337` and `sampling.stable_x_init: true`.
- The generator stayed `conditional_flow_matching` with `base_channels: 32`,
  `channel_multipliers: [1, 2, 4, 4]`, `condition_dim: 512`, Heun sampling,
  and 32 sample steps.
- Stage1 stayed a no-cycle training target in the executed metrics:
  `cycle_loss_raw`, `cycle_loss_normalized`, and `effective_cycle_loss_weight`
  are `0.0` in the final record.
- Stage2 was not run in this experiment; the config has `stage2.epochs: 0`.

## Key Fixes Needed For This Run

- FID/KID were moved outside training DDP. The local commit is
  `9c4e476 fix: run training fid kid outside ddp`.
- Plot refresh became strict, so missing Stage1 quality plot inputs fail instead
  of silently producing partial output. The local commit is
  `252f157 fix: fail fast on missing stage1 quality plots`.
- The quality DataLoader file-descriptor leak was fixed, allowing the epoch-53
  recovery path. The local commit is
  `9cc3c88 fix: resume stage1 after quality fd leak`.
- Resume handling was fixed so Stage1 epoch numbering continued correctly after
  resume. The final metrics include `stage_epoch: 199`, `stage_epoch_0based: 199`,
  and `stage_epoch_1based: 200`.
- `stable_epochs` was adjusted to `29` in the long200_v4 config so the gate did
  not stop the run before epoch 200. The local commit is
  `df7533c configs: finish stage1 long200 after gate`.

## Quality Metrics

FID and KID use 3969 generated images and 3969 real single-face validation
images. NIQE uses 512 generated images per epoch.

| Epoch | FID | KID mean | KID std | NIQE mean | NIQE std |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 88.638969 | 0.079384 | 0.009279 | 5.973798 | 1.205115 |
| 40 | 77.769257 | 0.061926 | 0.008339 | 5.873528 | 1.035605 |
| 60 | 72.207748 | 0.063197 | 0.009428 | 6.172364 | 1.296557 |
| 80 | 75.867142 | 0.063737 | 0.009755 | 5.298255 | 1.100471 |
| 100 | 85.779221 | 0.077884 | 0.010812 | 4.630718 | 0.824113 |
| 120 | 80.820648 | 0.068689 | 0.009837 | 4.383054 | 1.061292 |
| 140 | 64.225830 | 0.053980 | 0.007537 | 5.628162 | 1.351609 |
| 160 | 89.180779 | 0.080501 | 0.009739 | 4.906436 | 0.921413 |
| 180 | 55.054893 | 0.045338 | 0.007130 | 6.162786 | 1.381737 |
| 200 | 49.216141 | 0.035547 | 0.006894 | 6.109232 | 1.331761 |

The best FID and best KID in this table are both epoch 200:
`FID=49.21614074707031`, `KID mean=0.03554704040288925`.
NIQE is noisy and does not track FID/KID monotonically; the best NIQE in the full
per-epoch series is epoch 2, but FID/KID were not run at that epoch.

## Face Stability And Training Trend

Single-face stability is strong by the end of Stage1. Over epochs 101-200, the
average `face_detect_ge1` is 99.72%, the average `single_face_eq1` is 99.70%,
the average multi-face rate is 0.02%, and the average zero-face rate is 0.28%.
Over epochs 172-200, the multi-face rate is 0.00% on average and the zero-face
rate is about 0.25%.

The training loss and gradient trend are stable rather than explosive. In the
plot JSON, loss moves from `0.06713453485667706` at epoch 1 to
`0.05814494377523661` at epoch 200. Grad norm moves from
`0.19816504304011662` to `0.07980940988858541` over the same span.

## Raw Versus EMA

This run cannot answer whether EMA is better. The manifest records:

- `ema_config.enabled: false`
- `ema_config.evaluate_raw: true`
- `ema_config.evaluate_ema: false`
- `best_model: raw`

So all metrics here are raw-model metrics. There is no EMA comparison artifact
for this run.

## Latent Cosine

Latent cosine is not the main Stage1 target in this run. It is useful as a
secondary diagnostic, but this Stage1 run is judged mainly by single-face
stability and generated-image distribution quality. The epoch-200 latent cosine
is `0.6387721505016088`, but it should not be used alone to select the Stage1
checkpoint.

## Checkpoint Recommendation

Based on the metrics JSON, epoch 200 is the best FID/KID point. The checkpoint
folder contains these current top-level checkpoint files:

- `artifacts/checkpoints/g_medium_v1_stage1_long200_v4/best.pt`
- `artifacts/checkpoints/g_medium_v1_stage1_long200_v4/best_stage1.pt`
- `artifacts/checkpoints/g_medium_v1_stage1_long200_v4/last.pt`
- `artifacts/checkpoints/g_medium_v1_stage1_long200_v4/best_single_face.pt`
- historical `best_single_face_epoch_*.pt` files

There is no separate top-level `epoch_0200.pt` file. For Stage2 M0, use the
best-FID Stage1 checkpoint from this directory, with `best.pt` or
`best_stage1.pt` as the intended artifact names. Use `last.pt` only when the
exact latest/resume checkpoint is needed.

## Stage2 Recommendation

Stage2 is reasonable as a conditional next step, not as a claim that Stage1 image
quality is solved. The condition is that Stage2 should use the epoch-200
best-FID/latest Stage1 state as M0 and should keep quality metrics enabled. The
reason is simple: Stage1 single-face stability is already high, and distribution
quality improved from `FID=88.63896942138672` at epoch 20 to
`FID=49.21614074707031` at epoch 200, but FID is still above the 15-35 target
band.

## What This Experiment Cannot Prove

- It cannot prove any privacy result. No privacy TAR/EER/AUC result is recorded
  here, and Stage2 was not completed.
- It cannot prove EMA is better, because EMA was disabled.
- It cannot prove the final anonymization quality of the full SAFA pipeline,
  because this is Stage1 only.
- It cannot prove that latent cosine alone is a good checkpoint selector.
- It cannot prove that FID will enter the 15-35 target band without Stage2 or a
  further generator-quality change.
