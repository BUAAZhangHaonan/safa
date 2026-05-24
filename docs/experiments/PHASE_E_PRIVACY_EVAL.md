# Phase E Privacy Eval

Date: 2026-05-25

## Completed Scope

Phase E has protocol code for privacy ROC metrics, but the full privacy metrics
were not run for the current fixed16 raw single-face artifact. The evaluation
generated images and computed the pre-privacy guard metrics, then exited with
code 1 because the fail-fast privacy guard did not pass.

This document records only the artifacts and metrics that already exist. It does
not claim a privacy pass, and it does not record any new training or evaluation
started for this document update.

## Protocol Status

- Privacy ROC code exists in `src/safa/evaluation/runner.py`.
- Privacy recognizer export code exists in
  `scripts/export_privacy_recognizers.py`.
- Full privacy metrics are blocked for the current artifact.
- The block is expected behavior: the guard requires
  `latent_cosine_mean >= 0.95`, but the current result is
  `0.8763568043956591`.
- `privacy_skipped` is `true` in the result JSON.
- `metrics.privacy` is empty in the result JSON.

## Evaluation Artifact

- Result JSON:
  `artifacts/eval/stability_balanced_debug_monitor10_rawbest_fixed16_single_face/result.json`
- Per-sample rows:
  `artifacts/eval/stability_balanced_debug_monitor10_rawbest_fixed16_single_face/per_sample.jsonl`
- Generated images:
  `artifacts/eval/stability_balanced_debug_monitor10_rawbest_fixed16_single_face/generated_images`
- Generated PNG count: 3969
- Dataset index: `data/index/val_single_face.jsonl`
- Dataset samples: 3969
- Feature cache: `artifacts/e0_features/val_single_face`
- E0 checkpoint: `artifacts/checkpoints/e0/best.pt`
- G checkpoint:
  `artifacts/checkpoints/stability_balanced_debug_monitor10_rawbest_fixed16/best.pt`
- Sampling seed: 1337
- Stable x init: true

## Guard Metrics Already Obtained

| Metric | Value |
| --- | ---: |
| face_detect_ge1_rate | 0.998740236835475 |
| single_face_eq1_rate | 0.9972285210380448 |
| multi_face_rate | 0.0015117157974300832 |
| zero_face_rate | 0.0012597631645250692 |
| latent_cosine_mean | 0.8763568043956591 |
| latent_cosine_threshold | 0.95 |
| label_accuracy_generated | 0.5152431342907533 |

The face-detection side of the guard is high, but the latent-cosine side fails.
Because `latent_cosine_mean` is below the `0.95` threshold, full privacy metrics
were skipped instead of being reported.

## Generation Quality Already Obtained

Quality metrics were computed for the same fixed16 raw single-face artifact:

- Quality JSON:
  `artifacts/eval/stability_balanced_debug_monitor10_rawbest_fixed16_single_face/generation_quality.json`
- Real images: 3969
- Generated images: 3969
- FID: 124.33562469482422
- KID mean: 0.12673257291316986
- KID std: 0.011795842088758945
- NIQE mean: 4.491209701214922
- NIQE std: 0.6980395217797578

These FID, KID, and NIQE numbers compare 3969 generated single PNG files against
3969 real single-face validation images. They are not paired PSNR or SSIM.

## Current Decision

Mark Phase E privacy evaluation as blocked for this artifact, not passed. The
next privacy run should only report full privacy ROC metrics after the fail-fast
guard passes.

## What Phase E Does Not Prove

- It does not prove privacy improvement.
- It does not prove a privacy pass.
- It does not provide full privacy ROC metrics for this artifact.
- It does not compare privacy recognizers, because the recognizer pass was
  skipped by the guard.
