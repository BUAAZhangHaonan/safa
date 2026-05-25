# MEDIUM V1 E0 Training

Date: 2026-05-25

## Current Decision

E0-medium training completed and produced the checkpoint intended for the
medium_v1 generator configs:
`artifacts/checkpoints/e0_medium_v1/best.pt`.

This is not a clean same-protocol proof that E0-medium is better than the old
E0. E0-medium was validated on `data/index/val_single_face.jsonl` with 3969
samples. The existing old E0 checkpoint metrics were validated on
`data/index/val.jsonl` with 4000 samples, and they do not include
`balanced_accuracy` or `macro_f1`. The old E0 numbers are reference numbers only.

On raw accuracy alone, E0-medium best is higher than the old E0 reference:
`0.563114134542706` versus `0.555`. That is a +0.008114134542706 absolute
difference, but it is not an equivalent validation comparison.

## Artifacts Checked

| Item | Path or value |
| --- | --- |
| E0-medium config | `configs/medium_v1/train_e0_medium_v1.yaml` |
| E0-medium best checkpoint | `artifacts/checkpoints/e0_medium_v1/best.pt` |
| E0-medium best sha256 | `d7d2c57a552155776b8c15a4e52e43ec5082fc046aa0aabb4e9709685f7e3d1a` |
| E0-medium manifest | `artifacts/checkpoints/e0_medium_v1/manifest.json` |
| E0-medium last metrics | `artifacts/checkpoints/e0_medium_v1/last_metrics.json` |
| E0-medium log | `artifacts/logs/train_e0_medium_v1.log` |
| Old E0 best checkpoint | `artifacts/checkpoints/e0/best.pt` |
| Old E0 best sha256 | `5f165c520fad315dd1550676c6515c3480585e8ea0dcf1841fd678c8f1963e0f` |
| Old E0 manifest | `artifacts/checkpoints/e0/manifest.json` |
| Old E0 last metrics | `artifacts/checkpoints/e0/last_metrics.json` |
| Single-face validation index | `data/index/val_single_face.jsonl`, 3969 rows |
| Full validation index | `data/index/val.jsonl`, 4000 rows |

The E0-medium best and last checkpoint files were also opened with the project
Python environment. Their checkpoint payloads contain `metrics`, `model_config`,
and `model_state_dict`. The metrics in the checkpoint match the manifest and
`last_metrics.json` values below.

## E0 Metrics

| Run | Validation protocol | Samples | Best epoch | Best accuracy | Best balanced accuracy | Best macro F1 | Last epoch | Last accuracy | Last balanced accuracy | Last macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0-medium | `val_single_face` | 3969 | 7 | 0.563114134542706 | 0.563122977172707 | 0.5623512571187093 | 22 | 0.5416981607457798 | 0.5419439260388013 | 0.5389591434080089 |
| Old E0 reference | `val` | 4000 | 9 | 0.555 | n/a | n/a | 9 | 0.555 | n/a | n/a |

E0-medium stopped by early stopping at epoch 22:
`no improvement for 15 epochs`. The best checkpoint stayed epoch 7.

The old E0 reference has `num_samples=4000` in both its manifest and checkpoint
metrics. That matches the full validation index, not the single-face validation
index. Previous docs also report the old E0 as `accuracy=55.5%, epoch 9`.

## What Changed

- The E0-medium train index is `data/index/train_balanced_medium.jsonl`, with
  30000 rows, instead of the full `data/index/train.jsonl`.
- The validation index is fixed to `data/index/val_single_face.jsonl`, with
  3969 rows.
- The output directory changed to `artifacts/checkpoints/e0_medium_v1`.
- The current medium config uses `epochs: 60`, `warmup_epochs: 5`,
  `early_stopping_patience: 15`, and `class_weight: false`.

## What Stayed Fixed

- Seed: `1337`.
- Image size: `224`.
- Number of classes: `8`.
- Embedding dimension: `512`.
- Learning rate: `0.0003`.
- Weight decay: `0.0001`.
- Label smoothing: `0.1`.
- Augmentation: `strong`.
- ImageNet initialization: `IMAGENET1K_V2`.
- E0 embeddings remain L2-normalized.

## Required Questions

### 1. What changed?

The E0 training data, validation protocol, output directory, class weighting,
warmup length, max epoch count, and early-stopping patience changed for
medium_v1. The main protocol change is that validation is now single-face only.

### 2. What stayed fixed?

The E0 architecture contract stayed fixed: 8 affective classes, 512-dimensional
L2-normalized embeddings, ImageNet ResNet initialization, strong augmentation,
label smoothing, seed, learning rate, and weight decay.

### 3. Which E0 checkpoint was used?

Use `artifacts/checkpoints/e0_medium_v1/best.pt` for medium_v1 work. The old E0
reference checkpoint is `artifacts/checkpoints/e0/best.pt`.

### 4. Which G checkpoint was used?

No G checkpoint was used for this E0 training record. The medium_v1 G configs
point to the E0-medium checkpoint, but this document does not run or evaluate G.

### 5. Raw or EMA?

Not applicable for E0. The E0 checkpoints contain model weights and metrics, not
raw-versus-EMA variants.

### 6. FID, KID, and NIQE?

Not applicable for E0. These are generated-image quality metrics and require a G
evaluation output. This document records only E0 classifier training.

### 7. Single-face protocol?

E0-medium uses the fixed single-face validation index:
`data/index/val_single_face.jsonl`. The single-face manifest reports 3969
single-face rows from the 4000-row full validation index, with 29 multi-face rows
and 2 zero-face rows removed.

The old E0 reference metrics are not from this protocol. They are from the
4000-row full validation index, so they are not directly equivalent.

### 8. Latent cosine?

Not applicable for E0. Latent cosine is a G evaluation metric comparing
generated images against source embeddings.

### 9. Privacy guard?

Not applicable for E0. Privacy guard is part of G evaluation before identity
privacy metrics. No privacy recognizer or guard pass was run here.

### 10. Why is this not a direct pass over old E0?

The direct comparison gate is missing old E0 metrics on
`data/index/val_single_face.jsonl`. The old E0 checkpoint only has existing
4000-sample validation metrics and no `balanced_accuracy` or `macro_f1`.

### 11. What can this not prove?

- It cannot prove E0-medium is better than old E0 under the exact same
  validation protocol.
- It cannot prove G quality, privacy, or latent preservation.
- It cannot prove FID, KID, NIQE, TAR, EER, AUC, or privacy ROC behavior.

### 12. What is the next step?

If a strict E0 comparison is needed, evaluate old E0 on
`data/index/val_single_face.jsonl` with the same metric set used for
E0-medium. For medium_v1 G work, use `artifacts/checkpoints/e0_medium_v1/best.pt`
as the E0 checkpoint because it is the completed single-face E0-medium result.

## Curve Status

Expected path: `artifacts/plots/medium_v1/e0_curves.png`.

The curve was not generated. `scripts/plot_medium_v1_curves.py --help` exposes
only the G curve interface:

- `--stage1-json`
- `--m0-json`
- `--m1-json`
- `--out-dir`

There is no E0 curve mode or E0 history JSON argument in the current script.
The available E0 artifacts provide best and last metrics, but not a structured
per-epoch history file suitable for this script. No placeholder or fake curve was
created.
