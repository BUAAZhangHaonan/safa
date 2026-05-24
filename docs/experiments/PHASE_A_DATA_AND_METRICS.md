# Phase A Data And Metrics Outputs

Date: 2026-05-24

## Completed Scope

Phase A produced the balanced train indexes, the single-face validation index,
the balanced debug E0 feature cache, and the metric plumbing needed to avoid
selecting checkpoints only by any-face detection rate.

This document records generated artifacts and their current limits. It does not
claim any new training, evaluation, or cache run beyond the existing Phase A
outputs listed here.

## Balanced Train Indexes

Debug split:

- Index: `data/index/train_balanced_debug.jsonl`
- Manifest: `data/index/train_balanced_debug_manifest.json`
- Rows: 8000
- Class counts: 1000 samples per class for labels 0 through 7
- Seed: 1337
- Output SHA256: `3cc061f9d5f4ceded00ae1ce3e8897a1b210d4d93f05b49f8c5f06f831fb1068`

Medium split:

- Index: `data/index/train_balanced_medium.jsonl`
- Manifest: `data/index/train_balanced_medium_manifest.json`
- Rows: 30000
- Class counts: 3750 samples per class for labels 0 through 7
- Seed: 1337
- Output SHA256: `bcdab61f555ffae532646e0f0402537070149fb0e775ef30d7ceccdd53924544`

## Single-Face Validation Index

- Source index: `data/index/val.jsonl`
- Output index: `data/index/val_single_face.jsonl`
- Manifest: `data/index/val_single_face_manifest.json`
- Detector: `insightface_buffalo_l`
- Device recorded by manifest: `cuda:0`
- Source rows: 4000
- Single-face rows: 3969
- Zero-face rows: 2
- Multi-face rows: 29
- Output SHA256: `da14e23eacefecbc2948d1374fb93961a13d017a9183aa1fe2a2f62b33a4b4ea`

## Balanced Debug Feature Cache

- Cache directory: `artifacts/e0_features/train_balanced_debug`
- Manifest: `artifacts/e0_features/train_balanced_debug/manifest.json`
- Shard: `artifacts/e0_features/train_balanced_debug/features.pt`
- Index path: `data/index/train_balanced_debug.jsonl`
- Encoder checkpoint: `artifacts/checkpoints/e0/best.pt`
- Samples: 8000
- Feature dimension: 512
- Dtype: `float32`
- L2 normalized: true
- Class counts: 1000 samples per class for labels 0 through 7
- Shard SHA256: `cc0f3d312dbb26263089fbe0762e6646889ce8a7f250992c3e2cb77285b6fe67`

The cache manifest and shard were verified with
`safa.data.feature_cache.load_feature_cache` on CPU. The cache files remain
ignored by Git because `artifacts/` and `*.pt` are ignored and the repository has
no tracked `artifacts/e0_features` convention.

## Metrics And Quality Scripts

- Single-face metrics are implemented: `face_detect_ge1_rate`,
  `single_face_eq1_rate`, `zero_face_rate`, and `multi_face_rate`.
- Evaluation summaries and face-detection guards include those single-face rate
  fields.
- Checkpoint composite selection uses `validation_latent_cosine_mean *
  validation_single_face_eq1_rate`, not the legacy any-face rate.
- FID/KID/IQA dependencies are installed.
- The quality script is implemented at `scripts/eval_generation_quality.py` for
  FID, KID, and pyIQA no-reference IQA.
- Real `generation_quality` metrics have not been run yet.

## Stale Artifacts

Old full train/val feature caches and old checkpoints were produced under the
previous schema or previous metric policy. They should be rebuilt or rerun before
being used as current Phase A evidence.

## What Phase A Does Not Prove

- It does not prove EMA stability.
- It does not prove gradient conflict.
- It does not prove a privacy pass.
