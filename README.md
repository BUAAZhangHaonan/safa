# Samplewise Affective Face Anonymization

Minimal validation code for a paper idea: generate an anonymized face from a frozen affective embedding while preserving the same frozen encoder representation sample by sample.

The first validation is intentionally small. It checks whether the full chain runs:

1. Build a strict AffectNet image index.
2. Train an AffectNet emotion encoder `E0`.
3. Freeze `E0` and cache 512-dimensional L2-normalized embeddings.
4. Train a conditional flow matching generator `G(z) -> x_hat` where `z = E0(x)` is the only public input.
5. Evaluate affective preservation and ArcFace face-detection readiness before any privacy evaluation.

## Core Rules

- `E0` is a ResNet-50 emotion encoder trained on 8 AffectNet classes.
- `E0` outputs a 512-dimensional L2-normalized embedding.
- `G` receives only `z`; no image, identity feature, landmark, pose, background, or external noise input is allowed.
- Identity recognizers are never used during training.
- Privacy evaluation is guarded by ArcFace detection rate and latent cosine thresholds.
- Missing files, invalid labels, missing checkpoints, CPU-only execution, and NaNs are hard errors.

## Metric Semantics

- `face_detect_ge1_rate` is the count >= 1 rate.
- `single_face_eq1_rate` is the count == 1 rate.
- `zero_face_rate` is the count == 0 rate.
- `multi_face_rate` is the count > 1 rate.
- Legacy `face_detection_rate` and eval `face_detection.detected.mean` remain ge1 metrics for old report compatibility.
- New checkpoint composite uses `validation_*_latent_cosine_mean * validation_*_single_face_eq1_rate`. Old reports used `face_detection_rate`/ge1.
- `lambda_cycle` is a legacy compatibility field. Read `effective_cycle_loss_weight` for the actual cycle loss weight, especially for uncertainty-weighted runs.
- `stage_epoch` remains the legacy 0-based epoch. New metrics also write `stage_epoch_0based` and `stage_epoch_1based`.
- Quality metrics prefer explicit fields such as `quality_raw_niqe_mean`, `quality_raw_niqe_std`, `quality_raw_fid`, `quality_raw_kid_mean`, and `quality_raw_kid_std`; older names remain aliases only.

## Main Commands

```bash
python -m safa.cli.build_index --root /home/hdd3/zhanghaonan/AffectNet --out data/index/train.jsonl
python -m safa.cli.train_e0 --config configs/train_e0.yaml
python -m safa.cli.cache_e0 --config configs/cache_e0.yaml
python -m safa.cli.cache_e0 --config configs/cache_e0_val.yaml
python -m safa.cli.train_g --config configs/train_g.yaml
python -m safa.cli.eval --config configs/eval.yaml
python -m safa.cli.smoke --config configs/smoke.yaml
```

Long runs should be launched through the scripts in `scripts/`. They start `tmux` sessions, set `CUDA_VISIBLE_DEVICES`, and run through a Linux RAM guard that stops the job if server memory reaches 90%.

Default GPU assignment on 4029:

- `train_e0`: physical GPU 0.
- `cache_e0` train split: physical GPU 1.
- `cache_e0` val split: physical GPU 2.
- `smoke`: physical GPU 3.
- `train_g`: physical GPU 1.
- `eval_safa`: physical GPU 2.

The Python config still uses `device: cuda:0`; the scripts map that visible device to the selected physical GPU through `SAFA_CUDA_VISIBLE_DEVICES`.
