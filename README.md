# Samplewise Affective Face Anonymization

Minimal validation code for a paper idea: generate an anonymized face from a frozen affective embedding while preserving the same frozen encoder representation sample by sample.

The first validation is intentionally small. It checks whether the full chain runs:

1. Build a strict AffectNet image index.
2. Train an AffectNet emotion encoder `E0`.
3. Freeze `E0` and cache 512-dimensional L2-normalized embeddings.
4. Train a generator `G(z) -> x_hat` where `z = E0(x)` is the only input.
5. Evaluate affective preservation, empirical unlinkability, and anti-steganography perturbations.

## Core Rules

- `E0` is a ResNet-50 emotion encoder trained on 8 AffectNet classes.
- `E0` outputs a 512-dimensional L2-normalized embedding.
- `G` receives only `z`; no image, identity feature, landmark, pose, background, or noise input is allowed.
- Identity recognizers are never used during training.
- Missing files, invalid labels, missing checkpoints, CPU-only execution, and NaNs are hard errors.

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

The Python config still uses `device: cuda:0`; the scripts map that visible device to the selected physical GPU through `CUDA_VISIBLE_DEVICES`.
