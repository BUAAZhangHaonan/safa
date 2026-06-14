# 4029 Minimal Validation Runbook

Repository path:

```bash
/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization
```

Environment rules:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CONDA_BIN=/home/hdd3/zhanghaonan/anaconda3/bin/conda
export PYTHON_BIN=/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python
export HTTP_PROXY=http://<proxy-host>:<proxy-port>
export HTTPS_PROXY=http://<proxy-host>:<proxy-port>
export MAX_RAM_FRACTION=0.90
```

Use only physical GPUs `0,1,2,3`. Do not fall back to CPU. GPU memory is allowed to fill until OOM, but server RAM must stay below 90%.

Default physical GPU assignment:

- `scripts/run_train_e0_tmux.sh`: GPU 0.
- `scripts/run_cache_e0.sh`: GPU 1.
- `scripts/run_cache_e0_val.sh`: GPU 2.
- `scripts/run_smoke_tmux.sh`: GPU 3.
- `scripts/run_train_g_tmux.sh`: GPU 1.
- `scripts/run_eval.sh`: GPU 2.

The configs keep `device: cuda:0`. Each script maps `cuda:0` to the physical GPU through `SAFA_CUDA_VISIBLE_DEVICES`. This intentionally ignores any inherited `CUDA_VISIBLE_DEVICES` from the login shell.

## Setup

```bash
cd /home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization
$CONDA_BIN create -y -n safa python=3.12
$PYTHON_BIN -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
$PYTHON_BIN -m pip install --no-cache-dir -e ".[test,privacy,export,quality]"
$PYTHON_BIN -m pip check
```

If ImageNet, InsightFace, FaceNet, or AdaFace weights cannot be downloaded, stop and provide local checkpoint paths. Do not replace them with random weights.

Evaluation-only privacy recognizer assets:

```bash
export HTTP_PROXY=http://<proxy-host>:<proxy-port>
export HTTPS_PROXY=http://<proxy-host>:<proxy-port>
export HF_HUB_DISABLE_XET=1
PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 $PYTHON_BIN scripts/export_privacy_recognizers.py --out-dir artifacts/privacy --which adaface
```

The `safa` runtime environment intentionally does not install `facenet-pytorch`; its package metadata requires older torch and torchvision versions. Runtime evaluation uses exported TorchScript recognizers only. Export FaceNet in the separate `facenet` conda environment:

```bash
export FACENET_PYTHON_BIN=/home/hdd3/zhanghaonan/anaconda3/envs/facenet/bin/python

$CONDA_BIN create -y -n facenet python=3.10
$FACENET_PYTHON_BIN -m pip install --no-cache-dir \
  torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121
$FACENET_PYTHON_BIN -m pip install --no-cache-dir \
  facenet-pytorch==2.6.0 \
  numpy==1.26.4 \
  Pillow==10.2.0
$FACENET_PYTHON_BIN -m pip check
PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 $FACENET_PYTHON_BIN scripts/export_privacy_recognizers.py --out-dir artifacts/privacy --which facenet
test -s artifacts/privacy/facenet.pt
```

After export, verify the TorchScript asset through the `safa` runtime before running privacy evaluation.

## Data Repair

The received AffectNet `training.csv` may contain one known row pointing to a missing image. Repair it explicitly and keep the audit artifact:

```bash
$PYTHON_BIN scripts/repair_affectnet_missing_row.py \
  --csv /home/hdd3/zhanghaonan/AffectNet/training.csv \
  --out-dir artifacts/data_fixes
```

## Milestone Commands

```bash
$PYTHON_BIN -m safa.cli.build_index --root /home/hdd3/zhanghaonan/AffectNet --out data/index/train.jsonl --split train --only-split train --label-policy affectnet8 --csv-image-prefix Manually_Annotated_Images
$PYTHON_BIN -m safa.cli.build_index --root /home/hdd3/zhanghaonan/AffectNet --out data/index/val.jsonl --split val --only-split val --label-policy affectnet8 --csv-image-prefix Manually_Annotated_Images
```

```bash
scripts/run_train_e0_tmux.sh
```

```bash
scripts/run_cache_e0.sh
scripts/run_cache_e0_val.sh
```

Phase A cache rule: caches created before the current Phase A index/filtering flow are stale. Rebuild both train and val E0 feature caches after changing the Phase A index, the single-face filter, or the E0 checkpoint. Do not reuse old cache directories with new indexes.

```bash
scripts/run_smoke_tmux.sh
scripts/run_train_g_tmux.sh
scripts/run_eval.sh
```

Run generation quality as a fail-fast check after evaluation samples exist:

```bash
PYTHONPATH=src $PYTHON_BIN scripts/eval_generation_quality.py \
  --real-index data/index/val.jsonl \
  --generated-dir artifacts/eval/samples \
  --output artifacts/eval/generation_quality.json
```

If dependencies from the `quality` extra are missing, the generated sample directory is empty, or any metric cannot be computed, the command exits nonzero. Stop and fix that cause; do not treat quality evaluation failure as a warning.

The `tmux` scripts start in the background and print the log path. Set `ATTACH=1` only when an interactive terminal should attach to the session.

## Required Checks

- `pytest` or `python -m unittest discover tests` passes in the remote environment.
- `artifacts/checkpoints/e0/manifest.json` reports `passes_majority_baseline=true`.
- `artifacts/smoke/smoke_result.json` exists after smoke.
- `artifacts/eval/g_val.json` and `artifacts/eval/per_sample.jsonl` exist after evaluation.
- `artifacts/eval/generation_quality.json` exists after the fail-fast generation quality command.
- Privacy recognizers are not imported or used by training commands.
- FaceNet and AdaFace TorchScript checkpoints exist at the configured paths before evaluation. If they are missing, evaluation must stop.
- If ArcFace detects zero faces on generated images, report this as a generator image-quality failure. Do not bypass detection or add post-processing.
