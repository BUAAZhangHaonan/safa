# 4029 Minimal Validation Runbook

Repository path:

```bash
/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization
```

Environment rules:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHON_BIN=/home/hdd3/zhanghaonan/anaconda3/bin/python
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

The configs keep `device: cuda:0`. Each script maps `cuda:0` to the physical GPU through `CUDA_VISIBLE_DEVICES`.

## Setup

```bash
cd /home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization
$PYTHON_BIN -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
$PYTHON_BIN -m pip install --no-cache-dir -e ".[test]"
$PYTHON_BIN -m pip install --no-cache-dir insightface onnxruntime-gpu
$PYTHON_BIN -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
```

If ImageNet, InsightFace, FaceNet, or AdaFace weights cannot be downloaded, stop and provide local checkpoint paths. Do not replace them with random weights.

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

```bash
scripts/run_smoke_tmux.sh
scripts/run_train_g_tmux.sh
scripts/run_eval.sh
```

The `tmux` scripts start in the background and print the log path. Set `ATTACH=1` only when an interactive terminal should attach to the session.

## Required Checks

- `pytest` or `python -m unittest discover tests` passes in the remote environment.
- `artifacts/checkpoints/e0/manifest.json` reports `passes_majority_baseline=true`.
- `artifacts/smoke/smoke_result.json` exists after smoke.
- `artifacts/eval/g_val.json` and `artifacts/eval/per_sample.jsonl` exist after evaluation.
- Privacy recognizers are not imported or used by training commands.
- FaceNet and AdaFace TorchScript checkpoints exist at the configured paths before evaluation. If they are missing, evaluation must stop.
