# 4029 Minimal Validation Runbook

Repository path:

```bash
/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization
```

Environment rules:

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHON_BIN=/home/hdd3/zhanghaonan/anaconda3/bin/python
export HTTP_PROXY=http://<proxy-host>:<proxy-port>
export HTTPS_PROXY=http://<proxy-host>:<proxy-port>
```

Use only GPUs `0,1,2,3`. Start with GPU `0`. Do not fall back to CPU.

## Setup

```bash
cd /home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization
$PYTHON_BIN -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
$PYTHON_BIN -m pip install --no-cache-dir -e ".[test]"
$PYTHON_BIN -m pip install --no-cache-dir insightface onnxruntime-gpu
$PYTHON_BIN -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
```

If ImageNet, InsightFace, FaceNet, or AdaFace weights cannot be downloaded, stop and provide local checkpoint paths. Do not replace them with random weights.

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
```

Use a validation cache config before evaluation:

```bash
python -m safa.cli.cache_e0 --config configs/cache_e0.yaml
```

```bash
scripts/run_smoke_tmux.sh
scripts/run_train_g_tmux.sh
scripts/run_eval.sh
```

## Required Checks

- `pytest` or `python -m unittest discover tests` passes in the remote environment.
- `artifacts/checkpoints/e0/manifest.json` reports `passes_majority_baseline=true`.
- `artifacts/smoke/smoke_result.json` exists after smoke.
- `artifacts/eval/g_val.json` exists after evaluation.
- Privacy recognizers are not imported or used by training commands.
