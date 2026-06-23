#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${PYTHONPATH:-src}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT_DETECTED="$(nvidia-smi -L | grep -c '^GPU ' || true)"
    if [[ "${GPU_COUNT_DETECTED}" -gt 0 ]]; then
      CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT_DETECTED - 1)))"
      export CUDA_VISIBLE_DEVICES
    fi
  fi
fi

GPU_COUNT="${MEANFLOW_GPU_COUNT:-}"
if [[ -z "$GPU_COUNT" ]]; then
  GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)"
fi
if [[ "$GPU_COUNT" -lt 1 ]]; then
  GPU_COUNT=1
fi

case "$GPU_COUNT" in
  1)
    DEFAULT_PER=384
    DEFAULT_GLOBAL=384
    ;;
  2)
    DEFAULT_PER=384
    DEFAULT_GLOBAL=768
    ;;
  3)
    DEFAULT_PER=256
    DEFAULT_GLOBAL=768
    ;;
  *)
    DEFAULT_PER=256
    DEFAULT_GLOBAL=$((256 * GPU_COUNT))
    ;;
esac

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-$DEFAULT_PER}"
GLOBAL_BATCH="${GLOBAL_BATCH:-$DEFAULT_GLOBAL}"
RUNTIME_CONFIG="${MEANFLOW_RUNTIME_CONFIG:-configs/medium_v2/experiments/e15_meanflow_sit_b_face_mixed_resume_h100_runtime.yaml}"

python scripts/h100/prepare_h100_bundle.py \
  --bundle-root "$ROOT" \
  --runtime-config "$RUNTIME_CONFIG" \
  --gpu-count "$GPU_COUNT" \
  --per-device-batch "$PER_DEVICE_BATCH" \
  --global-batch "$GLOBAL_BATCH" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p artifacts/logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="artifacts/logs/e15_meanflow_h100_resume_${GPU_COUNT}gpu_${GLOBAL_BATCH}gb_${STAMP}.log"
echo "runtime_config=$RUNTIME_CONFIG"
echo "gpu_count=$GPU_COUNT per_device_batch=$PER_DEVICE_BATCH global_batch=$GLOBAL_BATCH"
echo "log=$LOG"

if [[ "$GPU_COUNT" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$GPU_COUNT" \
    -m safa.cli.train_g \
    --config "$RUNTIME_CONFIG" 2>&1 | tee "$LOG"
else
  python -m safa.cli.train_g \
    --config "$RUNTIME_CONFIG" 2>&1 | tee "$LOG"
fi
