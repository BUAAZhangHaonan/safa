#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export HTTP_PROXY="${HTTP_PROXY:-http://<proxy-host>:<proxy-port>}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://<proxy-host>:<proxy-port>}"
PYTHON_BIN="${PYTHON_BIN:-/home/hdd3/zhanghaonan/anaconda3/bin/python}"
MAX_RAM_FRACTION="${MAX_RAM_FRACTION:-0.90}"
mkdir -p artifacts/logs
tmux new-session -d -s cache_e0_val "$PYTHON_BIN scripts/guarded_run.py --max-ram-fraction $MAX_RAM_FRACTION -- $PYTHON_BIN -m safa.cli.cache_e0 --config configs/cache_e0_val.yaml 2>&1 | tee artifacts/logs/cache_e0_val.log"
echo "Started tmux session cache_e0_val. Log: artifacts/logs/cache_e0_val.log"
if [[ "${ATTACH:-0}" == "1" ]]; then
  tmux attach -t cache_e0_val
fi
