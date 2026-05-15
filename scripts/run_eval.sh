#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export HTTP_PROXY="${HTTP_PROXY:-http://<proxy-host>:<proxy-port>}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://<proxy-host>:<proxy-port>}"
PYTHON_BIN="${PYTHON_BIN:-/home/hdd3/zhanghaonan/anaconda3/bin/python}"
"$PYTHON_BIN" -m safa.cli.eval --config configs/eval.yaml
