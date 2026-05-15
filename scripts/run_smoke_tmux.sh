#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p artifacts/logs
tmux new-session -d -s smoke_safa "python -m safa.cli.smoke --config configs/smoke.yaml 2>&1 | tee artifacts/logs/smoke.log"
tmux attach -t smoke_safa

