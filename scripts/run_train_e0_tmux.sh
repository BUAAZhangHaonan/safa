#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
tmux new-session -d -s train_e0 "python -m safa.cli.train_e0 --config configs/train_e0.yaml 2>&1 | tee artifacts/logs/train_e0.log"
tmux attach -t train_e0

