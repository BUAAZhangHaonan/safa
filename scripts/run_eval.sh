#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${SAFA_CUDA_VISIBLE_DEVICES:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export HTTP_PROXY="${HTTP_PROXY:-}"
export HTTPS_PROXY="${HTTPS_PROXY:-}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python}"
MAX_RAM_FRACTION="${MAX_RAM_FRACTION:-0.90}"
mkdir -p artifacts/logs
RUN_ENV="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS HTTP_PROXY=$HTTP_PROXY HTTPS_PROXY=$HTTPS_PROXY PYTHONPATH=$PYTHONPATH"
tmux new-session -d -s eval_safa "$RUN_ENV $PYTHON_BIN scripts/guarded_run.py --max-ram-fraction $MAX_RAM_FRACTION -- $PYTHON_BIN -m safa.cli.eval --config configs/eval.yaml 2>&1 | tee artifacts/logs/eval_safa.log"
echo "Started tmux session eval_safa. Log: artifacts/logs/eval_safa.log"
if [[ "${ATTACH:-0}" == "1" ]]; then
  tmux attach -t eval_safa
fi
