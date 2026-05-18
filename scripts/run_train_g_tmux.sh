#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${SAFA_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HTTP_PROXY="${HTTP_PROXY:-}"
export HTTPS_PROXY="${HTTPS_PROXY:-}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python}"
MAX_RAM_FRACTION="${MAX_RAM_FRACTION:-0.90}"
mkdir -p artifacts/logs
RUN_ENV="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS HTTP_PROXY=$HTTP_PROXY HTTPS_PROXY=$HTTPS_PROXY PYTHONPATH=$PYTHONPATH"
tmux new-session -d -s train_g "$RUN_ENV $PYTHON_BIN scripts/guarded_run.py --max-ram-fraction $MAX_RAM_FRACTION -- $PYTHON_BIN -m torch.distributed.run --standalone --nproc_per_node=4 -m safa.cli.train_g --config configs/train_g.yaml 2>&1 | tee artifacts/logs/train_g.log"
echo "Started tmux session train_g. Log: artifacts/logs/train_g.log"
if [[ "${ATTACH:-0}" == "1" ]]; then
  tmux attach -t train_g
fi
