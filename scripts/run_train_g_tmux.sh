#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization}"
CONFIG="${CONFIG:-configs/train_g_v2_best.yaml}"
SESSION="${SESSION:-train_g_v2_best}"
LOG="${LOG:-artifacts/logs/train_g_v2_best.log}"

usage() {
  echo "Usage: $0 [--config PATH] [--log PATH] [--session NAME]" >&2
}

require_value() {
  local option="$1"
  if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
    echo "$option requires a value" >&2
    usage
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      require_value "$@"
      CONFIG="$2"
      shift 2
      ;;
    --log)
      require_value "$@"
      LOG="$2"
      shift 2
      ;;
    --session)
      require_value "$@"
      SESSION="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${SAFA_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HTTP_PROXY="${HTTP_PROXY:-}"
export HTTPS_PROXY="${HTTPS_PROXY:-}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python}"
mkdir -p "$(dirname "$LOG")"
RUN_ENV="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS HTTP_PROXY=$HTTP_PROXY HTTPS_PROXY=$HTTPS_PROXY PYTHONPATH=$PYTHONPATH"
tmux new-session -d -s "$SESSION" "$RUN_ENV $PYTHON_BIN scripts/guarded_run.py --max-ram-fraction 0.90 -- $PYTHON_BIN -m torch.distributed.run --standalone --nproc_per_node=4 -m safa.cli.train_g --config $CONFIG 2>&1 | tee $LOG"
echo "Started tmux session $SESSION. Log: $LOG"
if [[ "${ATTACH:-0}" == "1" ]]; then
  tmux attach -t "$SESSION"
fi
