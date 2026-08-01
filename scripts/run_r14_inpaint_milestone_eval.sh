#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization"
PYTHON_BIN="/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"
OUTPUT_ROOT="artifacts/r14_inpaint_milestone_eval/v2"
SESSION="safa-r14-inpaint-milestone-eval-v1"
LOG="$OUTPUT_ROOT/logs/pipeline.log"

MODE="launch"
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--dry-run | --inside-tmux]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --dry-run) MODE="dry-run" ;;
    --inside-tmux) MODE="inside-tmux" ;;
    -h|--help) echo "Usage: $0 [--dry-run | --inside-tmux]"; exit 0 ;;
    *) echo "Usage: $0 [--dry-run | --inside-tmux]" >&2; exit 2 ;;
  esac
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="1,2"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

case "$MODE" in
  dry-run)
    "$PYTHON_BIN" scripts/run_r14_inpaint_milestone_eval.py --mode dry-run --output-root "$OUTPUT_ROOT"
    ;;
  inside-tmux)
    "$PYTHON_BIN" scripts/run_r14_inpaint_milestone_eval.py --mode run --output-root "$OUTPUT_ROOT"
    ;;
  launch)
    "$PYTHON_BIN" scripts/run_r14_inpaint_milestone_eval.py --mode dry-run --output-root "$OUTPUT_ROOT"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "refusing duplicate eval session: $SESSION" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$LOG")"
    TMUX_COMMAND="env CUDA_VISIBLE_DEVICES=1,2 PYTHONPATH=$(printf '%q' "$PYTHONPATH") OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 bash scripts/run_r14_inpaint_milestone_eval.sh --inside-tmux 2>&1 | tee -- $(printf '%q' "$LOG")"
    tmux new-session -d -s "$SESSION" "$TMUX_COMMAND"
    echo "Started $SESSION on physical GPU1,2. Log: $LOG"
    ;;
esac
