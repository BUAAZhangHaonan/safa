#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization"
PYTHON_BIN="/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"
CONFIG="configs/medium_v2/experiments/r14_inpaint_resume_gpu01_step2560.yaml"
MANIFEST="artifacts/r14_inpaint_feasibility/v1/manifests/smoke8.jsonl"
OUTPUT_DIR="artifacts/r14_inpaint_resume_gpu01/batch4_smoke_v1"
LOG="artifacts/r14_inpaint_resume_gpu01/logs/batch4_smoke_v1.log"
SESSION="safa-r14-inpaint-batch4-smoke-v1"
GPU_LIST="0,1"
NPROC=2

MODE="launch"
if [[ $# -gt 1 ]]; then
  printf 'Usage: %s [--dry-run | --inside-tmux]\n' "$0" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --dry-run) MODE="dry-run" ;;
    --inside-tmux) MODE="inside-tmux" ;;
    *) printf 'Usage: %s [--dry-run]\n' "$0" >&2; exit 2 ;;
  esac
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

SMOKE=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  "--nproc_per_node=$NPROC"
  scripts/run_r14_inpaint_resume_batch4_smoke.py
  --config "$CONFIG"
  --manifest "$MANIFEST"
  --output-dir "$OUTPUT_DIR"
)

print_command() {
  printf 'R14 batch4 smoke (physical GPU%s):\n  ' "$GPU_LIST"
  printf '%q ' "${SMOKE[@]}"
  printf '\nOutput: %s\nLog: %s\n' "$OUTPUT_DIR" "$LOG"
}

validate_launch_state() {
  "$PYTHON_BIN" scripts/validate_r14_inpaint_resume_gpu01.py --mode static
  [[ "$(git branch --show-current)" == "master" ]]
  [[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/master)" ]]
  git diff --quiet
  git diff --cached --quiet
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    printf 'tmux session already exists: %s\n' "$SESSION" >&2
    exit 1
  fi
  [[ ! -e "$OUTPUT_DIR" ]]
  [[ ! -e "$LOG" ]]
  for gpu in 0 1; do
    IFS=',' read -r used_mib free_mib utilization < <(
      nvidia-smi -i "$gpu" \
        --query-gpu=memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
    )
    used_mib="${used_mib// /}"
    free_mib="${free_mib// /}"
    utilization="${utilization// /}"
    if (( used_mib >= 1024 || free_mib < 22000 || utilization >= 90 )); then
      printf 'GPU%s is not empty enough for the batch4 discovery smoke: used=%s free=%s util=%s\n' \
        "$gpu" "$used_mib" "$free_mib" "$utilization" >&2
      exit 1
    fi
  done
}

case "$MODE" in
  dry-run)
    validate_launch_state
    print_command
    ;;
  inside-tmux)
    if [[ "${R14_BATCH4_SMOKE_PREFLIGHT_PASSED:-}" != "1" ]]; then
      printf '%s\n' "--inside-tmux is reserved for the validated launcher" >&2
      exit 2
    fi
    print_command
    "${SMOKE[@]}"
    ;;
  launch)
    validate_launch_state
    mkdir -p "$(dirname "$LOG")"
    TMUX_PAYLOAD=(
      env
      "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
      "PYTHONPATH=$PYTHONPATH"
      "OMP_NUM_THREADS=$OMP_NUM_THREADS"
      "MKL_NUM_THREADS=$MKL_NUM_THREADS"
      "PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
      "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
      "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
      "R14_BATCH4_SMOKE_PREFLIGHT_PASSED=1"
      bash scripts/run_r14_inpaint_resume_batch4_smoke.sh --inside-tmux
    )
    printf -v TMUX_COMMAND '%q ' "${TMUX_PAYLOAD[@]}"
    TMUX_COMMAND="set -o pipefail; ${TMUX_COMMAND}2>&1 | tee -- $(printf '%q' "$LOG")"
    tmux new-session -d -s "$SESSION" "$TMUX_COMMAND"
    printf 'Started %s. Log: %s\n' "$SESSION" "$LOG"
    ;;
esac
