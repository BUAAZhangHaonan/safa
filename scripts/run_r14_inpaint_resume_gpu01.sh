#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization"
PYTHON_BIN="/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"
CONFIG="configs/medium_v2/experiments/r14_inpaint_resume_gpu01_step2688.yaml"
ARTIFACT_ROOT="artifacts/r14_inpaint_resume_gpu01/v1"
CHECKPOINT_ROOT="checkpoints/r14_inpaint_resume_gpu01_step2688"
SESSION="safa-r14-inpaint-resume-gpu01-v1"
LOG="$ARTIFACT_ROOT/logs/train.log"
GPU_LIST="0,1"
NPROC=2
NCCL_IB_DISABLE_VALUE="1"
NCCL_P2P_DISABLE_VALUE="0"

usage() {
  printf '%s\n' \
    "Usage: $0 [--dry-run | --validate]" \
    "Default: validate once, then resume R14 training in tmux session $SESSION." >&2
}

MODE="launch"
if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --dry-run) MODE="dry-run" ;;
    --validate) MODE="validate" ;;
    --inside-tmux) MODE="inside-tmux" ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE="$NCCL_IB_DISABLE_VALUE"
export NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE_VALUE"

TRAIN=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  "--nproc_per_node=$NPROC"
  -m safa.cli.train_g
  --config "$CONFIG"
)
VERIFY=(
  "$PYTHON_BIN" scripts/validate_r14_inpaint_resume_gpu01.py
  --mode artifact
)

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

print_pipeline() {
  printf '%s\n' \
    "Locked R14 resume training (CUDA_VISIBLE_DEVICES=$GPU_LIST, nproc=$NPROC):" \
    "  source step: 2432" \
    "  additional optimizer steps: 256" \
    "  required final step: 2688" \
    "  output: $CHECKPOINT_ROOT"
  print_command "${TRAIN[@]}"
  print_command "${VERIFY[@]}"
}

run_pipeline() {
  "${TRAIN[@]}"
  "${VERIFY[@]}"
}

case "$MODE" in
  dry-run)
    "$PYTHON_BIN" scripts/validate_r14_inpaint_resume_gpu01.py --mode static
    print_pipeline
    ;;
  validate)
    "$PYTHON_BIN" scripts/validate_r14_inpaint_resume_gpu01.py --mode resource
    print_pipeline
    ;;
  inside-tmux)
    if [[ "${R14_RESUME_PREFLIGHT_PASSED:-}" != "1" ]]; then
      printf '%s\n' "--inside-tmux is reserved for the validated launcher" >&2
      exit 2
    fi
    printf 'Git SHA: %s\n' "$(git rev-parse HEAD)"
    printf '%s\n' \
      "Source checkpoint: checkpoints/r14_inpaint_feasibility_2560step/last.pt" \
      "Source SHA256: a176d5521782a16ba488fe5d727cec61ddcf35d07fe75316f00f281ef423b7bf" \
      "Per-GPU preflight peak: 8192 MiB (R14 four-GPU measured upper bound)"
    print_pipeline
    run_pipeline
    ;;
  launch)
    "$PYTHON_BIN" scripts/validate_r14_inpaint_resume_gpu01.py --mode resource
    mkdir -p "$(dirname "$LOG")"
    TMUX_PAYLOAD=(
      env
      "CUDA_VISIBLE_DEVICES=$GPU_LIST"
      "PYTHONPATH=$PYTHONPATH"
      "OMP_NUM_THREADS=$OMP_NUM_THREADS"
      "MKL_NUM_THREADS=$MKL_NUM_THREADS"
      "PYTHONUNBUFFERED=1"
      "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
      "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
      "R14_RESUME_PREFLIGHT_PASSED=1"
      bash scripts/run_r14_inpaint_resume_gpu01.sh --inside-tmux
    )
    printf -v TMUX_COMMAND '%q ' "${TMUX_PAYLOAD[@]}"
    TMUX_COMMAND="set -o pipefail; ${TMUX_COMMAND}2>&1 | tee -- $(printf '%q' "$LOG")"
    tmux new-session -d -s "$SESSION" "$TMUX_COMMAND"
    printf 'Started %s on physical GPU0,1. Log: %s\n' "$SESSION" "$LOG"
    ;;
  *)
    printf 'unreachable mode: %s\n' "$MODE" >&2
    exit 2
    ;;
esac
