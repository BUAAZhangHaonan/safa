#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization"
PYTHON_BIN="/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"
CONFIG="configs/medium_v2/experiments/r14_inpaint_feasibility_2560step.yaml"
ARTIFACT_ROOT="artifacts/r14_inpaint_feasibility/v1"
CHECKPOINT_ROOT="checkpoints/r14_inpaint_feasibility_2560step"
SESSION="safa-r14-inpaint-v1"
LOG="$ARTIFACT_ROOT/logs/pipeline.log"
GPU_LIST="0,1,2,3"
NPROC=4
NCCL_IB_DISABLE_VALUE="1"
NCCL_P2P_DISABLE_VALUE="0"

usage() {
  printf '%s\n' \
    "Usage: $0 [--dry-run | --validate]" \
    "Default: validate once, then start the locked pipeline in tmux session $SESSION." >&2
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

SMOKE=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  "--nproc_per_node=$NPROC"
  scripts/run_r14_inpaint_smoke.py
  --config "$CONFIG"
  --manifest "$ARTIFACT_ROOT/manifests/smoke8.jsonl"
  --output-dir "$ARTIFACT_ROOT/smoke8"
)
TRAIN=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  "--nproc_per_node=$NPROC"
  -m safa.cli.train_g
  --config "$CONFIG"
)
EXPORT=(
  "$PYTHON_BIN" scripts/export_r14_inpaint_ema.py
  --checkpoint "$CHECKPOINT_ROOT/last.pt"
  --output "$CHECKPOINT_ROOT/final_ema.pt"
  --metadata-output "$CHECKPOINT_ROOT/final_ema.json"
)
GENERATE=(
  "$PYTHON_BIN" -m torch.distributed.run
  --standalone
  "--nproc_per_node=$NPROC"
  scripts/run_r14_inpaint_generation.py
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT_ROOT/final_ema.pt"
  --manifest "$ARTIFACT_ROOT/manifests/regular32.jsonl"
  --output-dir "$ARTIFACT_ROOT/regular32_generation"
)
EVALUATE=(
  "$PYTHON_BIN" scripts/evaluate_r14_inpaint_feasibility.py
  --config "$CONFIG"
  --manifest "$ARTIFACT_ROOT/manifests/regular32.jsonl"
  --generation-dir "$ARTIFACT_ROOT/regular32_generation"
  --output-dir "$ARTIFACT_ROOT/regular32_evaluation"
)
RENDER=(
  "$PYTHON_BIN" scripts/render_r14_inpaint_visual8.py
  --manifest "$ARTIFACT_ROOT/manifests/visual8.jsonl"
  --generation-dir "$ARTIFACT_ROOT/regular32_generation"
  --output-dir "$ARTIFACT_ROOT/visual8"
)
CLOSE=(
  "$PYTHON_BIN" scripts/close_r14_inpaint_feasibility.py
  --evaluation "$ARTIFACT_ROOT/regular32_evaluation/summary.json"
  --visual "$ARTIFACT_ROOT/visual8/summary.json"
  --output-dir "$ARTIFACT_ROOT"
)

VERIFY() {
  "$PYTHON_BIN" scripts/validate_r14_inpaint_feasibility.py --mode artifact --stage "$1"
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

print_pipeline() {
  printf '%s\n' "Locked R14 pipeline (CUDA_VISIBLE_DEVICES=$GPU_LIST):"
  print_command "${SMOKE[@]}"
  print_command "$PYTHON_BIN" scripts/validate_r14_inpaint_feasibility.py --mode artifact --stage smoke
  print_command "${TRAIN[@]}"
  print_command "$PYTHON_BIN" scripts/validate_r14_inpaint_feasibility.py --mode artifact --stage train
  print_command "${EXPORT[@]}"
  print_command "$PYTHON_BIN" scripts/validate_r14_inpaint_feasibility.py --mode artifact --stage export
  print_command "${GENERATE[@]}"
  print_command "$PYTHON_BIN" scripts/validate_r14_inpaint_feasibility.py --mode artifact --stage generation
  print_command "${EVALUATE[@]}"
  print_command "$PYTHON_BIN" scripts/validate_r14_inpaint_feasibility.py --mode artifact --stage evaluation
  print_command "${RENDER[@]}"
  print_command "$PYTHON_BIN" scripts/validate_r14_inpaint_feasibility.py --mode artifact --stage visual
  print_command "${CLOSE[@]}"
  print_command "$PYTHON_BIN" scripts/validate_r14_inpaint_feasibility.py --mode artifact --stage conclusion
}

run_pipeline() {
  "${SMOKE[@]}"
  VERIFY smoke
  "${TRAIN[@]}"
  VERIFY train
  "${EXPORT[@]}"
  VERIFY export
  "${GENERATE[@]}"
  VERIFY generation
  "${EVALUATE[@]}"
  VERIFY evaluation
  "${RENDER[@]}"
  VERIFY visual
  "${CLOSE[@]}"
  VERIFY conclusion
}

case "$MODE" in
  dry-run)
    "$PYTHON_BIN" scripts/validate_r14_inpaint_feasibility.py --mode static
    print_pipeline
    ;;
  validate)
    "$PYTHON_BIN" scripts/validate_r14_inpaint_feasibility.py --mode resource
    print_pipeline
    ;;
  inside-tmux)
    "$PYTHON_BIN" scripts/validate_r14_inpaint_feasibility.py --mode static
    run_pipeline
    ;;
  launch)
    "$PYTHON_BIN" scripts/validate_r14_inpaint_feasibility.py --mode resource
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
      bash scripts/run_r14_inpaint_feasibility.sh --inside-tmux
    )
    printf -v TMUX_COMMAND '%q ' "${TMUX_PAYLOAD[@]}"
    TMUX_COMMAND+="2>&1 | tee -- $(printf '%q' "$LOG")"
    tmux new-session -d -s "$SESSION" "$TMUX_COMMAND"
    printf 'Started %s on physical GPU0,1,2,3. Log: %s\n' "$SESSION" "$LOG"
    ;;
  *)
    printf 'unreachable mode: %s\n' "$MODE" >&2
    exit 2
    ;;
esac
