#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization}"
CONFIG="${CONFIG:-configs/train_g_v2_best.yaml}"
SESSION="${SESSION:-train_g_v2_best}"
LOG="${LOG:-artifacts/logs/train_g_v2_best.log}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

usage() {
  echo "Usage: $0 [--config PATH] [--log PATH] [--session NAME] [--nproc-per-node N]" >&2
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
    --nproc-per-node|--nproc_per_node)
      require_value "$@"
      NPROC_PER_NODE="$2"
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
VALIDATION_CONFIG="$CONFIG" VALIDATION_NPROC="$NPROC_PER_NODE" "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import yaml


def require_positive_int(config: dict, field: str, path: Path) -> int:
    if field not in config:
        raise SystemExit(f"{path}: missing required field {field}")
    value = config[field]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit(f"{path}: {field} must be a positive integer, got {value!r}")
    return int(value)


config_path = Path(os.environ["VALIDATION_CONFIG"])
if not config_path.is_file():
    raise SystemExit(f"config not found: {config_path}")
try:
    nproc = int(os.environ["VALIDATION_NPROC"])
except ValueError as exc:
    raise SystemExit(f"--nproc_per_node must be a positive integer, got {os.environ['VALIDATION_NPROC']!r}") from exc
if nproc <= 0:
    raise SystemExit(f"--nproc_per_node must be a positive integer, got {nproc!r}")
with config_path.open("r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle)
if not isinstance(payload, dict):
    raise SystemExit(f"{config_path}: YAML config must be a mapping")
global_batch_size = require_positive_int(payload, "global_batch_size", config_path)
per_device_batch_size = require_positive_int(payload, "per_device_batch_size", config_path)
if global_batch_size % per_device_batch_size != 0:
    raise SystemExit(
        f"{config_path}: global_batch_size / per_device_batch_size must be an integer, "
        f"got {global_batch_size}/{per_device_batch_size}"
    )
expected_nproc = global_batch_size // per_device_batch_size
if nproc != expected_nproc:
    raise SystemExit(
        f"--nproc_per_node={nproc} does not match "
        f"global_batch_size / per_device_batch_size = {expected_nproc}"
    )
PY
mkdir -p "$(dirname "$LOG")"
TMUX_PAYLOAD=(
  env
  "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  "MKL_NUM_THREADS=$MKL_NUM_THREADS"
  "HTTP_PROXY=$HTTP_PROXY"
  "HTTPS_PROXY=$HTTPS_PROXY"
  "PYTHONPATH=$PYTHONPATH"
  "$PYTHON_BIN"
  scripts/guarded_run.py
  --max-ram-fraction
  0.90
  --
  "$PYTHON_BIN"
  -m
  torch.distributed.run
  --standalone
  "--nproc_per_node=$NPROC_PER_NODE"
  -m
  safa.cli.train_g
  --config
  "$CONFIG"
)
printf -v TMUX_COMMAND '%q ' "${TMUX_PAYLOAD[@]}"
TMUX_COMMAND+="2>&1 | tee -- $(printf '%q' "$LOG")"
tmux new-session -d -s "$SESSION" "$TMUX_COMMAND"
echo "Started tmux session $SESSION. Log: $LOG"
if [[ "${ATTACH:-0}" == "1" ]]; then
  tmux attach -t "$SESSION"
fi
