#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/k100/miniconda3/envs/pt210_cu130_fa4/bin/python}"
NPROC_PER_NODE=4
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DRY_RUN=1
SKIP_E16_CHECK=0
ONE=""
E16_PATTERN="${SAFA_E16_PATTERN:-e16_meanflow_sit_l2_face_mixed_2400ep}"
DEFAULT_E16_PATTERN="e16_meanflow_sit_l2_face_mixed_2400ep"

LOG_DIR="artifacts/logs"
RUN_DIR="artifacts/run"

CONFIGS=(
  "configs/medium_v2/experiments/e22_sit_diffusion_b2_face_mixed_2400ep.yaml"
  "configs/medium_v2/experiments/e23_latent_consistency_b2_face_mixed_2400ep.yaml"
  "configs/medium_v2/experiments/e19_meanflow_sit_b2_face_mixed_2400ep.yaml"
  "configs/medium_v2/experiments/e20_rectified_flow_sit_b2_face_mixed_2400ep.yaml"
  "configs/medium_v2/experiments/e17_sit_diffusion_l2_face_mixed_2400ep.yaml"
  "configs/medium_v2/experiments/e18_latent_consistency_l2_face_mixed_2400ep.yaml"
  "configs/medium_v2/experiments/e21_rectified_flow_sit_l2_face_mixed_2400ep.yaml"
)

usage() {
  cat <<'EOF'
Usage: scripts/h100/run_generation_baseline_ddp_h100.sh [options]

Default mode is dry-run. Pass --run to start training.

Options:
  --run                  Start sequential torchrun training.
  --dry-run              Print the queue plan without starting training.
  --repo-root PATH       Repository root. Defaults to the script's repo.
  --python PATH          Python used for runtime YAML generation.
  --nproc-per-node N     torchrun processes per node. Defaults to 4.
  --timestamp VALUE      Timestamp for log, pid, and status files.
  --skip-e16-check       Do not block while an E16 process is running.
  --one NAME_OR_CONFIG   Run one experiment by name or config path.
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      DRY_RUN=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    --skip-e16-check)
      SKIP_E16_CHECK=1
      shift
      ;;
    --one)
      ONE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

case "${NPROC_PER_NODE}" in
  ''|*[!0-9]*|0)
    echo "--nproc-per-node must be a positive integer" >&2
    exit 64
    ;;
esac

torchrun_bin_for_python() {
  local candidate
  candidate="$(dirname "${PYTHON_BIN}")/torchrun"
  if [[ -x "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
  else
    printf 'torchrun\n'
  fi
}

TORCHRUN_BIN="$(torchrun_bin_for_python)"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC_PER_NODE - 1)))"
  export CUDA_VISIBLE_DEVICES
fi
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

experiment_name_for_config() {
  local config="$1"
  basename "${config}" .yaml
}

runtime_config_for_config() {
  local config="$1"
  local name
  name="$(experiment_name_for_config "${config}")"
  printf '%s/%s_h100_ddp_runtime.yaml\n' "$(dirname "${config}")" "${name}"
}

batch_for_config() {
  local name="$1"
  case "${name}" in
    e19_*|e20_*|e22_*|e23_*)
      printf '32 128\n'
      ;;
    e17_*|e18_*|e21_*)
      printf '16 64\n'
      ;;
    *)
      echo "no H100 batch mapping for ${name}" >&2
      return 1
      ;;
  esac
}

matches_one() {
  local config="$1"
  local wanted="$2"
  local name base
  name="$(experiment_name_for_config "${config}")"
  base="$(basename "${config}")"
  [[ "${wanted}" == "${name}" || "${wanted}" == "${base}" || "${wanted}" == "${config}" || "${wanted}" == "${REPO_ROOT}/${config}" ]]
}

select_configs() {
  local config matched=0
  for config in "${CONFIGS[@]}"; do
    if [[ -z "${ONE}" ]] || matches_one "${config}" "${ONE}"; then
      printf '%s\n' "${config}"
      matched=1
    fi
  done
  if [[ -n "${ONE}" && "${matched}" -eq 0 ]]; then
    echo "no config matched --one ${ONE}" >&2
    return 65
  fi
}

write_runtime_config() {
  local config="$1"
  local runtime_config="$2"
  local per_device_batch="$3"
  local global_batch="$4"

  mkdir -p "$(dirname "${REPO_ROOT}/${runtime_config}")"
  "${PYTHON_BIN}" - "${REPO_ROOT}/${config}" "${REPO_ROOT}/${runtime_config}" "${per_device_batch}" "${global_batch}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml


source = Path(sys.argv[1])
target = Path(sys.argv[2])
per_device_batch = int(sys.argv[3])
global_batch = int(sys.argv[4])

with source.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

config["device"] = "cuda:0"
config["distributed"] = dict(config.get("distributed") or {})
config["distributed"]["backend"] = "nccl"
config["global_batch_size"] = global_batch
config["per_device_batch_size"] = per_device_batch
config["num_workers"] = 8

config["validation"] = dict(config.get("validation") or {})
config["validation"]["batch_size"] = 16

stages = dict(config.get("stages") or {})
stage2 = dict(stages.get("stage2") or {})
quality_eval = dict(stage2.get("quality_eval") or {})
output_dir = quality_eval.get("output_dir")
if output_dir:
    output_dir = str(output_dir)
    if not output_dir.endswith("_h100_ddp"):
        output_dir = f"{output_dir}_h100_ddp"
    quality_eval["output_dir"] = output_dir
quality_eval["distribution_cuda_visible_devices"] = "0"
quality_eval["distribution_device"] = "cuda:0"
stage2["quality_eval"] = quality_eval
stages["stage2"] = stage2
config["stages"] = stages

with target.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
}

train_command_string() {
  local runtime_config="$1"
  printf '%q ' \
    "${TORCHRUN_BIN}" \
    --standalone \
    "--nproc_per_node=${NPROC_PER_NODE}" \
    -m safa.cli.train_g \
    --config "${runtime_config}"
}

train_command_array() {
  local runtime_config="$1"
  TRAIN_COMMAND=(
    "${TORCHRUN_BIN}"
    --standalone
    "--nproc_per_node=${NPROC_PER_NODE}"
    -m
    safa.cli.train_g
    --config
    "${runtime_config}"
  )
}

log_path_for_name() {
  local name="$1"
  printf '%s/%s_h100_ddp_%s.log\n' "${LOG_DIR}" "${name}" "${TIMESTAMP}"
}

pid_path_for_name() {
  local name="$1"
  printf '%s/%s_h100_ddp_%s.pid\n' "${RUN_DIR}" "${name}" "${TIMESTAMP}"
}

status_path_for_name() {
  local name="$1"
  printf '%s/%s_h100_ddp_%s.status\n' "${RUN_DIR}" "${name}" "${TIMESTAMP}"
}

prepare_runtime_configs() {
  local config name runtime_config per_device_batch global_batch
  while IFS= read -r config; do
    [[ -n "${config}" ]] || continue
    name="$(experiment_name_for_config "${config}")"
    read -r per_device_batch global_batch < <(batch_for_config "${name}")
    runtime_config="$(runtime_config_for_config "${config}")"
    write_runtime_config "${config}" "${runtime_config}" "${per_device_batch}" "${global_batch}"
  done < <(select_configs)
}

print_plan() {
  local mode="$1"
  local config name runtime_config per_device_batch global_batch
  echo "${mode}"
  echo "repo_root: ${REPO_ROOT}"
  echo "timestamp: ${TIMESTAMP}"
  echo "python: ${PYTHON_BIN}"
  echo "torchrun: ${TORCHRUN_BIN}"
  echo "nproc_per_node: ${NPROC_PER_NODE}"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
  echo "logs: ${LOG_DIR}"
  echo "pids: ${RUN_DIR}"
  echo "order:"
  while IFS= read -r config; do
    [[ -n "${config}" ]] || continue
    name="$(experiment_name_for_config "${config}")"
    read -r per_device_batch global_batch < <(batch_for_config "${name}")
    runtime_config="$(runtime_config_for_config "${config}")"
    echo "  - ${name}"
    echo "    source_config: ${config}"
    echo "    runtime_config: ${runtime_config}"
    echo "    batch: per_device=${per_device_batch} global=${global_batch}"
    echo "    command: $(train_command_string "${runtime_config}")"
    echo "    log: $(log_path_for_name "${name}")"
    echo "    pid: $(pid_path_for_name "${name}")"
    echo "    status: $(status_path_for_name "${name}")"
  done < <(select_configs)
}

matching_e16_processes() {
  ps -eo pid=,args= | while read -r pid args; do
    [[ -n "${pid:-}" ]] || continue
    if [[ "${args}" == *"${E16_PATTERN}"* && "${args}" == *"safa.cli.train_g"* ]]; then
      printf '%s %s\n' "${pid}" "${args}"
    elif [[ "${args}" == *"${E16_PATTERN}"* && "${E16_PATTERN}" != "${DEFAULT_E16_PATTERN}" ]]; then
      printf '%s %s\n' "${pid}" "${args}"
    fi
  done
}

run_queue() {
  local config name runtime_config log_path pid_path status_path train_pid exit_code
  mkdir -p "${REPO_ROOT}/${LOG_DIR}" "${REPO_ROOT}/${RUN_DIR}"
  cd "${REPO_ROOT}"
  while IFS= read -r config; do
    [[ -n "${config}" ]] || continue
    name="$(experiment_name_for_config "${config}")"
    runtime_config="$(runtime_config_for_config "${config}")"
    log_path="$(log_path_for_name "${name}")"
    pid_path="$(pid_path_for_name "${name}")"
    status_path="$(status_path_for_name "${name}")"

    echo "starting ${name}"
    train_command_array "${runtime_config}"
    "${TRAIN_COMMAND[@]}" > "${log_path}" 2>&1 &
    train_pid=$!
    echo "${train_pid}" > "${pid_path}"
    set +e
    wait "${train_pid}"
    exit_code=$?
    set -e
    echo "${exit_code}" > "${status_path}"
    if [[ "${exit_code}" -ne 0 ]]; then
      echo "training failed for ${name}; see ${log_path}" >&2
      return "${exit_code}"
    fi
    echo "finished ${name}"
  done < <(select_configs)
}

if [[ "${SKIP_E16_CHECK}" -eq 0 ]]; then
  e16_matches="$(matching_e16_processes || true)"
  if [[ -n "${e16_matches}" ]]; then
    echo "E16 training is still running; H100 DDP generation baseline queue will not start."
    echo "${e16_matches}"
    exit 2
  fi
fi

prepare_runtime_configs

if [[ "${DRY_RUN}" -eq 1 ]]; then
  print_plan "DRY RUN: pass --run to start training"
  exit 0
fi

print_plan "RUN: starting H100 DDP generation baseline queue"
run_queue
