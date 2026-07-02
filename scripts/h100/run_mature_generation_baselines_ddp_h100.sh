#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/k100/miniconda3/envs/pt210_cu130_fa4/bin/python}"
NPROC_PER_NODE=4
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DRY_RUN=1
INCLUDE_E16=0
ONE=""
LOG_DIR="artifacts/logs"
RUN_DIR="artifacts/run"

E19_CONFIG="configs/medium_v2/experiments/e19_meanflow_sit_b2_face_mixed_2400ep.yaml"
E16_CONFIG="configs/medium_v2/experiments/e16_meanflow_sit_l2_face_mixed_2400ep.yaml"

usage() {
  cat <<'EOF'
Usage: scripts/h100/run_mature_generation_baselines_ddp_h100.sh [options]

Default mode is dry-run. Pass --run to start sequential 4-GPU torchrun training.
By default this mature queue runs E19 only and marks E16 as skip-current. Use --include-e16 to opt in.

Options:
  --run                  Start training.
  --dry-run              Write runtime YAMLs and print the plan without starting training.
  --repo-root PATH       Repository root. Defaults to the script's repo.
  --python PATH          Python used for runtime YAML generation.
  --nproc-per-node N     torchrun processes per node. Defaults to 4.
  --timestamp VALUE      Timestamp for log, pid, and status files.
  --include-e16          Include E16 L/2 after E19.
  --one NAME_OR_CONFIG   Run one experiment by name or config path.
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) DRY_RUN=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --timestamp) TIMESTAMP="$2"; shift 2 ;;
    --include-e16) INCLUDE_E16=1; shift ;;
    --one) ONE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

case "${NPROC_PER_NODE}" in
  ''|*[!0-9]*|0) echo "--nproc-per-node must be a positive integer" >&2; exit 64 ;;
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
  basename "$1" .yaml
}

runtime_config_for_config() {
  local config="$1" name
  name="$(experiment_name_for_config "${config}")"
  printf '%s/%s_h100_mature_ddp_runtime.yaml\n' "$(dirname "${config}")" "${name}"
}

batch_for_config() {
  local name="$1"
  case "${name}" in
    e19_*) printf '32 128\n' ;;
    e16_*) printf '16 64\n' ;;
    *) echo "no H100 mature batch mapping for ${name}" >&2; return 1 ;;
  esac
}

matches_one() {
  local config="$1" wanted="$2" name base
  name="$(experiment_name_for_config "${config}")"
  base="$(basename "${config}")"
  [[ "${wanted}" == "${name}" || "${wanted}" == "${base}" || "${wanted}" == "${config}" || "${wanted}" == "${REPO_ROOT}/${config}" ]]
}

all_configs() {
  printf '%s\n' "${E19_CONFIG}"
  if [[ "${INCLUDE_E16}" -eq 1 || -n "${ONE}" ]]; then
    printf '%s\n' "${E16_CONFIG}"
  fi
}

select_configs() {
  local config matched=0
  while IFS= read -r config; do
    [[ -n "${config}" ]] || continue
    if [[ -z "${ONE}" ]] || matches_one "${config}" "${ONE}"; then
      printf '%s\n' "${config}"
      matched=1
    fi
  done < <(all_configs)
  if [[ -n "${ONE}" && "${matched}" -eq 0 ]]; then
    echo "no config matched --one ${ONE}" >&2
    return 65
  fi
}

write_runtime_config() {
  local config="$1" runtime_config="$2" per_device_batch="$3" global_batch="$4"
  mkdir -p "$(dirname "${REPO_ROOT}/${runtime_config}")"
  "${PYTHON_BIN}" - "${REPO_ROOT}/${config}" "${REPO_ROOT}/${runtime_config}" "${REPO_ROOT}" "${per_device_batch}" "${global_batch}" <<'PYCODE'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
repo_root = Path(sys.argv[3])
per_device_batch = int(sys.argv[4])
global_batch = int(sys.argv[5])

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
    if not output_dir.endswith("_h100_mature_ddp"):
        output_dir = f"{output_dir}_h100_mature_ddp"
    quality_eval["output_dir"] = output_dir
quality_eval["distribution_cuda_visible_devices"] = "0"
quality_eval["distribution_device"] = "cuda:0"
stage2["quality_eval"] = quality_eval
stages["stage2"] = stage2
config["stages"] = stages

out_dir = str(config.get("out_dir") or "")
last_checkpoint = repo_root / out_dir / "last.pt" if out_dir else None
if last_checkpoint is not None and last_checkpoint.is_file():
    config["resume_from"] = f"{out_dir}/last.pt"
    config["resume_mode"] = "training_state"
    config["resume_optimizer_state"] = True
else:
    config["resume_from"] = ""
    config["resume_mode"] = "model_weights_only"
    config["resume_optimizer_state"] = False

with target.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PYCODE
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
  printf '%s/%s_h100_mature_ddp_%s.log\n' "${LOG_DIR}" "$1" "${TIMESTAMP}"
}

pid_path_for_name() {
  printf '%s/%s_h100_mature_ddp_%s.pid\n' "${RUN_DIR}" "$1" "${TIMESTAMP}"
}

status_path_for_name() {
  printf '%s/%s_h100_mature_ddp_%s.status\n' "${RUN_DIR}" "$1" "${TIMESTAMP}"
}

resume_from_for_runtime() {
  "${PYTHON_BIN}" - "$1" <<'PYCODE'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    print((yaml.safe_load(handle) or {}).get("resume_from") or "")
PYCODE
}

print_plan() {
  local mode="$1" config name runtime_config per_device_batch global_batch resume_from
  echo "${mode}"
  echo "repo_root: ${REPO_ROOT}"
  echo "timestamp: ${TIMESTAMP}"
  echo "python: ${PYTHON_BIN}"
  echo "torchrun: ${TORCHRUN_BIN}"
  echo "nproc_per_node: ${NPROC_PER_NODE}"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
  if [[ "${INCLUDE_E16}" -eq 0 && -z "${ONE}" ]]; then
    echo "skip-current: e16_meanflow_sit_l2_face_mixed_2400ep"
  fi
  echo "logs: ${LOG_DIR}"
  echo "pids: ${RUN_DIR}"
  echo "order:"
  while IFS= read -r config; do
    [[ -n "${config}" ]] || continue
    name="$(experiment_name_for_config "${config}")"
    read -r per_device_batch global_batch < <(batch_for_config "${name}")
    runtime_config="$(runtime_config_for_config "${config}")"
    resume_from="$(resume_from_for_runtime "${REPO_ROOT}/${runtime_config}")"
    echo "  - ${name}"
    echo "    source_config: ${config}"
    echo "    runtime_config: ${runtime_config}"
    echo "    batch: per_device=${per_device_batch} global=${global_batch}"
    echo "    resume_from: ${resume_from:-none}"
    echo "    command: $(train_command_string "${runtime_config}")"
    echo "    log: $(log_path_for_name "${name}")"
    echo "    pid: $(pid_path_for_name "${name}")"
    echo "    status: $(status_path_for_name "${name}")"
  done < <(select_configs)
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

prepare_runtime_configs

if [[ "${DRY_RUN}" -eq 1 ]]; then
  print_plan "DRY RUN: mature MeanFlow-SiT H100 DDP queue"
  exit 0
fi

print_plan "RUN: starting mature MeanFlow-SiT H100 DDP queue"
run_queue
