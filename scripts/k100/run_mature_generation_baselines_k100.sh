#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/k100/miniconda3/envs/pt210_cu130_fa4/bin/python}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DRY_RUN=1
INCLUDE_E19=0
ONE=""
LOG_DIR="artifacts/logs"
RUN_DIR="artifacts/run"
CU13_LIBRARY_PATH="/home/k100/miniconda3/envs/pt210_cu130_fa4/lib/python3.12/site-packages/nvidia/cu13/lib"

E16_CONFIG="configs/medium_v2/experiments/e16_meanflow_sit_l2_face_mixed_2400ep.yaml"
E19_CONFIG="configs/medium_v2/experiments/e19_meanflow_sit_b2_face_mixed_2400ep.yaml"
usage() {
  cat <<'EOF'
Usage: scripts/k100/run_mature_generation_baselines_k100.sh [options]

Default mode is dry-run. Pass --run to start sequential single-GPU training.
This mature queue defaults to E16 L/2. Use --include-e19 or --one for E19 B/2.

Options:
  --run              Start training.
  --dry-run          Write runtime YAMLs and print the plan without starting training.
  --repo-root PATH   Repository root. Defaults to the script's repo.
  --python PATH      Python used for runtime YAML generation and training.
  --timestamp VALUE  Timestamp for log, pid, and status files.
  --include-e19      Include E19 B/2 after E16.
  --one NAME_OR_CONFIG
                     Run one experiment by name or config path.
  -h, --help         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) DRY_RUN=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --timestamp) TIMESTAMP="$2"; shift 2 ;;
    --include-e19) INCLUDE_E19=1; shift ;;
    --one) ONE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0"
fi
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [[ -d "${CU13_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${CU13_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

experiment_name_for_config() {
  basename "$1" .yaml
}

runtime_config_for_config() {
  local config="$1" name
  name="$(experiment_name_for_config "${config}")"
  printf '%s/%s_k100_runtime.yaml\n' "$(dirname "${config}")" "${name}"
}

batch_for_config() {
  local name="$1"
  case "${name}" in
    e16_*|e19_*) printf '32 32\n' ;;
    *) echo "no K100 mature batch mapping for ${name}" >&2; return 1 ;;
  esac
}

matches_one() {
  local config="$1" wanted="$2" name base
  name="$(experiment_name_for_config "${config}")"
  base="$(basename "${config}")"
  [[ "${wanted}" == "${name}" || "${wanted}" == "${base}" || "${wanted}" == "${config}" || "${wanted}" == "${REPO_ROOT}/${config}" ]]
}

all_configs() {
  printf '%s\n' "${E16_CONFIG}"
  if [[ "${INCLUDE_E19}" -eq 1 || -n "${ONE}" ]]; then
    printf '%s\n' "${E19_CONFIG}"
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
config["distributed"]["backend"] = "gloo"
config["global_batch_size"] = global_batch
config["per_device_batch_size"] = per_device_batch
config["disable_eval"] = True
config["eval"] = {"enabled": False}
config["visualization"] = {"enabled": False}
config["validation"] = dict(config.get("validation") or {})
config["validation"]["enabled"] = False
config["validation"]["max_samples"] = 0
config["validation"]["batch_size"] = 1
config["validation"]["face_detection"] = dict(config["validation"].get("face_detection") or {})
config["validation"]["face_detection"]["enabled"] = False

stages = dict(config.get("stages") or {})
stage2 = dict(stages.get("stage2") or {})
quality_eval = dict(stage2.get("quality_eval") or {})
quality_eval["enabled"] = False
quality_eval["metrics"] = []
quality_eval["niqe_interval_epochs"] = 1_000_000_000
quality_eval["distribution_interval_epochs"] = 1_000_000_000
quality_eval["niqe_max_samples"] = 0
quality_eval["distribution_max_samples"] = 0
quality_eval["quality_num_workers"] = 0
quality_eval["distribution_timeout_seconds"] = 1
quality_eval["distribution_cuda_visible_devices"] = ""
quality_eval["distribution_device"] = "cpu"
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
  printf '%q -m safa.cli.train_g --config %q' "${PYTHON_BIN}" "${runtime_config}"
}

log_path_for_name() {
  printf '%s/%s_k100_%s.log\n' "${LOG_DIR}" "$1" "${TIMESTAMP}"
}

pid_path_for_name() {
  printf '%s/%s_k100_%s.pid\n' "${RUN_DIR}" "$1" "${TIMESTAMP}"
}

status_path_for_name() {
  printf '%s/%s_k100_%s.status\n' "${RUN_DIR}" "$1" "${TIMESTAMP}"
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
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
  echo "eval: disabled"
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
    "${PYTHON_BIN}" -m safa.cli.train_g --config "${runtime_config}" > "${log_path}" 2>&1 &
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
  print_plan "DRY RUN: mature MeanFlow-SiT K100 queue"
  exit 0
fi

print_plan "RUN: starting mature MeanFlow-SiT K100 queue"
run_queue
