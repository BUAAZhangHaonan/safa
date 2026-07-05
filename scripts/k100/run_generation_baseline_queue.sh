#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/k100/miniconda3/envs/pt210_cu130_fa4/bin/python}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DRY_RUN=1
ABLATION_ONLY=0
E16_PATTERN="${SAFA_E16_PATTERN:-e16_meanflow_sit_l2_face_mixed_2400ep}"

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
Usage: scripts/k100/run_generation_baseline_queue.sh [--ablation-only] [--run] [--repo-root PATH] [--python PATH] [--timestamp VALUE]

Default mode refuses to run. Pass --ablation-only to acknowledge this is the internal ablation queue.
With --ablation-only, default mode is dry-run. Add --run to start training.
E17/E18/E20/E21/E22/E23 are internal ablation experiments, not paper main-table mature baselines.
The script exits without starting anything while an E16 train_g process is still running.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      DRY_RUN=0
      shift
      ;;
    --ablation-only)
      ABLATION_ONLY=1
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
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    --e16-pattern)
      E16_PATTERN="$2"
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

if [[ "${ABLATION_ONLY}" -ne 1 ]]; then
  echo "Refusing to run generation baseline queue by default: E17/E18/E20/E21/E22/E23 are internal ablation experiments, not paper main-table mature baselines." >&2
  echo "Pass --ablation-only to dry-run or run this internal ablation queue." >&2
  exit 64
fi

LOG_DIR="artifacts/logs"
RUN_DIR="artifacts/run"

experiment_name_for_config() {
  local config="$1"
  basename "${config}" .yaml
}

train_command_for_config() {
  local config="$1"
  printf '%q -m safa.cli.train_g --config %q' "${PYTHON_BIN}" "${config}"
}

matching_e16_processes() {
  ps -eo pid=,args= | while read -r pid args; do
    if [[ "${args}" == *"safa.cli.train_g"* && "${args}" == *"${E16_PATTERN}"* ]]; then
      printf '%s %s\n' "${pid}" "${args}"
    elif [[ "${args}" == *"${E16_PATTERN}"* && "${E16_PATTERN}" != "e16_meanflow_sit_l2_face_mixed_2400ep" ]]; then
      printf '%s %s\n' "${pid}" "${args}"
    fi
  done
}

print_plan() {
  local mode="$1"
  echo "${mode}"
  echo "repo_root: ${REPO_ROOT}"
  echo "timestamp: ${TIMESTAMP}"
  echo "python: ${PYTHON_BIN}"
  echo "logs: ${LOG_DIR}"
  echo "pids: ${RUN_DIR}"
  echo "order:"
  for config in "${CONFIGS[@]}"; do
    local name
    name="$(experiment_name_for_config "${config}")"
    echo "  - ${name}"
    echo "    config: ${config}"
    echo "    command: $(train_command_for_config "${config}")"
    echo "    log: ${LOG_DIR}/${name}_${TIMESTAMP}.log"
    echo "    pid: ${RUN_DIR}/${name}_${TIMESTAMP}.pid"
  done
}

run_queue() {
  mkdir -p "${REPO_ROOT}/${LOG_DIR}" "${REPO_ROOT}/${RUN_DIR}"
  cd "${REPO_ROOT}"
  for config in "${CONFIGS[@]}"; do
    local name log_path pid_path status_path
    name="$(experiment_name_for_config "${config}")"
    log_path="${LOG_DIR}/${name}_${TIMESTAMP}.log"
    pid_path="${RUN_DIR}/${name}_${TIMESTAMP}.pid"
    status_path="${RUN_DIR}/${name}_${TIMESTAMP}.status"
    echo "starting ${name}"
    "${PYTHON_BIN}" -m safa.cli.train_g --config "${config}" > "${log_path}" 2>&1 &
    local train_pid=$!
    echo "${train_pid}" > "${pid_path}"
    wait "${train_pid}"
    local exit_code=$?
    echo "${exit_code}" > "${status_path}"
    if [[ "${exit_code}" -ne 0 ]]; then
      echo "training failed for ${name}; see ${log_path}" >&2
      return "${exit_code}"
    fi
    echo "finished ${name}"
  done
}

e16_matches="$(matching_e16_processes || true)"
if [[ -n "${e16_matches}" ]]; then
  echo "E16 training is still running; generation baseline queue will not start."
  echo "${e16_matches}"
  exit 2
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  print_plan "DRY RUN: internal ablation queue; pass --run with --ablation-only to start training"
  exit 0
fi

print_plan "RUN: starting internal ablation generation baseline queue"
run_queue
