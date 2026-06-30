#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_DIR="${ROOT_DIR}/artifacts/external/generation_baselines"
WEIGHT_DIR="${ROOT_DIR}/artifacts/checkpoints/external/meanflow_sit"
MANIFEST_PATH="${ROOT_DIR}/artifacts/manifests/generation_baseline_weights_manifest.json"
VERIFY_SCRIPT="${ROOT_DIR}/scripts/external/verify_generation_baseline_weights.py"
PREFERRED_PYTHON="/home/k100/miniconda3/envs/pt210_cu130_fa4/bin/python"
PYTHON_BIN="${PYTHON:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${PREFERRED_PYTHON}" ]]; then
    PYTHON_BIN="${PREFERRED_PYTHON}"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

clone_if_missing() {
  local url="$1"
  local dest="$2"
  if [[ -d "${dest}/.git" ]]; then
    echo "skip existing repo: ${dest}"
    return 0
  fi
  if [[ -e "${dest}" ]]; then
    echo "skip existing path without cloning: ${dest}"
    return 0
  fi
  git clone --depth 1 "${url}" "${dest}"
}

gdown_is_available() {
  "${PYTHON_BIN}" -m gdown --help >/dev/null 2>&1 || command -v gdown >/dev/null 2>&1
}

run_gdown() {
  if "${PYTHON_BIN}" -m gdown --help >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m gdown "$@"
    return $?
  fi
  if command -v gdown >/dev/null 2>&1; then
    gdown "$@"
    return $?
  fi
  return 127
}

try_download_meanflow_b2() {
  local out_path="${WEIGHT_DIR}/zhuyu_sit_b_2_imagenet256.pt"
  if [[ -f "${out_path}" ]]; then
    echo "MeanFlow-SiT B/2 weight already exists: ${out_path}"
    return 0
  fi

  echo "MeanFlow-SiT B/2 weight is missing: ${out_path}"
  echo "Set MEANFLOW_SIT_B2_GDRIVE_URL or MEANFLOW_SIT_B2_GDRIVE_ID to let this script try gdown."
  if ! gdown_is_available; then
    echo "gdown is not available via ${PYTHON_BIN} -m gdown or PATH; skipping automatic B/2 download."
    return 0
  fi

  mkdir -p "${WEIGHT_DIR}"
  if [[ -n "${MEANFLOW_SIT_B2_GDRIVE_URL:-}" ]]; then
    run_gdown --fuzzy "${MEANFLOW_SIT_B2_GDRIVE_URL}" -O "${out_path}" || rm -f "${out_path}"
  elif [[ -n "${MEANFLOW_SIT_B2_GDRIVE_ID:-}" ]]; then
    run_gdown "${MEANFLOW_SIT_B2_GDRIVE_ID}" -O "${out_path}" || rm -f "${out_path}"
  fi
}

main() {
  mkdir -p "${CODE_DIR}" "${WEIGHT_DIR}" "$(dirname "${MANIFEST_PATH}")"

  clone_if_missing "https://github.com/Gsunshine/meanflow.git" "${CODE_DIR}/Gsunshine_meanflow"
  clone_if_missing "https://github.com/zhuyu-cs/MeanFlow.git" "${CODE_DIR}/zhuyu-cs_MeanFlow"
  clone_if_missing "https://github.com/facebookresearch/DiT.git" "${CODE_DIR}/facebookresearch_DiT"
  clone_if_missing "https://github.com/openai/consistency_models.git" "${CODE_DIR}/openai_consistency_models"

  local l2_path="${WEIGHT_DIR}/zhuyu_sit_l_2_imagenet256.pt"
  if [[ -f "${l2_path}" ]]; then
    echo "MeanFlow-SiT L/2 weight exists; verifying only: ${l2_path}"
  else
    echo "MeanFlow-SiT L/2 weight is missing; this script does not guess a download URL."
  fi

  try_download_meanflow_b2

  "${PYTHON_BIN}" "${VERIFY_SCRIPT}" --root "${ROOT_DIR}" --manifest "${MANIFEST_PATH}" --require-existing all
  echo "wrote manifest: ${MANIFEST_PATH}"
}

main "$@"
