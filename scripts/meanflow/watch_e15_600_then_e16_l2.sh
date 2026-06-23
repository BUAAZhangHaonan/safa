#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/k100/miniconda3/envs/pt210_cu130_fa4/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TARGET_EPOCH="${TARGET_EPOCH:-600}"
GRACE_SECONDS="${GRACE_SECONDS:-900}"
OOM_WATCH_SECONDS="${OOM_WATCH_SECONDS:-1800}"

E15_EXP="e15_meanflow_sit_b_face_mixed_resume_e14_2400ep"
E15_CONFIG="configs/medium_v2/experiments/${E15_EXP}.yaml"
E15_CKPT_DIR="artifacts/checkpoints/${E15_EXP}"
E15_METRICS="${E15_CKPT_DIR}/last_metrics.json"
E15_LAST="${E15_CKPT_DIR}/last.pt"

E16_EXP="e16_meanflow_sit_l2_face_mixed_2400ep"
E16_TEMPLATE_CONFIG="configs/medium_v2/experiments/${E16_EXP}.yaml"
E16_RUNTIME_CONFIG="configs/medium_v2/experiments/${E16_EXP}_runtime.yaml"
E16_CKPT_DIR="artifacts/checkpoints/${E16_EXP}"
E16_PRETRAIN="artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_l_2_imagenet256.pt"
E16_PRETRAIN_SOURCE="zhuyu-cs/MeanFlow ImageNet256 sit_l_2_meanflow_ema.pt"
E16_LOG_DIR="artifacts/logs"
E16_LOG="${E16_LOG_DIR}/${E16_EXP}_$(date +%Y%m%d_%H%M%S).log"

RUN_DIR="artifacts/run"
LOCK_DIR="${RUN_DIR}/${E16_EXP}_switch.lock"
WATCH_LOG="${E16_LOG_DIR}/${E16_EXP}_watcher.log"
WATCH_PID_FILE="${RUN_DIR}/${E16_EXP}_watcher.pid"
E16_PID_FILE="${RUN_DIR}/${E16_EXP}.pid"

mkdir -p "$RUN_DIR" "$E16_LOG_DIR"

log() {
  printf "[%s] %s\n" "$(date +%F_%T%z)" "$*" | tee -a "$WATCH_LOG"
}

cleanup_lock() {
  rm -rf "$LOCK_DIR"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [[ -f "$WATCH_PID_FILE" ]] && kill -0 "$(cat "$WATCH_PID_FILE")" 2>/dev/null; then
    log "watcher already running pid=$(cat "$WATCH_PID_FILE"); exiting"
    exit 0
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
fi
trap cleanup_lock EXIT
printf "%s\n" "$$" > "$WATCH_PID_FILE"

read_e15_epoch() {
  "$PYTHON_BIN" - "$E15_METRICS" <<PY
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print(0)
    raise SystemExit(0)
try:
    payload = json.loads(p.read_text())
except Exception:
    print(0)
    raise SystemExit(0)
value = payload.get("stage_epoch_1based")
if value is None:
    value = int(payload.get("stage_epoch", -1)) + 1
print(int(value or 0))
PY
}

find_train_main_pid() {
  local config="$1"
  ps -eo pid=,ppid=,stat=,args= | "$PYTHON_BIN" -c '
import sys
config = sys.argv[1]
rows = []
for line in sys.stdin:
    if "-m safa.cli.train_g" not in line or config not in line:
        continue
    parts = line.strip().split(None, 3)
    if len(parts) < 4:
        continue
    pid, ppid, stat, args = parts
    argv0 = args.split(None, 1)[0] if args else ""
    if "python" not in argv0:
        continue
    rows.append((int(pid), int(ppid), stat, args))
if not rows:
    raise SystemExit(0)
# DataLoader workers inherit the command line and usually have the trainer PID as PPID.
# The lowest PID is the launcher Python process for this single-GPU job.
print(sorted(rows, key=lambda item: item[0])[0][0])
' "$config"
}

process_is_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

wait_process_exit() {
  local pid="$1"
  local deadline=$((SECONDS + GRACE_SECONDS))
  while process_is_alive "$pid"; do
    if (( SECONDS >= deadline )); then
      return 1
    fi
    sleep 10
  done
  return 0
}

wait_checkpoint_stable() {
  if [[ ! -f "$E15_LAST" ]]; then
    return 1
  fi
  local first second
  first="$(stat -c "%s:%Y" "$E15_LAST")"
  sleep 20
  second="$(stat -c "%s:%Y" "$E15_LAST")"
  [[ "$first" == "$second" ]]
}

write_e16_runtime_config() {
  local batch="$1"
  "$PYTHON_BIN" - "$E16_TEMPLATE_CONFIG" "$E16_RUNTIME_CONFIG" "$batch" <<PY
import sys
from pathlib import Path
import yaml
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
batch = int(sys.argv[3])
config = yaml.safe_load(src.read_text())
config["per_device_batch_size"] = batch
config["global_batch_size"] = batch
config["resume_from"] = ""
config["resume_optimizer_state"] = False
config.setdefault("generator", {})["sit_pretrained_path"] = "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_l_2_imagenet256.pt"
config["generator"]["sit_pretrained_state_key"] = ""
config["generator"]["sit_pretrained_source"] = "zhuyu-cs/MeanFlow ImageNet256 sit_l_2_meanflow_ema.pt"
config["generator"]["sit_pretrained_source_repo"] = "https://github.com/zhuyu-cs/MeanFlow; https://drive.google.com/drive/folders/1oWt6tdm5WIeVaZnBuUVheKIG3cNDffl9?usp=drive_link"
config.setdefault("stages", {}).setdefault("stage2", {}).setdefault("quality_eval", {})["output_dir"] = "artifacts/eval/e16_meanflow_sit_l2_face_mixed_2400ep/quality"
yaml.safe_dump(config, dst.open("w"), sort_keys=False, allow_unicode=True)
PY
}

find_e16_pid() {
  find_train_main_pid "$E16_RUNTIME_CONFIG" || true
  find_train_main_pid "$E16_TEMPLATE_CONFIG" || true
}

launch_e16() {
  local batch="$1"
  if [[ -f "$E16_CKPT_DIR/last.pt" ]]; then
    log "E16 checkpoint already exists at ${E16_CKPT_DIR}/last.pt; refusing to start duplicate training"
    exit 0
  fi
  if [[ ! -f "$E16_PRETRAIN" ]]; then
    log "missing E16 L/2 pretrained weight: ${E16_PRETRAIN}"
    exit 2
  fi
  local pretrain_size
  pretrain_size="$(stat -c "%s" "$E16_PRETRAIN")"
  log "using E16 L/2 pretrained weight: ${E16_PRETRAIN} size=${pretrain_size} source=${E16_PRETRAIN_SOURCE}"
  local existing
  existing="$(find_e16_pid | head -n 1 || true)"
  if [[ -n "$existing" ]] && process_is_alive "$existing"; then
    log "E16 already running pid=$existing; exiting"
    printf "%s\n" "$existing" > "$E16_PID_FILE"
    exit 0
  fi
  write_e16_runtime_config "$batch"
  log "starting E16 ${E16_EXP} with batch=${batch}; warm-starting from L/2 ImageNet256 pretrained weight"
  log "runtime_config=${E16_RUNTIME_CONFIG} log=${E16_LOG}"
  nohup env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True LD_LIBRARY_PATH="/home/k100/miniconda3/envs/pt210_cu130_fa4/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}" "$PYTHON_BIN" -m safa.cli.train_g --config "$E16_RUNTIME_CONFIG" > "$E16_LOG" 2>&1 &
  local pid="$!"
  printf "%s\n" "$pid" > "$E16_PID_FILE"
  log "E16 launched pid=${pid}"
}

log "watcher started target_epoch=${TARGET_EPOCH} poll_seconds=${POLL_SECONDS}"
if [[ ! -f "$E16_TEMPLATE_CONFIG" ]]; then
  log "missing E16 template config: ${E16_TEMPLATE_CONFIG}"
  exit 2
fi
if [[ ! -f "$E16_PRETRAIN" ]]; then
  log "missing E16 L/2 pretrained weight while waiting: ${E16_PRETRAIN}"
  exit 2
fi

while true; do
  epoch="$(read_e15_epoch)"
  e15_pid="$(find_train_main_pid "$E15_CONFIG" || true)"
  log "E15 stage_epoch_1based=${epoch} pid=${e15_pid:-none}"
  if (( epoch >= TARGET_EPOCH )); then
    break
  fi
  if [[ -z "${e15_pid:-}" ]]; then
    log "E15 training process is not running before target epoch; watcher keeps polling metrics"
  fi
  sleep "$POLL_SECONDS"
done

if [[ -n "${e15_pid:-}" ]] && process_is_alive "$e15_pid"; then
  log "target reached; sending SIGINT to E15 pid=${e15_pid}"
  kill -INT "$e15_pid" || true
  if ! wait_process_exit "$e15_pid"; then
    log "E15 did not exit after SIGINT within ${GRACE_SECONDS}s; sending SIGTERM"
    kill -TERM "$e15_pid" || true
    wait_process_exit "$e15_pid" || log "E15 still alive after SIGTERM; not escalating to kill -9"
  fi
else
  log "target reached and no live E15 main process found"
fi

if wait_checkpoint_stable; then
  log "E15 last checkpoint is present and stable: ${E15_LAST}"
else
  log "warning: E15 last checkpoint was not stable after stop check; continuing because metrics already reached target"
fi

launch_e16 64
sleep "$OOM_WATCH_SECONDS"
e16_pid="$(cat "$E16_PID_FILE" 2>/dev/null || true)"
if [[ -n "$e16_pid" ]] && process_is_alive "$e16_pid"; then
  log "E16 still running after ${OOM_WATCH_SECONDS}s with batch=64; watcher done"
  exit 0
fi

if grep -Eiq "out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED" "$E16_LOG" 2>/dev/null; then
  log "E16 batch=64 exited with OOM; retrying once with batch=32"
  E16_LOG="${E16_LOG_DIR}/${E16_EXP}_batch32_$(date +%Y%m%d_%H%M%S).log"
  launch_e16 32
  log "E16 batch=32 launched; watcher done"
  exit 0
fi

log "E16 process exited before ${OOM_WATCH_SECONDS}s without detected OOM; inspect ${E16_LOG}"
exit 1
