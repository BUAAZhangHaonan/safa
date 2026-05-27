#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple, Sequence


DEFAULT_STATUS = Path("artifacts/monitor/medium_v1_stage2_m0_status.json")
DEFAULT_EVENTS = Path("artifacts/monitor/medium_v1_stage2_m0_events.jsonl")
DEFAULT_LOG = Path("artifacts/monitor/medium_v1_stage2_m0.log")
DEFAULT_METRICS = Path("artifacts/checkpoints/g_medium_v1_stage2_m0/last_metrics.json")
DEFAULT_QUALITY_DIR = Path("artifacts/eval/g_medium_v1_stage2_m0/quality")
DEFAULT_TRAIN_LOG = Path("artifacts/logs/train_g_medium_v1_stage2_m0_gpu3_6.log")
DEFAULT_TMUX_SESSION = "train_g_medium_v1_stage2_m0_gpu3_6"
DEFAULT_PROCESS_PATTERN = "safa.cli.train_g --config configs/medium_v1/train_g_medium_v1_stage2_m0.yaml"
DEFAULT_GPU_INDICES = (3, 4, 5, 6)
ERROR_PATTERNS = (
    re.compile(r"traceback", re.IGNORECASE),
    re.compile(r"runtimeerror", re.IGNORECASE),
    re.compile(r"cuda out of memory", re.IGNORECASE),
    re.compile(r"outofmemoryerror", re.IGNORECASE),
    re.compile(r"\bexception\b", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_])nan(?![A-Za-z0-9_])", re.IGNORECASE),
    re.compile(r"\bkilled\b", re.IGNORECASE),
)
MAX_SEEN_ITEMS = 256
INITIAL_LOG_TAIL_BYTES = 200_000


class CommandResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


class MonitorPaths(NamedTuple):
    status: Path = DEFAULT_STATUS
    events: Path = DEFAULT_EVENTS
    log: Path = DEFAULT_LOG
    metrics: Path = DEFAULT_METRICS
    quality_dir: Path = DEFAULT_QUALITY_DIR
    train_log: Path = DEFAULT_TRAIN_LOG


class MonitorConfig(NamedTuple):
    tmux_session: str = DEFAULT_TMUX_SESSION
    process_pattern: str = DEFAULT_PROCESS_PATTERN
    gpu_indices: tuple[int, ...] = DEFAULT_GPU_INDICES
    gpu_memory_high_ratio: float = 0.98
    gpu_memory_low_mb: int = 1000
    gpu_idle_util_pct: int = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_command(command: Sequence[str]) -> CommandResult:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_read_error": f"invalid json: {exc}"}


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_log(path: Path, events: list[dict]) -> None:
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            detail = event.get("summary") or event.get("path") or event.get("reason", "")
            handle.write(f"{event['time']} {event['type']}: {detail}\n")


def check_tmux(session: str, command_runner: Callable[[Sequence[str]], CommandResult]) -> bool:
    result = command_runner(("tmux", "has-session", "-t", session))
    return result.returncode == 0


def list_processes(pattern: str, command_runner: Callable[[Sequence[str]], CommandResult]) -> list[dict]:
    result = command_runner(("pgrep", "-af", pattern))
    if result.returncode not in (0, 1):
        return []
    processes = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, cmd = line.partition(" ")
        if pid.isdigit():
            processes.append({"pid": int(pid), "cmd": cmd})
    return processes


def parse_gpu_csv(stdout: str) -> list[dict]:
    gpus = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index, memory_used, memory_total, util = (int(part) for part in parts)
        except ValueError:
            continue
        ratio = memory_used / memory_total if memory_total else 0.0
        gpus.append(
            {
                "index": index,
                "memory_used_mb": memory_used,
                "memory_total_mb": memory_total,
                "memory_used_ratio": round(ratio, 4),
                "utilization_gpu_pct": util,
            }
        )
    return gpus


def query_gpus(indices: tuple[int, ...], command_runner: Callable[[Sequence[str]], CommandResult]) -> list[dict]:
    result = command_runner(
        (
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(index) for index in indices),
        )
    )
    if result.returncode != 0:
        return []
    return parse_gpu_csv(result.stdout)


def find_quality_jsons(quality_dir: Path) -> list[str]:
    if not quality_dir.exists():
        return []
    paths = []
    for path in quality_dir.rglob("*.json"):
        name = path.name.lower()
        if "fid" in name or "kid" in name or "distribution" in name:
            paths.append(path.as_posix())
    return sorted(paths)


def read_new_error_lines(log_path: Path, previous_state: dict) -> tuple[list[str], int]:
    try:
        size = log_path.stat().st_size
    except FileNotFoundError:
        return [], 0
    previous_offset = previous_state.get("train_log_offset")
    if isinstance(previous_offset, int) and 0 <= previous_offset <= size:
        offset = previous_offset
    else:
        offset = 0 if size <= INITIAL_LOG_TAIL_BYTES else size - INITIAL_LOG_TAIL_BYTES
    with log_path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
    text = chunk.decode("utf-8", errors="replace")
    errors = []
    for raw_line in text.replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if line and any(pattern.search(line) for pattern in ERROR_PATTERNS):
            errors.append(line[:500])
    return errors[-20:], size


def summarize_metrics(metrics: dict) -> dict:
    keys = (
        "stage",
        "stage_epoch",
        "stage_epoch_0based",
        "stage_epoch_1based",
        "loss",
        "flow_matching_mse",
        "cycle_loss_normalized",
        "quality_raw_niqe_mean",
        "validation_ema_source_prediction_preserved",
        "validation_ema_face_detection_rate",
        "validation_source_prediction_preserved",
        "validation_face_detection_rate",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def assess_gpu_abnormal(snapshot: dict, config: MonitorConfig) -> tuple[bool, list[str]]:
    gpus = snapshot.get("gpus", [])
    by_index = {gpu.get("index"): gpu for gpu in gpus}
    reasons = []
    for index in config.gpu_indices:
        gpu = by_index.get(index)
        if gpu is None:
            reasons.append(f"gpu{index}:missing")
            continue
        if gpu.get("memory_used_ratio", 0.0) >= config.gpu_memory_high_ratio:
            reasons.append(f"gpu{index}:memory_high:{gpu.get('memory_used_mb')}MB/{gpu.get('memory_total_mb')}MB")
        if snapshot.get("processes") and gpu.get("memory_used_mb", 0) <= config.gpu_memory_low_mb:
            reasons.append(f"gpu{index}:memory_low:{gpu.get('memory_used_mb')}MB")
    if snapshot.get("processes") and gpus and all(gpu.get("utilization_gpu_pct", 0) <= config.gpu_idle_util_pct for gpu in gpus):
        reasons.append("gpu3-6:all_idle")
    return bool(reasons), reasons


def collect_snapshot(paths: MonitorPaths, config: MonitorConfig, previous_status: dict, command_runner: Callable[[Sequence[str]], CommandResult]) -> dict:
    previous_state = previous_status.get("state", {}) if isinstance(previous_status, dict) else {}
    metrics = read_json(paths.metrics)
    snapshot = {
        "tmux_alive": check_tmux(config.tmux_session, command_runner),
        "processes": list_processes(config.process_pattern, command_runner),
        "last_metrics": summarize_metrics(metrics),
        "last_metrics_path": paths.metrics.as_posix(),
        "quality_dir": paths.quality_dir.as_posix(),
        "quality_jsons": find_quality_jsons(paths.quality_dir),
        "gpus": query_gpus(config.gpu_indices, command_runner),
    }
    errors, log_offset = read_new_error_lines(paths.train_log, previous_state)
    snapshot["errors"] = errors
    snapshot["train_log_path"] = paths.train_log.as_posix()
    snapshot["train_log_offset"] = log_offset
    gpu_abnormal, gpu_reasons = assess_gpu_abnormal(snapshot, config)
    snapshot["gpu_abnormal"] = gpu_abnormal
    snapshot["gpu_abnormal_reasons"] = gpu_reasons
    return snapshot


def _bounded_unique(values: list[str]) -> list[str]:
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen[-MAX_SEEN_ITEMS:]


def build_events(snapshot: dict, previous_status: dict, now: str) -> tuple[list[dict], dict]:
    previous_state = previous_status.get("state", {}) if isinstance(previous_status, dict) else {}
    events: list[dict] = []
    metrics = snapshot.get("last_metrics", {})
    epoch = metrics.get("stage_epoch_1based")
    previous_epoch = previous_state.get("last_epoch_1based")
    if isinstance(epoch, int) and (not isinstance(previous_epoch, int) or epoch > previous_epoch):
        events.append({"time": now, "type": "epoch_completed", "epoch_1based": epoch, "summary": json.dumps(metrics, sort_keys=True)})

    seen_quality = list(previous_state.get("seen_quality_jsons", []))
    for path in snapshot.get("quality_jsons", []):
        if path not in seen_quality:
            events.append({"time": now, "type": "quality_json", "path": path, "summary": path})
            seen_quality.append(path)

    seen_errors = list(previous_state.get("seen_error_signatures", []))
    for line in snapshot.get("errors", []):
        if line not in seen_errors:
            events.append({"time": now, "type": "error_keyword", "summary": line})
            seen_errors.append(line)

    previous_health = previous_state.get("last_health", {})
    tmux_alive = bool(snapshot.get("tmux_alive"))
    process_alive = bool(snapshot.get("processes"))
    gpu_abnormal = bool(snapshot.get("gpu_abnormal"))
    if previous_health.get("tmux_alive", True) and not tmux_alive:
        events.append({"time": now, "type": "tmux_dead", "summary": "training tmux session is not alive"})
    if previous_health.get("process_alive", True) and not process_alive and tmux_alive:
        events.append({"time": now, "type": "process_dead", "summary": "training process is not alive"})
    if not previous_health.get("gpu_abnormal", False) and gpu_abnormal:
        events.append({"time": now, "type": "gpu_abnormal", "summary": "; ".join(snapshot.get("gpu_abnormal_reasons", []))})

    state = {
        "last_epoch_1based": epoch if isinstance(epoch, int) else previous_epoch,
        "seen_quality_jsons": _bounded_unique(seen_quality),
        "seen_error_signatures": _bounded_unique(seen_errors),
        "train_log_offset": snapshot.get("train_log_offset", previous_state.get("train_log_offset", 0)),
        "last_health": {"tmux_alive": tmux_alive, "process_alive": process_alive, "gpu_abnormal": gpu_abnormal},
    }
    return events, state


def compact_status(snapshot: dict, state: dict, events: list[dict], now: str) -> dict:
    return {
        "time": now,
        "tmux_alive": snapshot.get("tmux_alive"),
        "process_alive": bool(snapshot.get("processes")),
        "process_count": len(snapshot.get("processes", [])),
        "last_metrics": snapshot.get("last_metrics", {}),
        "quality_dir": snapshot.get("quality_dir"),
        "quality_json_count": len(snapshot.get("quality_jsons", [])),
        "latest_quality_jsons": snapshot.get("quality_jsons", [])[-5:],
        "gpus": snapshot.get("gpus", []),
        "gpu_abnormal": snapshot.get("gpu_abnormal"),
        "gpu_abnormal_reasons": snapshot.get("gpu_abnormal_reasons", []),
        "new_event_count": len(events),
        "new_event_types": [event["type"] for event in events],
        "paths": {"events": DEFAULT_EVENTS.as_posix(), "log": DEFAULT_LOG.as_posix()},
        "state": state,
    }


def run_once(paths: MonitorPaths, config: MonitorConfig, command_runner: Callable[[Sequence[str]], CommandResult] = run_command) -> int:
    now = utc_now()
    previous_status = read_json(paths.status)
    snapshot = collect_snapshot(paths, config, previous_status, command_runner)
    events, state = build_events(snapshot, previous_status, now)
    status = compact_status(snapshot, state, events, now)
    write_json_atomic(paths.status, status)
    append_jsonl(paths.events, events)
    append_log(paths.log, events)
    return 0


def loop(paths: MonitorPaths, config: MonitorConfig, interval_seconds: int) -> int:
    while True:
        run_once(paths, config)
        time.sleep(interval_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact monitor for medium_v1 stage2 M0 training.")
    parser.add_argument("--once", action="store_true", help="run one monitor pass and exit")
    parser.add_argument("--interval", type=int, default=300, help="seconds between monitor passes")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--quality-dir", type=Path, default=DEFAULT_QUALITY_DIR)
    parser.add_argument("--train-log", type=Path, default=DEFAULT_TRAIN_LOG)
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--process-pattern", default=DEFAULT_PROCESS_PATTERN)
    parser.add_argument("--gpus", default=",".join(str(index) for index in DEFAULT_GPU_INDICES))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    gpu_indices = tuple(int(part) for part in args.gpus.split(",") if part.strip())
    paths = MonitorPaths(status=args.status, events=args.events, log=args.log, metrics=args.metrics, quality_dir=args.quality_dir, train_log=args.train_log)
    config = MonitorConfig(tmux_session=args.tmux_session, process_pattern=args.process_pattern, gpu_indices=gpu_indices)
    if args.once:
        return run_once(paths, config)
    return loop(paths, config, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
