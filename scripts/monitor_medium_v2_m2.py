#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence


DEFAULT_STATUS = Path("artifacts/monitor/medium_v2_stage2_m2_gram_weighted_status.json")
DEFAULT_EVENTS = Path("artifacts/monitor/medium_v2_stage2_m2_gram_weighted_events.jsonl")
DEFAULT_LOG = Path("artifacts/monitor/medium_v2_stage2_m2_gram_weighted.log")
DEFAULT_RUN_DIR = Path("artifacts/checkpoints/g_medium_v2_stage2_m2_gram_weighted")
DEFAULT_METRICS = DEFAULT_RUN_DIR / "last_metrics.json"
DEFAULT_HISTORY = DEFAULT_RUN_DIR / "history.json"
DEFAULT_QUALITY_DIR = Path("artifacts/eval/g_medium_v2_stage2_m2_gram_weighted/quality")
DEFAULT_TRAIN_LOG = Path("artifacts/logs/train_g_medium_v2_stage2_m2_gram_weighted_gpu3_6.log")
DEFAULT_TMUX_SESSION = "train_g_medium_v2_stage2_m2_gram_weighted_gpu3_6"
DEFAULT_PROCESS_PATTERN = "safa.cli.train_g --config configs/medium_v2/train_g_medium_v2_stage2_m2_gram_weighted.yaml"
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
    history: Path = DEFAULT_HISTORY
    quality_dir: Path = DEFAULT_QUALITY_DIR
    train_log: Path = DEFAULT_TRAIN_LOG


class MonitorConfig(NamedTuple):
    tmux_session: str = DEFAULT_TMUX_SESSION
    process_pattern: str = DEFAULT_PROCESS_PATTERN
    gpu_indices: tuple[int, ...] = DEFAULT_GPU_INDICES


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_command(command: Sequence[str]) -> CommandResult:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_read_error": f"invalid json: {exc}"}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            detail = row.get("summary") or row.get("path") or ""
            handle.write(f"{row['time']} {row['type']}: {detail}\n")


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def int_value(value: Any) -> int | None:
    number = finite_number(value)
    if number is None or int(number) != number:
        return None
    return int(number)


def check_tmux(session: str, command_runner: Callable[[Sequence[str]], CommandResult]) -> bool:
    return command_runner(("tmux", "has-session", "-t", session)).returncode == 0


def capture_tmux(session: str, command_runner: Callable[[Sequence[str]], CommandResult]) -> str:
    result = command_runner(("tmux", "capture-pane", "-pt", session, "-S", "-120"))
    if result.returncode != 0:
        return ""
    return result.stdout


def list_processes(pattern: str, command_runner: Callable[[Sequence[str]], CommandResult]) -> list[dict[str, Any]]:
    result = command_runner(("pgrep", "-af", pattern))
    if result.returncode not in (0, 1):
        return []
    processes = []
    for line in result.stdout.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit():
            processes.append({"pid": int(pid), "cmd": cmd})
    return processes


def parse_gpu_csv(stdout: str) -> list[dict[str, int]]:
    gpus = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index, util, memory_used, memory_total = (int(part) for part in parts)
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "utilization_gpu_pct": util,
                "memory_used_mb": memory_used,
                "memory_total_mb": memory_total,
            }
        )
    return gpus


def query_gpus(indices: tuple[int, ...], command_runner: Callable[[Sequence[str]], CommandResult]) -> list[dict[str, int]]:
    result = command_runner(
        (
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(index) for index in indices),
        )
    )
    if result.returncode != 0:
        return []
    return parse_gpu_csv(result.stdout)


def parse_live_progress(text: str) -> dict[str, Any]:
    matches = list(re.finditer(r"train_g\s+stage(?P<stage>\d+)\s+epoch=(?P<epoch>\d+):\s*(?P<percent>\d+(?:\.\d+)?)%.*?(?P<batch>\d+)/(?:\s*)?(?P<total>\d+)", text))
    if not matches:
        return {}
    match = matches[-1]
    epoch0 = int(match.group("epoch"))
    return {
        "stage_epoch_0based": epoch0,
        "stage_epoch_1based": epoch0 + 1,
        "batch": int(match.group("batch")),
        "total_batches": int(match.group("total")),
        "percent": float(match.group("percent")),
    }


def load_latest_history_row(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if isinstance(payload, dict):
        rows = payload.get("history", payload.get("epochs", []))
    else:
        rows = payload
    if isinstance(rows, list) and rows and isinstance(rows[-1], dict):
        return rows[-1]
    return {}


def first_metric(metrics: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in metrics:
            number = finite_number(metrics[name])
            if number is not None:
                return number
    return None


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    latest_epoch: dict[str, Any] = {}
    epoch1 = int_value(metrics.get("stage_epoch_1based"))
    epoch0 = int_value(metrics.get("stage_epoch_0based"))
    if epoch1 is None and epoch0 is None:
        stage_epoch = int_value(metrics.get("stage_epoch"))
        if stage_epoch is not None:
            epoch0 = stage_epoch
            epoch1 = stage_epoch + 1
    if epoch0 is not None:
        latest_epoch["stage_epoch_0based"] = epoch0
    if epoch1 is not None:
        latest_epoch["stage_epoch_1based"] = epoch1

    validation = {
        "cosine": first_metric(
            metrics,
            (
                "validation_raw_latent_cosine_mean",
                "validation_ema_latent_cosine_mean",
                "validation_latent_cosine_mean",
                "validation_cosine",
            ),
        ),
        "single_face": first_metric(
            metrics,
            (
                "validation_raw_single_face_eq1_rate",
                "validation_ema_single_face_eq1_rate",
                "validation_single_face_eq1_rate",
                "validation_face_detection_rate",
            ),
        ),
        "source_preserved": first_metric(
            metrics,
            (
                "validation_raw_source_prediction_preserved",
                "validation_ema_source_prediction_preserved",
                "validation_source_prediction_preserved",
            ),
        ),
    }
    validation = {key: value for key, value in validation.items() if value is not None}
    extra = {key: metrics[key] for key in ("loss", "flow_loss_raw", "cycle_loss_raw", "repr_point_loss", "repr_relation_loss", "repr_loss") if key in metrics}
    return {"latest_epoch": latest_epoch, "validation": validation, "training": extra}


def _quality_epoch(path: Path) -> int:
    for text in (path.parent.name, path.name):
        match = re.search(r"epoch_(\d{4,})", text)
        if match:
            return int(match.group(1))
    return -1


def latest_quality_metrics(quality_dir: Path) -> dict[str, Any]:
    if not quality_dir.is_dir():
        return {}
    quality: dict[str, Any] = {}
    latest_epoch = -1
    for path in sorted(quality_dir.rglob("*.json")):
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        epoch = _quality_epoch(path)
        if epoch >= latest_epoch:
            latest_epoch = epoch
            quality["epoch"] = epoch if epoch >= 0 else None
        iqa = payload.get("iqa")
        if isinstance(iqa, dict) and str(iqa.get("method", "")).lower() == "niqe":
            value = finite_number(iqa.get("mean"))
            if value is not None:
                quality["niqe"] = value
                quality["niqe_path"] = path.as_posix()
        for source, target in (("fid", "fid"), ("kid", "kid"), ("kid_mean", "kid_mean"), ("kid_std", "kid_std")):
            value = finite_number(payload.get(source))
            if value is not None:
                quality[target] = value
                quality[f"{target}_path"] = path.as_posix()
    return quality


def read_error_signatures(log_path: Path, previous_status: dict[str, Any]) -> tuple[list[str], int]:
    try:
        size = log_path.stat().st_size
    except FileNotFoundError:
        return [], 0
    previous_offset = previous_status.get("state", {}).get("train_log_offset")
    if isinstance(previous_offset, int) and 0 <= previous_offset <= size:
        offset = previous_offset
    else:
        offset = 0 if size <= INITIAL_LOG_TAIL_BYTES else size - INITIAL_LOG_TAIL_BYTES
    with log_path.open("rb") as handle:
        handle.seek(offset)
        text = handle.read().decode("utf-8", errors="replace")
    errors = []
    for raw_line in text.replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if line and any(pattern.search(line) for pattern in ERROR_PATTERNS):
            errors.append(line[:500])
    return errors[-20:], size


def merge_epoch(metric_epoch: dict[str, Any], live_progress: dict[str, Any]) -> dict[str, Any]:
    if metric_epoch:
        result = dict(metric_epoch)
        result["source"] = "metrics"
        return result
    if live_progress:
        result = dict(live_progress)
        result["source"] = "tmux_capture"
        return result
    return {}


def build_events(status: dict[str, Any], previous_status: dict[str, Any], now: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    previous_state = previous_status.get("state", {}) if isinstance(previous_status, dict) else {}
    events: list[dict[str, Any]] = []
    epoch = status.get("latest_epoch", {}).get("stage_epoch_1based")
    previous_epoch = previous_state.get("last_epoch_1based")
    if isinstance(epoch, int) and (not isinstance(previous_epoch, int) or epoch > previous_epoch):
        events.append({"time": now, "type": "epoch_observed", "epoch_1based": epoch, "summary": json.dumps(status.get("latest_epoch", {}), sort_keys=True)})
    seen_errors = list(previous_state.get("seen_error_signatures", []))
    for line in status.get("error_signatures", []):
        if line not in seen_errors:
            events.append({"time": now, "type": "error_signature", "summary": line})
            seen_errors.append(line)
    quality = status.get("latest_quality", {})
    quality_paths = sorted(str(value) for key, value in quality.items() if key.endswith("_path"))
    seen_quality = list(previous_state.get("seen_quality_paths", []))
    for path in quality_paths:
        if path not in seen_quality:
            events.append({"time": now, "type": "quality_metric", "path": path, "summary": path})
            seen_quality.append(path)
    state = {
        "last_epoch_1based": epoch if isinstance(epoch, int) else previous_epoch,
        "seen_error_signatures": seen_errors[-256:],
        "seen_quality_paths": seen_quality[-256:],
        "train_log_offset": status.get("train_log_offset", previous_state.get("train_log_offset", 0)),
    }
    return events, state


def run_once(paths: MonitorPaths, config: MonitorConfig, command_runner: Callable[[Sequence[str]], CommandResult] = run_command) -> int:
    now = utc_now()
    previous_status = read_json(paths.status)
    tmux_alive = check_tmux(config.tmux_session, command_runner)
    tmux_capture = capture_tmux(config.tmux_session, command_runner) if tmux_alive else ""
    processes = list_processes(config.process_pattern, command_runner)
    metrics = read_json(paths.metrics)
    if not metrics:
        metrics = load_latest_history_row(paths.history)
    metric_summary = summarize_metrics(metrics if isinstance(metrics, dict) else {})
    errors, log_offset = read_error_signatures(paths.train_log, previous_status if isinstance(previous_status, dict) else {})
    status = {
        "time": now,
        "tmux_alive": tmux_alive,
        "process_alive": bool(processes),
        "process_count": len(processes),
        "gpus": query_gpus(config.gpu_indices, command_runner),
        "latest_epoch": merge_epoch(metric_summary.get("latest_epoch", {}), parse_live_progress(tmux_capture)),
        "latest_validation": metric_summary.get("validation", {}),
        "latest_quality": latest_quality_metrics(paths.quality_dir),
        "training_metrics": metric_summary.get("training", {}),
        "error_signatures": errors,
        "train_log_offset": log_offset,
        "paths": {"status": paths.status.as_posix(), "events": paths.events.as_posix(), "log": paths.log.as_posix(), "train_log": paths.train_log.as_posix(), "metrics": paths.metrics.as_posix(), "history": paths.history.as_posix(), "quality_dir": paths.quality_dir.as_posix()},
    }
    events, state = build_events(status, previous_status if isinstance(previous_status, dict) else {}, now)
    status["new_event_count"] = len(events)
    status["new_event_types"] = [event["type"] for event in events]
    status["state"] = state
    write_json_atomic(paths.status, status)
    append_jsonl(paths.events, events)
    append_log(paths.log, events)
    return 0


def loop(paths: MonitorPaths, config: MonitorConfig, interval_seconds: int) -> int:
    while True:
        run_once(paths, config)
        time.sleep(interval_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight monitor for medium_v2 stage2 M2 gram-weighted training.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--quality-dir", type=Path, default=DEFAULT_QUALITY_DIR)
    parser.add_argument("--train-log", type=Path, default=DEFAULT_TRAIN_LOG)
    parser.add_argument("--tmux-session", default=DEFAULT_TMUX_SESSION)
    parser.add_argument("--process-pattern", default=DEFAULT_PROCESS_PATTERN)
    parser.add_argument("--gpus", default=",".join(str(index) for index in DEFAULT_GPU_INDICES))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = MonitorPaths(status=args.status, events=args.events, log=args.log, metrics=args.metrics, history=args.history, quality_dir=args.quality_dir, train_log=args.train_log)
    config = MonitorConfig(tmux_session=args.tmux_session, process_pattern=args.process_pattern, gpu_indices=tuple(int(part) for part in args.gpus.split(",") if part.strip()))
    if args.once:
        return run_once(paths, config)
    return loop(paths, config, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
