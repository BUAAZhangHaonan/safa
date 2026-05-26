#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple, Sequence


DEFAULT_STATUS = Path("artifacts/monitor/medium_v1_stage2_supervisor_status.json")
DEFAULT_EVENTS = Path("artifacts/monitor/medium_v1_stage2_supervisor_events.jsonl")
DEFAULT_LOG = Path("artifacts/monitor/medium_v1_stage2_supervisor.log")
DEFAULT_M0_METRICS = Path("artifacts/checkpoints/g_medium_v1_stage2_m0/last_metrics.json")
DEFAULT_M0_LAST = Path("artifacts/checkpoints/g_medium_v1_stage2_m0/last.pt")
DEFAULT_M0_BEST = Path("artifacts/checkpoints/g_medium_v1_stage2_m0/best_raw_utility.pt")
DEFAULT_M1_LAST = Path("artifacts/checkpoints/g_medium_v1_stage2_m1_uw/last.pt")
DEFAULT_M1_BEST = Path("artifacts/checkpoints/g_medium_v1_stage2_m1_uw/best_raw_utility.pt")
DEFAULT_M0_TRAIN_LOG = Path("artifacts/logs/train_g_medium_v1_stage2_m0_gpu3_6.log")
DEFAULT_M1_TRAIN_LOG = Path("artifacts/logs/train_g_medium_v1_stage2_m1_uw_gpu3_6.log")
DEFAULT_M0_SESSION = "train_g_medium_v1_stage2_m0_gpu3_6"
DEFAULT_M1_SESSION = "train_g_medium_v1_stage2_m1_uw_gpu3_6"
DEFAULT_M0_PROCESS_PATTERN = "safa.cli.train_g --config configs/medium_v1/train_g_medium_v1_stage2_m0.yaml"
DEFAULT_M1_CONFIG = "configs/medium_v1/train_g_medium_v1_stage2_m1_uw.yaml"
DEFAULT_GPUS = "3,4,5,6"
DEFAULT_PYTHON = "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"
REQUIRED_M0_EPOCHS = 200


class CommandResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


class SupervisorPaths(NamedTuple):
    status: Path = DEFAULT_STATUS
    events: Path = DEFAULT_EVENTS
    log: Path = DEFAULT_LOG
    m0_metrics: Path = DEFAULT_M0_METRICS
    m0_last_checkpoint: Path = DEFAULT_M0_LAST
    m0_best_checkpoint: Path = DEFAULT_M0_BEST
    m1_last_checkpoint: Path = DEFAULT_M1_LAST
    m1_best_checkpoint: Path = DEFAULT_M1_BEST
    m0_train_log: Path = DEFAULT_M0_TRAIN_LOG
    m1_train_log: Path = DEFAULT_M1_TRAIN_LOG


class SupervisorConfig(NamedTuple):
    m0_session: str = DEFAULT_M0_SESSION
    m1_session: str = DEFAULT_M1_SESSION
    m0_process_pattern: str = DEFAULT_M0_PROCESS_PATTERN
    m1_config: str = DEFAULT_M1_CONFIG
    gpu_visible_devices: str = DEFAULT_GPUS
    python_bin: str = DEFAULT_PYTHON
    required_m0_epochs: int = REQUIRED_M0_EPOCHS


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
            detail = event.get("summary") or event.get("reason") or ""
            handle.write(f"{event['time']} {event['type']}: {detail}\n")


def check_tmux(session: str, command_runner: Callable[[Sequence[str]], CommandResult]) -> bool:
    return command_runner(("tmux", "has-session", "-t", session)).returncode == 0


def list_processes(pattern: str, command_runner: Callable[[Sequence[str]], CommandResult]) -> list[dict]:
    result = command_runner(("pgrep", "-af", pattern))
    if result.returncode not in (0, 1):
        return []
    processes = []
    for line in result.stdout.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit():
            processes.append({"pid": int(pid), "cmd": cmd})
    return processes


def m0_epoch(metrics: dict) -> int | None:
    for key in ("stage_epoch_1based", "epoch_1based", "stage_epoch"):
        value = metrics.get(key)
        if isinstance(value, int):
            return value
    return None


def build_m1_command(paths: SupervisorPaths, config: SupervisorConfig) -> str:
    log_path = paths.m1_train_log.as_posix()
    return (
        f"cd {Path.cwd().as_posix()} && "
        f"mkdir -p {Path(log_path).parent.as_posix()} && "
        f"CUDA_VISIBLE_DEVICES={config.gpu_visible_devices} "
        "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HTTP_PROXY= HTTPS_PROXY= PYTHONPATH=src "
        f"{config.python_bin} scripts/guarded_run.py --max-ram-fraction 0.90 -- "
        f"{config.python_bin} -m torch.distributed.run --standalone --nproc_per_node=4 "
        f"-m safa.cli.train_g --config {config.m1_config} 2>&1 | tee {log_path}"
    )


def launch_m1(paths: SupervisorPaths, config: SupervisorConfig, command_runner: Callable[[Sequence[str]], CommandResult]) -> CommandResult:
    return command_runner(("tmux", "new-session", "-d", "-s", config.m1_session, build_m1_command(paths, config)))


def decide_and_act(
    paths: SupervisorPaths,
    config: SupervisorConfig,
    command_runner: Callable[[Sequence[str]], CommandResult],
) -> tuple[dict, list[dict]]:
    now = utc_now()
    previous = read_json(paths.status)
    metrics = read_json(paths.m0_metrics)
    epoch = m0_epoch(metrics)
    m0_complete = isinstance(epoch, int) and epoch >= config.required_m0_epochs
    m0_tmux_alive = check_tmux(config.m0_session, command_runner)
    m1_tmux_alive = check_tmux(config.m1_session, command_runner)
    m0_processes = list_processes(config.m0_process_pattern, command_runner)
    m0_checkpoints_present = paths.m0_last_checkpoint.exists() and paths.m0_best_checkpoint.exists()
    m1_outputs_present = paths.m1_last_checkpoint.exists() or paths.m1_best_checkpoint.exists()
    events: list[dict] = []

    status = {
        "time": now,
        "state": "waiting_m0",
        "m0_epoch_1based": epoch,
        "required_m0_epochs": config.required_m0_epochs,
        "m0_complete": m0_complete,
        "m0_tmux_alive": m0_tmux_alive,
        "m0_process_count": len(m0_processes),
        "m0_checkpoints_present": m0_checkpoints_present,
        "m1_tmux_alive": m1_tmux_alive,
        "m1_outputs_present": m1_outputs_present,
        "m1_session": config.m1_session,
        "m1_config": config.m1_config,
        "gpu_visible_devices": config.gpu_visible_devices,
        "paths": {"events": paths.events.as_posix(), "log": paths.log.as_posix()},
    }

    if m1_tmux_alive:
        status["state"] = "m1_running"
        return status, events
    if m1_outputs_present:
        status["state"] = "m1_exists"
        status["reason"] = "m1_checkpoint_exists"
        return status, events

    if not m0_complete:
        if not m0_tmux_alive:
            status["state"] = "blocked"
            status["reason"] = "m0_ended_before_completion"
            if previous.get("state") != "blocked" or previous.get("reason") != status["reason"]:
                events.append(
                    {
                        "time": now,
                        "type": "blocked",
                        "reason": status["reason"],
                        "summary": f"M0 ended before epoch {config.required_m0_epochs}; last_epoch={epoch}",
                    }
                )
        return status, events

    if not m0_checkpoints_present:
        missing = [
            path.as_posix()
            for path in (paths.m0_last_checkpoint, paths.m0_best_checkpoint)
            if not path.exists()
        ]
        status["state"] = "blocked"
        status["reason"] = "m0_checkpoints_missing"
        status["missing_checkpoints"] = missing
        if previous.get("state") != "blocked" or previous.get("reason") != status["reason"]:
            events.append(
                {
                    "time": now,
                    "type": "blocked",
                    "reason": status["reason"],
                    "summary": "M0 completed but required checkpoints are missing: " + ", ".join(missing),
                }
            )
        return status, events

    launch = launch_m1(paths, config, command_runner)
    if launch.returncode == 0:
        status["state"] = "m1_started"
        status["launch_command"] = build_m1_command(paths, config)
        events.append(
            {
                "time": now,
                "type": "m1_started",
                "summary": f"started {config.m1_session} on CUDA_VISIBLE_DEVICES={config.gpu_visible_devices}",
            }
        )
    else:
        status["state"] = "blocked"
        status["reason"] = "m1_launch_failed"
        status["launch_stderr"] = launch.stderr[-1000:]
        events.append(
            {
                "time": now,
                "type": "error",
                "reason": status["reason"],
                "summary": launch.stderr[-500:] or launch.stdout[-500:] or "tmux new-session failed",
            }
        )
    return status, events


def run_once(
    paths: SupervisorPaths,
    config: SupervisorConfig,
    command_runner: Callable[[Sequence[str]], CommandResult] = run_command,
) -> int:
    status, events = decide_and_act(paths, config, command_runner)
    write_json_atomic(paths.status, status)
    append_jsonl(paths.events, events)
    append_log(paths.log, events)
    return 0


def loop(paths: SupervisorPaths, config: SupervisorConfig, interval_seconds: int) -> int:
    while True:
        run_once(paths, config)
        time.sleep(interval_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervise medium_v1 stage2 M0 to M1 progression.")
    parser.add_argument("--once", action="store_true", help="run one supervisor pass and exit")
    parser.add_argument("--interval", type=int, default=600, help="seconds between supervisor passes")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--m0-metrics", type=Path, default=DEFAULT_M0_METRICS)
    parser.add_argument("--m0-last-checkpoint", type=Path, default=DEFAULT_M0_LAST)
    parser.add_argument("--m0-best-checkpoint", type=Path, default=DEFAULT_M0_BEST)
    parser.add_argument("--m1-last-checkpoint", type=Path, default=DEFAULT_M1_LAST)
    parser.add_argument("--m1-best-checkpoint", type=Path, default=DEFAULT_M1_BEST)
    parser.add_argument("--m0-train-log", type=Path, default=DEFAULT_M0_TRAIN_LOG)
    parser.add_argument("--m1-train-log", type=Path, default=DEFAULT_M1_TRAIN_LOG)
    parser.add_argument("--m0-session", default=DEFAULT_M0_SESSION)
    parser.add_argument("--m1-session", default=DEFAULT_M1_SESSION)
    parser.add_argument("--m0-process-pattern", default=DEFAULT_M0_PROCESS_PATTERN)
    parser.add_argument("--m1-config", default=DEFAULT_M1_CONFIG)
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = SupervisorPaths(
        status=args.status,
        events=args.events,
        log=args.log,
        m0_metrics=args.m0_metrics,
        m0_last_checkpoint=args.m0_last_checkpoint,
        m0_best_checkpoint=args.m0_best_checkpoint,
        m1_last_checkpoint=args.m1_last_checkpoint,
        m1_best_checkpoint=args.m1_best_checkpoint,
        m0_train_log=args.m0_train_log,
        m1_train_log=args.m1_train_log,
    )
    config = SupervisorConfig(
        m0_session=args.m0_session,
        m1_session=args.m1_session,
        m0_process_pattern=args.m0_process_pattern,
        m1_config=args.m1_config,
        gpu_visible_devices=args.gpus,
        python_bin=args.python_bin,
    )
    if args.once:
        return run_once(paths, config)
    return loop(paths, config, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
