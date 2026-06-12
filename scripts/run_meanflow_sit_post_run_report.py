#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/home/k100/miniconda3/envs/pt210_cu130_fa4/bin/python"
DEFAULT_TRAIN_SESSION = "safa_e11_meanflow_sit_k100_200ep_20260613_053015"
DEFAULT_OUTPUT_JSON = Path("artifacts/reports/e11_meanflow_sit_stage1_report.json")
DEFAULT_OUTPUT_MD = Path("artifacts/reports/e11_meanflow_sit_stage1_report.md")
DEFAULT_SLEEP_SECONDS = 300


@dataclass(frozen=True)
class MonitorPlan:
    repo_root: Path
    python: str
    session: str
    monitor_session: str
    timestamp: str
    sleep_seconds: int
    timeout_seconds: int
    output_json: Path
    output_md: Path
    log_path: Path
    dry_run: bool
    wait_only: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for the K100 e11 MeanFlow-SiT tmux run to finish, then write the Stage1 report.",
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--session", default=DEFAULT_TRAIN_SESSION)
    parser.add_argument("--monitor-session", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--sleep-seconds", type=int, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--wait-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_monitor_plan(args: argparse.Namespace) -> MonitorPlan:
    timestamp = str(args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    monitor_session = str(args.monitor_session or f"safa_e11_meanflow_sit_report_monitor_{timestamp}")
    log_path = Path(args.log) if args.log is not None else Path(f"artifacts/logs/e11_meanflow_sit_report_monitor_{timestamp}.log")
    return MonitorPlan(
        repo_root=Path(args.repo_root),
        python=str(args.python),
        session=str(args.session),
        monitor_session=monitor_session,
        timestamp=timestamp,
        sleep_seconds=int(args.sleep_seconds),
        timeout_seconds=int(args.timeout_seconds),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        log_path=log_path,
        dry_run=bool(args.dry_run),
        wait_only=bool(args.wait_only),
    )


def build_report_command(plan: MonitorPlan) -> list[str]:
    return [
        plan.python,
        "scripts/run_meanflow_sit_stage1_report.py",
        "--repo-root",
        str(plan.repo_root),
        "--runs",
        "e11",
        "e8",
        "--train-session",
        "",
        "--output-json",
        str(plan.output_json),
        "--output-md",
        str(plan.output_md),
    ]


def build_wait_command(plan: MonitorPlan) -> list[str]:
    return [
        plan.python,
        "scripts/run_meanflow_sit_post_run_report.py",
        "--wait-only",
        "--repo-root",
        str(plan.repo_root),
        "--python",
        plan.python,
        "--session",
        plan.session,
        "--sleep-seconds",
        str(plan.sleep_seconds),
        "--timeout-seconds",
        str(plan.timeout_seconds),
        "--output-json",
        str(plan.output_json),
        "--output-md",
        str(plan.output_md),
    ]


def build_tmux_start_command(plan: MonitorPlan) -> list[str]:
    wait_command = shlex.join(build_wait_command(plan))
    shell_command = " && ".join(
        [
            f"cd {shlex.quote(str(plan.repo_root))}",
            f"mkdir -p {shlex.quote(str(plan.log_path.parent))}",
            f"{wait_command} > {shlex.quote(str(plan.log_path))} 2>&1",
        ]
    )
    return ["tmux", "new-session", "-d", "-s", plan.monitor_session, shell_command]


def render_dry_run(plan: MonitorPlan) -> str:
    lines = ["DRY RUN: no tmux monitor started and no report written."]
    lines.append(f"training_session: {plan.session}")
    lines.append(f"monitor_session: {plan.monitor_session}")
    lines.append(f"sleep_seconds: {plan.sleep_seconds}")
    lines.append(f"output_json: {plan.output_json}")
    lines.append(f"output_md: {plan.output_md}")
    lines.append(f"tmux_command: {shlex.join(build_tmux_start_command(plan))}")
    return "\n".join(lines) + "\n"


def wait_for_training_then_report(plan: MonitorPlan) -> None:
    start = time.monotonic()
    while tmux_session_exists(plan.session):
        if plan.timeout_seconds > 0 and time.monotonic() - start > plan.timeout_seconds:
            raise TimeoutError(f"timed out waiting for tmux session: {plan.session}")
        time.sleep(plan.sleep_seconds)
    subprocess.run(build_report_command(plan), cwd=plan.repo_root, check=True)


def validate_monitor_start(plan: MonitorPlan) -> None:
    if tmux_session_exists(plan.monitor_session):
        raise RuntimeError(f"tmux monitor session already exists: {plan.monitor_session}")
    resolved_log = _resolve(plan.repo_root, plan.log_path)
    if resolved_log.exists():
        raise FileExistsError(f"refusing to overwrite existing monitor log: {resolved_log}")


def start_tmux_monitor(plan: MonitorPlan) -> None:
    validate_monitor_start(plan)
    _resolve(plan.repo_root, plan.log_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_tmux_start_command(plan), cwd=plan.repo_root, check=True)


def tmux_session_exists(session: str) -> bool:
    if not session:
        return False
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_monitor_plan(args)
    if plan.dry_run:
        print(render_dry_run(plan), end="")
        return 0
    if plan.wait_only:
        wait_for_training_then_report(plan)
        return 0
    start_tmux_monitor(plan)
    print(f"started monitor_session={plan.monitor_session} log={plan.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
