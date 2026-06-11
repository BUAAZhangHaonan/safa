#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_MEANFLOW_SESSION = "safa_meanflow_e9_gpu6_200ep_20260612_034831"
DEFAULT_MEANFLOW_LOG = Path("artifacts/logs/e9_meanflow_200ep_gpu6_20260612_034831_b4.log")
DEFAULT_MEANFLOW_CHECKPOINTS = (
    Path("artifacts/checkpoints/g_medium_v2_meanflow_200ep/best_stage2.pt"),
    Path("artifacts/checkpoints/g_medium_v2_meanflow_200ep/last.pt"),
)
DEFAULT_DDIM_CONFIG = Path("configs/medium_v2/experiments/e10_ddim_200ep.yaml")
DEFAULT_DDIM_CHECKPOINTS = (
    Path("artifacts/checkpoints/g_medium_v2_ddim_200ep/best_stage2.pt"),
    Path("artifacts/checkpoints/g_medium_v2_ddim_200ep/last.pt"),
)
DEFAULT_PYTHON = "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"
DEFAULT_POLL_SECONDS = 300
DEFAULT_REPORT_SUMMARY = Path("artifacts/eval/generator_comparison_summary.json")
DEFAULT_COMPARISON_SCRIPT = Path("scripts/run_generator_comparison_report.py")
ERROR_KEYWORDS = ("OutOfMemoryError", "RuntimeError", "exit_status=1", "CUDA error")


@dataclass(frozen=True)
class StagePlan:
    name: str
    session: str
    log_path: Path
    checkpoints: tuple[Path, ...]


@dataclass(frozen=True)
class QueuePlan:
    repo_root: Path
    timestamp: str
    poll_seconds: int
    python: str
    queue_log: Path
    dry_run: bool
    skip_report: bool
    start_ddim_if_meanflow_done: bool
    meanflow: StagePlan
    ddim: StagePlan
    ddim_config: Path
    comparison_script: Path
    report_summary: Path
    report_device: str


class QueueLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def append_output(self, label: str, output: str) -> None:
        if not output:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{label}]\n")
            handle.write(output)
            if not output.endswith("\n"):
                handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue e10 DDIM training on GPU6 after e9 MeanFlow finishes, then run the generator comparison report.",
        epilog=(
            "Use --dry-run to print the planned waits and commands without writing artifacts, "
            "starting tmux sessions, or running training."
        ),
    )
    parser.add_argument("--meanflow-session", default=DEFAULT_MEANFLOW_SESSION)
    parser.add_argument("--meanflow-log", type=Path, default=DEFAULT_MEANFLOW_LOG)
    parser.add_argument("--ddim-session", default=None)
    parser.add_argument("--ddim-log", type=Path, default=None)
    parser.add_argument("--ddim-config", type=Path, default=DEFAULT_DDIM_CONFIG)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timestamp", default=None, help="Override timestamp for reproducible dry-runs and tests.")
    parser.add_argument("--queue-log", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--start-ddim-if-meanflow-done", dest="start_ddim_if_meanflow_done", action="store_true", default=True)
    parser.add_argument("--no-start-ddim-if-meanflow-done", dest="start_ddim_if_meanflow_done", action="store_false")
    parser.add_argument("--comparison-script", type=Path, default=DEFAULT_COMPARISON_SCRIPT)
    parser.add_argument("--report-summary-out", type=Path, default=DEFAULT_REPORT_SUMMARY)
    parser.add_argument("--report-device", default="cuda:0")
    return parser.parse_args(argv)


def build_queue_plan(args: argparse.Namespace) -> QueuePlan:
    timestamp = str(args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    ddim_session = args.ddim_session or f"safa_ddim_e10_gpu6_200ep_{timestamp}"
    ddim_log = Path(args.ddim_log) if args.ddim_log is not None else Path(f"artifacts/logs/e10_ddim_200ep_gpu6_{timestamp}.log")
    queue_log = Path(args.queue_log) if args.queue_log is not None else Path(f"artifacts/logs/gpu6_generator_queue_{timestamp}.log")

    return QueuePlan(
        repo_root=Path(args.repo_root),
        timestamp=timestamp,
        poll_seconds=int(args.poll_seconds),
        python=str(args.python),
        queue_log=queue_log,
        dry_run=bool(args.dry_run),
        skip_report=bool(args.skip_report),
        start_ddim_if_meanflow_done=bool(args.start_ddim_if_meanflow_done),
        meanflow=StagePlan(
            name="meanflow-e9",
            session=str(args.meanflow_session),
            log_path=Path(args.meanflow_log),
            checkpoints=DEFAULT_MEANFLOW_CHECKPOINTS,
        ),
        ddim=StagePlan(
            name="ddim-e10",
            session=ddim_session,
            log_path=ddim_log,
            checkpoints=DEFAULT_DDIM_CHECKPOINTS,
        ),
        ddim_config=Path(args.ddim_config),
        comparison_script=Path(args.comparison_script),
        report_summary=Path(args.report_summary_out),
        report_device=str(args.report_device),
    )


def build_tmux_start_command(stage: StagePlan, plan: QueuePlan) -> list[str]:
    train_command = shlex.join(
        [
            plan.python,
            "-m",
            "safa.cli.train_g",
            "--config",
            str(plan.ddim_config),
        ]
    )
    shell_command = " && ".join(
        [
            f"cd {shlex.quote(str(plan.repo_root))}",
            "export CUDA_VISIBLE_DEVICES=6",
            "export PYTHONPATH=src",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            f"{train_command} > {shlex.quote(str(stage.log_path))} 2>&1",
        ]
    )
    return ["tmux", "new-session", "-d", "-s", stage.session, shell_command]


def build_report_command(plan: QueuePlan) -> list[str]:
    return [
        plan.python,
        str(plan.comparison_script),
        "--runs",
        "e8",
        "e9",
        "e10",
        "--device",
        plan.report_device,
        "--python",
        plan.python,
    ]


def build_gpu6_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base or {})
    env["CUDA_VISIBLE_DEVICES"] = "6"
    env["PYTHONPATH"] = "src"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def render_dry_run(plan: QueuePlan) -> str:
    lines = ["DRY RUN: no commands executed, no waits, no artifacts written."]
    lines.append(f"repo_root: {plan.repo_root}")
    lines.append(f"poll_seconds: {plan.poll_seconds}")
    lines.append(f"queue_log: {plan.queue_log}")
    lines.append(f"meanflow_session: {plan.meanflow.session}")
    lines.append(f"meanflow_log: {plan.meanflow.log_path}")
    lines.append("meanflow_checkpoints:")
    lines.extend(f"  - {path}" for path in plan.meanflow.checkpoints)
    lines.append(f"start_ddim_if_meanflow_done: {plan.start_ddim_if_meanflow_done}")
    lines.append(f"ddim_session: {plan.ddim.session}")
    lines.append(f"ddim_log: {plan.ddim.log_path}")
    lines.append("ddim_checkpoints:")
    lines.extend(f"  - {path}" for path in plan.ddim.checkpoints)
    lines.append(f"error_keywords: {', '.join(ERROR_KEYWORDS)}")
    lines.append("commands:")
    lines.append(f"  [ddim:start] {shlex.join(build_tmux_start_command(plan.ddim, plan))}")
    if plan.skip_report:
        lines.append("  [report] skipped")
    else:
        lines.append(
            "  [report] CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src "
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {shlex.join(build_report_command(plan))}"
        )
        lines.append(f"report_summary: {plan.report_summary}")
    return "\n".join(lines) + "\n"


def first_existing_checkpoint(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def find_error_keywords(log_path: Path, keywords: Sequence[str] = ERROR_KEYWORDS) -> list[str]:
    if not log_path.is_file():
        return []
    found: set[str] = set()
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for keyword in keywords:
                if keyword in line:
                    found.add(keyword)
    return [keyword for keyword in keywords if keyword in found]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_queue_plan(args)
    if plan.dry_run:
        print(render_dry_run(plan), end="")
        return 0

    logger = QueueLogger(_resolve_path(plan.repo_root, plan.queue_log))
    logger.log(f"queue_start timestamp={plan.timestamp} repo_root={plan.repo_root}")
    logger.log(f"queue_log={plan.queue_log}")
    logger.log(f"meanflow_session={plan.meanflow.session} meanflow_log={plan.meanflow.log_path}")
    logger.log(f"ddim_session={plan.ddim.session} ddim_log={plan.ddim.log_path}")
    logger.log(f"poll_seconds={plan.poll_seconds}")

    if tmux_session_exists(plan.meanflow.session):
        logger.log(f"waiting_for_session stage={plan.meanflow.name} session={plan.meanflow.session}")
        wait_for_session_to_end(plan.meanflow.session, plan.poll_seconds, logger)
    else:
        logger.log(f"session_not_running stage={plan.meanflow.name} session={plan.meanflow.session}")

    meanflow_checkpoint = validate_stage_completed(plan.meanflow, plan, logger)
    if meanflow_checkpoint is None:
        logger.log("queue_failed reason=meanflow_validation_failed action=do_not_start_ddim")
        return 1

    if not plan.start_ddim_if_meanflow_done:
        logger.log("queue_stop reason=start_ddim_if_meanflow_done_false")
        return 0

    if not launch_ddim(plan, logger):
        logger.log("queue_failed reason=ddim_launch_failed")
        return 1

    wait_for_session_to_end(plan.ddim.session, plan.poll_seconds, logger)
    ddim_checkpoint = validate_stage_completed(plan.ddim, plan, logger)
    if ddim_checkpoint is None:
        logger.log("queue_failed reason=ddim_validation_failed action=skip_report")
        return 1

    if plan.skip_report:
        logger.log("report_skipped")
        logger.log("queue_complete")
        return 0

    if not run_report(plan, logger):
        logger.log("queue_failed reason=report_failed")
        return 1

    logger.log(f"report_summary={plan.report_summary}")
    logger.log("queue_complete")
    return 0


def tmux_session_exists(session: str) -> bool:
    result = subprocess.run(("tmux", "has-session", "-t", session), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    return result.returncode == 0


def tmux_pane_pid(session: str) -> str | None:
    result = subprocess.run(
        ("tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    pid = result.stdout.strip().splitlines()
    return pid[0] if pid else None


def wait_for_session_to_end(session: str, poll_seconds: int, logger: QueueLogger) -> None:
    while tmux_session_exists(session):
        pane_pid = tmux_pane_pid(session) or "unknown"
        logger.log(f"session_alive session={session} pane_pid={pane_pid}")
        time.sleep(poll_seconds)
    logger.log(f"session_finished session={session}")


def validate_stage_completed(stage: StagePlan, plan: QueuePlan, logger: QueueLogger) -> Path | None:
    log_path = _resolve_path(plan.repo_root, stage.log_path)
    checkpoint_paths = tuple(_resolve_path(plan.repo_root, path) for path in stage.checkpoints)

    if not log_path.is_file():
        logger.log(f"stage_failed stage={stage.name} reason=missing_log log={stage.log_path}")
        return None

    errors = find_error_keywords(log_path)
    if errors:
        logger.log(f"stage_failed stage={stage.name} reason=error_keywords keywords={','.join(errors)} log={stage.log_path}")
        return None

    checkpoint = first_existing_checkpoint(checkpoint_paths)
    if checkpoint is None:
        joined = ",".join(str(path) for path in stage.checkpoints)
        logger.log(f"stage_failed stage={stage.name} reason=missing_checkpoint checkpoints={joined}")
        return None

    logger.log(f"stage_ok stage={stage.name} checkpoint={_display_path(plan.repo_root, checkpoint)} log={stage.log_path}")
    return checkpoint


def launch_ddim(plan: QueuePlan, logger: QueueLogger) -> bool:
    _resolve_path(plan.repo_root, plan.ddim.log_path).parent.mkdir(parents=True, exist_ok=True)
    command = build_tmux_start_command(plan.ddim, plan)
    logger.log(f"launch_ddim command={shlex.join(command)}")
    result = subprocess.run(command, cwd=plan.repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    logger.append_output("tmux stdout", result.stdout)
    logger.append_output("tmux stderr", result.stderr)
    if result.returncode != 0:
        logger.log(f"launch_ddim_failed returncode={result.returncode}")
        return False

    pane_pid = tmux_pane_pid(plan.ddim.session) or "unknown"
    logger.log(f"launch_ddim_ok session={plan.ddim.session} pane_pid={pane_pid} log={plan.ddim.log_path}")
    return True


def run_report(plan: QueuePlan, logger: QueueLogger) -> bool:
    command = build_report_command(plan)
    env = build_gpu6_env(os.environ)
    logger.log(
        "run_report env=CUDA_VISIBLE_DEVICES=6,PYTHONPATH=src,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"command={shlex.join(command)}"
    )
    result = subprocess.run(command, cwd=plan.repo_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    logger.append_output("report output", result.stdout)
    if result.returncode != 0:
        logger.log(f"run_report_failed returncode={result.returncode}")
        return False
    logger.log(f"run_report_ok summary={plan.report_summary}")
    return True


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _display_path(repo_root: Path, path: Path) -> Path:
    try:
        return path.relative_to(repo_root)
    except ValueError:
        return path


if __name__ == "__main__":
    raise SystemExit(main())
