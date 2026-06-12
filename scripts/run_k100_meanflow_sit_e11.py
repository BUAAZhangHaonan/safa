#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import yaml


DEFAULT_CONFIG = Path("configs/medium_v2/experiments/e11_meanflow_sit_b_stage1_200ep.yaml")
DEFAULT_PYTHON = "/home/k100/miniconda3/envs/pt210_cu130_fa4/bin/python"
DEFAULT_CU13_LIBRARY_PATH = "/home/k100/miniconda3/envs/pt210_cu130_fa4/lib/python3.12/site-packages/nvidia/cu13/lib"
DEFAULT_SESSION_PREFIX = "safa_e11_meanflow_sit_k100_200ep"
DEFAULT_PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"


@dataclass(frozen=True)
class RunPlan:
    repo_root: Path
    config: Path
    python: str
    timestamp: str
    session: str
    log_path: Path
    dry_run: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the e11 MeanFlow-SiT 200 epoch run on K100 in tmux.",
        epilog="Use --dry-run to print the tmux command without starting training.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_run_plan(args: argparse.Namespace) -> RunPlan:
    timestamp = str(args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    session = str(args.session or f"{DEFAULT_SESSION_PREFIX}_{timestamp}")
    log_path = Path(args.log) if args.log is not None else Path(f"artifacts/logs/e11_meanflow_sit_k100_200ep_{timestamp}.log")
    return RunPlan(
        repo_root=Path(args.repo_root),
        config=Path(args.config),
        python=str(args.python),
        timestamp=timestamp,
        session=session,
        log_path=log_path,
        dry_run=bool(args.dry_run),
    )


def build_k100_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base or {})
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = "src"
    env["PYTORCH_CUDA_ALLOC_CONF"] = DEFAULT_PYTORCH_CUDA_ALLOC_CONF
    env["LD_LIBRARY_PATH"] = DEFAULT_CU13_LIBRARY_PATH + (f":{existing_ld}" if existing_ld else "")
    return env


def build_tmux_start_command(plan: RunPlan) -> list[str]:
    train_command = shlex.join(
        [
            plan.python,
            "-m",
            "safa.cli.train_g",
            "--config",
            str(plan.config),
        ]
    )
    shell_command = " && ".join(
        [
            f"cd {shlex.quote(str(plan.repo_root))}",
            f"mkdir -p {shlex.quote(str(plan.log_path.parent))}",
            "export CUDA_VISIBLE_DEVICES=0",
            "export PYTHONPATH=src",
            f"export PYTORCH_CUDA_ALLOC_CONF={shlex.quote(DEFAULT_PYTORCH_CUDA_ALLOC_CONF)}",
            f"export LD_LIBRARY_PATH={shlex.quote(DEFAULT_CU13_LIBRARY_PATH)}:${{LD_LIBRARY_PATH:-}}",
            f"{train_command} > {shlex.quote(str(plan.log_path))} 2>&1",
        ]
    )
    return ["tmux", "new-session", "-d", "-s", plan.session, shell_command]


def render_dry_run(plan: RunPlan) -> str:
    lines = ["DRY RUN: no training started, no tmux session created."]
    lines.append(f"repo_root: {plan.repo_root}")
    lines.append(f"config: {plan.config}")
    lines.append(f"session: {plan.session}")
    lines.append(f"log_path: {plan.log_path}")
    lines.append(f"CUDA_VISIBLE_DEVICES: 0")
    lines.append(f"LD_LIBRARY_PATH_PREFIX: {DEFAULT_CU13_LIBRARY_PATH}")
    lines.append(f"command: {shlex.join(build_tmux_start_command(plan))}")
    return "\n".join(lines) + "\n"


def validate_prerequisites(plan: RunPlan) -> None:
    config_path = _resolve_path(plan.repo_root, plan.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"e11 config not found: {config_path}")
    config = _load_yaml(config_path)
    generator_config = config.get("generator")
    if not isinstance(generator_config, dict):
        raise ValueError(f"e11 config missing generator mapping: {config_path}")
    checkpoint_path = generator_config.get("sit_pretrained_path")
    if not checkpoint_path:
        raise ValueError("e11 config generator.sit_pretrained_path is required")
    resolved_checkpoint = _resolve_path(plan.repo_root, Path(str(checkpoint_path)))
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"MeanFlow-SiT pretrained checkpoint not found: {resolved_checkpoint}")
    vae_path = config.get("vae_path")
    if not vae_path:
        raise ValueError("e11 config vae_path is required for K100 long run")
    resolved_vae = _resolve_path(plan.repo_root, Path(str(vae_path)))
    if not resolved_vae.exists():
        raise FileNotFoundError(f"VAE path not found: {resolved_vae}")
    resolved_log = _resolve_path(plan.repo_root, plan.log_path)
    if resolved_log.exists():
        raise FileExistsError(f"refusing to overwrite existing log: {resolved_log}")
    if tmux_session_exists(plan.session):
        raise RuntimeError(f"tmux session already exists: {plan.session}")


def tmux_session_exists(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_run_plan(args)
    if plan.dry_run:
        print(render_dry_run(plan), end="")
        return 0
    validate_prerequisites(plan)
    log_path = _resolve_path(plan.repo_root, plan.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_tmux_start_command(plan)
    subprocess.run(command, cwd=plan.repo_root, env=build_k100_env(os.environ), check=True)
    print(f"started session={plan.session} log={plan.log_path}")
    return 0


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


if __name__ == "__main__":
    sys.exit(main())
