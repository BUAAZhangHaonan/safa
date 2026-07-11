#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import yaml


DEFAULT_PYTHON = "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"
DEFAULT_CONFIGS = (
    (0, Path("configs/medium_v2/experiments/r7_coupled_embedding_cap025_lr1e4_gpu0.yaml")),
    (1, Path("configs/medium_v2/experiments/r7_independent_prior_cap025_lr1e4_gpu1.yaml")),
    (2, Path("configs/medium_v2/experiments/r7_independent_prior_cap005_lr1e4_gpu2.yaml")),
    (3, Path("configs/medium_v2/experiments/r7_independent_prior_cap025_lr5e5_gpu3.yaml")),
)
EVAL_SCRIPT = Path("scripts/r5_eval.py")


@dataclass(frozen=True)
class RunContract:
    physical_gpu: int
    gpu_uuid: str
    config: Path
    experiment_name: str
    checkpoint: Path
    env: Mapping[str, str]
    train_command: tuple[str, ...]
    eval_command: tuple[str, ...]
    train_log: Path
    eval_log: Path
    eval_output_dir: Path


@dataclass(frozen=True)
class GpuState:
    index: int
    uuid: str
    free_memory_mib: int
    compute_processes: tuple[str, ...]


@dataclass(frozen=True)
class MatrixPlan:
    repo_root: Path
    python: str
    phase: str
    execute: bool
    preflight_only: bool
    log_dir: Path
    lock_path: Path
    runs: tuple[RunContract, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight, print, or explicitly launch the four-GPU R7 matrix.",
        epilog="Dry-run is the default. Commands run only when --execute is explicit.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--phase", choices=("train", "eval", "all"), default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def build_matrix_plan(args: argparse.Namespace) -> MatrixPlan:
    repo_root = Path(args.repo_root).resolve()
    python = str(args.python)
    phase = str(args.phase)
    log_dir = Path("artifacts/logs") / f"r7_matrix_{phase}"
    runs = []
    for physical_gpu, config_path in DEFAULT_CONFIGS:
        payload = _load_yaml(repo_root / config_path)
        experiment_name = str(payload.get("experiment_name", ""))
        checkpoint = Path(str(payload.get("out_dir", ""))) / "last.pt"
        env = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONPATH": "src",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "SAFA_REPO_ROOT": str(repo_root),
        }
        runs.append(
            RunContract(
                physical_gpu=physical_gpu,
                gpu_uuid="",
                config=config_path,
                experiment_name=experiment_name,
                checkpoint=checkpoint,
                env=env,
                train_command=(python, "-m", "safa.cli.train_g", "--config", str(config_path)),
                eval_command=(
                    python,
                    str(EVAL_SCRIPT),
                    str(checkpoint),
                    experiment_name,
                    "0",
                    "--full-ft",
                ),
                train_log=log_dir / f"{experiment_name}_train.log",
                eval_log=log_dir / f"{experiment_name}_eval.log",
                eval_output_dir=Path(f"artifacts/r5_eval_{experiment_name}"),
            )
        )
    return MatrixPlan(
        repo_root=repo_root,
        python=python,
        phase=phase,
        execute=bool(args.execute),
        preflight_only=bool(args.preflight_only),
        log_dir=log_dir,
        lock_path=Path("artifacts/r7_independent_prior_matrix.lock"),
        runs=tuple(runs),
    )


def validate_preflight(
    plan: MatrixPlan,
    *,
    gpu_states: Mapping[int, GpuState] | None = None,
) -> MatrixPlan:
    if not plan.repo_root.is_dir():
        raise FileNotFoundError(f"repo root not found: {plan.repo_root}")
    python_path = Path(plan.python)
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise FileNotFoundError(f"Python executable not found or not executable: {python_path}")
    if not (plan.repo_root / EVAL_SCRIPT).is_file():
        raise FileNotFoundError(f"R7 evaluation script not found: {plan.repo_root / EVAL_SCRIPT}")
    if [run.physical_gpu for run in plan.runs] != [0, 1, 2, 3]:
        raise ValueError("R7 matrix must pin exactly physical GPUs 0,1,2,3")
    if len({run.config for run in plan.runs}) != 4:
        raise ValueError("R7 matrix config paths must be unique")
    if len({run.checkpoint for run in plan.runs}) != 4:
        raise ValueError("R7 matrix checkpoint paths must be unique")

    for run in plan.runs:
        config_path = plan.repo_root / run.config
        if not config_path.is_file():
            raise FileNotFoundError(f"R7 config not found: {config_path}")
        payload = _load_yaml(config_path)
        if payload.get("experiment_name") != run.experiment_name:
            raise ValueError(f"experiment_name mismatch in {run.config}")
        if payload.get("device") != "cuda:0":
            raise ValueError(f"{run.config} must use logical device cuda:0")
        if payload.get("global_batch_size") != payload.get("per_device_batch_size"):
            raise ValueError(f"{run.config} must be an independent single-GPU run")
        if payload.get("out_dir") != str(run.checkpoint.parent):
            raise ValueError(f"out_dir mismatch in {run.config}")
        many_to_many = payload.get("many_to_many")
        if not isinstance(many_to_many, dict) or many_to_many.get("pairing_strategy") != "balanced_epoch_cycle":
            raise ValueError(f"{run.config} must use balanced_epoch_cycle pairing")
        for field in (
            "train_index",
            "train_features",
            "e0_checkpoint",
            "resume_from",
            "vae_path",
        ):
            value = payload.get(field)
            if not value or not (plan.repo_root / str(value)).exists():
                raise FileNotFoundError(f"{run.config} references missing {field}: {value!r}")
    validate_artifact_paths(plan)
    states = gpu_states if gpu_states is not None else query_gpu_states()
    _validate_gpu_states(states)
    return _bind_gpu_uuids(plan, states)


def validate_artifact_paths(plan: MatrixPlan) -> None:
    resolved_log_dir = _resolve(plan.repo_root, plan.log_dir)
    if resolved_log_dir.exists():
        raise FileExistsError(f"refusing to overwrite R7 log directory: {resolved_log_dir}")
    for run in plan.runs:
        checkpoint = _resolve(plan.repo_root, run.checkpoint)
        output_dir = checkpoint.parent
        eval_output_dir = _resolve(plan.repo_root, run.eval_output_dir)
        if plan.phase in {"train", "all"}:
            if output_dir.exists():
                raise FileExistsError(f"refusing to overwrite R7 output directory: {output_dir}")
            if eval_output_dir.exists():
                raise FileExistsError(f"refusing to reuse R7 eval directory: {eval_output_dir}")
        else:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"R7 checkpoint required for eval not found: {checkpoint}")
            if eval_output_dir.exists():
                raise FileExistsError(f"refusing to overwrite R7 eval directory: {eval_output_dir}")


def query_gpu_states() -> dict[int, GpuState]:
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    states = {}
    uuid_to_index = {}
    for line in gpu_result.stdout.splitlines():
        if not line.strip():
            continue
        index_text, uuid, memory_text = (part.strip() for part in line.split(",", maxsplit=2))
        index = int(index_text)
        uuid_to_index[uuid] = index
        states[index] = GpuState(
            index=index,
            uuid=uuid,
            free_memory_mib=int(memory_text),
            compute_processes=(),
        )

    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes: dict[int, list[str]] = {index: [] for index in states}
    for line in process_result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, process_name = (part.strip() for part in line.split(",", maxsplit=2))
        index = uuid_to_index.get(uuid)
        if index is not None:
            processes[index].append(f"pid={pid} {process_name}")
    return {
        index: GpuState(
            index=index,
            uuid=state.uuid,
            free_memory_mib=state.free_memory_mib,
            compute_processes=tuple(processes[index]),
        )
        for index, state in states.items()
    }


def _validate_gpu_states(states: Mapping[int, GpuState]) -> None:
    for index in range(4):
        state = states.get(index)
        if state is None:
            raise RuntimeError(f"nvidia-smi did not report required GPU {index}")
        if state.compute_processes:
            details = ", ".join(state.compute_processes)
            raise RuntimeError(f"GPU {index} has an external compute process: {details}")
        if state.free_memory_mib < 20000:
            raise RuntimeError(
                f"GPU {index} requires at least 20000 MiB free memory, got {state.free_memory_mib} MiB"
            )


def _bind_gpu_uuids(plan: MatrixPlan, states: Mapping[int, GpuState]) -> MatrixPlan:
    runs = []
    for run in plan.runs:
        state = states[run.physical_gpu]
        env = dict(run.env)
        env["CUDA_VISIBLE_DEVICES"] = state.uuid
        runs.append(replace(run, gpu_uuid=state.uuid, env=env))
    return replace(plan, runs=tuple(runs))


def build_process_command(run: RunContract, plan: MatrixPlan) -> tuple[str, ...]:
    commands = []
    if plan.phase in {"train", "all"}:
        commands.append(
            f"{shlex.join(run.train_command)} > {shlex.quote(str(run.train_log))} 2>&1"
        )
    if plan.phase in {"eval", "all"}:
        commands.append(
            f"{shlex.join(run.eval_command)} > {shlex.quote(str(run.eval_log))} 2>&1"
        )
    return ("/bin/bash", "-lc", " && ".join(commands))


def render_dry_run(plan: MatrixPlan) -> str:
    lines = ["DRY RUN: no commands executed and no files or directories written."]
    lines.append(f"repo_root: {plan.repo_root}")
    lines.append(f"phase: {plan.phase}")
    for run in plan.runs:
        env_text = " ".join(f"{key}={value}" for key, value in run.env.items())
        lines.append(f"[gpu{run.physical_gpu}] {env_text}")
        lines.append(f"  train: {shlex.join(run.train_command)}")
        lines.append(f"  eval: {shlex.join(run.eval_command)}")
        lines.append(f"  supervisor: {shlex.join(build_process_command(run, plan))}")
    return "\n".join(lines) + "\n"


def launch_matrix(plan: MatrixPlan) -> int:
    log_dir = _resolve(plan.repo_root, plan.log_dir)
    log_dir.mkdir(parents=True, exist_ok=False)
    processes = []
    try:
        for run in plan.runs:
            env = dict(os.environ)
            env.update(run.env)
            process = subprocess.Popen(build_process_command(run, plan), cwd=plan.repo_root, env=env)
            processes.append((run, process))
    except Exception as exc:
        failed_index = len(processes)
        statuses = []
        for run, process in processes:
            process.terminate()
            exit_code = int(process.wait())
            statuses.append(
                {
                    "experiment_name": run.experiment_name,
                    "physical_gpu": run.physical_gpu,
                    "gpu_uuid": run.gpu_uuid,
                    "exit_code": exit_code,
                    "status": "terminated_after_launch_error",
                }
            )
        failed_run = plan.runs[failed_index]
        statuses.append(
            {
                "experiment_name": failed_run.experiment_name,
                "physical_gpu": failed_run.physical_gpu,
                "gpu_uuid": failed_run.gpu_uuid,
                "exit_code": None,
                "status": "launch_failed",
            }
        )
        for run in plan.runs[failed_index + 1 :]:
            statuses.append(
                {
                    "experiment_name": run.experiment_name,
                    "physical_gpu": run.physical_gpu,
                    "gpu_uuid": run.gpu_uuid,
                    "exit_code": None,
                    "status": "not_started",
                }
            )
        _write_status(
            log_dir,
            plan,
            statuses,
            overall_status="failed",
            launch_error=f"{type(exc).__name__}: {exc}",
        )
        return 1

    statuses = []
    for run, process in processes:
        exit_code = int(process.wait())
        statuses.append(
            {
                "experiment_name": run.experiment_name,
                "physical_gpu": run.physical_gpu,
                "gpu_uuid": run.gpu_uuid,
                "exit_code": exit_code,
                "status": "passed" if exit_code == 0 else "failed",
            }
        )
    failed = any(status["exit_code"] != 0 for status in statuses)
    _write_status(log_dir, plan, statuses, overall_status="failed" if failed else "passed")
    return 1 if failed else 0


def _write_status(
    log_dir: Path,
    plan: MatrixPlan,
    statuses: list[dict],
    *,
    overall_status: str,
    launch_error: str | None = None,
) -> None:
    payload = {
        "phase": plan.phase,
        "overall_status": overall_status,
        "runs": statuses,
    }
    if launch_error is not None:
        payload["launch_error"] = launch_error
    (log_dir / "matrix_status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def execute_matrix(plan: MatrixPlan) -> int:
    lock_path = _resolve(plan.repo_root, plan.lock_path)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        bound_plan = validate_preflight(plan)
        if plan.preflight_only:
            print(f"PREFLIGHT OK: {len(bound_plan.runs)} independent runs pinned to GPUs 0-3")
            return 0
        return launch_matrix(bound_plan)
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_matrix_plan(args)
    if plan.execute:
        return execute_matrix(plan)
    plan = validate_preflight(plan)
    print(f"PREFLIGHT OK: {len(plan.runs)} independent runs pinned to GPUs 0-3")
    if plan.preflight_only:
        return 0
    print(render_dry_run(plan), end="")
    return 0


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"YAML config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
