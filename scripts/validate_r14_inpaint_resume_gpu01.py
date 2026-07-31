#!/usr/bin/env python3
"""Fail-closed preparation and completion checks for the R14 two-GPU resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/medium_v2/experiments/r14_inpaint_resume_gpu01_step2560.yaml")
SOURCE_ROOT = Path("checkpoints/r14_inpaint_feasibility_2560step")
SOURCE_CHECKPOINT = SOURCE_ROOT / "last.pt"
SOURCE_CHECKPOINT_SIZE = 2_103_293_800
SOURCE_CHECKPOINT_SHA256 = "a176d5521782a16ba488fe5d727cec61ddcf35d07fe75316f00f281ef423b7bf"
SOURCE_METRICS_SHA256 = "d9801ef8289f7036f0ef80a34113ee117fad8eca825c9797e07a2ce6e4c9d401"
SOURCE_HISTORY_SHA256 = "d823cb0918a44ddb59eedf7e50b6ef159238e3c56e917ad8aa2ef8c7771b577f"
CHECKPOINT_ROOT = Path("checkpoints/r14_inpaint_resume_gpu01_step2560")
ARTIFACT_ROOT = Path("artifacts/r14_inpaint_resume_gpu01/v1")
LOG_PATH = ARTIFACT_ROOT / "logs/train.log"
SMOKE_SUMMARY = Path("artifacts/r14_inpaint_resume_gpu01/batch4_smoke_v1/summary.json")
SESSION = "safa-r14-inpaint-resume-gpu01-v1"
SOURCE_WRITER_SESSION = "safa-r14-inpaint-v1"
GPU_BINDINGS = {
    0: "GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
    1: "GPU-dfaeaa7c-32c8-ebb4-aa59-ab7f829805f1",
}
NCCL_TRANSPORT_ENV = {
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_DISABLE": "0",
}
# Max reserved VRAM observed on both ranks in the full-state batch=4 smoke.
PROJECTED_PEAK_MIB = 5744
RESUME_CONTRACT = {
    "contract_type": "safa_r14_epoch_boundary_world_size_resume_v1",
    "source_global_step": 2432,
    "source_completed_stage2_epochs": 19,
    "source_world_size": 4,
    "source_global_batch_size": 8,
    "source_per_device_batch_size": 2,
    "samples_per_epoch": 1024,
    "target_world_size": 2,
    "target_global_batch_size": 8,
    "target_per_device_batch_size": 4,
    "additional_optimizer_steps": 128,
    "target_global_step": 2560,
}
ALLOWED_SOURCE_UNTRACKED = {
    str(SOURCE_ROOT / "last_metrics.json"),
    str(SOURCE_ROOT / "metrics_history.jsonl"),
}


class R14ResumeError(RuntimeError):
    pass


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R14ResumeError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise R14ResumeError(f"{name} must be a sequence")
    return value


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise R14ResumeError(f"{name} differs: expected {expected!r}, got {actual!r}")


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14ResumeError(f"cannot read JSON {path}: {exc}") from exc
    return _mapping(payload, str(path))


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise R14ResumeError(f"blank JSONL row: {path}:{line_number}")
                rows.append(_mapping(json.loads(line), f"{path}:{line_number}"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14ResumeError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _validate_batch4_smoke(path: Path | None = None) -> int:
    summary_path = REPO_ROOT / SMOKE_SUMMARY if path is None else path
    payload = _read_json(summary_path)
    expected = {
        "contract_type": "safa_r14_resume_batch4_memory_smoke_v1",
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_global_step": 2432,
        "smoke_optimizer_step": 2433,
        "formal_target_global_step": 2560,
        "world_size": 2,
        "per_device_batch_size": 4,
        "global_batch_size": 8,
        "sample_count": 8,
        "ddp_backward_and_adamw_step": True,
        "ema_loaded_and_updated": True,
        "source_face_pixels_enter_context_encoder": False,
        "source_z_finite_l2_normalized": True,
        "loss_finite": True,
        "gradients_finite": True,
        "vae_frozen": True,
        "max_peak_reserved_mib": float(PROJECTED_PEAK_MIB),
    }
    for key, value in expected.items():
        _require_equal(payload.get(key), value, f"batch4 smoke.{key}")
    rows = _sequence(payload.get("per_rank_memory"), "batch4 smoke.per_rank_memory")
    _require_equal(len(rows), 2, "batch4 smoke rank count")
    _require_equal(
        {int(_mapping(row, "batch4 smoke rank").get("rank")) for row in rows},
        {0, 1},
        "batch4 smoke ranks",
    )
    for row in rows:
        reserved = _mapping(row, "batch4 smoke rank").get("peak_reserved_mib")
        if not isinstance(reserved, (int, float)) or not math.isfinite(float(reserved)):
            raise R14ResumeError("batch4 smoke rank peak_reserved_mib is missing or non-finite")
        _require_equal(float(reserved), float(PROJECTED_PEAK_MIB), "batch4 smoke rank peak_reserved_mib")
    return PROJECTED_PEAK_MIB


def _run(argv: Sequence[str]) -> str:
    result = subprocess.run(
        list(argv), cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise R14ResumeError(f"command failed ({' '.join(argv)}): {detail}")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise R14ResumeError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _validate_nccl_transport_environment(
    environ: Mapping[str, str] | None = None,
) -> None:
    values = os.environ if environ is None else environ
    for name, expected in NCCL_TRANSPORT_ENV.items():
        _require_equal(values.get(name), expected, f"environment.{name}")


def _validate_config(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise R14ResumeError(f"cannot read config {path}: {exc}") from exc
    config = _mapping(payload, str(path))
    required = {
        "experiment_name": "r14_inpaint_resume_gpu01_step2560",
        "r14_contract": "safa_r14_face_region_inpaint_feasibility_v1",
        "seed": 1337,
        "sampling_seed": 1337,
        "global_batch_size": 8,
        "per_device_batch_size": 4,
        "amp": False,
        "out_dir": str(CHECKPOINT_ROOT),
        "resume_from": str(SOURCE_CHECKPOINT),
        "resume_mode": "training_state",
        "resume_checkpoint_model": "raw",
        "resume_optimizer_state": True,
        "generator_trainable": "full",
    }
    for key, expected in required.items():
        _require_equal(config.get(key), expected, f"config.{key}")
    if "resume_from_sha256" in config:
        raise R14ResumeError("config must not retain the legacy E15 resume hash field")
    _require_equal(config.get("r14_resume_contract"), RESUME_CONTRACT, "config.r14_resume_contract")
    _require_equal(
        config.get("optimizer_step_contract"),
        {"contract_type": "safa_r14_exact_optimizer_steps_v1", "required_steps": 2560},
        "config.optimizer_step_contract",
    )
    _require_equal(
        config.get("optimizer_checkpoint_contract"),
        {"contract_type": "safa_r14_optimizer_checkpoint_steps_v1", "save_steps": [2560]},
        "config.optimizer_checkpoint_contract",
    )
    _require_equal(
        _mapping(config.get("distributed"), "config.distributed").get("backend"),
        "nccl",
        "config.distributed.backend",
    )
    _require_equal(
        _mapping(config.get("stages"), "config.stages")["stage2"]["epochs"],
        20,
        "config.stages.stage2.epochs",
    )
    _require_equal(
        _mapping(config.get("r14_spatial"), "config.r14_spatial").get("pair_manifest"),
        "artifacts/r14_inpaint_feasibility/v1/manifests/train_pairs.jsonl",
        "config.r14_spatial.pair_manifest",
    )
    generator = _mapping(config.get("generator"), "config.generator")
    _require_equal(generator.get("model_type"), "meanflow_sit_inpaint", "config.generator.model_type")
    _require_equal(
        _mapping(generator.get("inpaint"), "config.generator.inpaint").get("conditioning"),
        "cached_source_z_only",
        "config.generator.inpaint.conditioning",
    )
    from safa.training.g_loop import _validate_train_g_config

    _validate_train_g_config(dict(config))
    return config


def _validate_source_files(*, load_checkpoint: bool) -> None:
    checkpoint = REPO_ROOT / SOURCE_CHECKPOINT
    if not checkpoint.is_file():
        raise R14ResumeError(f"source checkpoint is missing: {SOURCE_CHECKPOINT}")
    _require_equal(checkpoint.stat().st_size, SOURCE_CHECKPOINT_SIZE, "source checkpoint size")
    _require_equal(_sha256(checkpoint), SOURCE_CHECKPOINT_SHA256, "source checkpoint SHA256")
    metrics_path = REPO_ROOT / SOURCE_ROOT / "last_metrics.json"
    history_path = REPO_ROOT / SOURCE_ROOT / "metrics_history.jsonl"
    _require_equal(_sha256(metrics_path), SOURCE_METRICS_SHA256, "source last_metrics SHA256")
    _require_equal(_sha256(history_path), SOURCE_HISTORY_SHA256, "source metrics_history SHA256")
    external = _read_json(metrics_path)
    expected = {
        "global_step": 2432,
        "required_optimizer_steps": 2560,
        "stage": "stage2",
        "stage_epoch_0based": 18,
        "stage_epoch_1based": 19,
        "world_size": 4,
        "global_batch_size": 8,
        "per_device_batch_size": 2,
        "optimizer_resumed": False,
    }
    for key, value in expected.items():
        _require_equal(external.get(key), value, f"source.last_metrics.{key}")
    external_history = _read_jsonl(history_path)
    _require_equal(len(external_history), 19, "source metrics_history row count")
    _require_equal(external_history[-1].get("global_step"), 2432, "source metrics_history final step")
    if load_checkpoint:
        _validate_source_checkpoint_payload(checkpoint)


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise R14ResumeError(f"cannot load checkpoint {path}: {exc}") from exc
    return _mapping(payload, str(path))


def _validate_source_checkpoint_payload(path: Path) -> None:
    payload = _load_checkpoint(path)
    metrics = _mapping(payload.get("metrics"), "source checkpoint.metrics")
    expected_metrics = {
        "global_step": 2432,
        "required_optimizer_steps": 2560,
        "stage": "stage2",
        "stage_epoch_0based": 18,
        "stage_epoch_1based": 19,
        "world_size": 4,
        "global_batch_size": 8,
        "per_device_batch_size": 2,
    }
    for key, value in expected_metrics.items():
        _require_equal(metrics.get(key), value, f"source checkpoint.metrics.{key}")
    history = _sequence(payload.get("history"), "source checkpoint.history")
    _require_equal(len(history), 19, "source checkpoint history length")
    _require_equal(
        _mapping(history[-1], "source checkpoint.history[-1]").get("global_step"),
        2432,
        "source checkpoint final history step",
    )
    training = _mapping(payload.get("training_config"), "source checkpoint.training_config")
    for key, value in {
        "world_size": 4,
        "global_batch_size": 8,
        "per_device_batch_size": 2,
    }.items():
        _require_equal(training.get(key), value, f"source checkpoint.training_config.{key}")
    _require_equal(
        _mapping(training.get("stages"), "source checkpoint.training_config.stages")["stage2"]["epochs"],
        20,
        "source checkpoint.training_config.stages.stage2.epochs",
    )
    raw_state = _mapping(payload.get("model_state_dict"), "source checkpoint.model_state_dict")
    ema_state = _mapping(payload.get("ema_model_state_dict"), "source checkpoint.ema_model_state_dict")
    if not raw_state or not ema_state:
        raise R14ResumeError("source checkpoint must contain non-empty raw and EMA model states")
    if not any("context_embedder" in str(key) for key in raw_state):
        raise R14ResumeError("source raw model state lacks the R14 context embedder")
    optimizer = _mapping(payload.get("optimizer_state_dict"), "source checkpoint.optimizer_state_dict")
    state = _mapping(optimizer.get("state"), "source checkpoint.optimizer_state_dict.state")
    groups = _sequence(optimizer.get("param_groups"), "source checkpoint.optimizer_state_dict.param_groups")
    if not state:
        raise R14ResumeError("source optimizer state is empty")
    _require_equal(len(groups), 1, "source optimizer param-group count")


def validate_static() -> None:
    _validate_nccl_transport_environment()
    config_path = REPO_ROOT / CONFIG
    if not config_path.is_file():
        raise R14ResumeError(f"missing resume config: {CONFIG}")
    config = _validate_config(config_path)
    _validate_source_files(load_checkpoint=True)
    pair_manifest = REPO_ROOT / str(config["r14_spatial"]["pair_manifest"])
    rows = _read_jsonl(pair_manifest)
    _require_equal(len(rows), 1024, "training pair count")
    required_paths = (
        Path(str(config["train_index"])),
        Path(str(config["train_features"])),
        Path(str(config["e0_checkpoint"])),
        Path(str(config["vae_path"])),
    )
    missing = [str(path) for path in required_paths if not (REPO_ROOT / path).exists()]
    if missing:
        raise R14ResumeError(f"resume dependencies are missing: {missing}")
    for executable in ("tmux", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise R14ResumeError(f"required executable is unavailable: {executable}")


def _cpu_percent(sample_seconds: float = 0.2) -> float:
    def snapshot() -> tuple[int, int]:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    total0, idle0 = snapshot()
    time.sleep(sample_seconds)
    total1, idle1 = snapshot()
    delta = total1 - total0
    if delta <= 0:
        raise R14ResumeError("cannot measure CPU utilization")
    return 100.0 * (1.0 - (idle1 - idle0) / delta)


def _memory_and_swap_percent() -> tuple[float, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0])
    memory_total = values.get("MemTotal", 0)
    memory_available = values.get("MemAvailable", 0)
    if memory_total <= 0 or not 0 <= memory_available <= memory_total:
        raise R14ResumeError("cannot measure main-memory utilization")
    swap_total = values.get("SwapTotal", 0)
    swap_free = values.get("SwapFree", 0)
    if swap_total < 0 or not 0 <= swap_free <= swap_total:
        raise R14ResumeError("cannot measure swap utilization")
    memory = 100.0 * (memory_total - memory_available) / memory_total
    swap = 0.0 if swap_total == 0 else 100.0 * (swap_total - swap_free) / swap_total
    return memory, swap


def _validate_worktree() -> None:
    _require_equal(_run(["git", "branch", "--show-current"]), "master", "git branch")
    _require_equal(_run(["git", "rev-parse", "HEAD"]), _run(["git", "rev-parse", "origin/master"]), "HEAD/origin")
    status = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    observed_untracked: set[str] = set()
    for line in status.splitlines():
        if not line:
            continue
        if not line.startswith("?? "):
            raise R14ResumeError(f"tracked worktree change blocks launch: {line}")
        observed_untracked.add(line[3:])
    unexpected = observed_untracked - ALLOWED_SOURCE_UNTRACKED
    if unexpected:
        raise R14ResumeError(f"unexpected untracked paths block launch: {sorted(unexpected)}")
    missing = ALLOWED_SOURCE_UNTRACKED - observed_untracked
    if missing:
        raise R14ResumeError(f"source checkpoint files are not all preserved as untracked inputs: {sorted(missing)}")


def _validate_process_isolation(process_table: str, sessions: Sequence[str]) -> None:
    active_sessions = set(sessions)
    if SESSION in active_sessions:
        raise R14ResumeError(f"conflicting tmux session already exists: {SESSION}")
    if SOURCE_WRITER_SESSION in active_sessions:
        raise R14ResumeError(
            f"the original R14 source-checkpoint writer tmux is still running: {SOURCE_WRITER_SESSION}"
        )
    lines = process_table.splitlines()
    resume_config = str(CONFIG)
    source_config = "configs/medium_v2/experiments/r14_inpaint_feasibility_2560step.yaml"
    if any("safa.cli.train_g" in line and resume_config in line for line in lines):
        raise R14ResumeError("the locked two-GPU resume command is already running")
    if any(
        ("safa.cli.train_g" in line and source_config in line)
        or "scripts/run_r14_inpaint_feasibility.sh" in line
        for line in lines
    ):
        raise R14ResumeError("the original R14 source-checkpoint writer is still running")


def _validate_resources() -> Mapping[str, Any]:
    measured_peak_mib = _validate_batch4_smoke()
    cpu = _cpu_percent()
    memory, swap = _memory_and_swap_percent()
    for label, value in (("CPU", cpu), ("main-memory", memory)):
        if not math.isfinite(value) or value >= 90.0:
            raise R14ResumeError(f"{label} utilization must be below 90%, got {value:.2f}%")
    if not math.isfinite(swap):
        raise R14ResumeError("swap utilization could not be measured")
    free_disk = shutil.disk_usage(REPO_ROOT).free
    if free_disk < 24 * 1024**3:
        raise R14ResumeError(
            f"repository filesystem must have at least 24 GiB free, got {free_disk / 1024**3:.2f} GiB"
        )
    query = _run([
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    observed: dict[int, tuple[str, int, int, int, int]] = {}
    for line in query.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise R14ResumeError(f"unexpected nvidia-smi row: {line!r}")
        observed[int(parts[0])] = (parts[1], *(int(value) for value in parts[2:]))
    for index, expected_uuid in GPU_BINDINGS.items():
        if index not in observed:
            raise R14ResumeError(f"GPU{index} is unavailable")
        uuid, total_mib, used_mib, free_mib, utilization = observed[index]
        _require_equal(uuid, expected_uuid, f"GPU{index} UUID")
        projected_used_mib = used_mib + measured_peak_mib
        projected_free_mib = free_mib - measured_peak_mib
        if total_mib <= 0 or projected_used_mib / total_mib >= 0.85:
            raise R14ResumeError(
                f"GPU{index} projected memory occupancy with the measured "
                f"{measured_peak_mib} MiB R14 peak must remain below 85%"
            )
        if projected_free_mib < 4096:
            raise R14ResumeError(
                f"GPU{index} must retain at least 4 GiB after the measured "
                f"{measured_peak_mib} MiB R14 peak"
            )
        if utilization >= 90:
            raise R14ResumeError(f"GPU{index} utilization must be below 90%, got {utilization}%")
    _validate_worktree()
    tmux_result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        text=True,
        capture_output=True,
        check=False,
    )
    sessions = tmux_result.stdout.splitlines() if tmux_result.returncode == 0 else []
    process_table = _run(["ps", "-eo", "args="])
    _validate_process_isolation(process_table, sessions)
    generated = (CHECKPOINT_ROOT, LOG_PATH)
    present = [str(path) for path in generated if (REPO_ROOT / path).exists()]
    if present:
        raise R14ResumeError(f"refusing to reuse resume outputs: {present}")
    return {
        "cpu_percent": round(cpu, 2),
        "main_memory_percent": round(memory, 2),
        "swap_percent": round(swap, 2),
        "disk_free_gib": round(free_disk / 1024**3, 2),
        "projected_peak_mib": measured_peak_mib,
        "gpus": {
            str(index): {
                "memory_used_mib": observed[index][2],
                "memory_free_mib": observed[index][3],
                "utilization_percent": observed[index][4],
            }
            for index in GPU_BINDINGS
        },
    }


def _validate_final_metrics(metrics: Mapping[str, Any], label: str) -> None:
    expected = {
        "global_step": 2560,
        "required_optimizer_steps": 2560,
        "stage": "stage2",
        "stage_epoch_0based": 19,
        "stage_epoch_1based": 20,
        "world_size": 2,
        "global_batch_size": 8,
        "per_device_batch_size": 4,
        "optimizer_resumed": True,
    }
    for key, value in expected.items():
        _require_equal(metrics.get(key), value, f"{label}.{key}")


def _validate_checkpoint_states(payload: Mapping[str, Any], label: str) -> None:
    raw_state = _mapping(payload.get("model_state_dict"), f"{label}.model_state_dict")
    ema_state = _mapping(payload.get("ema_model_state_dict"), f"{label}.ema_model_state_dict")
    optimizer = _mapping(payload.get("optimizer_state_dict"), f"{label}.optimizer_state_dict")
    optimizer_state = _mapping(optimizer.get("state"), f"{label}.optimizer_state_dict.state")
    if not raw_state or not ema_state or not optimizer_state:
        raise R14ResumeError(f"{label} must contain raw, EMA, and optimizer states")


def validate_artifact() -> None:
    _validate_source_files(load_checkpoint=False)
    root = REPO_ROOT / CHECKPOINT_ROOT
    last = root / "last.pt"
    step = root / "step_00002560.pt"
    manifest_path = root / "manifest.json"
    for path in (last, step, manifest_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise R14ResumeError(f"resume training output is missing or empty: {path}")
    completion = _read_json(root / "completion.json")
    expected_completion = {
        "contract_type": "safa_r14_inpaint_exact_optimizer_steps_v1",
        "completed": True,
        "optimizer_steps": 2560,
        "ema_available": True,
        "checkpoint": str(CHECKPOINT_ROOT / "last.pt"),
        "manifest": str(CHECKPOINT_ROOT / "manifest.json"),
    }
    for key, value in expected_completion.items():
        _require_equal(completion.get(key), value, f"completion.{key}")
    external = _read_json(root / "last_metrics.json")
    _validate_final_metrics(external, "last_metrics")
    history_rows = _read_jsonl(root / "metrics_history.jsonl")
    _require_equal(len(history_rows), 1, "new metrics_history row count")
    _validate_final_metrics(history_rows[0], "metrics_history[0]")
    manifest = _read_json(manifest_path)
    _require_equal(manifest.get("checkpoint"), str(CHECKPOINT_ROOT / "last.pt"), "manifest.checkpoint")
    _validate_final_metrics(_mapping(manifest.get("metrics"), "manifest.metrics"), "manifest.metrics")
    manifest_history = _sequence(manifest.get("history"), "manifest.history")
    _require_equal(len(manifest_history), 20, "manifest history length")
    _validate_final_metrics(
        _mapping(manifest_history[-1], "manifest.history[-1]"),
        "manifest.history[-1]",
    )
    _require_equal(
        manifest.get("distributed"),
        {"enabled": True, "world_size": 2, "backend": "nccl"},
        "manifest.distributed",
    )
    payload = _load_checkpoint(last)
    _validate_final_metrics(_mapping(payload.get("metrics"), "checkpoint.metrics"), "checkpoint.metrics")
    history = _sequence(payload.get("history"), "checkpoint.history")
    _require_equal(len(history), 20, "checkpoint history length")
    _require_equal(
        _mapping(history[0], "checkpoint.history[0]").get("global_step"),
        128,
        "checkpoint first history step",
    )
    _validate_final_metrics(_mapping(history[-1], "checkpoint.history[-1]"), "checkpoint.history[-1]")
    training = _mapping(payload.get("training_config"), "checkpoint.training_config")
    _require_equal(training.get("r14_contract"), "safa_r14_face_region_inpaint_feasibility_v1", "checkpoint.training_config.r14_contract")
    _require_equal(training.get("r14_resume_contract"), RESUME_CONTRACT, "checkpoint.training_config.r14_resume_contract")
    for key, value in {
        "world_size": 2,
        "global_batch_size": 8,
        "per_device_batch_size": 4,
    }.items():
        _require_equal(training.get(key), value, f"checkpoint.training_config.{key}")
    _validate_checkpoint_states(payload, "checkpoint")

    step_payload = _load_checkpoint(step)
    step_metrics = _mapping(step_payload.get("metrics"), "step checkpoint.metrics")
    expected_step_metrics = {
        "global_step": 2560,
        "required_optimizer_steps": 2560,
        "stage": "stage2",
        "stage_epoch_0based": 19,
        "stage_epoch_1based": 20,
        "world_size": 2,
        "global_batch_size": 8,
        "per_device_batch_size": 4,
        "checkpoint_kind": "optimizer_step",
    }
    for key, value in expected_step_metrics.items():
        _require_equal(step_metrics.get(key), value, f"step checkpoint.metrics.{key}")
    step_history = _sequence(step_payload.get("history"), "step checkpoint.history")
    _require_equal(len(step_history), 19, "step checkpoint history length")
    _require_equal(
        _mapping(step_history[-1], "step checkpoint.history[-1]").get("global_step"),
        2432,
        "step checkpoint source history final step",
    )
    step_training = _mapping(
        step_payload.get("training_config"), "step checkpoint.training_config"
    )
    _require_equal(
        step_training.get("r14_resume_contract"),
        RESUME_CONTRACT,
        "step checkpoint.training_config.r14_resume_contract",
    )
    for key, value in {
        "world_size": 2,
        "global_batch_size": 8,
        "per_device_batch_size": 4,
    }.items():
        _require_equal(
            step_training.get(key), value, f"step checkpoint.training_config.{key}"
        )
    _validate_checkpoint_states(step_payload, "step checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static", "resource", "artifact"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    resources: Mapping[str, Any] | None = None
    if args.mode == "static":
        validate_static()
    elif args.mode == "resource":
        validate_static()
        resources = _validate_resources()
    else:
        validate_artifact()
    result: dict[str, Any] = {"mode": args.mode, "status": "pass"}
    if resources is not None:
        result["resources"] = resources
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
