#!/usr/bin/env python3
"""Static, resource, and completion checks for the 100-epoch GPU1/2 resume."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "safa_r14_resume_base_validator", SCRIPT_DIR / "validate_r14_inpaint_resume_gpu01.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the shared R14 validator")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

REPO_ROOT = BASE.REPO_ROOT
CONFIG = Path("configs/medium_v2/experiments/r14_inpaint_resume_gpu12_epoch100.yaml")
CHECKPOINT_ROOT = Path("checkpoints/r14_inpaint_resume_gpu12_epoch100")
ARTIFACT_ROOT = Path("artifacts/r14_inpaint_resume_gpu12_epoch100/v1")
LOG_PATH = ARTIFACT_ROOT / "logs/train.log"
SESSION = "safa-r14-inpaint-resume-gpu12-epoch100-v1"
RESUME_CONTRACT = {
    "contract_type": "safa_r14_long_resume_v1",
    "source_global_step": 2432,
    "source_completed_stage2_epochs": 19,
    "source_world_size": 4,
    "source_global_batch_size": 8,
    "source_per_device_batch_size": 2,
    "samples_per_epoch": 1024,
    "target_world_size": 2,
    "target_global_batch_size": 8,
    "target_per_device_batch_size": 4,
    "additional_optimizer_steps": 10368,
    "target_global_step": 12800,
    "target_completed_stage2_epochs": 100,
    "checkpoint_steps": [2560, 5120, 7680, 10240, 12800],
}
GPU_BINDINGS = {
    1: "GPU-dfaeaa7c-32c8-ebb4-aa59-ab7f829805f1",
    2: "GPU-e27fe71d-eaf7-3eb5-d0ff-c1c63b4f6b02",
}


def _configure_base() -> None:
    BASE.CONFIG = CONFIG
    BASE.CHECKPOINT_ROOT = CHECKPOINT_ROOT
    BASE.ARTIFACT_ROOT = ARTIFACT_ROOT
    BASE.LOG_PATH = LOG_PATH
    BASE.SESSION = SESSION
    BASE.RESUME_CONTRACT = RESUME_CONTRACT
    BASE.GPU_BINDINGS = GPU_BINDINGS
    BASE.PROJECTED_PEAK_MIB = 5744


_configure_base()
R14ResumeError = BASE.R14ResumeError


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    return BASE._mapping(value, name)


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    BASE._require_equal(actual, expected, name)


def _read_json(path: Path) -> Mapping[str, Any]:
    return BASE._read_json(path)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return BASE._read_jsonl(path)


def _validate_config(path: Path) -> Mapping[str, Any]:
    try:
        config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
    except (OSError, yaml.YAMLError) as exc:
        raise R14ResumeError(f"cannot read config {path}: {exc}") from exc
    required = {
        "experiment_name": "r14_inpaint_resume_gpu12_epoch100",
        "r14_contract": "safa_r14_face_region_inpaint_feasibility_v1",
        "seed": 1337,
        "sampling_seed": 1337,
        "global_batch_size": 8,
        "per_device_batch_size": 4,
        "amp": False,
        "out_dir": str(CHECKPOINT_ROOT),
        "resume_from": str(BASE.SOURCE_CHECKPOINT),
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
        {"contract_type": "safa_r14_exact_optimizer_steps_v1", "required_steps": 12800},
        "config.optimizer_step_contract",
    )
    _require_equal(
        config.get("optimizer_checkpoint_contract"),
        {
            "contract_type": "safa_r14_optimizer_checkpoint_steps_v1",
            "save_steps": [2560, 5120, 7680, 10240, 12800],
        },
        "config.optimizer_checkpoint_contract",
    )
    _require_equal(_mapping(config.get("distributed"), "config.distributed").get("backend"), "nccl", "config.distributed.backend")
    stages = _mapping(config.get("stages"), "config.stages")
    _require_equal(_mapping(stages.get("stage2"), "config.stages.stage2").get("epochs"), 100, "config.stages.stage2.epochs")
    spatial = _mapping(config.get("r14_spatial"), "config.r14_spatial")
    _require_equal(spatial.get("pair_manifest"), "artifacts/r14_inpaint_feasibility/v1/manifests/train_pairs.jsonl", "config.r14_spatial.pair_manifest")
    generator = _mapping(config.get("generator"), "config.generator")
    _require_equal(generator.get("model_type"), "meanflow_sit_inpaint", "config.generator.model_type")
    _require_equal(_mapping(generator.get("inpaint"), "config.generator.inpaint").get("conditioning"), "cached_source_z_only", "config.generator.inpaint.conditioning")
    from safa.training.g_loop import _validate_train_g_config

    _validate_train_g_config(dict(config))
    return config


def validate_static() -> None:
    BASE._validate_nccl_transport_environment()
    path = REPO_ROOT / CONFIG
    if not path.is_file():
        raise R14ResumeError(f"missing long-resume config: {CONFIG}")
    config = _validate_config(path)
    BASE._validate_source_files(load_checkpoint=True)
    pair_manifest = REPO_ROOT / str(_mapping(config.get("r14_spatial"), "config.r14_spatial")["pair_manifest"])
    BASE._require_equal(len(_read_jsonl(pair_manifest)), 1024, "training pair count")
    required_paths = (
        Path(str(config["train_index"])),
        Path(str(config["train_features"])),
        Path(str(config["e0_checkpoint"])),
        Path(str(config["vae_path"])),
    )
    missing = [str(path) for path in required_paths if not (REPO_ROOT / path).exists()]
    if missing:
        raise R14ResumeError(f"resume dependencies are missing: {missing}")


def _validate_final_metrics(metrics: Mapping[str, Any], label: str) -> None:
    expected = {
        "global_step": 12800,
        "required_optimizer_steps": 12800,
        "stage": "stage2",
        "stage_epoch_0based": 99,
        "stage_epoch_1based": 100,
        "world_size": 2,
        "global_batch_size": 8,
        "per_device_batch_size": 4,
        "optimizer_resumed": True,
    }
    for key, value in expected.items():
        _require_equal(metrics.get(key), value, f"{label}.{key}")


def validate_artifact() -> None:
    BASE._validate_source_files(load_checkpoint=False)
    root = REPO_ROOT / CHECKPOINT_ROOT
    for step in RESUME_CONTRACT["checkpoint_steps"]:
        path = root / f"step_{step:08d}.pt"
        if not path.is_file() or path.stat().st_size <= 0:
            raise R14ResumeError(f"missing checkpoint step artifact: {path}")
    last = root / "last.pt"
    for path in (last, root / "manifest.json"):
        if not path.is_file() or path.stat().st_size <= 0:
            raise R14ResumeError(f"long-resume output is missing or empty: {path}")
    completion = _read_json(root / "completion.json")
    for key, value in {
        "contract_type": "safa_r14_inpaint_exact_optimizer_steps_v1",
        "completed": True,
        "optimizer_steps": 12800,
        "ema_available": True,
        "checkpoint": str(CHECKPOINT_ROOT / "last.pt"),
        "manifest": str(CHECKPOINT_ROOT / "manifest.json"),
    }.items():
        _require_equal(completion.get(key), value, f"completion.{key}")
    _validate_final_metrics(_read_json(root / "last_metrics.json"), "last_metrics")
    rows = _read_jsonl(root / "metrics_history.jsonl")
    _require_equal(len(rows), 81, "new metrics_history row count")
    _validate_final_metrics(rows[-1], "metrics_history[-1]")
    manifest = _read_json(root / "manifest.json")
    _require_equal(manifest.get("checkpoint"), str(CHECKPOINT_ROOT / "last.pt"), "manifest.checkpoint")
    _validate_final_metrics(_mapping(manifest.get("metrics"), "manifest.metrics"), "manifest.metrics")
    _require_equal(len(manifest.get("history", [])), 100, "manifest history length")
    _require_equal(manifest.get("distributed"), {"enabled": True, "world_size": 2, "backend": "nccl"}, "manifest.distributed")
    payload = BASE._load_checkpoint(last)
    _validate_final_metrics(_mapping(payload.get("metrics"), "checkpoint.metrics"), "checkpoint.metrics")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != 100:
        raise R14ResumeError("checkpoint history must contain 100 completed epochs")
    _validate_final_metrics(_mapping(history[-1], "checkpoint.history[-1]"), "checkpoint.history[-1]")
    training = _mapping(payload.get("training_config"), "checkpoint.training_config")
    _require_equal(training.get("r14_contract"), "safa_r14_face_region_inpaint_feasibility_v1", "checkpoint.training_config.r14_contract")
    _require_equal(training.get("r14_resume_contract"), RESUME_CONTRACT, "checkpoint.training_config.r14_resume_contract")
    for key, value in {"world_size": 2, "global_batch_size": 8, "per_device_batch_size": 4}.items():
        _require_equal(training.get(key), value, f"checkpoint.training_config.{key}")
    BASE._validate_checkpoint_states(payload, "checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static", "resource", "artifact"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    if args.mode == "static":
        validate_static()
    elif args.mode == "resource":
        validate_static()
        resources = BASE._validate_resources()
        print(json.dumps({"mode": args.mode, "resources": resources, "status": "pass"}, sort_keys=True))
        return
    else:
        validate_artifact()
    print(json.dumps({"mode": args.mode, "status": "pass"}, sort_keys=True))


if __name__ == "__main__":
    main()
