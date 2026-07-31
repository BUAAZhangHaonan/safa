#!/usr/bin/env python3
"""Fail-closed validation for the bounded R14 inpainting feasibility run."""
from __future__ import annotations

import argparse
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
CONFIG = Path("configs/medium_v2/experiments/r14_inpaint_feasibility_256step.yaml")
ARTIFACT_ROOT = Path("artifacts/r14_inpaint_feasibility/v1")
CHECKPOINT_ROOT = Path("checkpoints/r14_inpaint_feasibility_256step")
SESSION = "safa-r14-inpaint-v1"
E15_PATH = Path(
    "artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/"
    "last_nopretrained.pt"
)
E15_SHA256 = "4690717781db58a6021d57d124300a9b212f0a5043cf3028fb5de4d9c835cc4d"
GPU_BINDINGS = {
    1: "GPU-dfaeaa7c-32c8-ebb4-aa59-ab7f829805f1",
    2: "GPU-e27fe71d-eaf7-3eb5-d0ff-c1c63b4f6b02",
}
MANIFEST_COUNTS = {
    "manifests/smoke8.jsonl": 8,
    "manifests/regular32.jsonl": 32,
    "manifests/visual8.jsonl": 8,
}
ENTRYPOINTS = (
    "scripts/prepare_r14_inpaint_manifests.py",
    "scripts/run_r14_inpaint_smoke.py",
    "scripts/export_r14_inpaint_ema.py",
    "scripts/run_r14_inpaint_generation.py",
    "scripts/evaluate_r14_inpaint_feasibility.py",
    "scripts/render_r14_inpaint_visual8.py",
    "scripts/close_r14_inpaint_feasibility.py",
)


class R14LaunchError(RuntimeError):
    pass


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R14LaunchError(f"{name} must be a mapping")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14LaunchError(f"cannot read JSON {path}: {exc}") from exc
    return _mapping(value, str(path))


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise R14LaunchError(f"blank JSONL row: {path}:{line_number}")
                rows.append(_mapping(json.loads(line), f"{path}:{line_number}"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14LaunchError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _run(argv: Sequence[str]) -> str:
    result = subprocess.run(argv, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise R14LaunchError(f"command failed ({' '.join(argv)}): {detail}")
    return result.stdout.strip()


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise R14LaunchError(f"{name} differs: expected {expected!r}, got {actual!r}")


def _validate_config(path: Path) -> None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise R14LaunchError(f"cannot read config {path}: {exc}") from exc
    config = _mapping(payload, str(path))
    required = {
        "r14_contract": "safa_r14_face_region_inpaint_feasibility_v1",
        "seed": 1337,
        "sampling_seed": 1337,
        "global_batch_size": 4,
        "per_device_batch_size": 2,
        "amp": True,
        "resume_from": str(E15_PATH),
        "resume_from_sha256": E15_SHA256,
        "resume_mode": "model_weights_only",
        "resume_checkpoint_model": "ema",
        "out_dir": str(CHECKPOINT_ROOT),
    }
    for key, expected in required.items():
        _require_equal(config.get(key), expected, f"config.{key}")
    _require_equal(
        config.get("optimizer_step_contract"),
        {"contract_type": "safa_r14_exact_optimizer_steps_v1", "required_steps": 256},
        "config.optimizer_step_contract",
    )
    _require_equal(_mapping(config.get("distributed"), "config.distributed").get("backend"), "nccl", "config.distributed.backend")
    generator = _mapping(config.get("generator"), "config.generator")
    _require_equal(generator.get("model_type"), "meanflow_sit_inpaint", "config.generator.model_type")
    inpaint = _mapping(generator.get("inpaint"), "config.generator.inpaint")
    locked_inpaint = {
        "mask_source": "affectnet_original_bbox",
        "bbox_expansion_pixels": 0,
        "morphology": None,
        "source_face_removed_before_context_encoder": True,
        "flow_and_loss_region": "inside_latent_mask_only",
        "outside_latent_projection": "context_after_every_flow_step",
        "outside_pixel_contract": "bit_exact_context_pixels",
    }
    for key, expected in locked_inpaint.items():
        _require_equal(inpaint.get(key), expected, f"config.generator.inpaint.{key}")
    pairing = _mapping(config.get("r14_pairing"), "config.r14_pairing")
    _require_equal(pairing.get("expression_relation"), "same", "config.r14_pairing.expression_relation")
    _require_equal(pairing.get("identity_relation"), "different_sample_id", "config.r14_pairing.identity_relation")
    _require_equal(pairing.get("require_source_target_distinct"), True, "config.r14_pairing.require_source_target_distinct")
    data = _mapping(config.get("r14_spatial"), "config.r14_spatial")
    locked_data = {
        "contract_type": "safa_r14_spatial_training_v1",
        "pair_manifest": str(ARTIFACT_ROOT / "manifests/train_pairs.jsonl"),
        "horizontal_flip_probability": 0.0,
    }
    for key, expected in locked_data.items():
        _require_equal(data.get(key), expected, f"config.r14_spatial.{key}")
    _require_equal(config.get("train_index"), "data/index/train_face_mixed_e14_4029avail.jsonl", "config.train_index")
    _require_equal(config.get("train_features"), "artifacts/e0_features/train_face_mixed_e14_e0_medium_v1", "config.train_features")
    _require_equal(config.get("eval_index"), "data/index/val_face_mixed_e14.jsonl", "config.eval_index")
    _require_equal(config.get("eval_features"), "artifacts/e0_features/val_face_mixed_e14_e0_medium_v1", "config.eval_features")
    _require_equal(config.get("e0_checkpoint"), "artifacts/checkpoints/e0_medium_v1/best.pt", "config.e0_checkpoint")
    forbidden_sequence_fields = ("learning_rates", "loss_weights", "mask_candidates", "training_steps_grid")
    for field in forbidden_sequence_fields:
        if field in config:
            raise R14LaunchError(f"config contains forbidden search field: {field}")
    from safa.training.g_loop import _validate_train_g_config

    _validate_train_g_config(dict(config))


def _validate_bbox_and_landmarks(row: Mapping[str, Any], bbox_name: str, landmarks_name: str, label: str) -> None:
    bbox = row.get(bbox_name)
    if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox):
        raise R14LaunchError(f"{label} lacks an integer four-value {bbox_name}")
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise R14LaunchError(f"{label} has a non-positive {bbox_name}")
    landmarks = row.get(landmarks_name)
    if not isinstance(landmarks, list) or len(landmarks) != 68:
        raise R14LaunchError(f"{label} must contain 68 points in {landmarks_name}")
    for point in landmarks:
        if not isinstance(point, list) or len(point) != 2:
            raise R14LaunchError(f"{label} has invalid {landmarks_name}")
        for value in point:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise R14LaunchError(f"{label} has non-finite {landmarks_name}")


def _validate_eval_manifest(path: Path, count: int) -> None:
    rows = _read_jsonl(path)
    if len(rows) != count:
        raise R14LaunchError(f"{path} must contain exactly {count} rows, got {len(rows)}")
    sample_ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in sample_ids):
        raise R14LaunchError(f"{path} contains an invalid sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise R14LaunchError(f"{path} contains duplicate sample_id values")
    expected_fields = {
        "contract_version",
        "sample_id",
        "image_path",
        "affect_label",
        "bbox_xywh",
        "landmarks68",
    }
    for index, row in enumerate(rows):
        if set(row) != expected_fields:
            raise R14LaunchError(f"{path}:{index + 1} fields differ from safa_r14_spatial_eval_v1")
        _require_equal(row.get("contract_version"), "safa_r14_spatial_eval_v1", f"{path}:{index + 1}.contract_version")
        if type(row.get("affect_label")) is not int or row["affect_label"] not in range(8):
            raise R14LaunchError(f"{path}:{index + 1} has an invalid affect_label")
        image_path = row.get("image_path")
        if not isinstance(image_path, str) or not Path(image_path).is_absolute():
            raise R14LaunchError(f"{path}:{index + 1} image_path must be absolute")
        _validate_bbox_and_landmarks(row, "bbox_xywh", "landmarks68", f"{path}:{index + 1}")


def _validate_train_manifest(path: Path) -> None:
    rows = _read_jsonl(path)
    if len(rows) != 1024:
        raise R14LaunchError(f"train_pairs.jsonl must contain exactly 1024 pairs, got {len(rows)}")
    pair_ids = [row.get("pair_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in pair_ids) or len(set(pair_ids)) != len(pair_ids):
        raise R14LaunchError("train_pairs.jsonl has invalid or duplicate pair_id values")
    expected_fields = {
        "contract_version", "pair_id", "source_sample_id", "target_sample_id", "affect_label",
        "source_image_path", "target_image_path", "source_bbox_xywh", "target_bbox_xywh",
        "source_landmarks68", "target_landmarks68",
    }
    for index, row in enumerate(rows):
        if set(row) != expected_fields:
            raise R14LaunchError(f"{path}:{index + 1} fields differ from safa_r14_spatial_pair_v1")
        _require_equal(row.get("contract_version"), "safa_r14_spatial_pair_v1", f"{path}:{index + 1}.contract_version")
        if type(row.get("affect_label")) is not int or row["affect_label"] not in range(8):
            raise R14LaunchError(f"{path}:{index + 1} has an invalid affect_label")
        source_id = row.get("source_sample_id")
        target_id = row.get("target_sample_id")
        if not isinstance(source_id, str) or not isinstance(target_id, str) or not source_id or not target_id or source_id == target_id:
            raise R14LaunchError(f"{path}:{index + 1} violates source_sample_id != target_sample_id")
        _validate_bbox_and_landmarks(row, "source_bbox_xywh", "source_landmarks68", f"{path}:{index + 1}")
        _validate_bbox_and_landmarks(row, "target_bbox_xywh", "target_landmarks68", f"{path}:{index + 1}")


def _validate_cache_membership(
    index_path: Path,
    feature_dir: Path,
    required_sample_ids: Sequence[str],
    expected_index_path: str,
    label: str,
) -> None:
    index_rows = _read_jsonl(index_path)
    index_ids = [row.get("sample_id") for row in index_rows]
    if any(not isinstance(value, str) or not value for value in index_ids):
        raise R14LaunchError(f"{label} index contains an invalid sample_id")
    if len(set(index_ids)) != len(index_ids):
        raise R14LaunchError(f"{label} index contains duplicate sample_id values")
    cache_manifest = _read_json(feature_dir / "manifest.json")
    _require_equal(cache_manifest.get("index_path"), expected_index_path, f"{label} cache index_path")
    cache_ids = cache_manifest.get("sample_ids")
    if not isinstance(cache_ids, list) or cache_ids != index_ids:
        raise R14LaunchError(f"{label} cache sample_ids must exactly match its bound index order")
    if not (feature_dir / "features.pt").is_file():
        raise R14LaunchError(f"{label} cache lacks features.pt")
    available = set(index_ids)
    missing = sorted(set(required_sample_ids) - available)
    if missing:
        preview = ", ".join(missing[:4])
        raise R14LaunchError(f"{label} cache/index lacks {len(missing)} required sample IDs: {preview}")


def validate_static() -> None:
    config_path = REPO_ROOT / CONFIG
    if not config_path.is_file():
        raise R14LaunchError(f"missing config: {CONFIG}")
    _validate_config(config_path)
    if not (REPO_ROOT / E15_PATH).is_file():
        raise R14LaunchError(f"missing E15 source checkpoint: {E15_PATH}")
    for entrypoint in ENTRYPOINTS:
        if not (REPO_ROOT / entrypoint).is_file():
            raise R14LaunchError(f"missing pipeline entrypoint: {entrypoint}")
    for relative, count in MANIFEST_COUNTS.items():
        _validate_eval_manifest(REPO_ROOT / ARTIFACT_ROOT / relative, count)
    regular_ids = [row["sample_id"] for row in _read_jsonl(REPO_ROOT / ARTIFACT_ROOT / "manifests/regular32.jsonl")]
    smoke_ids = [row["sample_id"] for row in _read_jsonl(REPO_ROOT / ARTIFACT_ROOT / "manifests/smoke8.jsonl")]
    visual_ids = [row["sample_id"] for row in _read_jsonl(REPO_ROOT / ARTIFACT_ROOT / "manifests/visual8.jsonl")]
    if smoke_ids != visual_ids or any(sample_id not in regular_ids for sample_id in smoke_ids):
        raise R14LaunchError("smoke8/visual8 must be the same ordered subset of regular32")
    train_pairs = REPO_ROOT / ARTIFACT_ROOT / "manifests/train_pairs.jsonl"
    _validate_train_manifest(train_pairs)
    eval_ids = regular_ids
    _validate_cache_membership(
        REPO_ROOT / "data/index/val_face_mixed_e14.jsonl",
        REPO_ROOT / "artifacts/e0_features/val_face_mixed_e14_e0_medium_v1",
        eval_ids,
        "data/index/val_face_mixed_e14.jsonl",
        "evaluation",
    )
    train_source_ids = [
        row["source_sample_id"] for row in _read_jsonl(train_pairs)
    ]
    _validate_cache_membership(
        REPO_ROOT / "data/index/train_face_mixed_e14_4029avail.jsonl",
        REPO_ROOT / "artifacts/e0_features/train_face_mixed_e14_e0_medium_v1",
        train_source_ids,
        "data/index/train_face_mixed_e14_4029avail.jsonl",
        "training",
    )
    for executable in ("tmux", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise R14LaunchError(f"required executable is unavailable: {executable}")


def _cpu_percent(sample_seconds: float = 0.2) -> float:
    def snapshot() -> tuple[int, int]:
        fields = (Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]).split()[1:]
        values = [int(value) for value in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    total0, idle0 = snapshot()
    time.sleep(sample_seconds)
    total1, idle1 = snapshot()
    delta = total1 - total0
    if delta <= 0:
        raise R14LaunchError("cannot measure CPU utilization")
    return 100.0 * (1.0 - (idle1 - idle0) / delta)


def _memory_percent() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0])
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    if total <= 0 or not 0 <= available <= total:
        raise R14LaunchError("cannot measure main-memory utilization")
    return 100.0 * (total - available) / total


def _validate_resources() -> None:
    cpu = _cpu_percent()
    memory = _memory_percent()
    if not math.isfinite(cpu) or cpu >= 90.0:
        raise R14LaunchError(f"CPU utilization must be below 90%, got {cpu:.2f}%")
    if not math.isfinite(memory) or memory >= 90.0:
        raise R14LaunchError(f"main-memory utilization must be below 90%, got {memory:.2f}%")
    free_disk = shutil.disk_usage(REPO_ROOT).free
    if free_disk < 24 * 1024**3:
        raise R14LaunchError(f"repository filesystem must have at least 24 GiB free, got {free_disk / 1024**3:.2f} GiB")
    query = _run([
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    observed: dict[int, tuple[str, int, int, int, int]] = {}
    for line in query.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise R14LaunchError(f"unexpected nvidia-smi row: {line!r}")
        index = int(parts[0])
        observed[index] = (parts[1], *(int(value) for value in parts[2:]))
    for index, expected_uuid in GPU_BINDINGS.items():
        if index not in observed:
            raise R14LaunchError(f"GPU{index} is unavailable")
        uuid, total_mib, used_mib, free_mib, utilization = observed[index]
        _require_equal(uuid, expected_uuid, f"GPU{index} UUID")
        if total_mib <= 0 or used_mib / total_mib >= 0.90:
            raise R14LaunchError(f"GPU{index} memory occupancy must be below 90%")
        if free_mib < 4096:
            raise R14LaunchError(f"GPU{index} must retain at least 4 GiB free before launch")
        if utilization >= 90:
            raise R14LaunchError(f"GPU{index} utilization must be below 90%, got {utilization}%")
    if _run(["git", "branch", "--show-current"]) != "master":
        raise R14LaunchError("R14 feasibility must launch from master")
    if _run(["git", "status", "--porcelain"]):
        raise R14LaunchError("R14 feasibility requires a clean worktree")
    sessions = _run(["tmux", "list-sessions", "-F", "#{session_name}"]) if subprocess.run(["tmux", "has-session"], capture_output=True).returncode == 0 else ""
    if SESSION in sessions.splitlines():
        raise R14LaunchError(f"tmux session already exists: {SESSION}")
    generated = (
        ARTIFACT_ROOT / "smoke8",
        ARTIFACT_ROOT / "regular32_generation",
        ARTIFACT_ROOT / "regular32_evaluation",
        ARTIFACT_ROOT / "visual8",
        ARTIFACT_ROOT / "summary.json",
        ARTIFACT_ROOT / "conclusion.md",
        ARTIFACT_ROOT / "logs/pipeline.log",
        CHECKPOINT_ROOT,
    )
    present = [str(path) for path in generated if (REPO_ROOT / path).exists()]
    if present:
        raise R14LaunchError(f"refusing to reuse R14 outputs: {present}")


def _require_path(payload: Mapping[str, Any], keys: Sequence[str], expected: Any, name: str) -> None:
    value: Any = payload
    for key in keys:
        value = _mapping(value, name).get(key)
    _require_equal(value, expected, name)


def validate_artifact(stage: str) -> None:
    root = REPO_ROOT / ARTIFACT_ROOT
    if stage == "smoke":
        payload = _read_json(root / "smoke8/summary.json")
        checks = {
            "sample_count": 8,
            "source_face_pixels_enter_context_encoder": False,
            "outside_mask_bit_exact": True,
            "same_seed_noise_deterministic": True,
            "masked_loss_finite": True,
            "masked_gradients_finite": True,
            "vae_frozen": True,
        }
        for key, expected in checks.items():
            _require_equal(payload.get(key), expected, f"smoke.{key}")
    elif stage == "train":
        checkpoint = REPO_ROOT / CHECKPOINT_ROOT / "last.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise R14LaunchError("training did not produce a non-empty last.pt")
        payload = _read_json(REPO_ROOT / CHECKPOINT_ROOT / "completion.json")
        _require_equal(payload.get("contract_type"), "safa_r14_inpaint_exact_optimizer_steps_v1", "train.contract_type")
        _require_equal(payload.get("optimizer_steps"), 256, "train.optimizer_steps")
        _require_equal(payload.get("completed"), True, "train.completed")
        _require_equal(payload.get("ema_available"), True, "train.ema_available")
    elif stage == "export":
        export = REPO_ROOT / CHECKPOINT_ROOT / "final_ema.pt"
        if not export.is_file() or export.stat().st_size <= 0:
            raise R14LaunchError("EMA export is missing or empty")
        payload = _read_json(REPO_ROOT / CHECKPOINT_ROOT / "final_ema.json")
        _require_equal(payload.get("checkpoint_model"), "ema", "export.checkpoint_model")
        _require_equal(payload.get("optimizer_steps"), 256, "export.optimizer_steps")
    elif stage == "generation":
        payload = _read_json(root / "regular32_generation/completion.json")
        _require_equal(payload.get("sample_count"), 32, "generation.sample_count")
        _require_equal(payload.get("outside_mask_bit_exact_all"), True, "generation.outside_mask_bit_exact_all")
        _require_equal(payload.get("candidate_count"), 32, "generation.candidate_count")
    elif stage == "evaluation":
        payload = _read_json(root / "regular32_evaluation/summary.json")
        _require_equal(payload.get("sample_count"), 32, "evaluation.sample_count")
        for role in ("source", "native", "candidate"):
            _require_path(payload, ("arcface", role, "exact_one"), 32, f"evaluation.arcface.{role}.exact_one")
        for key in ("e0", "delta_e0", "delta_edev", "arcface_u95", "full_niqe", "full_sharpness", "roi_niqe", "roi_sharpness"):
            value = payload.get("metrics", {}).get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise R14LaunchError(f"evaluation metric {key} is missing or non-finite")
        for key in ("exact_one", "representation", "privacy", "full_quality", "roi_quality"):
            if not isinstance(payload.get("gate", {}).get(key), bool):
                raise R14LaunchError(f"evaluation gate {key} must be explicit boolean")
    elif stage == "visual":
        payload = _read_json(root / "visual8/summary.json")
        _require_equal(payload.get("sample_count"), 8, "visual.sample_count")
        contact_sheet = root / "visual8/contact_sheet.png"
        if not contact_sheet.is_file() or contact_sheet.stat().st_size <= 0:
            raise R14LaunchError("visual8 contact sheet is missing or empty")
    elif stage == "conclusion":
        payload = _read_json(root / "summary.json")
        allowed = {
            "no_go_exact_one",
            "no_go_representation",
            "no_go_privacy",
            "no_go_full_quality",
            "no_go_copied_background_metric_inflation",
            "no_go_visual_severe",
            "numeric_pass_visual_review_pending",
            "feasibility_pass_stage128_allowed",
        }
        if payload.get("classification") not in allowed:
            raise R14LaunchError("R14 conclusion classification is missing or unlocked")
        if not (root / "conclusion.md").is_file():
            raise R14LaunchError("R14 conclusion.md is missing")
    else:
        raise R14LaunchError(f"unknown artifact stage: {stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static", "resource", "artifact"), required=True)
    parser.add_argument("--stage", choices=("smoke", "train", "export", "generation", "evaluation", "visual", "conclusion"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    if args.mode == "static":
        validate_static()
    elif args.mode == "resource":
        validate_static()
        _validate_resources()
    else:
        if args.stage is None:
            raise R14LaunchError("--stage is required with --mode artifact")
        validate_artifact(args.stage)
    print(json.dumps({"status": "pass", "mode": args.mode, "stage": args.stage}, sort_keys=True))


if __name__ == "__main__":
    main()
