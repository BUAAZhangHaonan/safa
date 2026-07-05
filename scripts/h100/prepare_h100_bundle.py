#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import yaml

DEFAULT_TEMPLATE_CONFIG = Path("configs/medium_v2/experiments/e15_meanflow_sit_b_face_mixed_resume_h100_2400ep.yaml")
DEFAULT_RUNTIME_CONFIG = Path("configs/medium_v2/experiments/e15_meanflow_sit_b_face_mixed_resume_h100_runtime.yaml")
TRAIN_INDEX_BUNDLE = Path("data/index_bundle/train_face_mixed_e15_h100.jsonl")
VAL_INDEX_BUNDLE = Path("data/index_bundle/val_face_mixed_e15_h100.jsonl")
TRAIN_INDEX_RUNTIME = Path("data/index_runtime/train_face_mixed_e15_h100.jsonl")
VAL_INDEX_RUNTIME = Path("data/index_runtime/val_face_mixed_e15_h100.jsonl")
TRAIN_CACHE_BUNDLE = Path("artifacts/e0_features/train_face_mixed_e15_h100_e0_medium_v1")
VAL_CACHE_BUNDLE = Path("artifacts/e0_features/val_face_mixed_e15_h100_e0_medium_v1")
TRAIN_CACHE_RUNTIME = Path("artifacts/e0_features/train_face_mixed_e15_h100_e0_medium_v1_runtime")
VAL_CACHE_RUNTIME = Path("artifacts/e0_features/val_face_mixed_e15_h100_e0_medium_v1_runtime")
E15_H100_CKPT_DIR = Path("artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep")
E0_CKPT = Path("artifacts/checkpoints/e0_medium_v1/best.pt")
VAE_PATH = Path("artifacts/checkpoints/external/sd-vae-ft-ema")
MEANFLOW_PRETRAINED = Path("artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_b_4_imagenet256.pt")
QUALITY_DIR = Path("artifacts/eval/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/quality")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_batch_for_gpu_count(gpu_count: int) -> tuple[int, int]:
    if gpu_count <= 1:
        return 384, 384
    if gpu_count == 2:
        return 384, 768
    if gpu_count >= 4:
        return 256, 1024
    return 256, 256 * gpu_count


def detect_gpu_count(cuda_visible_devices: str | None = None) -> int:
    visible = cuda_visible_devices if cuda_visible_devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        tokens = [item.strip() for item in visible.split(",") if item.strip()]
        if tokens:
            return len(tokens)
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return 1
    count = sum(1 for line in output.splitlines() if line.strip().startswith("GPU "))
    return max(1, count)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    if not rows:
        raise ValueError(f"Index contains no rows: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve_bundle_path(bundle_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return bundle_root / path


def _abspath(bundle_root: Path, path: str | Path) -> str:
    return str(_resolve_bundle_path(bundle_root, path).resolve())


def _rewrite_index_absolute(bundle_root: Path, source: Path, target: Path) -> None:
    rows = _read_jsonl(source)
    rewritten: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        image_path = _resolve_bundle_path(bundle_root, str(new_row["image_path"])).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Bundle image missing for {new_row['sample_id']}: {image_path}")
        dataset_root = _resolve_bundle_path(bundle_root, str(new_row["dataset_root"])).resolve()
        new_row["image_path"] = str(image_path)
        new_row["dataset_root"] = str(dataset_root)
        rewritten.append(new_row)
    _write_jsonl(target, rewritten)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _prepare_feature_cache(bundle_root: Path, source_cache: Path, runtime_cache: Path, runtime_index: Path, e0_checkpoint: Path) -> None:
    runtime_cache.mkdir(parents=True, exist_ok=True)
    source_features = source_cache / "features.pt"
    source_manifest = source_cache / "manifest.json"
    if not source_features.is_file():
        raise FileNotFoundError(f"Feature shard missing: {source_features}")
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Feature manifest missing: {source_manifest}")
    target_features = runtime_cache / "features.pt"
    _link_or_copy(source_features, target_features)

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["index_path"] = str(runtime_index.resolve())
    manifest["index_sha256"] = sha256_file(runtime_index)
    manifest["encoder_checkpoint"] = str(e0_checkpoint.resolve())
    manifest["encoder_checkpoint_sha256"] = sha256_file(e0_checkpoint)
    manifest["shard"] = "features.pt"
    manifest["shard_sha256"] = sha256_file(target_features)
    (runtime_cache / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _first_visible_cuda_device(cuda_visible_devices: str | None) -> str:
    if not cuda_visible_devices:
        return "0"
    tokens = [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
    return tokens[0] if tokens else "0"


def prepare_bundle(
    bundle_root: str | Path,
    *,
    config_path: str | Path | None = None,
    runtime_config_path: str | Path | None = None,
    gpu_count: int | None = None,
    per_device_batch: int | None = None,
    global_batch: int | None = None,
    cuda_visible_devices: str | None = None,
) -> Path:
    bundle = Path(bundle_root).resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(f"Bundle root does not exist: {bundle}")

    resolved_gpu_count = gpu_count or detect_gpu_count(cuda_visible_devices)
    default_per, default_global = default_batch_for_gpu_count(resolved_gpu_count)
    per_device = int(per_device_batch or os.environ.get("PER_DEVICE_BATCH") or default_per)
    global_size = int(global_batch or os.environ.get("GLOBAL_BATCH") or default_global)
    if global_size != per_device * resolved_gpu_count:
        raise ValueError(
            "global_batch_size must equal per_device_batch_size * gpu_count: "
            f"{global_size} != {per_device} * {resolved_gpu_count}"
        )

    template = _resolve_bundle_path(bundle, config_path or DEFAULT_TEMPLATE_CONFIG)
    runtime_config = _resolve_bundle_path(bundle, runtime_config_path or DEFAULT_RUNTIME_CONFIG)
    if not template.is_file():
        raise FileNotFoundError(f"Template config does not exist: {template}")

    runtime_train_index = _resolve_bundle_path(bundle, TRAIN_INDEX_RUNTIME)
    runtime_val_index = _resolve_bundle_path(bundle, VAL_INDEX_RUNTIME)
    _rewrite_index_absolute(bundle, _resolve_bundle_path(bundle, TRAIN_INDEX_BUNDLE), runtime_train_index)
    _rewrite_index_absolute(bundle, _resolve_bundle_path(bundle, VAL_INDEX_BUNDLE), runtime_val_index)

    e0_checkpoint = _resolve_bundle_path(bundle, E0_CKPT)
    _prepare_feature_cache(
        bundle,
        _resolve_bundle_path(bundle, TRAIN_CACHE_BUNDLE),
        _resolve_bundle_path(bundle, TRAIN_CACHE_RUNTIME),
        runtime_train_index,
        e0_checkpoint,
    )
    _prepare_feature_cache(
        bundle,
        _resolve_bundle_path(bundle, VAL_CACHE_BUNDLE),
        _resolve_bundle_path(bundle, VAL_CACHE_RUNTIME),
        runtime_val_index,
        e0_checkpoint,
    )

    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Template config must contain a mapping: {template}")

    config["experiment_name"] = "e15_meanflow_sit_b_face_mixed_h100_resume_2400ep"
    config["device"] = "cuda:0"
    config["global_batch_size"] = global_size
    config["per_device_batch_size"] = per_device
    config["train_index"] = str(runtime_train_index.resolve())
    config["train_features"] = _abspath(bundle, TRAIN_CACHE_RUNTIME)
    config["e0_checkpoint"] = str(e0_checkpoint.resolve())
    config["out_dir"] = _abspath(bundle, E15_H100_CKPT_DIR)
    config["resume_from"] = _abspath(bundle, E15_H100_CKPT_DIR / "last.pt")
    config["resume_mode"] = "training_state"
    config["resume_optimizer_state"] = True
    config["vae_path"] = _abspath(bundle, VAE_PATH)
    config.setdefault("distributed", {})["backend"] = "nccl"
    config.setdefault("generator", {})["attention_backend"] = "auto"
    config["generator"]["sit_pretrained_path"] = _abspath(bundle, MEANFLOW_PRETRAINED)

    stage2 = config.setdefault("stages", {}).setdefault("stage2", {})
    stage2["epochs"] = 2400
    quality = stage2.setdefault("quality_eval", {})
    quality["real_index"] = str(runtime_val_index.resolve())
    quality["output_dir"] = _abspath(bundle, QUALITY_DIR)
    quality["distribution_cuda_visible_devices"] = _first_visible_cuda_device(cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES"))
    quality["distribution_device"] = os.environ.get("MEANFLOW_DISTRIBUTION_DEVICE", "cuda:0")

    validation = config.setdefault("validation", {})
    validation["index"] = str(runtime_val_index.resolve())
    validation["features"] = _abspath(bundle, VAL_CACHE_RUNTIME)

    resume_path = str(config["resume_from"])
    if "/e14_" in resume_path or "checkpoints/e14" in resume_path:
        raise ValueError(f"Refusing to write config that resumes from E14 checkpoint: {resume_path}")
    if not Path(resume_path).is_file():
        raise FileNotFoundError(f"E15 H100 snapshot checkpoint does not exist: {resume_path}")

    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return runtime_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an unpacked MeanFlow E15 H100 bundle for local training.")
    parser.add_argument("--bundle-root", default=".", help="Unpacked bundle root. Defaults to current directory.")
    parser.add_argument("--config", default=str(DEFAULT_TEMPLATE_CONFIG), help="Template config path, relative to bundle root unless absolute.")
    parser.add_argument("--runtime-config", default=str(DEFAULT_RUNTIME_CONFIG), help="Output runtime config path.")
    parser.add_argument("--gpu-count", type=int, default=None, help="Override detected GPU count.")
    parser.add_argument("--per-device-batch", type=int, default=None, help="Override per-device batch size.")
    parser.add_argument("--global-batch", type=int, default=None, help="Override global batch size.")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"), help="Visible CUDA device list.")
    args = parser.parse_args()

    runtime = prepare_bundle(
        args.bundle_root,
        config_path=args.config,
        runtime_config_path=args.runtime_config,
        gpu_count=args.gpu_count,
        per_device_batch=args.per_device_batch,
        global_batch=args.global_batch,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    print(runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
