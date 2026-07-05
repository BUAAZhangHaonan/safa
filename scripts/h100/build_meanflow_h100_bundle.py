#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from PIL import Image
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_TRAIN_INDEX = Path("data/index/train_face_mixed_e14.jsonl")
SOURCE_VAL_INDEX = Path("data/index/val_face_mixed_e14.jsonl")
SOURCE_TRAIN_CACHE = Path("artifacts/e0_features/train_face_mixed_e14_e0_medium_v1")
SOURCE_VAL_CACHE = Path("artifacts/e0_features/val_face_mixed_e14_e0_medium_v1")
SOURCE_E15_CKPT_DIR = Path("artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_resume_e14_2400ep")
SOURCE_E15_CONFIG = Path("configs/medium_v2/experiments/e15_meanflow_sit_b_face_mixed_resume_e14_2400ep.yaml")
BUNDLE_CKPT_DIR = Path("artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep")
BUNDLE_TRAIN_CACHE = Path("artifacts/e0_features/train_face_mixed_e15_h100_e0_medium_v1")
BUNDLE_VAL_CACHE = Path("artifacts/e0_features/val_face_mixed_e15_h100_e0_medium_v1")
BUNDLE_TRAIN_INDEX = Path("data/index_bundle/train_face_mixed_e15_h100.jsonl")
BUNDLE_VAL_INDEX = Path("data/index_bundle/val_face_mixed_e15_h100.jsonl")
H100_TEMPLATE_CONFIG = Path("configs/medium_v2/experiments/e15_meanflow_sit_b_face_mixed_resume_h100_2400ep.yaml")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _bundle_image_relpath(row: dict[str, Any]) -> Path:
    sample_id = str(row["sample_id"])
    dataset_version = str(row.get("dataset_version", ""))
    if dataset_version == "affectnet" or sample_id.startswith(("train:", "val:")):
        if ":" in sample_id:
            rel = Path(sample_id.split(":", 1)[1])
        else:
            rel = Path(row["image_path"]).name
        return Path("data/face_256_q95/affectnet") / str(row["split"]) / rel.with_suffix(".jpg")
    if dataset_version.startswith("celebahq") or sample_id.startswith("celebahq_"):
        return Path("data/face_256_q95/celeba_hq") / f"{sample_id}.jpg"
    if dataset_version.startswith("ffhq") or sample_id.startswith("ffhq_"):
        return Path("data/face_256_q95/ffhq") / f"{sample_id}.jpg"
    raise ValueError(f"Unsupported dataset row: sample_id={sample_id} dataset_version={dataset_version}")


def _dataset_root_for(row: dict[str, Any]) -> str:
    sample_id = str(row["sample_id"])
    dataset_version = str(row.get("dataset_version", ""))
    if dataset_version == "affectnet" or sample_id.startswith(("train:", "val:")):
        return "data/face_256_q95/affectnet"
    if dataset_version.startswith("celebahq") or sample_id.startswith("celebahq_"):
        return "data/face_256_q95/celeba_hq"
    if dataset_version.startswith("ffhq") or sample_id.startswith("ffhq_"):
        return "data/face_256_q95/ffhq"
    raise ValueError(f"Unsupported dataset row: sample_id={sample_id} dataset_version={dataset_version}")


def _dataset_version_for(row: dict[str, Any]) -> str:
    sample_id = str(row["sample_id"])
    dataset_version = str(row.get("dataset_version", ""))
    if dataset_version == "affectnet" or sample_id.startswith(("train:", "val:")):
        return "affectnet-256-q95"
    if dataset_version.startswith("celebahq") or sample_id.startswith("celebahq_"):
        return "celebahq-256-q95"
    if dataset_version.startswith("ffhq") or sample_id.startswith("ffhq_"):
        return "ffhq-256-q95"
    raise ValueError(f"Unsupported dataset row: sample_id={sample_id} dataset_version={dataset_version}")


def _resize_one(task: tuple[str, str]) -> tuple[str, bool, str | None]:
    source, target = task
    target_path = Path(target)
    if target_path.is_file() and target_path.stat().st_size > 0:
        return target, False, None
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image = image.convert("RGB")
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
            image.save(tmp_path, format="JPEG", quality=95, subsampling=0, optimize=False)
        tmp_path.replace(target_path)
        return target, True, None
    except Exception as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return target, False, f"{source}: {exc}"


def _make_bundle_rows(rows: list[dict[str, Any]], bundle_root: Path) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    bundle_rows: list[dict[str, Any]] = []
    resize_tasks: list[tuple[str, str]] = []
    for row in rows:
        target_rel = _bundle_image_relpath(row)
        target_abs = bundle_root / target_rel
        bundle_row = dict(row)
        bundle_row["image_path"] = str(target_rel)
        bundle_row["dataset_root"] = _dataset_root_for(row)
        bundle_row["dataset_version"] = _dataset_version_for(row)
        bundle_rows.append(bundle_row)
        resize_tasks.append((str(Path(row["image_path"])), str(target_abs)))
    return bundle_rows, resize_tasks


def resize_images(tasks: list[tuple[str, str]], workers: int) -> dict[str, int]:
    done = 0
    written = 0
    skipped = 0
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_resize_one, task) for task in tasks]
        for future in as_completed(futures):
            _, did_write, error = future.result()
            done += 1
            if error:
                errors.append(error)
            elif did_write:
                written += 1
            else:
                skipped += 1
            if done % 5000 == 0:
                print(f"resized progress {done}/{len(tasks)} written={written} skipped={skipped} errors={len(errors)}", flush=True)
    if errors:
        raise RuntimeError("Image resize failed:\n" + "\n".join(errors[:20]))
    return {"total": len(tasks), "written": written, "skipped": skipped}


def rsync_path(source: Path, target: Path, *, exclude_pycache: bool = True) -> None:
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        source_arg = str(source) + "/"
        target_arg = str(target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        source_arg = str(source)
        target_arg = str(target)
    command = ["rsync", "-a", "--delete"]
    if exclude_pycache:
        command += ["--exclude", "__pycache__", "--exclude", "*.pyc", "--exclude", ".pytest_cache"]
    command += [source_arg, target_arg]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def copy_code(bundle_root: Path) -> None:
    for directory in ["src", "scripts", "configs", "tests", "docs"]:
        source = REPO_ROOT / directory
        if source.exists():
            rsync_path(source, bundle_root / directory)
    for filename in ["pyproject.toml", "README.md", "AGENTS.md"]:
        source = REPO_ROOT / filename
        if source.exists():
            target = bundle_root / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def copy_required_weights(bundle_root: Path) -> None:
    for source_rel in [
        Path("artifacts/checkpoints/e0_medium_v1/best.pt"),
        Path("artifacts/checkpoints/external/sd-vae-ft-ema"),
        Path("artifacts/checkpoints/external/meanflow_sit"),
    ]:
        source = REPO_ROOT / source_rel
        target = bundle_root / source_rel
        if source.is_dir():
            rsync_path(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    for key in ["model_state_dict", "ema_model_state_dict", "optimizer_state_dict", "metrics"]:
        if key not in payload:
            raise ValueError(f"checkpoint missing {key}")
    return payload


def freeze_e15_snapshot(bundle_root: Path, retries: int = 6, sleep_seconds: int = 20) -> dict[str, Any]:
    source = REPO_ROOT / SOURCE_E15_CKPT_DIR / "last.pt"
    target_dir = bundle_root / BUNDLE_CKPT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "last.pt"
    tmp = target_dir / "last.pt.tmp"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        shutil.copy2(source, tmp)
        try:
            payload = _load_checkpoint(tmp)
            tmp.replace(target)
            metrics = payload.get("metrics", {})
            for name in ["last_metrics.json", "metrics_history.jsonl"]:
                src = REPO_ROOT / SOURCE_E15_CKPT_DIR / name
                if src.is_file():
                    shutil.copy2(src, target_dir / name)
            return {
                "stage_epoch_1based": metrics.get("stage_epoch_1based"),
                "raw_stage_epoch": metrics.get("stage_epoch"),
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            }
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to copy a readable E15 checkpoint after {retries} attempts: {last_error}")


def copy_feature_caches(bundle_root: Path) -> None:
    for source_rel, target_rel in [
        (SOURCE_TRAIN_CACHE, BUNDLE_TRAIN_CACHE),
        (SOURCE_VAL_CACHE, BUNDLE_VAL_CACHE),
    ]:
        source = REPO_ROOT / source_rel
        target = bundle_root / target_rel
        target.mkdir(parents=True, exist_ok=True)
        for name in ["features.pt", "manifest.json"]:
            shutil.copy2(source / name, target / name)


def write_h100_template_config(bundle_root: Path) -> None:
    source_config = yaml.safe_load((REPO_ROOT / SOURCE_E15_CONFIG).read_text(encoding="utf-8"))
    source_config["experiment_name"] = "e15_meanflow_sit_b_face_mixed_h100_resume_2400ep"
    source_config["global_batch_size"] = 384
    source_config["per_device_batch_size"] = 384
    source_config["train_index"] = "data/index_runtime/train_face_mixed_e15_h100.jsonl"
    source_config["train_features"] = "artifacts/e0_features/train_face_mixed_e15_h100_e0_medium_v1_runtime"
    source_config["e0_checkpoint"] = "artifacts/checkpoints/e0_medium_v1/best.pt"
    source_config["out_dir"] = str(BUNDLE_CKPT_DIR)
    source_config["resume_from"] = str(BUNDLE_CKPT_DIR / "last.pt")
    source_config["resume_mode"] = "training_state"
    source_config["resume_optimizer_state"] = True
    source_config["vae_path"] = "artifacts/checkpoints/external/sd-vae-ft-ema"
    source_config.setdefault("distributed", {})["backend"] = "nccl"
    source_config.setdefault("generator", {})["attention_backend"] = "auto"
    source_config["generator"]["sit_pretrained_path"] = "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_b_4_imagenet256.pt"
    quality = source_config["stages"]["stage2"]["quality_eval"]
    quality["real_index"] = "data/index_runtime/val_face_mixed_e15_h100.jsonl"
    quality["output_dir"] = "artifacts/eval/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/quality"
    quality["distribution_cuda_visible_devices"] = "0"
    quality["distribution_device"] = "cuda:0"
    source_config["validation"]["index"] = "data/index_runtime/val_face_mixed_e15_h100.jsonl"
    source_config["validation"]["features"] = "artifacts/e0_features/val_face_mixed_e15_h100_e0_medium_v1_runtime"

    target = bundle_root / H100_TEMPLATE_CONFIG
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(source_config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def create_archive(export_root: Path, bundle_parent: Path, bundle_name: str, compression_level: int) -> Path:
    archive = export_root / f"{bundle_parent.name}.tar.zst"
    if archive.exists():
        archive.unlink()
    command = [
        "tar",
        "--use-compress-program",
        f"zstd -T0 -{compression_level}",
        "-cf",
        str(archive),
        "-C",
        str(bundle_parent.parent),
        bundle_parent.name,
    ]
    subprocess.run(command, check=True)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a <100GB MeanFlow E15 H100 resume bundle.")
    parser.add_argument("--export-root", default="artifacts/exports")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 8)))
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--skip-archive", action="store_true")
    args = parser.parse_args()

    export_root = REPO_ROOT / args.export_root
    bundle_parent = export_root / f"meanflow_e15_h100_bundle_{args.date}"
    bundle_root = bundle_parent / "meanflow_e15_h100_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)

    print(f"bundle_root={bundle_root}", flush=True)
    copy_code(bundle_root)
    write_h100_template_config(bundle_root)
    copy_required_weights(bundle_root)
    copy_feature_caches(bundle_root)

    train_rows = read_jsonl(REPO_ROOT / SOURCE_TRAIN_INDEX)
    val_rows = read_jsonl(REPO_ROOT / SOURCE_VAL_INDEX)
    bundle_train_rows, train_tasks = _make_bundle_rows(train_rows, bundle_root)
    bundle_val_rows, val_tasks = _make_bundle_rows(val_rows, bundle_root)
    write_jsonl(bundle_root / BUNDLE_TRAIN_INDEX, bundle_train_rows)
    write_jsonl(bundle_root / BUNDLE_VAL_INDEX, bundle_val_rows)

    resize_summary = resize_images(train_tasks + val_tasks, workers=args.workers)
    snapshot = freeze_e15_snapshot(bundle_root)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bundle": bundle_root.name,
        "train_count": len(bundle_train_rows),
        "val_count": len(bundle_val_rows),
        "image_format": "JPEG q95 256x256 RGB",
        "checkpoint": snapshot,
        "resize": resize_summary,
        "paths": {
            "checkpoint": str(BUNDLE_CKPT_DIR / "last.pt"),
            "train_index": str(BUNDLE_TRAIN_INDEX),
            "val_index": str(BUNDLE_VAL_INDEX),
            "template_config": str(H100_TEMPLATE_CONFIG),
        },
    }
    manifest_path = bundle_root / "BUNDLE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    archive = None
    if not args.skip_archive:
        archive = create_archive(export_root, bundle_parent, bundle_root.name, args.compression_level)
        sha_path = archive.with_suffix(archive.suffix + ".sha256")
        sha_path.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
        manifest["archive"] = {"path": str(archive), "bytes": archive.stat().st_size, "sha256": sha256_file(archive)}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"bundle_root": str(bundle_root), "archive": str(archive) if archive else None, **manifest}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
