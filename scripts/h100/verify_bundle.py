#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_h100_bundle import prepare_bundle, sha256_file  # noqa: E402


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _require(path: Path, *, file: bool | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if file is True and not path.is_file():
        raise FileNotFoundError(f"Expected file: {path}")
    if file is False and not path.is_dir():
        raise FileNotFoundError(f"Expected directory: {path}")


def _archive_contains(archive: Path, required: list[str]) -> None:
    if archive.suffix == ".zst":
        # Python stdlib cannot read zstd tar streams. The caller also runs tar -tf.
        return
    with tarfile.open(archive, "r:*") as tar:
        names = set(tar.getnames())
    missing = [item for item in required if not any(name.endswith(item) for name in names)]
    if missing:
        raise ValueError(f"Archive missing required entries: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an unpacked MeanFlow H100 resume bundle.")
    parser.add_argument("--bundle-root", default=".")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--per-device-batch", type=int, default=None)
    parser.add_argument("--global-batch", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--expected-train", type=int, default=130000)
    parser.add_argument("--expected-val", type=int, default=3969)
    parser.add_argument("--archive", default=None)
    args = parser.parse_args()

    bundle = Path(args.bundle_root).resolve()
    _require(bundle, file=False)
    runtime_config = prepare_bundle(
        bundle,
        gpu_count=args.gpu_count,
        per_device_batch=args.per_device_batch,
        global_batch=args.global_batch,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))

    required_files = [
        Path(config["resume_from"]),
        Path(config["train_index"]),
        Path(config["validation"]["index"]),
        Path(config["e0_checkpoint"]),
        Path(config["generator"]["sit_pretrained_path"]),
        Path(config["train_features"]) / "manifest.json",
        Path(config["train_features"]) / "features.pt",
        Path(config["validation"]["features"]) / "manifest.json",
        Path(config["validation"]["features"]) / "features.pt",
    ]
    for path in required_files:
        _require(path, file=True)
    _require(Path(config["vae_path"]), file=False)

    train_count = _count_jsonl(Path(config["train_index"]))
    val_count = _count_jsonl(Path(config["validation"]["index"]))
    if train_count != args.expected_train:
        raise ValueError(f"train index count mismatch: {train_count} != {args.expected_train}")
    if val_count != args.expected_val:
        raise ValueError(f"val index count mismatch: {val_count} != {args.expected_val}")
    if "/e14_" in str(config["resume_from"]) or "checkpoints/e14" in str(config["resume_from"]):
        raise ValueError(f"resume_from points at E14 checkpoint: {config['resume_from']}")

    checkpoint_summary = {"loaded": False}
    try:
        import torch

        ckpt = torch.load(config["resume_from"], map_location="cpu", weights_only=False)
        metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
        checkpoint_summary = {
            "loaded": True,
            "stage_epoch_1based": metrics.get("stage_epoch_1based"),
            "has_model": isinstance(ckpt, dict) and "model_state_dict" in ckpt,
            "has_ema": isinstance(ckpt, dict) and "ema_model_state_dict" in ckpt,
            "has_optimizer": isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt,
        }
    except ImportError:
        checkpoint_summary = {"loaded": False, "reason": "torch not installed"}

    if args.archive:
        archive = Path(args.archive).resolve()
        _require(archive, file=True)
        if archive.stat().st_size >= 100_000_000_000:
            raise ValueError(f"Archive is >=100GB: {archive.stat().st_size}")
        _archive_contains(
            archive,
            [
                "scripts/h100/setup_meanflow_env.sh",
                "scripts/h100/train_meanflow_h100.sh",
                "docs/meanflow_h100_resume_bundle.md",
            ],
        )

    summary = {
        "bundle_root": str(bundle),
        "runtime_config": str(runtime_config),
        "train_count": train_count,
        "val_count": val_count,
        "per_device_batch_size": config["per_device_batch_size"],
        "global_batch_size": config["global_batch_size"],
        "distributed_backend": config["distributed"]["backend"],
        "resume_from": config["resume_from"],
        "resume_sha256": sha256_file(Path(config["resume_from"])),
        "checkpoint": checkpoint_summary,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
