from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "h100" / "prepare_h100_bundle.py"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_h100_bundle", PREPARE_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_batch_policy_targets_h100_memory_and_scales_global_batch() -> None:
    module = _load_prepare_module()

    assert module.default_batch_for_gpu_count(1) == (384, 384)
    assert module.default_batch_for_gpu_count(2) == (384, 768)
    assert module.default_batch_for_gpu_count(4) == (256, 1024)


def test_prepare_runtime_files_use_bundle_paths_and_never_e14_resume(tmp_path: Path) -> None:
    module = _load_prepare_module()
    bundle = tmp_path / "bundle"
    train_image = bundle / "data" / "face_256_q95" / "affectnet" / "train" / "a.jpg"
    val_image = bundle / "data" / "face_256_q95" / "affectnet" / "val" / "b.jpg"
    train_image.parent.mkdir(parents=True)
    val_image.parent.mkdir(parents=True)
    train_image.write_bytes(b"not-a-real-image-but-present")
    val_image.write_bytes(b"not-a-real-image-but-present")

    train_index = bundle / "data" / "index_bundle" / "train_face_mixed_e15_h100.jsonl"
    val_index = bundle / "data" / "index_bundle" / "val_face_mixed_e15_h100.jsonl"
    train_index.parent.mkdir(parents=True)
    train_row = {
        "sample_id": "train:a.jpg",
        "image_path": "data/face_256_q95/affectnet/train/a.jpg",
        "label": 0,
        "split": "train",
        "dataset_root": "data/face_256_q95/affectnet",
        "dataset_version": "affectnet-256-q95",
    }
    val_row = dict(train_row, sample_id="val:b.jpg", image_path="data/face_256_q95/affectnet/val/b.jpg", split="val")
    train_index.write_text(json.dumps(train_row, sort_keys=True) + "\n", encoding="utf-8")
    val_index.write_text(json.dumps(val_row, sort_keys=True) + "\n", encoding="utf-8")

    e0 = bundle / "artifacts" / "checkpoints" / "e0_medium_v1" / "best.pt"
    e0.parent.mkdir(parents=True)
    e0.write_bytes(b"checkpoint")
    ckpt = bundle / "artifacts" / "checkpoints" / "e15_meanflow_sit_b_face_mixed_h100_resume_2400ep" / "last.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"meanflow")
    vae = bundle / "artifacts" / "checkpoints" / "external" / "sd-vae-ft-ema"
    meanflow = bundle / "artifacts" / "checkpoints" / "external" / "meanflow_sit" / "zhuyu_sit_b_4_imagenet256.pt"
    vae.mkdir(parents=True)
    meanflow.parent.mkdir(parents=True)
    meanflow.write_bytes(b"pretrained")

    config_path = bundle / "configs" / "medium_v2" / "experiments" / "e15_meanflow_sit_b_face_mixed_h100_resume_2400ep.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump({
        "experiment_name": "e15_meanflow_sit_b_face_mixed_h100_resume_2400ep",
        "global_batch_size": 512,
        "per_device_batch_size": 512,
        "train_index": "data/index_bundle/train_face_mixed_e15_h100.jsonl",
        "train_features": "artifacts/e0_features/train_face_mixed_e15_h100_e0_medium_v1",
        "e0_checkpoint": "artifacts/checkpoints/e0_medium_v1/best.pt",
        "out_dir": "artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep",
        "resume_from": "artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last.pt",
        "resume_mode": "training_state",
        "resume_optimizer_state": True,
        "vae_path": "artifacts/checkpoints/external/sd-vae-ft-ema",
        "distributed": {"backend": "gloo"},
        "generator": {"attention_backend": "auto", "sit_pretrained_path": "artifacts/checkpoints/external/meanflow_sit/zhuyu_sit_b_4_imagenet256.pt"},
        "stages": {"stage2": {"epochs": 2400, "quality_eval": {"real_index": "data/index_bundle/val_face_mixed_e15_h100.jsonl", "output_dir": "artifacts/eval/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/quality", "distribution_cuda_visible_devices": "0", "distribution_device": "cuda:0"}}},
        "validation": {"index": "data/index_bundle/val_face_mixed_e15_h100.jsonl", "features": "artifacts/e0_features/val_face_mixed_e15_h100_e0_medium_v1"},
    }), encoding="utf-8")

    for name, source_index in [("train", train_index), ("val", val_index)]:
        cache = bundle / "artifacts" / "e0_features" / f"{name}_face_mixed_e15_h100_e0_medium_v1"
        cache.mkdir(parents=True)
        (cache / "features.pt").write_bytes(b"features")
        (cache / "manifest.json").write_text(json.dumps({
            "dataset": "AffectNet",
            "index_path": str(source_index),
            "index_sha256": "old",
            "encoder_checkpoint": str(e0),
            "encoder_checkpoint_sha256": "old",
            "num_samples": 1,
            "feature_dim": 512,
            "l2_normalized": True,
            "dtype": "float32",
            "shard": "features.pt",
            "shard_sha256": module.sha256_file(cache / "features.pt"),
            "sample_ids": ["train:a.jpg" if name == "train" else "val:b.jpg"],
            "labels": [0],
        }, sort_keys=True), encoding="utf-8")

    runtime_config = module.prepare_bundle(
        bundle,
        config_path=config_path,
        gpu_count=2,
        per_device_batch=384,
        global_batch=768,
        cuda_visible_devices="0,1",
    )

    config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    assert config["distributed"]["backend"] == "nccl"
    assert config["per_device_batch_size"] == 384
    assert config["global_batch_size"] == 768
    assert "e14" not in config["resume_from"]
    assert config["resume_from"] == str(ckpt.resolve())
    assert config["train_index"].endswith("data/index_runtime/train_face_mixed_e15_h100.jsonl")
    assert config["validation"]["index"].endswith("data/index_runtime/val_face_mixed_e15_h100.jsonl")

    runtime_row = json.loads(Path(config["train_index"]).read_text(encoding="utf-8").splitlines()[0])
    assert Path(runtime_row["image_path"]).is_absolute()
    assert Path(runtime_row["image_path"]).is_file()

    manifest = json.loads(Path(config["train_features"]).joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["index_path"] == config["train_index"]
    assert manifest["index_sha256"] == module.sha256_file(Path(config["train_index"]))
    assert manifest["encoder_checkpoint"] == config["e0_checkpoint"]
