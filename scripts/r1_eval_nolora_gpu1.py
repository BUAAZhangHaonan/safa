#!/usr/bin/env python3
"""Round 1 GPU 1: eval a checkpoint WITHOUT LoRA wrap (for sweep_lora_baseline_full / sweep_lora_adaln)."""
from __future__ import annotations
import sys
import os
import json
import numpy as np
import torch
import cv2
from pathlib import Path

REPO = Path("/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization")
sys.path.insert(0, str(REPO / "src"))

CKPT_NAME = os.environ.get("R1_CKPT_NAME", "sweep_lora_baseline_full_gpu0")
CKPT = REPO / f"artifacts/checkpoints/{CKPT_NAME}/best.pt"
VAL_INDEX = REPO / "data/index/val_face_mixed_e14.jsonl"
VAL_FEATURES = REPO / "artifacts/e0_features/val_face_mixed_e14_e0_medium_v1"
E0_CKPT = REPO / "artifacts/checkpoints/e0_medium_v1/best.pt"
VAE_PATH = REPO / "artifacts/checkpoints/external/sd-vae-ft-ema"
OUT_DIR = REPO / f"artifacts/r1_eval2048_{CKPT_NAME}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "result.json"

DEVICE = "cuda:0"
SEED = 1337


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    payload = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    state_dict = payload["model_state_dict"]
    has_lora = any("lora_a" in k for k in state_dict.keys())
    print(f"[r1_eval] CKPT={CKPT_NAME} has_lora={has_lora}")

    if has_lora:
        from safa.models.generator import build_generator, require_generator_model_config
        from safa.models.peft_lora import wrap_backbone_with_lora_target
        import safa.evaluation.runner as runner
        # Detect LoRA target from keys
        lora_paths = set()
        for k in state_dict.keys():
            if "lora_a" in k:
                # strip prefix vector_field.blocks.N. and suffix .lora_a.weight
                parts = k.split(".")
                # vector_field.blocks.{N}.{path}.lora_a.weight -> path parts [3:-2]
                path = ".".join(parts[3:-2])
                lora_paths.add(path)
        print(f"[r1_eval] detected LoRA targets: {sorted(lora_paths)}")
        def patched_load(checkpoint_path, cfg, dev):
            pl = torch.load(str(checkpoint_path), map_location=dev, weights_only=False)
            mc = require_generator_model_config(pl, str(checkpoint_path))
            if "sit_pretrained_path" in mc:
                mc["sit_pretrained_path"] = ""
            gen = build_generator(mc)
            wrap_backbone_with_lora_target(
                gen.vector_field, target_modules=sorted(lora_paths),
                rank=8, alpha=4.0,
            )
            gen.load_state_dict(pl["model_state_dict"])
            return gen.to(dev).eval()
        runner._load_generator = patched_load
    else:
        # Patch sit_pretrained_path on the fly by creating a patched checkpoint
        if "sit_pretrained_path" in payload.get("model_config", {}):
            print(f"[r1_eval] patching sit_pretrained_path to empty")
            import copy
            p2 = copy.deepcopy(payload)
            p2["model_config"]["sit_pretrained_path"] = ""
            patched_ckpt = CKPT.parent / f"{CKPT.stem}_nopretrained.pt"
            torch.save(p2, patched_ckpt)
            print(f"[r1_eval] wrote patched ckpt {patched_ckpt}")

    config = {
        "seed": SEED, "device": DEVICE, "num_workers": 4, "batch_size": 32,
        "image_size": 256, "index": str(VAL_INDEX), "features": str(VAL_FEATURES),
        "e0_checkpoint": str(E0_CKPT),
        "g_checkpoint": str(CKPT.parent / f"{CKPT.stem}_nopretrained.pt") if (not has_lora and "sit_pretrained_path" in payload.get("model_config", {})) else str(CKPT),
        "checkpoint_model": "raw", "out_json": str(OUT_JSON),
        "per_sample_jsonl": str(OUT_DIR / "per_sample.jsonl"),
        "sample_dir": str(OUT_DIR / "samples"),
        "generated_image_dir": str(OUT_DIR / "generated_images"),
        "face_detection": {"enabled": True, "model_name": "buffalo_l",
                          "threshold": 0.5, "single_face_eq1_threshold": 0.98,
                          "latent_cosine_threshold": 0.95},
        "privacy": {"enabled": False}, "anti_steg": {"enabled": False},
        "vae_path": str(VAE_PATH), "vae_scaling_factor": 0.18215,
        "pixel_image_size": 256, "latent_training": True,
    }

    print(f"[r1_eval] running full eval (3969 samples) on {CKPT_NAME}")
    from safa.evaluation.runner import run_eval_from_config
    run_eval_from_config(config)

    with VAL_INDEX.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    real_paths = [Path(r["image_path"]) for r in rows[:2048]]
    gen_files = sorted((OUT_DIR / "generated_images").glob("*.png"))[:2048]

    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)
    fid_metric.eval()
    from torchvision import transforms
    from PIL import Image
    to_tensor = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    for p in real_paths:
        img = Image.open(p).convert("RGB")
        t = (to_tensor(img).unsqueeze(0).to(DEVICE) * 255).to(torch.uint8)
        fid_metric.update(t, real=True)
    for gf in gen_files:
        img = Image.open(gf).convert("RGB")
        t = (to_tensor(img).unsqueeze(0).to(DEVICE) * 255).to(torch.uint8)
        fid_metric.update(t, real=False)
    fid_value = fid_metric.compute().item()
    print(f"[r1_eval] FID(2048) = {fid_value:.4f}")

    sharp_values = []
    for gf in gen_files[:512]:
        img_np = cv2.imread(str(gf), cv2.IMREAD_GRAYSCALE)
        if img_np is None: continue
        sharp_values.append(cv2.Laplacian(img_np, cv2.CV_64F).var())
    sharp_mean = float(np.mean(sharp_values)) if sharp_values else float("nan")

    payload_out = json.loads(OUT_JSON.read_text())
    payload_out["fid_2048"] = fid_value
    payload_out["sharpness_mean_generated"] = sharp_mean
    payload_out["n_real"] = len(real_paths)
    payload_out["n_generated"] = len(gen_files)
    OUT_JSON.write_text(json.dumps(payload_out, indent=2))
    print(f"[r1_eval] DONE {CKPT_NAME} FID={fid_value:.4f} sharp={sharp_mean:.2f}")


if __name__ == "__main__":
    main()
