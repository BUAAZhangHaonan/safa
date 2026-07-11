#!/usr/bin/env python3
"""Round 5 universal eval. Usage:
python scripts/r5_eval.py <ckpt_path> <out_dir_name> <gpu> [--lora-target] [--full-ft]

Computes: face detection, FID(2048), sharpness, latent cosine, spearman, source_pred_preserved.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def resolve_repo_root(env=None, *, script_path=None) -> Path:
    environment = os.environ if env is None else env
    override = environment.get("SAFA_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    source = Path(__file__) if script_path is None else Path(script_path)
    return source.resolve().parents[1]


REPO = resolve_repo_root()
sys.path.insert(0, str(REPO / "src"))

CKPT = Path(sys.argv[1])
OUT_NAME = sys.argv[2]
GPU = int(sys.argv[3]) if len(sys.argv) > 3 else 0
USE_LORA_TARGET = "--lora-target" in sys.argv
USE_FULL_FT = "--full-ft" in sys.argv  # no wrapping needed

VAL_INDEX = REPO / "data/index/val_face_mixed_e14.jsonl"
VAL_FEATURES = REPO / "artifacts/e0_features/val_face_mixed_e14_e0_medium_v1"
E0_CKPT = REPO / "artifacts/checkpoints/e0_medium_v1/best.pt"
VAE_PATH = REPO / "artifacts/checkpoints/external/sd-vae-ft-ema"
OUT_DIR = REPO / f"artifacts/r5_eval_{OUT_NAME}"
OUT_JSON = OUT_DIR / "result.json"
DEVICE = f"cuda:{GPU}"
SEED = 1337

print(f"[r5_eval] ckpt={CKPT}")
print(f"[r5_eval] out={OUT_DIR}")
print(f"[r5_eval] device={DEVICE}")
print(f"[r5_eval] lora_target={USE_LORA_TARGET} full_ft={USE_FULL_FT}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with VAL_INDEX.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    payload = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    state_dict = payload["model_state_dict"]
    has_lora_target = any(("attn.qkv" in k or "attn.proj" in k) and "lora_a" in k for k in state_dict)
    has_lora_peft = any("lora_a" in k for k in state_dict)
    print(f"[r5_eval] has_lora_target={has_lora_target}, has_lora_peft={has_lora_peft}")

    from safa.models.generator import build_generator, require_generator_model_config
    from safa.models.peft_lora import wrap_backbone_with_lora_target
    import safa.evaluation.runner as runner

    def patched_load(checkpoint_path, cfg, dev):
        pl = torch.load(str(checkpoint_path), map_location=dev, weights_only=False)
        mc = require_generator_model_config(pl, str(checkpoint_path))
        gen = build_generator(mc)
        if has_lora_target:
            wrap_backbone_with_lora_target(
                gen.vector_field,
                target_modules=["attn.qkv", "attn.proj"],
                rank=8, alpha=4.0,
            )
        sd = pl["model_state_dict"]
        gen.load_state_dict(sd)
        return gen.to(dev).eval()

    runner._load_generator = patched_load

    config = {
        "seed": SEED, "device": DEVICE, "num_workers": 4, "batch_size": 32,
        "image_size": 256, "index": str(VAL_INDEX), "features": str(VAL_FEATURES),
        "e0_checkpoint": str(E0_CKPT), "g_checkpoint": str(CKPT),
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
    print("[r5_eval] running full eval (3969 samples)")
    from safa.evaluation.runner import run_eval_from_config
    run_eval_from_config(config)

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
    print(f"[r5_eval] FID(2048) = {fid_value:.4f}")
    sharp_values = []
    for gf in gen_files[:512]:
        img_np = cv2.imread(str(gf), cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            continue
        sharp_values.append(cv2.Laplacian(img_np, cv2.CV_64F).var())
    sharp_mean = float(np.mean(sharp_values)) if sharp_values else float("nan")
    payload_out = json.loads(OUT_JSON.read_text())
    payload_out["fid_2048"] = fid_value
    payload_out["sharpness_mean_generated"] = sharp_mean
    payload_out["ckpt"] = str(CKPT)
    OUT_JSON.write_text(json.dumps(payload_out, indent=2))
    # Extract key metrics
    cos_mean = payload_out.get("metrics", {}).get("latent_cosine", {}).get("mean", "N/A")
    face_rate = payload_out.get("metrics", {}).get("face_detection", {}).get("face_detect_ge1_rate", {}).get("mean", "N/A")
    src_pres = payload_out.get("metrics", {}).get("source_prediction_preserved", {}).get("mean", "N/A")
    print(f"[r5_eval] DONE  FID={fid_value:.4f}  sharp={sharp_mean:.2f}  cos={cos_mean}  face={face_rate}  src_pres={src_pres}")


if __name__ == "__main__":
    main()
