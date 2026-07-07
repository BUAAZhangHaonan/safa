#!/usr/bin/env python3
"""Round 1 GPU 3: 2048-sample FID + Sharpness + face_det for sweep_lora_qv_gpu2.

Goal: verify whether 256-sample FID is misleading by computing 2048-sample FID
on the most promising LoRA QV checkpoint. Also collect Sharpness + face_det.
"""
from __future__ import annotations
import sys
import json
import numpy as np
import torch
import cv2
from pathlib import Path

REPO = Path("/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization")
sys.path.insert(0, str(REPO / "src"))

CKPT = REPO / "artifacts/checkpoints/sweep_lora_qv_gpu2/best.pt"
VAL_INDEX = REPO / "data/index/val_face_mixed_e14.jsonl"
VAL_FEATURES = REPO / "artifacts/e0_features/val_face_mixed_e14_e0_medium_v1"
E0_CKPT = REPO / "artifacts/checkpoints/e0_medium_v1/best.pt"
VAE_PATH = REPO / "artifacts/checkpoints/external/sd-vae-ft-ema"
OUT_JSON = REPO / "artifacts/r1_eval2048_qv/result.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES=3 maps to cuda:0
N_SAMPLES = 2048
BATCH = 32
SEED = 1337


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Use the full val index (feature cache validates index SHA256, can't subset).
    # For FID we'll use first N_SAMPLES rows from the full index.
    with VAL_INDEX.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows = rows[:N_SAMPLES]  # for FID sampling
    subset_index = VAL_INDEX  # use full index for eval
    print(f"[r1_eval2048] using full index {VAL_INDEX} ({len(rows)} rows for FID subsample)")

    # Probe checkpoint to detect LoRA keys
    payload = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    state_dict = payload["model_state_dict"]
    has_lora = any("lora_a" in k for k in state_dict.keys())
    print(f"[r1_eval2048] checkpoint has LoRA keys: {has_lora}")

    # Monkey-patch _load_generator to wrap LoRA before load_state_dict
    from safa.models.generator import build_generator, require_generator_model_config
    from safa.models.peft_lora import wrap_backbone_with_lora_target
    import safa.evaluation.runner as runner

    def patched_load(checkpoint_path, cfg, dev):
        pl = torch.load(str(checkpoint_path), map_location=dev, weights_only=False)
        mc = require_generator_model_config(pl, str(checkpoint_path))
        gen = build_generator(mc)
        if has_lora:
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
        "seed": SEED,
        "device": DEVICE,
        "num_workers": 4,
        "batch_size": BATCH,
        "image_size": 256,
        "index": str(VAL_INDEX),
        "features": str(VAL_FEATURES),
        "e0_checkpoint": str(E0_CKPT),
        "g_checkpoint": str(CKPT),
        "checkpoint_model": "raw",
        "out_json": str(OUT_JSON),
        "per_sample_jsonl": str(OUT_JSON.parent / "per_sample.jsonl"),
        "sample_dir": str(OUT_JSON.parent / "samples"),
        "generated_image_dir": str(OUT_JSON.parent / "generated_images"),
        "face_detection": {"enabled": True, "model_name": "buffalo_l",
                          "threshold": 0.5, "single_face_eq1_threshold": 0.98,
                          "latent_cosine_threshold": 0.95},
        "privacy": {"enabled": False},
        "anti_steg": {"enabled": False},
        "vae_path": str(VAE_PATH),
        "vae_scaling_factor": 0.18215,
        "pixel_image_size": 256,
        "latent_training": True,
    }

    from safa.evaluation.runner import run_eval_from_config
    result = run_eval_from_config(config)
    print(f"[r1_eval2048] eval result written to {OUT_JSON}")

    # Compute 2048-sample FID
    gen_dir = Path(config["generated_image_dir"])
    n_gen = len(list(gen_dir.glob("*.png")))
    print(f"[r1_eval2048] computing FID with {n_gen} generated images")

    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)
    fid_metric.eval()

    from torchvision import transforms
    from PIL import Image
    to_tensor = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

    real_paths = [Path(r["image_path"]) for r in rows]
    for p in real_paths:
        img = Image.open(p).convert("RGB")
        t = (to_tensor(img).unsqueeze(0).to(DEVICE) * 255).to(torch.uint8)
        fid_metric.update(t, real=True)

    gen_files = sorted(gen_dir.glob("*.png"))
    for gf in gen_files:
        img = Image.open(gf).convert("RGB")
        t = (to_tensor(img).unsqueeze(0).to(DEVICE) * 255).to(torch.uint8)
        fid_metric.update(t, real=False)

    fid_value = fid_metric.compute().item()
    print(f"[r1_eval2048] FID(2048) = {fid_value:.4f}")

    sharp_values = []
    for gf in gen_files[:512]:
        img_np = cv2.imread(str(gf), cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            continue
        sharp_values.append(cv2.Laplacian(img_np, cv2.CV_64F).var())
    sharp_mean = float(np.mean(sharp_values)) if sharp_values else float("nan")
    print(f"[r1_eval2048] Sharpness(mean Laplacian var, n={len(sharp_values)}) = {sharp_mean:.2f}")

    result_payload = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    result_payload["fid_2048"] = fid_value
    result_payload["sharpness_mean"] = sharp_mean
    result_payload["n_generated"] = len(gen_files)
    result_payload["n_real"] = len(real_paths)
    OUT_JSON.write_text(json.dumps(result_payload, indent=2))
    print(f"[r1_eval2048] DONE, wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
