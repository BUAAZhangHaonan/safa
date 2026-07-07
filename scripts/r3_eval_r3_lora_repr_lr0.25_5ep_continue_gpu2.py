#!/usr/bin/env python3
"""Round 3 GPU 2: 2048-sample FID + Sharpness for r3_lora_repr_lr0.25_5ep_continue_gpu2."""
from __future__ import annotations
import sys, json, os
import numpy as np
import torch, cv2
from pathlib import Path

REPO = Path("/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization")
sys.path.insert(0, str(REPO / "src"))

CKPT = REPO / "artifacts/checkpoints/r3_lora_repr_lr0.25_5ep_continue_gpu2/best.pt"
VAL_INDEX = REPO / "data/index/val_face_mixed_e14.jsonl"
VAL_FEATURES = REPO / "artifacts/e0_features/val_face_mixed_e14_e0_medium_v1"
E0_CKPT = REPO / "artifacts/checkpoints/e0_medium_v1/best.pt"
VAE_PATH = REPO / "artifacts/checkpoints/external/sd-vae-ft-ema"
OUT_DIR = REPO / "artifacts/r3_eval_r3_lora_repr_lr0.25_5ep_continue_gpu2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "result.json"
DEVICE = "cuda:2"
SEED = 1337


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    config = {{
        "seed": SEED, "device": DEVICE, "num_workers": 4, "batch_size": 32,
        "image_size": 256, "index": str(VAL_INDEX), "features": str(VAL_FEATURES),
        "e0_checkpoint": str(E0_CKPT), "g_checkpoint": str(CKPT),
        "checkpoint_model": "raw", "out_json": str(OUT_JSON),
        "per_sample_jsonl": str(OUT_DIR / "per_sample.jsonl"),
        "sample_dir": str(OUT_DIR / "samples"),
        "generated_image_dir": str(OUT_DIR / "generated_images"),
        "face_detection": {{"enabled": True, "model_name": "buffalo_l",
                          "threshold": 0.5, "single_face_eq1_threshold": 0.98,
                          "latent_cosine_threshold": 0.95}},
        "privacy": {{"enabled": False}}, "anti_steg": {{"enabled": False}},
        "vae_path": str(VAE_PATH), "vae_scaling_factor": 0.18215,
        "pixel_image_size": 256, "latent_training": True,
    }}
    print(f"[r3_eval_r3_lora_repr_lr0.25_5ep_continue_gpu2] running full eval (3969 samples)")
    from safa.evaluation.runner import run_eval_from_config
    run_eval_from_config(config)
    print(f"[r3_eval_r3_lora_repr_lr0.25_5ep_continue_gpu2] eval done, computing FID")
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
    print(f"[r3_eval_r3_lora_repr_lr0.25_5ep_continue_gpu2] FID(2048) = {{fid_value:.4f}}")
    sharp_values = []
    for gf in gen_files[:512]:
        img_np = cv2.imread(str(gf), cv2.IMREAD_GRAYSCALE)
        if img_np is None: continue
        sharp_values.append(cv2.Laplacian(img_np, cv2.CV_64F).var())
    sharp_mean = float(np.mean(sharp_values)) if sharp_values else float("nan")
    payload = json.loads(OUT_JSON.read_text())
    payload["fid_2048"] = fid_value
    payload["sharpness_mean_generated"] = sharp_mean
    payload["n_real"] = len(real_paths)
    payload["n_generated"] = len(gen_files)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[r3_eval_r3_lora_repr_lr0.25_5ep_continue_gpu2] DONE, FID={{fid_value:.4f}} sharp={{sharp_mean:.2f}}")


if __name__ == "__main__":
    main()
