#!/usr/bin/env python3
"""Round 1 GPU 3 FID-only script: compute 2048-sample FID using already-generated images.

Reuses artifacts/r1_eval2048_qv/generated_images/ (3969 images from previous full-eval run).
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

VAL_INDEX = REPO / "data/index/val_face_mixed_e14.jsonl"
GEN_DIR = REPO / "artifacts/r1_eval2048_qv/generated_images"
RESULT_JSON = REPO / "artifacts/r1_eval2048_qv/result.json"

DEVICE = "cuda:0"
N_SAMPLES = 2048
SEED = 1337


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Real images: first 2048 rows from val index
    with VAL_INDEX.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    real_rows = rows[:N_SAMPLES]
    real_paths = [Path(r["image_path"]) for r in real_rows]
    print(f"[r1_fid_only] using {len(real_paths)} real images")

    # Generated images: sorted files (named by global_index from previous full eval)
    gen_files = sorted(GEN_DIR.glob("*.png"))
    gen_files = gen_files[:N_SAMPLES]
    print(f"[r1_fid_only] using {len(gen_files)} generated images")

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
    print(f"[r1_fid_only] FID(2048) = {fid_value:.4f}")

    # Sharpness on 512 generated images
    sharp_values = []
    for gf in gen_files[:512]:
        img_np = cv2.imread(str(gf), cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            continue
        sharp_values.append(cv2.Laplacian(img_np, cv2.CV_64F).var())
    sharp_mean = float(np.mean(sharp_values)) if sharp_values else float("nan")
    print(f"[r1_fid_only] Sharpness(mean Laplacian var, n={len(sharp_values)}) = {sharp_mean:.2f}")

    # Append to result
    payload = json.loads(RESULT_JSON.read_text())
    payload["fid_2048"] = fid_value
    payload["sharpness_mean_generated"] = sharp_mean
    payload["n_real"] = len(real_paths)
    payload["n_generated"] = len(gen_files)
    RESULT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[r1_fid_only] DONE")


if __name__ == "__main__":
    main()
