#!/usr/bin/env python3
"""Generic 256-sample FID for any checkpoint's pre-generated images.
Use CKPT_NAME env var to specify which checkpoint's generated_images/ to use."""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

REPO = Path("/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization")
sys.path.insert(0, str(REPO / "src"))

CKPT_NAME = os.environ.get("CKPT_NAME", "r1_eval2048_r1_lora_repr_only_lr0.5_gpu1")
GEN_DIR = REPO / f"artifacts/{CKPT_NAME}/generated_images"
VAL_INDEX = REPO / "data/index/val_face_mixed_e14.jsonl"
OUT = REPO / f"artifacts/{CKPT_NAME}/fid_256.json"

DEVICE = "cuda:0"
SEED = 1337
N = 256


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with VAL_INDEX.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rng = np.random.default_rng(SEED)
    real_idx = rng.choice(int(len(rows)), size=N, replace=False)
    real_paths = [Path(rows[i]["image_path"]) for i in real_idx]

    gen_files_all = sorted(GEN_DIR.glob("*.png"))
    print(f"[fid256_generic] CKPT={CKPT_NAME} found {len(gen_files_all)} gen")
    if not gen_files_all:
        print(f"[fid256_generic] ERROR: no gen images")
        return
    gen_idx = rng.choice(int(len(gen_files_all)), size=N, replace=False)
    gen_files = [gen_files_all[i] for i in gen_idx]

    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)
    fid.eval()
    from torchvision import transforms
    from PIL import Image
    to_t = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

    for p in real_paths:
        img = Image.open(p).convert("RGB")
        t = (to_t(img).unsqueeze(0).to(DEVICE) * 255).to(torch.uint8)
        fid.update(t, real=True)
    for gf in gen_files:
        img = Image.open(gf).convert("RGB")
        t = (to_t(img).unsqueeze(0).to(DEVICE) * 255).to(torch.uint8)
        fid.update(t, real=False)

    fid_value = fid.compute().item()
    print(f"[fid256_generic] {CKPT_NAME} FID(256) = {fid_value:.4f}")
    OUT.write_text(json.dumps({"ckpt": CKPT_NAME, "fid_256": fid_value, "n": N}, indent=2))


if __name__ == "__main__":
    main()
