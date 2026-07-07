#!/usr/bin/env python3
"""Round 1 GPU 3: 256-sample FID for e15 base."""
from __future__ import annotations
import sys
import json
import numpy as np
import torch
from pathlib import Path

REPO = Path("/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization")
sys.path.insert(0, str(REPO / "src"))

GEN_DIR = REPO / "artifacts/r1_eval2048_e15base/generated_images"
VAL_INDEX = REPO / "data/index/val_face_mixed_e14.jsonl"
OUT = REPO / "artifacts/r1_eval2048_e15base/fid_256.json"

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
    real_rows = [rows[i] for i in real_idx]
    real_paths = [Path(r["image_path"]) for r in real_rows]

    gen_files_all = sorted(GEN_DIR.glob("*.png"))
    print(f"[fid256_e15] found {len(gen_files_all)} gen images")
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
    print(f"[fid256_e15] e15 FID(256) = {fid_value:.4f}")
    OUT.write_text(json.dumps({"ckpt": "e15_base", "fid_256": fid_value, "n": N}, indent=2))


if __name__ == "__main__":
    main()
