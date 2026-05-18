#!/usr/bin/env python3
"""Visualize SAFA results: original vs generated face comparison grid."""

import sys
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safa.models.e0 import load_e0_checkpoint
from safa.models.generator import ConditionalFlowGenerator
from safa.training.transforms import eval_transform, generator_image_transform
from safa.training.losses import normalize_for_e0

EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

def load_image(path: str):
    img = Image.open(path).convert("RGB")
    return img

def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """Convert [C,H,W] tensor in [0,1] to [H,W,C] numpy array."""
    return t.permute(1, 2, 0).cpu().numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--g-checkpoint", type=str, required=True, help="Path to G best.pt")
    parser.add_argument("--e0-checkpoint", type=str, required=True, help="Path to E0 best.pt")
    parser.add_argument("--val-index", type=str, default="data/index/val.jsonl")
    parser.add_argument("--output", type=str, default="artifacts/visualizations/comparison_grid.png")
    parser.add_argument("--n-images", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load E0
    print("Loading E0...")
    e0 = load_e0_checkpoint(args.e0_checkpoint, device=device)
    e0.eval()

    # Load G
    print("Loading G...")
    payload = torch.load(args.g_checkpoint, map_location="cpu")
    config = payload.get("model_config", {})
    generator = ConditionalFlowGenerator(config)
    generator.load_state_dict(payload["model_state_dict"])
    generator.to(device)
    generator.eval()

    # Load val index
    print("Loading val index...")
    samples = []
    with open(args.val_index) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line.strip())
                samples.append(rec)

    # Pick evenly spaced samples
    indices = np.linspace(0, len(samples) - 1, args.n_images, dtype=int)
    selected = [samples[i] for i in indices]

    # Create figure
    n = args.n_images
    cols = 2  # original | generated
    fig, axes = plt.subplots(n, cols, figsize=(8, n * 2.5))

    with torch.no_grad():
        for idx, rec in enumerate(selected):
            # Load original image
            img_path = rec["image_path"]
            img_pil = load_image(img_path)

            # For E0: ImageNet normalized
            img_e0 = eval_transform(224)(img_pil).unsqueeze(0).to(device)
            e0_out = e0(img_e0)
            z = e0_out["embedding"]  # [1, 512]
            pred_label = e0_out["logits"].argmax(dim=1).item()
            orig_label = rec["label"]

            # For G: [0,1] range
            img_g = generator_image_transform(224)(img_pil).unsqueeze(0).to(device)

            # Generate
            generated = generator.sample(z, steps=args.sample_steps)  # [1, 3, 224, 224]

            # E0 on generated
            gen_e0_input = normalize_for_e0(generated)
            gen_e0_out = e0(gen_e0_input)
            z_hat = gen_e0_out["embedding"]
            gen_pred_label = gen_e0_out["logits"].argmax(dim=1).item()

            cos_sim = torch.nn.functional.cosine_similarity(z, z_hat, dim=1).item()

            # Plot original
            ax_orig = axes[idx, 0]
            ax_orig.imshow(img_pil.resize((224, 224)))
            ax_orig.set_title(f"Original: {EMOTION_LABELS[orig_label]}", fontsize=9)
            ax_orig.axis("off")

            # Plot generated
            ax_gen = axes[idx, 1]
            gen_np = tensor_to_np(generated[0])
            gen_np = np.clip(gen_np, 0, 1)
            ax_gen.imshow(gen_np)
            ax_gen.set_title(
                f"Generated: {EMOTION_LABELS[gen_pred_label]}\n"
                f"cos={cos_sim:.3f}",
                fontsize=9,
                color="green" if orig_label == gen_pred_label else "red"
            )
            ax_gen.axis("off")

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison grid to {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
