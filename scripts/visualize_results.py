#!/usr/bin/env python3
"""SAFA reconstruction visualization: 4x4 grid of original vs generated pairs.

For each of 16 validation images:
  1. E0 encodes original -> z
  2. G generates from z -> reconstructed image
  3. E0 re-encodes generated -> z_hat
  4. Compute cosine(z, z_hat) and predicted emotion labels

Output: PNG grid with annotations showing emotion labels and cosine similarity.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [_SCRIPT_DIR, os.path.join(_SCRIPT_DIR, "src")]:
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from safa.models.generator import build_generator, require_generator_model_config
from safa.models.e0 import load_e0_checkpoint, freeze_e0
from safa.data.dataset import AffectNetRecords
from safa.training.transforms import eval_transform
from safa.utils.sampling import make_x_init_for_sample_ids, sampling_base_seed_from_config

EMOTION_LABELS = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt",
]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def parse_args():
    p = argparse.ArgumentParser(description="SAFA comparison grid visualizer")
    p.add_argument("--e0-checkpoint", default="artifacts/checkpoints/e0/best.pt",
                   help="Path to E0 best.pt checkpoint")
    p.add_argument("--g-checkpoint", default="artifacts/checkpoints/g/best.pt",
                   help="Path to G best.pt checkpoint")
    p.add_argument("--val-index", default="data/index/val.jsonl",
                   help="Path to validation JSONL index file")
    p.add_argument("--out-path", default="artifacts/visualizations/comparison_grid.png",
                   help="Where to save the output PNG")
    p.add_argument("--num-samples", type=int, default=16,
                   help="Number of samples to visualize (default 16)")
    p.add_argument("--sample-steps", type=int, default=32,
                   help="Heun sampler steps for G (default 32)")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device (default cuda:0)")
    p.add_argument("--sampling-seed", type=int, default=None,
                   help="Stable x_init base seed. Defaults to checkpoint training_config seed when present.")
    return p.parse_args()


def load_generator(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    g_config = require_generator_model_config(ckpt, path)
    model = build_generator(g_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def denormalize_imagenet(tensor_chw):
    t = tensor_chw.cpu().clone()
    for c in range(3):
        t[c] = t[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return t.clamp(0.0, 1.0)


def chw_to_numpy(tensor_chw):
    return (tensor_chw.cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)


def normalize_for_e0(images_bchw):
    mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(images_bchw.device)
    std = IMAGENET_STD.view(1, 3, 1, 1).to(images_bchw.device)
    return (images_bchw - mean) / std


def main():
    args = parse_args()
    device = torch.device(args.device)
    num_samples = args.num_samples

    print(f"Loading E0 from {args.e0_checkpoint} ...")
    e0, _ = load_e0_checkpoint(args.e0_checkpoint, device=str(device))
    e0.to(device)
    freeze_e0(e0)

    print(f"Loading G  from {args.g_checkpoint} ...")
    g, g_ckpt = load_generator(args.g_checkpoint, device)
    image_size = _checkpoint_image_size(g_ckpt)
    sampling_seed = _sampling_seed(args.sampling_seed, g_ckpt)

    print(f"Loading val set from {args.val_index} ...")
    dataset = AffectNetRecords(
        index_path=args.val_index,
        transform=eval_transform(image_size),
    )
    total = min(num_samples, len(dataset))
    print(f"  Dataset size: {len(dataset)}, using {total} samples")

    indices = np.linspace(0, len(dataset) - 1, total, dtype=int)

    results = []

    with torch.no_grad():
        for count, idx in enumerate(indices):
            sample = dataset[int(idx)]
            img_tensor = sample["image"].unsqueeze(0).to(device)
            true_label = sample["label"]

            e0_out = e0(img_tensor)
            z = e0_out["embedding"].squeeze(0)
            pred_orig = e0_out["logits"].argmax(dim=1).item()

            sample_id = sample["sample_id"]
            x_init = make_x_init_for_sample_ids([sample_id], sampling_seed, image_size, z.device, z.dtype)
            gen_img = g.sample(z.unsqueeze(0), steps=args.sample_steps, x_init=x_init)

            gen_normalized = normalize_for_e0(gen_img)
            gen_e0_out = e0(gen_normalized)
            z_hat = gen_e0_out["embedding"].squeeze(0)
            pred_gen = gen_e0_out["logits"].argmax(dim=1).item()

            cos_sim = F.cosine_similarity(z.unsqueeze(0), z_hat.unsqueeze(0)).item()

            results.append({
                "original": img_tensor.squeeze(0),
                "generated": gen_img.squeeze(0),
                "true_label": true_label,
                "pred_orig": pred_orig,
                "pred_gen": pred_gen,
                "cos_sim": cos_sim,
            })

            if (count + 1) % 4 == 0 or count == total - 1:
                print(f"  Processed {count + 1}/{total}")

    print("Building comparison grid ...")
    n_cols = 4
    n_rows = (total + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(22, 5.5 * n_rows), dpi=150)
    outer = gridspec.GridSpec(n_rows, n_cols, wspace=0.08, hspace=0.30)

    for i, r in enumerate(results):
        row, col = divmod(i, n_cols)

        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, col], wspace=0.03)

        orig_vis = denormalize_imagenet(r["original"])
        ax_o = fig.add_subplot(inner[0])
        ax_o.imshow(chw_to_numpy(orig_vis))
        ax_o.axis("off")
        true_emotion = EMOTION_LABELS[r["true_label"]]
        pred_emotion_orig = EMOTION_LABELS[r["pred_orig"]]
        ax_o.set_title(
            f"Original\ntrue: {true_emotion} | pred: {pred_emotion_orig}",
            fontsize=8, pad=3,
        )

        ax_g = fig.add_subplot(inner[1])
        ax_g.imshow(chw_to_numpy(r["generated"]))
        ax_g.axis("off")
        pred_emotion_gen = EMOTION_LABELS[r["pred_gen"]]
        cos_val = r["cos_sim"]
        color = "green" if r["pred_orig"] == r["pred_gen"] else "red"
        ax_g.set_title(
            f"Generated\npred: {pred_emotion_gen} | cos: {cos_val:.3f}",
            fontsize=8, pad=3, color=color,
        )

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(args.out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved grid to {args.out_path}")

    cos_sims = [r["cos_sim"] for r in results]
    label_consistent = sum(1 for r in results if r["pred_orig"] == r["pred_gen"])
    true_correct = sum(1 for r in results if r["pred_orig"] == r["true_label"])

    print(f"\n--- Summary ({total} samples) ---")
    print(f"  Cosine sim   : mean={np.mean(cos_sims):.4f}  "
          f"min={np.min(cos_sims):.4f}  max={np.max(cos_sims):.4f}")
    print(f"  Label match  : {label_consistent}/{total} "
          f"({100 * label_consistent / total:.1f}% orig==gen)")
    print(f"  E0 accuracy  : {true_correct}/{total} "
          f"({100 * true_correct / total:.1f}% pred==true)")


def _sampling_seed(arg_seed: int | None, checkpoint: dict) -> int:
    if arg_seed is not None:
        return int(arg_seed)
    training_config = checkpoint.get("training_config")
    if isinstance(training_config, dict):
        return sampling_base_seed_from_config(training_config)
    raise KeyError("Pass --sampling-seed or use a checkpoint with training_config.seed/sampling_seed")


def _checkpoint_image_size(checkpoint: dict) -> int:
    model_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if not isinstance(model_config, dict) or "image_size" not in model_config:
        raise ValueError("Generator checkpoint missing model_config.image_size")
    value = model_config["image_size"]
    if isinstance(value, bool):
        raise ValueError(f"Generator checkpoint model_config.image_size must be a positive integer, got {value!r}")
    try:
        image_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Generator checkpoint model_config.image_size must be a positive integer, got {value!r}") from exc
    if image_size <= 0:
        raise ValueError(f"Generator checkpoint model_config.image_size must be a positive integer, got {value!r}")
    return image_size


if __name__ == "__main__":
    main()
