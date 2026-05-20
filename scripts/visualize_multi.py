#!/usr/bin/env python3
"""SAFA multi-experiment comparison: one grid showing same samples across all models."""

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

from safa.models.generator import build_generator
from safa.models.e0 import load_e0_checkpoint, freeze_e0
from safa.data.dataset import AffectNetRecords
from safa.training.transforms import eval_transform

EMOTION_LABELS = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt",
]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def parse_args():
    p = argparse.ArgumentParser(description="SAFA multi-experiment comparison grid")
    p.add_argument("--e0-checkpoint", default="artifacts/checkpoints/e0/best.pt")
    p.add_argument("--val-index", default="data/index/val.jsonl")
    p.add_argument("--out-dir", default="artifacts/visualizations")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--sample-steps", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_generator(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    g_config = ckpt["model_config"]
    model = build_generator(g_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("metrics", {})


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


def process_sample(e0, generator, img_tensor, device, sample_steps):
    """Process one image through E0 -> G -> E0 pipeline."""
    with torch.no_grad():
        e0_out = e0(img_tensor)
        z = e0_out["embedding"]
        pred_orig = e0_out["logits"].argmax(dim=1).item()

        gen_img = generator.sample(z, steps=sample_steps)
        gen_clamped = gen_img.clamp(-1, 1)

        gen_normalized = normalize_for_e0(gen_clamped)
        gen_e0_out = e0(gen_normalized)
        z_hat = gen_e0_out["embedding"]
        pred_gen = gen_e0_out["logits"].argmax(dim=1).item()

        cos_sim = F.cosine_similarity(z, z_hat).item()

    return {
        "generated": gen_clamped.squeeze(0),
        "pred_orig": pred_orig,
        "pred_gen": pred_gen,
        "cos_sim": cos_sim,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Define all experiments
    experiments = [
        ("V1 (Round 1)", "artifacts/checkpoints/g/best.pt"),
        ("V2 (λ ramp)", "artifacts/checkpoints/g_v2/best.pt"),
        ("Abl A (scratch)", "artifacts/ablation/ablation_a_combined/last.pt"),
        ("Abl B (resume, λ=0.05)", "artifacts/ablation/ablation_b_aggressive/best.pt"),
    ]

    # Filter to only existing checkpoints
    valid_experiments = []
    for name, path in experiments:
        if os.path.isfile(path):
            valid_experiments.append((name, path))
        else:
            print(f"  SKIP {name}: {path} not found")

    experiments = valid_experiments
    if not experiments:
        print("No valid checkpoints found. Aborting.")
        return

    print(f"Loading E0 from {args.e0_checkpoint} ...")
    e0, _ = load_e0_checkpoint(args.e0_checkpoint, device=str(device))
    e0.to(device)
    freeze_e0(e0)

    print(f"Loading val set from {args.val_index} ...")
    dataset = AffectNetRecords(index_path=args.val_index, transform=eval_transform(224))
    num_samples = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    print(f"  Dataset size: {len(dataset)}, using {num_samples} samples")

    # Load all generators
    generators = {}
    exp_metrics = {}
    for name, path in experiments:
        print(f"Loading G from {path} ...")
        gen, metrics = load_generator(path, device)
        generators[name] = gen
        exp_metrics[name] = metrics
        cos = metrics.get("validation_latent_cosine_mean", -1)
        fd = metrics.get("validation_face_detection_rate", -1)
        print(f"  {name}: cosine={cos:.4f}, face_det={fd:.4f}")

    # Process all samples
    all_results = {}  # {exp_name: [results_per_sample]}
    originals = []

    for name, gen in generators.items():
        all_results[name] = []

    for count, idx in enumerate(indices):
        sample = dataset[int(idx)]
        img_tensor = sample["image"].unsqueeze(0).to(device)
        true_label = sample["label"]
        originals.append({
            "image": img_tensor.squeeze(0),
            "true_label": true_label,
        })

        for name, gen in generators.items():
            r = process_sample(e0, gen, img_tensor, device, args.sample_steps)
            all_results[name].append(r)

        if (count + 1) % 4 == 0 or count == num_samples - 1:
            print(f"  Processed {count + 1}/{num_samples}")

    # === Build multi-experiment comparison grid ===
    n_exp = len(experiments)
    n_cols = n_exp + 1  # original + each experiment
    n_rows = num_samples

    fig_w = 4.0 * n_cols
    fig_h = 4.0 * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)

    # Title row
    fig.suptitle("SAFA Multi-Experiment Comparison", fontsize=16, fontweight="bold", y=0.98)

    outer = gridspec.GridSpec(n_rows, n_cols, wspace=0.05, hspace=0.25,
                              top=0.95, bottom=0.02, left=0.02, right=0.98)

    for row_idx in range(num_samples):
        orig = originals[row_idx]
        true_emo = EMOTION_LABELS[orig["true_label"]]

        # Original image column
        ax = fig.add_subplot(outer[row_idx, 0])
        ax.imshow(chw_to_numpy(denormalize_imagenet(orig["image"])))
        ax.axis("off")
        if row_idx == 0:
            ax.set_title("Original", fontsize=10, fontweight="bold", pad=4)
        ax.text(0.5, -0.05, true_emo, transform=ax.transAxes,
                ha="center", va="top", fontsize=8, style="italic")

        # Each experiment column
        for exp_idx, (name, _) in enumerate(experiments):
            col = exp_idx + 1
            r = all_results[name][row_idx]
            ax = fig.add_subplot(outer[row_idx, col])

            gen_vis = r["generated"]
            gen_vis = denormalize_imagenet(gen_vis) if gen_vis.min() < -0.5 else gen_vis.clamp(0, 1)
            ax.imshow(chw_to_numpy(gen_vis.clamp(0, 1)))
            ax.axis("off")

            if row_idx == 0:
                m = exp_metrics[name]
                cos_m = m.get("validation_latent_cosine_mean", -1)
                fd_m = m.get("validation_face_detection_rate", -1)
                ax.set_title(f"{name}\ncos={cos_m:.3f} fd={fd_m:.3f}",
                            fontsize=9, fontweight="bold", pad=4)

            pred_emo = EMOTION_LABELS[r["pred_gen"]]
            color = "green" if r["pred_orig"] == r["pred_gen"] else "red"
            ax.text(0.5, -0.05, f"{pred_emo} ({r['cos_sim']:.3f})",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7, color=color)

    out_path = os.path.join(args.out_dir, "multi_experiment_comparison.png")
    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"\nSaved multi-experiment grid to {out_path}")

    # === Print summary table ===
    print(f"\n{'='*80}")
    print(f"{'Experiment':<30s} {'Cosine':>8s} {'FaceDet':>8s} {'Label%':>8s} {'Composite':>10s}")
    print(f"{'-'*80}")

    for name, _ in experiments:
        m = exp_metrics[name]
        cos_m = m.get("validation_latent_cosine_mean", -1)
        fd_m = m.get("validation_face_detection_rate", -1)
        composite = cos_m * fd_m
        label_match = sum(1 for r in all_results[name] if r["pred_orig"] == r["pred_gen"])
        label_pct = 100.0 * label_match / num_samples
        print(f"{name:<30s} {cos_m:>8.4f} {fd_m:>8.4f} {label_pct:>7.1f}% {composite:>10.4f}")

    print(f"{'='*80}")

    # === Metrics bar chart ===
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    names = [n for n, _ in experiments]
    cosines = [exp_metrics[n].get("validation_latent_cosine_mean", 0) for n in names]
    face_dets = [exp_metrics[n].get("validation_face_detection_rate", 0) for n in names]
    composites = [c * f for c, f in zip(cosines, face_dets)]

    x = np.arange(len(names))
    width = 0.6

    axes[0].bar(x, cosines, width, color="steelblue")
    axes[0].set_title("Cosine Similarity", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    for i, v in enumerate(cosines):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    axes[1].bar(x, face_dets, width, color="coral")
    axes[1].set_title("Face Detection Rate", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.15)
    for i, v in enumerate(face_dets):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    axes[2].bar(x, composites, width, color="seagreen")
    axes[2].set_title("Composite Score (cos × fd)", fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[2].set_ylim(0, 1.05)
    for i, v in enumerate(composites):
        axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    fig2.suptitle("SAFA Experiment Metrics Comparison", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    chart_path = os.path.join(args.out_dir, "metrics_comparison.png")
    fig2.savefig(chart_path, bbox_inches="tight", dpi=120)
    plt.close(fig2)
    print(f"Saved metrics chart to {chart_path}")


if __name__ == "__main__":
    main()
