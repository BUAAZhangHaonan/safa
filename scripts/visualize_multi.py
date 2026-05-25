#!/usr/bin/env python3
"""SAFA multi-experiment comparison: one grid showing same samples across all models."""

import argparse
import math
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [_SCRIPT_DIR, os.path.join(_SCRIPT_DIR, "src")]:
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

EMOTION_LABELS = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description="SAFA multi-experiment comparison grid")
    p.add_argument("--e0-checkpoint", default="artifacts/checkpoints/e0/best.pt")
    p.add_argument("--val-index", default="data/index/val.jsonl")
    p.add_argument("--out-dir", default="artifacts/visualizations")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--sample-steps", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sampling-seed", type=int, default=None,
                   help="Stable x_init base seed. Defaults to checkpoint training_config seed when all checkpoints agree.")
    return p.parse_args()


def load_generator(path, device):
    import torch
    from safa.models.generator import build_generator, require_generator_model_config

    ckpt = torch.load(path, map_location=device, weights_only=False)
    g_config = require_generator_model_config(ckpt, path)
    model = build_generator(g_config).to(device)
    checkpoint_model = _checkpoint_model_from_checkpoint(ckpt, path)
    state_key = "ema_model_state_dict" if checkpoint_model == "ema" else "model_state_dict"
    if state_key not in ckpt or ckpt[state_key] is None:
        raise ValueError(f"{path} is marked as {checkpoint_model} but missing {state_key}")
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, ckpt.get("metrics", {}), ckpt.get("training_config"), _checkpoint_image_size(ckpt), checkpoint_model


def denormalize_imagenet(tensor_chw):
    t = tensor_chw.cpu().clone()
    for c in range(3):
        t[c] = t[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return t.clamp(0.0, 1.0)


def chw_to_numpy(tensor_chw):
    import numpy as np

    return (tensor_chw.cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)


def normalize_for_e0(images_bchw):
    import torch

    mean = torch.tensor(IMAGENET_MEAN, device=images_bchw.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images_bchw.device).view(1, 3, 1, 1)
    return (images_bchw - mean) / std


def process_sample(e0, generator, img_tensor, device, sample_steps, sample_id, sampling_seed, image_size):
    """Process one image through E0 -> G -> E0 pipeline."""
    import torch
    import torch.nn.functional as F
    from safa.utils.sampling import make_x_init_for_sample_ids

    with torch.no_grad():
        e0_out = e0(img_tensor)
        z = e0_out["embedding"]
        pred_orig = e0_out["logits"].argmax(dim=1).item()

        x_init = make_x_init_for_sample_ids([sample_id], sampling_seed, image_size, z.device, z.dtype)
        gen_img = generator.sample(z, steps=sample_steps, x_init=x_init)
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
    import numpy as np
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from safa.models.e0 import load_e0_checkpoint, freeze_e0
    from safa.data.dataset import AffectNetRecords
    from safa.training.transforms import eval_transform

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

    # Load all generators
    generators = {}
    exp_metrics = {}
    training_configs = []
    image_sizes = []
    checkpoint_models = {}
    metric_summaries = {}
    for name, path in experiments:
        print(f"Loading G from {path} ...")
        gen, metrics, training_config, image_size, checkpoint_model = load_generator(path, device)
        generators[name] = gen
        exp_metrics[name] = metrics
        checkpoint_models[name] = checkpoint_model
        metric_summaries[name] = _checkpoint_metric_summary(metrics, checkpoint_model=checkpoint_model, checkpoint_label=name)
        training_configs.append(training_config)
        image_sizes.append(image_size)
        summary = metric_summaries[name]
        print(f"  {name}: model={checkpoint_model} cosine={summary['cosine']:.4f}, single_face={summary['single_face_eq1']:.4f}")

    image_size = _shared_image_size(image_sizes)

    print(f"Loading val set from {args.val_index} ...")
    dataset = AffectNetRecords(index_path=args.val_index, transform=eval_transform(image_size))
    num_samples = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    print(f"  Dataset size: {len(dataset)}, using {num_samples} samples")

    sampling_seed = _sampling_seed(args.sampling_seed, training_configs)

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
            r = process_sample(e0, gen, img_tensor, device, args.sample_steps, sample["sample_id"], sampling_seed, image_size)
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
                summary = metric_summaries[name]
                cos_m = summary["cosine"]
                sf_m = summary["single_face_eq1"]
                ax.set_title(f"{name}\ncos={cos_m:.3f} sf={sf_m:.3f}",
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
    print(f"{'Experiment':<30s} {'Cosine':>8s} {'Single':>8s} {'Label%':>8s} {'Composite':>10s}")
    print(f"{'-'*80}")

    for name, _ in experiments:
        m = exp_metrics[name]
        summary = metric_summaries[name]
        cos_m = summary["cosine"]
        sf_m = summary["single_face_eq1"]
        composite = cos_m * sf_m
        label_match = sum(1 for r in all_results[name] if r["pred_orig"] == r["pred_gen"])
        label_pct = 100.0 * label_match / num_samples
        print(f"{name:<30s} {cos_m:>8.4f} {sf_m:>8.4f} {label_pct:>7.1f}% {composite:>10.4f}")

    print(f"{'='*80}")

    # === Metrics bar chart ===
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    names = [n for n, _ in experiments]
    cosines = [metric_summaries[n]["cosine"] for n in names]
    single_faces = [metric_summaries[n]["single_face_eq1"] for n in names]
    composites = [c * f for c, f in zip(cosines, single_faces)]

    x = np.arange(len(names))
    width = 0.6

    axes[0].bar(x, cosines, width, color="steelblue")
    axes[0].set_title("Cosine Similarity", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    for i, v in enumerate(cosines):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    axes[1].bar(x, single_faces, width, color="coral")
    axes[1].set_title("Single-Face Rate", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.15)
    for i, v in enumerate(single_faces):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    axes[2].bar(x, composites, width, color="seagreen")
    axes[2].set_title("Composite Score (cos x single)", fontsize=11)
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


def _sampling_seed(arg_seed: int | None, training_configs: list) -> int:
    from safa.utils.sampling import sampling_base_seed_from_config

    if arg_seed is not None:
        return int(arg_seed)
    seeds = []
    for config in training_configs:
        if isinstance(config, dict):
            seeds.append(sampling_base_seed_from_config(config))
    unique = set(seeds)
    if len(seeds) == len(training_configs) and len(unique) == 1:
        return seeds[0]
    raise KeyError("Pass --sampling-seed or use checkpoints with one shared training_config.seed/sampling_seed")


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


def _shared_image_size(image_sizes: list[int]) -> int:
    unique = sorted(set(image_sizes))
    if len(unique) != 1:
        raise RuntimeError(f"Multi-checkpoint visualization requires one shared model_config.image_size, got {unique}")
    return unique[0]


def _checkpoint_model_from_checkpoint(checkpoint: dict, path: str) -> str:
    value = checkpoint.get("checkpoint_model")
    if value is None and os.path.basename(str(path)).startswith("best_ema"):
        value = "ema"
    if value is None:
        value = "raw"
    value = str(value)
    if value not in ("raw", "ema"):
        raise ValueError(f"{path}: checkpoint_model must be raw or ema, got {value!r}")
    return value


def _checkpoint_metric_summary(metrics: dict, *, checkpoint_model: str, checkpoint_label: str) -> dict:
    if checkpoint_model not in ("raw", "ema"):
        raise ValueError(f"{checkpoint_label}: checkpoint_model must be raw or ema, got {checkpoint_model!r}")
    if checkpoint_model == "ema":
        cosine_field = "validation_ema_latent_cosine_mean"
        single_field = "validation_ema_single_face_eq1_rate"
        ge1_field = "validation_ema_face_detect_ge1_rate"
        _require_metric(metrics, single_field, f"EMA checkpoint metrics require {single_field}: {checkpoint_label}")
        return {
            "cosine": _metric_value(metrics, cosine_field, checkpoint_label),
            "single_face_eq1": _metric_value(metrics, single_field, checkpoint_label),
            "face_detect_ge1": _metric_value(metrics, ge1_field, checkpoint_label),
            "cosine_source": cosine_field,
            "single_face_source": single_field,
            "face_detect_ge1_source": ge1_field,
        }
    cosine_field = _first_metric_field(metrics, ("validation_raw_latent_cosine_mean", "validation_latent_cosine_mean"), checkpoint_label)
    single_field = _first_metric_field(metrics, ("validation_raw_single_face_eq1_rate", "validation_single_face_eq1_rate"), checkpoint_label)
    ge1_field = _first_metric_field(
        metrics,
        ("validation_raw_face_detect_ge1_rate", "validation_face_detect_ge1_rate", "validation_face_detection_rate"),
        checkpoint_label,
    )
    return {
        "cosine": _metric_value(metrics, cosine_field, checkpoint_label),
        "single_face_eq1": _metric_value(metrics, single_field, checkpoint_label),
        "face_detect_ge1": _metric_value(metrics, ge1_field, checkpoint_label),
        "cosine_source": cosine_field,
        "single_face_source": single_field,
        "face_detect_ge1_source": ge1_field,
    }


def _first_metric_field(metrics: dict, fields: tuple[str, ...], checkpoint_label: str) -> str:
    for field in fields:
        if field in metrics:
            return field
    raise ValueError(f"{checkpoint_label}: missing metric field; expected one of {fields}")


def _require_metric(metrics: dict, field: str, message: str) -> None:
    if field not in metrics:
        raise ValueError(message)


def _metric_value(metrics: dict, field: str, checkpoint_label: str) -> float:
    _require_metric(metrics, field, f"{checkpoint_label}: missing metric field {field}")
    value = metrics[field]
    if isinstance(value, bool):
        raise ValueError(f"{checkpoint_label}.{field} must be numeric, got bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{checkpoint_label}.{field} must be finite, got {value!r}")
    return number


if __name__ == "__main__":
    main()
