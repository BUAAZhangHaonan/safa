"""Evaluate a generator checkpoint with different ODE step counts.

Reports cosine similarity for each step count to diagnose train/val step mismatch.
Usage: python step_count_diagnostic.py --checkpoint <path> [--max_samples 128] [--batch_size 32]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from safa.data.feature_dataset import FeatureAlignedAffectNet
from safa.models.e0 import load_e0_checkpoint
from safa.models.generator import build_generator
from safa.training.losses import normalize_for_e0
from safa.training.transforms import generator_image_transform
from safa.utils.sampling import make_x_init_for_sample_ids, sampling_base_seed_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint")
    parser.add_argument("--e0_checkpoint", default="artifacts/checkpoints/e0/best.pt")
    parser.add_argument("--val_index", default="data/index/val.jsonl")
    parser.add_argument("--val_features", default="artifacts/e0_features/val")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--step_counts", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--sampling-seed", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load generator
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    generator = build_generator(ckpt["model_config"]).to(device)
    generator.load_state_dict(ckpt["model_state_dict"])
    generator.eval()
    sampling_seed = _sampling_seed(args.sampling_seed, ckpt)
    print(f"Loaded generator from {args.checkpoint}")
    if "metrics" in ckpt:
        print(f"  Checkpoint cosine: {ckpt['metrics'].get('validation_latent_cosine_mean', 'N/A')}")
        print(f"  Checkpoint stage: {ckpt['metrics'].get('stage', 'N/A')}, epoch: {ckpt['metrics'].get('stage_epoch', 'N/A')}")

    # Load E0
    e0, _ = load_e0_checkpoint(args.e0_checkpoint, device=str(device))
    e0.to(device)
    e0.eval()

    # Load validation data
    dataset = FeatureAlignedAffectNet(
        args.val_index,
        args.val_features,
        args.e0_checkpoint,
        transform=generator_image_transform(args.image_size),
    )
    indices = list(range(min(args.max_samples, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    results = {}
    for steps in args.step_counts:
        total_cosine = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in loader:
                z = batch["z"].to(device, non_blocking=True)
                sample_ids = list(batch["sample_id"])
                x_init = make_x_init_for_sample_ids(sample_ids, sampling_seed, args.image_size, z.device, z.dtype)
                generated = generator.sample(z, steps=steps, x_init=x_init)
                e0_out = e0(normalize_for_e0(generated))
                cosine = F.cosine_similarity(e0_out["embedding"], z, dim=1)
                total_cosine += float(cosine.sum().cpu())
                total_samples += int(z.shape[0])
        mean_cosine = total_cosine / total_samples
        results[f"steps_{steps}"] = mean_cosine
        print(f"  steps={steps:3d}: cosine={mean_cosine:.4f}  (n={total_samples})")

    # Save results
    out_path = Path(args.checkpoint).parent / "step_count_diagnostic.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")


def _sampling_seed(arg_seed: int | None, checkpoint: dict) -> int:
    if arg_seed is not None:
        return int(arg_seed)
    training_config = checkpoint.get("training_config")
    if isinstance(training_config, dict):
        return sampling_base_seed_from_config(training_config)
    raise KeyError("Pass --sampling-seed or use a checkpoint with training_config.seed/sampling_seed")


if __name__ == "__main__":
    main()
