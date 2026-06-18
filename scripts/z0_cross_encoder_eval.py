"""z_0 cross-encoder universality evaluation.

Computes pairwise cosine similarity s_X(i,j) = cos(E_0_X(x_i), E_0_X(x_j))
for each E_0 variant X, then reports the 4x4 Spearman rank correlation matrix
to test whether z_0 is a "universal deep representation" across encoders
or E_0-specific.

Usage:
    python scripts/z0_cross_encoder_eval.py \
        --val-index data/index/val_single_face.jsonl \
        --encoders orig:artifacts/checkpoints/e0_medium_v1/best.pt \
                   resnet18:artifacts/checkpoints/e0_resnet18/best.pt \
                   vgg16:artifacts/checkpoints/e0_vgg16/best.pt \
                   dinov2_large:artifacts/checkpoints/e0_dinov2_large/best.pt \
        --num-pairs 500 \
        --device cuda:0 \
        --output artifacts/eval/z0_cross_encoder/result.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--val-index", type=Path, required=True, help="val_single_face.jsonl")
    p.add_argument(
        "--encoders",
        nargs="+",
        required=True,
        help="name:checkpoint_path pairs, e.g. orig:artifacts/.../best.pt",
    )
    p.add_argument("--num-pairs", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-images", type=int, default=2000, help="cap val set size for speed")
    return p.parse_args()


def read_index(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "image_path" in d:
                rows.append(d)
    return rows


def load_image(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    t = torch.from_numpy(_to_np(img)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def _to_np(img):
    import numpy as np
    return numpy_array(img)


def numpy_array(img):
    import numpy as np
    return np.asarray(img)


def load_encoder(checkpoint_path: str, device: str):
    sys.path.insert(0, "src")
    from safa.models.e0 import load_e0_checkpoint
    model, _ = load_e0_checkpoint(checkpoint_path, device=device)
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_embeddings(model, image_paths: list[str], image_size: int, batch_size: int, device: str) -> torch.Tensor:
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = torch.stack([load_image(p, image_size) for p in batch_paths]).to(device)
        out = model(images)
        embeddings.append(out["embedding"].detach().cpu())
    return torch.cat(embeddings, dim=0)


def sample_pairs(n: int, num_pairs: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    pairs = []
    for _ in range(num_pairs):
        i = rng.randrange(n)
        j = rng.randrange(n)
        while j == i:
            j = rng.randrange(n)
        pairs.append((i, j))
    return pairs


def pairwise_cosine(z_matrix: torch.Tensor, pairs: list[tuple[int, int]]) -> list[float]:
    """z_matrix: [N, D]. Returns list of cos(z_i, z_j) for each pair."""
    result = []
    for i, j in pairs:
        cos = F.cosine_similarity(z_matrix[i : i + 1], z_matrix[j : j + 1], dim=1).item()
        result.append(cos)
    return result


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    n = len(a)
    ra = _rank(a)
    rb = _rank(b)
    mean_ra = sum(ra) / n
    mean_rb = sum(rb) / n
    num = sum((x - mean_ra) * (y - mean_rb) for x, y in zip(ra, rb))
    den_a = math.sqrt(sum((x - mean_ra) ** 2 for x in ra))
    den_b = math.sqrt(sum((x - mean_rb) ** 2 for x in rb))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_index(args.val_index)
    if len(rows) > args.max_images:
        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.max_images)
    image_paths = [r["image_path"] for r in rows]
    print(f"[info] using {len(image_paths)} val images", flush=True)

    encoders = []
    for spec in args.encoders:
        if ":" not in spec:
            raise ValueError(f"Invalid encoder spec (need name:path): {spec}")
        name, path = spec.split(":", 1)
        encoders.append((name, path))
    print(f"[info] encoders: {[(n, p) for n, p in encoders]}", flush=True)

    embeddings_by_encoder = {}
    for name, path in encoders:
        print(f"[info] loading {name} from {path}", flush=True)
        model = load_encoder(path, args.device)
        z = extract_embeddings(model, image_paths, args.image_size, args.batch_size, args.device)
        embeddings_by_encoder[name] = z
        print(f"[info] {name}: extracted {z.shape}", flush=True)
        del model
        torch.cuda.empty_cache()

    pairs = sample_pairs(len(image_paths), args.num_pairs, args.seed)
    print(f"[info] sampled {len(pairs)} pairs", flush=True)

    cos_by_encoder = {}
    for name, z in embeddings_by_encoder.items():
        cos_vals = pairwise_cosine(z, pairs)
        cos_by_encoder[name] = cos_vals
        mean_cos = sum(cos_vals) / len(cos_vals)
        print(f"[info] {name}: mean pairwise cos = {mean_cos:.4f}", flush=True)

    encoder_names = [n for n, _ in encoders]
    correlation_matrix = {}
    for n1 in encoder_names:
        correlation_matrix[n1] = {}
        for n2 in encoder_names:
            rho = spearman(cos_by_encoder[n1], cos_by_encoder[n2])
            correlation_matrix[n1][n2] = round(rho, 4)

    off_diag = []
    for i, n1 in enumerate(encoder_names):
        for j, n2 in enumerate(encoder_names):
            if i < j:
                off_diag.append(correlation_matrix[n1][n2])
    avg_off_diag = sum(off_diag) / len(off_diag) if off_diag else 0.0

    if "orig" in encoder_names:
        orig_correlations = []
        for n in encoder_names:
            if n != "orig":
                orig_correlations.append(correlation_matrix["orig"][n])
        avg_vs_orig = sum(orig_correlations) / len(orig_correlations) if orig_correlations else 0.0
    else:
        avg_vs_orig = None

    if avg_off_diag > 0.7:
        verdict = "UNIVERSAL: z_0 transfers across encoders (assumption holds)"
    elif avg_off_diag < 0.3:
        verdict = "PRIVATE: z_0 is encoder-specific (assumption violated)"
    else:
        verdict = "PARTIAL: z_0 transfers weakly (further investigation needed)"

    result = {
        "num_images": len(image_paths),
        "num_pairs": len(pairs),
        "encoders": encoder_names,
        "spearman_matrix": correlation_matrix,
        "avg_off_diagonal_spearman": round(avg_off_diag, 4),
        "avg_vs_orig_spearman": round(avg_vs_orig, 4) if avg_vs_orig is not None else None,
        "verdict": verdict,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Spearman correlation matrix ===")
    print("            " + "  ".join(f"{n[:10]:>10}" for n in encoder_names))
    for n1 in encoder_names:
        row = "  ".join(f"{correlation_matrix[n1][n2]:>10.4f}" for n2 in encoder_names)
        print(f"{n1[:10]:>10}  {row}")
    print(f"\nAvg off-diagonal: {avg_off_diag:.4f}")
    if avg_vs_orig is not None:
        print(f"Avg vs orig:      {avg_vs_orig:.4f}")
    print(f"Verdict: {verdict}")
    print(f"\n[info] wrote {args.output}")


if __name__ == "__main__":
    main()
