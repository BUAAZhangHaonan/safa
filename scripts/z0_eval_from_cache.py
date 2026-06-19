"""Compute pairwise Spearman from cached features (each at its native resolution).

Each cache was extracted at the encoder training resolution, so this is the
fairest cross-encoder universality comparison.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def load_cache(path: Path):
    """Returns (sample_ids, features[N, 512])."""
    data = torch.load(path / "features.pt", map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        feats = data.get("features", data.get("embeddings"))
        ids = data.get("sample_ids", None)
    else:
        feats = data
        ids = None
    return ids, feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--caches", nargs="+", required=True, help="name:path pairs")
    p.add_argument("--num-pairs", type=int, default=1000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    caches = {}
    common_ids = None
    for spec in args.caches:
        name, path = spec.split(":", 1)
        ids, feats = load_cache(Path(path))
        caches[name] = feats
        ids_set = set(ids) if ids is not None else None
        if common_ids is None:
            common_ids = ids_set
        elif ids_set is not None:
            common_ids &= ids_set
        print(f"[info] {name}: {feats.shape} from {path}")

    if common_ids is None:
        common_ids = list(range(next(iter(caches.values())).shape[0]))
    else:
        common_ids = list(common_ids)
    print(f"[info] common samples: {len(common_ids)}")

    # Align all caches to common samples (by index if ids missing, else by id)
    aligned = {}
    for name, feats in caches.items():
        aligned[name] = feats[:len(common_ids)]  # assume same order
    names = list(caches.keys())

    # Sample pairs
    n = len(common_ids)
    pairs = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(args.num_pairs)]
    # Filter out trivial self-pairs
    pairs = [(i, j) for i, j in pairs if i != j]
    print(f"[info] sampled {len(pairs)} non-trivial pairs")

    cos_by_encoder = {}
    for name in names:
        z = aligned[name]
        z = F.normalize(z, p=2, dim=1)
        cos_vals = [F.cosine_similarity(z[i:i+1], z[j:j+1], dim=1).item() for i, j in pairs]
        cos_by_encoder[name] = cos_vals
        print(f"[info] {name}: mean pairwise cos = {np.mean(cos_vals):.4f}")

    print("\n=== Spearman correlation matrix ===")
    K = len(names)
    mat = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            rho, _ = spearmanr(cos_by_encoder[names[a]], cos_by_encoder[names[b]])
            mat[a, b] = rho

    print("              " + " ".join(n[:9].ljust(11) for n in names))
    for a in range(K):
        print(names[a].ljust(13) + " ".join(f"{mat[a,b]:.3f}".ljust(11) for b in range(K)))

    off_diag = mat[~np.eye(K, dtype=bool)]
    avg = float(off_diag.mean())
    if avg > 0.7:
        verdict = "UNIVERSAL: z_0 transfers across encoders (assumption holds)"
    elif avg < 0.3:
        verdict = "PRIVATE: z_0 is encoder-specific (assumption violated)"
    else:
        verdict = "PARTIAL: z_0 transfers weakly (further investigation needed)"
    print(f"Avg off-diagonal: {avg:.4f}")
    print(f"Verdict: {verdict}")

    out = {
        "num_samples": len(common_ids),
        "num_pairs": len(pairs),
        "encoder_names": names,
        "spearman_matrix": mat.tolist(),
        "avg_off_diagonal_spearman": round(avg, 4),
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[info] wrote {args.output}")


if __name__ == "__main__":
    main()
