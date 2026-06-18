"""Offline identity privacy evaluation.

Loads generated images from a finished eval run and computes identity
similarity between source x_0 and generated \hat{x} using the configured
recognizers (arcface / facenet / adaface). Pairs where any recognizer
fails to embed (e.g. generated image has no face) are skipped with reason
and the skip rate is reported.

Usage:
  python -m scripts.identity_privacy_eval \\
    --result artifacts/eval/<run>/best/result.json \\
    --out artifacts/eval/<run>/best/privacy_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "safa").is_dir():
            return parent
    return here.parent.parent


sys.path.insert(0, str((_resolve_repo_root() / "src")))

from safa.evaluation.recognizers import build_recognizers  # noqa: E402


def _load_image_tensor(path: str | Path) -> torch.Tensor:
    """Load image as (1, 3, H, W) tensor with values in [0, 1]."""
    import numpy as np
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _embed_safe(recognizer, image_tensor, device):
    """Embed a single image tensor (1, 3, H, W). Returns (embedding, None) or (None, reason)."""
    try:
        emb = recognizer.embed(image_tensor.to(device))
        return emb.detach().to(device), None
    except RuntimeError as exc:
        msg = str(exc)
        if "expected exactly one face" in msg or "No face" in msg:
            return None, "no_unique_face"
        return None, f"runtime_error:{msg[:80]}"
    except Exception as exc:  # noqa: BLE001
        return None, f"exception:{type(exc).__name__}"


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "count": int(t.numel()),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)) if t.numel() > 1 else 0.0,
        "min": float(t.min()),
        "max": float(t.max()),
        "p10": float(t.quantile(0.10)),
        "p25": float(t.quantile(0.25)),
        "p50": float(t.quantile(0.50)),
        "p75": float(t.quantile(0.75)),
        "p90": float(t.quantile(0.90)),
        "p95": float(t.quantile(0.95)),
        "ge_0.3_ratio": float((t >= 0.3).float().mean()),
        "ge_0.5_ratio": float((t >= 0.5).float().mean()),
        "ge_0.6_ratio": float((t >= 0.6).float().mean()),
        "ge_0.7_ratio": float((t >= 0.7).float().mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="path to result.json from a finished eval")
    parser.add_argument("--out", required=True, help="path to write privacy_report.json")
    parser.add_argument("--recognizers", default="arcface,facenet,adaface",
                        help="comma-separated recognizer names (must match those declared in result.json)")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-batch-size", type=int, default=8)
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    result_path = Path(args.result)
    with open(result_path) as f:
        result = json.load(f)

    generated_dir = repo_root / result["artifacts"]["generated_image_dir"]
    per_sample_path = repo_root / result["artifacts"]["per_sample_jsonl"]

    index_path = repo_root / result["dataset"]["index"]
    sample_to_source = {}
    with open(index_path) as f:
        for line in f:
            row = json.loads(line)
            sample_to_source[row["sample_id"]] = row["image_path"]
    print(f"[identity_privacy_eval] loaded {len(sample_to_source)} source paths from {index_path.name}")

    recognizer_assets = result.get("recognizer_assets", [])
    selected = set(args.recognizers.split(","))
    assets = [r for r in recognizer_assets if r["name"] in selected]
    if not assets:
        raise RuntimeError(f"No recognizers selected. Available: {[r['name'] for r in recognizer_assets]}")

    print(f"[identity_privacy_eval] building recognizers: {[r['name'] for r in assets]}")
    device = torch.device(args.device)
    recognizers = build_recognizers(assets, str(device))

    samples = []
    with open(per_sample_path) as f:
        for line in f:
            row = json.loads(line)
            samples.append(row)
    if args.max_samples and args.max_samples < len(samples):
        samples = samples[: args.max_samples]
    print(f"[identity_privacy_eval] {len(samples)} samples to evaluate")

    per_recognizer: dict[str, dict] = {r.name: {"cosines": [], "skipped": 0, "skip_reasons": {}} for r in recognizers}
    per_sample_out = []

    for sample in tqdm(samples, desc="privacy"):
        artifacts = sample.get("artifacts", {})
        generated_rel = artifacts.get("generated_image_path")
        if not generated_rel:
            continue
        generated_path = repo_root / generated_rel
        if not generated_path.exists():
            for r in recognizers:
                per_recognizer[r.name]["skipped"] += 1
                per_recognizer[r.name]["skip_reasons"]["missing_file"] = (
                    per_recognizer[r.name]["skip_reasons"].get("missing_file", 0) + 1
                )
            continue

        source_rel = sample_to_source.get(sample["sample_id"])
        source_path = Path(source_rel) if source_rel else None
        if source_path is None or not source_path.exists():
            for r in recognizers:
                per_recognizer[r.name]["skipped"] += 1
                per_recognizer[r.name]["skip_reasons"]["missing_source"] = (
                    per_recognizer[r.name]["skip_reasons"].get("missing_source", 0) + 1
                )
            continue

        gen_img = _load_image_tensor(generated_path)
        src_img = _load_image_tensor(source_path)

        row_out = {"sample_id": sample.get("sample_id"), "per_recognizer": {}}
        for r in recognizers:
            src_emb, src_err = _embed_safe(r, src_img, device)
            gen_emb, gen_err = _embed_safe(r, gen_img, device)
            if src_err or gen_err:
                per_recognizer[r.name]["skipped"] += 1
                reason = src_err or gen_err
                per_recognizer[r.name]["skip_reasons"][reason] = (
                    per_recognizer[r.name]["skip_reasons"].get(reason, 0) + 1
                )
                row_out["per_recognizer"][r.name] = {"skipped": reason}
                continue
            cos = float(F.cosine_similarity(src_emb, gen_emb, dim=1)[0])
            per_recognizer[r.name]["cosines"].append(cos)
            row_out["per_recognizer"][r.name] = {"cos": cos}
        per_sample_out.append(row_out)

    summary = {
        "result_file": str(result_path),
        "checkpoint_g": result.get("checkpoints", {}).get("g"),
        "num_samples_total": len(samples),
        "num_samples_processed": len(per_sample_out),
        "per_recognizer": {},
    }
    for name, bucket in per_recognizer.items():
        stats = _stats(bucket["cosines"])
        stats["skipped"] = bucket["skipped"]
        stats["skip_rate"] = bucket["skipped"] / max(1, len(samples))
        stats["skip_reasons"] = bucket["skip_reasons"]
        summary["per_recognizer"][name] = stats

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    per_sample_out_path = out_path.parent / (out_path.stem + "_per_sample.jsonl")
    with open(per_sample_out_path, "w") as f:
        for row in per_sample_out:
            f.write(json.dumps(row) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\n[identity_privacy_eval] report -> {out_path}")
    print(f"[identity_privacy_eval] per-sample -> {per_sample_out_path}")


if __name__ == "__main__":
    main()
