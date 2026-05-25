#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_IQA_METHOD = "niqe"
DEFAULT_METRICS = ("fid", "kid", "niqe")
SUPPORTED_METRICS = frozenset(DEFAULT_METRICS)
REAL_IMAGE_METRICS = frozenset(("fid", "kid"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_jsonl_index(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            if "image_path" not in row:
                raise ValueError(f"{path}:{line_no}: missing required field 'image_path'")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: real index contains no rows")
    return rows


def real_image_paths(real_index: Path) -> list[Path]:
    paths = [Path(str(row["image_path"])) for row in read_jsonl_index(real_index)]
    for image_path in paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"real image does not exist: {image_path}")
    if not paths:
        raise ValueError("real index contains no images")
    return paths


def limited_paths(paths: list[Path], *, max_count: int | None, seed: int | None) -> list[Path]:
    if max_count is None:
        return list(paths)
    if isinstance(max_count, bool) or int(max_count) <= 0:
        raise ValueError(f"max_count must be a positive integer, got {max_count!r}")
    limit = int(max_count)
    if limit >= len(paths):
        return list(paths)
    if seed is None:
        return list(paths[:limit])
    seed_text = str(int(seed))
    return sorted(
        paths,
        key=lambda path: hashlib.sha256(f"{seed_text}\0{path.as_posix()}".encode("utf-8")).hexdigest(),
    )[:limit]


def generated_image_paths(generated_dir: Path) -> list[Path]:
    if not generated_dir.is_dir():
        raise NotADirectoryError(f"generated-dir is not a directory: {generated_dir}")
    paths = sorted(
        path
        for path in generated_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError("generated-dir contains no supported images")
    return paths


def quality_eval_device(device: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for quality evaluation") from exc
    requested = str(device)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def prepare_metric_for_device(metric, device):
    if metric is None:
        return None, None
    if hasattr(metric, "to"):
        return metric.to(device), device
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for quality evaluation") from exc
    return metric, torch.device("cpu")


def image_to_device(image, device):
    if device is None:
        return image
    if hasattr(image, "to"):
        return image.to(device, non_blocking=True)
    return image


def seed_metric_randomness(seed: int | None, device) -> None:
    if seed is None:
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for quality evaluation") from exc
    torch.manual_seed(int(seed))
    if getattr(device, "type", None) == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def load_image_uint8(path: Path):
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("numpy, torch, and pillow are required for quality evaluation") from exc

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def create_fid_metric():
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise RuntimeError("torchmetrics[image] is required for FID evaluation") from exc
    return FrechetInceptionDistance(feature=2048, normalize=False)


def create_kid_metric():
    try:
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError as exc:
        raise RuntimeError("torchmetrics[image] is required for KID evaluation") from exc
    return KernelInceptionDistance(subset_size=50, normalize=False)


def create_iqa_metric(method: str):
    try:
        import pyiqa
    except ImportError as exc:
        raise RuntimeError("pyiqa is required for IQA evaluation") from exc
    return pyiqa.create_metric(method)


def metric_scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1)
        if value.numel() != 1:
            raise ValueError(f"metric returned {value.numel()} values where one scalar was expected")
        number = float(value[0].item())
    else:
        number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metric returned non-finite value {number!r}")
    return number


def metric_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu().reshape(-1)
        numbers = [float(item) for item in tensor.tolist()]
    elif isinstance(value, (list, tuple)):
        numbers = []
        for item in value:
            numbers.extend(metric_values(item))
    else:
        numbers = [float(value)]
    if not numbers:
        raise ValueError("metric returned no values")
    for number in numbers:
        if not math.isfinite(number):
            raise ValueError(f"metric returned non-finite value {number!r}")
    return numbers


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize empty IQA values")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": float(mean), "std": float(math.sqrt(variance))}


def normalize_metrics(metrics: Iterable[str] | None) -> tuple[str, ...]:
    values = DEFAULT_METRICS if metrics is None else tuple(metrics)
    if not values:
        raise ValueError("metrics must be a non-empty list")
    parsed = []
    for value in values:
        name = str(value).lower()
        if name not in SUPPORTED_METRICS:
            raise ValueError(f"unsupported quality metric: {value!r}")
        if name in parsed:
            raise ValueError(f"duplicate quality metric: {name!r}")
        parsed.append(name)
    return tuple(parsed)


def evaluate_generation_quality(
    *,
    real_index: Path | None,
    generated_dir: Path,
    output: Path,
    iqa_method: str = DEFAULT_IQA_METHOD,
    metrics: Iterable[str] | None = None,
    max_generated: int | None = None,
    max_real: int | None = None,
    subset_seed: int | None = 1337,
    device: str = "auto",
) -> dict[str, Any]:
    metric_names = normalize_metrics(metrics)
    generated_paths = limited_paths(generated_image_paths(generated_dir), max_count=max_generated, seed=subset_seed)
    needs_real_images = any(name in REAL_IMAGE_METRICS for name in metric_names)
    if needs_real_images:
        if real_index is None:
            raise ValueError("real-index is required when FID or KID metrics are enabled")
        real_paths = limited_paths(real_image_paths(real_index), max_count=max_real, seed=subset_seed)
    else:
        real_paths = []

    selected_device = quality_eval_device(device)
    fid = create_fid_metric() if "fid" in metric_names else None
    kid = create_kid_metric() if "kid" in metric_names else None
    iqa = create_iqa_metric(iqa_method) if "niqe" in metric_names else None
    fid, fid_device = prepare_metric_for_device(fid, selected_device)
    kid, kid_device = prepare_metric_for_device(kid, selected_device)
    iqa, iqa_device = prepare_metric_for_device(iqa, selected_device)
    if fid is not None and hasattr(fid, "eval"):
        fid.eval()
    if kid is not None and hasattr(kid, "eval"):
        kid.eval()
    if iqa is not None and hasattr(iqa, "eval"):
        iqa.eval()

    iqa_values: list[float] = []
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for quality evaluation") from exc

    with torch.no_grad():
        for path in real_paths:
            image = load_image_uint8(path)
            if fid is not None:
                fid.update(image_to_device(image, fid_device), real=True)
            if kid is not None:
                kid.update(image_to_device(image, kid_device), real=True)

        for path in generated_paths:
            image = load_image_uint8(path)
            if fid is not None:
                fid.update(image_to_device(image, fid_device), real=False)
            if kid is not None:
                kid.update(image_to_device(image, kid_device), real=False)
            if iqa is not None:
                iqa_image = image_to_device(image, iqa_device)
                iqa_values.extend(metric_values(iqa(iqa_image.float().div(255.0))))

        payload = {
            "metrics": list(metric_names),
            "num_generated": len(generated_paths),
        }
        if needs_real_images:
            payload["num_real"] = len(real_paths)
        if fid is not None:
            payload["fid"] = metric_scalar(fid.compute())
        if kid is not None:
            seed_metric_randomness(subset_seed, kid_device)
            kid_mean, kid_std = kid.compute()
            payload["kid_mean"] = metric_scalar(kid_mean)
            payload["kid_std"] = metric_scalar(kid_std)
        if iqa is not None:
            iqa_summary = mean_std(iqa_values)
            payload["iqa"] = {
                "method": iqa_method,
                "mean": iqa_summary["mean"],
                "std": iqa_summary["std"],
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated image quality with FID, KID, and NIQE."
    )
    parser.add_argument("--real-index", type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-generated", type=int, default=None)
    parser.add_argument("--max-real", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        choices=sorted(SUPPORTED_METRICS),
        help="Quality metrics to run. Default: fid kid niqe.",
    )
    parser.add_argument(
        "--iqa-method",
        default=DEFAULT_IQA_METHOD,
        help=f"pyIQA no-reference metric to run. Default: {DEFAULT_IQA_METHOD}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evaluate_generation_quality(
            real_index=args.real_index,
            generated_dir=args.generated_dir,
            output=args.output,
            iqa_method=args.iqa_method,
            metrics=args.metrics,
            max_generated=args.max_generated,
            max_real=args.max_real,
            subset_seed=args.subset_seed,
            device=args.device,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
