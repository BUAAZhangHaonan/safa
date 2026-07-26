"""Locked KID evaluation for canonical screening smoke8 and screen512."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
from typing import Any, Mapping

from safa.closeout.canonical_screening import CanonicalScreeningError, sha256_file


def _load_quality_module(binding: Mapping[str, Any]):
    path = Path(str(binding.get("path", ""))).resolve()
    expected = binding.get("sha256")
    if not path.is_file() or sha256_file(path) != expected:
        raise CanonicalScreeningError("canonical KID quality script binding mismatch")
    module_name = f"_safa_canonical_quality_{expected}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CanonicalScreeningError("quality script has no importable module specification")
    loader_get_data = getattr(spec.loader, "get_data", None)
    if not callable(loader_get_data):
        raise CanonicalScreeningError("quality script loader cannot read locked bytes")
    source = loader_get_data(str(path))
    if not isinstance(source, bytes) or hashlib.sha256(source).hexdigest() != expected:
        raise CanonicalScreeningError("quality script bytes differ during import")
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def evaluate_locked_kid(
    *,
    quality_script: Mapping[str, Any],
    real_index: Path,
    generated_dir: Path,
    sample_id_manifest: Path,
    per_sample_jsonl: Path,
    subset_seed: int,
    subset_size: int,
    device: str,
) -> dict[str, Any]:
    if type(subset_size) is not int or subset_size < 2:
        raise CanonicalScreeningError("KID subset_size must be an integer >= 2")
    module = _load_quality_module(quality_script)
    manifest_ids, real_paths, generated_paths = module.manifest_image_paths(
        real_index=real_index,
        generated_dir=generated_dir,
        sample_id_manifest=sample_id_manifest,
        per_sample_jsonl=per_sample_jsonl,
    )
    if (
        len(manifest_ids) != len(real_paths)
        or len(manifest_ids) != len(generated_paths)
        or len(manifest_ids) < subset_size
    ):
        raise CanonicalScreeningError(
            "KID manifest coverage is smaller than the locked subset size"
        )
    try:
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError as exc:
        raise RuntimeError("torchmetrics[image] is required for canonical KID") from exc
    metric = KernelInceptionDistance(subset_size=subset_size, normalize=False)
    selected_device = module.quality_eval_device(device)
    metric, metric_device = module.prepare_metric_for_device(metric, selected_device)
    if hasattr(metric, "eval"):
        metric.eval()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for canonical KID") from exc
    with torch.no_grad():
        for path in real_paths:
            image = module.load_image_uint8(path)
            metric.update(module.image_to_device(image, metric_device), real=True)
        for path in generated_paths:
            image = module.load_image_uint8(path)
            metric.update(module.image_to_device(image, metric_device), real=False)
    module.seed_metric_randomness(subset_seed, metric_device)
    kid_mean, kid_std = metric.compute()
    mean = module.metric_scalar(kid_mean)
    std = module.metric_scalar(kid_std)
    if not math.isfinite(mean) or not math.isfinite(std):
        raise CanonicalScreeningError("canonical KID result is non-finite")
    return {
        "kid_mean": mean,
        "kid_std": std,
        "kid_subset_size": subset_size,
        "num_real": len(real_paths),
        "num_generated": len(generated_paths),
        "sample_id_manifest_sha256": sha256_file(sample_id_manifest),
        "per_sample_jsonl_sha256": sha256_file(per_sample_jsonl),
        "real_asset_manifest_sha256": module.asset_manifest_digest(
            real_paths, manifest_ids
        ),
        "generated_asset_manifest_sha256": module.asset_manifest_digest(
            generated_paths, manifest_ids
        ),
    }
