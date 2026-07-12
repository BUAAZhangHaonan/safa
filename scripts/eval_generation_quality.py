#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_IQA_METHOD = "niqe"
DEFAULT_METRICS = ("fid", "kid", "niqe")
SUPPORTED_METRICS = frozenset((*DEFAULT_METRICS, "sharpness"))
REAL_IMAGE_METRICS = frozenset(("fid", "kid"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_manifest_digest(paths: Sequence[Path], labels: Sequence[str]) -> str:
    if len(paths) != len(labels):
        raise ValueError("asset digest labels and paths disagree")
    return hashlib.sha256(
        "".join(
            f"{label}\t{sha256_file(path)}\n"
            for label, path in zip(labels, paths, strict=True)
        ).encode("utf-8")
    ).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reusable_quality_payload(
    output: Path,
    *,
    contract: Mapping[str, Any],
    metric_names: Sequence[str],
    num_generated: int,
    num_real: int | None,
) -> dict[str, Any] | None:
    if not output.is_file():
        return None
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("quality_contract") != dict(contract):
        return None
    if payload.get("metrics") != list(metric_names) or payload.get("num_generated") != num_generated:
        return None
    if num_real is not None and payload.get("num_real") != num_real:
        return None
    scalar_fields = {
        "fid": ("fid",),
        "kid": ("kid_mean", "kid_std"),
    }
    for metric_name, fields in scalar_fields.items():
        if metric_name in metric_names and any(
            not isinstance(payload.get(field), (int, float))
            or not math.isfinite(float(payload[field]))
            for field in fields
        ):
            return None
    if "niqe" in metric_names:
        iqa = payload.get("iqa")
        if not isinstance(iqa, Mapping) or any(
            not isinstance(iqa.get(field), (int, float))
            or not math.isfinite(float(iqa[field]))
            for field in ("mean", "std")
        ):
            return None
    if "sharpness" in metric_names:
        sharpness = payload.get("sharpness")
        if not isinstance(sharpness, Mapping) or any(
            not isinstance(sharpness.get(field), (int, float))
            or not math.isfinite(float(sharpness[field]))
            for field in ("mean", "median", "std", "p05", "p10", "p90", "p95")
        ):
            return None
    return payload


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
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
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: JSONL contains no rows")
    return rows


def read_jsonl_index(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl_objects(path)
    for line_no, row in enumerate(rows, start=1):
        if "image_path" not in row:
            raise ValueError(f"{path}:{line_no}: missing required field 'image_path'")
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


def _unique_sample_rows(path: Path, *, label: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ordered_ids: list[str] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(read_jsonl_objects(path), start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}:{line_no}: {label} requires a non-empty string sample_id")
        if sample_id in rows_by_id:
            raise ValueError(f"duplicate sample_id in {label}: {sample_id!r}")
        ordered_ids.append(sample_id)
        rows_by_id[sample_id] = row
    return ordered_ids, rows_by_id


def _id_set_difference_message(label: str, *, expected: set[str], actual: set[str]) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing IDs {missing!r}")
    if extra:
        details.append(f"extra IDs {extra!r}")
    return f"{label} sample IDs do not match manifest: " + "; ".join(details)


def manifest_image_paths(
    *,
    real_index: Path,
    generated_dir: Path,
    sample_id_manifest: Path,
    per_sample_jsonl: Path,
) -> tuple[list[str], list[Path], list[Path]]:
    manifest_ids, _ = _unique_sample_rows(sample_id_manifest, label="sample-ID manifest")
    manifest_set = set(manifest_ids)
    _, real_rows = _unique_sample_rows(real_index, label="real index")
    missing_real = manifest_set - set(real_rows)
    if missing_real:
        raise ValueError(f"real index is missing manifest IDs {sorted(missing_real)!r}")

    _, generated_rows = _unique_sample_rows(per_sample_jsonl, label="per-sample JSONL")
    if set(generated_rows) != manifest_set:
        raise ValueError(
            _id_set_difference_message("per-sample JSONL", expected=manifest_set, actual=set(generated_rows))
        )

    real_paths: list[Path] = []
    generated_paths: list[Path] = []
    for sample_id in manifest_ids:
        real_value = real_rows[sample_id].get("image_path")
        if not isinstance(real_value, str) or not real_value:
            raise ValueError(f"real index sample {sample_id!r} is missing image_path")
        real_path = Path(real_value)
        if not real_path.is_file():
            raise FileNotFoundError(f"real image does not exist: {real_path}")
        real_paths.append(real_path)

        generated_row = generated_rows[sample_id]
        generated_value = generated_row.get("generated", generated_row.get("generated_image_path"))
        if not isinstance(generated_value, str) or not generated_value:
            raise ValueError(f"per-sample JSONL sample {sample_id!r} is missing generated path")
        generated_path = Path(generated_value)
        if not generated_path.is_file():
            raise FileNotFoundError(f"generated image does not exist: {generated_path}")
        generated_paths.append(generated_path)

    resolved_selected = [path.resolve() for path in generated_paths]
    if len(set(resolved_selected)) != len(resolved_selected):
        raise ValueError("per-sample JSONL maps multiple sample IDs to the same generated image")
    resolved_directory = {path.resolve() for path in generated_image_paths(generated_dir)}
    resolved_expected = set(resolved_selected)
    if resolved_directory != resolved_expected:
        missing_files = sorted(str(path) for path in resolved_expected - resolved_directory)
        extra_files = sorted(str(path) for path in resolved_directory - resolved_expected)
        details = []
        if missing_files:
            details.append(f"missing files {missing_files!r}")
        if extra_files:
            details.append(f"extra files {extra_files!r}")
        raise ValueError("generated-dir files do not match per-sample JSONL: " + "; ".join(details))
    return manifest_ids, real_paths, generated_paths


def sample_id_digest(sample_ids: Iterable[str]) -> str:
    canonical = "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def laplacian_variance(path: Path) -> float:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for Sharpness evaluation") from exc
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"failed to read generated image for Sharpness: {path}")
    value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if not math.isfinite(value):
        raise ValueError(f"Sharpness returned a non-finite value for {path}")
    return value


def sharpness_summary(values: list[float]) -> dict[str, float | str]:
    if not values:
        raise ValueError("cannot summarize empty Sharpness values")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for Sharpness evaluation") from exc
    array = np.asarray(values, dtype=np.float64)
    return {
        "definition": "grayscale_laplacian_variance",
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "p05": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def quality_eval_device(device: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for quality evaluation") from exc
    requested = str(device)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(requested)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is not available: {requested}")
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {selected.index} is unavailable; visible CUDA device count is {torch.cuda.device_count()}"
            )
    return selected


def prepare_metric_for_device(metric, device):
    if metric is None:
        return None, None
    if hasattr(metric, "to"):
        try:
            return metric.to(device), device
        except Exception as exc:
            metric_name = type(metric).__name__
            raise RuntimeError(
                f"failed to move quality metric {metric_name} to device {device}: {exc}"
            ) from exc
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for quality evaluation") from exc
    if torch.device(device).type != "cpu":
        raise RuntimeError(f"quality metric {type(metric).__name__} does not support .to(device); cannot use {device}")
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
    sample_id_manifest: Path | None = None,
    per_sample_jsonl: Path | None = None,
    generation_result: Path | None = None,
    reuse_valid_output: bool = False,
) -> dict[str, Any]:
    metric_names = normalize_metrics(metrics)
    needs_real_images = any(name in REAL_IMAGE_METRICS for name in metric_names)
    manifest_mode = sample_id_manifest is not None or per_sample_jsonl is not None
    manifest_ids: list[str] = []
    if manifest_mode:
        if sample_id_manifest is None or per_sample_jsonl is None:
            raise ValueError("manifest mode requires both --sample-id-manifest and --per-sample-jsonl")
        if max_real is not None or max_generated is not None:
            raise ValueError("manifest mode rejects both --max-real and --max-generated")
        if real_index is None:
            raise ValueError("manifest mode requires --real-index")
        manifest_ids, joined_real_paths, generated_paths = manifest_image_paths(
            real_index=real_index,
            generated_dir=generated_dir,
            sample_id_manifest=sample_id_manifest,
            per_sample_jsonl=per_sample_jsonl,
        )
        contract_real_paths = joined_real_paths
        real_paths = joined_real_paths if needs_real_images else []
    else:
        generated_paths = limited_paths(
            generated_image_paths(generated_dir), max_count=max_generated, seed=subset_seed
        )
        if needs_real_images:
            if real_index is None:
                raise ValueError("real-index is required when FID or KID metrics are enabled")
            real_paths = limited_paths(real_image_paths(real_index), max_count=max_real, seed=subset_seed)
        else:
            real_paths = []
        contract_real_paths = real_paths

    labels = manifest_ids if manifest_mode else [path.name for path in generated_paths]
    real_labels = manifest_ids if manifest_mode else [path.name for path in contract_real_paths]
    quality_contract: dict[str, Any] = {
        "schema_version": 1,
        "metrics": list(metric_names),
        "sample_id_manifest_sha256": (
            sha256_file(sample_id_manifest) if sample_id_manifest is not None else None
        ),
        "per_sample_jsonl_sha256": (
            sha256_file(per_sample_jsonl) if per_sample_jsonl is not None else None
        ),
        "real_asset_manifest_sha256": asset_manifest_digest(contract_real_paths, real_labels),
        "generated_asset_manifest_sha256": asset_manifest_digest(generated_paths, labels),
    }
    if generation_result is not None:
        try:
            generation_payload = json.loads(generation_result.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid generation result JSON: {generation_result}") from exc
        if not isinstance(generation_payload, Mapping) or generation_payload.get("status") != "complete":
            raise ValueError("generation result must be a complete object")
        arm_digest = generation_payload.get("arm_config_sha256")
        if (
            not isinstance(arm_digest, str)
            or len(arm_digest) != 64
            or any(character not in "0123456789abcdef" for character in arm_digest)
        ):
            raise ValueError("generation result requires a 64-character arm config digest")
        quality_contract.update(
            {
                "generation_result_sha256": sha256_file(generation_result),
                "arm_config_sha256": arm_digest,
            }
        )
    cached = (
        reusable_quality_payload(
            output,
            contract=quality_contract,
            metric_names=metric_names,
            num_generated=len(generated_paths),
            num_real=len(real_paths) if needs_real_images else None,
        )
        if reuse_valid_output
        else None
    )
    if cached is not None:
        return cached

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
    sharpness_values: list[float] = []
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
            if "sharpness" in metric_names:
                sharpness_values.append(laplacian_variance(path))

        payload = {
            "metrics": list(metric_names),
            "num_generated": len(generated_paths),
            "quality_contract": quality_contract,
        }
        if needs_real_images:
            payload["num_real"] = len(real_paths)
        if manifest_mode:
            payload["sample_id_manifest"] = str(sample_id_manifest)
            payload["sample_id_count"] = len(manifest_ids)
            payload["sample_id_sha256"] = sample_id_digest(manifest_ids)
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
        if "sharpness" in metric_names:
            payload["sharpness"] = sharpness_summary(sharpness_values)

    atomic_write_json(output, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated image quality with FID, KID, and NIQE."
    )
    parser.add_argument("--real-index", type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-id-manifest", type=Path)
    parser.add_argument("--per-sample-jsonl", type=Path)
    parser.add_argument("--generation-result", type=Path)
    parser.add_argument("--reuse-valid-output", action="store_true")
    parser.add_argument("--max-generated", type=int, default=None)
    parser.add_argument("--max-real", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=1337)
    parser.add_argument("--seed", type=int, dest="subset_seed")
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
            sample_id_manifest=args.sample_id_manifest,
            per_sample_jsonl=args.per_sample_jsonl,
            generation_result=args.generation_result,
            reuse_valid_output=args.reuse_valid_output,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
