#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import numbers
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence


METHODS = (
    "native_nfe1",
    "transport_nfe2",
    "transport_nfe5",
    "paper_nfe8",
)
FFT_BANDS = (
    ("radial_0p00_0p25", 0.0, 0.25),
    ("radial_0p25_0p50", 0.25, 0.50),
    ("radial_0p50_0p75", 0.50, 0.75),
    ("radial_0p75_1p00", 0.75, 1.0),
)
LAPLACIAN_SCALES = (1.0, 0.75, 0.5)
PRIMARY_FREQUENCY_METRIC = "radial_0p50_1p00"
ROI_SIZE = 112


class FrequencyDiagnosticError(ValueError):
    """Raised when the diagnostic contract or evidence is invalid."""


def _reject_json_constant(value: str) -> None:
    raise FrequencyDiagnosticError(f"JSON contains non-finite constant: {value}")


def read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise FrequencyDiagnosticError(f"invalid {label}: {path}") from exc


def read_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as exc:
                raise FrequencyDiagnosticError(
                    f"invalid {label} JSONL row {line_number}: {path}"
                ) from exc
            if not isinstance(row, Mapping):
                raise FrequencyDiagnosticError(
                    f"{label} row {line_number} must be an object"
                )
            rows.append(row)
    if not rows:
        raise FrequencyDiagnosticError(f"{label} is empty: {path}")
    return rows


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise FrequencyDiagnosticError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise FrequencyDiagnosticError(f"{label} must be finite")
    return normalized


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FrequencyDiagnosticError(f"{label} must be an object")
    return value


def resolve_repo_path(repo_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FrequencyDiagnosticError(f"{label} must be a non-empty path")
    raw = Path(value)
    path = (raw if raw.is_absolute() else repo_root / raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def ordered_sample_ids(path: Path) -> list[str]:
    rows = read_jsonl(path, "sample manifest")
    result: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise FrequencyDiagnosticError(
                f"sample manifest row {index} has an invalid sample_id"
            )
        if sample_id in seen:
            raise FrequencyDiagnosticError(
                f"sample manifest contains duplicate sample_id: {sample_id}"
            )
        seen.add(sample_id)
        result.append(sample_id)
    return result


def method_asset_paths(
    path: Path,
    sample_ids: Sequence[str],
    repo_root: Path,
    method: str,
) -> dict[str, Path]:
    rows = read_jsonl(path, f"{method} assets")
    by_id: dict[str, Path] = {}
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise FrequencyDiagnosticError(
                f"{method} asset row {index} has an invalid sample_id"
            )
        if sample_id in by_id:
            raise FrequencyDiagnosticError(
                f"{method} assets contain duplicate sample_id: {sample_id}"
            )
        by_id[sample_id] = resolve_repo_path(
            repo_root, row.get("generated"), f"{method} image for {sample_id}"
        )
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise FrequencyDiagnosticError(
            f"{method} assets are missing {len(missing)} ordered sample IDs; "
            f"first={missing[0]}"
        )
    return {sample_id: by_id[sample_id] for sample_id in sample_ids}


def serialize_exact_one_face(faces: Any, image_shape: Sequence[int]) -> dict[str, Any]:
    import numpy as np

    if not isinstance(faces, Sequence):
        raise FrequencyDiagnosticError("ArcFace analyzer returned a non-sequence")
    if len(faces) != 1:
        raise FrequencyDiagnosticError(
            f"ArcFace exact-one requirement failed: face_count={len(faces)}"
        )
    face = faces[0]
    bbox = np.asarray(getattr(face, "bbox", None), dtype=np.float64)
    kps = np.asarray(getattr(face, "kps", None), dtype=np.float64)
    score = finite_float(getattr(face, "det_score", None), "ArcFace det_score")
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise FrequencyDiagnosticError("ArcFace bbox must contain four finite values")
    if kps.shape != (5, 2) or not np.isfinite(kps).all():
        raise FrequencyDiagnosticError(
            "ArcFace landmarks must contain five finite 2D points"
        )
    if len(image_shape) != 3 or int(image_shape[0]) <= 0 or int(image_shape[1]) <= 0:
        raise FrequencyDiagnosticError("decoded image shape is invalid")
    height, width = int(image_shape[0]), int(image_shape[1])
    area = max(0.0, float(bbox[2] - bbox[0])) * max(
        0.0, float(bbox[3] - bbox[1])
    )
    if area <= 0.0:
        raise FrequencyDiagnosticError("ArcFace bbox has non-positive area")
    return {
        "face_count": 1,
        "bbox": [float(value) for value in bbox],
        "det_score": score,
        "kps": [[float(value) for value in point] for point in kps],
        "bbox_area_ratio": area / float(height * width),
    }


def align_face(
    image: Any,
    detection: Mapping[str, Any],
    aligner: Callable[[Any, Any], Any] | None = None,
) -> Any:
    import numpy as np

    landmarks = np.asarray(detection["kps"], dtype=np.float32)
    if aligner is None:
        try:
            from insightface.utils import face_align
        except ImportError as exc:
            raise RuntimeError("insightface is required for ArcFace alignment") from exc

        def aligner(raw_image: Any, raw_landmarks: Any) -> Any:
            return face_align.norm_crop(
                raw_image,
                landmark=raw_landmarks,
                image_size=ROI_SIZE,
                mode="arcface",
            )

    crop = np.asarray(aligner(image, landmarks))
    if crop.shape != (ROI_SIZE, ROI_SIZE, 3):
        raise FrequencyDiagnosticError(
            f"aligned ArcFace ROI must be {ROI_SIZE}x{ROI_SIZE}x3; got {crop.shape}"
        )
    if not np.isfinite(crop).all():
        raise FrequencyDiagnosticError("aligned ArcFace ROI contains non-finite values")
    return crop


def grayscale_float(image: Any) -> Any:
    import cv2
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 2:
        gray = array
    else:
        raise FrequencyDiagnosticError(f"image has invalid shape: {array.shape}")
    gray = np.asarray(gray, dtype=np.float64)
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        raise FrequencyDiagnosticError("image is too small for frequency analysis")
    if not np.isfinite(gray).all():
        raise FrequencyDiagnosticError("image contains non-finite values")
    if np.issubdtype(array.dtype, np.integer):
        gray /= float(np.iinfo(array.dtype).max)
    return gray


def radial_fft_energy(gray: Any) -> dict[str, float]:
    import numpy as np

    array = grayscale_float(gray)
    height, width = array.shape
    centered = array - float(array.mean())
    window = np.outer(np.hanning(height), np.hanning(width))
    window_power = float(np.square(window).sum())
    if window_power <= 0.0 or not math.isfinite(window_power):
        raise FrequencyDiagnosticError("FFT window has invalid power")
    spectrum = np.fft.fftshift(np.fft.fft2(centered * window))
    power = np.square(np.abs(spectrum)) / window_power
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    radius = np.sqrt(np.square(fy[:, None]) + np.square(fx[None, :]))
    radius /= math.sqrt(0.5**2 + 0.5**2)
    result: dict[str, float] = {}
    for name, lower, upper in FFT_BANDS:
        mask = (radius >= lower) & (radius < upper if upper < 1.0 else radius <= upper)
        if not bool(mask.any()):
            raise FrequencyDiagnosticError(f"FFT radial band is empty: {name}")
        result[name] = finite_float(float(power[mask].mean()), name)
    high_mask = (radius >= 0.5) & (radius <= 1.0)
    result[PRIMARY_FREQUENCY_METRIC] = finite_float(
        float(power[high_mask].mean()), PRIMARY_FREQUENCY_METRIC
    )
    return result


def multiscale_laplacian(gray: Any) -> dict[str, float]:
    import cv2

    array = grayscale_float(gray)
    result: dict[str, float] = {}
    for scale in LAPLACIAN_SCALES:
        if scale == 1.0:
            scaled = array
        else:
            height = max(8, int(round(array.shape[0] * scale)))
            width = max(8, int(round(array.shape[1] * scale)))
            scaled = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
        value = float(cv2.Laplacian(scaled, cv2.CV_64F).var())
        result[f"scale_{scale:.2f}"] = finite_float(
            value, f"Laplacian scale {scale:.2f}"
        )
    return result


def image_metrics(image: Any) -> dict[str, Any]:
    gray = grayscale_float(image)
    return {
        "fft_energy": radial_fft_energy(gray),
        "laplacian_variance": multiscale_laplacian(gray),
        "gray_std": finite_float(float(gray.std()), "grayscale standard deviation"),
    }


def positive_ratio(numerator: Any, denominator: Any, label: str) -> float:
    top = finite_float(numerator, f"{label} numerator")
    bottom = finite_float(denominator, f"{label} denominator")
    if bottom <= 0.0:
        raise FrequencyDiagnosticError(f"{label} denominator must be positive")
    ratio = top / bottom
    if ratio < 0.0:
        raise FrequencyDiagnosticError(f"{label} ratio must be non-negative")
    return finite_float(ratio, f"{label} ratio")


def method_transfers(method_metrics: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for region in ("roi", "full"):
        current = require_mapping(method_metrics.get(region), f"{region} metrics")
        reference = require_mapping(native.get(region), f"native {region} metrics")
        current_fft = require_mapping(current.get("fft_energy"), f"{region} FFT")
        reference_fft = require_mapping(reference.get("fft_energy"), f"native {region} FFT")
        fft_ratios = {
            key: positive_ratio(current_fft.get(key), reference_fft.get(key), f"{region} {key}")
            for key in (*[name for name, _, _ in FFT_BANDS], PRIMARY_FREQUENCY_METRIC)
        }
        current_lap = require_mapping(current.get("laplacian_variance"), f"{region} Laplacian")
        reference_lap = require_mapping(reference.get("laplacian_variance"), f"native {region} Laplacian")
        lap_ratios = {
            key: positive_ratio(current_lap.get(key), reference_lap.get(key), f"{region} {key}")
            for key in reference_lap
        }
        result[region] = {
            "fft_energy_ratio": fft_ratios,
            "laplacian_variance_ratio": lap_ratios,
        }
    return result


def evaluate_decision(rows: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise FrequencyDiagnosticError("decision requires non-empty per-sample rows")
    enabled = rule.get("enabled")
    if not isinstance(enabled, bool):
        raise FrequencyDiagnosticError("decision rule enabled must be boolean")
    def roi_high(row: Mapping[str, Any], method: str) -> float:
        return finite_float(
            require_mapping(
                require_mapping(row.get("methods"), "sample methods").get(method),
                f"sample {method} metrics",
            )["roi"]["fft_energy"][PRIMARY_FREQUENCY_METRIC],
            f"sample {method} ROI high-frequency energy",
        )

    has_method_metrics = all(isinstance(row.get("methods"), Mapping) for row in rows)
    monotonic_count = sum(bool(row.get("roi_nfe1_gt_nfe2_gt_nfe5")) for row in rows)
    nfe5_ratios = [
        finite_float(
            require_mapping(
                require_mapping(row.get("transfers"), "sample transfers").get("transport_nfe5"),
                "sample NFE5 transfer",
            )["roi"]["fft_energy_ratio"][PRIMARY_FREQUENCY_METRIC],
            "sample NFE5 ROI high-frequency ratio",
        )
        for row in rows
    ]
    median_ratio = finite_float(statistics.median(nfe5_ratios), "median NFE5 ROI ratio")
    result: dict[str, Any] = {
        "enabled": enabled,
        "sample_count": len(rows),
        "monotonic_count": monotonic_count,
        "monotonic_fraction": monotonic_count / len(rows),
        "median_nfe5_to_nfe1_roi_high_frequency_ratio": median_ratio,
    }
    if has_method_metrics:
        result["pairwise_counts"] = {
            "nfe1_gt_nfe2": sum(
                roi_high(row, "native_nfe1") > roi_high(row, "transport_nfe2")
                for row in rows
            ),
            "nfe2_gt_nfe5": sum(
                roi_high(row, "transport_nfe2") > roi_high(row, "transport_nfe5")
                for row in rows
            ),
            "nfe1_gt_nfe5": sum(
                roi_high(row, "native_nfe1") > roi_high(row, "transport_nfe5")
                for row in rows
            ),
            "nfe5_gt_paper": sum(
                roi_high(row, "transport_nfe5") > roi_high(row, "paper_nfe8")
                for row in rows
            ),
        }
    if not enabled:
        result["classification"] = "descriptive_only"
        return result
    required = rule.get("required_monotonic_count")
    if isinstance(required, bool) or not isinstance(required, int) or required <= 0:
        raise FrequencyDiagnosticError("required_monotonic_count must be positive")
    if required > len(rows):
        raise FrequencyDiagnosticError("required_monotonic_count exceeds sample count")
    maximum = finite_float(rule.get("median_nfe5_ratio_max"), "NFE5 median ratio maximum")
    if maximum <= 0.0:
        raise FrequencyDiagnosticError("NFE5 median ratio maximum must be positive")
    count_pass = monotonic_count >= required
    ratio_pass = median_ratio <= maximum
    result.update(
        {
            "required_monotonic_count": required,
            "median_nfe5_ratio_max": maximum,
            "monotonic_count_pass": count_pass,
            "median_ratio_pass": ratio_pass,
            "confirmed": count_pass and ratio_pass,
            "classification": (
                "face_roi_sampler_low_pass_confirmed"
                if count_pass and ratio_pass
                else "face_roi_sampler_low_pass_not_confirmed"
            ),
        }
    )
    return result


def summarize_dataset(rows: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    method_summary: dict[str, Any] = {}
    for method in METHODS:
        summary: dict[str, Any] = {}
        for region in ("roi", "full"):
            ratios = [
                row["transfers"][method][region]["fft_energy_ratio"][PRIMARY_FREQUENCY_METRIC]
                for row in rows
            ]
            laplacian = [
                row["transfers"][method][region]["laplacian_variance_ratio"]["scale_1.00"]
                for row in rows
            ]
            summary[region] = {
                "mean_high_frequency_ratio": finite_float(
                    statistics.fmean(ratios), f"{method} {region} mean FFT ratio"
                ),
                "median_high_frequency_ratio": finite_float(
                    statistics.median(ratios), f"{method} {region} median FFT ratio"
                ),
                "mean_laplacian_ratio_scale_1p00": finite_float(
                    statistics.fmean(laplacian), f"{method} {region} mean Laplacian ratio"
                ),
                "median_laplacian_ratio_scale_1p00": finite_float(
                    statistics.median(laplacian), f"{method} {region} median Laplacian ratio"
                ),
            }
        method_summary[method] = summary
    return {
        "sample_count": len(rows),
        "primary_metric": f"roi.fft_energy.{PRIMARY_FREQUENCY_METRIC}",
        "methods": method_summary,
        "decision": evaluate_decision(rows, rule),
    }


def build_analyzer(arcface_request: Path, device: str) -> Any:
    request = require_mapping(read_json(arcface_request, "ArcFace request"), "ArcFace request")
    if request.get("task") != "arcface":
        raise FrequencyDiagnosticError("ArcFace request task must be arcface")
    config = require_mapping(request.get("config"), "ArcFace request config")
    contract = require_mapping(config.get("arcface"), "ArcFace runtime contract")
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from safa.evaluation.r9_evaluator_worker import _production_face_analyzer_factory

    return _production_face_analyzer_factory(contract, device)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_conclusion(summary: Mapping[str, Any]) -> str:
    datasets = require_mapping(summary.get("datasets"), "summary datasets")
    tail = require_mapping(datasets.get("sharpness_tail32"), "tail32 summary")
    decision = require_mapping(tail.get("decision"), "tail32 decision")
    confirmed = decision.get("confirmed") is True
    ratio_pass = decision.get("median_ratio_pass") is True
    tail_methods = require_mapping(tail.get("methods"), "tail32 method summary")
    tail_nfe5 = require_mapping(tail_methods.get("transport_nfe5"), "tail32 NFE5")
    prefix = require_mapping(datasets.get("prefix128"), "prefix128 summary")
    prefix_methods = require_mapping(prefix.get("methods"), "prefix128 method summary")
    prefix_nfe5 = require_mapping(
        prefix_methods.get("transport_nfe5"), "prefix128 NFE5"
    )
    roi_median = finite_float(
        require_mapping(tail_nfe5.get("roi"), "tail32 NFE5 ROI").get(
            "median_high_frequency_ratio"
        ),
        "tail32 NFE5 ROI median",
    )
    full_median = finite_float(
        require_mapping(tail_nfe5.get("full"), "tail32 NFE5 full image").get(
            "median_high_frequency_ratio"
        ),
        "tail32 NFE5 full median",
    )
    prefix_median = finite_float(
        require_mapping(prefix_nfe5.get("roi"), "prefix128 NFE5 ROI").get(
            "median_high_frequency_ratio"
        ),
        "prefix128 NFE5 ROI median",
    )
    if confirmed:
        headline = "The aligned face ROI confirms the predeclared sampler low-pass rule."
        direction = (
            "Stop eta, interval, and checkpoint searches. Preserve the one-step endpoint and "
            "move to a faithful sampler-consistency or endpoint high-frequency objective."
        )
    elif ratio_pass:
        headline = (
            "The contraction magnitude is large, but the aligned face ROI misses the "
            "predeclared monotonic-count rule."
        )
        direction = (
            "Do not call this a global sampler low-pass confirmation. The ROI and full-image "
            "controls both lose high frequency on tail32, while prefix128 is near neutral; "
            "the next diagnostic should explain the non-monotonic tail exceptions rather than "
            "open another eta or schedule grid."
        )
    else:
        headline = "The aligned face ROI does not confirm the predeclared sampler low-pass rule."
        direction = (
            "Compare the full-image control with the aligned ROI before changing the model; "
            "a full-image-only loss means the old sharpness gate is dominated by background or scale."
        )
    lines = [
        "# R12 face-frequency diagnostic",
        "",
        headline,
        "",
        f"- Tail32 monotonic count: `{decision['monotonic_count']}/32` "
        f"(required `>= {decision.get('required_monotonic_count')}`).",
        "- Tail32 median NFE5/NFE1 aligned-ROI high-frequency ratio: "
        f"`{decision['median_nfe5_to_nfe1_roi_high_frequency_ratio']:.6f}` "
        f"(required `<= {decision.get('median_nfe5_ratio_max')}`).",
        f"- Classification: `{decision['classification']}`.",
        f"- Tail32 NFE5 median high-frequency ratio, ROI/full: "
        f"`{roi_median:.6f}/{full_median:.6f}`.",
        f"- Prefix128 NFE5 median aligned-ROI high-frequency ratio: "
        f"`{prefix_median:.6f}`.",
        "",
        direction,
        "",
        "This is a diagnostic over existing images. It is not a screening survivor, privacy "
        "result, Full gate, or formal winner.",
    ]
    return "\n".join(lines) + "\n"


def run(request_path: Path, output_dir: Path, device: str) -> dict[str, Any]:
    import cv2

    repo_root = Path(__file__).resolve().parents[1]
    request_path = request_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    request = require_mapping(read_json(request_path, "diagnostic request"), "diagnostic request")
    if request.get("schema_version") != 1 or request.get("contract_type") != "safa_r12_face_frequency_request_v1":
        raise FrequencyDiagnosticError("diagnostic request identity mismatch")
    arcface_request = resolve_repo_path(
        repo_root, request.get("arcface_request"), "locked R9 ArcFace request"
    )
    raw_datasets = request.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise FrequencyDiagnosticError("diagnostic request datasets must be non-empty")

    prepared: list[dict[str, Any]] = []
    dataset_ids: set[str] = set()
    for index, raw_dataset in enumerate(raw_datasets):
        dataset = require_mapping(raw_dataset, f"dataset {index}")
        dataset_id = dataset.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id or dataset_id in dataset_ids:
            raise FrequencyDiagnosticError(f"dataset {index} has an invalid or duplicate ID")
        dataset_ids.add(dataset_id)
        manifest = resolve_repo_path(repo_root, dataset.get("sample_id_manifest"), f"{dataset_id} manifest")
        sample_ids = ordered_sample_ids(manifest)
        raw_methods = require_mapping(dataset.get("methods"), f"{dataset_id} methods")
        if set(raw_methods) != set(METHODS):
            raise FrequencyDiagnosticError(f"{dataset_id} must bind exactly {METHODS}")
        assets = {
            method: method_asset_paths(
                resolve_repo_path(repo_root, raw_methods[method], f"{dataset_id} {method} JSONL"),
                sample_ids,
                repo_root,
                method,
            )
            for method in METHODS
        }
        rule = require_mapping(dataset.get("decision_rule"), f"{dataset_id} decision rule")
        prepared.append(
            {
                "dataset_id": dataset_id,
                "manifest": str(manifest),
                "sample_ids": sample_ids,
                "assets": assets,
                "rule": dict(rule),
            }
        )

    analyzer = build_analyzer(arcface_request, device)
    cache: dict[Path, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    for dataset in prepared:
        dataset_rows: list[dict[str, Any]] = []
        for ordinal, sample_id in enumerate(dataset["sample_ids"]):
            methods: dict[str, Any] = {}
            image_shape: tuple[int, int, int] | None = None
            for method in METHODS:
                path = dataset["assets"][method][sample_id]
                if path not in cache:
                    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if image is None:
                        raise FrequencyDiagnosticError(f"failed to decode image: {path}")
                    if image_shape is not None and tuple(image.shape) != image_shape:
                        raise FrequencyDiagnosticError(
                            f"{sample_id} method images have inconsistent shapes"
                        )
                    detection = serialize_exact_one_face(analyzer.get(image), image.shape)
                    roi = align_face(image, detection)
                    cache[path] = {
                        "path": str(path),
                        "image_shape": [int(value) for value in image.shape],
                        "arcface": detection,
                        "roi": image_metrics(roi),
                        "full": image_metrics(image),
                    }
                observed = cache[path]
                current_shape = tuple(observed["image_shape"])
                if image_shape is None:
                    image_shape = current_shape
                elif current_shape != image_shape:
                    raise FrequencyDiagnosticError(
                        f"{sample_id} method images have inconsistent shapes"
                    )
                methods[method] = observed
            native = methods["native_nfe1"]
            transfers = {
                method: method_transfers(methods[method], native) for method in METHODS
            }
            roi_high = {
                method: methods[method]["roi"]["fft_energy"][PRIMARY_FREQUENCY_METRIC]
                for method in METHODS
            }
            row = {
                "dataset_id": dataset["dataset_id"],
                "ordinal": ordinal,
                "sample_id": sample_id,
                "methods": methods,
                "transfers": transfers,
                "roi_nfe1_gt_nfe2_gt_nfe5": (
                    roi_high["native_nfe1"]
                    > roi_high["transport_nfe2"]
                    > roi_high["transport_nfe5"]
                ),
                "roi_nfe1_gt_nfe2_gt_nfe5_gt_paper": (
                    roi_high["native_nfe1"]
                    > roi_high["transport_nfe2"]
                    > roi_high["transport_nfe5"]
                    > roi_high["paper_nfe8"]
                ),
            }
            dataset_rows.append(row)
            all_rows.append(row)
        dataset_summaries[dataset["dataset_id"]] = summarize_dataset(
            dataset_rows, dataset["rule"]
        )

    summary = {
        "schema_version": 1,
        "contract_type": "safa_r12_face_frequency_summary_v1",
        "boundary": {
            "existing_images_only": True,
            "new_generation": False,
            "privacy_evaluated": False,
            "formal_gate": False,
        },
        "inputs": {
            "request": str(request_path),
            "arcface_request": str(arcface_request),
            "device": device,
            "detector": "buffalo_l",
            "det_size": [224, 224],
            "aligned_roi_size": [ROI_SIZE, ROI_SIZE],
            "ordered_dataset_ids": [dataset["dataset_id"] for dataset in prepared],
        },
        "metric_contract": {
            "alignment": "ArcFace five-point norm_crop mode=arcface",
            "fft_window": "separable_hann",
            "fft_band_radius_normalization": "distance divided by corner Nyquist radius",
            "fft_bands": [
                {"name": name, "lower_inclusive": lower, "upper": upper}
                for name, lower, upper in FFT_BANDS
            ],
            "primary_frequency_metric": PRIMARY_FREQUENCY_METRIC,
            "laplacian_scales": list(LAPLACIAN_SCALES),
        },
        "unique_image_count": len(cache),
        "row_count": len(all_rows),
        "datasets": dataset_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "per_sample.jsonl").open("x", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "conclusion.md").write_text(
        build_conclusion(summary), encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure aligned-face and full-image frequency transfer over existing R11 outputs."
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args.request, args.output_dir, args.device)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
