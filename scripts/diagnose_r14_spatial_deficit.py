#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import numbers
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np


DATASETS = ("regular32", "sharpness_tail32")
ARMS = ("u12", "u16")
BOOTSTRAP_SEED = 91637
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_QUANTILES = (0.025, 0.975)
GRADIENT_SIGMAS = (0.0, 1.0, 2.0)
BACKGROUND_SHARE_THRESHOLD = 0.5
LAPLACIAN = "centered_squared_laplacian_energy"
GRADIENT = "multiscale_gradient_energy"
IDENTITY_ROWS = Path(
    "artifacts/r12_arcface_absolute_identity_risk/v1/per_sample.jsonl"
)
QUALITY_ROOT = Path("artifacts/r12_seed_aligned_trajectory/evaluation_v1")


class SpatialDeficitError(ValueError):
    """Raised when existing evidence violates the diagnostic contract."""


def reject_constant(value: str) -> None:
    raise SpatialDeficitError(f"JSON contains non-finite constant: {value}")


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=reject_constant)
            if not isinstance(row, Mapping):
                raise SpatialDeficitError(f"row {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise SpatialDeficitError(f"empty JSONL: {path}")
    return rows


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise SpatialDeficitError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SpatialDeficitError(f"{label} must be finite")
    return result


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialDeficitError(f"{label} must be an object")
    return value


def exact_one_bbox(value: Any, label: str) -> list[float]:
    detection = mapping(value, label)
    if detection.get("face_count") != 1:
        raise SpatialDeficitError(f"{label} exact-one requirement failed")
    raw = detection.get("bbox")
    if not isinstance(raw, list) or len(raw) != 4:
        raise SpatialDeficitError(f"{label} bbox must contain four values")
    bbox = [finite(item, f"{label} bbox") for item in raw]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise SpatialDeficitError(f"{label} bbox has non-positive area")
    return bbox


def bbox_union_mask(
    image_shape: Sequence[int],
    native_bbox: Sequence[float],
    candidate_bbox: Sequence[float],
) -> np.ndarray:
    if len(image_shape) != 3:
        raise SpatialDeficitError("image shape must be HxWxC")
    height, width, channels = [int(value) for value in image_shape]
    if height <= 0 or width <= 0 or channels != 3:
        raise SpatialDeficitError("image shape must be positive HxWx3")
    x = np.arange(width, dtype=np.float64)[None, :] + 0.5
    y = np.arange(height, dtype=np.float64)[:, None] + 0.5
    mask = np.zeros((height, width), dtype=bool)
    for label, raw in (("native", native_bbox), ("candidate", candidate_bbox)):
        if len(raw) != 4:
            raise SpatialDeficitError(f"{label} bbox must have four values")
        x1, y1, x2, y2 = [finite(item, f"{label} bbox") for item in raw]
        if x2 <= x1 or y2 <= y1:
            raise SpatialDeficitError(f"{label} bbox has non-positive area")
        mask |= (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
    count = int(mask.sum())
    if count <= 0 or count >= height * width:
        raise SpatialDeficitError("bbox union must leave two non-empty regions")
    return mask


def centered_laplacian_energy(gray: Any) -> np.ndarray:
    import cv2

    array = np.asarray(gray)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise SpatialDeficitError("grayscale input must be a finite matrix")
    laplacian = cv2.Laplacian(array, cv2.CV_64F)
    result = np.square(laplacian - float(laplacian.mean()))
    if not np.isfinite(result).all():
        raise SpatialDeficitError("Laplacian energy is non-finite")
    return result


def gradient_energy_maps(gray: Any) -> dict[str, np.ndarray]:
    import cv2

    array = np.asarray(gray, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise SpatialDeficitError("grayscale input must be a finite matrix")
    result = {}
    for sigma in GRADIENT_SIGMAS:
        smooth = array if sigma == 0.0 else cv2.GaussianBlur(
            array,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        gx = cv2.Sobel(
            smooth, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101
        )
        gy = cv2.Sobel(
            smooth, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101
        )
        result[f"sigma_{sigma:.1f}"] = np.square(gx) + np.square(gy)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise SpatialDeficitError("gradient energy is non-finite")
    return result


def partition(energy: Any, mask: Any) -> dict[str, float]:
    array = np.asarray(energy, dtype=np.float64)
    region = np.asarray(mask)
    if array.ndim != 2 or region.dtype != bool or region.shape != array.shape:
        raise SpatialDeficitError("energy/mask shapes are invalid")
    if not np.isfinite(array).all():
        raise SpatialDeficitError("energy is non-finite")
    face = float(array[region].sum())
    background = float(array[~region].sum())
    total = face + background
    direct = float(array.sum())
    if abs(total - direct) > 1e-12 * max(1.0, abs(direct)):
        raise SpatialDeficitError("regional sums do not reconstruct total energy")
    return {"face": face, "background": background, "total": total}


def pair_partition(native: Any, candidate: Any, mask: Any) -> dict[str, Any]:
    native_parts = partition(native, mask)
    candidate_parts = partition(candidate, mask)
    face = native_parts["face"] - candidate_parts["face"]
    background = native_parts["background"] - candidate_parts["background"]
    total = face + background
    return {
        "native": native_parts,
        "candidate": candidate_parts,
        "deficit_native_minus_candidate": {
            "face": face,
            "background": background,
            "total": total,
        },
    }


def background_share(face: Any, background: Any) -> float:
    face_values = np.asarray(face, dtype=np.float64)
    background_values = np.asarray(background, dtype=np.float64)
    if (
        face_values.ndim != 1
        or background_values.shape != face_values.shape
        or face_values.size == 0
        or not np.isfinite(face_values).all()
        or not np.isfinite(background_values).all()
    ):
        raise SpatialDeficitError("regional deficit arrays are invalid")
    face_positive = float(np.maximum(face_values, 0.0).sum())
    background_positive = float(np.maximum(background_values, 0.0).sum())
    denominator = face_positive + background_positive
    if denominator <= 0.0:
        raise SpatialDeficitError("positive regional deficit is empty")
    return background_positive / denominator


def bootstrap_share(face: Any, background: Any, indices: Any) -> dict[str, Any]:
    face_values = np.asarray(face, dtype=np.float64)
    background_values = np.asarray(background, dtype=np.float64)
    samples = np.asarray(indices)
    if (
        samples.ndim != 2
        or samples.shape[1] != face_values.size
        or not np.issubdtype(samples.dtype, np.integer)
        or samples.size == 0
        or int(samples.min()) < 0
        or int(samples.max()) >= face_values.size
    ):
        raise SpatialDeficitError("bootstrap indices are invalid")
    positive_face = np.maximum(face_values[samples], 0.0).sum(axis=1)
    positive_background = np.maximum(background_values[samples], 0.0).sum(axis=1)
    denominator = positive_face + positive_background
    if bool((denominator <= 0.0).any()):
        raise SpatialDeficitError("bootstrap sample has no positive regional deficit")
    shares = positive_background / denominator
    interval = np.quantile(shares, BOOTSTRAP_QUANTILES)
    return {
        "point": background_share(face_values, background_values),
        "ci95": [float(interval[0]), float(interval[1])],
    }


def rankdata(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or array.size == 0:
        raise SpatialDeficitError("rank input is invalid")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pearson(left: Any, right: Any) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise SpatialDeficitError("correlation arrays are invalid")
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(np.square(x).sum() * np.square(y).sum()))
    if denominator <= 0.0:
        raise SpatialDeficitError("correlation is undefined")
    return float((x * y).sum()) / denominator


def quality_rows(path: Path) -> tuple[list[str], dict[str, float]]:
    payload = mapping(read_json(path), "quality")
    rows = mapping(payload.get("per_sample_metrics"), "quality rows").get("rows")
    if not isinstance(rows, list) or len(rows) != 32:
        raise SpatialDeficitError("quality evidence must contain 32 rows")
    order, result = [], {}
    for raw in rows:
        row = mapping(raw, "quality row")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in result:
            raise SpatialDeficitError("quality sample_id is invalid or duplicate")
        order.append(sample_id)
        result[sample_id] = finite(row.get("sharpness"), "quality sharpness")
    return order, result


def quality_path(dataset: str, arm: str) -> Path:
    folder = "regular32" if dataset == "regular32" else "sharpness_tail32"
    label = "native" if arm == "native" else (
        f"{arm}_regular32" if dataset == "regular32" else f"{arm}_tail32"
    )
    return QUALITY_ROOT / folder / "quality" / label / "quality.json"


def load_rows(repo_root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped = {dataset: {arm: [] for arm in ARMS} for dataset in DATASETS}
    seen = set()
    for raw in read_jsonl(repo_root / IDENTITY_ROWS):
        row = mapping(raw, "identity row")
        dataset, arm, sample_id = row.get("dataset_id"), row.get("arm"), row.get("sample_id")
        if dataset not in DATASETS or arm not in ARMS or not isinstance(sample_id, str):
            raise SpatialDeficitError("identity row has invalid dataset/arm/sample_id")
        key = (dataset, arm, sample_id)
        if key in seen:
            raise SpatialDeficitError("duplicate identity row")
        seen.add(key)
        ordinal = row.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise SpatialDeficitError("identity ordinal must be an integer")
        native_path = Path(str(row.get("native_path"))).resolve()
        candidate_path = Path(str(row.get("candidate_path"))).resolve()
        if not native_path.is_file() or not candidate_path.is_file():
            raise FileNotFoundError(f"missing image for {sample_id}")
        grouped[dataset][arm].append(
            {
                "sample_id": sample_id,
                "ordinal": ordinal,
                "native_path": native_path,
                "candidate_path": candidate_path,
                "native_bbox": exact_one_bbox(row.get("native_detection"), "native detection"),
                "candidate_bbox": exact_one_bbox(
                    row.get("candidate_detection"), "candidate detection"
                ),
            }
        )
    for dataset in DATASETS:
        orders = []
        for arm in ARMS:
            grouped[dataset][arm].sort(key=lambda row: row["ordinal"])
            rows = grouped[dataset][arm]
            if len(rows) != 32 or [row["ordinal"] for row in rows] != list(range(32)):
                raise SpatialDeficitError(f"{dataset}/{arm} rows are incomplete")
            orders.append([row["sample_id"] for row in rows])
        if orders[0] != orders[1]:
            raise SpatialDeficitError(f"{dataset} horizon orders differ")
    return grouped


def analyze_pair(
    row: Mapping[str, Any],
    dataset: str,
    arm: str,
    official_native: float,
    official_candidate: float,
) -> dict[str, Any]:
    import cv2

    native = cv2.imread(str(row["native_path"]), cv2.IMREAD_COLOR)
    candidate = cv2.imread(str(row["candidate_path"]), cv2.IMREAD_COLOR)
    native_gray = cv2.imread(str(row["native_path"]), cv2.IMREAD_GRAYSCALE)
    candidate_gray = cv2.imread(str(row["candidate_path"]), cv2.IMREAD_GRAYSCALE)
    if (
        native is None
        or candidate is None
        or native_gray is None
        or candidate_gray is None
        or native.shape != candidate.shape
        or native_gray.shape != candidate_gray.shape
        or native.shape[:2] != native_gray.shape
    ):
        raise SpatialDeficitError("native/candidate decoding or shape mismatch")
    mask = bbox_union_mask(native.shape, row["native_bbox"], row["candidate_bbox"])
    laplacian = pair_partition(
        centered_laplacian_energy(native_gray),
        centered_laplacian_energy(candidate_gray),
        mask,
    )
    pixels = int(mask.size)
    measured_native = laplacian["native"]["total"] / pixels
    measured_candidate = laplacian["candidate"]["total"] / pixels
    tolerance = 1e-9 * max(1.0, official_native, official_candidate)
    error = max(
        abs(measured_native - official_native),
        abs(measured_candidate - official_candidate),
    )
    if error > tolerance:
        raise SpatialDeficitError("energy does not reproduce official sharpness")
    native_gradient = gradient_energy_maps(native_gray)
    candidate_gradient = gradient_energy_maps(candidate_gray)
    gradient_scales = {
        key: pair_partition(native_gradient[key], candidate_gradient[key], mask)
        for key in native_gradient
    }
    gradient = pair_partition(
        np.mean(np.stack(list(native_gradient.values())), axis=0),
        np.mean(np.stack(list(candidate_gradient.values())), axis=0),
        mask,
    )
    return {
        "dataset_id": dataset,
        "arm": arm,
        "ordinal": row["ordinal"],
        "sample_id": row["sample_id"],
        "image_shape": list(native.shape),
        "paths": {"native": str(row["native_path"]), "candidate": str(row["candidate_path"])},
        "mask": {
            "definition": "native_candidate_bbox_union_by_pixel_center",
            "native_bbox": row["native_bbox"],
            "candidate_bbox": row["candidate_bbox"],
            "face_pixels": int(mask.sum()),
            "background_pixels": int((~mask).sum()),
            "face_fraction": float(mask.mean()),
            "expansion_pixels": 0,
            "morphology": None,
        },
        "official_sharpness": {
            "native": official_native,
            "candidate": official_candidate,
            "deficit_native_minus_candidate": official_native - official_candidate,
            "max_abs_reproduction_error": error,
        },
        "metrics": {
            LAPLACIAN: laplacian,
            GRADIENT: {"aggregate_equal_scale_mean": gradient, "scales": gradient_scales},
        },
    }


def metric_arrays(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, np.ndarray]:
    result = {"face": [], "background": [], "total": []}
    for row in rows:
        data = mapping(mapping(row.get("metrics"), "metrics").get(metric), metric)
        if metric == GRADIENT:
            data = mapping(data.get("aggregate_equal_scale_mean"), "gradient aggregate")
        deficit = mapping(data.get("deficit_native_minus_candidate"), "deficit")
        pixels = int(row["image_shape"][0]) * int(row["image_shape"][1])
        for region in result:
            result[region].append(finite(deficit.get(region), f"{metric}/{region}") / pixels)
    return {key: np.asarray(value) for key, value in result.items()}


def summarize_metric(
    rows: Sequence[Mapping[str, Any]], metric: str, indices: np.ndarray
) -> dict[str, Any]:
    values = metric_arrays(rows, metric)
    residual = np.abs(values["face"] + values["background"] - values["total"])
    if float(residual.max()) > 1e-9 * max(1.0, float(np.abs(values["total"]).max())):
        raise SpatialDeficitError("regional deficits do not reconstruct total")
    return {
        "mean_deficit_per_pixel": {
            key: float(value.mean()) for key, value in values.items()
        },
        "positive_background_deficit_share": bootstrap_share(
            values["face"], values["background"], indices
        ),
        "positive_full_deficit_fraction": float((values["total"] > 0.0).mean()),
        "relation_to_full_image_deficit": {
            "face_pearson": pearson(values["face"], values["total"]),
            "face_spearman": pearson(rankdata(values["face"]), rankdata(values["total"])),
            "background_pearson": pearson(values["background"], values["total"]),
            "background_spearman": pearson(
                rankdata(values["background"]), rankdata(values["total"])
            ),
            "regional_sum_pearson": pearson(
                values["face"] + values["background"], values["total"]
            ),
            "max_abs_reconstruction_error": float(residual.max()),
        },
    }


def horizon_consistency(rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    if [row["sample_id"] for row in rows_by_arm["u12"]] != [
        row["sample_id"] for row in rows_by_arm["u16"]
    ]:
        raise SpatialDeficitError("horizon rows are not paired")
    result = {}
    for metric in (LAPLACIAN, GRADIENT):
        left, right = metric_arrays(rows_by_arm["u12"], metric), metric_arrays(
            rows_by_arm["u16"], metric
        )
        result[metric] = {
            region: {
                "pearson": pearson(left[region], right[region]),
                "spearman": pearson(rankdata(left[region]), rankdata(right[region])),
            }
            for region in ("face", "background", "total")
        }
    return result


def conclusion(summary: Mapping[str, Any]) -> str:
    decision = mapping(summary.get("decision"), "decision")
    tail = mapping(mapping(summary.get("datasets"), "datasets").get("sharpness_tail32"), "tail")
    lines = ["# R14 spatial sharpness-deficit diagnostic", ""]
    lines.append(
        "The predeclared evidence "
        + ("supports" if decision["architecture_supported"] else "rejects")
        + " background-preserving face-only generation as the next mainline."
    )
    lines.extend(["", "Tail32 centered-Laplacian positive background-deficit shares:"])
    for arm in ARMS:
        share = tail["arms"][arm][LAPLACIAN]["positive_background_deficit_share"]
        lines.append(
            f"- {arm}: {share['point']:.6f} "
            f"(95% bootstrap [{share['ci95'][0]:.6f}, {share['ci95'][1]:.6f}]); "
            f"support={decision['arms'][arm]['support']}."
        )
    lines.extend(
        [
            "",
            f"Decision: {decision['classification']}. Both point estimates and both "
            "lower bounds must be strictly above 0.5.",
            "",
            decision["next_direction"],
            "",
            "Face exact-one detections and face-region deficits remain separately reported. "
            "This existing-image diagnostic is not a gate survivor, privacy proof, Full result, "
            "or formal winner.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(output_dir.parent)
    source_rows = load_rows(repo_root)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = {
        dataset: rng.integers(0, 32, size=(BOOTSTRAP_ITERATIONS, 32), dtype=np.int64)
        for dataset in DATASETS
    }
    all_rows, datasets = [], {}
    for dataset in DATASETS:
        orders, sharpness = {}, {}
        for label in ("native", *ARMS):
            orders[label], sharpness[label] = quality_rows(
                repo_root / quality_path(dataset, label)
            )
        expected = [row["sample_id"] for row in source_rows[dataset]["u12"]]
        if any(orders[label] != expected for label in orders):
            raise SpatialDeficitError(f"{dataset} quality order mismatch")
        analyzed = {}
        arm_summaries = {}
        for arm in ARMS:
            rows = [
                analyze_pair(
                    row,
                    dataset,
                    arm,
                    sharpness["native"][row["sample_id"]],
                    sharpness[arm][row["sample_id"]],
                )
                for row in source_rows[dataset][arm]
            ]
            analyzed[arm] = rows
            all_rows.extend(rows)
            arm_summaries[arm] = {
                "sample_count": 32,
                "exact_one_native_candidate_count": 32,
                "mean_face_mask_fraction": statistics.fmean(
                    row["mask"]["face_fraction"] for row in rows
                ),
                "official_sharpness_max_abs_reproduction_error": max(
                    row["official_sharpness"]["max_abs_reproduction_error"] for row in rows
                ),
                LAPLACIAN: summarize_metric(rows, LAPLACIAN, indices[dataset]),
                GRADIENT: summarize_metric(rows, GRADIENT, indices[dataset]),
            }
        datasets[dataset] = {
            "sample_count": 32,
            "arms": arm_summaries,
            "u12_u16_consistency": horizon_consistency(analyzed),
        }
    decision_arms = {}
    for arm in ARMS:
        share = datasets["sharpness_tail32"]["arms"][arm][LAPLACIAN][
            "positive_background_deficit_share"
        ]
        decision_arms[arm] = {
            "point": share["point"],
            "bootstrap_lower": share["ci95"][0],
            "point_pass": share["point"] > BACKGROUND_SHARE_THRESHOLD,
            "bootstrap_lower_pass": share["ci95"][0] > BACKGROUND_SHARE_THRESHOLD,
        }
        decision_arms[arm]["support"] = (
            decision_arms[arm]["point_pass"]
            and decision_arms[arm]["bootstrap_lower_pass"]
        )
    supported = all(decision_arms[arm]["support"] for arm in ARMS)
    decision = {
        "predeclared_rule": (
            "tail32 positive background centered-Laplacian deficit share point and "
            "paired-bootstrap lower bound must both be strictly greater than 0.5 "
            "at u12 and u16"
        ),
        "threshold": BACKGROUND_SHARE_THRESHOLD,
        "arms": decision_arms,
        "architecture_supported": supported,
        "classification": (
            "background_preserving_face_only_generation_supported"
            if supported
            else "background_preserving_face_only_generation_rejected"
        ),
        "next_direction": (
            "Run one bounded face-only architecture probe that preserves native background "
            "pixels exactly and keeps face ROI/privacy gates explicit."
            if supported
            else "Do not spend the next cycle on native-background compositing. Focus on "
            "face-region decoder/sampler detail while retaining full-image quality and privacy gates."
        ),
    }
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r14_spatial_deficit_summary_v1",
        "boundary": {
            "existing_images_only": True,
            "new_generation": False,
            "new_training": False,
            "formal_gate": False,
            "privacy_proof": False,
        },
        "mask_contract": {
            "roles": ["native", "candidate"],
            "coordinate_rule": "original_coordinate_pixel_center_inside_half_open_bbox",
            "combine": "union",
            "expansion_pixels": 0,
            "morphology": None,
        },
        "metrics": {
            LAPLACIAN: (
                "(Laplacian - full-image mean Laplacian)^2; total/pixels exactly "
                "reproduces official grayscale Laplacian variance"
            ),
            GRADIENT: {
                "operator": "Sobel squared magnitude after Gaussian smoothing",
                "gaussian_sigmas": list(GRADIENT_SIGMAS),
                "aggregation": "equal_scale_mean",
            },
        },
        "positive_deficit_share_definition": (
            "sum(max(background native-candidate regional energy,0)) divided by "
            "positive face plus positive background regional deficit"
        ),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "iterations": BOOTSTRAP_ITERATIONS,
            "quantiles": list(BOOTSTRAP_QUANTILES),
            "unit": "paired sample_id",
        },
        "inputs": {"r12_identity_rows": str((repo_root / IDENTITY_ROWS).resolve())},
        "datasets": datasets,
        "decision": decision,
    }
    output_dir.mkdir()
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "per_sample.jsonl").open("x", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "conclusion.md").write_text(conclusion(summary), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Partition existing R12 sharpness deficits by locked face bboxes."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
