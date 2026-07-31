#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_r12_face_frequency import (  # noqa: E402
    PRIMARY_FREQUENCY_METRIC,
    FrequencyDiagnosticError,
    align_face,
    build_analyzer,
    image_metrics,
    serialize_exact_one_face,
)


ARMS = ("u12", "u16")
BOOTSTRAP_SEED = 91637
BOOTSTRAP_ITERATIONS = 10_000
TOP_K = 8


class LpipsConflictError(FrequencyDiagnosticError):
    """Raised when LPIPS-conflict evidence is absent or invalid."""


def _reject_json_constant(value: str) -> None:
    raise LpipsConflictError(f"JSON contains non-finite constant: {value}")


def read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise LpipsConflictError(f"invalid {label}: {path}") from exc


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
                raise LpipsConflictError(
                    f"invalid {label} row {line_number}: {path}"
                ) from exc
            if not isinstance(row, Mapping):
                raise LpipsConflictError(f"{label} row {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise LpipsConflictError(f"{label} is empty: {path}")
    return rows


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LpipsConflictError(f"{label} must be an object")
    return value


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise LpipsConflictError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise LpipsConflictError(f"{label} must be finite")
    return normalized


def resolve_repo_file(repo_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise LpipsConflictError(f"{label} must be a non-empty path")
    raw = Path(value)
    path = (raw if raw.is_absolute() else repo_root / raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_map(
    rows: Sequence[Mapping[str, Any]], label: str
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    ordered: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise LpipsConflictError(f"{label} row {index} has invalid sample_id")
        if sample_id in by_id:
            raise LpipsConflictError(f"{label} contains duplicate sample_id: {sample_id}")
        ordered.append(sample_id)
        by_id[sample_id] = row
    return ordered, by_id


def require_order(
    expected: Sequence[str], observed: Sequence[str], label: str
) -> None:
    if list(observed) != list(expected):
        raise LpipsConflictError(f"{label} sample order does not match the manifest")


def quality_rows(path: Path, expected_ids: Sequence[str], label: str) -> dict[str, Mapping[str, Any]]:
    payload = require_mapping(read_json(path, label), label)
    per_sample = require_mapping(payload.get("per_sample_metrics"), f"{label} per-sample metrics")
    rows = per_sample.get("rows")
    if not isinstance(rows, list):
        raise LpipsConflictError(f"{label} per-sample rows must be a list")
    ordered, by_id = ordered_map(rows, label)
    require_order(expected_ids, ordered, label)
    for sample_id, row in by_id.items():
        finite_float(row.get("sharpness"), f"{label} sharpness for {sample_id}")
        finite_float(row.get("niqe"), f"{label} NIQE for {sample_id}")
    return by_id


def arcface_rows(path: Path, expected_ids: Sequence[str], label: str) -> dict[str, Mapping[str, Any]]:
    payload = require_mapping(read_json(path, label), label)
    rows = payload.get("result")
    if not isinstance(rows, list):
        raise LpipsConflictError(f"{label} result must be a list")
    ordered, by_id = ordered_map(rows, label)
    require_order(expected_ids, ordered, label)
    for sample_id, row in by_id.items():
        counts = [
            row.get("source_face_count"),
            row.get("native_face_count"),
            row.get("candidate_face_count"),
        ]
        if counts != [1, 1, 1]:
            raise LpipsConflictError(
                f"{label} exact-one failed for {sample_id}: counts={counts}"
            )
        finite_float(
            row.get("source_candidate_cosine"),
            f"{label} source-candidate cosine for {sample_id}",
        )
        finite_float(
            row.get("source_native_cosine"),
            f"{label} source-native cosine for {sample_id}",
        )
    return by_id


def formal_native_rows(
    path: Path, repo_root: Path
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    rows = read_jsonl(path, "formal native bindings")
    ordered, by_id = ordered_map(rows, "formal native bindings")
    for sample_id, row in by_id.items():
        native = resolve_repo_file(
            repo_root, row.get("formal_native"), f"formal native for {sample_id}"
        )
        digest = row.get("formal_native_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise LpipsConflictError(f"invalid formal native digest for {sample_id}")
        if sha256_file(native) != digest:
            raise LpipsConflictError(f"formal native digest mismatch for {sample_id}")
    return ordered, by_id


def run_rows(
    path: Path, expected_ids: Sequence[str], repo_root: Path, label: str
) -> dict[str, Mapping[str, Any]]:
    rows = read_jsonl(path, label)
    ordered, by_id = ordered_map(rows, label)
    require_order(expected_ids, ordered, label)
    for sample_id, row in by_id.items():
        resolve_repo_file(repo_root, row.get("generated"), f"{label} image for {sample_id}")
        for field in (
            "candidate_cosine",
            "native_cosine",
            "edev_cosine",
            "native_edev_cosine",
        ):
            finite_float(row.get(field), f"{label} {field} for {sample_id}")
    return by_id


def lpips_tensor(image: Any, device: str) -> Any:
    import cv2
    import torch

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise LpipsConflictError(f"LPIPS image has invalid shape: {array.shape}")
    if not np.isfinite(array).all():
        raise LpipsConflictError("LPIPS image contains non-finite values")
    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor


def lpips_distance(model: Any, first: Any, second: Any, device: str) -> float:
    import torch

    if tuple(np.asarray(first).shape) != tuple(np.asarray(second).shape):
        raise LpipsConflictError("LPIPS pair images must have identical shapes")
    with torch.no_grad():
        value = model(lpips_tensor(first, device), lpips_tensor(second, device))
    array = value.detach().cpu().numpy().reshape(-1)
    if array.size != 1:
        raise LpipsConflictError("LPIPS model must return one scalar per pair")
    distance = finite_float(float(array[0]), "LPIPS distance")
    if distance < 0.0:
        raise LpipsConflictError("LPIPS distance must be non-negative")
    return distance


def positive_ratio(numerator: Any, denominator: Any, label: str) -> float:
    numerator_value = finite_float(numerator, f"{label} numerator")
    denominator_value = finite_float(denominator, f"{label} denominator")
    if numerator_value < 0.0 or denominator_value <= 0.0:
        raise LpipsConflictError(
            f"{label} requires a non-negative numerator and positive denominator"
        )
    return numerator_value / denominator_value


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    if len(x) != len(y) or len(x) < 3:
        raise LpipsConflictError("Spearman inputs must have equal length >= 3")
    result = float(spearmanr(x, y).statistic)
    if not math.isfinite(result):
        raise LpipsConflictError("Spearman correlation is non-finite")
    return result


def top_indices(values: Sequence[float], top_k: int) -> list[int]:
    if top_k <= 0 or top_k >= len(values):
        raise LpipsConflictError("top_k must be between zero and sample count")
    return sorted(range(len(values)), key=lambda index: (-float(values[index]), index))[:top_k]


def association(
    predictor: Sequence[float],
    outcome: Sequence[float],
    bootstrap_indices: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise LpipsConflictError("association arrays must be equal one-dimensional vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise LpipsConflictError("association arrays contain non-finite values")
    rho = spearman(x, y)
    top_x = set(top_indices(x, top_k))
    top_y = set(top_indices(y, top_k))
    overlap = len(top_x & top_y)
    expected_overlap = top_k * top_k / len(x)
    enrichment = overlap / expected_overlap
    top_predictor_outcome_gap = float(
        np.mean(y[list(top_x)])
        - np.mean(y[[i for i in range(len(y)) if i not in top_x]])
    )
    top_failure_predictor_gap = float(
        np.mean(x[list(top_y)])
        - np.mean(x[[i for i in range(len(x)) if i not in top_y]])
    )

    bootstrap_rho: list[float] = []
    bootstrap_predictor_outcome_gap: list[float] = []
    bootstrap_failure_predictor_gap: list[float] = []
    dropped = 0
    for indices in bootstrap_indices:
        xb = x[indices]
        yb = y[indices]
        try:
            bootstrap_rho.append(spearman(xb, yb))
        except LpipsConflictError:
            dropped += 1
            continue
        selected_predictor = top_indices(xb, min(top_k, len(xb) - 1))
        selected_predictor_set = set(selected_predictor)
        predictor_remainder = [
            i for i in range(len(yb)) if i not in selected_predictor_set
        ]
        bootstrap_predictor_outcome_gap.append(
            float(np.mean(yb[selected_predictor]) - np.mean(yb[predictor_remainder]))
        )
        selected_failure = top_indices(yb, min(top_k, len(yb) - 1))
        selected_failure_set = set(selected_failure)
        failure_remainder = [
            i for i in range(len(xb)) if i not in selected_failure_set
        ]
        bootstrap_failure_predictor_gap.append(
            float(np.mean(xb[selected_failure]) - np.mean(xb[failure_remainder]))
        )
    minimum = int(math.ceil(0.99 * len(bootstrap_indices)))
    if (
        len(bootstrap_rho) < minimum
        or len(bootstrap_predictor_outcome_gap) < minimum
        or len(bootstrap_failure_predictor_gap) < minimum
    ):
        raise LpipsConflictError(
            f"too many degenerate bootstrap replicates: kept={len(bootstrap_rho)}"
        )
    rho_ci = np.percentile(bootstrap_rho, [2.5, 97.5])
    predictor_outcome_gap_ci = np.percentile(
        bootstrap_predictor_outcome_gap, [2.5, 97.5]
    )
    failure_predictor_gap_ci = np.percentile(
        bootstrap_failure_predictor_gap, [2.5, 97.5]
    )
    return {
        "spearman": rho,
        "spearman_ci95": [float(rho_ci[0]), float(rho_ci[1])],
        "top_k": top_k,
        "top_outlier_overlap": overlap,
        "random_expected_overlap": expected_overlap,
        "top_outlier_enrichment_factor": enrichment,
        "top_predictor_outcome_mean_gap": top_predictor_outcome_gap,
        "top_predictor_outcome_mean_gap_ci95": [
            float(predictor_outcome_gap_ci[0]),
            float(predictor_outcome_gap_ci[1]),
        ],
        "top_failure_group_predictor_mean_gap": top_failure_predictor_gap,
        "top_failure_group_predictor_mean_gap_ci95": [
            float(failure_predictor_gap_ci[0]),
            float(failure_predictor_gap_ci[1]),
        ],
        "bootstrap_kept": len(bootstrap_rho),
        "bootstrap_dropped_degenerate": dropped,
    }


def paired_mean_delta(
    first: Sequence[float], second: Sequence[float], bootstrap_indices: np.ndarray
) -> dict[str, Any]:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise LpipsConflictError("paired arrays are invalid")
    delta = b - a
    means = np.mean(delta[bootstrap_indices], axis=1)
    ci = np.percentile(means, [2.5, 97.5])
    return {
        "direction": "u16_minus_u12",
        "mean_delta": float(np.mean(delta)),
        "ci95": [float(ci[0]), float(ci[1])],
    }


def evaluate_target_license(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(cells) != set(ARMS):
        raise LpipsConflictError("target license requires u12 and u16 cells")
    directional: dict[str, bool] = {}
    statistical: dict[str, bool] = {}
    for arm in ARMS:
        cell = require_mapping(cells[arm], f"{arm} association")
        rho = finite_float(cell.get("spearman"), f"{arm} Spearman")
        failure_group_gap = finite_float(
            cell.get("top_failure_group_predictor_mean_gap"),
            f"{arm} failure-group predictor gap",
        )
        rho_ci = cell.get("spearman_ci95")
        if not isinstance(rho_ci, list) or len(rho_ci) != 2:
            raise LpipsConflictError(f"{arm} Spearman CI is invalid")
        directional[arm] = rho > 0.0 and failure_group_gap > 0.0
        statistical[arm] = finite_float(
            rho_ci[0], f"{arm} Spearman lower CI"
        ) > 0.0
    consistent = all(directional.values())
    supported = any(statistical.values())
    return {
        "directional_by_arm": directional,
        "statistical_support_by_arm": statistical,
        "positive_spearman_and_failure_enrichment_both_updates": consistent,
        "at_least_one_update_spearman_ci_lower_gt_zero": supported,
        "licensed": consistent and supported,
    }


def evaluate_shared_license(targets: Mapping[str, Mapping[str, Any]]) -> bool:
    expected = {"regular_privacy", "tail_sharpness"}
    if set(targets) != expected:
        raise LpipsConflictError(
            "shared LPIPS license requires regular privacy and tail sharpness targets"
        )
    licensed: list[bool] = []
    for target_id in sorted(expected):
        target = require_mapping(targets[target_id], f"{target_id} license")
        value = target.get("licensed")
        if not isinstance(value, bool):
            raise LpipsConflictError(f"{target_id} license must be boolean")
        licensed.append(value)
    return all(licensed)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_lpips_model(device: str) -> Any:
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("lpips is required for the perceptual diagnostic") from exc
    model = lpips.LPIPS(net="alex", version="0.1", verbose=False)
    return model.eval().to(device)


def build_conclusion(summary: Mapping[str, Any]) -> str:
    decision = require_mapping(summary.get("decision"), "summary decision")
    targets = require_mapping(decision.get("targets"), "license targets")
    lines = [
        "# R12 native-anchor LPIPS conflict diagnostic",
        "",
        (
            "Native-anchor LPIPS conflict projection is licensed by the predeclared evidence rule."
            if decision.get("lpips_projection_licensed") is True
            else "Native-anchor LPIPS conflict projection is rejected as ungrounded by the predeclared evidence rule."
        ),
        "",
    ]
    for target_id in ("regular_privacy", "tail_sharpness"):
        target = require_mapping(targets.get(target_id), target_id)
        lines.append(f"- `{target_id}` licensed: `{str(target['licensed']).lower()}`.")
        for arm in ARMS:
            association_row = summary["associations"][target_id][arm]
            lines.append(
                f"  - {arm}: Spearman `{association_row['spearman']:.6f}` "
                f"(95% bootstrap CI `{association_row['spearman_ci95'][0]:.6f}, "
                f"{association_row['spearman_ci95'][1]:.6f}`), top-outlier enrichment "
                f"`{association_row['top_outlier_enrichment_factor']:.3f}x`, top-8 failure-group "
                f"ROI LPIPS mean gap `{association_row['top_failure_group_predictor_mean_gap']:.6f}`."
            )
    lines.extend(
        [
            "",
            "LPIPS and quality distances use candidate and exact formal native pixels only. "
            "Source pixels are never read by the quality-distance path; source information enters "
            "only through archived ArcFace cosine scalars.",
            "",
            "This diagnostic does not promote an arm and is not privacy proof, a Full gate, or a formal winner.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(request_path: Path, output_dir: Path, device: str) -> dict[str, Any]:
    import cv2

    repo_root = Path(__file__).resolve().parents[1]
    request_path = request_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    request = require_mapping(read_json(request_path, "diagnostic request"), "diagnostic request")
    if request.get("schema_version") != 1 or request.get("contract_type") != "safa_r12_lpips_conflict_request_v1":
        raise LpipsConflictError("diagnostic request identity mismatch")
    statistics_contract = require_mapping(
        request.get("statistics"), "statistics contract"
    )
    if statistics_contract != {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "top_k": TOP_K,
        "paired_resampling_unit": "sample_id",
    }:
        raise LpipsConflictError("statistics contract differs from the predeclared rule")
    license_contract = require_mapping(request.get("license_rule"), "license rule")
    expected_license = {
        "cell_direction": "spearman_gt_0_and_top8_failure_group_roi_lpips_gap_gt_0",
        "consistency": "cell_direction_true_for_u12_and_u16_per_target",
        "statistical_support": "at_least_one_update_spearman_ci95_lower_gt_0_per_target",
        "overall": "regular_privacy_and_tail_sharpness_targets_licensed",
    }
    if dict(license_contract) != expected_license:
        raise LpipsConflictError("license rule differs from the predeclared rule")
    arcface_request = resolve_repo_file(
        repo_root, request.get("arcface_request"), "locked ArcFace request"
    )
    datasets = request.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise LpipsConflictError("request must contain regular32 and sharpness_tail32")

    prepared: list[dict[str, Any]] = []
    seen_datasets: set[str] = set()
    for raw in datasets:
        dataset = require_mapping(raw, "dataset")
        dataset_id = dataset.get("dataset_id")
        if dataset_id not in {"regular32", "sharpness_tail32"} or dataset_id in seen_datasets:
            raise LpipsConflictError(f"invalid or duplicate dataset ID: {dataset_id}")
        seen_datasets.add(str(dataset_id))
        native_path = resolve_repo_file(
            repo_root, dataset.get("formal_native_bindings"), f"{dataset_id} native bindings"
        )
        sample_ids, natives = formal_native_rows(native_path, repo_root)
        if len(sample_ids) != 32:
            raise LpipsConflictError(f"{dataset_id} must contain exactly 32 samples")
        native_quality = quality_rows(
            resolve_repo_file(repo_root, dataset.get("native_quality"), f"{dataset_id} native quality"),
            sample_ids,
            f"{dataset_id} native quality",
        )
        raw_arms = require_mapping(dataset.get("arms"), f"{dataset_id} arms")
        if set(raw_arms) != set(ARMS):
            raise LpipsConflictError(f"{dataset_id} must bind u12 and u16")
        arms: dict[str, Any] = {}
        for arm in ARMS:
            arm_config = require_mapping(raw_arms[arm], f"{dataset_id} {arm}")
            run_path = resolve_repo_file(
                repo_root, arm_config.get("run_rows"), f"{dataset_id} {arm} run rows"
            )
            quality_path = resolve_repo_file(
                repo_root, arm_config.get("quality"), f"{dataset_id} {arm} quality"
            )
            arcface_path = resolve_repo_file(
                repo_root, arm_config.get("arcface"), f"{dataset_id} {arm} ArcFace"
            )
            arms[arm] = {
                "run": run_rows(run_path, sample_ids, repo_root, f"{dataset_id} {arm} run"),
                "quality": quality_rows(
                    quality_path, sample_ids, f"{dataset_id} {arm} quality"
                ),
                "arcface": arcface_rows(
                    arcface_path, sample_ids, f"{dataset_id} {arm} ArcFace"
                ),
            }
        prepared.append(
            {
                "dataset_id": dataset_id,
                "sample_ids": sample_ids,
                "natives": natives,
                "native_quality": native_quality,
                "arms": arms,
            }
        )
    if seen_datasets != {"regular32", "sharpness_tail32"}:
        raise LpipsConflictError("both required datasets must be present")

    analyzer = build_analyzer(arcface_request, device)
    lpips_model = build_lpips_model(device)
    observation_cache: dict[Path, dict[str, Any]] = {}

    def observe(path: Path) -> dict[str, Any]:
        if path not in observation_cache:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise LpipsConflictError(f"failed to decode image: {path}")
            detection = serialize_exact_one_face(analyzer.get(image), image.shape)
            roi = align_face(image, detection)
            observation_cache[path] = {
                "path": str(path),
                "image": image,
                "roi_image": roi,
                "arcface_alignment": detection,
                "roi_metrics": image_metrics(roi),
                "full_metrics": image_metrics(image),
            }
        return observation_cache[path]

    all_rows: list[dict[str, Any]] = []
    rows_by_dataset_arm: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dataset in prepared:
        dataset_id = str(dataset["dataset_id"])
        rows_by_dataset_arm[dataset_id] = {arm: [] for arm in ARMS}
        for ordinal, sample_id in enumerate(dataset["sample_ids"]):
            native_binding = dataset["natives"][sample_id]
            native_path = resolve_repo_file(
                repo_root,
                native_binding.get("formal_native"),
                f"formal native for {sample_id}",
            )
            native_observation = observe(native_path)
            native_sharpness = finite_float(
                dataset["native_quality"][sample_id].get("sharpness"),
                f"native sharpness for {sample_id}",
            )
            if native_sharpness <= 0.0:
                raise LpipsConflictError(f"native sharpness must be positive: {sample_id}")
            for arm in ARMS:
                evidence = dataset["arms"][arm]
                run_row = evidence["run"][sample_id]
                candidate_path = resolve_repo_file(
                    repo_root,
                    run_row.get("generated"),
                    f"{dataset_id} {arm} candidate for {sample_id}",
                )
                candidate_observation = observe(candidate_path)
                candidate_sharpness = finite_float(
                    evidence["quality"][sample_id].get("sharpness"),
                    f"{dataset_id} {arm} sharpness for {sample_id}",
                )
                arcface = evidence["arcface"][sample_id]
                roi_lpips = lpips_distance(
                    lpips_model,
                    native_observation["roi_image"],
                    candidate_observation["roi_image"],
                    device,
                )
                full_lpips = lpips_distance(
                    lpips_model,
                    native_observation["image"],
                    candidate_observation["image"],
                    device,
                )
                privacy_delta = finite_float(
                    arcface.get("source_candidate_cosine"), "source-candidate cosine"
                ) - finite_float(arcface.get("source_native_cosine"), "source-native cosine")
                sharpness_ratio = positive_ratio(
                    candidate_sharpness, native_sharpness, "sharpness ratio"
                )
                roi_fft_native = native_observation["roi_metrics"]["fft_energy"][PRIMARY_FREQUENCY_METRIC]
                roi_fft_candidate = candidate_observation["roi_metrics"]["fft_energy"][PRIMARY_FREQUENCY_METRIC]
                full_fft_native = native_observation["full_metrics"]["fft_energy"][PRIMARY_FREQUENCY_METRIC]
                full_fft_candidate = candidate_observation["full_metrics"]["fft_energy"][PRIMARY_FREQUENCY_METRIC]
                roi_laplacian_retention = {
                    scale: positive_ratio(
                        candidate_observation["roi_metrics"]["laplacian_variance"][scale],
                        native_observation["roi_metrics"]["laplacian_variance"][scale],
                        f"ROI Laplacian retention at {scale}",
                    )
                    for scale in native_observation["roi_metrics"]["laplacian_variance"]
                }
                full_laplacian_retention = {
                    scale: positive_ratio(
                        candidate_observation["full_metrics"]["laplacian_variance"][scale],
                        native_observation["full_metrics"]["laplacian_variance"][scale],
                        f"full-image Laplacian retention at {scale}",
                    )
                    for scale in native_observation["full_metrics"]["laplacian_variance"]
                }
                row = {
                    "dataset_id": dataset_id,
                    "arm": arm,
                    "ordinal": ordinal,
                    "sample_id": sample_id,
                    "native_path": str(native_path),
                    "candidate_path": str(candidate_path),
                    "roi_lpips_native_candidate": roi_lpips,
                    "full_lpips_native_candidate": full_lpips,
                    "privacy_delta_source_candidate_minus_source_native": privacy_delta,
                    "candidate_e0": finite_float(run_row.get("candidate_cosine"), "candidate e0"),
                    "native_e0": finite_float(run_row.get("native_cosine"), "native e0"),
                    "delta_e0": finite_float(run_row.get("candidate_cosine"), "candidate e0")
                    - finite_float(run_row.get("native_cosine"), "native e0"),
                    "candidate_edev": finite_float(run_row.get("edev_cosine"), "candidate edev"),
                    "native_edev": finite_float(run_row.get("native_edev_cosine"), "native edev"),
                    "delta_edev": finite_float(run_row.get("edev_cosine"), "candidate edev")
                    - finite_float(run_row.get("native_edev_cosine"), "native edev"),
                    "native_sharpness": native_sharpness,
                    "candidate_sharpness": candidate_sharpness,
                    "sharpness_ratio": sharpness_ratio,
                    "sharpness_loss": 1.0 - sharpness_ratio,
                    "roi_fft_high_retention": positive_ratio(
                        roi_fft_candidate, roi_fft_native, "ROI FFT retention"
                    ),
                    "full_fft_high_retention": positive_ratio(
                        full_fft_candidate, full_fft_native, "full-image FFT retention"
                    ),
                    "roi_laplacian_retention": roi_laplacian_retention,
                    "full_laplacian_retention": full_laplacian_retention,
                    "native_arcface_alignment": native_observation["arcface_alignment"],
                    "candidate_arcface_alignment": candidate_observation["arcface_alignment"],
                }
                # Every numeric result must be finite before it enters statistics.
                def validate_numeric(value: Any, label: str) -> None:
                    if isinstance(value, Mapping):
                        for key, child in value.items():
                            validate_numeric(child, f"{label}.{key}")
                    elif isinstance(value, list):
                        for index, child in enumerate(value):
                            validate_numeric(child, f"{label}[{index}]")
                    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
                        finite_float(value, label)

                validate_numeric(row, f"{dataset_id}.{arm}.{sample_id}")
                rows_by_dataset_arm[dataset_id][arm].append(row)
                all_rows.append(row)

    bootstrap_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, 32, size=(BOOTSTRAP_ITERATIONS, 32), dtype=np.int64
    )
    associations: dict[str, Any] = {
        "regular_privacy": {},
        "tail_sharpness": {},
        "descriptive": {},
    }
    for arm in ARMS:
        regular = rows_by_dataset_arm["regular32"][arm]
        tail = rows_by_dataset_arm["sharpness_tail32"][arm]
        associations["regular_privacy"][arm] = association(
            [row["roi_lpips_native_candidate"] for row in regular],
            [row["privacy_delta_source_candidate_minus_source_native"] for row in regular],
            bootstrap_indices,
            TOP_K,
        )
        associations["tail_sharpness"][arm] = association(
            [row["roi_lpips_native_candidate"] for row in tail],
            [row["sharpness_loss"] for row in tail],
            bootstrap_indices,
            TOP_K,
        )
        associations["descriptive"][arm] = {}
        for dataset_id, rows in (("regular32", regular), ("sharpness_tail32", tail)):
            associations["descriptive"][arm][dataset_id] = {
                "full_lpips_vs_privacy": association(
                    [row["full_lpips_native_candidate"] for row in rows],
                    [row["privacy_delta_source_candidate_minus_source_native"] for row in rows],
                    bootstrap_indices,
                    TOP_K,
                ),
                "full_lpips_vs_sharpness_loss": association(
                    [row["full_lpips_native_candidate"] for row in rows],
                    [row["sharpness_loss"] for row in rows],
                    bootstrap_indices,
                    TOP_K,
                ),
                "roi_lpips_vs_delta_e0_spearman": spearman(
                    [row["roi_lpips_native_candidate"] for row in rows],
                    [row["delta_e0"] for row in rows],
                ),
                "roi_lpips_vs_delta_edev_spearman": spearman(
                    [row["roi_lpips_native_candidate"] for row in rows],
                    [row["delta_edev"] for row in rows],
                ),
            }

    target_decisions = {
        "regular_privacy": evaluate_target_license(associations["regular_privacy"]),
        "tail_sharpness": evaluate_target_license(associations["tail_sharpness"]),
    }
    licensed = evaluate_shared_license(target_decisions)

    paired_changes: dict[str, Any] = {}
    for dataset_id in ("regular32", "sharpness_tail32"):
        u12 = rows_by_dataset_arm[dataset_id]["u12"]
        u16 = rows_by_dataset_arm[dataset_id]["u16"]
        if [row["sample_id"] for row in u12] != [row["sample_id"] for row in u16]:
            raise LpipsConflictError(f"{dataset_id} u12/u16 rows are not paired")
        paired_changes[dataset_id] = {
            "roi_lpips": paired_mean_delta(
                [row["roi_lpips_native_candidate"] for row in u12],
                [row["roi_lpips_native_candidate"] for row in u16],
                bootstrap_indices,
            ),
            "privacy_delta": paired_mean_delta(
                [row["privacy_delta_source_candidate_minus_source_native"] for row in u12],
                [row["privacy_delta_source_candidate_minus_source_native"] for row in u16],
                bootstrap_indices,
            ),
            "sharpness_loss": paired_mean_delta(
                [row["sharpness_loss"] for row in u12],
                [row["sharpness_loss"] for row in u16],
                bootstrap_indices,
            ),
        }

    top_outliers: dict[str, Any] = {"regular_privacy": {}, "tail_sharpness": {}}
    for arm in ARMS:
        for target_id, dataset_id, field in (
            ("regular_privacy", "regular32", "privacy_delta_source_candidate_minus_source_native"),
            ("tail_sharpness", "sharpness_tail32", "sharpness_loss"),
        ):
            rows = rows_by_dataset_arm[dataset_id][arm]
            selected = sorted(rows, key=lambda row: (-row[field], row["sample_id"]))[:TOP_K]
            top_outliers[target_id][arm] = [
                {
                    "rank": rank,
                    "sample_id": row["sample_id"],
                    "roi_lpips": row["roi_lpips_native_candidate"],
                    "full_lpips": row["full_lpips_native_candidate"],
                    "privacy_delta": row["privacy_delta_source_candidate_minus_source_native"],
                    "sharpness_loss": row["sharpness_loss"],
                    "sharpness_ratio": row["sharpness_ratio"],
                    "roi_fft_high_retention": row["roi_fft_high_retention"],
                    "delta_e0": row["delta_e0"],
                    "delta_edev": row["delta_edev"],
                }
                for rank, row in enumerate(selected, start=1)
            ]

    summary = {
        "schema_version": 1,
        "contract_type": "safa_r12_lpips_conflict_summary_v1",
        "boundary": {
            "existing_images_only": True,
            "new_generation": False,
            "new_training": False,
            "source_pixels_read_for_quality_distance": False,
            "formal_gate": False,
        },
        "inputs": {
            "request": str(request_path),
            "arcface_request": str(arcface_request),
            "device": device,
            "lpips": "lpips-0.1-alex",
            "ordered_datasets": [str(dataset["dataset_id"]) for dataset in prepared],
            "unique_observed_image_count": len(observation_cache),
            "joined_row_count": len(all_rows),
        },
        "statistics_contract": dict(statistics_contract),
        "license_rule": dict(license_contract),
        "associations": associations,
        "paired_changes": paired_changes,
        "decision": {
            "targets": target_decisions,
            "lpips_projection_licensed": licensed,
            "classification": (
                "native_anchor_lpips_projection_licensed"
                if licensed
                else "native_anchor_lpips_rejected_ungrounded"
            ),
            "arm_promotion": "forbidden",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "top_outliers.json", top_outliers)
    with (output_dir / "per_sample.jsonl").open("x", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "conclusion.md").write_text(build_conclusion(summary), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether native-anchor face-ROI LPIPS is grounded in R12 failures."
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
