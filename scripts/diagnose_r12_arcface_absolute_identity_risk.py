#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_r12_face_frequency import (  # noqa: E402
    FrequencyDiagnosticError,
    build_analyzer,
    serialize_exact_one_face,
)


ARMS = ("u12", "u16")
DATASETS = ("regular32", "sharpness_tail32")
BOOTSTRAP_SEED = 91637
BOOTSTRAP_ITERATIONS = 10_000
TOP_K = 8
GALLERY_SIZE = 64
EMBEDDING_DIMENSION = 512
EMBEDDING_NORM_TOLERANCE = 1e-5
ARCHIVED_COSINE_TOLERANCE = 1e-6
RELATIVE_DELTA_GATE = 0.02
ENRICHMENT_THRESHOLD = 2.0
TOP8_CANDIDATE_COSINE_MEDIAN_MAX = 0.1
SOURCE_NATIVE_SPEARMAN_MAX = -0.5


class IdentityRiskDiagnosticError(FrequencyDiagnosticError):
    """Raised when identity-risk inputs or observations violate the contract."""


def _reject_json_constant(value: str) -> None:
    raise IdentityRiskDiagnosticError(f"JSON contains non-finite constant: {value}")


def read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except json.JSONDecodeError as exc:
        raise IdentityRiskDiagnosticError(f"invalid {label}: {path}") from exc


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
                raise IdentityRiskDiagnosticError(
                    f"invalid {label} JSONL row {line_number}: {path}"
                ) from exc
            if not isinstance(row, Mapping):
                raise IdentityRiskDiagnosticError(
                    f"{label} row {line_number} must be an object"
                )
            rows.append(row)
    if not rows:
        raise IdentityRiskDiagnosticError(f"{label} is empty: {path}")
    return rows


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IdentityRiskDiagnosticError(f"{label} must be an object")
    return value


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise IdentityRiskDiagnosticError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise IdentityRiskDiagnosticError(f"{label} must be finite")
    return result


def resolve_repo_file(repo_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise IdentityRiskDiagnosticError(f"{label} must be a non-empty path")
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
            raise IdentityRiskDiagnosticError(f"{label} row {index} has invalid sample_id")
        if sample_id in by_id:
            raise IdentityRiskDiagnosticError(
                f"{label} contains duplicate sample_id: {sample_id}"
            )
        ordered.append(sample_id)
        by_id[sample_id] = row
    return ordered, by_id


def require_order(
    expected: Sequence[str], observed: Sequence[str], label: str
) -> None:
    if list(observed) != list(expected):
        raise IdentityRiskDiagnosticError(
            f"{label} sample order does not match formal native bindings"
        )


def formal_native_rows(
    path: Path, repo_root: Path
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ordered, raw_by_id = ordered_map(read_jsonl(path, "formal native bindings"), "formal native bindings")
    if len(ordered) != 32:
        raise IdentityRiskDiagnosticError("each dataset must contain exactly 32 samples")
    by_id: dict[str, dict[str, Any]] = {}
    for sample_id in ordered:
        row = raw_by_id[sample_id]
        source = resolve_repo_file(repo_root, row.get("source"), f"source for {sample_id}")
        native = resolve_repo_file(
            repo_root, row.get("formal_native"), f"formal native for {sample_id}"
        )
        digest = row.get("formal_native_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise IdentityRiskDiagnosticError(
                f"invalid formal native digest for {sample_id}"
            )
        if sha256_file(native) != digest:
            raise IdentityRiskDiagnosticError(
                f"formal native digest mismatch for {sample_id}"
            )
        by_id[sample_id] = {"source": source, "native": native}
    return ordered, by_id


def run_rows(
    path: Path,
    expected_ids: Sequence[str],
    formal: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
    label: str,
) -> dict[str, dict[str, Any]]:
    ordered, raw_by_id = ordered_map(read_jsonl(path, label), label)
    require_order(expected_ids, ordered, label)
    result: dict[str, dict[str, Any]] = {}
    for sample_id in ordered:
        row = raw_by_id[sample_id]
        source = resolve_repo_file(repo_root, row.get("source"), f"{label} source")
        if source != formal[sample_id]["source"]:
            raise IdentityRiskDiagnosticError(
                f"{label} source binding differs from formal binding for {sample_id}"
            )
        candidate = resolve_repo_file(
            repo_root, row.get("generated"), f"{label} candidate for {sample_id}"
        )
        result[sample_id] = {"candidate": candidate}
    return result


def arcface_rows(
    path: Path, expected_ids: Sequence[str], label: str
) -> dict[str, dict[str, float]]:
    payload = require_mapping(read_json(path, label), label)
    rows = payload.get("result")
    if not isinstance(rows, list):
        raise IdentityRiskDiagnosticError(f"{label} result must be a list")
    ordered, raw_by_id = ordered_map(rows, label)
    require_order(expected_ids, ordered, label)
    result: dict[str, dict[str, float]] = {}
    for sample_id in ordered:
        row = raw_by_id[sample_id]
        counts = [
            row.get("source_face_count"),
            row.get("native_face_count"),
            row.get("candidate_face_count"),
        ]
        if counts != [1, 1, 1]:
            raise IdentityRiskDiagnosticError(
                f"{label} archived exact-one failed for {sample_id}: {counts}"
            )
        result[sample_id] = {
            "source_native": finite_float(
                row.get("source_native_cosine"),
                f"{label} source-native cosine for {sample_id}",
            ),
            "source_candidate": finite_float(
                row.get("source_candidate_cosine"),
                f"{label} source-candidate cosine for {sample_id}",
            ),
        }
    return result


def extract_exact_one_embedding(
    faces: Any,
    image_shape: Sequence[int],
    embedding_dimension: int = EMBEDDING_DIMENSION,
    norm_tolerance: float = EMBEDDING_NORM_TOLERANCE,
) -> tuple[dict[str, Any], np.ndarray]:
    try:
        detection = serialize_exact_one_face(faces, image_shape)
    except FrequencyDiagnosticError as exc:
        raise IdentityRiskDiagnosticError(str(exc)) from exc
    embedding = np.asarray(
        getattr(faces[0], "normed_embedding", None), dtype=np.float64
    )
    if embedding.shape != (embedding_dimension,) or not np.isfinite(embedding).all():
        raise IdentityRiskDiagnosticError(
            f"ArcFace normed embedding must contain {embedding_dimension} finite values"
        )
    norm = float(np.linalg.norm(embedding))
    if not math.isfinite(norm) or abs(norm - 1.0) > norm_tolerance:
        raise IdentityRiskDiagnosticError(
            f"ArcFace normed embedding norm differs from one: {norm}"
        )
    return detection, embedding


def retrieval_metrics(
    query: np.ndarray, gallery: np.ndarray, true_index: int
) -> dict[str, Any]:
    query = np.asarray(query, dtype=np.float64)
    gallery = np.asarray(gallery, dtype=np.float64)
    if gallery.ndim != 2 or query.shape != (gallery.shape[1],):
        raise IdentityRiskDiagnosticError("query and gallery embedding shapes are invalid")
    if gallery.shape[0] < 2 or not 0 <= true_index < gallery.shape[0]:
        raise IdentityRiskDiagnosticError("gallery size or true index is invalid")
    if not np.isfinite(query).all() or not np.isfinite(gallery).all():
        raise IdentityRiskDiagnosticError("query or gallery contains non-finite values")
    scores = gallery @ query
    if not np.isfinite(scores).all():
        raise IdentityRiskDiagnosticError("retrieval cosine scores are non-finite")
    true_score = float(scores[true_index])
    impostors = np.delete(scores, true_index)
    max_impostor = float(np.max(impostors))
    # Ties are conservatively counted ahead of the true source.  Therefore a
    # positive margin is exactly equivalent to strict rank one.
    rank = 1 + int(np.count_nonzero(impostors >= true_score))
    percentile = float((gallery.shape[0] - rank) / (gallery.shape[0] - 1))
    margin = true_score - max_impostor
    return {
        "true_source_cosine": true_score,
        "true_source_rank": rank,
        "retrieval_percentile": percentile,
        "max_impostor_cosine": max_impostor,
        "max_impostor_margin": margin,
        "recall_at_1": int(rank <= 1),
        "recall_at_5": int(rank <= 5),
        "positive_margin": bool(margin > 0.0),
    }


def paired_bootstrap_mean_delta(
    native: Sequence[float],
    candidate: Sequence[float],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    native_array = np.asarray(native, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    indices = np.asarray(bootstrap_indices)
    if (
        native_array.ndim != 1
        or native_array.shape != candidate_array.shape
        or not np.isfinite(native_array).all()
        or not np.isfinite(candidate_array).all()
    ):
        raise IdentityRiskDiagnosticError("paired bootstrap arrays are invalid")
    if indices.ndim != 2 or indices.shape[1] != native_array.size:
        raise IdentityRiskDiagnosticError("paired bootstrap index shape is invalid")
    if indices.size == 0 or np.min(indices) < 0 or np.max(indices) >= native_array.size:
        raise IdentityRiskDiagnosticError("paired bootstrap indices are invalid")
    delta = candidate_array - native_array
    means = np.mean(delta[indices], axis=1)
    ci = np.percentile(means, [2.5, 97.5])
    return {
        "direction": "candidate_minus_native",
        "mean_delta": float(np.mean(delta)),
        "ci95": [float(ci[0]), float(ci[1])],
    }


def top_k_enrichment(events: Sequence[bool], top_indices: Sequence[int]) -> dict[str, Any]:
    event_array = np.asarray(events, dtype=bool)
    indices = np.asarray(top_indices, dtype=np.int64)
    if event_array.ndim != 1 or event_array.size == 0:
        raise IdentityRiskDiagnosticError("enrichment events must be a non-empty vector")
    if indices.ndim != 1 or indices.size == 0 or len(set(indices.tolist())) != indices.size:
        raise IdentityRiskDiagnosticError("top-k indices must be unique and non-empty")
    if np.min(indices) < 0 or np.max(indices) >= event_array.size:
        raise IdentityRiskDiagnosticError("top-k indices are out of range")
    event_count = int(np.count_nonzero(event_array))
    observed = int(np.count_nonzero(event_array[indices]))
    expected = float(indices.size * event_count / event_array.size)
    if event_count == 0:
        enrichment = 0.0
    else:
        enrichment = float(observed / expected)
    return {
        "event_count_all": event_count,
        "event_rate_all": float(event_count / event_array.size),
        "observed_in_top_k": observed,
        "event_rate_top_k": float(observed / indices.size),
        "random_expected_overlap": expected,
        "enrichment_factor": enrichment,
    }


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if (
        x_array.ndim != 1
        or x_array.shape != y_array.shape
        or x_array.size < 2
        or not np.isfinite(x_array).all()
        or not np.isfinite(y_array).all()
    ):
        raise IdentityRiskDiagnosticError("Spearman inputs are invalid")
    value = float(spearmanr(x_array, y_array).statistic)
    if not math.isfinite(value):
        raise IdentityRiskDiagnosticError("Spearman result is non-finite")
    return value


def aggregate_retrieval(rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, Any]:
    if not rows:
        raise IdentityRiskDiagnosticError("retrieval aggregation is empty")
    cells = [require_mapping(row.get(role), f"{role} retrieval") for row in rows]
    ranks = [int(cell["true_source_rank"]) for cell in cells]
    return {
        "mean_true_source_cosine": float(
            statistics.fmean(finite_float(cell["true_source_cosine"], "true cosine") for cell in cells)
        ),
        "mean_true_source_rank": float(statistics.fmean(ranks)),
        "median_true_source_rank": float(statistics.median(ranks)),
        "mean_retrieval_percentile": float(
            statistics.fmean(
                finite_float(cell["retrieval_percentile"], "retrieval percentile")
                for cell in cells
            )
        ),
        "mean_max_impostor_margin": float(
            statistics.fmean(
                finite_float(cell["max_impostor_margin"], "impostor margin")
                for cell in cells
            )
        ),
        "recall_at_1": float(statistics.fmean(int(cell["recall_at_1"]) for cell in cells)),
        "recall_at_5": float(statistics.fmean(int(cell["recall_at_5"]) for cell in cells)),
    }


def paired_retrieval_summary(
    rows: Sequence[Mapping[str, Any]], bootstrap_indices: np.ndarray
) -> dict[str, Any]:
    fields = (
        "true_source_cosine",
        "retrieval_percentile",
        "max_impostor_margin",
        "recall_at_1",
        "recall_at_5",
    )
    return {
        field: paired_bootstrap_mean_delta(
            [finite_float(row["native"][field], f"native {field}") for row in rows],
            [finite_float(row["candidate"][field], f"candidate {field}") for row in rows],
            bootstrap_indices,
        )
        for field in fields
    }


def evaluate_decision(
    regular: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if set(regular) != set(ARMS):
        raise IdentityRiskDiagnosticError("decision requires regular u12 and u16")
    arm_checks: dict[str, dict[str, bool]] = {}
    for arm in ARMS:
        cell = require_mapping(regular[arm], f"regular {arm}")
        paired = require_mapping(cell.get("paired_candidate_minus_native"), f"{arm} paired")
        percentile = require_mapping(paired.get("retrieval_percentile"), f"{arm} percentile")
        recall5 = require_mapping(paired.get("recall_at_5"), f"{arm} recall5")
        percentile_ci = percentile.get("ci95")
        recall5_ci = recall5.get("ci95")
        if not isinstance(percentile_ci, list) or len(percentile_ci) != 2:
            raise IdentityRiskDiagnosticError(f"{arm} percentile CI is invalid")
        if not isinstance(recall5_ci, list) or len(recall5_ci) != 2:
            raise IdentityRiskDiagnosticError(f"{arm} recall5 CI is invalid")
        percentile_supported = finite_float(percentile_ci[0], "percentile lower CI") > 0.0
        recall5_supported = finite_float(recall5_ci[0], "recall5 lower CI") > 0.0
        top8 = require_mapping(cell.get("top8_relative_delta_outliers"), f"{arm} top8")
        high_rank = require_mapping(top8.get("candidate_recall_at_5"), f"{arm} top8 recall5")
        positive_margin = require_mapping(
            top8.get("candidate_positive_margin"), f"{arm} top8 positive margin"
        )
        recall5_enriched = finite_float(
            high_rank.get("enrichment_factor"), "recall5 enrichment"
        ) >= ENRICHMENT_THRESHOLD
        margin_enriched = finite_float(
            positive_margin.get("enrichment_factor"), "positive-margin enrichment"
        ) >= ENRICHMENT_THRESHOLD
        relative_failed = finite_float(
            cell.get("mean_locked_relative_delta"), "mean relative delta"
        ) > RELATIVE_DELTA_GATE
        baseline_anticorrelation = finite_float(
            cell.get("relative_delta_vs_source_native_spearman"), "baseline Spearman"
        ) <= SOURCE_NATIVE_SPEARMAN_MAX
        low_absolute_cosine = finite_float(
            top8.get("candidate_true_source_cosine_median"), "top8 candidate cosine median"
        ) <= TOP8_CANDIDATE_COSINE_MEDIAN_MAX
        arm_checks[arm] = {
            "retrieval_percentile_ci95_lower_gt_0": percentile_supported,
            "recall_at_5_ci95_lower_gt_0": recall5_supported,
            "joint_supported_retrieval_increase": percentile_supported and recall5_supported,
            "top8_recall_at_5_enrichment_ge_2": recall5_enriched,
            "top8_positive_margin_enrichment_ge_2": margin_enriched,
            "locked_relative_delta_failed": relative_failed,
            "relative_delta_source_native_spearman_le_minus_0p5": baseline_anticorrelation,
            "top8_candidate_cosine_median_le_0p1": low_absolute_cosine,
            "top8_both_enrichments_lt_2": not recall5_enriched and not margin_enriched,
        }
    actual = all(
        checks["joint_supported_retrieval_increase"]
        and checks["top8_recall_at_5_enrichment_ge_2"]
        and checks["top8_positive_margin_enrichment_ge_2"]
        for checks in arm_checks.values()
    )
    baseline = (
        not actual
        and all(
            checks["locked_relative_delta_failed"]
            and checks["relative_delta_source_native_spearman_le_minus_0p5"]
            and checks["top8_candidate_cosine_median_le_0p1"]
            and checks["top8_both_enrichments_lt_2"]
            for checks in arm_checks.values()
        )
        and not any(
            checks["joint_supported_retrieval_increase"]
            for checks in arm_checks.values()
        )
    )
    classification = (
        "actual_retrieval_leakage_supported"
        if actual
        else "baseline_conditioned_metric_geometry"
        if baseline
        else "identity_risk_inconclusive"
    )
    return {
        "classification": classification,
        "actual_retrieval_leakage_supported": actual,
        "baseline_conditioned_metric_geometry": baseline,
        "checks_by_arm": arm_checks,
        "locked_relative_delta_gate_changed": False,
    }


def validate_request(request: Mapping[str, Any]) -> None:
    if (
        request.get("schema_version") != 1
        or request.get("contract_type")
        != "safa_r12_arcface_absolute_identity_risk_request_v1"
    ):
        raise IdentityRiskDiagnosticError("diagnostic request identity mismatch")
    expected_gallery = {
        "ordered_datasets": list(DATASETS),
        "expected_source_count": GALLERY_SIZE,
        "ranking": "strict_true_score_greater_than_every_impostor",
        "percentile": "(gallery_size-rank)/(gallery_size-1)",
    }
    expected_statistics = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "paired_resampling_unit": "sample_id",
        "top_k": TOP_K,
    }
    expected_gate = {
        "relative_arcface_delta_direction": "source_candidate_minus_source_native",
        "relative_arcface_delta_upper_bound": RELATIVE_DELTA_GATE,
        "gate_is_unchanged": True,
    }
    expected_numeric = {
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_norm_absolute_tolerance": EMBEDDING_NORM_TOLERANCE,
        "archived_cosine_absolute_tolerance": ARCHIVED_COSINE_TOLERANCE,
        "missing_nonfinite_or_non_exact_one": "fail_closed",
    }
    expected_decision = {
        "candidate_cosine_top8_median_max_for_baseline_geometry": TOP8_CANDIDATE_COSINE_MEDIAN_MAX,
        "source_native_spearman_max_for_baseline_geometry": SOURCE_NATIVE_SPEARMAN_MAX,
        "top8_enrichment_threshold": ENRICHMENT_THRESHOLD,
        "high_true_source_rank": "candidate_rank_le_5",
        "positive_margin": "candidate_true_source_score_gt_max_impostor",
        "actual_retrieval_leakage_supported": "both_regular_arms_have_percentile_and_recall5_paired_delta_ci95_lower_gt_0_and_top8_recall5_and_positive_margin_enrichment_ge_2",
        "baseline_conditioned_metric_geometry": "both_regular_arms_fail_locked_relative_delta_and_have_source_native_spearman_le_minus_0p5_and_top8_candidate_cosine_median_le_0p1_and_both_top8_enrichments_lt_2_and_no_arm_has_joint_supported_percentile_and_recall5_increase",
        "otherwise": "identity_risk_inconclusive",
    }
    for field, expected in (
        ("gallery", expected_gallery),
        ("statistics", expected_statistics),
        ("locked_gate_context", expected_gate),
        ("numeric_contract", expected_numeric),
        ("decision_rule", expected_decision),
    ):
        if dict(require_mapping(request.get(field), field)) != expected:
            raise IdentityRiskDiagnosticError(
                f"{field} differs from the predeclared diagnostic contract"
            )


def git_sha(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise IdentityRiskDiagnosticError("failed to resolve a full Git SHA")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def build_conclusion(summary: Mapping[str, Any]) -> str:
    decision = require_mapping(summary.get("decision"), "summary decision")
    regular = require_mapping(summary["datasets"].get("regular32"), "regular summary")
    lines = [
        "# R12 ArcFace absolute identity-risk diagnostic",
        "",
        f"Predeclared classification: `{decision['classification']}`.",
        "",
    ]
    for arm in ARMS:
        cell = require_mapping(regular.get(arm), f"regular {arm}")
        paired = require_mapping(cell["paired_candidate_minus_native"], f"{arm} paired")
        top8 = require_mapping(cell["top8_relative_delta_outliers"], f"{arm} top8")
        lines.append(
            f"- `{arm}`: locked relative delta mean `{cell['mean_locked_relative_delta']:.6f}`; "
            f"candidate Recall@1/5 `{cell['candidate']['recall_at_1']:.4f}/"
            f"{cell['candidate']['recall_at_5']:.4f}`; candidate-minus-native percentile "
            f"delta `{paired['retrieval_percentile']['mean_delta']:.6f}` "
            f"(95% CI `{paired['retrieval_percentile']['ci95'][0]:.6f}, "
            f"{paired['retrieval_percentile']['ci95'][1]:.6f}`); top-8 Recall@5/positive-margin "
            f"enrichment `{top8['candidate_recall_at_5']['enrichment_factor']:.3f}x/"
            f"{top8['candidate_positive_margin']['enrichment_factor']:.3f}x`."
        )
    lines.extend(
        [
            "",
            "The locked source-candidate minus source-native gate remains unchanged at `<= 0.02`. ",
            "This closed-set diagnostic only distinguishes baseline-conditioned relative-metric "
            "geometry from supported retrieval leakage on the existing 64-source gallery.",
            "",
            "Source pixels are read only by the locked buffalo_l identity analyzer. No image was "
            "generated and no model was trained. This result is not a privacy proof, a Full gate, "
            "or a formal winner.",
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
    validate_request(request)
    arcface_request = resolve_repo_file(
        repo_root, request.get("arcface_request"), "locked ArcFace request"
    )
    raw_datasets = request.get("datasets")
    if not isinstance(raw_datasets, list) or len(raw_datasets) != len(DATASETS):
        raise IdentityRiskDiagnosticError("request must contain the two locked datasets")

    prepared: list[dict[str, Any]] = []
    dataset_ids: list[str] = []
    all_sample_ids: list[str] = []
    for raw_dataset in raw_datasets:
        dataset = require_mapping(raw_dataset, "dataset")
        dataset_id = dataset.get("dataset_id")
        if dataset_id not in DATASETS or dataset_id in dataset_ids:
            raise IdentityRiskDiagnosticError(
                f"invalid or duplicate dataset ID: {dataset_id}"
            )
        dataset_ids.append(str(dataset_id))
        native_path = resolve_repo_file(
            repo_root,
            dataset.get("formal_native_bindings"),
            f"{dataset_id} formal native bindings",
        )
        sample_ids, formal = formal_native_rows(native_path, repo_root)
        overlap = set(all_sample_ids).intersection(sample_ids)
        if overlap:
            raise IdentityRiskDiagnosticError(
                f"source gallery sample IDs overlap across datasets: {sorted(overlap)[0]}"
            )
        all_sample_ids.extend(sample_ids)
        raw_arms = require_mapping(dataset.get("arms"), f"{dataset_id} arms")
        if set(raw_arms) != set(ARMS):
            raise IdentityRiskDiagnosticError(f"{dataset_id} must bind u12 and u16")
        arms: dict[str, Any] = {}
        for arm in ARMS:
            arm_config = require_mapping(raw_arms[arm], f"{dataset_id} {arm}")
            arms[arm] = {
                "run": run_rows(
                    resolve_repo_file(
                        repo_root,
                        arm_config.get("run_rows"),
                        f"{dataset_id} {arm} run rows",
                    ),
                    sample_ids,
                    formal,
                    repo_root,
                    f"{dataset_id} {arm} run rows",
                ),
                "arcface": arcface_rows(
                    resolve_repo_file(
                        repo_root,
                        arm_config.get("arcface"),
                        f"{dataset_id} {arm} archived ArcFace",
                    ),
                    sample_ids,
                    f"{dataset_id} {arm} archived ArcFace",
                ),
            }
        prepared.append(
            {
                "dataset_id": str(dataset_id),
                "sample_ids": sample_ids,
                "formal": formal,
                "arms": arms,
            }
        )
    if tuple(dataset_ids) != DATASETS or len(all_sample_ids) != GALLERY_SIZE:
        raise IdentityRiskDiagnosticError(
            "dataset order or combined source-gallery size differs from the contract"
        )

    role_paths: dict[str, list[Path]] = {"source": [], "native": [], "candidate": []}
    for dataset in prepared:
        for sample_id in dataset["sample_ids"]:
            role_paths["source"].append(dataset["formal"][sample_id]["source"])
            role_paths["native"].append(dataset["formal"][sample_id]["native"])
            for arm in ARMS:
                role_paths["candidate"].append(
                    dataset["arms"][arm]["run"][sample_id]["candidate"]
                )
    if len(set(role_paths["source"])) != 64 or len(set(role_paths["native"])) != 64:
        raise IdentityRiskDiagnosticError("source and native paths must each be unique")
    if len(set(role_paths["candidate"])) != 128:
        raise IdentityRiskDiagnosticError("candidate paths must be unique across both arms")
    if any(
        set(role_paths[first]).intersection(role_paths[second])
        for first, second in (("source", "native"), ("source", "candidate"), ("native", "candidate"))
    ):
        raise IdentityRiskDiagnosticError("image roles must bind disjoint files")

    analyzer = build_analyzer(arcface_request, device)
    observation_cache: dict[Path, dict[str, Any]] = {}

    def observe(path: Path) -> dict[str, Any]:
        if path not in observation_cache:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise IdentityRiskDiagnosticError(f"failed to decode image: {path}")
            faces = analyzer.get(image)
            detection, embedding = extract_exact_one_embedding(faces, image.shape)
            observation_cache[path] = {
                "detection": detection,
                "embedding": embedding,
            }
        return observation_cache[path]

    gallery_observations: list[dict[str, Any]] = []
    for dataset in prepared:
        for sample_id in dataset["sample_ids"]:
            gallery_observations.append(observe(dataset["formal"][sample_id]["source"]))
    gallery = np.stack([row["embedding"] for row in gallery_observations], axis=0)
    if gallery.shape != (GALLERY_SIZE, EMBEDDING_DIMENSION):
        raise IdentityRiskDiagnosticError("combined source gallery has an invalid shape")

    all_rows: list[dict[str, Any]] = []
    rows_by_dataset_arm: dict[str, dict[str, list[dict[str, Any]]]] = {
        dataset_id: {arm: [] for arm in ARMS} for dataset_id in DATASETS
    }
    gallery_offset = 0
    for dataset in prepared:
        dataset_id = dataset["dataset_id"]
        for ordinal, sample_id in enumerate(dataset["sample_ids"]):
            true_index = gallery_offset + ordinal
            source_observation = gallery_observations[true_index]
            native_path = dataset["formal"][sample_id]["native"]
            native_observation = observe(native_path)
            native_retrieval = retrieval_metrics(
                native_observation["embedding"], gallery, true_index
            )
            for arm in ARMS:
                candidate_path = dataset["arms"][arm]["run"][sample_id]["candidate"]
                candidate_observation = observe(candidate_path)
                candidate_retrieval = retrieval_metrics(
                    candidate_observation["embedding"], gallery, true_index
                )
                archived = dataset["arms"][arm]["arcface"][sample_id]
                for name, recomputed, expected in (
                    ("source-native", native_retrieval["true_source_cosine"], archived["source_native"]),
                    ("source-candidate", candidate_retrieval["true_source_cosine"], archived["source_candidate"]),
                ):
                    if abs(float(recomputed) - float(expected)) > ARCHIVED_COSINE_TOLERANCE:
                        raise IdentityRiskDiagnosticError(
                            f"{dataset_id} {arm} {sample_id} recomputed {name} cosine "
                            f"differs from archived value: {recomputed} versus {expected}"
                        )
                relative_delta = (
                    candidate_retrieval["true_source_cosine"]
                    - native_retrieval["true_source_cosine"]
                )
                row = {
                    "dataset_id": dataset_id,
                    "arm": arm,
                    "ordinal": ordinal,
                    "gallery_true_index": true_index,
                    "sample_id": sample_id,
                    "source_path": str(dataset["formal"][sample_id]["source"]),
                    "native_path": str(native_path),
                    "candidate_path": str(candidate_path),
                    "source_detection": source_observation["detection"],
                    "native_detection": native_observation["detection"],
                    "candidate_detection": candidate_observation["detection"],
                    "archived_source_native_cosine": archived["source_native"],
                    "archived_source_candidate_cosine": archived["source_candidate"],
                    "locked_relative_delta_source_candidate_minus_source_native": relative_delta,
                    "native": native_retrieval,
                    "candidate": candidate_retrieval,
                    "paired_candidate_minus_native": {
                        "true_source_cosine": relative_delta,
                        "true_source_rank": candidate_retrieval["true_source_rank"]
                        - native_retrieval["true_source_rank"],
                        "retrieval_percentile": candidate_retrieval["retrieval_percentile"]
                        - native_retrieval["retrieval_percentile"],
                        "max_impostor_margin": candidate_retrieval["max_impostor_margin"]
                        - native_retrieval["max_impostor_margin"],
                        "recall_at_1": candidate_retrieval["recall_at_1"]
                        - native_retrieval["recall_at_1"],
                        "recall_at_5": candidate_retrieval["recall_at_5"]
                        - native_retrieval["recall_at_5"],
                    },
                }
                rows_by_dataset_arm[dataset_id][arm].append(row)
                all_rows.append(row)
        gallery_offset += len(dataset["sample_ids"])

    if len(observation_cache) != 256 or len(all_rows) != 128:
        raise IdentityRiskDiagnosticError(
            "observed-image or joined-row count differs from the locked 256/128 contract"
        )
    bootstrap_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, 32, size=(BOOTSTRAP_ITERATIONS, 32), dtype=np.int64
    )
    dataset_summaries: dict[str, dict[str, Any]] = {}
    top_outliers: dict[str, dict[str, Any]] = {}
    for dataset_id in DATASETS:
        dataset_summaries[dataset_id] = {}
        top_outliers[dataset_id] = {}
        for arm in ARMS:
            rows = rows_by_dataset_arm[dataset_id][arm]
            deltas = [
                finite_float(
                    row["locked_relative_delta_source_candidate_minus_source_native"],
                    "locked relative delta",
                )
                for row in rows
            ]
            top_indices = sorted(
                range(len(rows)), key=lambda index: (-deltas[index], rows[index]["sample_id"])
            )[:TOP_K]
            top_rows = [rows[index] for index in top_indices]
            top_summary = {
                "top_k": TOP_K,
                "candidate_true_source_cosine_median": float(
                    statistics.median(
                        finite_float(row["candidate"]["true_source_cosine"], "candidate cosine")
                        for row in top_rows
                    )
                ),
                "candidate_recall_at_5": top_k_enrichment(
                    [bool(row["candidate"]["recall_at_5"]) for row in rows], top_indices
                ),
                "candidate_positive_margin": top_k_enrichment(
                    [bool(row["candidate"]["positive_margin"]) for row in rows], top_indices
                ),
            }
            dataset_summaries[dataset_id][arm] = {
                "sample_count": len(rows),
                "mean_locked_relative_delta": float(statistics.fmean(deltas)),
                "relative_delta_vs_source_native_spearman": spearman(
                    deltas,
                    [
                        finite_float(row["native"]["true_source_cosine"], "native cosine")
                        for row in rows
                    ],
                ),
                "native": aggregate_retrieval(rows, "native"),
                "candidate": aggregate_retrieval(rows, "candidate"),
                "paired_candidate_minus_native": paired_retrieval_summary(
                    rows, bootstrap_indices
                ),
                "top8_relative_delta_outliers": top_summary,
            }
            top_outliers[dataset_id][arm] = [
                {
                    "rank": rank,
                    "sample_id": row["sample_id"],
                    "locked_relative_delta": row[
                        "locked_relative_delta_source_candidate_minus_source_native"
                    ],
                    "source_native_cosine": row["native"]["true_source_cosine"],
                    "source_candidate_cosine": row["candidate"]["true_source_cosine"],
                    "candidate_true_source_rank": row["candidate"]["true_source_rank"],
                    "candidate_retrieval_percentile": row["candidate"]["retrieval_percentile"],
                    "candidate_max_impostor_margin": row["candidate"]["max_impostor_margin"],
                    "candidate_recall_at_1": row["candidate"]["recall_at_1"],
                    "candidate_recall_at_5": row["candidate"]["recall_at_5"],
                }
                for rank, row in enumerate(top_rows, start=1)
            ]

    decision = evaluate_decision(dataset_summaries["regular32"])
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r12_arcface_absolute_identity_risk_summary_v1",
        "boundary": {
            "existing_images_only": True,
            "new_generation": False,
            "new_training": False,
            "source_pixels_read_only_by_identity_metric": True,
            "locked_relative_delta_gate_unchanged": True,
            "formal_gate": False,
            "privacy_proof": False,
        },
        "inputs": {
            "git_sha": git_sha(repo_root),
            "request": str(request_path),
            "request_sha256": sha256_file(request_path),
            "arcface_request": str(arcface_request),
            "arcface_request_sha256": sha256_file(arcface_request),
            "device": device,
            "gallery_size": GALLERY_SIZE,
            "unique_observed_image_count": len(observation_cache),
            "joined_row_count": len(all_rows),
            "exact_one_observation_count": len(observation_cache),
        },
        "statistics_contract": dict(request["statistics"]),
        "locked_gate_context": dict(request["locked_gate_context"]),
        "decision_rule": dict(request["decision_rule"]),
        "datasets": dataset_summaries,
        "decision": decision,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "top_outliers.json", top_outliers)
    with (output_dir / "per_sample.jsonl").open("x", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "conclusion.md").write_text(
        build_conclusion(summary), encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare absolute closed-set ArcFace retrieval risk with the locked relative gate "
            "on existing R12 images."
        )
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
