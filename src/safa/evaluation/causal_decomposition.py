from __future__ import annotations

import math
from typing import Any, Mapping


DATASET_IDS = ("prefix128", "sharpness_tail32")
ROLE_IDS = ("native", "transport_only_nfe5", "paper_eta_0p125")
FORBIDDEN_CLASSIFIER_METRIC_TOKENS = ("fid", "kid", "u95")


class CausalDecompositionError(RuntimeError):
    pass


def _reject_noncausal_metrics(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(
                token in normalized
                for token in FORBIDDEN_CLASSIFIER_METRIC_TOKENS
            ):
                raise CausalDecompositionError(
                    f"{label} contains non-classifier metric {key!r}"
                )
            _reject_noncausal_metrics(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_noncausal_metrics(item, label=f"{label}[{index}]")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CausalDecompositionError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CausalDecompositionError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise CausalDecompositionError(f"{label} must be finite")
    return result


def classify_causal_decomposition(
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the locked quality-only R11 causal branch rule.

    FID/KID may be reported separately for prefix128, but are deliberately not
    accepted as classifier evidence. Geometry and privacy are not evaluated for
    this transport-only diagnostic.
    """
    _reject_noncausal_metrics(evidence, label="causal evidence")
    if set(evidence) != set(DATASET_IDS):
        raise CausalDecompositionError(
            f"causal evidence datasets must be {list(DATASET_IDS)!r}"
        )
    datasets: list[dict[str, Any]] = []
    for dataset_id in DATASET_IDS:
        row = evidence[dataset_id]
        if set(row) != {"sample_count", "quality", "representation"}:
            raise CausalDecompositionError(
                f"{dataset_id} causal evidence fields are not canonical"
            )
        sample_count = row["sample_count"]
        expected_count = 128 if dataset_id == "prefix128" else 32
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count != expected_count
        ):
            raise CausalDecompositionError(
                f"{dataset_id}.sample_count must be {expected_count}"
            )
        quality = row["quality"]
        representation = row["representation"]
        if not isinstance(quality, Mapping) or set(quality) != set(ROLE_IDS):
            raise CausalDecompositionError(
                f"{dataset_id}.quality roles are not canonical"
            )
        if (
            not isinstance(representation, Mapping)
            or set(representation) != set(ROLE_IDS)
        ):
            raise CausalDecompositionError(
                f"{dataset_id}.representation roles are not canonical"
            )
        normalized_quality: dict[str, dict[str, float]] = {}
        normalized_representation: dict[str, dict[str, float]] = {}
        for role in ROLE_IDS:
            role_quality = quality[role]
            role_representation = representation[role]
            if not isinstance(role_quality, Mapping) or set(role_quality) != {
                "niqe",
                "sharpness",
            }:
                raise CausalDecompositionError(
                    f"{dataset_id}.{role}.quality fields are not canonical"
                )
            if (
                not isinstance(role_representation, Mapping)
                or set(role_representation) != {"e0_cosine", "edev_cosine"}
            ):
                raise CausalDecompositionError(
                    f"{dataset_id}.{role}.representation fields are not canonical"
                )
            normalized_quality[role] = {
                metric: _finite(
                    role_quality[metric],
                    f"{dataset_id}.{role}.{metric}",
                )
                for metric in ("niqe", "sharpness")
            }
            normalized_representation[role] = {
                metric: _finite(
                    role_representation[metric],
                    f"{dataset_id}.{role}.{metric}",
                )
                for metric in ("e0_cosine", "edev_cosine")
            }
        native = normalized_quality["native"]
        transport = normalized_quality["transport_only_nfe5"]
        paper = normalized_quality["paper_eta_0p125"]
        niqe_limit = native["niqe"] + 0.10
        sharpness_floor = 0.95 * native["sharpness"]
        transport_quality_pass = (
            transport["niqe"] <= niqe_limit
            and transport["sharpness"] >= sharpness_floor
        )
        paper_quality_pass = (
            paper["niqe"] <= niqe_limit
            and paper["sharpness"] >= sharpness_floor
        )
        datasets.append(
            {
                "dataset_id": dataset_id,
                "sample_count": sample_count,
                "quality": normalized_quality,
                "representation_report_only": normalized_representation,
                "thresholds": {
                    "niqe_max": niqe_limit,
                    "sharpness_min": sharpness_floor,
                },
                "transport_quality_pass": transport_quality_pass,
                "paper_quality_pass": paper_quality_pass,
            }
        )
    transport_passes_both = all(
        row["transport_quality_pass"] for row in datasets
    )
    paper_fails_quality_on_either = any(
        not row["paper_quality_pass"] for row in datasets
    )
    classification = (
        "correction_limited"
        if transport_passes_both and paper_fails_quality_on_either
        else "schedule_branch"
    )
    return {
        "schema_version": 1,
        "contract_type": "safa_r11_causal_decomposition_classification_v1",
        "classification": classification,
        "transport_passes_both_datasets": transport_passes_both,
        "paper_fails_quality_on_either_dataset": paper_fails_quality_on_either,
        "classifier_metrics": ["niqe", "sharpness"],
        "representation": {
            "status": "report_only",
            "used_for_classification": False,
        },
        "geometry": {"status": "not_evaluated"},
        "privacy": {"status": "not_evaluated"},
        "candidate_promotion": {
            "status": "forbidden",
            "reason": "transport_only_nfe5_is_a_causal_diagnostic_not_a_survivor",
        },
        "datasets": datasets,
    }
