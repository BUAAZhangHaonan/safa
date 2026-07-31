from __future__ import annotations

import math
from typing import Any, Mapping


DATASET_IDS = ("prefix128", "sharpness_tail32")
ROLE_IDS = ("native", "paper_eta_0p125", "schedule_nfe2")
FORBIDDEN_CLASSIFIER_METRIC_TOKENS = (
    "fid",
    "kid",
    "u95",
    "arcface",
    "privacy",
)


class ScheduleNFE2Error(RuntimeError):
    pass


def _reject_nonclassifier_metrics(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(
                token in normalized
                for token in FORBIDDEN_CLASSIFIER_METRIC_TOKENS
            ):
                raise ScheduleNFE2Error(
                    f"{label} contains non-classifier metric {key!r}"
                )
            _reject_nonclassifier_metrics(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonclassifier_metrics(item, label=f"{label}[{index}]")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ScheduleNFE2Error(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScheduleNFE2Error(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ScheduleNFE2Error(f"{label} must be finite")
    return result


def classify_schedule_nfe2(
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the locked quality-only NFE2 schedule diagnostic rule."""
    _reject_nonclassifier_metrics(evidence, label="NFE2 classifier evidence")
    if set(evidence) != set(DATASET_IDS):
        raise ScheduleNFE2Error(
            f"NFE2 evidence datasets must be {list(DATASET_IDS)!r}"
        )
    datasets: list[dict[str, Any]] = []
    for dataset_id in DATASET_IDS:
        row = evidence[dataset_id]
        if set(row) != {"sample_count", "quality", "representation"}:
            raise ScheduleNFE2Error(
                f"{dataset_id} evidence fields are not canonical"
            )
        expected_count = 128 if dataset_id == "prefix128" else 32
        sample_count = row["sample_count"]
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count != expected_count
        ):
            raise ScheduleNFE2Error(
                f"{dataset_id}.sample_count must be {expected_count}"
            )
        quality = row["quality"]
        representation = row["representation"]
        if not isinstance(quality, Mapping) or set(quality) != set(ROLE_IDS):
            raise ScheduleNFE2Error(
                f"{dataset_id}.quality roles are not canonical"
            )
        if (
            not isinstance(representation, Mapping)
            or set(representation) != set(ROLE_IDS)
        ):
            raise ScheduleNFE2Error(
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
                raise ScheduleNFE2Error(
                    f"{dataset_id}.{role}.quality fields are not canonical"
                )
            if (
                not isinstance(role_representation, Mapping)
                or set(role_representation) != {"e0_cosine", "edev_cosine"}
            ):
                raise ScheduleNFE2Error(
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
        candidate = normalized_quality["schedule_nfe2"]
        niqe_limit = native["niqe"] + 0.10
        sharpness_floor = 0.95 * native["sharpness"]
        nfe2_quality_pass = (
            candidate["niqe"] <= niqe_limit
            and candidate["sharpness"] >= sharpness_floor
        )
        datasets.append(
            {
                "dataset_id": dataset_id,
                "sample_count": sample_count,
                "quality": {
                    "native": native,
                    "schedule_nfe2": candidate,
                },
                "parent_paper_report_only": normalized_quality[
                    "paper_eta_0p125"
                ],
                "representation_report_only": normalized_representation,
                "thresholds": {
                    "niqe_max": niqe_limit,
                    "sharpness_min": sharpness_floor,
                },
                "nfe2_quality_pass": nfe2_quality_pass,
            }
        )
    passes_both = all(row["nfe2_quality_pass"] for row in datasets)
    classification = (
        "schedule_limited" if passes_both else "mixed_guidance_failure"
    )
    return {
        "schema_version": 1,
        "contract_type": "safa_r11_schedule_nfe2_classification_v1",
        "classification": classification,
        "nfe2_quality_passes_both_datasets": passes_both,
        "classifier_metrics": ["niqe", "sharpness"],
        "split_route": {
            "stop_required": not passes_both,
            "status": "stop" if not passes_both else "diagnostic_complete",
        },
        "representation": {
            "status": "report_only",
            "used_for_classification": False,
        },
        "geometry": {"status": "not_evaluated"},
        "privacy": {"status": "not_evaluated"},
        "candidate_promotion": {
            "status": "forbidden",
            "reason": "schedule_nfe2_is_a_causal_diagnostic_not_a_survivor",
        },
        "datasets": datasets,
    }
