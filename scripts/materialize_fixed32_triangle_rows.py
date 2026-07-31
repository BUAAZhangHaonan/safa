#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from safa.evaluation.triangle32_evaluation import (
    canonical_digest,
    load_arm_set,
    validate_generation_result,
)
from safa.evaluation.triangle_screening import TriangleScreeningError


def _json(path: Path) -> Any:
    if not path.is_file():
        raise TriangleScreeningError(f"missing authoritative input: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TriangleScreeningError(f"invalid JSON in {path}") from exc


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise TriangleScreeningError(f"missing authoritative input: {path}")
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise TriangleScreeningError(
                    f"{path}:{line_number}: blank rows are forbidden"
                )
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise TriangleScreeningError(
                    f"{path}:{line_number}: row must be an object"
                )
            rows.append(row)
    return rows


def _indexed(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise TriangleScreeningError(f"{label} row {index} has invalid sample_id")
        if sample_id in result:
            raise TriangleScreeningError(f"{label} duplicate sample_id: {sample_id}")
        result[sample_id] = row
    return result


def _ordered_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_sample_ids: Sequence[str],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    result = _indexed(rows, label)
    if list(result) != list(expected_sample_ids):
        raise TriangleScreeningError(
            f"{label} sample IDs/order disagree with fixed32"
        )
    return result


def _quality_payload(
    path: Path, label: str, *, expected_metrics: Sequence[str]
) -> Mapping[str, Any]:
    payload = _json(path)
    if not isinstance(payload, Mapping):
        raise TriangleScreeningError(f"{label} quality output must be an object")
    metrics = payload.get("metrics")
    if metrics != list(expected_metrics):
        raise TriangleScreeningError(
            f"{label} quality output metrics disagree with the locked dataset"
        )
    per_sample = payload.get("per_sample_metrics")
    if (
        not isinstance(per_sample, Mapping)
        or per_sample.get("contract_type")
        != "safa_r9_quality_per_sample_metrics_v1"
        or not isinstance(per_sample.get("rows"), list)
    ):
        raise TriangleScreeningError(
            f"{label} official per-sample quality metrics are missing"
        )
    if "fid" not in expected_metrics and any(
        field in payload for field in ("fid", "kid_mean", "kid_std")
    ):
        raise TriangleScreeningError(
            f"{label} tail quality output contains forbidden FID/KID"
        )
    return payload


def _finite(row: Mapping[str, Any], field: str, label: str) -> float:
    value = row.get(field)
    if isinstance(value, bool):
        raise TriangleScreeningError(f"{label}.{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TriangleScreeningError(f"{label}.{field} must be finite") from exc
    if not math.isfinite(result):
        raise TriangleScreeningError(f"{label}.{field} must be finite")
    return result


def _integer(row: Mapping[str, Any], field: str, label: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TriangleScreeningError(
            f"{label}.{field} must be a nonnegative integer"
        )
    return value


def _identity_cosines(
    row: Mapping[str, Any], *, label: str
) -> tuple[float | None, float | None]:
    legacy_fields = sorted(
        field for field in ("native_cosine", "candidate_cosine") if field in row
    )
    if legacy_fields:
        raise TriangleScreeningError(
            f"{label} uses forbidden legacy ArcFace fields: {legacy_fields}"
        )
    counts = tuple(
        _integer(row, field, label)
        for field in (
            "source_face_count",
            "native_face_count",
            "candidate_face_count",
        )
    )
    all_roles_exact_one = counts == (1, 1, 1)
    values: list[float | None] = []
    for field, pair_exact_one in (
        ("source_native_cosine", counts[0] == counts[1] == 1),
        ("source_candidate_cosine", counts[0] == counts[2] == 1),
    ):
        if pair_exact_one:
            if row.get(field) is None and not all_roles_exact_one:
                values.append(None)
            else:
                value = _finite(row, field, label)
                if value < -1.0 or value > 1.0:
                    raise TriangleScreeningError(f"{label}.{field} must be in [-1,1]")
                values.append(value)
        else:
            if row.get(field) is not None:
                raise TriangleScreeningError(
                    f"{label}.{field} must be null when its role pair is not exact-one"
                )
            values.append(None)
    return values[0], values[1]


def _representation_cosines(
    native_row: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    *,
    candidate_label: str,
) -> tuple[float, float, float, float]:
    return (
        _finite(native_row, "native_cosine", "shared native representation"),
        _finite(candidate_row, "e0_cosine", candidate_label),
        _finite(
            native_row, "native_edev_cosine", "shared native representation"
        ),
        _finite(candidate_row, "edev_cosine", candidate_label),
    )


def _direct_meanflow_representation_cosines(
    row: Mapping[str, Any], *, candidate_label: str
) -> tuple[float, float, float, float]:
    forbidden = sorted(
        field for field in ("e0_cosine", "candidate_edev_cosine") if field in row
    )
    if forbidden:
        raise TriangleScreeningError(
            f"{candidate_label} uses forbidden representation aliases: {forbidden}"
        )
    values = (
        _finite(row, "native_cosine", candidate_label),
        _finite(row, "candidate_cosine", candidate_label),
        _finite(row, "native_edev_cosine", candidate_label),
        _finite(row, "edev_cosine", candidate_label),
    )
    if any(value < -1.0 or value > 1.0 for value in values):
        raise TriangleScreeningError(
            f"{candidate_label} representation cosines must be in [-1,1]"
        )
    if _finite(row, "cosine", candidate_label) != values[1]:
        raise TriangleScreeningError(
            f"{candidate_label}.cosine must equal candidate_cosine exactly"
        )
    return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official_arcface(
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    arm_id: str,
    sample_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    if (
        request.get("schema_version") != 1
        or request.get("contract_type") != "safa_r9_phase_evaluator_request_v1"
        or request.get("task") != "arcface"
        or request.get("evaluator_request_sha256")
        != canonical_digest(request, "evaluator_request_sha256")
    ):
        raise TriangleScreeningError(f"{arm_id} ArcFace request is not canonical")
    payload = request.get("payload")
    samples = payload.get("samples") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("arm_id") != arm_id
        or not isinstance(samples, list)
        or [row.get("sample_id") for row in samples if isinstance(row, Mapping)]
        != list(sample_ids)
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "sample_id",
                "source",
                "native",
                "candidate",
                "source_sha256",
                "native_sha256",
                "candidate_sha256",
            }
            for row in samples
        )
    ):
        raise TriangleScreeningError(
            f"{arm_id} ArcFace request is not exact three-role evidence"
        )
    if (
        result.get("schema_version") != 1
        or result.get("contract_type") != "safa_r9_phase_evaluator_output_v1"
        or result.get("task") != "arcface"
        or result.get("evaluator_request_sha256")
        != request["evaluator_request_sha256"]
        or result.get("evaluator_output_sha256")
        != canonical_digest(result, "evaluator_output_sha256")
        or not isinstance(result.get("result"), list)
    ):
        raise TriangleScreeningError(
            f"{arm_id} ArcFace result is not the bound official evaluator output"
        )
    return result["result"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize stage32 triangle rows from official evaluator outputs."
    )
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--native-representation-rows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-arm-id")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    diagnostic_path = args.diagnostic_manifest.resolve()
    arm_set = load_arm_set(diagnostic_path)
    arm_ids = list(arm_set.arm_ids)
    baseline_arm_id = args.baseline_arm_id or arm_set.baseline_arm_id
    if baseline_arm_id not in arm_ids:
        raise TriangleScreeningError(
            "baseline arm ID must explicitly name an arm in the arm-set manifest"
        )

    selection_path = args.selection_manifest.resolve()
    if (
        selection_path != arm_set.selection_manifest
        or _sha256(selection_path) != arm_set.selection_manifest_sha256
    ):
        raise TriangleScreeningError("selection manifest disagrees with arm-set lock")
    selection_rows = _jsonl(selection_path)
    selection_ids = [row.get("sample_id") for row in selection_rows]
    if (
        len(selection_ids) != arm_set.sample_count
        or len(set(selection_ids)) != arm_set.sample_count
        or any(not isinstance(value, str) for value in selection_ids)
    ):
        raise TriangleScreeningError(
            f"selection manifest must contain {arm_set.sample_count} unique IDs"
        )

    is_r11 = (
        arm_set.contract_type == "safa_r11_initial_noise_evaluation_dataset_v1"
    )
    if is_r11:
        if args.native_representation_rows is not None:
            raise TriangleScreeningError(
                "R11 materialization reads native representations directly "
                "from each generation row"
            )
        native_representation_path = None
        native_representation = None
    else:
        if args.native_representation_rows is None:
            raise TriangleScreeningError(
                "legacy materialization requires native representation rows"
            )
        native_representation_path = args.native_representation_rows.resolve()
        native_representation = _ordered_index(
            _jsonl(native_representation_path),
            expected_sample_ids=selection_ids,
            label="shared native representation",
        )
    runs_root = args.runs_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    native_quality_path = evaluation_root / "quality" / "native" / "quality.json"
    native_quality_payload = _quality_payload(
        native_quality_path,
        "native",
        expected_metrics=arm_set.quality_metrics,
    )
    native_quality = _ordered_index(
        native_quality_payload["per_sample_metrics"]["rows"],
        expected_sample_ids=selection_ids,
        label="native quality",
    )
    native_fid = (
        _finite(native_quality_payload, "fid", "native quality")
        if arm_set.stage != 32
        else None
    )
    native_kid = (
        _finite(native_quality_payload, "kid_mean", "native quality")
        if arm_set.stage != 32
        else None
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    materialized_arms: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for arm_id in arm_ids:
        run_path = validate_generation_result(arm_set, runs_root, arm_id)
        candidate_quality_path = (
            evaluation_root / "quality" / arm_id / "quality.json"
        )
        arcface_request_path = evaluation_root / "arcface" / arm_id / "request.json"
        arcface_result_path = evaluation_root / "arcface" / arm_id / "result.json"
        run_rows = _ordered_index(
            _jsonl(run_path),
            expected_sample_ids=selection_ids,
            label=f"{arm_id} generation",
        )
        candidate_quality_payload = _quality_payload(
            candidate_quality_path,
            arm_id,
            expected_metrics=arm_set.quality_metrics,
        )
        candidate_quality = _ordered_index(
            candidate_quality_payload["per_sample_metrics"]["rows"],
            expected_sample_ids=selection_ids,
            label=f"{arm_id} quality",
        )
        request = _json(arcface_request_path)
        result = _json(arcface_result_path)
        if not isinstance(request, Mapping) or not isinstance(result, Mapping):
            raise TriangleScreeningError(f"{arm_id} ArcFace envelope must be an object")
        config = request.get("config")
        detector = config.get("arcface") if isinstance(config, Mapping) else None
        if not isinstance(detector, Mapping) or detector.get("model_name") != "buffalo_l":
            raise TriangleScreeningError(
                f"{arm_id} ArcFace request must bind buffalo_l"
            )
        arcface_rows = _official_arcface(
            request, result, arm_id=arm_id, sample_ids=selection_ids
        )
        arcface = _ordered_index(
            arcface_rows,
            expected_sample_ids=selection_ids,
            label=f"{arm_id} ArcFace",
        )
        rows: list[dict[str, Any]] = []
        for sample_id in selection_ids:
            generated = run_rows[sample_id]
            native_q = native_quality[sample_id]
            candidate_q = candidate_quality[sample_id]
            identity = arcface[sample_id]
            if is_r11:
                native_e0, candidate_e0, native_edev, candidate_edev = (
                    _direct_meanflow_representation_cosines(
                        generated, candidate_label=arm_id
                    )
                )
            else:
                assert native_representation is not None
                native_e0, candidate_e0, native_edev, candidate_edev = (
                    _representation_cosines(
                        native_representation[sample_id],
                        generated,
                        candidate_label=arm_id,
                    )
                )
            source_native_cosine, source_candidate_cosine = _identity_cosines(
                identity, label=f"{arm_id} ArcFace"
            )
            rows.append(
                {
                    "arm_id": arm_id,
                    "stage": arm_set.stage,
                    "sample_id": sample_id,
                    "native_e0": native_e0,
                    "candidate_e0": candidate_e0,
                    "native_edev": native_edev,
                    "candidate_edev": candidate_edev,
                    "native_niqe": _finite(native_q, "niqe", "native quality"),
                    "candidate_niqe": _finite(
                        candidate_q, "niqe", f"{arm_id} quality"
                    ),
                    "native_sharpness": _finite(
                        native_q, "sharpness", "native quality"
                    ),
                    "candidate_sharpness": _finite(
                        candidate_q, "sharpness", f"{arm_id} quality"
                    ),
                    "source_face_count": _integer(
                        identity, "source_face_count", f"{arm_id} ArcFace"
                    ),
                    "native_face_count": _integer(
                        identity, "native_face_count", f"{arm_id} ArcFace"
                    ),
                    "candidate_face_count": _integer(
                        identity, "candidate_face_count", f"{arm_id} ArcFace"
                    ),
                    "source_native_cosine": source_native_cosine,
                    "source_candidate_cosine": source_candidate_cosine,
                }
            )
        row_path = output_dir / f"{arm_id}.jsonl"
        with row_path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
        arm_binding: dict[str, Any] = {
            "arm_id": arm_id,
            "rows_path": str(row_path),
        }
        if arm_set.stage != 32:
            arm_binding.update(
                {
                    "fid": _finite(candidate_quality_payload, "fid", arm_id),
                    "kid": _finite(candidate_quality_payload, "kid_mean", arm_id),
                }
            )
        materialized_arms.append(arm_binding)
        provenance.append(
            {
                "arm_id": arm_id,
                "generation_rows_sha256": _sha256(run_path),
                "candidate_quality_output_sha256": _sha256(candidate_quality_path),
                "arcface_request_sha256": _sha256(arcface_request_path),
                "arcface_result_sha256": _sha256(arcface_result_path),
                "materialized_rows_sha256": _sha256(row_path),
            }
        )
    request_path = output_dir / "screening_request.json"
    with request_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": arm_set.stage,
                    "baseline_arm_id": baseline_arm_id,
                    "selection_manifest_path": str(selection_path),
                    "selection_manifest_sha256": _sha256(selection_path),
                    "selection_role": arm_set.selection_role,
                    "arms": materialized_arms,
                    **(
                        {}
                        if arm_set.stage == 32
                        else {"native_fid": native_fid, "native_kid": native_kid}
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    summary_path = output_dir / "materialization_summary.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "contract_type": (
                        "safa_r11_typed_triangle_rows_v1"
                        if is_r11
                        else "safa_triangle_fixed32_rows_v1"
                    ),
                    "selection_role": arm_set.selection_role,
                    "sample_count": arm_set.sample_count,
                    "stage": arm_set.stage,
                    "metric_contract": {
                        "fid_kid": (
                            "required"
                            if arm_set.stage != 32
                            else "forbidden_absent"
                        ),
                        "quality": list(arm_set.quality_metrics),
                        "privacy": (
                            "arcface_delta_u95_shared_pcg64_2000"
                            if arm_set.stage != 32
                            else "point_arcface_delta"
                        ),
                        "arcface_detector": "buffalo_l",
                    },
                    "diagnostic_manifest_sha256": _sha256(diagnostic_path),
                    "selection_manifest_sha256": _sha256(selection_path),
                    "native_representation_rows": (
                        {
                            "binding": "direct_per_arm_generation_rows",
                            "fields": [
                                "native_cosine",
                                "candidate_cosine",
                                "native_edev_cosine",
                                "edev_cosine",
                            ],
                        }
                        if is_r11
                        else {
                            "path": str(native_representation_path),
                            "sha256": _sha256(native_representation_path),
                        }
                    ),
                    "native_quality_output_sha256": _sha256(native_quality_path),
                    "arms": provenance,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TriangleScreeningError as exc:
        print(f"fixed32 materialization failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
