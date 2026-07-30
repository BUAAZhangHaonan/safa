#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

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


def _quality_rows(path: Path, label: str) -> list[Mapping[str, Any]]:
    payload = _json(path)
    if not isinstance(payload, Mapping):
        raise TriangleScreeningError(f"{label} quality output must be an object")
    metrics = payload.get("metrics")
    if metrics != ["niqe", "sharpness"]:
        raise TriangleScreeningError(
            f"{label} quality output must contain only NIQE and sharpness"
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
    return per_sample["rows"]


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
    if isinstance(value, bool) or not isinstance(value, int):
        raise TriangleScreeningError(f"{label}.{field} must be an integer")
    return value


def _identity_cosines(
    row: Mapping[str, Any], *, label: str
) -> tuple[float | None, float | None]:
    counts = tuple(
        _integer(row, field, label)
        for field in (
            "source_face_count",
            "native_face_count",
            "candidate_face_count",
        )
    )
    if counts == (1, 1, 1):
        return (
            _finite(row, "source_native_cosine", label),
            _finite(row, "source_candidate_cosine", label),
        )
    if (
        row.get("source_native_cosine") is not None
        or row.get("source_candidate_cosine") is not None
    ):
        raise TriangleScreeningError(
            f"{label} must omit ArcFace cosines when any required role is not exact-one"
        )
    return None, None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize stage32 triangle rows from official evaluator outputs."
    )
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    diagnostic_path = args.diagnostic_manifest.resolve()
    diagnostic = _json(diagnostic_path)
    if (
        not isinstance(diagnostic, Mapping)
        or diagnostic.get("contract_type")
        != "safa_r10_triangle_fixed32_diagnostic_preparation_v1"
        or diagnostic.get("registration") != "new_diagnostic_not_r9_continuation"
    ):
        raise TriangleScreeningError("invalid fixed32 diagnostic manifest")
    arms = diagnostic.get("arms")
    if not isinstance(arms, list) or len(arms) != 4:
        raise TriangleScreeningError("fixed32 diagnostic must contain four arms")
    arm_ids = [arm.get("arm_id") for arm in arms if isinstance(arm, Mapping)]
    if arm_ids != [
        "eta0p125_baseline",
        "eta0p125_disable_i1",
        "eta0p125_disable_i2",
        "eta0p125_disable_i3",
    ]:
        raise TriangleScreeningError("fixed32 arm IDs/order disagree with the lock")

    selection_path = args.selection_manifest.resolve()
    selection_rows = _jsonl(selection_path)
    selection_ids = [row.get("sample_id") for row in selection_rows]
    if (
        len(selection_ids) != 32
        or len(set(selection_ids)) != 32
        or any(not isinstance(value, str) for value in selection_ids)
    ):
        raise TriangleScreeningError("selection manifest must contain 32 unique IDs")

    runs_root = args.runs_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    native_quality_path = evaluation_root / "quality" / "native" / "quality.json"
    native_quality = _indexed(
        _quality_rows(native_quality_path, "native"), "native quality"
    )
    if set(native_quality) != set(selection_ids):
        raise TriangleScreeningError("native quality rows do not match fixed32")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    materialized_arms: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for arm_id in arm_ids:
        run_path = runs_root / arm_id / "per_sample.jsonl"
        candidate_quality_path = (
            evaluation_root / "quality" / arm_id / "quality.json"
        )
        arcface_request_path = evaluation_root / "arcface" / arm_id / "request.json"
        arcface_result_path = evaluation_root / "arcface" / arm_id / "result.json"
        run_rows = _indexed(_jsonl(run_path), f"{arm_id} generation")
        candidate_quality = _indexed(
            _quality_rows(candidate_quality_path, arm_id), f"{arm_id} quality"
        )
        request = _json(arcface_request_path)
        result = _json(arcface_result_path)
        if not isinstance(request, Mapping):
            raise TriangleScreeningError(f"{arm_id} ArcFace request must be an object")
        config = request.get("config")
        detector = config.get("arcface") if isinstance(config, Mapping) else None
        if not isinstance(detector, Mapping) or detector.get("model_name") != "buffalo_l":
            raise TriangleScreeningError(
                f"{arm_id} ArcFace request must bind buffalo_l"
            )
        if not isinstance(result, Mapping) or not isinstance(result.get("result"), list):
            raise TriangleScreeningError(
                f"{arm_id} ArcFace result must be the official evaluator output"
            )
        arcface = _indexed(result["result"], f"{arm_id} ArcFace")
        expected = set(selection_ids)
        if any(set(rows) != expected for rows in (run_rows, candidate_quality, arcface)):
            raise TriangleScreeningError(
                f"{arm_id} evidence does not match the fixed32 manifest"
            )
        rows: list[dict[str, Any]] = []
        for sample_id in selection_ids:
            generated = run_rows[sample_id]
            native_q = native_quality[sample_id]
            candidate_q = candidate_quality[sample_id]
            identity = arcface[sample_id]
            source_native_cosine, source_candidate_cosine = _identity_cosines(
                identity, label=f"{arm_id} ArcFace"
            )
            rows.append(
                {
                    "arm_id": arm_id,
                    "stage": 32,
                    "sample_id": sample_id,
                    "native_e0": _finite(generated, "native_cosine", arm_id),
                    "candidate_e0": _finite(generated, "candidate_cosine", arm_id),
                    "native_edev": _finite(
                        generated, "native_edev_cosine", arm_id
                    ),
                    "candidate_edev": _finite(generated, "edev_cosine", arm_id),
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
        materialized_arms.append({"arm_id": arm_id, "rows_path": str(row_path)})
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
                    "stage": 32,
                    "baseline_arm_id": "eta0p125_baseline",
                    "selection_manifest_path": str(selection_path),
                    "arms": materialized_arms,
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
                    "contract_type": "safa_triangle_fixed32_rows_v1",
                    "metric_contract": {
                        "fid_kid": "absent_not_interpreted",
                        "quality": ["niqe", "sharpness"],
                        "privacy": "point_arcface_delta",
                        "arcface_detector": "buffalo_l",
                    },
                    "diagnostic_manifest_sha256": _sha256(diagnostic_path),
                    "selection_manifest_sha256": _sha256(selection_path),
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
