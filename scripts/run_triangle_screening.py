#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from safa.evaluation.triangle_screening import (
    TriangleScreeningError,
    evaluate_arms,
    join_eligibility_evidence,
    select_eligible512,
    write_outputs,
)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise TriangleScreeningError(f"missing input: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TriangleScreeningError(f"invalid JSON in {path}: {exc}") from exc


def _read_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise TriangleScreeningError(f"missing per-sample input: {path}")
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise TriangleScreeningError(
                    f"{path}:{line_number}: blank JSONL rows are forbidden"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TriangleScreeningError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, Mapping):
                raise TriangleScreeningError(
                    f"{path}:{line_number}: row must be an object"
                )
            rows.append(row)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic SAFA triangle screening gate."
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    request = _read_json(args.request.resolve())
    if not isinstance(request, Mapping):
        raise TriangleScreeningError("request must be a JSON object")
    required = ("stage", "baseline_arm_id", "arms")
    if request.get("stage") != 32:
        required += ("native_fid", "native_kid")
    missing = [field for field in required if field not in request]
    if missing:
        raise TriangleScreeningError(
            f"request is missing fields: {', '.join(missing)}"
        )
    arms = request["arms"]
    if not isinstance(arms, list):
        raise TriangleScreeningError("request.arms must be a list")
    if request["stage"] == 32 and any(
        field in request for field in ("native_fid", "native_kid")
    ):
        raise TriangleScreeningError(
            "stage32 requests forbid native FID/KID fields"
        )
    materialized: list[dict[str, Any]] = []
    request_root = args.request.resolve().parent
    selection_manifest = None
    if "selection_manifest_path" in request:
        selection_path = Path(request["selection_manifest_path"])
        if not selection_path.is_absolute():
            selection_path = request_root / selection_path
        selection_path = selection_path.resolve()
        selection_manifest = _read_rows(selection_path)
        if len(selection_manifest) != request["stage"]:
            raise TriangleScreeningError(
                "locked selection manifest count must equal the screening stage"
            )
        declared_sha256 = request.get("selection_manifest_sha256")
        if (
            not isinstance(declared_sha256, str)
            or hashlib.sha256(selection_path.read_bytes()).hexdigest()
            != declared_sha256
        ):
            raise TriangleScreeningError("locked selection manifest SHA256 differs")
    elif request["stage"] != 8:
        eligibility = request.get("eligibility")
        if not isinstance(eligibility, Mapping):
            raise TriangleScreeningError(
                "stages 32/128/512 require request.eligibility"
            )
        eligibility_fields = (
            "full_evidence_path",
            "affectnet_labels_path",
            "native_sharpness_path",
        )
        eligibility_missing = [
            field for field in eligibility_fields if field not in eligibility
        ]
        if eligibility_missing:
            raise TriangleScreeningError(
                "request.eligibility is missing fields: "
                + ", ".join(eligibility_missing)
            )
        paths = []
        for field in eligibility_fields:
            value = eligibility[field]
            if not isinstance(value, str) or not value:
                raise TriangleScreeningError(
                    f"request.eligibility.{field} must be a non-empty string"
                )
            path = Path(value)
            paths.append(path if path.is_absolute() else request_root / path)
        joined = join_eligibility_evidence(
            _read_rows(paths[0].resolve()),
            _read_rows(paths[1].resolve()),
            _read_rows(paths[2].resolve()),
        )
        selected512 = select_eligible512(
            joined,
            expected_eligible_count=eligibility.get("expected_eligible_count", 2045),
        )
        selection_manifest = selected512[: request["stage"]]
    for index, arm in enumerate(arms):
        if not isinstance(arm, Mapping):
            raise TriangleScreeningError(f"request.arms[{index}] must be an object")
        if request["stage"] == 32 and any(
            field in arm for field in ("fid", "kid")
        ):
            raise TriangleScreeningError(
                "stage32 arm requests forbid FID/KID fields"
            )
        if request["stage"] != 32 and any(
            field not in arm for field in ("fid", "kid")
        ):
            raise TriangleScreeningError(
                "non-stage32 arm requests require FID/KID fields"
            )
        row_path_value = arm.get("rows_path")
        if not isinstance(row_path_value, str) or not row_path_value:
            raise TriangleScreeningError(
                f"request.arms[{index}].rows_path must be a non-empty string"
            )
        row_path = Path(row_path_value)
        if not row_path.is_absolute():
            row_path = request_root / row_path
        materialized.append(
            {
                "arm_id": arm.get("arm_id"),
                "fid": arm.get("fid"),
                "kid": arm.get("kid"),
                "rows": _read_rows(row_path.resolve()),
            }
        )
    results = evaluate_arms(
        materialized,
        stage=request["stage"],
        native_fid=request.get("native_fid"),
        native_kid=request.get("native_kid"),
        baseline_arm_id=request["baseline_arm_id"],
        expected_sample_ids=(
            None
            if selection_manifest is None
            else [row["sample_id"] for row in selection_manifest]
        ),
    )
    write_outputs(
        args.output_dir,
        results,
        stage=request["stage"],
        baseline_arm_id=request["baseline_arm_id"],
        selection_manifest=selection_manifest,
        selection_metadata=(
            None
            if "selection_role" not in request
            else {
                "selection_role": request["selection_role"],
                "manifest_sha256": request["selection_manifest_sha256"],
            }
        ),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TriangleScreeningError as exc:
        print(f"triangle screening failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
