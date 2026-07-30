#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from safa.evaluation.triangle32_evaluation import (
    LEGACY_ARM_IDS,
    load_arm_set,
    validate_generation_result,
)
from safa.evaluation.triangle_screening import TriangleScreeningError


ARM_IDS = LEGACY_ARM_IDS
REGISTERED_REAL_INDEX = (
    Path(__file__).resolve().parents[1]
    / "data/index/val_face_mixed_e14.jsonl"
).resolve()


def _rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise TriangleScreeningError(f"missing fixed32 rows: {path}")
    result: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise TriangleScreeningError(
                    f"{path}:{line_number}: row must be an object"
                )
            result.append(row)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare immutable fixed32 role views for NIQE/sharpness evaluation."
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--real-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm-set-manifest", type=Path)
    parser.add_argument("--native-per-sample", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root = args.runs_root.resolve()
    selection_path = args.selection_manifest.resolve()
    real_index_path = args.real_index.resolve()
    if real_index_path != REGISTERED_REAL_INDEX:
        raise TriangleScreeningError(
            f"real index must be the registered E14 index: {REGISTERED_REAL_INDEX}"
        )
    if not real_index_path.is_file():
        raise TriangleScreeningError(
            f"registered E14 real index is missing: {real_index_path}"
        )
    real_index_binding = {
        "path": str(real_index_path),
        "sha256": _sha256(real_index_path),
    }
    arm_set = load_arm_set(args.arm_set_manifest)
    if (
        arm_set.selection_manifest is not None
        and (
            selection_path != arm_set.selection_manifest
            or _sha256(selection_path) != arm_set.selection_manifest_sha256
        )
    ):
        raise TriangleScreeningError("selection manifest disagrees with arm-set lock")
    selection = _rows(selection_path)
    sample_ids = [row.get("sample_id") for row in selection]
    if (
        len(sample_ids) != 32
        or len(set(sample_ids)) != 32
        or any(not isinstance(value, str) for value in sample_ids)
    ):
        raise TriangleScreeningError("selection manifest must contain 32 unique IDs")

    run_paths = {
        arm_id: validate_generation_result(arm_set, runs_root, arm_id)
        for arm_id in arm_set.arm_ids
    }
    run_rows = {arm_id: _rows(path) for arm_id, path in run_paths.items()}
    for arm_id, rows in run_rows.items():
        if [row.get("sample_id") for row in rows] != sample_ids:
            raise TriangleScreeningError(
                f"{arm_id} per-sample order disagrees with fixed32"
            )
    native_source_path = (
        args.native_per_sample.resolve()
        if args.native_per_sample is not None
        else run_paths[ARM_IDS[0]]
    )
    baseline = _rows(native_source_path)
    if [row.get("sample_id") for row in baseline] != sample_ids:
        raise TriangleScreeningError("native per-sample order disagrees with fixed32")
    native_view: list[dict[str, Any]] = []
    for row in baseline:
        native = row.get("native")
        if not isinstance(native, str) or not native:
            raise TriangleScreeningError("baseline row is missing native path")
        view = dict(row)
        view["generated"] = native
        view["candidate"] = native
        view["mode"] = "native"
        native_view.append(view)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    native_path = output_dir / "native_per_sample.jsonl"
    with native_path.open("x", encoding="utf-8") as handle:
        for row in native_view:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest_path = output_dir / "quality_input_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": 2,
                    "contract_type": "safa_triangle_fixed32_quality_inputs_v2",
                    "metrics": ["niqe", "sharpness"],
                    "real_index": real_index_binding,
                    "selection_manifest": str(selection_path),
                    "selection_manifest_sha256": _sha256(selection_path),
                    "native": {
                        "per_sample": str(native_path),
                        "per_sample_sha256": _sha256(native_path),
                        "real_index": real_index_binding,
                        "role_binding": "generated_equals_existing_native",
                    },
                    "candidates": [
                        {
                            "arm_id": arm_id,
                            "per_sample": str(run_paths[arm_id]),
                            "per_sample_sha256": _sha256(run_paths[arm_id]),
                            "real_index": real_index_binding,
                            "role_binding": "existing_generated_candidate",
                        }
                        for arm_id in arm_set.arm_ids
                    ],
                    "arm_set_manifest": (
                        None
                        if arm_set.manifest_path is None
                        else {
                            "path": str(arm_set.manifest_path),
                            "sha256": _sha256(arm_set.manifest_path),
                        }
                    ),
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
        print(f"fixed32 quality input preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
