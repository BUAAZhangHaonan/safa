#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from safa.evaluation.triangle_screening import TriangleScreeningError


ARM_IDS = (
    "eta0p125_baseline",
    "eta0p125_disable_i1",
    "eta0p125_disable_i2",
    "eta0p125_disable_i3",
)


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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root = args.runs_root.resolve()
    selection_path = args.selection_manifest.resolve()
    selection = _rows(selection_path)
    sample_ids = [row.get("sample_id") for row in selection]
    if (
        len(sample_ids) != 32
        or len(set(sample_ids)) != 32
        or any(not isinstance(value, str) for value in sample_ids)
    ):
        raise TriangleScreeningError("selection manifest must contain 32 unique IDs")

    run_paths = {arm_id: runs_root / arm_id / "per_sample.jsonl" for arm_id in ARM_IDS}
    run_rows = {arm_id: _rows(path) for arm_id, path in run_paths.items()}
    for arm_id, rows in run_rows.items():
        if [row.get("sample_id") for row in rows] != sample_ids:
            raise TriangleScreeningError(
                f"{arm_id} per-sample order disagrees with fixed32"
            )
    baseline = run_rows["eta0p125_baseline"]
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
                    "schema_version": 1,
                    "contract_type": "safa_triangle_fixed32_quality_inputs_v1",
                    "metrics": ["niqe", "sharpness"],
                    "selection_manifest": str(selection_path),
                    "selection_manifest_sha256": _sha256(selection_path),
                    "native": {
                        "per_sample": str(native_path),
                        "per_sample_sha256": _sha256(native_path),
                        "role_binding": "generated_equals_existing_native",
                    },
                    "candidates": [
                        {
                            "arm_id": arm_id,
                            "per_sample": str(run_paths[arm_id]),
                            "per_sample_sha256": _sha256(run_paths[arm_id]),
                            "role_binding": "existing_generated_candidate",
                        }
                        for arm_id in ARM_IDS
                    ],
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
