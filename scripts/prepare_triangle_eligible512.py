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
    join_eligibility_evidence,
    select_eligible512,
)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise TriangleScreeningError(f"missing input: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TriangleScreeningError(f"invalid JSON in {path}") from exc


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise TriangleScreeningError(f"missing input: {path}")
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise TriangleScreeningError(
                    f"{path}:{line_number}: blank rows are forbidden"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_by_sample(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise TriangleScreeningError(
                f"{label} row {index} has invalid sample_id"
            )
        if sample_id in result:
            raise TriangleScreeningError(
                f"{label} has duplicate sample_id: {sample_id}"
            )
        result[sample_id] = row
    return result


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the authoritative R9 Full eligible512 triangle manifest."
    )
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    full_root = args.full_root.resolve()
    automatic_path = full_root / "automatic_evidence.json"
    arcface_path = full_root / "evaluator_evidence" / "arcface" / "winner.json"
    arcface_request_path = (
        full_root / "evaluator_runs" / "arcface" / "winner" / "request.json"
    )
    automatic = _read_json(automatic_path)
    arcface_evidence = _read_json(arcface_path)
    arcface_request = _read_json(arcface_request_path)
    if not all(
        isinstance(value, Mapping)
        for value in (automatic, arcface_evidence, arcface_request)
    ):
        raise TriangleScreeningError("authoritative R9 Full inputs must be objects")

    arcface_config = arcface_request.get("config")
    if not isinstance(arcface_config, Mapping):
        raise TriangleScreeningError("ArcFace request config is missing")
    detector = arcface_config.get("arcface")
    if (
        not isinstance(detector, Mapping)
        or detector.get("model_name") != "buffalo_l"
    ):
        raise TriangleScreeningError(
            "ArcFace request must bind model_name=buffalo_l"
        )
    arcface = arcface_evidence.get("arcface")
    if not isinstance(arcface, Mapping) or not isinstance(
        arcface.get("rows"), list
    ):
        raise TriangleScreeningError("ArcFace per-sample rows are missing")
    face_rows = _unique_by_sample(arcface["rows"], "ArcFace")

    manifest_contract = automatic.get("manifest")
    source_index_contract = automatic.get("source_index")
    arms = automatic.get("arms")
    if (
        not isinstance(manifest_contract, Mapping)
        or not isinstance(source_index_contract, Mapping)
        or not isinstance(arms, list)
        or len(arms) != 1
        or not isinstance(arms[0], Mapping)
        or arms[0].get("arm_id") != "paper_eta_0p125"
    ):
        raise TriangleScreeningError(
            "automatic evidence must bind one paper_eta_0p125 Full arm"
        )
    manifest_path = Path(str(manifest_contract.get("path"))).resolve()
    source_index_path = Path(str(source_index_contract.get("path"))).resolve()
    for path, contract, label in (
        (manifest_path, manifest_contract, "Full manifest"),
        (source_index_path, source_index_contract, "source index"),
    ):
        if _sha256(path) != contract.get("sha256"):
            raise TriangleScreeningError(f"{label} SHA256 mismatch")
    manifest_rows = _read_jsonl(manifest_path)
    manifest_ids = [row.get("sample_id") for row in manifest_rows]
    if (
        len(manifest_ids) != 2048
        or len(set(manifest_ids)) != 2048
        or any(not isinstance(sample_id, str) for sample_id in manifest_ids)
    ):
        raise TriangleScreeningError(
            "Full manifest must contain 2048 unique sample IDs"
        )
    if set(face_rows) != set(manifest_ids):
        raise TriangleScreeningError(
            "ArcFace rows do not map exactly to the Full manifest"
        )

    paired = arms[0].get("paired_metric_rows")
    if not isinstance(paired, Mapping) or not isinstance(paired.get("rows"), list):
        raise TriangleScreeningError("native sharpness rows are missing")
    paired_rows = _unique_by_sample(paired["rows"], "paired metrics")
    if set(paired_rows) != set(manifest_ids):
        raise TriangleScreeningError(
            "paired metric rows do not map exactly to the Full manifest"
        )
    source_rows = _unique_by_sample(_read_jsonl(source_index_path), "source index")
    if not set(manifest_ids).issubset(source_rows):
        raise TriangleScreeningError(
            "source index does not cover every Full manifest sample"
        )

    full_evidence_rows = [
        {
            "sample_id": sample_id,
            "source_detector": "buffalo_l",
            "native_detector": "buffalo_l",
            "source_face_count": face_rows[sample_id].get("source_face_count"),
            "native_face_count": face_rows[sample_id].get("native_face_count"),
        }
        for sample_id in manifest_ids
    ]
    label_rows = [
        {
            "sample_id": sample_id,
            "label": source_rows[sample_id].get("label"),
        }
        for sample_id in manifest_ids
    ]
    sharpness_rows = [
        {
            "sample_id": sample_id,
            "native_sharpness": paired_rows[sample_id].get("native_sharpness"),
        }
        for sample_id in manifest_ids
    ]
    joined = join_eligibility_evidence(
        full_evidence_rows, label_rows, sharpness_rows
    )
    eligible512 = select_eligible512(joined, expected_eligible_count=2045)
    eligible_pool = [
        row
        for row in joined
        if row["source_detector"] == "buffalo_l"
        and row["native_detector"] == "buffalo_l"
        and row["source_face_count"] == 1
        and row["native_face_count"] == 1
    ]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "eligible_pool": output_dir / "eligible_pool_2045.jsonl",
        "eligible512": output_dir / "eligible512.jsonl",
        "prefix32": output_dir / "prefix32.jsonl",
        "prefix128": output_dir / "prefix128.jsonl",
        "summary": output_dir / "summary.json",
    }
    _write_jsonl(outputs["eligible_pool"], eligible_pool)
    _write_jsonl(outputs["eligible512"], eligible512)
    _write_jsonl(outputs["prefix32"], eligible512[:32])
    _write_jsonl(outputs["prefix128"], eligible512[:128])
    summary = {
        "schema_version": 1,
        "contract_type": "safa_triangle_eligible512_preparation_v1",
        "selector": "safa-triangle-512-v1",
        "counts": {
            "full": 2048,
            "eligible": len(eligible_pool),
            "selected": len(eligible512),
            "prefix32": 32,
            "prefix128": 128,
        },
        "inputs": {
            "automatic_evidence": {
                "path": str(automatic_path),
                "sha256": _sha256(automatic_path),
            },
            "arcface_evidence": {
                "path": str(arcface_path),
                "sha256": _sha256(arcface_path),
            },
            "arcface_request": {
                "path": str(arcface_request_path),
                "sha256": _sha256(arcface_request_path),
            },
            "full_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "source_index": {
                "path": str(source_index_path),
                "sha256": _sha256(source_index_path),
            },
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in outputs.items()
            if name != "summary"
        },
    }
    with outputs["summary"].open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TriangleScreeningError as exc:
        print(f"triangle eligibility preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
