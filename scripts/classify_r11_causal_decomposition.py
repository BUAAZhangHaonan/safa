#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from safa.evaluation.causal_decomposition import (
    CausalDecompositionError,
    classify_causal_decomposition,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise CausalDecompositionError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CausalDecompositionError(f"{label} must be an object")
    return value


def _rows(path: Path, label: str) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise CausalDecompositionError(f"{label} is missing: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(not isinstance(row, Mapping) for row in rows):
        raise CausalDecompositionError(f"{label} rows must be objects")
    return rows


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


def _bound_path(binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) - {"path", "sha256", "expected_absent_at_preparation"}:
        raise CausalDecompositionError(f"{label} binding fields are not canonical")
    path = Path(str(binding.get("path"))).resolve()
    declared = binding.get("sha256")
    expected_absent = binding.get("expected_absent_at_preparation", False)
    if expected_absent not in (True, False):
        raise CausalDecompositionError(
            f"{label}.expected_absent_at_preparation must be boolean"
        )
    if not path.is_file():
        raise CausalDecompositionError(f"{label} is missing: {path}")
    if declared is not None and _sha256(path) != declared:
        raise CausalDecompositionError(f"{label} SHA256 differs")
    return path


def _quality(
    path: Path,
    *,
    dataset_id: str,
    sample_ids: Sequence[str],
    label: str,
) -> tuple[dict[str, float], dict[str, float] | None]:
    payload = _json(path, label)
    expected_metrics = (
        ["fid", "kid", "niqe", "sharpness"]
        if dataset_id == "prefix128"
        else ["niqe", "sharpness"]
    )
    if payload.get("metrics") != expected_metrics:
        raise CausalDecompositionError(
            f"{label} metrics must be exactly {expected_metrics!r}"
        )
    if dataset_id == "sharpness_tail32" and any(
        key in payload for key in ("fid", "kid", "kid_mean", "kid_std")
    ):
        raise CausalDecompositionError(f"{label} tail output contains FID/KID")
    per_sample = payload.get("per_sample_metrics")
    rows = per_sample.get("rows") if isinstance(per_sample, Mapping) else None
    if not isinstance(rows, list):
        raise CausalDecompositionError(f"{label} per-sample rows are missing")
    if [row.get("sample_id") for row in rows if isinstance(row, Mapping)] != list(
        sample_ids
    ):
        raise CausalDecompositionError(f"{label} sample order differs")
    niqe: list[float] = []
    sharpness: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "sample_id",
            "niqe",
            "sharpness",
        }:
            raise CausalDecompositionError(
                f"{label} row {index} fields are not canonical"
            )
        niqe.append(_finite(row["niqe"], f"{label}[{index}].niqe"))
        sharpness.append(
            _finite(row["sharpness"], f"{label}[{index}].sharpness")
        )
    quality = {
        "niqe": sum(niqe) / len(niqe),
        "sharpness": sum(sharpness) / len(sharpness),
    }
    descriptive = None
    if dataset_id == "prefix128":
        descriptive = {
            "fid": _finite(payload.get("fid"), f"{label}.fid"),
            "kid_mean": _finite(payload.get("kid_mean"), f"{label}.kid_mean"),
            "kid_std": _finite(payload.get("kid_std"), f"{label}.kid_std"),
        }
    return quality, descriptive


def _representation(
    path: Path, *, sample_ids: Sequence[str], label: str
) -> dict[str, float]:
    rows = _rows(path, label)
    if [row.get("sample_id") for row in rows] != list(sample_ids):
        raise CausalDecompositionError(f"{label} sample order differs")
    e0 = []
    edev = []
    for index, row in enumerate(rows):
        e0.append(
            _finite(
                row.get("e0_cosine", row.get("candidate_cosine")),
                f"{label}[{index}].e0_cosine",
            )
        )
        edev.append(
            _finite(row.get("edev_cosine"), f"{label}[{index}].edev_cosine")
        )
    return {
        "e0_cosine": sum(e0) / len(e0),
        "edev_cosine": sum(edev) / len(edev),
    }


def materialize_evidence(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    if (
        contract.get("schema_version") != 1
        or contract.get("contract_type")
        != "safa_r11_transport_causal_decomposition_v1"
        or not isinstance(contract.get("datasets"), list)
    ):
        raise CausalDecompositionError("causal contract identity differs")
    evidence: dict[str, Mapping[str, Any]] = {}
    descriptive: dict[str, Any] = {}
    for dataset in contract["datasets"]:
        if not isinstance(dataset, Mapping):
            raise CausalDecompositionError("causal dataset row must be an object")
        dataset_id = str(dataset.get("dataset_id"))
        selection_path = _bound_path(dataset["selection_manifest"], "selection")
        selection_rows = _rows(selection_path, f"{dataset_id} selection")
        sample_ids = [str(row.get("sample_id")) for row in selection_rows]
        if len(sample_ids) != dataset.get("sample_count") or len(set(sample_ids)) != len(
            sample_ids
        ):
            raise CausalDecompositionError(
                f"{dataset_id} selection count/order is invalid"
            )
        roles = dataset.get("roles")
        if not isinstance(roles, Mapping) or set(roles) != {
            "native",
            "transport_only_nfe5",
            "paper_eta_0p125",
        }:
            raise CausalDecompositionError(f"{dataset_id} role bindings differ")
        quality: dict[str, dict[str, float]] = {}
        representation: dict[str, dict[str, float]] = {}
        descriptive_roles: dict[str, dict[str, float]] = {}
        for role, binding in roles.items():
            if not isinstance(binding, Mapping) or set(binding) != {
                "quality_output",
                "representation_rows",
            }:
                raise CausalDecompositionError(
                    f"{dataset_id}.{role} binding fields differ"
                )
            quality[role], role_descriptive = _quality(
                _bound_path(binding["quality_output"], f"{dataset_id}.{role}.quality"),
                dataset_id=dataset_id,
                sample_ids=sample_ids,
                label=f"{dataset_id}.{role}.quality",
            )
            representation[role] = _representation(
                _bound_path(
                    binding["representation_rows"],
                    f"{dataset_id}.{role}.representation",
                ),
                sample_ids=sample_ids,
                label=f"{dataset_id}.{role}.representation",
            )
            if role_descriptive is not None:
                descriptive_roles[role] = role_descriptive
        evidence[dataset_id] = {
            "sample_count": len(sample_ids),
            "quality": quality,
            "representation": representation,
        }
        if descriptive_roles:
            descriptive[dataset_id] = {
                "status": "descriptive_only_not_used_for_classification",
                "metrics": descriptive_roles,
            }
    return evidence, descriptive


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify the locked native/transport/paper causal decomposition."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = _json(args.contract.resolve(), "causal contract")
    evidence, descriptive = materialize_evidence(contract)
    result = classify_causal_decomposition(evidence)
    result["descriptive_metrics"] = descriptive
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace classifier output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CausalDecompositionError as exc:
        print(f"causal classification failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
