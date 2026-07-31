#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

from safa.evaluation.meanflow_guidance_runner import (
    locked_r11_nfe2_schedule_contract,
)
from safa.evaluation.schedule_nfe2 import (
    ScheduleNFE2Error,
    classify_schedule_nfe2,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_sha256(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    ).hexdigest()


def _json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ScheduleNFE2Error(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ScheduleNFE2Error(f"{label} must be an object")
    return value


def _rows(path: Path, label: str) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise ScheduleNFE2Error(f"{label} is missing: {path}")
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(not isinstance(row, Mapping) for row in values):
        raise ScheduleNFE2Error(f"{label} rows must be objects")
    return values


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


def _bound_path(binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) - {"path", "sha256", "expected_absent_at_preparation"}:
        raise ScheduleNFE2Error(f"{label} binding fields are not canonical")
    expected_absent = binding.get("expected_absent_at_preparation", False)
    if expected_absent not in (True, False):
        raise ScheduleNFE2Error(
            f"{label}.expected_absent_at_preparation must be boolean"
        )
    path = Path(str(binding.get("path"))).resolve()
    if not path.is_file():
        raise ScheduleNFE2Error(f"{label} is missing: {path}")
    declared = binding.get("sha256")
    if declared is not None and _sha256(path) != declared:
        raise ScheduleNFE2Error(f"{label} SHA256 differs")
    return path


def _quality(
    path: Path,
    *,
    dataset_id: str,
    sample_ids: Sequence[str],
    label: str,
    generation_result: Path | None,
) -> tuple[dict[str, float], dict[str, float] | None]:
    payload = _json(path, label)
    expected_metrics = (
        ["fid", "kid", "niqe", "sharpness"]
        if dataset_id == "prefix128"
        else ["niqe", "sharpness"]
    )
    if payload.get("metrics") != expected_metrics:
        raise ScheduleNFE2Error(
            f"{label} metrics must be exactly {expected_metrics!r}"
        )
    if dataset_id == "sharpness_tail32" and any(
        key in payload for key in ("fid", "kid", "kid_mean", "kid_std")
    ):
        raise ScheduleNFE2Error(f"{label} tail output contains FID/KID")
    if payload.get("sample_id_count") != len(sample_ids):
        raise ScheduleNFE2Error(f"{label} sample count differs")
    if payload.get("sample_id_sha256") != _id_sha256(sample_ids):
        raise ScheduleNFE2Error(f"{label} ordered sample SHA256 differs")
    per_sample = payload.get("per_sample_metrics")
    rows = per_sample.get("rows") if isinstance(per_sample, Mapping) else None
    if not isinstance(rows, list):
        raise ScheduleNFE2Error(f"{label} per-sample rows are missing")
    if [row.get("sample_id") for row in rows if isinstance(row, Mapping)] != list(
        sample_ids
    ):
        raise ScheduleNFE2Error(f"{label} sample order differs")
    niqe: list[float] = []
    sharpness: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "sample_id",
            "niqe",
            "sharpness",
        }:
            raise ScheduleNFE2Error(
                f"{label} row {index} fields are not canonical"
            )
        niqe.append(_finite(row["niqe"], f"{label}[{index}].niqe"))
        sharpness.append(
            _finite(row["sharpness"], f"{label}[{index}].sharpness")
        )
    quality_contract = payload.get("quality_contract")
    if not isinstance(quality_contract, Mapping):
        raise ScheduleNFE2Error(f"{label}.quality_contract is missing")
    if generation_result is None:
        if any(
            field in quality_contract
            for field in ("generation_result_sha256", "arm_config_sha256")
        ):
            raise ScheduleNFE2Error(
                f"{label} reused quality must not claim a generation result"
            )
    else:
        generation = _json(generation_result, f"{label} generation result")
        if quality_contract.get("generation_result_sha256") != _sha256(
            generation_result
        ):
            raise ScheduleNFE2Error(
                f"{label} generation-result SHA256 differs"
            )
        if quality_contract.get("arm_config_sha256") != generation.get(
            "arm_config_sha256"
        ):
            raise ScheduleNFE2Error(f"{label} arm SHA256 differs")
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
        raise ScheduleNFE2Error(f"{label} sample order differs")
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


def _validate_generation(
    path: Path,
    *,
    sample_ids: Sequence[str],
    dataset_id: str,
    prepared_config: Mapping[str, Any],
) -> None:
    result = _json(path, f"{dataset_id} NFE2 generation result")
    schedule = locked_r11_nfe2_schedule_contract()
    if result.get("status") != "complete" or result.get("sample_count") != len(
        sample_ids
    ):
        raise ScheduleNFE2Error(
            f"{dataset_id} NFE2 generation is not complete"
        )
    config = result.get("config")
    if not isinstance(config, Mapping):
        raise ScheduleNFE2Error(f"{dataset_id} NFE2 config is missing")
    if dict(config) != dict(prepared_config):
        raise ScheduleNFE2Error(
            f"{dataset_id} executed config differs from prepared config"
        )
    if (
        config.get("arm_name") != "schedule_nfe2"
        or config.get("sampling_seed") != 7919
        or config.get("batch_size") != 2
        or config.get("r11_nfe2_schedule_contract") != schedule
    ):
        raise ScheduleNFE2Error(
            f"{dataset_id} NFE2 generation config differs"
        )
    if result.get("nfe") != {
        "candidate": 2,
        "candidate_algorithm": 2,
        "candidate_diagnostic": 0,
        "matched_native": 0,
    }:
        raise ScheduleNFE2Error(f"{dataset_id} aggregate NFE differs")
    rows = _rows(
        Path(str(result["artifacts"]["per_sample_jsonl"])),
        f"{dataset_id} NFE2 rows",
    )
    if [row.get("sample_id") for row in rows] != list(sample_ids):
        raise ScheduleNFE2Error(f"{dataset_id} generation order differs")
    for index, row in enumerate(rows):
        if (
            row.get("candidate_algorithm_nfe") != 2
            or row.get("candidate_diagnostic_nfe") != 0
            or row.get("native_nfe") != 0
            or row.get("candidate_trace") != schedule["expected_algorithm_trace"]
            or row.get("candidate_diagnostic_trace") != []
        ):
            raise ScheduleNFE2Error(
                f"{dataset_id} generation row {index} NFE/trace differs"
            )


def _validate_reused_assets(
    path: Path,
    *,
    sample_ids: Sequence[str],
    dataset_id: str,
) -> str:
    rows = _rows(path, f"{dataset_id} reused asset bindings")
    if [row.get("sample_id") for row in rows] != list(sample_ids):
        raise ScheduleNFE2Error(f"{dataset_id} reused asset order differs")
    expected_fields = {
        "ordinal",
        "sample_id",
        "source",
        "source_sha256",
        "native",
        "native_sha256",
        "paper",
        "paper_sha256",
    }
    for ordinal, row in enumerate(rows):
        if set(row) != expected_fields or row["ordinal"] != ordinal:
            raise ScheduleNFE2Error(
                f"{dataset_id} reused asset row {ordinal} is not canonical"
            )
        for role in ("source", "native", "paper"):
            asset = Path(str(row[role]))
            if not asset.is_file() or _sha256(asset) != row[f"{role}_sha256"]:
                raise ScheduleNFE2Error(
                    f"{dataset_id} reused {role} asset {ordinal} differs"
                )
    return _sha256(path)


def materialize_evidence(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any], dict[str, Any]]:
    if (
        contract.get("schema_version") != 1
        or contract.get("contract_type")
        != "safa_r11_schedule_nfe2_diagnostic_v1"
        or not isinstance(contract.get("datasets"), list)
    ):
        raise ScheduleNFE2Error("NFE2 diagnostic contract identity differs")
    parent_binding = contract.get("parent_causal_classification")
    if not isinstance(parent_binding, Mapping):
        raise ScheduleNFE2Error("parent causal classification binding is missing")
    parent_path = _bound_path(parent_binding, "parent causal classification")
    parent = _json(parent_path, "parent causal classification")
    if (
        parent.get("classification") != "schedule_branch"
        or parent.get("candidate_promotion", {}).get("status") != "forbidden"
    ):
        raise ScheduleNFE2Error(
            "parent causal classification is not the locked schedule_branch"
        )

    evidence: dict[str, Mapping[str, Any]] = {}
    descriptive: dict[str, Any] = {}
    provenance: dict[str, Any] = {
        "parent_causal_classification": {
            "path": str(parent_path),
            "sha256": _sha256(parent_path),
            "classification": "schedule_branch",
        },
        "reuse_asset_bindings": {},
    }
    for dataset in contract["datasets"]:
        if not isinstance(dataset, Mapping):
            raise ScheduleNFE2Error("NFE2 dataset row must be an object")
        dataset_id = str(dataset.get("dataset_id"))
        selection_path = _bound_path(dataset["selection_manifest"], "selection")
        selection_rows = _rows(selection_path, f"{dataset_id} selection")
        sample_ids = [str(row.get("sample_id")) for row in selection_rows]
        if (
            len(sample_ids) != dataset.get("sample_count")
            or len(set(sample_ids)) != len(sample_ids)
            or _id_sha256(sample_ids) != dataset.get("ordered_sample_id_sha256")
        ):
            raise ScheduleNFE2Error(
                f"{dataset_id} selection count/order is invalid"
            )
        reused_path = _bound_path(
            dataset["reuse_asset_bindings"],
            f"{dataset_id} reused asset bindings",
        )
        provenance["reuse_asset_bindings"][dataset_id] = {
            "path": str(reused_path),
            "sha256": _validate_reused_assets(
                reused_path,
                sample_ids=sample_ids,
                dataset_id=dataset_id,
            ),
        }
        config_path = _bound_path(
            dataset["nfe2_config"],
            f"{dataset_id} prepared NFE2 config",
        )
        prepared_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(prepared_config, Mapping):
            raise ScheduleNFE2Error(
                f"{dataset_id} prepared NFE2 config must be a mapping"
            )
        generation_result = _bound_path(
            dataset["nfe2_generation_result"],
            f"{dataset_id} NFE2 generation result",
        )
        _validate_generation(
            generation_result,
            sample_ids=sample_ids,
            dataset_id=dataset_id,
            prepared_config=prepared_config,
        )
        provenance.setdefault("nfe2_configs", {})[dataset_id] = {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "arm_config_sha256": prepared_config.get("arm_config_sha256"),
        }
        roles = dataset.get("roles")
        if not isinstance(roles, Mapping) or set(roles) != {
            "native",
            "paper_eta_0p125",
            "schedule_nfe2",
        }:
            raise ScheduleNFE2Error(f"{dataset_id} role bindings differ")
        quality: dict[str, dict[str, float]] = {}
        representation: dict[str, dict[str, float]] = {}
        descriptive_roles: dict[str, dict[str, float]] = {}
        for role, binding in roles.items():
            if not isinstance(binding, Mapping) or set(binding) != {
                "quality_output",
                "representation_rows",
            }:
                raise ScheduleNFE2Error(
                    f"{dataset_id}.{role} binding fields differ"
                )
            quality[role], role_descriptive = _quality(
                _bound_path(binding["quality_output"], f"{dataset_id}.{role}.quality"),
                dataset_id=dataset_id,
                sample_ids=sample_ids,
                label=f"{dataset_id}.{role}.quality",
                generation_result=(
                    generation_result if role == "schedule_nfe2" else None
                ),
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
    return evidence, descriptive, provenance


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify the locked R11 coarse-schedule NFE2 diagnostic."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = _json(args.contract.resolve(), "NFE2 diagnostic contract")
    evidence, descriptive, provenance = materialize_evidence(contract)
    result = classify_schedule_nfe2(evidence)
    result["descriptive_metrics"] = descriptive
    result["provenance"] = provenance
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
    except ScheduleNFE2Error as exc:
        print(f"NFE2 classification failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
