#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_r12_seed_aligned_trajectory as prep
from safa.evaluation.triangle32_evaluation import load_arm_set


class R12PreparationValidationError(RuntimeError):
    pass


def read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise R12PreparationValidationError(f"missing preparation artifact: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            R12PreparationValidationError(f"non-finite JSON value in {path}: {token}")
        ),
    )
    if not isinstance(value, Mapping):
        raise R12PreparationValidationError(f"preparation artifact is not an object: {path}")
    return value


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise R12PreparationValidationError(f"{label}: {actual!r} != {expected!r}")


def validate(preparation_root: Path = prep.DEFAULT_OUTPUT) -> dict[str, Any]:
    root = preparation_root.resolve()
    manifest = read_json(root / "preparation_manifest.json")
    require_equal(manifest.get("status"), "prepared_not_launched", "manifest status")
    require_equal(manifest.get("launch_status"), "not_started", "manifest launch status")

    generation_path = root / "generation_ledger.json"
    evaluation_path = root / "evaluation_ledger.json"
    selection_path = root / "selection_contract.json"
    generation = read_json(generation_path)
    evaluation = read_json(evaluation_path)
    selection = read_json(selection_path)
    require_equal(
        manifest.get("generation_ledger", {}).get("sha256"),
        prep.sha256(generation_path),
        "generation ledger digest",
    )
    require_equal(
        manifest.get("evaluation_ledger", {}).get("sha256"),
        prep.sha256(evaluation_path),
        "evaluation ledger digest",
    )
    require_equal(
        manifest.get("selection_contract", {}).get("sha256"),
        prep.sha256(selection_path),
        "selection contract digest",
    )

    jobs = generation.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 4:
        raise R12PreparationValidationError("generation ledger must contain exactly four jobs")
    expected_ids = [f"generate__{spec['arm_id']}" for spec in prep.ARMS]
    require_equal([row.get("job_id") for row in jobs], expected_ids, "generation job order")
    require_equal(
        [row.get("physical_gpu", {}).get("physical_index") for row in jobs],
        [0, 1, 2, 3],
        "generation GPU order",
    )
    require_equal([row.get("num_updates") for row in jobs], [12, 16, 12, 16], "update budgets")
    require_equal(
        [row.get("expected_candidate_nfe") for row in jobs],
        [13, 17, 13, 17],
        "candidate NFE budgets",
    )
    for spec, job in zip(prep.ARMS, jobs, strict=True):
        prep.validate_config(spec)
        gpu = int(spec["gpu_index"])
        require_equal(
            job.get("physical_gpu", {}).get("uuid"), prep.GPU_UUIDS[gpu], f"{job['job_id']} GPU UUID"
        )
        require_equal(job.get("environment", {}).get("CUDA_VISIBLE_DEVICES"), prep.GPU_UUIDS[gpu], f"{job['job_id']} CUDA visibility")
        require_equal(job.get("attempt_limit"), 1, f"{job['job_id']} attempt limit")
        require_equal(job.get("retry_count"), 0, f"{job['job_id']} retry count")
        output = Path(str(job.get("output_dir", "")))
        if output.exists():
            raise R12PreparationValidationError(
                f"prepared-not-launched output already exists: {output}"
            )

    contracts = {
        "regular32": ("u12_regular32", "u16_regular32"),
        "sharpness_tail32": ("u12_tail32", "u16_tail32"),
    }
    binding_counts: dict[str, int] = {}
    for dataset_id, expected_arms in contracts.items():
        contract_path = root / "evaluation_contracts" / f"{dataset_id}.json"
        arm_set = load_arm_set(contract_path)
        require_equal(arm_set.arm_ids, expected_arms, f"{dataset_id} arm order")
        declared = manifest.get("evaluation_contracts", {}).get(dataset_id, {})
        require_equal(declared.get("sha256"), prep.sha256(contract_path), f"{dataset_id} contract digest")

        binding_path = root / "formal_native_bindings" / f"{dataset_id}.jsonl"
        rows = prep.read_jsonl(binding_path, f"{dataset_id} formal native binding")
        ids = prep.ordered_ids(rows, f"{dataset_id} formal native binding")
        selected_ids = prep.ordered_ids(
            prep.read_jsonl(Path(prep.DATASETS[dataset_id]["selection"]), dataset_id), dataset_id
        )
        require_equal(ids, selected_ids, f"{dataset_id} binding order")
        for row in rows:
            linked = Path(str(row.get("generated", "")))
            formal = Path(str(row.get("formal_native", "")))
            declared_hash = row.get("formal_native_sha256")
            if not linked.is_file() or not formal.is_file():
                raise R12PreparationValidationError(
                    f"{dataset_id} native binding target is missing: {row.get('sample_id')}"
                )
            require_equal(prep.sha256(linked), declared_hash, f"{dataset_id} linked native hash")
            require_equal(prep.sha256(formal), declared_hash, f"{dataset_id} formal native hash")
        binding_counts[dataset_id] = len(rows)

    require_equal(evaluation.get("fid_kid_interpretation"), "forbidden", "stage32 FID/KID")
    quality_jobs = evaluation.get("quality_jobs")
    arcface_jobs = evaluation.get("arcface_jobs")
    if not isinstance(quality_jobs, list) or len(quality_jobs) != 6:
        raise R12PreparationValidationError("evaluation ledger must contain six quality jobs")
    if not isinstance(arcface_jobs, list) or len(arcface_jobs) != 4:
        raise R12PreparationValidationError("evaluation ledger must contain four ArcFace jobs")
    for job in quality_jobs:
        require_equal(job.get("metrics"), ["niqe", "sharpness"], f"{job.get('job_id')} metrics")
        require_equal(job.get("fid_kid_interpretation"), "forbidden", f"{job.get('job_id')} FID/KID")
        require_equal(job.get("attempt_limit"), 1, f"{job.get('job_id')} attempt limit")
        require_equal(job.get("retry_count"), 0, f"{job.get('job_id')} retry count")
    require_equal(
        [job.get("physical_gpu", {}).get("physical_index") for job in arcface_jobs],
        [0, 1, 2, 3],
        "ArcFace GPU order",
    )

    horizons = selection.get("paired_horizons")
    if not isinstance(horizons, list) or len(horizons) != 2:
        raise R12PreparationValidationError("selection contract must contain u12 and u16 only")
    require_equal([row.get("num_updates") for row in horizons], [12, 16], "selection horizons")
    require_equal([row.get("selection_eligible") for row in horizons], [True, True], "selection eligibility")
    require_equal(selection.get("pair_score"), "minimum_R_Q_P_across_regular32_and_tail32", "pair score")
    require_equal(selection.get("gates", {}).get("fid_kid"), "forbidden_at_stage32", "stage32 gates")
    require_equal(
        selection.get("legacy_tail_scope", {}).get("selection_basis"),
        "full_image_laplacian_sharpness",
        "legacy tail selection basis",
    )
    require_equal(
        selection.get("legacy_tail_scope", {}).get("post_hoc_roi_tail_reselection"),
        "forbidden",
        "post-hoc tail selection",
    )
    require_equal(selection.get("prefix_binding", {}).get("u12_loss_history_length"), 13, "u12 history length")
    require_equal(selection.get("prefix_binding", {}).get("u16_loss_history_length"), 17, "u16 history length")

    return {
        "status": "validated_prepared_not_launched",
        "generation_jobs": 4,
        "generation_gpu_indices": [0, 1, 2, 3],
        "update_budgets": [12, 16],
        "candidate_nfe_budgets": [13, 17],
        "binding_counts": binding_counts,
        "quality_metrics": ["niqe", "sharpness"],
        "fid_kid_interpretation": "forbidden",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R12 without launching any job.")
    parser.add_argument("--preparation-root", type=Path, default=prep.DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(validate(args.preparation_root), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
