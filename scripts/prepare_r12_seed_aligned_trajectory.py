#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from safa.evaluation.meanflow_guidance_runner import (
    resolve_frozen_effective_guidance_config,
)
from safa.evaluation.r9_determinism import validate_r9_execution_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/r12_seed_aligned_trajectory/preparation_v1"
RUNS_ROOT = REPO_ROOT / "artifacts/r12_seed_aligned_trajectory/runs_v1"
EVALUATION_ROOT = REPO_ROOT / "artifacts/r12_seed_aligned_trajectory/evaluation_v1"
RESULTS_ROOT = REPO_ROOT / "artifacts/r12_seed_aligned_trajectory/results_v1"
INTERPRETER = Path("/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python")
REAL_INDEX = REPO_ROOT / "data/index/val_face_mixed_e14.jsonl"
ARCFACE_TEMPLATE = (
    REPO_ROOT
    / "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v9/full/evaluator_runs/arcface/winner/request.json"
)
GPU_UUIDS = (
    "GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
    "GPU-dfaeaa7c-32c8-ebb4-aa59-ab7f829805f1",
    "GPU-e27fe71d-eaf7-3eb5-d0ff-c1c63b4f6b02",
    "GPU-61ea2925-9905-7f56-cd64-7a792a32efef",
)
DATASETS = {
    "regular32": {
        "selection": REPO_ROOT
        / "artifacts/r10_triangle_exploration/preparation_v1/prefix32.jsonl",
        "formal_reuse": REPO_ROOT
        / "artifacts/r11_causal_decomposition/preparation_v1/reuse/"
        "prefix128/native_reuse.jsonl",
        "arms": ("u12_regular32", "u16_regular32"),
    },
    "sharpness_tail32": {
        "selection": REPO_ROOT
        / "artifacts/r11_initial_noise_sharpness_probe/preparation_v1/tail32.jsonl",
        "formal_reuse": REPO_ROOT
        / "artifacts/r11_causal_decomposition/preparation_v1/reuse/"
        "sharpness_tail32/native_reuse.jsonl",
        "arms": ("u12_tail32", "u16_tail32"),
    },
}
ARMS = (
    {
        "arm_id": "u12_regular32",
        "dataset_id": "regular32",
        "num_updates": 12,
        "gpu_index": 0,
    },
    {
        "arm_id": "u16_regular32",
        "dataset_id": "regular32",
        "num_updates": 16,
        "gpu_index": 1,
    },
    {
        "arm_id": "u12_tail32",
        "dataset_id": "sharpness_tail32",
        "num_updates": 12,
        "gpu_index": 2,
    },
    {
        "arm_id": "u16_tail32",
        "dataset_id": "sharpness_tail32",
        "num_updates": 16,
        "gpu_index": 3,
    },
)


class R12PreparationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise R12PreparationError(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise R12PreparationError(f"{label} has a blank row at {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise R12PreparationError(f"{label} row {line_number} is not an object")
        rows.append(value)
    return rows


def ordered_ids(rows: Sequence[Mapping[str, Any]], label: str) -> list[str]:
    result = [row.get("sample_id") for row in rows]
    if (
        len(result) != 32
        or any(not isinstance(value, str) or not value for value in result)
        or len(set(result)) != 32
    ):
        raise R12PreparationError(f"{label} must contain 32 unique ordered IDs")
    return [str(value) for value in result]


def ordered_id_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{sample_id}\n" for sample_id in ids).encode()).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _config_path(arm_id: str) -> Path:
    return REPO_ROOT / "configs/medium_v2/experiments" / f"r12_initial_noise_fixed_eta05_{arm_id}.yaml"


def validate_config(spec: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    arm_id = str(spec["arm_id"])
    path = _config_path(arm_id)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise R12PreparationError(f"{arm_id} config is not an object")
    selection = Path(str(DATASETS[str(spec["dataset_id"])]["selection"]))
    expected = {
        "experiment_contract": "safa_r9_meanflow_v1",
        "mode": "initial_noise",
        "phase": "diagnose",
        "projection": "fixed_radius",
        "eta": 0.5,
        "num_updates": int(spec["num_updates"]),
        "seed": 7919,
        "sampling_seed": 7919,
        "attention_backend": "native",
        "batch_size": 2,
        "max_samples": 32,
        "contact_sheets": False,
        "sample_id_manifest": str(selection.relative_to(REPO_ROOT)),
        "sample_id_manifest_sha256": sha256(selection),
        "quality_metrics": ["niqe", "sharpness"],
        "calibration_metrics": ["e0_cosine", "edev_cosine", "niqe", "sharpness"],
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise R12PreparationError(
                f"{arm_id}.{field} differs: {value.get(field)!r} != {expected_value!r}"
            )
    policy = value.get("determinism_policy")
    if not isinstance(policy, Mapping) or value.get("determinism_policy_sha256") != (
        "ea6a4e81627a993066d9b1a3ca4ae791a0bcb3e21e399a5d2cb27811aa22147f"
    ):
        raise R12PreparationError(f"{arm_id} strict determinism policy differs")
    effective = resolve_frozen_effective_guidance_config(value)
    execution_contract = validate_r9_execution_config(effective)
    if execution_contract.get("attention_backend") != "native":
        raise R12PreparationError(f"{arm_id} effective attention backend is not native")
    if effective.get("r9_phase_contract", {}).get("edev_required") is not True:
        raise R12PreparationError(f"{arm_id} effective Edev scoring is disabled")
    return path, effective


def build_native_binding(
    *, dataset_id: str, output_root: Path
) -> tuple[Path, list[dict[str, Any]]]:
    dataset = DATASETS[dataset_id]
    selection_ids = ordered_ids(read_jsonl(Path(dataset["selection"]), dataset_id), dataset_id)
    formal_rows = read_jsonl(Path(dataset["formal_reuse"]), f"{dataset_id} formal reuse")
    by_id = {row.get("sample_id"): row for row in formal_rows}
    if len(by_id) != len(formal_rows) or any(sample_id not in by_id for sample_id in selection_ids):
        raise R12PreparationError(f"{dataset_id} formal native IDs differ")
    native_dir = output_root / "formal_native" / dataset_id
    native_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for ordinal, sample_id in enumerate(selection_ids):
        source_row = by_id[sample_id]
        formal_path = Path(str(source_row.get("generated", ""))).resolve()
        declared = source_row.get("generated_sha256")
        if not formal_path.is_file() or sha256(formal_path) != declared:
            raise R12PreparationError(f"{dataset_id} formal native differs: {sample_id}")
        for field in ("e0_cosine", "edev_cosine"):
            value = source_row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise R12PreparationError(f"{dataset_id} {sample_id} invalid {field}")
        linked = native_dir / f"{ordinal:08d}.png"
        os.symlink(formal_path, linked)
        rows.append(
            {
                "sample_id": sample_id,
                "ordinal": ordinal,
                "source": source_row.get("source"),
                "generated": str(linked),
                "formal_native": str(formal_path),
                "formal_native_sha256": str(declared),
                "e0_cosine": float(source_row["e0_cosine"]),
                "edev_cosine": float(source_row["edev_cosine"]),
            }
        )
    binding_path = output_root / "formal_native_bindings" / f"{dataset_id}.jsonl"
    write_jsonl(binding_path, rows)
    return binding_path, rows


def _generation_job(spec: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    arm_id = str(spec["arm_id"])
    gpu = int(spec["gpu_index"])
    output = RUNS_ROOT / arm_id
    log = REPO_ROOT / "artifacts/r12_seed_aligned_trajectory/logs_v1" / f"{arm_id}.log"
    return {
        "job_id": f"generate__{arm_id}",
        "kind": "generation",
        "arm_id": arm_id,
        "dataset_id": spec["dataset_id"],
        "num_updates": spec["num_updates"],
        "expected_candidate_nfe": int(spec["num_updates"]) + 1,
        "physical_gpu": {"physical_index": gpu, "uuid": GPU_UUIDS[gpu]},
        "logical_device": "cuda:0",
        "environment": {"CUDA_VISIBLE_DEVICES": GPU_UUIDS[gpu]},
        "tmux_session": f"safa-r12-{arm_id.replace('_', '-')}",
        "argv": [
            str(INTERPRETER),
            "scripts/run_meanflow_flow_map_guidance.py",
            "--config",
            str(config_path.relative_to(REPO_ROOT)),
            "--output-dir",
            str(output.relative_to(REPO_ROOT)),
            "--shard-index",
            "0",
            "--num-shards",
            "1",
        ],
        "output_dir": str(output),
        "log_path": str(log),
        "attempt_limit": 1,
        "retry_count": 0,
    }


def _quality_job(
    *, dataset_id: str, role: str, gpu: int, binding_path: Path | None = None
) -> dict[str, Any]:
    selection = Path(DATASETS[dataset_id]["selection"])
    output = EVALUATION_ROOT / dataset_id / "quality" / role / "quality.json"
    log = EVALUATION_ROOT / dataset_id / "logs" / f"quality__{role}.log"
    if role == "native":
        assert binding_path is not None
        generated_dir = binding_path.parent.parent / "formal_native" / dataset_id
        per_sample = binding_path
        generation_result = None
    else:
        generated_dir = RUNS_ROOT / role / "generated_images"
        per_sample = RUNS_ROOT / role / "per_sample.jsonl"
        generation_result = RUNS_ROOT / role / "generation_result.json"
    argv = [
        str(INTERPRETER),
        "scripts/run_r11_quality_evaluation.py",
        "--real-index",
        str(REAL_INDEX),
        "--generated-dir",
        str(generated_dir),
        "--output",
        str(output),
        "--sample-id-manifest",
        str(selection),
        "--per-sample-jsonl",
        str(per_sample),
        "--seed",
        "7919",
        "--device",
        "cuda:0",
        "--metrics",
        "niqe",
        "sharpness",
    ]
    if generation_result is not None:
        argv.extend(("--generation-result", str(generation_result)))
    return {
        "job_id": f"quality__{dataset_id}__{role}",
        "kind": "quality",
        "dataset_id": dataset_id,
        "role": role,
        "physical_gpu": {"physical_index": gpu, "uuid": GPU_UUIDS[gpu]},
        "environment": {"CUDA_VISIBLE_DEVICES": GPU_UUIDS[gpu]},
        "argv": argv,
        "output_path": str(output),
        "log_path": str(log),
        "metrics": ["niqe", "sharpness"],
        "fid_kid_interpretation": "forbidden",
        "attempt_limit": 1,
        "retry_count": 0,
    }


def prepare(output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace R12 preparation: {output_root}")
    output_root.mkdir(parents=True)
    bindings: dict[str, Path] = {}
    selection_ids: dict[str, list[str]] = {}
    for dataset_id, dataset in DATASETS.items():
        binding_path, _ = build_native_binding(dataset_id=dataset_id, output_root=output_root)
        bindings[dataset_id] = binding_path
        selection_ids[dataset_id] = ordered_ids(
            read_jsonl(Path(dataset["selection"]), dataset_id), dataset_id
        )

    config_rows = []
    generation_jobs = []
    for spec in ARMS:
        config_path, effective = validate_config(spec)
        config_rows.append(
            {
                **dict(spec),
                "path": str(config_path),
                "sha256": sha256(config_path),
                "effective_arm_config_sha256": effective["arm_config_sha256"],
            }
        )
        generation_jobs.append(_generation_job(spec, config_path))

    generation_ledger_path = output_root / "generation_ledger.json"
    write_json(
        generation_ledger_path,
        {
            "schema_version": 1,
            "contract_type": "safa_r12_seed_aligned_generation_ledger_v1",
            "attempt_limit": 1,
            "retry_count": 0,
            "jobs": generation_jobs,
        },
    )

    dataset_contracts: dict[str, Path] = {}
    for dataset_id, dataset in DATASETS.items():
        ids = selection_ids[dataset_id]
        selection = Path(dataset["selection"])
        arms = [row for row in config_rows if row["dataset_id"] == dataset_id]
        contract_path = output_root / "evaluation_contracts" / f"{dataset_id}.json"
        write_json(
            contract_path,
            {
                "schema_version": 1,
                "contract_type": "safa_r12_seed_aligned_evaluation_dataset_v1",
                "registration": "seed_aligned_fixed_radius_trajectory",
                "selection_role": dataset_id,
                "sample_count": 32,
                "stage": 32,
                "quality_metrics": ["niqe", "sharpness"],
                "sampling_seed": 7919,
                "selection_manifest": {
                    "path": str(selection),
                    "sha256": sha256(selection),
                    "sample_count": 32,
                    "ordered_sample_id_sha256": ordered_id_digest(ids),
                },
                "baseline_arm_id": str(dataset["arms"][0]),
                "arms": arms,
            },
        )
        dataset_contracts[dataset_id] = contract_path

    selection_contract_path = output_root / "selection_contract.json"
    write_json(
        selection_contract_path,
        {
            "schema_version": 1,
            "contract_type": "safa_r12_seed_aligned_selection_v1",
            "stage": 32,
            "gates": {
                "source_native_candidate_exact_one": "32/32",
                "arcface_point_delta_max": 0.02,
                "e0_min": 0.75,
                "delta_e0_min": 0.30,
                "delta_edev_min": 0.05,
                "niqe_max": "native+0.10",
                "sharpness_min": "max(300,0.95*native)",
                "fid_kid": "forbidden_at_stage32",
            },
            "paired_horizons": [
                {
                    "num_updates": 12,
                    "regular_arm_id": "u12_regular32",
                    "tail_arm_id": "u12_tail32",
                    "selection_eligible": True,
                    "role": "natural_75_percent_update_budget",
                    "preregistered_reason": (
                        "12/16 updates is the natural 75% compute horizon and historical "
                        "aggregate E0 clears 0.75 by more than 0.02 on both datasets"
                    ),
                    "update_budget_reduction_vs_u16": 0.25,
                    "candidate_nfe_reduction_vs_u16": 4.0 / 17.0,
                },
                {
                    "num_updates": 16,
                    "regular_arm_id": "u16_regular32",
                    "tail_arm_id": "u16_tail32",
                    "selection_eligible": True,
                    "role": "aligned_original_horizon_comparator",
                },
            ],
            "pair_score": "minimum_R_Q_P_across_regular32_and_tail32",
            "advance_rule": "one_horizon_may_advance_only_if_both_datasets_pass_all_gates",
            "legacy_tail_scope": {
                "selection_basis": "full_image_laplacian_sharpness",
                "known_confound": "background_high_frequency_content",
                "allowed_claim": "full_image_sharpness_gate_recovery_or_failure",
                "forbidden_claim_without_agreeing_roi_evidence": "face_detail_restoration",
                "post_hoc_roi_tail_reselection": "forbidden",
            },
            "outcomes": [
                "early_stop_quality_recovery",
                "early_stop_gate_recovery",
                "early_stop_not_needed_at32",
                "full_horizon_required",
                "early_stop_representation_limited",
                "initial_noise_quality_limited",
                "tail_fragility",
                "face_or_privacy_limited",
                "protocol_binding_failure",
            ],
            "prefix_binding": {
                "same_ordered_sample_ids_and_batch_pairs": True,
                "native_bytes_equal_across_horizons_and_formal_seed7919": True,
                "u12_loss_history_length": 13,
                "u16_loss_history_length": 17,
                "u12_candidate_nfe": 13,
                "u16_candidate_nfe": 17,
                "u12_loss_history_equals_u16_prefix": True,
                "mismatch_action": "protocol_binding_failure_no_metric_interpretation",
            },
            "hard_stop": {
                "no_additional_update_horizon": True,
                "no_additional_eta": True,
                "no_training": True,
            "no_128_512_full_without_selected_pair_survivor": True,
            },
        },
    )

    request_builds = []
    for dataset_id, contract_path in dataset_contracts.items():
        request_builds.append(
            {
                "job_id": f"arcface_request_build__{dataset_id}",
                "kind": "cpu_preparation",
                "argv": [
                    str(INTERPRETER),
                    "scripts/build_fixed32_arcface_requests.py",
                    "--diagnostic-manifest",
                    str(contract_path),
                    "--selection-manifest",
                    str(DATASETS[dataset_id]["selection"]),
                    "--runs-root",
                    str(RUNS_ROOT),
                    "--template-request",
                    str(ARCFACE_TEMPLATE),
                    "--output-root",
                    str(EVALUATION_ROOT / dataset_id / "arcface"),
                    "--device",
                    "cuda:0",
                ],
                "attempt_limit": 1,
                "retry_count": 0,
            }
        )

    quality_jobs = [
        _quality_job(dataset_id="regular32", role="native", gpu=0, binding_path=bindings["regular32"]),
        _quality_job(dataset_id="regular32", role="u12_regular32", gpu=1),
        _quality_job(dataset_id="regular32", role="u16_regular32", gpu=2),
        _quality_job(dataset_id="sharpness_tail32", role="native", gpu=3, binding_path=bindings["sharpness_tail32"]),
        _quality_job(dataset_id="sharpness_tail32", role="u12_tail32", gpu=0),
        _quality_job(dataset_id="sharpness_tail32", role="u16_tail32", gpu=1),
    ]
    arcface_jobs = []
    for gpu, spec in enumerate(ARMS):
        dataset_id = str(spec["dataset_id"])
        arm_id = str(spec["arm_id"])
        request = EVALUATION_ROOT / dataset_id / "arcface" / arm_id / "request.json"
        result = EVALUATION_ROOT / dataset_id / "arcface" / arm_id / "result.json"
        arcface_jobs.append(
            {
                "job_id": f"arcface__{dataset_id}__{arm_id}",
                "kind": "arcface",
                "dataset_id": dataset_id,
                "role": arm_id,
                "physical_gpu": {"physical_index": gpu, "uuid": GPU_UUIDS[gpu]},
                "environment": {"CUDA_VISIBLE_DEVICES": GPU_UUIDS[gpu]},
                "argv": [
                    str(INTERPRETER),
                    "scripts/run_r9_phase_evaluator.py",
                    "--request",
                    str(request),
                    "--output",
                    str(result),
                ],
                "request_path": str(request),
                "output_path": str(result),
                "log_path": str(EVALUATION_ROOT / dataset_id / "logs" / f"arcface__{arm_id}.log"),
                "attempt_limit": 1,
                "retry_count": 0,
            }
        )

    evaluation_ledger_path = output_root / "evaluation_ledger.json"
    write_json(
        evaluation_ledger_path,
        {
            "schema_version": 1,
            "contract_type": "safa_r12_seed_aligned_evaluation_ledger_v1",
            "request_build_jobs": request_builds,
            "quality_jobs": quality_jobs,
            "arcface_jobs": arcface_jobs,
            "postprocess_job": {
                "argv": [
                    str(INTERPRETER),
                    "scripts/classify_r12_seed_aligned_trajectory.py",
                    "--preparation-root",
                    str(output_root),
                    "--output-dir",
                    str(RESULTS_ROOT),
                ],
                "retry_count": 0,
            },
            "fid_kid_interpretation": "forbidden",
            "attempt_limit": 1,
            "retry_count": 0,
        },
    )

    manifest = {
        "schema_version": 1,
        "contract_type": "safa_r12_seed_aligned_preparation_v1",
        "status": "prepared_not_launched",
        "scientific_question": (
            "seed-aligned natural 75%-update u12 horizon versus original u16 horizon"
        ),
        "fixed_semantics": {
            "checkpoint": "E15",
            "mode": "initial_noise",
            "projection": "fixed_radius",
            "eta": 0.5,
            "seed": 7919,
            "sampling_seed": 7919,
            "attention_backend": "native",
            "strict_r9_determinism": True,
            "batch_size": 2,
            "sample_count_per_job": 32,
            "retry_count": 0,
        },
        "configs": config_rows,
        "formal_native_bindings": {
            dataset_id: {
                "path": str(path),
                "sha256": sha256(path),
                "sample_count": 32,
                "required_post_generation_check": "all regenerated native bytes equal formal native",
            }
            for dataset_id, path in bindings.items()
        },
        "generation_ledger": {"path": str(generation_ledger_path), "sha256": sha256(generation_ledger_path)},
        "evaluation_ledger": {"path": str(evaluation_ledger_path), "sha256": sha256(evaluation_ledger_path)},
        "selection_contract": {"path": str(selection_contract_path), "sha256": sha256(selection_contract_path)},
        "evaluation_contracts": {
            dataset_id: {"path": str(path), "sha256": sha256(path)}
            for dataset_id, path in dataset_contracts.items()
        },
        "launch_status": "not_started",
    }
    write_json(output_root / "preparation_manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the bounded R12 seed-aligned trajectory matrix.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = prepare(args.output_root)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
