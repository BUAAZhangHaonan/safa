#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from safa.evaluation.meanflow_guidance_runner import (
    R11_NFE2_SCHEDULE_CONTRACT_FIELD,
    R11_SCHEDULE_NFE2_CONTRACT_TYPE,
    R9_GUIDANCE_INTERVAL_CONTRACT_FIELD,
    R9_PHASE_CONTRACT_FIELD,
    locked_r11_nfe2_schedule_contract,
    resolve_frozen_effective_guidance_config,
)


REPO = Path(__file__).resolve().parents[1]
PYTHON = "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"
PARENT_ROOT = Path("artifacts/r11_causal_decomposition/preparation_v1")
FORMAL_SHARDS = Path(
    "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v9/full/winner/shards"
)
GPU_UUIDS = (
    "GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
    "GPU-dfaeaa7c-32c8-ebb4-aa59-ab7f829805f1",
    "GPU-e27fe71d-eaf7-3eb5-d0ff-c1c63b4f6b02",
    "GPU-61ea2925-9905-7f56-cd64-7a792a32efef",
)
DATASETS = (
    {
        "dataset_id": "prefix128",
        "sample_count": 128,
        "selection": Path(
            "artifacts/r10_triangle_exploration/preparation_v1/prefix128.jsonl"
        ),
        "parent_config": Path(
            "configs/medium_v2/experiments/"
            "r11_transport_only_nfe5_prefix128.yaml"
        ),
        "gpu_index": 0,
    },
    {
        "dataset_id": "sharpness_tail32",
        "sample_count": 32,
        "selection": Path(
            "artifacts/r11_initial_noise_sharpness_probe/preparation_v1/tail32.jsonl"
        ),
        "parent_config": Path(
            "configs/medium_v2/experiments/"
            "r11_transport_only_nfe5_sharpness_tail32.yaml"
        ),
        "gpu_index": 1,
    },
)


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


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


def _rows(path: Path) -> list[dict[str, Any]]:
    resolved = _absolute(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{resolved}:{line_number}: row is not an object")
        rows.append(value)
    return rows


def _json(path: Path) -> dict[str, Any]:
    resolved = _absolute(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain an object")
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n"
            )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)


def _binding(path: Path, *, exists: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path)}
    if exists:
        resolved = _absolute(path)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        result["sha256"] = _sha256(resolved)
    else:
        result["expected_absent_at_preparation"] = True
    return result


def _asset(
    path_value: Any,
    declared_sha256: Any,
    *,
    label: str,
    require_formal_target: bool,
) -> tuple[str, str]:
    raw = _absolute(Path(str(path_value)))
    resolved = raw.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    if "seed1337" in str(raw).lower() or "seed_1337" in str(raw).lower():
        raise ValueError(f"{label} points to forbidden seed-1337 evidence")
    digest = _sha256(resolved)
    if digest != declared_sha256:
        raise ValueError(f"{label} SHA256 differs")
    if require_formal_target:
        formal_root = _absolute(FORMAL_SHARDS).resolve()
        if formal_root not in resolved.parents:
            raise ValueError(
                f"{label} must resolve under the R9 formal winner shards"
            )
    return str(resolved), digest


def _reuse_bindings(dataset_id: str, sample_ids: Sequence[str]) -> list[dict[str, Any]]:
    parent_dataset = PARENT_ROOT / "reuse" / dataset_id
    external = _rows(parent_dataset / "native_external.jsonl")
    native = _rows(parent_dataset / "native_reuse.jsonl")
    paper = _rows(parent_dataset / "paper_reuse.jsonl")
    for label, rows in (
        ("external native", external),
        ("native reuse", native),
        ("paper reuse", paper),
    ):
        if [row.get("sample_id") for row in rows] != list(sample_ids):
            raise ValueError(f"{dataset_id} {label} sample order differs")
    bindings: list[dict[str, Any]] = []
    for ordinal, sample_id in enumerate(sample_ids):
        external_row = external[ordinal]
        native_row = native[ordinal]
        paper_row = paper[ordinal]
        source, source_sha = _asset(
            external_row["source"],
            external_row["source_sha256"],
            label=f"{dataset_id} source {ordinal}",
            require_formal_target=False,
        )
        native_path, native_sha = _asset(
            external_row["native"],
            external_row["native_sha256"],
            label=f"{dataset_id} native {ordinal}",
            require_formal_target=True,
        )
        if native_row.get("generated_sha256") != native_sha:
            raise ValueError(f"{dataset_id} native reuse SHA differs at {ordinal}")
        paper_path, paper_sha = _asset(
            paper_row["generated"],
            paper_row["generated_sha256"],
            label=f"{dataset_id} paper {ordinal}",
            require_formal_target=True,
        )
        bindings.append(
            {
                "ordinal": ordinal,
                "sample_id": sample_id,
                "source": source,
                "source_sha256": source_sha,
                "native": native_path,
                "native_sha256": native_sha,
                "paper": paper_path,
                "paper_sha256": paper_sha,
            }
        )
    return bindings


def _effective_config(
    parent: Mapping[str, Any],
    *,
    dataset_id: str,
    sample_count: int,
    output_root: Path,
) -> dict[str, Any]:
    config = dict(parent)
    for field in (
        "arm_config_sha256",
        R9_GUIDANCE_INTERVAL_CONTRACT_FIELD,
        "locked_schedule",
    ):
        config.pop(field, None)
    run_dir = output_root / "runs" / dataset_id / "schedule_nfe2"
    config.update(
        {
            "experiment_name": f"r11_schedule_nfe2__{dataset_id}",
            "arm_name": "schedule_nfe2",
            "causal_contract_type": R11_SCHEDULE_NFE2_CONTRACT_TYPE,
            "out_dir": str(run_dir),
            "guided_times": [1.0, 0.25],
            "unguided_times": [0.25, 0.0],
            R11_NFE2_SCHEDULE_CONTRACT_FIELD: (
                locked_r11_nfe2_schedule_contract()
            ),
            "asset_digest_cache": str(
                output_root / "asset_digests" / f"{dataset_id}.json"
            ),
        }
    )
    effective = resolve_frozen_effective_guidance_config(config)
    mutable_fields = {
        "experiment_name",
        "arm_name",
        "causal_contract_type",
        "out_dir",
        "guided_times",
        "unguided_times",
        R11_NFE2_SCHEDULE_CONTRACT_FIELD,
        "asset_digest_cache",
        R9_GUIDANCE_INTERVAL_CONTRACT_FIELD,
        "locked_schedule",
        "arm_config_sha256",
    }
    for field in set(parent) | set(effective):
        if field not in mutable_fields and parent.get(field) != effective.get(field):
            raise ValueError(
                f"{dataset_id} NFE2 changed non-schedule field {field!r}"
            )
    interval = effective[R9_GUIDANCE_INTERVAL_CONTRACT_FIELD]
    schedule = locked_r11_nfe2_schedule_contract()
    if (
        effective["batch_size"] != 2
        or effective["sampling_seed"] != 7919
        or effective["seed"] != 7919
        or effective["max_samples"] != sample_count
        or effective["active_guidance_intervals"] != []
        or effective["collect_interval_diagnostics"] is not False
        or interval["expected_algorithm_nfe"] != 2
        or interval["expected_diagnostic_nfe"] != 0
        or interval["expected_algorithm_trace"]
        != schedule["expected_algorithm_trace"]
        or not effective[R9_PHASE_CONTRACT_FIELD]["edev_required"]
        or effective["external_native_contract"]
        != parent["external_native_contract"]
    ):
        raise ValueError(f"{dataset_id} NFE2 invariant differs")
    locked = effective["locked_schedule"]
    if (
        locked["guided_times"] != [1.0, 0.25]
        or locked["unguided_times"] != [0.25, 0.0]
        or locked["parent_guided_times"] != [1.0, 0.75, 0.5, 0.25]
        or locked["parent_unguided_times"] != [0.25, 0.125, 0.0]
    ):
        raise ValueError(f"{dataset_id} locked NFE2 schedule differs")
    return effective


def _tmux_spec(job_id: str, ledger_path: Path, gpu_index: int) -> dict[str, Any]:
    return {
        "session": f"safa-r11-nfe2-{job_id.replace('__', '-').replace('_', '-')}",
        "remain_on_exit": True,
        "gpu_uuid": GPU_UUIDS[gpu_index],
        "argv": [
            PYTHON,
            "scripts/run_r11_schedule_nfe2_job.py",
            "--ledger",
            str(ledger_path),
            "--job-id",
            job_id,
        ],
    }


def _generation_job(
    *,
    dataset_id: str,
    gpu_index: int,
    output_root: Path,
    config_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    job_id = f"generate__{dataset_id}"
    run_dir = output_root / "runs" / dataset_id / "schedule_nfe2"
    asset_digest = output_root / "asset_digests" / f"{dataset_id}.json"
    log = output_root / "logs" / f"{job_id}.log"
    receipt = output_root / "attempt_receipts" / f"{job_id}.json"
    return {
        "job_id": job_id,
        "job_type": "generation",
        "wave": 0,
        "retry_limit": 0,
        "attempt_limit": 1,
        "gpu": {"index": gpu_index, "uuid": GPU_UUIDS[gpu_index]},
        "env": {"CUDA_VISIBLE_DEVICES": GPU_UUIDS[gpu_index]},
        "argv": [
            PYTHON,
            "scripts/run_meanflow_flow_map_guidance.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(run_dir),
        ],
        "tmux": _tmux_spec(job_id, ledger_path, gpu_index),
        "config_binding": _binding(config_path, exists=True),
        "output_path": str(run_dir),
        "asset_digest_path": str(asset_digest),
        "log_path": str(log),
        "attempt_receipt_path": str(receipt),
        "prerequisite_paths": [str(config_path)],
        "fresh_output_paths": [
            str(run_dir),
            str(asset_digest),
            str(log),
            str(receipt),
        ],
        "success_output_paths": [
            str(run_dir / "completion.json"),
            str(run_dir / "generation_result.json"),
            str(run_dir / "per_sample.jsonl"),
            str(run_dir / "sample_id_manifest.jsonl"),
            str(asset_digest),
        ],
        "expected_algorithm_nfe": 2,
        "expected_diagnostic_nfe": 0,
        "expected_matched_native_nfe": 0,
    }


def _quality_job(
    *,
    dataset_id: str,
    gpu_index: int,
    sample_count: int,
    selection: Path,
    config_path: Path,
    output_root: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    job_id = f"quality__{dataset_id}__schedule_nfe2"
    run_dir = output_root / "runs" / dataset_id / "schedule_nfe2"
    output = output_root / "quality" / dataset_id / "schedule_nfe2" / "quality.json"
    log = output_root / "logs" / f"{job_id}.log"
    receipt = output_root / "attempt_receipts" / f"{job_id}.json"
    generation_result = run_dir / "generation_result.json"
    metrics = (
        ["fid", "kid", "niqe", "sharpness"]
        if dataset_id == "prefix128"
        else ["niqe", "sharpness"]
    )
    argv = [
        PYTHON,
        "scripts/run_r11_quality_evaluation.py",
        "--real-index",
        "data/index/val_face_mixed_e14.jsonl",
        "--generated-dir",
        str(run_dir / "generated_images"),
        "--output",
        str(output),
        "--sample-id-manifest",
        str(selection),
        "--per-sample-jsonl",
        str(run_dir / "per_sample.jsonl"),
        "--device",
        "cuda:0",
        "--metrics",
        *metrics,
    ]
    if dataset_id == "prefix128":
        argv.extend(["--kid-subset-size", "127"])
    argv.extend(["--generation-result", str(generation_result)])
    return {
        "job_id": job_id,
        "job_type": "quality",
        "wave": 1,
        "retry_limit": 0,
        "attempt_limit": 1,
        "gpu": {"index": gpu_index, "uuid": GPU_UUIDS[gpu_index]},
        "env": {"CUDA_VISIBLE_DEVICES": GPU_UUIDS[gpu_index]},
        "argv": argv,
        "tmux": _tmux_spec(job_id, ledger_path, gpu_index),
        "config_binding": _binding(config_path, exists=True),
        "dataset_id": dataset_id,
        "role": "schedule_nfe2",
        "sample_count": sample_count,
        "metrics": metrics,
        "classifier_metrics": ["niqe", "sharpness"],
        "fid_kid_role": (
            "descriptive_only" if dataset_id == "prefix128" else "forbidden"
        ),
        "output_path": str(output),
        "log_path": str(log),
        "attempt_receipt_path": str(receipt),
        "prerequisite_paths": [str(config_path), str(generation_result)],
        "fresh_output_paths": [str(output), str(log), str(receipt)],
        "success_output_paths": [str(output)],
    }


def prepare(output_root: Path, config_dir: Path) -> dict[str, Any]:
    output_root_abs = _absolute(output_root)
    config_paths = {
        str(spec["dataset_id"]): (
            config_dir / f"r11_schedule_nfe2_{spec['dataset_id']}.yaml"
        )
        for spec in DATASETS
    }
    if output_root_abs.exists():
        raise FileExistsError(output_root_abs)
    existing_configs = [
        str(path) for path in config_paths.values() if _absolute(path).exists()
    ]
    if existing_configs:
        raise FileExistsError(f"NFE2 configs already exist: {existing_configs!r}")
    if not _absolute(Path(PYTHON)).is_file():
        raise FileNotFoundError(PYTHON)

    parent_classification_path = PARENT_ROOT / "causal_classification.json"
    parent_classification = _json(parent_classification_path)
    if (
        parent_classification.get("classification") != "schedule_branch"
        or parent_classification.get("candidate_promotion", {}).get("status")
        != "forbidden"
    ):
        raise ValueError("parent causal evidence is not locked schedule_branch")

    prepared: dict[str, dict[str, Any]] = {}
    for spec in DATASETS:
        dataset_id = str(spec["dataset_id"])
        sample_count = int(spec["sample_count"])
        selection = Path(spec["selection"])
        selection_rows = _rows(selection)
        sample_ids = [str(row["sample_id"]) for row in selection_rows]
        if len(sample_ids) != sample_count or len(set(sample_ids)) != sample_count:
            raise ValueError(f"{dataset_id} selection count/uniqueness differs")
        parent_config = yaml.safe_load(
            _absolute(Path(spec["parent_config"])).read_text(encoding="utf-8")
        )
        if not isinstance(parent_config, dict):
            raise ValueError(f"{dataset_id} parent config must be a mapping")
        expected_external = PARENT_ROOT / "reuse" / dataset_id / "native_external.jsonl"
        if (
            parent_config.get("sampling_seed") != 7919
            or parent_config.get("batch_size") != 2
            or parent_config.get("external_native_contract", {}).get("manifest")
            != str(expected_external)
            or parent_config.get("external_native_contract", {}).get(
                "ordered_sample_id_sha256"
            )
            != _id_sha256(sample_ids)
        ):
            raise ValueError(f"{dataset_id} parent causal binding differs")
        prepared[dataset_id] = {
            "sample_ids": sample_ids,
            "reuse_bindings": _reuse_bindings(dataset_id, sample_ids),
            "effective_config": _effective_config(
                parent_config,
                dataset_id=dataset_id,
                sample_count=sample_count,
                output_root=output_root,
            ),
        }

    for spec in DATASETS:
        dataset_id = str(spec["dataset_id"])
        reuse_path = output_root / "reuse_asset_bindings" / f"{dataset_id}.jsonl"
        _write_jsonl(
            _absolute(reuse_path),
            prepared[dataset_id]["reuse_bindings"],
        )
        _write_yaml(
            _absolute(config_paths[dataset_id]),
            prepared[dataset_id]["effective_config"],
        )

    ledger_path = output_root / "launch_ledger.json"
    jobs: list[dict[str, Any]] = []
    dataset_contracts: list[dict[str, Any]] = []
    for spec in DATASETS:
        dataset_id = str(spec["dataset_id"])
        sample_count = int(spec["sample_count"])
        selection = Path(spec["selection"])
        gpu_index = int(spec["gpu_index"])
        config_path = config_paths[dataset_id]
        jobs.append(
            _generation_job(
                dataset_id=dataset_id,
                gpu_index=gpu_index,
                output_root=output_root,
                config_path=config_path,
                ledger_path=ledger_path,
            )
        )
        jobs.append(
            _quality_job(
                dataset_id=dataset_id,
                gpu_index=gpu_index,
                sample_count=sample_count,
                selection=selection,
                config_path=config_path,
                output_root=output_root,
                ledger_path=ledger_path,
            )
        )
        parent_dataset = PARENT_ROOT / "reuse" / dataset_id
        parent_quality = PARENT_ROOT / "quality" / dataset_id
        run_dir = output_root / "runs" / dataset_id / "schedule_nfe2"
        nfe2_quality = (
            output_root / "quality" / dataset_id / "schedule_nfe2" / "quality.json"
        )
        dataset_contracts.append(
            {
                "dataset_id": dataset_id,
                "sample_count": sample_count,
                "ordered_sample_id_sha256": _id_sha256(
                    prepared[dataset_id]["sample_ids"]
                ),
                "selection_manifest": _binding(selection, exists=True),
                "reuse_asset_bindings": _binding(
                    output_root
                    / "reuse_asset_bindings"
                    / f"{dataset_id}.jsonl",
                    exists=True,
                ),
                "external_native_contract": _binding(
                    parent_dataset / "native_external.jsonl",
                    exists=True,
                ),
                "nfe2_config": _binding(config_path, exists=True),
                "nfe2_generation_result": _binding(
                    run_dir / "generation_result.json",
                    exists=False,
                ),
                "roles": {
                    "native": {
                        "quality_output": _binding(
                            parent_quality / "native" / "quality.json",
                            exists=True,
                        ),
                        "representation_rows": _binding(
                            parent_dataset / "native_reuse.jsonl",
                            exists=True,
                        ),
                    },
                    "paper_eta_0p125": {
                        "quality_output": _binding(
                            parent_quality
                            / "paper_eta_0p125"
                            / "quality.json",
                            exists=True,
                        ),
                        "representation_rows": _binding(
                            parent_dataset / "paper_reuse.jsonl",
                            exists=True,
                        ),
                    },
                    "schedule_nfe2": {
                        "quality_output": _binding(nfe2_quality, exists=False),
                        "representation_rows": _binding(
                            run_dir / "per_sample.jsonl",
                            exists=False,
                        ),
                    },
                },
            }
        )

    if [job["wave"] for job in jobs] != [0, 1, 0, 1]:
        raise AssertionError("NFE2 jobs must be two independent 0/1 waves")
    fresh_paths = [
        path for job in jobs for path in job["fresh_output_paths"]
    ]
    if len(fresh_paths) != len(set(fresh_paths)) or any(
        _absolute(Path(path)).exists() for path in fresh_paths
    ):
        raise ValueError("NFE2 fresh output paths are not unique and absent")

    contract_path = output_root / "diagnostic_contract.json"
    classifier_output = output_root / "classification.json"
    _write_json(
        _absolute(contract_path),
        {
            "schema_version": 1,
            "contract_type": "safa_r11_schedule_nfe2_diagnostic_v1",
            "parent_causal_classification": _binding(
                parent_classification_path,
                exists=True,
            ),
            "parent_causal_contract": _binding(
                PARENT_ROOT / "causal_contract.json",
                exists=True,
            ),
            "classifier": {
                "rule": (
                    "schedule_limited iff schedule_nfe2 passes "
                    "NIQE<=native+0.10 and sharpness>=0.95*native on both "
                    "datasets; otherwise mixed_guidance_failure and stop "
                    "the split route"
                ),
                "metrics": ["niqe", "sharpness"],
                "prefix_fid_kid": "descriptive_only",
                "tail_fid_kid": "forbidden",
                "representation": "report_only",
                "geometry": "not_evaluated",
                "privacy": "not_evaluated",
                "candidate_promotion": "forbidden",
                "classifier_output": _binding(classifier_output, exists=False),
            },
            "datasets": dataset_contracts,
        },
    )
    _write_json(
        _absolute(ledger_path),
        {
            "schema_version": 1,
            "contract_type": "safa_r11_schedule_nfe2_launch_ledger_v1",
            "status": "prepared_not_launched",
            "retry_limit": 0,
            "attempt_limit": 1,
            "execution": {
                "mode": "independent_one_job_per_tmux",
                "runner": "scripts/run_r11_schedule_nfe2_job.py",
                "durable_attempt_receipts": True,
                "automatic_retry": False,
            },
            "resource_policy": {
                "gpu_indices": [0, 1, 2, 3],
                "gpu_uuids": list(GPU_UUIDS),
                "cpu_memory_max_fraction": 0.9,
                "gpu_memory_max_fraction": 0.9,
            },
            "jobs": jobs,
        },
    )
    manifest = {
        "schema_version": 1,
        "contract_type": "safa_r11_schedule_nfe2_preparation_v1",
        "status": "prepared_not_launched",
        "parent_evidence": {
            "classification": "schedule_branch",
            "causal_classification": _binding(
                parent_classification_path,
                exists=True,
            ),
            "causal_contract": _binding(
                PARENT_ROOT / "causal_contract.json",
                exists=True,
            ),
            "formal_campaign": "r9-report-only-formal-v9",
            "formal_seed": 7919,
            "native_regeneration": False,
            "paper_regeneration": False,
        },
        "invariants": {
            "checkpoint": "E15",
            "sampling_seed": 7919,
            "batch_size": 2,
            "schedule_times": [1.0, 0.25, 0.0],
            "expected_algorithm_nfe": 2,
            "expected_diagnostic_nfe": 0,
            "expected_matched_native_nfe": 0,
            "active_guidance_intervals": [],
            "effective_edev": True,
            "retry_limit": 0,
            "attempt_limit": 1,
            "candidate_promotion": "forbidden",
        },
        "diagnostic_contract": _binding(contract_path, exists=True),
        "launch_ledger": _binding(ledger_path, exists=True),
        "configs": [
            _binding(config_paths[str(spec["dataset_id"])], exists=True)
            for spec in DATASETS
        ],
        "job_counts": {"generation": 2, "quality": 2},
        "classifier": {
            "argv": [
                PYTHON,
                "scripts/classify_r11_schedule_nfe2.py",
                "--contract",
                str(contract_path),
                "--output",
                str(classifier_output),
            ],
            "output": _binding(classifier_output, exists=False),
        },
    }
    _write_json(_absolute(output_root / "preparation_manifest.json"), manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the locked R11 coarse-schedule NFE2 diagnostic."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/r11_schedule_nfe2/preparation_v1"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/medium_v2/experiments"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = prepare(args.output_root, args.config_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
