#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from safa.evaluation.meanflow_guidance_runner import (
    R9_GUIDANCE_INTERVAL_CONTRACT_FIELD,
    R9_PHASE_CONTRACT_FIELD,
    finalize_effective_guidance_config,
    validate_guidance_config,
)


REPO = Path(__file__).resolve().parents[1]
FORMAL_ROOT = Path(
    "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v9"
)
FORMAL_TEMPLATE = FORMAL_ROOT / "runtime_configs/full/winner.yaml"
FORMAL_SHARDS = FORMAL_ROOT / "full/winner/shards"
FORMAL_FULL_MANIFEST = Path(
    "configs/medium_v2/experiments/r9_manifests/full_2048.jsonl"
)
DATASETS = (
    (
        "prefix128",
        128,
        Path("artifacts/r10_triangle_exploration/preparation_v1/prefix128.jsonl"),
        0,
    ),
    (
        "sharpness_tail32",
        32,
        Path(
            "artifacts/r11_initial_noise_sharpness_probe/preparation_v1/tail32.jsonl"
        ),
        1,
    ),
)
PYTHON = "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"
GPU_UUIDS = (
    "GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
    "GPU-dfaeaa7c-32c8-ebb4-aa59-ab7f829805f1",
    "GPU-e27fe71d-eaf7-3eb5-d0ff-c1c63b4f6b02",
    "GPU-61ea2925-9905-7f56-cd64-7a792a32efef",
)


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in ids).encode("utf-8")
    ).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    resolved = _absolute(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{resolved}:{line_number}: row is not an object")
        result.append(value)
    return result


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
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False),
        encoding="utf-8",
    )


def _formal_rows() -> dict[str, dict[str, Any]]:
    shard_paths = sorted(
        _absolute(FORMAL_SHARDS).glob("shard_*/per_sample.jsonl"),
        key=lambda path: int(path.parent.name.split("_")[1]),
    )
    if len(shard_paths) != 16:
        raise ValueError("formal reuse requires exactly 16 winner shards")
    by_id: dict[str, dict[str, Any]] = {}
    for path in shard_paths:
        for row in _rows(path):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in by_id:
                raise ValueError(f"invalid/duplicate formal sample ID {sample_id!r}")
            by_id[sample_id] = row
    full_ids = [str(row["sample_id"]) for row in _rows(FORMAL_FULL_MANIFEST)]
    if len(full_ids) != 2048 or len(set(full_ids)) != 2048:
        raise ValueError("formal Full manifest must contain 2,048 unique IDs")
    if set(by_id) != set(full_ids):
        raise ValueError("formal 16-shard rows do not exactly cover Full manifest")
    return by_id


def _asset(path_value: Any, *, label: str) -> tuple[str, str]:
    path = _absolute(Path(str(path_value)))
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a direct file: {path}")
    return str(path), _sha256(path)


def _flat_link(directory: Path, ordinal: int, source: Path) -> Path:
    link = directory / f"{ordinal:08d}.png"
    if link.exists() or link.is_symlink():
        raise FileExistsError(link)
    directory.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), link)
    if _sha256(link) != _sha256(source):
        raise ValueError(f"flat reuse link changed bytes: {link}")
    return link


def _effective_config(
    template: Mapping[str, Any],
    *,
    dataset_id: str,
    count: int,
    selection: Path,
    external_manifest: Path,
    output_root: Path,
    gpu_index: int,
) -> dict[str, Any]:
    config = dict(template)
    for field in (
        "arm_config_sha256",
        R9_GUIDANCE_INTERVAL_CONTRACT_FIELD,
        R9_PHASE_CONTRACT_FIELD,
    ):
        config.pop(field, None)
    selection_abs = _absolute(selection)
    config.update(
        {
            "experiment_name": f"r11_causal__transport_only_nfe5__{dataset_id}",
            "arm_name": "transport_only_nfe5",
            "causal_contract_type": "safa_r11_transport_only_nfe5_v1",
            "out_dir": str(
                output_root / "runs" / dataset_id / "transport_only_nfe5"
            ),
            "mode": "paper_algorithm_split",
            "phase": "calibrate",
            "seed": 7919,
            "sampling_seed": 7919,
            "device": "cuda:0",
            "sample_id_manifest": str(selection),
            "sample_id_manifest_sha256": _sha256(selection_abs),
            "calibration_sample_id_manifest": str(selection),
            "calibration_sample_id_manifest_sha256": _sha256(selection_abs),
            "max_samples": count,
            "calibration_samples": count,
            "batch_size": 2,
            "contact_sheets": False,
            "step_size": 0.125,
            "active_guidance_intervals": [],
            "collect_interval_diagnostics": False,
            "asset_digest_cache": str(
                output_root / "asset_digests" / f"{dataset_id}.json"
            ),
            "external_native_contract": {
                "schema_version": 1,
                "contract_type": "safa_r11_causal_external_native_v1",
                "manifest": str(external_manifest),
                "manifest_sha256": _sha256(_absolute(external_manifest)),
                "sample_count": count,
                "ordered_sample_id_sha256": _id_sha256(
                    [str(row["sample_id"]) for row in _rows(selection)]
                ),
            },
        }
    )
    validated = validate_guidance_config(config)
    effective = finalize_effective_guidance_config(
        validated,
        locked_schedule=validated["locked_schedule"],
    )
    interval = effective[R9_GUIDANCE_INTERVAL_CONTRACT_FIELD]
    if (
        interval["active_guidance_intervals"] != []
        or interval["expected_algorithm_nfe"] != 5
        or interval["expected_diagnostic_nfe"] != 0
        or not effective[R9_PHASE_CONTRACT_FIELD]["edev_required"]
    ):
        raise ValueError("effective causal config is not NFE5/Edev")
    return effective


def _binding(path: Path, *, exists: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path)}
    if exists:
        result["sha256"] = _sha256(_absolute(path))
    else:
        result["expected_absent_at_preparation"] = True
    return result


def _quality_job(
    *,
    job_id: str,
    wave: int,
    gpu_index: int,
    dataset_id: str,
    role: str,
    selection: Path,
    generated_dir: Path,
    per_sample: Path,
    output: Path,
    log_path: Path,
    generation_result: Path | None = None,
) -> dict[str, Any]:
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
        str(generated_dir),
        "--output",
        str(output),
        "--sample-id-manifest",
        str(selection),
        "--per-sample-jsonl",
        str(per_sample),
        "--device",
        "cuda:0",
        "--metrics",
        *metrics,
    ]
    if dataset_id == "prefix128":
        argv.extend(["--kid-subset-size", "127"])
    if generation_result is not None:
        argv.extend(["--generation-result", str(generation_result)])
    return {
        "job_id": job_id,
        "job_type": "quality",
        "wave": wave,
        "retry_limit": 0,
        "gpu": {"index": gpu_index, "uuid": GPU_UUIDS[gpu_index]},
        "env": {"CUDA_VISIBLE_DEVICES": GPU_UUIDS[gpu_index]},
        "argv": argv,
        "dataset_id": dataset_id,
        "role": role,
        "output_path": str(output),
        "log_path": str(log_path),
        "fresh_output_paths": [str(output), str(log_path)],
        "classifier_metrics": ["niqe", "sharpness"],
        "fid_kid_role": "descriptive_only"
        if dataset_id == "prefix128"
        else "forbidden",
    }


def prepare(output_root: Path, config_dir: Path) -> dict[str, Any]:
    output_root_abs = _absolute(output_root)
    if output_root_abs.exists():
        raise FileExistsError(output_root_abs)
    template = yaml.safe_load(_absolute(FORMAL_TEMPLATE).read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError("formal winner YAML must be a mapping")
    by_id = _formal_rows()
    dataset_contracts: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    quality_job_index = 0
    for dataset_id, count, selection, gpu_index in DATASETS:
        selection_rows = _rows(selection)
        ids = [str(row["sample_id"]) for row in selection_rows]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"{dataset_id} selection count/uniqueness differs")
        dataset_dir = output_root / "reuse" / dataset_id
        native_external_path = dataset_dir / "native_external.jsonl"
        native_view_path = dataset_dir / "native_reuse.jsonl"
        paper_view_path = dataset_dir / "paper_reuse.jsonl"
        native_flat = output_root_abs / "flat" / dataset_id / "native"
        paper_flat = output_root_abs / "flat" / dataset_id / "paper_eta_0p125"
        native_external: list[dict[str, Any]] = []
        native_view: list[dict[str, Any]] = []
        paper_view: list[dict[str, Any]] = []
        for ordinal, sample_id in enumerate(ids):
            formal = by_id.get(sample_id)
            if formal is None:
                raise ValueError(f"{dataset_id} ID absent from formal Full: {sample_id}")
            source, source_sha = _asset(formal["source"], label="formal source")
            native, native_sha = _asset(formal["native"], label="formal native")
            paper, paper_sha = _asset(formal["generated"], label="formal paper")
            native_link = _flat_link(native_flat, ordinal, Path(native))
            paper_link = _flat_link(paper_flat, ordinal, Path(paper))
            native_external.append(
                {
                    "ordinal": ordinal,
                    "sample_id": sample_id,
                    "source": source,
                    "source_sha256": source_sha,
                    "native": native,
                    "native_sha256": native_sha,
                    "e0_cosine": float(formal["native_cosine"]),
                    "edev_cosine": float(formal["native_edev_cosine"]),
                }
            )
            native_view.append(
                {
                    "ordinal": ordinal,
                    "sample_id": sample_id,
                    "source": source,
                    "generated": str(native_link),
                    "generated_sha256": native_sha,
                    "e0_cosine": float(formal["native_cosine"]),
                    "edev_cosine": float(formal["native_edev_cosine"]),
                }
            )
            paper_view.append(
                {
                    "ordinal": ordinal,
                    "sample_id": sample_id,
                    "source": source,
                    "generated": str(paper_link),
                    "generated_sha256": paper_sha,
                    "e0_cosine": float(formal["candidate_cosine"]),
                    "edev_cosine": float(formal["edev_cosine"]),
                }
            )
        _write_jsonl(_absolute(native_external_path), native_external)
        _write_jsonl(_absolute(native_view_path), native_view)
        _write_jsonl(_absolute(paper_view_path), paper_view)
        config_path = config_dir / f"r11_transport_only_nfe5_{dataset_id}.yaml"
        effective = _effective_config(
            template,
            dataset_id=dataset_id,
            count=count,
            selection=selection,
            external_manifest=native_external_path,
            output_root=output_root,
            gpu_index=gpu_index,
        )
        _write_yaml(_absolute(config_path), effective)
        run_dir = output_root / "runs" / dataset_id / "transport_only_nfe5"
        quality_dir = output_root / "quality" / dataset_id
        generation_log = output_root / "logs" / f"generate__{dataset_id}.log"
        asset_digest_path = Path(str(effective["asset_digest_cache"]))
        jobs.append(
            {
                "job_id": f"generate__{dataset_id}",
                "job_type": "generation",
                "wave": 0,
                "retry_limit": 0,
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
                "fresh_output": str(run_dir),
                "output_path": str(run_dir),
                "asset_digest_path": str(asset_digest_path),
                "log_path": str(generation_log),
                "fresh_output_paths": [
                    str(run_dir),
                    str(asset_digest_path),
                    str(generation_log),
                ],
            }
        )
        role_sources = {
            "native": (output_root / "flat" / dataset_id / "native", native_view_path),
            "paper_eta_0p125": (
                output_root / "flat" / dataset_id / "paper_eta_0p125",
                paper_view_path,
            ),
            "transport_only_nfe5": (
                run_dir / "generated_images",
                run_dir / "per_sample.jsonl",
            ),
        }
        role_bindings: dict[str, Any] = {}
        for role_index, role in enumerate(
            ("native", "paper_eta_0p125", "transport_only_nfe5")
        ):
            job_id = f"quality__{dataset_id}__{role}"
            generated_dir, per_sample = role_sources[role]
            quality_output = quality_dir / role / "quality.json"
            jobs.append(
                _quality_job(
                    job_id=job_id,
                    wave=1 + quality_job_index // 4,
                    gpu_index=quality_job_index % 4,
                    dataset_id=dataset_id,
                    role=role,
                    selection=selection,
                    generated_dir=generated_dir,
                    per_sample=per_sample,
                    output=quality_output,
                    log_path=output_root / "logs" / f"{job_id}.log",
                    generation_result=(
                        run_dir / "generation_result.json"
                        if role == "transport_only_nfe5"
                        else None
                    ),
                )
            )
            quality_job_index += 1
            role_bindings[role] = {
                "quality_output": _binding(quality_output, exists=False),
                "representation_rows": _binding(
                    per_sample,
                    exists=role != "transport_only_nfe5",
                ),
            }
        dataset_contracts.append(
            {
                "dataset_id": dataset_id,
                "sample_count": count,
                "ordered_sample_id_sha256": _id_sha256(ids),
                "selection_manifest": _binding(selection, exists=True),
                "roles": role_bindings,
            }
        )
    generation_jobs = [job for job in jobs if job["job_type"] == "generation"]
    quality_jobs = [job for job in jobs if job["job_type"] == "quality"]
    if len(generation_jobs) != 2 or len(quality_jobs) != 6:
        raise AssertionError("launch ledger must contain 2 generation + 6 quality jobs")
    causal_contract_path = output_root / "causal_contract.json"
    _write_json(
        _absolute(causal_contract_path),
        {
            "schema_version": 1,
            "contract_type": "safa_r11_transport_causal_decomposition_v1",
            "classifier": {
                "rule": (
                    "correction_limited iff transport passes NIQE<=native+0.10 "
                    "and sharpness>=0.95*native on both datasets while paper "
                    "fails either quality gate on at least one dataset"
                ),
                "metrics": ["niqe", "sharpness"],
                "prefix_fid_kid": "descriptive_only",
                "tail_fid_kid": "forbidden",
                "geometry": "not_evaluated",
                "privacy": "not_evaluated",
                "candidate_promotion": "forbidden",
            },
            "datasets": dataset_contracts,
        },
    )
    ledger_path = output_root / "launch_ledger.json"
    _write_json(
        _absolute(ledger_path),
        {
            "schema_version": 1,
            "contract_type": "safa_r11_causal_launch_ledger_v1",
            "status": "prepared_not_launched",
            "retry_limit": 0,
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
        "contract_type": "safa_r11_causal_preparation_v1",
        "status": "go_not_launched",
        "formal_reuse": {
            "campaign": "r9-report-only-formal-v9",
            "arm": "paper_eta_0p125",
            "full_verdict": "failed_locked_winner",
            "sample_count": 2048,
            "shards": 16,
            "seed": 7919,
            "regeneration": {"native": False, "paper": False},
        },
        "invariants": {
            "checkpoint": "E15",
            "vae_digest": template["vae_digest"],
            "sampling_seed": 7919,
            "step_size": 0.125,
            "active_guidance_intervals": [],
            "expected_algorithm_nfe": 5,
            "effective_edev": True,
            "retry_limit": 0,
        },
        "causal_contract": _binding(causal_contract_path, exists=True),
        "launch_ledger": _binding(ledger_path, exists=True),
        "configs": [
            _binding(
                config_dir / f"r11_transport_only_nfe5_{dataset_id}.yaml",
                exists=True,
            )
            for dataset_id, _, _, _ in DATASETS
        ],
        "job_counts": {"generation": 2, "quality": 6},
    }
    _write_json(_absolute(output_root / "preparation_manifest.json"), manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the locked R11 transport-only causal decomposition."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/r11_causal_decomposition/preparation_v1"),
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
