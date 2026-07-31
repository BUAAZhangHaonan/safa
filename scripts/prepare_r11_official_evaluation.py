#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_CONTRACTS = (
    REPO_ROOT
    / "artifacts/r11_initial_noise_sharpness_probe/preparation_v1/run_contracts.json"
)
REAL_INDEX = (REPO_ROOT / "data/index/val_face_mixed_e14.jsonl").resolve()
FORMAL_ARCFACE_TEMPLATE = (
    REPO_ROOT
    / "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v9/full/evaluator_runs/arcface/winner/request.json"
).resolve()
ARCFACE_WORKER_WRAPPER = (
    REPO_ROOT / "scripts/run_r9_phase_evaluator.py"
).resolve()
ARCFACE_WORKER_IMPLEMENTATION = (
    REPO_ROOT / "src/safa/evaluation/r9_evaluator_worker.py"
).resolve()
EXPECTED_DATASETS = {
    "prefix128": {
        "stage": 128,
        "sample_count": 128,
        "metrics": ["fid", "kid", "niqe", "sharpness"],
        "selection_role": "committed_prefix128",
        "baseline": "eta025_prefix128",
    },
    "sharpness_tail32": {
        "stage": 32,
        "sample_count": 32,
        "metrics": ["niqe", "sharpness"],
        "selection_role": "sharpness_tail32",
        "baseline": "eta025_tail32",
    },
}
GPU_BINDINGS = (
    {"physical_index": 0, "uuid": "GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6"},
    {"physical_index": 1, "uuid": "GPU-dfaeaa7c-32c8-ebb4-aa59-ab7f829805f1"},
    {"physical_index": 2, "uuid": "GPU-e27fe71d-eaf7-3eb5-d0ff-c1c63b4f6b02"},
    {"physical_index": 3, "uuid": "GPU-61ea2925-9905-7f56-cd64-7a792a32efef"},
)


class R11EvaluationPreparationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise R11EvaluationPreparationError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise R11EvaluationPreparationError(f"{label} must be an object")
    return value


def _repo_file(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise R11EvaluationPreparationError(f"{label} path is invalid")
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    if not path.is_file():
        raise R11EvaluationPreparationError(f"{label} is missing: {path}")
    return path


def _selection_binding(path: Path, expected_count: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    sample_ids = [
        row.get("sample_id") if isinstance(row, Mapping) else None for row in rows
    ]
    if (
        len(sample_ids) != expected_count
        or len(set(sample_ids)) != expected_count
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
    ):
        raise R11EvaluationPreparationError(
            f"{path} must contain {expected_count} unique ordered sample IDs"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "sample_count": expected_count,
        "ordered_sample_id_sha256": hashlib.sha256(
            "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
        ).hexdigest(),
    }


def _dataset_contracts(run_contracts_path: Path) -> dict[str, dict[str, Any]]:
    envelope = _json(run_contracts_path, "R11 run contracts")
    if (
        envelope.get("contract_type")
        != "safa_r11_initial_noise_sharpness_probe_runs_v1"
        or envelope.get("retry_count") != 0
        or not isinstance(envelope.get("runs"), list)
        or len(envelope["runs"]) != 4
    ):
        raise R11EvaluationPreparationError("R11 run contract envelope differs")
    grouped: dict[str, list[dict[str, Any]]] = {
        "prefix128": [],
        "sharpness_tail32": [],
    }
    for run in envelope["runs"]:
        if not isinstance(run, Mapping):
            raise R11EvaluationPreparationError("R11 run row must be an object")
        role = run.get("selection_role")
        dataset_id = (
            "prefix128"
            if role == "committed_prefix128"
            else "sharpness_tail32" if role == "sharpness_tail32" else None
        )
        if dataset_id is None:
            raise R11EvaluationPreparationError("unknown R11 selection role")
        config_path = _repo_file(run.get("config"), f"{run.get('arm_id')} config")
        if sha256_file(config_path) != run.get("config_sha256"):
            raise R11EvaluationPreparationError(
                f"{run.get('arm_id')} config SHA256 differs"
            )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected = EXPECTED_DATASETS[dataset_id]
        if (
            not isinstance(config, Mapping)
            or config.get("phase") != "calibration"
            or config.get("mode") != "initial_noise"
            or config.get("projection") != "fixed_radius"
            or config.get("num_updates") != 16
            or config.get("batch_size") != 2
            or config.get("sampling_seed") != 1337
            or config.get("max_samples") != expected["sample_count"]
            or run.get("sample_count") != expected["sample_count"]
            or run.get("eta") not in (0.25, 0.5)
        ):
            raise R11EvaluationPreparationError(
                f"{run.get('arm_id')} fixed generation semantics differ"
            )
        selection_path = _repo_file(
            config.get("sample_id_manifest"), f"{run.get('arm_id')} selection"
        )
        if (
            sha256_file(selection_path) != config.get("sample_id_manifest_sha256")
            or config.get("sample_id_manifest_sha256")
            != config.get("calibration_sample_id_manifest_sha256")
        ):
            raise R11EvaluationPreparationError(
                f"{run.get('arm_id')} selection SHA256 differs"
            )
        grouped[dataset_id].append(
            {
                "arm_id": run["arm_id"],
                "eta": run["eta"],
                "config": {
                    "path": str(config_path),
                    "sha256": sha256_file(config_path),
                },
                "run_root": str((REPO_ROOT / str(run["output_dir"])).resolve()),
                "selection_path": selection_path,
            }
        )
    datasets: dict[str, dict[str, Any]] = {}
    for dataset_id, arms in grouped.items():
        expected = EXPECTED_DATASETS[dataset_id]
        arms.sort(key=lambda arm: arm["eta"])
        if (
            [arm["eta"] for arm in arms] != [0.25, 0.5]
            or arms[0]["arm_id"] != expected["baseline"]
            or len({arm["selection_path"] for arm in arms}) != 1
        ):
            raise R11EvaluationPreparationError(
                f"{dataset_id} eta/baseline/selection matrix differs"
            )
        selection = _selection_binding(
            arms[0]["selection_path"], expected["sample_count"]
        )
        for arm in arms:
            arm.pop("selection_path")
        datasets[dataset_id] = {
            "schema_version": 1,
            "contract_type": "safa_r11_initial_noise_evaluation_dataset_v1",
            "registration": "official_r11_typed_evaluation",
            "dataset_id": dataset_id,
            "selection_role": dataset_id,
            "sample_count": expected["sample_count"],
            "stage": expected["stage"],
            "quality_metrics": expected["metrics"],
            "sampling_seed": 1337,
            "selection_manifest": selection,
            "baseline_arm_id": expected["baseline"],
            "arms": arms,
        }
    return datasets


def _launch_job(
    *,
    job_id: str,
    kind: str,
    dataset_id: str,
    role: str,
    wave: str,
    gpu_slot: int,
    argv: list[str],
    inputs: list[Path],
    outputs: list[Path],
    log_path: Path,
) -> dict[str, Any]:
    gpu = GPU_BINDINGS[gpu_slot]
    return {
        "job_id": job_id,
        "kind": kind,
        "dataset_id": dataset_id,
        "role": role,
        "wave": wave,
        "retry_count": 0,
        "physical_gpu": gpu,
        "environment": {"CUDA_VISIBLE_DEVICES": gpu["uuid"]},
        "logical_device": "cuda:0",
        "argv": argv,
        "input_paths": [str(path) for path in inputs],
        "output_path": str(outputs[0]),
        "log_path": str(log_path),
        "fresh_output_paths": [str(path) for path in (*outputs, log_path)],
    }


def _evaluation_launch_jobs(
    contracts: Mapping[str, Mapping[str, Any]],
    *,
    evaluation_root: Path,
    runs_root: Path,
) -> list[dict[str, Any]]:
    assignments = {
        ("prefix128", "native", "quality"): ("A", 0),
        ("prefix128", "eta025_prefix128", "quality"): ("A", 1),
        ("prefix128", "eta05_prefix128", "quality"): ("A", 2),
        ("sharpness_tail32", "native", "quality"): ("A", 3),
        ("sharpness_tail32", "eta025_tail32", "quality"): ("B", 0),
        ("sharpness_tail32", "eta05_tail32", "quality"): ("B", 1),
        ("prefix128", "eta025_prefix128", "arcface"): ("B", 2),
        ("prefix128", "eta05_prefix128", "arcface"): ("B", 3),
        ("sharpness_tail32", "eta025_tail32", "arcface"): ("C", 0),
        ("sharpness_tail32", "eta05_tail32", "arcface"): ("C", 1),
    }
    jobs: list[dict[str, Any]] = []
    for dataset_id, contract in contracts.items():
        dataset_root = evaluation_root / dataset_id
        selection = Path(str(contract["selection_manifest"]["path"]))
        metrics = list(contract["quality_metrics"])
        baseline = str(contract["baseline_arm_id"])
        quality_roles = ["native", *[str(arm["arm_id"]) for arm in contract["arms"]]]
        for role in quality_roles:
            is_native = role == "native"
            per_sample = (
                dataset_root / "inputs/native_per_sample.jsonl"
                if is_native
                else runs_root / role / "per_sample.jsonl"
            )
            generated_dir = (
                runs_root / baseline / "native_images"
                if is_native
                else runs_root / role / "generated_images"
            )
            output_path = dataset_root / "quality" / role / "quality.json"
            argv = [
                sys.executable,
                "scripts/run_r11_quality_evaluation.py",
                "--real-index",
                str(REAL_INDEX),
                "--generated-dir",
                str(generated_dir),
                "--output",
                str(output_path),
                "--sample-id-manifest",
                str(selection),
                "--per-sample-jsonl",
                str(per_sample),
            ]
            if not is_native:
                argv.extend(
                    [
                        "--generation-result",
                        str(runs_root / role / "generation_result.json"),
                    ]
                )
            if "kid" in metrics:
                argv.extend(
                    ["--kid-subset-size", str(int(contract["sample_count"]) - 1)]
                )
            argv.extend(
                [
                    "--seed",
                    str(contract["sampling_seed"]),
                    "--device",
                    "cuda:0",
                    "--metrics",
                    *metrics,
                ]
            )
            wave, gpu_slot = assignments[(dataset_id, role, "quality")]
            quality_job = _launch_job(
                    job_id=f"quality__{dataset_id}__{role}",
                    kind="quality",
                    dataset_id=dataset_id,
                    role=role,
                    wave=wave,
                    gpu_slot=gpu_slot,
                    argv=argv,
                    inputs=[selection, per_sample],
                    outputs=[output_path],
                    log_path=dataset_root / "logs" / f"quality__{role}.log",
                )
            quality_job["quality_output_path"] = str(output_path)
            jobs.append(quality_job)
        for arm in contract["arms"]:
            arm_id = str(arm["arm_id"])
            request_path = dataset_root / "arcface" / arm_id / "request.json"
            result_path = dataset_root / "arcface" / arm_id / "result.json"
            wave, gpu_slot = assignments[(dataset_id, arm_id, "arcface")]
            arcface_job = _launch_job(
                    job_id=f"arcface__{dataset_id}__{arm_id}",
                    kind="arcface",
                    dataset_id=dataset_id,
                    role=arm_id,
                    wave=wave,
                    gpu_slot=gpu_slot,
                    argv=[
                        sys.executable,
                        "scripts/run_r9_phase_evaluator.py",
                        "--request",
                        str(request_path),
                        "--output",
                        str(result_path),
                    ],
                    inputs=[request_path],
                    outputs=[result_path],
                    log_path=dataset_root / "logs" / f"arcface__{arm_id}.log",
                )
            arcface_job["request_path"] = str(request_path)
            arcface_job["result_path"] = str(result_path)
            jobs.append(arcface_job)
    expected_mapping = set(assignments)
    if (
        len(jobs) != 10
        or sum(job["kind"] == "quality" for job in jobs) != 6
        or sum(job["kind"] == "arcface" for job in jobs) != 4
        or len({job["job_id"] for job in jobs}) != 10
        or {
            (job["dataset_id"], job["role"], job["kind"]) for job in jobs
        }
        != expected_mapping
    ):
        raise R11EvaluationPreparationError("evaluation launch job matrix differs")
    fresh = [path for job in jobs for path in job["fresh_output_paths"]]
    if len(fresh) != len(set(fresh)):
        raise R11EvaluationPreparationError("evaluation launch outputs are not unique")
    existing = [path for path in fresh if Path(path).exists()]
    if existing:
        raise R11EvaluationPreparationError(
            f"evaluation launch outputs are not fresh: {existing[:4]!r}"
        )
    return jobs


def prepare(
    *,
    run_contracts_path: Path,
    output_dir: Path,
    write: bool,
) -> dict[str, Any]:
    contracts = _dataset_contracts(run_contracts_path.resolve())
    output = output_dir.resolve()
    evaluation_root = (
        REPO_ROOT / "artifacts/r11_initial_noise_sharpness_probe/evaluation_v1"
    ).resolve()
    runs_root = (
        REPO_ROOT / "artifacts/r11_initial_noise_sharpness_probe/runs_v1"
    ).resolve()
    datasets = []
    for dataset_id, contract in contracts.items():
        contract_path = output / f"{dataset_id}.json"
        dataset_root = evaluation_root / dataset_id
        commands = {
            "quality_prepare": [
                sys.executable,
                "scripts/prepare_fixed32_quality_inputs.py",
                "--runs-root",
                str(runs_root),
                "--selection-manifest",
                contract["selection_manifest"]["path"],
                "--real-index",
                str(REAL_INDEX),
                "--output-dir",
                str(dataset_root / "inputs"),
                "--arm-set-manifest",
                str(contract_path),
                "--device",
                "cuda:0",
            ],
            "arcface_request_build": [
                sys.executable,
                "scripts/build_fixed32_arcface_requests.py",
                "--diagnostic-manifest",
                str(contract_path),
                "--selection-manifest",
                contract["selection_manifest"]["path"],
                "--runs-root",
                str(runs_root),
                "--template-request",
                str(FORMAL_ARCFACE_TEMPLATE),
                "--output-root",
                str(dataset_root / "arcface"),
                "--device",
                "cuda:0",
            ],
            "materialize": [
                sys.executable,
                "scripts/materialize_fixed32_triangle_rows.py",
                "--diagnostic-manifest",
                str(contract_path),
                "--runs-root",
                str(runs_root),
                "--evaluation-root",
                str(dataset_root),
                "--selection-manifest",
                contract["selection_manifest"]["path"],
                "--output-dir",
                str(dataset_root / "rows"),
                "--baseline-arm-id",
                contract["baseline_arm_id"],
            ],
            "screen": [
                sys.executable,
                "scripts/run_triangle_screening.py",
                "--request",
                str(dataset_root / "rows/screening_request.json"),
                "--output-dir",
                str(dataset_root / "screening"),
            ],
        }
        datasets.append(
            {
                "dataset_id": dataset_id,
                "dataset_contract": str(contract_path),
                "expected_arcface_request_count": len(contract["arms"]),
                "commands": commands,
                "output_root": str(dataset_root),
            }
        )
    launch_jobs = _evaluation_launch_jobs(
        contracts, evaluation_root=evaluation_root, runs_root=runs_root
    )
    plan = {
        "schema_version": 1,
        "contract_type": "safa_r11_official_evaluation_preparation_v2",
        "retry_count": 0,
        "run_contracts": {
            "path": str(run_contracts_path.resolve()),
            "sha256": sha256_file(run_contracts_path.resolve()),
        },
        "real_index": {
            "path": str(REAL_INDEX),
            "sha256": sha256_file(REAL_INDEX),
        },
        "formal_arcface_template": {
            "path": str(FORMAL_ARCFACE_TEMPLATE),
            "sha256": sha256_file(FORMAL_ARCFACE_TEMPLATE),
        },
        "r11_arcface_worker": {
            "request_contract": "safa_r11_arcface_evaluator_request_v1",
            "output_contract": "safa_r11_arcface_evaluator_output_v1",
            "pair_policy": "pairwise_exact_one_v1",
            "wrapper": {
                "path": str(ARCFACE_WORKER_WRAPPER),
                "sha256": sha256_file(ARCFACE_WORKER_WRAPPER),
            },
            "implementation": {
                "path": str(ARCFACE_WORKER_IMPLEMENTATION),
                "sha256": sha256_file(ARCFACE_WORKER_IMPLEMENTATION),
            },
        },
        "expected_arcface_request_count": 4,
        "expected_quality_job_count": 6,
        "gpu_bindings": list(GPU_BINDINGS),
        "launch_jobs": launch_jobs,
        "launch_waves": [
            {
                "wave": wave,
                "job_ids": [
                    job["job_id"] for job in launch_jobs if job["wave"] == wave
                ],
            }
            for wave in ("A", "B", "C")
        ],
        "datasets": datasets,
    }
    if write:
        output.mkdir(parents=True, exist_ok=False)
        for dataset_id, contract in contracts.items():
            (output / f"{dataset_id}.json").write_text(
                json.dumps(contract, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        for dataset in plan["datasets"]:
            path = Path(dataset["dataset_contract"])
            dataset["dataset_contract_sha256"] = sha256_file(path)
        (output / "evaluation_plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return plan


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the official typed R11 quality/ArcFace/screening adapter."
    )
    parser.add_argument(
        "--run-contracts", type=Path, default=DEFAULT_RUN_CONTRACTS
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = prepare(
        run_contracts_path=args.run_contracts,
        output_dir=args.output_dir,
        write=not args.dry_run,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R11EvaluationPreparationError as exc:
        print(f"R11 evaluation preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
