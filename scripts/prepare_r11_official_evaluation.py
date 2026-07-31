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
    datasets = []
    for dataset_id, contract in contracts.items():
        contract_path = output / f"{dataset_id}.json"
        dataset_root = evaluation_root / dataset_id
        runs_root = (
            REPO_ROOT / "artifacts/r11_initial_noise_sharpness_probe/runs_v1"
        ).resolve()
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
    plan = {
        "schema_version": 1,
        "contract_type": "safa_r11_official_evaluation_preparation_v1",
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
        "expected_arcface_request_count": 4,
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
