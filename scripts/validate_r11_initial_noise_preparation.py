#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from safa.evaluation.meanflow_guidance_runner import (
    _edev_required_for_config,
    resolve_frozen_effective_guidance_config,
)


ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / "artifacts/r11_initial_noise_sharpness_probe/preparation_v1"
POOL = ROOT / "artifacts/r10_triangle_exploration/preparation_v1/eligible_pool_2045.jsonl"
PREFIX128 = ROOT / "artifacts/r10_triangle_exploration/preparation_v1/prefix128.jsonl"
TAIL32 = PREPARATION / "tail32.jsonl"
MANIFEST = PREPARATION / "preparation_manifest.json"
RUN_CONTRACTS = PREPARATION / "run_contracts.json"
EXPECTED_POOL_SHA256 = "56c2baf99779d8b9cf7460b34bc1e388ba1f8687e8073ca1d636f137dd2c00e8"
EXPECTED_PREFIX128_SHA256 = "9e1bfa90057c9b72f02a29340d91f0d87d4d4f35119733824b9659bd7ee8db89"
EXPECTED_TAIL32_SHA256 = "f38fb6f6542c267b6c7d9cbec9ce57abdf9b0657edfe8d060fe533178a9f5b29"
EXPECTED_EDEV_OUTPUTS = {
    "per_sample_required": ["edev_cosine", "native_edev_cosine"],
    "summary_required": ["candidate_edev_source", "native_edev_source"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    return rows


def expected_tail(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(pool) != 2045:
        raise ValueError(f"eligible pool must contain 2045 rows, got {len(pool)}")
    sample_ids = [row.get("sample_id") for row in pool]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("eligible pool has duplicate sample IDs")
    selected: list[dict[str, Any]] = []
    for label in range(8):
        label_rows = [row for row in pool if row.get("label") == label]
        if len(label_rows) < 4:
            raise ValueError(f"label {label} has fewer than four eligible rows")
        for row in label_rows:
            sharpness = row.get("native_sharpness")
            if (
                not isinstance(sharpness, (int, float))
                or isinstance(sharpness, bool)
                or not math.isfinite(float(sharpness))
            ):
                raise ValueError(f"{row.get('sample_id')}: invalid native sharpness")
        ranked = sorted(
            label_rows,
            key=lambda row: (-float(row["native_sharpness"]), str(row["sample_id"])),
        )
        for tail_rank, row in enumerate(ranked[:4]):
            selected.append(
                {
                    "label": label,
                    "matched_native_sharpness": row["native_sharpness"],
                    "sample_id": row["sample_id"],
                    "tail_rank": tail_rank,
                }
            )
    return selected


def validate_configs(run_contracts: dict[str, Any]) -> None:
    allowed_changes = {
        "experiment_name",
        "out_dir",
        "phase",
        "sample_id_manifest",
        "sample_id_manifest_sha256",
        "calibration_sample_id_manifest",
        "calibration_sample_id_manifest_sha256",
        "max_samples",
        "calibration_samples",
    }
    runs = run_contracts.get("runs")
    if not isinstance(runs, list) or len(runs) != 4:
        raise ValueError("run contract must contain exactly four runs")
    if {run.get("gpu_index") for run in runs} != {0, 1, 2, 3}:
        raise ValueError("run contracts must bind GPUs 0, 1, 2, and 3 exactly once")
    expected_matrix = {
        ("committed_prefix128", 0.25, 128),
        ("committed_prefix128", 0.5, 128),
        ("sharpness_tail32", 0.25, 32),
        ("sharpness_tail32", 0.5, 32),
    }
    actual_matrix: set[tuple[str, float, int]] = set()
    for run in runs:
        config_path = ROOT / str(run["config"])
        if sha256(config_path) != run.get("config_sha256"):
            raise ValueError(f"{run.get('arm_id')}: config SHA256 mismatch")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        eta = float(config["eta"])
        role = str(run["selection_role"])
        count = int(run["sample_count"])
        actual_matrix.add((role, eta, count))
        if (
            config.get("mode") != "initial_noise"
            or config.get("phase") != "calibration"
            or config.get("projection") != "fixed_radius"
            or config.get("num_updates") != 16
            or config.get("batch_size") != 2
            or config.get("seed") != 1337
            or config.get("sampling_seed") != 1337
            or config.get("max_samples") != count
        ):
            raise ValueError(f"{run.get('arm_id')}: frozen run semantics drifted")
        effective_config = resolve_frozen_effective_guidance_config(config)
        if not _edev_required_for_config(effective_config):
            raise ValueError(
                f"{run.get('arm_id')}: effective config must require Edev scoring"
            )
        if run.get("expected_outputs") != EXPECTED_EDEV_OUTPUTS:
            raise ValueError(
                f"{run.get('arm_id')}: required Edev output fields drifted"
            )
        base_name = (
            "r8_meanflow_noise_fixed_eta025.yaml"
            if eta == 0.25
            else "r8_meanflow_noise_fixed_eta05.yaml"
        )
        base = yaml.safe_load(
            (
                ROOT / "configs/medium_v2/experiments" / base_name
            ).read_text(encoding="utf-8")
        )
        reduced_config = {k: v for k, v in config.items() if k not in allowed_changes}
        reduced_base = {k: v for k, v in base.items() if k not in allowed_changes}
        if reduced_config != reduced_base:
            raise ValueError(f"{run.get('arm_id')}: R8 algorithm or asset drift")
        if run.get("environment") != {"CUDA_VISIBLE_DEVICES": str(run["gpu_index"])}:
            raise ValueError(f"{run.get('arm_id')}: CUDA binding drift")
        if run.get("retry_count", run_contracts.get("retry_count")) != 0:
            raise ValueError(f"{run.get('arm_id')}: retries must remain disabled")
    if actual_matrix != expected_matrix:
        raise ValueError(f"run matrix mismatch: {sorted(actual_matrix)}")


def main() -> int:
    for path, expected in (
        (POOL, EXPECTED_POOL_SHA256),
        (PREFIX128, EXPECTED_PREFIX128_SHA256),
        (TAIL32, EXPECTED_TAIL32_SHA256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"{path}: SHA256 mismatch")
    pool = read_jsonl(POOL)
    prefix = read_jsonl(PREFIX128)
    tail = read_jsonl(TAIL32)
    if tail != expected_tail(pool):
        raise ValueError("tail32 is not the exact top-four sharpness selection per label")
    if Counter(row["label"] for row in tail) != Counter({label: 4 for label in range(8)}):
        raise ValueError("tail32 must contain four samples for each of eight labels")
    if len({row["sample_id"] for row in tail}) != 32:
        raise ValueError("tail32 sample IDs must be unique")
    overlap = {row["sample_id"] for row in tail} & {
        row["sample_id"] for row in prefix
    }
    if len(overlap) != 2:
        raise ValueError(f"prefix128/tail32 overlap must be 2, got {len(overlap)}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("selection", {}).get("selector")
        != "top4_matched_native_sharpness_per_label_v1"
        or manifest.get("selection", {}).get("selection_role")
        != "sharpness_tail32"
    ):
        raise ValueError("selector metadata mismatch")
    run_contracts = json.loads(RUN_CONTRACTS.read_text(encoding="utf-8"))
    validate_configs(run_contracts)
    print(
        json.dumps(
            {
                "status": "validated",
                "eligible_pool": len(pool),
                "prefix128": len(prefix),
                "tail32": len(tail),
                "prefix128_tail32_overlap": len(overlap),
                "run_contracts": 4,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
