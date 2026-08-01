#!/usr/bin/env python3
"""Run the bounded R14 EMA-milestone regular32/tail32 evaluation.

This coordinator never trains or changes the official R14 evaluator. It only
exports an EMA state, launches the independent two-rank generator, invokes the
official NIQE/sharpness/ArcFace/E0/edev evaluator, and records a compact
per-checkpoint summary.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from safa.data.r14_spatial import build_eval_records, build_spatial_index_from_affectnet_csv, write_eval_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
CONFIG = REPO_ROOT / "configs/medium_v2/experiments/r14_inpaint_resume_gpu12_epoch100.yaml"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints/r14_inpaint_resume_gpu12_epoch100"
REGULAR_MANIFEST = REPO_ROOT / "artifacts/r14_inpaint_feasibility/v1/manifests/regular32.jsonl"
TAIL_IDS = REPO_ROOT / "artifacts/r11_initial_noise_sharpness_probe/preparation_v1/tail32.jsonl"
VAL_CSV = Path("/home/hdd3/zhanghaonan/AffectNet/validation.csv")
CHECKPOINT_STEPS = (2560, 5120, 7680, 10240, 12800)
DATASETS = ("regular32", "tail32")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{path} contains a non-object row")
    return rows


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _run(argv: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
    print("+", " ".join(str(value) for value in argv), flush=True)
    subprocess.run(list(argv), cwd=REPO_ROOT, env=dict(env) if env is not None else None, check=True)


def _validate_config() -> dict[str, object]:
    import yaml

    config = _mapping(yaml.safe_load(CONFIG.read_text(encoding="utf-8")), "config")
    if config.get("sampling_seed") != 1337 or config.get("seed") != 1337:
        raise RuntimeError("R14 milestone evaluation requires seed and sampling_seed equal to 1337")
    if config.get("eval_index") != "data/index/val_face_mixed_e14.jsonl":
        raise RuntimeError("R14 milestone evaluation requires the locked validation index")
    if config.get("e0_checkpoint") != "artifacts/checkpoints/e0_medium_v1/best.pt":
        raise RuntimeError("R14 milestone evaluation requires the locked E0 checkpoint")
    return dict(config)


def _materialize_tail_manifest(output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to replace tail manifest: {output}")
    tail = _read_jsonl(TAIL_IDS)
    ids = [str(row.get("sample_id")) for row in tail]
    if len(ids) != 32 or len(set(ids)) != 32 or any(not value or value == "None" for value in ids):
        raise RuntimeError("tail32 sample-ID manifest must contain 32 unique IDs")
    val_index = _read_json(REPO_ROOT / "data/index/val_face_mixed_e14.jsonl")
    by_id = {str(row["sample_id"]): row for row in val_index}
    missing = [sample_id for sample_id in ids if sample_id not in by_id]
    if missing:
        raise RuntimeError(f"tail32 IDs missing from validation index: {missing[:4]}")
    source_index = output.parent / "tail32_source_index.jsonl"
    source_index.parent.mkdir(parents=True, exist_ok=True)
    source_index.write_text(
        "".join(json.dumps(by_id[sample_id], sort_keys=True, allow_nan=False) + "\n" for sample_id in ids),
        encoding="utf-8",
    )
    spatial = build_spatial_index_from_affectnet_csv(source_index, [VAL_CSV])
    spatial_by_id = {record.sample_id: record for record in spatial}
    ordered = [spatial_by_id[sample_id] for sample_id in ids]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_eval_manifest(build_eval_records(ordered), output)
    return output


def _read_json(path: Path) -> list[dict[str, object]]:
    rows = _read_jsonl(path)
    return rows


def _validate_manifest(path: Path) -> list[dict[str, object]]:
    rows = _read_jsonl(path)
    if len(rows) != 32:
        raise RuntimeError(f"manifest must contain 32 rows: {path}")
    ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != 32:
        raise RuntimeError(f"manifest sample IDs must be 32 unique non-empty strings: {path}")
    return rows


def _env() -> dict[str, str]:
    values = dict(os.environ)
    values.update(
        {
            "CUDA_VISIBLE_DEVICES": "1,2",
            "PYTHONPATH": f"src{os.pathsep}{values.get('PYTHONPATH', '')}".rstrip(os.pathsep),
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONUNBUFFERED": "1",
            "NCCL_IB_DISABLE": "1",
            "NCCL_P2P_DISABLE": "0",
        }
    )
    return values


def _checkpoint_path(step: int) -> Path:
    path = CHECKPOINT_ROOT / f"step_{step:08d}.pt"
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing checkpoint milestone: {path}")
    return path


def _summarize_dataset(step: int, dataset: str, generation: Path, evaluation: Path) -> dict[str, object]:
    summary = _mapping(json.loads((evaluation / "summary.json").read_text(encoding="utf-8")), "official summary")
    metrics = _mapping(summary.get("metrics"), "official metrics")
    arcface = _mapping(summary.get("arcface"), "official arcface")
    arc_rows = _mapping(json.loads((evaluation / "arcface_result.json").read_text(encoding="utf-8")), "ArcFace result")
    result = arc_rows.get("result")
    if not isinstance(result, list) or len(result) != 32:
        raise RuntimeError("ArcFace result must contain exactly 32 rows")
    deltas: list[float] = []
    for row in result:
        row_map = _mapping(row, "ArcFace row")
        candidate = row_map.get("source_candidate_cosine")
        native = row_map.get("source_native_cosine")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in (candidate, native)):
            raise RuntimeError("ArcFace pair cosine is missing or non-finite")
        deltas.append(float(candidate) - float(native))
    point_delta = sum(deltas) / len(deltas)
    generation_completion = _mapping(json.loads((generation / "completion.json").read_text(encoding="utf-8")), "generation completion")
    return {
        "checkpoint_step": step,
        "dataset": dataset,
        "sample_count": 32,
        "generation_contract": generation_completion.get("contract_type"),
        "e0": metrics.get("e0"),
        "delta_e0": metrics.get("delta_e0"),
        "delta_edev": metrics.get("delta_edev"),
        "full_niqe": metrics.get("full_niqe"),
        "full_native_niqe": metrics.get("full_native_niqe"),
        "full_sharpness": metrics.get("full_sharpness"),
        "full_native_sharpness": metrics.get("full_native_sharpness"),
        "roi_niqe": metrics.get("roi_niqe"),
        "roi_native_niqe": metrics.get("roi_native_niqe"),
        "roi_sharpness": metrics.get("roi_sharpness"),
        "roi_native_sharpness": metrics.get("roi_native_sharpness"),
        "arcface_exact_one": dict(arcface),
        "arcface_delta_point": point_delta,
        "arcface_delta_u95": metrics.get("arcface_u95"),
        "official_summary": str((evaluation / "summary.json").relative_to(REPO_ROOT)),
    }


def _run_one(step: int, dataset: str, manifest: Path, root: Path, config: Path, env: Mapping[str, str]) -> dict[str, object]:
    checkpoint = _checkpoint_path(step)
    export_dir = root / "exports" / f"step_{step:08d}"
    export = export_dir / "ema.pt"
    metadata = export_dir / "ema.json"
    if not export.exists():
        _run(
            [str(PYTHON), "scripts/export_r14_inpaint_ema_milestone.py", "--checkpoint", str(checkpoint), "--output", str(export), "--metadata-output", str(metadata)],
            env=env,
        )
    run_root = root / "runs" / f"step_{step:08d}" / dataset
    generation = run_root / "generation"
    evaluation = run_root / "evaluation"
    _run(
        [str(PYTHON), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "scripts/run_r14_inpaint_milestone_generation.py", "--config", str(config), "--checkpoint", str(export), "--manifest", str(manifest), "--output-dir", str(generation)],
        env=env,
    )
    _run(
        [str(PYTHON), "scripts/evaluate_r14_inpaint_feasibility.py", "--config", str(config), "--manifest", str(manifest), "--generation-dir", str(generation), "--output-dir", str(evaluation)],
        env=env,
    )
    return _summarize_dataset(step, dataset, generation, evaluation)


def _dry_run(root: Path) -> dict[str, object]:
    _validate_config()
    regular = _validate_manifest(REGULAR_MANIFEST)
    tail = _validate_manifest(TAIL_IDS)
    missing = [step for step in CHECKPOINT_STEPS if not _checkpoint_path(step).is_file()]
    if missing:
        raise RuntimeError(f"missing milestones: {missing}")
    return {
        "status": "pass",
        "mode": "dry-run",
        "gpu_list": [1, 2],
        "world_size": 2,
        "seed": 1337,
        "sampling_seed": 1337,
        "sample_counts": {"regular32": len(regular), "tail32_ids": len(tail)},
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "official_metrics": ["niqe", "sharpness", "e0", "edev", "arcface_exact_one", "arcface_delta_point", "arcface_delta_u95"],
        "forbidden_metrics": ["fid", "kid"],
        "output_root": str(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "run"), required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts/r14_inpaint_milestone_eval/v1")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.mode == "dry-run":
        print(json.dumps(_dry_run(root), indent=2, sort_keys=True))
        return
    if root.exists():
        raise FileExistsError(f"refusing to reuse eval root: {root}")
    root.mkdir(parents=True, exist_ok=False)
    _validate_config()
    tail_manifest = _materialize_tail_manifest(root / "preparation" / "tail32.jsonl")
    _validate_manifest(REGULAR_MANIFEST)
    _validate_manifest(tail_manifest)
    env = _env()
    summary: list[dict[str, object]] = []
    _write_json(root / "run_manifest.json", {"contract_type": "safa_r14_inpaint_milestone_eval_v1", "seed": 1337, "sampling_seed": 1337, "gpu_list": [1, 2], "world_size": 2, "checkpoint_steps": list(CHECKPOINT_STEPS), "datasets": {"regular32": str(REGULAR_MANIFEST), "tail32": str(tail_manifest)}})
    for step in CHECKPOINT_STEPS:
        for dataset, manifest in (("regular32", REGULAR_MANIFEST), ("tail32", tail_manifest)):
            summary.append(_run_one(step, dataset, manifest, root, CONFIG, env))
            _write_json(root / "summary.json", {"contract_type": "safa_r14_inpaint_milestone_eval_v1", "sample_count_per_dataset": 32, "fid_kid_interpretation": "forbidden_on_32", "arms": summary})
    print(json.dumps({"status": "pass", "arms": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
