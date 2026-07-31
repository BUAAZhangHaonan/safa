#!/usr/bin/env python3
"""Thin R14 adapter over official full-image R/Q/P and exact-bbox ROI quality."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

import build_fixed32_arcface_requests as arc_builder
import eval_generation_quality as quality_core
from safa.data.r14_spatial import R14SpatialEvalDataset
from safa.evaluation.r9_evaluator_worker import build_worker_request
from safa.evaluation.r9_phase_results import ArcFaceEvaluationRequest, SampleEvidence
from safa.evaluation.triangle_screening import paired_bootstrap_upper
from safa.models.e0 import freeze_e0, load_e0_checkpoint
from safa.training.losses import normalize_for_e0
from safa.training.transforms import r14_joint_transform


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCFACE_TEMPLATE = REPO_ROOT / "artifacts/r9_meanflow_flow_map_guidance/campaigns/r9-report-only-formal-v9/full/evaluator_runs/arcface/winner/request.json"
EDEV_CHECKPOINT = REPO_ROOT / "artifacts/checkpoints/e0_resnet18/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _jsonl(path: Path) -> list[Mapping[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, Mapping) for row in rows):
        raise RuntimeError(f"{path} contains a non-object row")
    return rows


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _quality_mean(payload: Mapping[str, object], metric: str) -> float:
    block = payload.get("iqa" if metric == "niqe" else "sharpness")
    value = _mapping(block, metric).get("mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"official {metric} mean is missing or non-finite")
    return float(value)


def _official_quality(
    *,
    manifest: Path,
    directory: Path,
    per_sample: Path,
    output: Path,
) -> Mapping[str, object]:
    return quality_core.evaluate_generation_quality(
        real_index=manifest,
        generated_dir=directory,
        output=output,
        metrics=("niqe", "sharpness"),
        subset_seed=91637,
        device="cuda:0",
        sample_id_manifest=manifest,
        per_sample_jsonl=per_sample,
        reuse_valid_output=False,
    )


def _crop_exact_bbox(rows: Sequence[Mapping[str, object]], output: Path) -> tuple[Path, Path, Path, Path]:
    candidate_dir = output / "roi_candidate"
    native_dir = output / "roi_native"
    candidate_dir.mkdir()
    native_dir.mkdir()
    candidate_rows = []
    native_rows = []
    for index, row in enumerate(rows):
        bbox = row.get("bbox_xyxy_256")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RuntimeError("generation row lacks exact 256-space bbox")
        x1, y1, x2, y2 = (int(value) for value in bbox)
        if not (0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256):
            raise RuntimeError("generation row bbox is invalid")
        sample_id = str(row["sample_id"])
        candidate_path = candidate_dir / f"{index:04d}.png"
        native_path = native_dir / f"{index:04d}.png"
        with Image.open(str(row["generated"])) as image:
            image.convert("RGB").crop((x1, y1, x2, y2)).save(candidate_path, format="PNG")
        with Image.open(str(row["native"])) as image:
            image.convert("RGB").crop((x1, y1, x2, y2)).save(native_path, format="PNG")
        candidate_rows.append({"sample_id": sample_id, "generated": str(candidate_path.resolve())})
        native_rows.append({"sample_id": sample_id, "generated": str(native_path.resolve())})
    candidate_jsonl = output / "roi_candidate_per_sample.jsonl"
    native_jsonl = output / "roi_native_per_sample.jsonl"
    _write_jsonl(candidate_jsonl, candidate_rows)
    _write_jsonl(native_jsonl, native_rows)
    return candidate_dir, candidate_jsonl, native_dir, native_jsonl


def _official_arcface(rows: Sequence[Mapping[str, object]], output: Path) -> Mapping[str, object]:
    samples = tuple(
        SampleEvidence(
            sample_id=str(row["sample_id"]),
            source=Path(str(row["source"])),
            native=Path(str(row["native"])),
            candidate=Path(str(row["generated"])),
            source_sha256=arc_builder._sha256(Path(str(row["source"]))),
            native_sha256=arc_builder._sha256(Path(str(row["native"]))),
            candidate_sha256=arc_builder._sha256(Path(str(row["generated"]))),
        )
        for row in rows
    )
    template = arc_builder._template(ARCFACE_TEMPLATE)
    source_index_path, source_index_sha256 = arc_builder._source_index(template)
    config = arc_builder._production_config(template, device="cuda:0", work_root=output / "arcface_work")
    request = ArcFaceEvaluationRequest(
        phase="diagnose",
        logical_run_id="r14_inpaint_feasibility_regular32",
        arm_id="r14_inpaint_20ep_2560step",
        seed=91637,
        source_index_path=source_index_path,
        source_index_sha256=source_index_sha256,
        samples=samples,
        pair_policy="pairwise_exact_one_v1",
    )
    request_payload = build_worker_request(
        "arcface", request, config=config, contract_type="safa_r11_arcface_evaluator_request_v1"
    )
    request_path = output / "arcface_request.json"
    result_path = output / "arcface_result.json"
    _write_json(request_path, request_payload)
    subprocess.run(
        [sys.executable, "scripts/run_r9_phase_evaluator.py", "--request", str(request_path), "--output", str(result_path)],
        cwd=REPO_ROOT,
        check=True,
    )
    return _mapping(json.loads(result_path.read_text(encoding="utf-8")), "ArcFace result")


def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB")).float().div(255.0)


def _representation_rows(
    config: Mapping[str, object],
    manifest: Path,
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], Mapping[str, float]]:
    dataset = R14SpatialEvalDataset(
        manifest,
        Path(str(config["eval_index"])),
        Path(str(config["eval_features"])),
        Path(str(config["e0_checkpoint"])),
        r14_joint_transform(256, horizontal_flip_probability=0.0),
    )
    device = torch.device("cuda", 0)
    e0, _ = load_e0_checkpoint(Path(str(config["e0_checkpoint"])), device="cpu")
    edev, _ = load_e0_checkpoint(EDEV_CHECKPOINT, device="cpu")
    e0 = e0.to(device).eval()
    edev = edev.to(device).eval()
    freeze_e0(e0)
    freeze_e0(edev)
    result = []
    by_id = {str(row["sample_id"]): row for row in rows}
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            sample_id = str(item["sample_id"])
            row = by_id[sample_id]
            source_z = item["source_z"].to(device=device, dtype=torch.float32).unsqueeze(0)
            source = _load_rgb(Path(str(row["source"]))).unsqueeze(0).to(device)
            native = _load_rgb(Path(str(row["native"]))).unsqueeze(0).to(device)
            candidate = _load_rgb(Path(str(row["generated"]))).unsqueeze(0).to(device)
            native_e0 = e0(normalize_for_e0(native))["embedding"]
            candidate_e0 = e0(normalize_for_e0(candidate))["embedding"]
            source_edev = edev(normalize_for_e0(source))["embedding"]
            native_edev = edev(normalize_for_e0(native))["embedding"]
            candidate_edev = edev(normalize_for_e0(candidate))["embedding"]
            result.append(
                {
                    "sample_id": sample_id,
                    "native_e0": float(F.cosine_similarity(native_e0, source_z).item()),
                    "candidate_e0": float(F.cosine_similarity(candidate_e0, source_z).item()),
                    "native_edev": float(F.cosine_similarity(native_edev, source_edev).item()),
                    "candidate_edev": float(F.cosine_similarity(candidate_edev, source_edev).item()),
                }
            )
    def mean(field: str) -> float:
        return sum(float(row[field]) for row in result) / len(result)
    metrics = {
        "e0": mean("candidate_e0"),
        "delta_e0": mean("candidate_e0") - mean("native_e0"),
        "delta_edev": mean("candidate_edev") - mean("native_edev"),
    }
    return result, metrics


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse evaluation output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = _mapping(yaml.safe_load(args.config.read_text(encoding="utf-8")), "config")
    rows = _jsonl(args.generation_dir / "per_sample.jsonl")
    if len(rows) != 32:
        raise RuntimeError("R14 evaluation requires exactly 32 generation rows")
    candidate_quality = _official_quality(
        manifest=args.manifest,
        directory=args.generation_dir / "generated_images",
        per_sample=args.generation_dir / "per_sample.jsonl",
        output=args.output_dir / "full_candidate_quality.json",
    )
    native_rows = [{"sample_id": row["sample_id"], "generated": row["native"]} for row in rows]
    native_per_sample = args.output_dir / "full_native_per_sample.jsonl"
    _write_jsonl(native_per_sample, native_rows)
    native_quality = _official_quality(
        manifest=args.manifest,
        directory=args.generation_dir / "native_images",
        per_sample=native_per_sample,
        output=args.output_dir / "full_native_quality.json",
    )
    roi_candidate_dir, roi_candidate_rows, roi_native_dir, roi_native_rows = _crop_exact_bbox(rows, args.output_dir)
    roi_candidate_quality = _official_quality(
        manifest=args.manifest, directory=roi_candidate_dir, per_sample=roi_candidate_rows,
        output=args.output_dir / "roi_candidate_quality.json",
    )
    roi_native_quality = _official_quality(
        manifest=args.manifest, directory=roi_native_dir, per_sample=roi_native_rows,
        output=args.output_dir / "roi_native_quality.json",
    )
    arcface = _official_arcface(rows, args.output_dir)
    arc_rows = arcface.get("result")
    if not isinstance(arc_rows, list) or len(arc_rows) != 32:
        raise RuntimeError("official ArcFace result must contain 32 rows")
    counts = {
        role: sum(int(row[f"{role}_face_count"] == 1) for row in arc_rows)
        for role in ("source", "native", "candidate")
    }
    deltas = []
    for row in arc_rows:
        candidate = row.get("source_candidate_cosine")
        native = row.get("source_native_cosine")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in (candidate, native)):
            raise RuntimeError("ArcFace pair cosine is missing or non-finite")
        deltas.append(float(candidate) - float(native))
    arcface_u95 = paired_bootstrap_upper(deltas, iterations=2000, seed=91637)
    representation_rows, representation = _representation_rows(config, args.manifest, rows)
    _write_jsonl(args.output_dir / "representation_rows.jsonl", representation_rows)
    metrics = {
        **representation,
        "arcface_u95": arcface_u95,
        "full_niqe": _quality_mean(candidate_quality, "niqe"),
        "full_native_niqe": _quality_mean(native_quality, "niqe"),
        "full_sharpness": _quality_mean(candidate_quality, "sharpness"),
        "full_native_sharpness": _quality_mean(native_quality, "sharpness"),
        "roi_niqe": _quality_mean(roi_candidate_quality, "niqe"),
        "roi_native_niqe": _quality_mean(roi_native_quality, "niqe"),
        "roi_sharpness": _quality_mean(roi_candidate_quality, "sharpness"),
        "roi_native_sharpness": _quality_mean(roi_native_quality, "sharpness"),
    }
    gate = {
        "exact_one": all(value == 32 for value in counts.values()),
        "representation": metrics["e0"] >= 0.75 and metrics["delta_e0"] >= 0.30 and metrics["delta_edev"] >= 0.05,
        "privacy": metrics["arcface_u95"] <= 0.02,
        "full_quality": metrics["full_niqe"] <= metrics["full_native_niqe"] + 0.10 and metrics["full_sharpness"] >= max(300.0, 0.95 * metrics["full_native_sharpness"]),
        "roi_quality": metrics["roi_niqe"] <= metrics["roi_native_niqe"] + 0.10 and metrics["roi_sharpness"] >= max(300.0, 0.95 * metrics["roi_native_sharpness"]),
    }
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r14_inpaint_regular32_evaluation_v1",
        "sample_count": 32,
        "arcface": {role: {"exact_one": count} for role, count in counts.items()},
        "metrics": metrics,
        "gate": gate,
        "bootstrap": {"iterations": 2000, "seed": 91637, "unit": "sample_id", "privacy_statistic": "arcface_delta_u95"},
        "quality_thresholds": {"niqe_delta_max": 0.10, "sharpness_ratio_min": 0.95, "sharpness_absolute_min": 300.0},
        "claim_boundary": "regular32 feasibility only",
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
