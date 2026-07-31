#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from safa.evaluation.meanflow_guidance_runner import read_ordered_sample_manifest
from safa.evaluation.r12_latent_fourier_replay import (
    BOOTSTRAP_ITERATIONS,
    R12FourierReplayError,
    frozen_criteria,
    high_frequency_drift,
    merge_and_bind_replay_rows,
    read_jsonl,
)


ROOT = Path("artifacts/r12_latent_fourier_replay")
R12 = Path("artifacts/r12_seed_aligned_trajectory")


def _quality_rows(dataset_id: str, arm_id: str) -> tuple[dict, dict]:
    base = R12 / "evaluation_v1" / dataset_id / "quality"
    native = json.loads((base / "native" / "quality.json").read_text(encoding="utf-8"))
    candidate = json.loads(
        (base / arm_id / "quality.json").read_text(encoding="utf-8")
    )
    return (
        {row["sample_id"]: row for row in native["per_sample_metrics"]["rows"]},
        {row["sample_id"]: row for row in candidate["per_sample_metrics"]["rows"]},
    )


def _arcface_rows(dataset_id: str, arm_id: str) -> dict[str, dict]:
    path = R12 / "evaluation_v1" / dataset_id / "arcface" / arm_id / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["sample_id"]: row for row in payload["result"]}


def _dataset_rows(
    *,
    dataset_id: str,
    arm_id: str,
    manifest_path: Path,
    lane_root: Path,
) -> list[dict]:
    expected_ids = [
        row["sample_id"] for row in read_ordered_sample_manifest(manifest_path)
    ]
    existing = read_jsonl(R12 / "runs_v1" / arm_id / "per_sample.jsonl")
    replay = merge_and_bind_replay_rows(
        [lane_root / "lane0" / "per_sample.jsonl", lane_root / "lane1" / "per_sample.jsonl"],
        expected_sample_ids=expected_ids,
        existing_u16_rows=existing,
    )
    native_quality, candidate_quality = _quality_rows(dataset_id, arm_id)
    arcface = _arcface_rows(dataset_id, arm_id)
    rows = []
    for row in replay:
        sample_id = str(row["sample_id"])
        native = native_quality[sample_id]
        candidate = candidate_quality[sample_id]
        face = arcface[sample_id]
        if not (
            face["source_face_count"]
            == face["native_face_count"]
            == face["candidate_face_count"]
            == 1
        ):
            raise R12FourierReplayError(
                f"ArcFace exact-one binding failed for {sample_id}"
            )
        retention = float(candidate["sharpness"]) / float(native["sharpness"])
        privacy = float(face["source_candidate_cosine"]) - float(
            face["source_native_cosine"]
        )
        values = {
            "sample_id": sample_id,
            "dataset_id": dataset_id,
            "original_ordinal": int(row["original_ordinal"]),
            "h12": high_frequency_drift(row, 12),
            "h16": high_frequency_drift(row, 16),
            "sharpness_retention": retention,
            "arcface_delta": privacy,
        }
        if not all(
            math.isfinite(value)
            for key, value in values.items()
            if key not in {"sample_id", "dataset_id"}
        ):
            raise R12FourierReplayError(f"non-finite classified row {sample_id}")
        rows.append(values)
    return rows


def classify(replay_root: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse result directory {output_dir}")
    regular = _dataset_rows(
        dataset_id="regular32",
        arm_id="u16_regular32",
        manifest_path=Path(
            "artifacts/r10_triangle_exploration/preparation_v1/prefix32.jsonl"
        ),
        lane_root=replay_root / "regular32",
    )
    tail = _dataset_rows(
        dataset_id="sharpness_tail32",
        arm_id="u16_tail32",
        manifest_path=Path(
            "artifacts/r11_initial_noise_sharpness_probe/preparation_v1/tail32.jsonl"
        ),
        lane_root=replay_root / "sharpness_tail32",
    )
    decision = frozen_criteria(
        regular, tail, iterations=BOOTSTRAP_ITERATIONS
    )
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r12_latent_fourier_replay_result_v1",
        "status": "complete",
        "sample_counts": {"regular32": len(regular), "sharpness_tail32": len(tail)},
        "loss_history_binding": "exact_passed",
        "merge_order_binding": "exact_passed",
        "snapshot_steps": [0, 12, 16],
        "high_frequency_definition": "per_channel_rfft_radius_at_least_nyquist_over_2",
        "criteria": decision,
        "projection_implemented": False,
        "training_started": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "per_sample.jsonl").open("x", encoding="utf-8") as handle:
        for row in [*regular, *tail]:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    licensed = decision["fourier_projection_licensed"]
    lines = [
        "# R12 latent Fourier-shell replay diagnostic",
        "",
        f"- Fourier projection licensed: `{str(licensed).lower()}`.",
        f"- Stop required: `{str(decision['stop_required']).lower()}`.",
        "- Replay loss histories and ordered sample bindings passed exactly.",
        "- No images were written or evaluated by the replay.",
        "- No projection or training was run.",
    ]
    (output_dir / "conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify frozen R12 Fourier evidence.")
    parser.add_argument("--replay-root", type=Path, default=ROOT / "replay_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_v1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = classify(args.replay_root, args.output_dir)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
