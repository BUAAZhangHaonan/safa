#!/usr/bin/env python3
"""Materialize the only R14 train1024 and nested single-S eval manifests."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from safa.data.r14_spatial import (
    build_eval_records,
    build_self_free_same_label_pairs,
    build_spatial_index_from_affectnet_csv,
    write_eval_manifest,
    write_pair_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_INDEX = REPO_ROOT / "data/index/train_face_mixed_e14_4029avail.jsonl"
VAL_INDEX = REPO_ROOT / "data/index/val_face_mixed_e14.jsonl"
PREFIX32 = REPO_ROOT / "artifacts/r10_triangle_exploration/preparation_v1/prefix32.jsonl"
TRAIN_CSV = Path("/home/hdd3/zhanghaonan/AffectNet/training.csv")
VAL_CSV = Path("/home/hdd3/zhanghaonan/AffectNet/validation.csv")
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/r14_inpaint_feasibility/v1/manifests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _rows(path: Path) -> list[Mapping[str, object]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(value, Mapping) for value in values):
        raise RuntimeError(f"{path} contains a non-object row")
    return values


def _write_rows(path: Path, rows: list[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace R14 manifests: {output}")
    output.mkdir(parents=True, exist_ok=False)
    train_rows = _rows(TRAIN_INDEX)
    if len(train_rows) != 30000:
        raise RuntimeError(f"locked training index must contain 30000 rows, got {len(train_rows)}")
    train1024_rows = train_rows[:1024]
    train1024_index = output / "train1024_source_index.jsonl"
    _write_rows(train1024_index, train1024_rows)
    train_spatial = build_spatial_index_from_affectnet_csv(train1024_index, [TRAIN_CSV])
    pairs = build_self_free_same_label_pairs(
        train_spatial, train_spatial, pairing_seed=1337, pairs_per_source=1
    )
    if len(pairs) != 1024 or any(pair.source_sample_id == pair.target_sample_id for pair in pairs):
        raise RuntimeError("R14 train1024 pairing count or self-free invariant failed")
    write_pair_manifest(pairs, output / "train_pairs.jsonl")

    val_rows = _rows(VAL_INDEX)
    val_by_id = {str(row["sample_id"]): row for row in val_rows}
    selection = _rows(PREFIX32)
    ordered_ids = [str(row["sample_id"]) for row in selection]
    if len(ordered_ids) != 32 or len(set(ordered_ids)) != 32:
        raise RuntimeError("locked prefix32 must contain 32 unique IDs")
    if any(sample_id not in val_by_id for sample_id in ordered_ids):
        raise RuntimeError("locked prefix32 contains an ID missing from val index")
    regular_index = output / "regular32_source_index.jsonl"
    _write_rows(regular_index, [val_by_id[sample_id] for sample_id in ordered_ids])
    regular_spatial = build_spatial_index_from_affectnet_csv(regular_index, [VAL_CSV])
    regular_eval = build_eval_records(regular_spatial)
    write_eval_manifest(regular_eval, output / "regular32.jsonl")
    first_by_label = {}
    for record in regular_eval:
        first_by_label.setdefault(record.affect_label, record)
    if set(first_by_label) != set(range(8)):
        raise RuntimeError("regular32 does not contain all eight AffectNet labels")
    fixed8 = [first_by_label[label] for label in range(8)]
    write_eval_manifest(fixed8, output / "smoke8.jsonl")
    write_eval_manifest(fixed8, output / "visual8.jsonl")
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r14_inpaint_manifest_preparation_v1",
        "train_pair_count": len(pairs),
        "optimizer_steps": 256,
        "global_batch_size": 4,
        "regular32_count": len(regular_eval),
        "smoke8_count": len(fixed8),
        "visual8_count": len(fixed8),
        "pairing_seed": 1337,
        "bbox_expansion_pixels": 0,
        "morphology": None,
        "train_pair_role": "A_cached_affect_to_B_clean_target_context_mask",
        "eval_role": "single_S_cached_affect_original_context_mask",
    }
    with (output / "preparation_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
