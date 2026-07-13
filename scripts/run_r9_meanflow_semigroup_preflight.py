#!/usr/bin/env python3
from __future__ import annotations

import os

# This entry point is R9-only, so the cuBLAS contract is fixed before importing torch.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
from pathlib import Path
import sys

from safa.evaluation.meanflow_guidance_runner import (
    run_guidance_from_config,
    validate_guidance_config,
)
from safa.evaluation.r9_determinism import canonical_r9_arm_config_digest
from safa.evaluation.r9_semigroup_contracts import (
    canonical_r9_semigroup_preflight_payload,
    finalize_r9_semigroup_preflight,
)
from safa.utils.config import load_yaml


CONFIG = Path("configs/medium_v2/experiments/r9_meanflow_semigroup_preflight.yaml")
OUTPUT_ROOT = Path("artifacts/r9_meanflow_flow_map_guidance/semigroup/preflight/shards")
FINAL_ROOT = Path("artifacts/r9_meanflow_flow_map_guidance/semigroup")
VISUAL_REVIEW = FINAL_ROOT / "visual_review.json"
NUM_SHARDS = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run or execute one locked R9 semigroup preflight shard."
    )
    parser.add_argument("--shard-index", type=int, choices=range(NUM_SHARDS), default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    return parser.parse_args(argv)


def build_dry_run(args: argparse.Namespace) -> dict:
    config = _effective_config()
    return {
        "schema_version": 1,
        "execute": bool(args.execute),
        "allow_busy_gpus": bool(args.allow_busy_gpus),
        "config": str(CONFIG),
        "output_dir": str(OUTPUT_ROOT / f"shard_{args.shard_index}"),
        "shard": {"index": int(args.shard_index), "count": NUM_SHARDS},
        "arm_config_sha256": config["arm_config_sha256"],
        "preflight_contract": canonical_r9_semigroup_preflight_payload(config),
        "semigroup_preflight_contract_sha256": config[
            "semigroup_preflight_contract_sha256"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.execute and args.finalize:
            raise ValueError("--execute and --finalize are mutually exclusive")
        if args.finalize:
            result = _finalize()
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0
        dry_run = build_dry_run(args)
        if not args.execute:
            print(json.dumps(dry_run, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if not args.allow_busy_gpus:
            raise ValueError("--execute requires explicit --allow-busy-gpus authorization")
        config = _effective_config()
        result = run_guidance_from_config(
            config,
            output_dir=dry_run["output_dir"],
            shard_index=args.shard_index,
            num_shards=NUM_SHARDS,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _effective_config() -> dict:
    config = validate_guidance_config(load_yaml(CONFIG))
    config["arm_config_sha256"] = canonical_r9_arm_config_digest(config)
    return config


def _finalize() -> dict:
    if not VISUAL_REVIEW.is_file():
        raise FileNotFoundError(f"R9 finalize requires visual review: {VISUAL_REVIEW}")
    review = json.loads(VISUAL_REVIEW.read_text(encoding="utf-8"))
    if not isinstance(review, dict) or review.get("schema_version") != 1:
        raise ValueError("R9 visual review must be a schema_version=1 object")
    if review.get("sample_count") != 64 or not isinstance(review.get("splits"), dict):
        raise ValueError("R9 visual review must cover all 64 samples")
    split_review = review["splits"]
    if set(split_review) != {"0.25", "0.5", "0.75"}:
        raise ValueError("R9 visual review must cover splits 0.25, 0.5, and 0.75")
    visual_pass = {}
    for split, payload in split_review.items():
        if not isinstance(payload, dict) or not isinstance(payload.get("passed"), bool):
            raise ValueError(f"R9 visual review split {split} requires boolean passed")
        visual_pass[split] = payload["passed"]
    return finalize_r9_semigroup_preflight(
        _effective_config(),
        [OUTPUT_ROOT / f"shard_{index}" for index in range(NUM_SHARDS)],
        output_dir=FINAL_ROOT,
        visual_pass_by_split=visual_pass,
    )


if __name__ == "__main__":
    raise SystemExit(main())
