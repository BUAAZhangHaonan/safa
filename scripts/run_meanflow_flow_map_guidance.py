#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from safa.evaluation.meanflow_guidance_runner import SUPPORTED_MODES, run_guidance_from_config
from safa.utils.config import load_yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen MeanFlow flow-map guidance evaluation.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--num-updates", type=int)
    parser.add_argument("--projection", choices=("fixed_radius", "typical_shell"))
    parser.add_argument("--semigroup-report", type=Path)
    parser.add_argument("--schedule-manifest", type=Path)
    parser.add_argument("--t-cut", type=float)
    parser.add_argument(
        "--fmrg-variant",
        choices=("official_head_current_xt", "paper_algorithm_split"),
    )
    parser.add_argument("--sample-mode", choices=("flow_map1", "flow_map2"))
    parser.add_argument(
        "--optimization-mode",
        choices=("official_adam", "paper_normalized_direct_autograd"),
    )
    parser.add_argument("--num-optim-iters", type=int)
    return parser.parse_args(argv)


def resolved_config(args: argparse.Namespace) -> dict:
    config = load_yaml(args.config)
    if args.mode is not None and args.fmrg_variant is not None and args.mode != args.fmrg_variant:
        raise ValueError("--mode and --fmrg-variant disagree")
    overrides = {
        "mode": args.fmrg_variant if args.fmrg_variant is not None else args.mode,
        "max_samples": args.max_samples,
        "eta": args.eta,
        "step_size": args.step_size,
        "num_updates": args.num_updates,
        "projection": args.projection,
        "semigroup_report": args.semigroup_report,
        "schedule_manifest": args.schedule_manifest,
        "t_cut": args.t_cut,
        "sample_mode": args.sample_mode,
        "optimization_mode": args.optimization_mode,
        "num_optim_iters": args.num_optim_iters,
    }
    for field, value in overrides.items():
        if value is not None:
            config[field] = str(value) if isinstance(value, Path) else value
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = resolved_config(args)
        manifest = run_guidance_from_config(
            config,
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            explicit_t_cut=args.t_cut,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
