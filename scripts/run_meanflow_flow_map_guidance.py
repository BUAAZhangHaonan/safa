#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from safa.evaluation.meanflow_guidance_runner import (
    SUPPORTED_MODES,
    finalize_effective_guidance_config as _finalize_effective_guidance_config,
    resolve_frozen_effective_guidance_config,
    run_guidance_from_config,
)
from safa.utils.config import load_yaml


SEMANTIC_OVERRIDE_FIELDS = frozenset(
    {
        "mode",
        "max_samples",
        "eta",
        "step_size",
        "num_updates",
        "projection",
        "semigroup_report",
        "schedule_manifest",
        "t_cut",
        "sample_mode",
        "optimization_mode",
        "num_optim_iters",
        "out_dir",
        "phase",
        "sample_id_manifest",
        "sample_id_manifest_sha256",
        "contact_sheets",
    }
)


def finalize_effective_guidance_config(
    validated_config: Mapping[str, Any],
    *,
    locked_schedule: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _finalize_effective_guidance_config(
        validated_config,
        locked_schedule=locked_schedule,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen MeanFlow flow-map guidance evaluation."
    )
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


def semantic_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.mode is not None
        and args.fmrg_variant is not None
        and args.mode != args.fmrg_variant
    ):
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
    return {field: value for field, value in overrides.items() if value is not None}


def resolve_guidance_semantics(
    base_config: Mapping[str, Any], semantic_overrides: Mapping[str, Any]
) -> dict[str, Any]:
    unknown = sorted(set(semantic_overrides) - SEMANTIC_OVERRIDE_FIELDS)
    if unknown:
        raise ValueError(f"unsupported semantic guidance overrides: {unknown!r}")
    resolved = dict(base_config)
    for field, value in semantic_overrides.items():
        if value is None:
            resolved.pop(field, None)
        else:
            resolved[field] = str(value) if isinstance(value, Path) else value
    return resolved


def resolve_effective_guidance_config(
    base_config: Mapping[str, Any], semantic_overrides: Mapping[str, Any]
) -> dict[str, Any]:
    semantic_config = resolve_guidance_semantics(base_config, semantic_overrides)
    return resolve_frozen_effective_guidance_config(semantic_config)


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    return resolve_effective_guidance_config(
        load_yaml(args.config),
        semantic_overrides_from_args(args),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = resolved_config(args)
        manifest = run_guidance_from_config(
            config,
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            explicit_t_cut=None,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
