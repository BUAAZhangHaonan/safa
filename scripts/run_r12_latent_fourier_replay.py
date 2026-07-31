#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from safa.evaluation.meanflow_guidance_runner import (
    resolve_frozen_effective_guidance_config,
)
from safa.evaluation.r12_latent_fourier_replay import run_replay_lane
from safa.utils.config import load_yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay exact R12 initial-noise paths and save latent spectra only."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dataset-id", required=True, choices=("regular32", "sharpness_tail32")
    )
    parser.add_argument("--lane-index", required=True, type=int)
    parser.add_argument("--lane-count", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = resolve_frozen_effective_guidance_config(load_yaml(args.config))
        result = run_replay_lane(
            config,
            dataset_id=args.dataset_id,
            lane_index=args.lane_index,
            lane_count=args.lane_count,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
