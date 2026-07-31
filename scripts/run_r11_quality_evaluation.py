#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Sequence

CORE_PATH = Path(__file__).resolve().with_name("eval_generation_quality.py")
CORE_SPEC = importlib.util.spec_from_file_location(
    "r11_official_quality_core", CORE_PATH
)
if CORE_SPEC is None or CORE_SPEC.loader is None:
    raise RuntimeError(f"cannot load official quality core: {CORE_PATH}")
CORE_MODULE = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(CORE_MODULE)
evaluate_generation_quality = CORE_MODULE.evaluate_generation_quality


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official quality core with the locked R11 KID contract."
    )
    parser.add_argument("--real-index", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id-manifest", type=Path, required=True)
    parser.add_argument("--per-sample-jsonl", type=Path, required=True)
    parser.add_argument("--generation-result", type=Path)
    parser.add_argument("--reuse-valid-output", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        choices=("fid", "kid", "niqe", "sharpness"),
    )
    parser.add_argument("--kid-subset-size", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = tuple(args.metrics)
    sample_count = sum(
        1
        for line in args.sample_id_manifest.read_text(encoding="utf-8").splitlines()
        if line
    )
    if "kid" in metrics:
        if args.kid_subset_size != sample_count - 1:
            raise ValueError(
                "R11 KID subset size must equal locked sample_count - 1"
            )
    elif args.kid_subset_size is not None:
        raise ValueError("non-KID R11 quality forbids --kid-subset-size")
    evaluate_generation_quality(
        real_index=args.real_index,
        generated_dir=args.generated_dir,
        output=args.output,
        metrics=metrics,
        subset_seed=args.seed,
        kid_subset_size=(
            args.kid_subset_size if args.kid_subset_size is not None else 50
        ),
        device=args.device,
        sample_id_manifest=args.sample_id_manifest,
        per_sample_jsonl=args.per_sample_jsonl,
        generation_result=args.generation_result,
        reuse_valid_output=args.reuse_valid_output,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"R11 quality evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
