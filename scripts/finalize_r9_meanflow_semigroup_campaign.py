#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from safa.evaluation.r9_semigroup_campaign_closure import (
    finalize_campaign_semigroup_closure,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and immutably seal one campaign-specific R9 semigroup "
            "bootstrap preflight for a distinct formal campaign."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Exact immutable bootstrap CID semigroup runtime YAML.",
    )
    parser.add_argument(
        "--shard-root",
        required=True,
        type=Path,
        help="Exact bootstrap CID directory containing shard_0 through shard_3.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New canonical bootstrap-to-formal campaign closure directory.",
    )
    parser.add_argument(
        "--visual-review",
        required=True,
        type=Path,
        help="Complete v2 visual review bound to the rehashed evidence manifest.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = finalize_campaign_semigroup_closure(
        config_path=args.config,
        shard_root=args.shard_root,
        output_root=args.output_root,
        visual_review_path=args.visual_review,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
