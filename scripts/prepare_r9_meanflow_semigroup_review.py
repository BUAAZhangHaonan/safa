#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from safa.evaluation.r9_semigroup_campaign_closure import (
    prepare_campaign_semigroup_visual_review,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the immutable blinded 64-sample R9 semigroup review "
            "assignment for one bootstrap/formal campaign pair."
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
        "--formal-campaign-id",
        required=True,
        help="Distinct immutable formal R9 campaign ID to bind before review.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare_campaign_semigroup_visual_review(
        config_path=args.config,
        shard_root=args.shard_root,
        formal_campaign_id=args.formal_campaign_id,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
