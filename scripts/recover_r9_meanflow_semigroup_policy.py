#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from safa.evaluation.r9_semigroup_campaign_closure import (
    finalize_campaign_semigroup_policy_recovery,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seal the immutable user-authorized R9 semigroup recovery policy."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--policy-campaign-id", required=True)
    parser.add_argument("--formal-campaign-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--source-terminal-failure", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = finalize_campaign_semigroup_policy_recovery(
        config_path=args.config,
        shard_root=args.shard_root,
        policy_campaign_id=args.policy_campaign_id,
        formal_campaign_id=args.formal_campaign_id,
        output_root=args.output_root,
        visual_review_path=args.visual_review,
        source_terminal_failure_path=args.source_terminal_failure,
        user_recovery_authorization_id=args.authorization_id,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
