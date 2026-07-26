from __future__ import annotations

import argparse
import json
from pathlib import Path

from safa.closeout.ledger import build_closeout_snapshot, write_closeout_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable SAFA historical experiment closeout ledger."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write a new output directory. Without this flag the command is a dry run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = build_closeout_snapshot(args.repo_root)
    statuses: dict[str, int] = {}
    for row in snapshot["rows"]:
        status = row["status"]
        statuses[status] = statuses.get(status, 0) + 1
    summary = {
        "run_count": len(snapshot["rows"]),
        "artifact_count": len(snapshot["artifact_manifest"]),
        "statuses": dict(sorted(statuses.items())),
        "mode": "execute" if args.execute else "dry_run",
    }
    if args.execute:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required with --execute")
        written = write_closeout_snapshot(snapshot, args.output_dir)
        summary["output_dir"] = str(written)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
