#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from safa.evaluation.triangle_screening import (
    HISTORICAL_FAMILY_IDS,
    TriangleScreeningError,
    load_historical_primary_artifacts,
    select_historical24,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the locked historical triangle-screening 24."
    )
    parser.add_argument("--historical-primary-root", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.manifest_json.is_file():
        raise TriangleScreeningError(f"missing input: {args.manifest_json}")
    try:
        manifest_payload = json.loads(
            args.manifest_json.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise TriangleScreeningError(
            f"invalid JSON in {args.manifest_json}"
        ) from exc
    if not isinstance(manifest_payload, Mapping):
        raise TriangleScreeningError("manifest JSON must be an object")
    candidates, manifest = load_historical_primary_artifacts(
        args.historical_primary_root, manifest_payload
    )
    selected = select_historical24(
        candidates,
        manifest,
        family_ids=HISTORICAL_FAMILY_IDS,
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise TriangleScreeningError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "selected24.json"
    csv_path = output_dir / "selected24.csv"
    with json_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "contract_type": "safa_triangle_historical24_v1",
                    "candidate_count": 193,
                    "selected_count": 24,
                    "selected": selected,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        fieldnames = list(selected[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TriangleScreeningError as exc:
        print(f"triangle candidate selection failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
