from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from safa.data.affectnet_index import build_affectnet_index
from safa.data.index_schema import write_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict AffectNet JSONL index.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-version", default="affectnet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label-policy", choices=("strict_0_7", "affectnet8"), default="strict_0_7")
    parser.add_argument("--csv-image-prefix", default="Manually_Annotated_Images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_affectnet_index(
        root=args.root,
        default_split=args.split,
        dataset_version=args.dataset_version,
        limit=args.limit,
        label_policy=args.label_policy,
        csv_image_prefix=args.csv_image_prefix,
    )
    write_index(records, args.out)
    split_counts = Counter(record.split for record in records)
    label_counts = Counter(record.label for record in records)
    print(f"wrote {len(records)} records to {args.out}")
    print(f"split_counts={dict(sorted(split_counts.items()))}")
    print(f"label_counts={dict(sorted(label_counts.items()))}")


if __name__ == "__main__":
    main()
