from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil


DEFAULT_MISSING_REL = "2/9db2af5a1da8bd77355e8c6a655da519a899ecc42641bf254107bfc0.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove one confirmed missing AffectNet CSV row with an audit record.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--missing-rel", default=DEFAULT_MISSING_REL)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV does not exist: {args.csv}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    backup = args.out_dir / f"{args.csv.name}.before_missing_row_removal.bak"
    removed_json = args.out_dir / "removed_missing_training_row.json"
    if not backup.exists():
        shutil.copy2(args.csv, backup)
    rows = []
    removed = []
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV has no header: {args.csv}")
        for line_no, row in enumerate(reader, start=2):
            if row.get("subDirectory_filePath") == args.missing_rel:
                removed.append({"line_no": line_no, "row": row})
            else:
                rows.append(row)
    if len(removed) != 1:
        raise ValueError(f"Expected exactly one matching row for {args.missing_rel}, found {len(removed)}")
    tmp = args.csv.with_suffix(args.csv.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(args.csv)
    audit = {"csv": str(args.csv), "backup": str(backup), "removed_count": len(removed), "remaining_rows": len(rows), "removed": removed}
    removed_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

