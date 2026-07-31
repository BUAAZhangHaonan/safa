#!/usr/bin/env python3
"""Render the fixed eight R14 source/native/candidate triptychs."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[Mapping[str, object]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(value, Mapping) for value in values):
        raise RuntimeError(f"{path} contains a non-object row")
    return values


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse visual output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    visual_rows = _rows(args.manifest)
    if len(visual_rows) != 8:
        raise RuntimeError(f"visual8 manifest must contain 8 rows, got {len(visual_rows)}")
    generation = {str(row["sample_id"]): row for row in _rows(args.generation_dir / "per_sample.jsonl")}
    ordered_ids = [str(row["sample_id"]) for row in visual_rows]
    if len(set(ordered_ids)) != 8 or any(sample_id not in generation for sample_id in ordered_ids):
        raise RuntimeError("visual8 IDs must be eight unique regular32 IDs")
    contact = Image.new("RGB", (3 * 256, 8 * 284), "white")
    rendered = []
    for index, sample_id in enumerate(ordered_ids):
        row = generation[sample_id]
        triptych = Image.new("RGB", (3 * 256, 284), "white")
        draw = ImageDraw.Draw(triptych)
        for column, (label, field) in enumerate((("source", "source"), ("matched native", "native"), ("candidate", "generated"))):
            with Image.open(str(row[field])) as image:
                rgb = image.convert("RGB")
                if rgb.size != (256, 256):
                    raise RuntimeError(f"visual image must be 256x256: {row[field]}")
                triptych.paste(rgb, (column * 256, 28))
            draw.text((column * 256 + 6, 7), label, fill="black")
        path = args.output_dir / f"{index:02d}.png"
        triptych.save(path, format="PNG")
        contact.paste(triptych, (0, index * 284))
        rendered.append({"ordinal": index, "sample_id": sample_id, "path": str(path.resolve())})
    contact_path = args.output_dir / "contact_sheet.png"
    contact.save(contact_path, format="PNG")
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r14_inpaint_visual8_v1",
        "sample_count": 8,
        "columns": ["source", "matched_native", "candidate"],
        "rows": rendered,
        "contact_sheet": str(contact_path.resolve()),
        "review_status": "pending_human_review",
        "severe_count": None,
        "pass_rule": "severe_count == 0",
    }
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
