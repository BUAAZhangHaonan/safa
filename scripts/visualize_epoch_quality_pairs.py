#!/usr/bin/env python3
"""Visualize source/generated pairs from a quality-eval epoch directory."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TILE_WIDTH = 340
TILE_HEIGHT = 280
TITLE_HEIGHT = 58
IMAGE_SIZE = 132
GRID_COLUMNS = 4


def visualize_epoch_quality_pairs(
    quality_epoch_dir: Path | str | Sequence[str],
    index: Path | str | None = None,
    out_path: Path | str | None = None,
    num_samples: int = 16,
    seed: int | None = None,
) -> Path:
    """Write a source/generated pair grid for one quality epoch.

    Pairing is metadata-only. The function never infers sample identity from
    generated image filenames.
    """
    if isinstance(quality_epoch_dir, Sequence) and not isinstance(quality_epoch_dir, (str, bytes, Path)):
        args = parse_args(quality_epoch_dir)
        return visualize_epoch_quality_pairs(
            quality_epoch_dir=args.quality_epoch_dir,
            index=args.index,
            out_path=args.out_path,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    if index is None:
        raise ValueError("index is required")
    if out_path is None:
        raise ValueError("out_path is required")
    if isinstance(num_samples, bool) or int(num_samples) <= 0:
        raise ValueError(f"num_samples must be a positive integer, got {num_samples!r}")

    epoch_dir = Path(quality_epoch_dir)
    index_path = Path(index)
    output_path = Path(out_path)
    if not epoch_dir.is_dir():
        raise NotADirectoryError(f"quality epoch dir is not a directory: {epoch_dir}")
    if not index_path.is_file():
        raise FileNotFoundError(f"index not found: {index_path}")

    index_by_sample_id = _load_index(index_path)
    metadata_rows = _load_epoch_metadata(epoch_dir)
    selected_rows = _select_rows(metadata_rows, num_samples=int(num_samples), seed=seed)
    entries = [_entry_from_metadata(row, index_by_sample_id, epoch_dir, index_path.parent) for row in selected_rows]
    return _draw_grid(entries, output_path, title=epoch_dir.name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize source/generated pairs from a SAFA quality epoch")
    parser.add_argument("--quality-epoch-dir", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--out-path", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        visualize_epoch_quality_pairs(
            quality_epoch_dir=args.quality_epoch_dir,
            index=args.index,
            out_path=args.out_path,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from exc
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON metadata must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def _load_index(index_path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(index_path)
    if not rows:
        raise ValueError(f"index contains no rows: {index_path}")
    by_sample_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = _required_str(row, ("sample_id",), "index row")
        _required_str(row, ("image_path",), f"index row {sample_id!r}")
        if sample_id in by_sample_id:
            raise ValueError(f"duplicate sample_id in index: {sample_id}")
        by_sample_id[sample_id] = row
    return by_sample_id


def _load_epoch_metadata(epoch_dir: Path) -> list[dict[str, Any]]:
    per_sample_path = epoch_dir / "per_sample.jsonl"
    if per_sample_path.is_file():
        return _non_empty_metadata(_read_jsonl(per_sample_path), per_sample_path)

    result_path = epoch_dir / "result.json"
    if result_path.is_file():
        rows = _rows_from_result_json(result_path, _read_json(result_path))
        return _non_empty_metadata(rows, result_path)

    for path in sorted(epoch_dir.glob("*.json")):
        try:
            payload = _read_json(path)
        except json.JSONDecodeError:
            continue
        rows = _rows_from_result_json(path, payload)
        if rows:
            return _non_empty_metadata(rows, path)

    raise ValueError(
        "quality epoch metadata not found; expected per_sample.jsonl or result.json "
        "with artifacts.per_sample_jsonl or a per_sample list"
    )


def _rows_from_result_json(result_path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    per_sample_value = _nested_value(payload, ("artifacts", "per_sample_jsonl"))
    if isinstance(per_sample_value, str) and per_sample_value:
        per_sample_path = _resolve_path(result_path.parent, per_sample_value)
        if not per_sample_path.is_file():
            raise FileNotFoundError(f"per-sample metadata not found: {per_sample_path}")
        return _read_jsonl(per_sample_path)
    rows = payload.get("per_sample")
    if isinstance(rows, list):
        parsed = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{result_path}: per_sample[{index}] must be an object")
            parsed.append(row)
        return parsed
    return []


def _non_empty_metadata(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError(f"quality epoch metadata has no rows: {path}")
    return rows


def _select_rows(rows: list[dict[str, Any]], *, num_samples: int, seed: int | None) -> list[dict[str, Any]]:
    if seed is not None:
        selected = list(rows)
        random.Random(seed).shuffle(selected)
        return selected[:num_samples]
    return sorted(rows, key=lambda row: _required_str(row, ("sample_id",), "metadata row"))[:num_samples]


def _entry_from_metadata(
    row: dict[str, Any],
    index_by_sample_id: dict[str, dict[str, Any]],
    epoch_dir: Path,
    index_dir: Path,
) -> dict[str, Any]:
    sample_id = _required_str(row, ("sample_id",), "metadata row")
    if sample_id not in index_by_sample_id:
        raise KeyError(f"sample_id from quality metadata cannot join index: {sample_id}")
    index_row = index_by_sample_id[sample_id]
    source_path = _resolve_path(index_dir, _required_str(index_row, ("image_path",), f"index row {sample_id!r}"))
    generated_value = _nested_value(row, ("artifacts", "generated_image_path"))
    if generated_value is None:
        generated_value = row.get("generated_image_path")
    if not isinstance(generated_value, str) or not generated_value:
        raise ValueError(f"metadata row {sample_id!r} missing generated_image_path")
    generated_path = _resolve_path(epoch_dir, generated_value)
    if not source_path.is_file():
        raise FileNotFoundError(f"source image not found for {sample_id}: {source_path}")
    if not generated_path.is_file():
        raise FileNotFoundError(f"generated image not found for {sample_id}: {generated_path}")
    if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"source image has unsupported extension for {sample_id}: {source_path}")
    if generated_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"generated image has unsupported extension for {sample_id}: {generated_path}")
    return {
        "sample_id": sample_id,
        "label": row.get("label", index_row.get("label")),
        "source_path": source_path,
        "generated_path": generated_path,
    }


def _nested_value(data: dict[str, Any], keys: Sequence[str]) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _required_str(data: dict[str, Any], keys: Sequence[str], context: str) -> str:
    value = _nested_value(data, keys)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} missing {'.'.join(keys)}")
    return value


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return base_dir / path


def _draw_grid(entries: list[dict[str, Any]], out_path: Path, *, title: str) -> Path:
    if not entries:
        raise ValueError("no metadata rows selected")
    columns = min(GRID_COLUMNS, len(entries))
    rows = int(math.ceil(len(entries) / columns))
    canvas = Image.new("RGB", (columns * TILE_WIDTH, TITLE_HEIGHT + rows * TILE_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22)
    header_font = _font(12)
    text_font = _font(11)
    draw.text((canvas.width // 2, 18), f"{title} source/generated pairs", fill=(20, 20, 20), font=title_font, anchor="ma")

    for item_index, entry in enumerate(entries):
        row_index = item_index // columns
        col_index = item_index % columns
        x0 = col_index * TILE_WIDTH
        y0 = TITLE_HEIGHT + row_index * TILE_HEIGHT
        source_x = x0 + 28
        generated_x = x0 + 180
        image_y = y0 + 38
        draw.rectangle((x0, y0, x0 + TILE_WIDTH - 1, y0 + TILE_HEIGHT - 1), outline=(230, 230, 230))
        draw.text((source_x + IMAGE_SIZE // 2, y0 + 18), "source", fill=(70, 70, 70), font=header_font, anchor="ma")
        draw.text((generated_x + IMAGE_SIZE // 2, y0 + 18), "generated", fill=(70, 70, 70), font=header_font, anchor="ma")
        _paste_image(canvas, entry["source_path"], source_x, image_y)
        _paste_image(canvas, entry["generated_path"], generated_x, image_y)
        label = _entry_label(entry)
        draw.text((x0 + 18, y0 + 190), label, fill=(25, 25, 25), font=text_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


def _paste_image(canvas: Image.Image, path: Path, x: int, y: int) -> None:
    with Image.open(path) as image:
        image = image.convert("RGB")
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), resample)
        frame = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (245, 245, 245))
        frame.paste(image, ((IMAGE_SIZE - image.width) // 2, (IMAGE_SIZE - image.height) // 2))
    canvas.paste(frame, (x, y))


def _entry_label(entry: dict[str, Any]) -> str:
    sample_id = _short_sample_id(str(entry["sample_id"]))
    label = entry.get("label")
    if label is None:
        return f"id {sample_id}"
    return f"id {sample_id}  label {label}"


def _short_sample_id(sample_id: str) -> str:
    safe = re.sub(r"\s+", " ", sample_id)
    if len(safe) <= 42:
        return safe
    return "..." + safe[-39:]


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
