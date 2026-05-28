#!/usr/bin/env python3
"""Visualize source/generated pairs from a SAFA eval result artifact."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

_CANVAS_WIDTH = 560
_TITLE_HEIGHT = 60
_ROW_HEIGHT = 160
_IMAGE_SIZE = 120
_LEFT_X = 20
_RIGHT_X = 300
_IMAGE_Y_OFFSET = 10
_TEXT_Y_OFFSET = 134
_TEXT_LINE_HEIGHT = 12


def visualize_eval_pairs(
    result_json: Path | str | Sequence[str],
    out_path: Path | str | None = None,
    num_samples: int = 16,
    seed: int | None = None,
    sort_by: str = "sample_id",
) -> Path:
    """Write a source/generated visual audit PNG for an eval result.json."""
    if isinstance(result_json, Sequence) and not isinstance(result_json, (str, bytes, Path)) and out_path is None:
        args = _parse_args(result_json)
        return visualize_eval_pairs(
            args.result_json,
            args.out_path,
            num_samples=args.num_samples,
            seed=args.seed,
            sort_by=args.sort_by,
        )
    if out_path is None:
        raise ValueError("out_path is required")

    result_path = Path(result_json)
    output_path = Path(out_path)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples!r}")

    result = _read_json(result_path, "result json")
    per_sample_path = _resolve_required_path(result_path.parent, _require_path_value(result, ["artifacts", "per_sample_jsonl"]))
    index_path = _resolve_required_path(result_path.parent, _require_path_value(result, ["dataset", "index"]))
    if not per_sample_path.is_file():
        raise FileNotFoundError(f"per-sample jsonl not found: {per_sample_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"dataset index not found: {index_path}")

    index_by_sample_id = _load_index(index_path)
    rows = _load_per_sample_rows(per_sample_path)
    selected = _select_rows(rows, num_samples=num_samples, seed=seed, sort_by=sort_by)
    entries = [_entry_from_row(row, index_by_sample_id, result_path.parent) for row in selected]
    return _draw(entries, output_path)


def main(argv: Sequence[str] | None = None) -> Path:
    args = _parse_args(argv)
    return visualize_eval_pairs(
        args.result_json,
        args.out_path,
        num_samples=args.num_samples,
        seed=args.seed,
        sort_by=args.sort_by,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SAFA eval source/generated pairs")
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--out-path", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sort-by", default="sample_id")
    return parser.parse_args(argv)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _require_path_value(data: dict[str, Any], keys: Sequence[str]) -> str:
    value: Any = data
    dotted = ".".join(keys)
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted)
        value = value[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{dotted} must be a non-empty string")
    return value


def _resolve_required_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return base_dir / path


def _load_index(index_path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(index_path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise KeyError("dataset.index row missing sample_id")
        if "image_path" not in row:
            raise KeyError(f"dataset.index row {sample_id!r} missing image_path")
        if sample_id in index:
            raise ValueError(f"duplicate sample_id in dataset index: {sample_id}")
        index[sample_id] = row
    return index


def _load_per_sample_rows(per_sample_path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(per_sample_path)
    if not rows:
        raise ValueError(f"per-sample jsonl has no rows: {per_sample_path}")
    return rows


def _select_rows(rows: list[dict[str, Any]], num_samples: int, seed: int | None, sort_by: str) -> list[dict[str, Any]]:
    if seed is not None:
        selected = list(rows)
        random.Random(seed).shuffle(selected)
        return selected[:num_samples]
    return sorted(rows, key=lambda row: _sort_value(row, sort_by))[:num_samples]


def _sort_value(row: dict[str, Any], sort_by: str) -> Any:
    if sort_by == "sample_id":
        return _require_str(row, ["sample_id"])
    value = _nested_value(row, sort_by.split("."))
    if value is None and sort_by == "latent_cosine":
        value = _nested_value(row, ["affective", "latent_cosine"])
    if value is None:
        raise KeyError(sort_by)
    return value


def _nested_value(data: dict[str, Any], keys: Sequence[str]) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _entry_from_row(row: dict[str, Any], index_by_sample_id: dict[str, dict[str, Any]], base_dir: Path) -> dict[str, Any]:
    sample_id = _require_str(row, ["sample_id"])
    if sample_id not in index_by_sample_id:
        raise KeyError(f"sample_id cannot join dataset.index: {sample_id}")
    index_row = index_by_sample_id[sample_id]
    source_path = _resolve_required_path(base_dir, _require_str(index_row, ["image_path"]))
    generated_path = _resolve_required_path(base_dir, _require_str(row, ["artifacts", "generated_image_path"]))
    if not source_path.is_file():
        raise FileNotFoundError(f"source image not found for {sample_id}: {source_path}")
    if not generated_path.is_file():
        raise FileNotFoundError(f"generated image not found for {sample_id}: {generated_path}")
    return {
        "sample_id": sample_id,
        "label": _require_number(row, ["label"]),
        "latent_cosine": _require_number(row, ["affective", "latent_cosine"]),
        "source_prediction_preserved": _require_number(row, ["affective", "source_prediction_preserved"]),
        "face_count": _require_number(row, ["face_detection", "count"]),
        "single_face": _require_number(row, ["face_detection", "single_face_eq1_rate"]),
        "source_path": source_path,
        "generated_path": generated_path,
    }


def _require_str(data: dict[str, Any], keys: Sequence[str]) -> str:
    value = _nested_value(data, keys)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{'.'.join(keys)} must be a non-empty string")
    return value


def _require_number(data: dict[str, Any], keys: Sequence[str]) -> int | float:
    value = _nested_value(data, keys)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{'.'.join(keys)} must be a number")
    return value


def _draw(entries: list[dict[str, Any]], out_path: Path) -> Path:
    if not entries:
        raise ValueError("no samples selected")
    height = _TITLE_HEIGHT + _ROW_HEIGHT * len(entries)
    canvas = Image.new("RGB", (_CANVAS_WIDTH, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22)
    header_font = _font(13)
    text_font = _font(10)
    draw.text((_CANVAS_WIDTH // 2, 18), "visual audit", fill=(20, 20, 20), font=title_font, anchor="ma")
    draw.text((_LEFT_X + _IMAGE_SIZE // 2, 45), "source", fill=(60, 60, 60), font=header_font, anchor="ma")
    draw.text((_RIGHT_X + _IMAGE_SIZE // 2, 45), "generated", fill=(60, 60, 60), font=header_font, anchor="ma")

    for idx, entry in enumerate(entries):
        row_y = _TITLE_HEIGHT + idx * _ROW_HEIGHT
        _paste_image(canvas, entry["source_path"], _LEFT_X, row_y + _IMAGE_Y_OFFSET)
        _paste_image(canvas, entry["generated_path"], _RIGHT_X, row_y + _IMAGE_Y_OFFSET)
        label_lines = _label_lines(entry)
        for line_idx, line in enumerate(label_lines):
            y = row_y + _TEXT_Y_OFFSET + line_idx * _TEXT_LINE_HEIGHT
            draw.text((_LEFT_X, y), line, fill=(20, 20, 20), font=text_font)
            draw.text((_RIGHT_X, y), line, fill=(20, 20, 20), font=text_font)
        if idx < len(entries) - 1:
            draw.line((10, row_y + _ROW_HEIGHT - 1, _CANVAS_WIDTH - 10, row_y + _ROW_HEIGHT - 1), fill=(225, 225, 225))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


def _paste_image(canvas: Image.Image, path: Path, x: int, y: int) -> None:
    with Image.open(path) as image:
        image = image.convert("RGB")
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        image.thumbnail((_IMAGE_SIZE, _IMAGE_SIZE), resample)
        framed = Image.new("RGB", (_IMAGE_SIZE, _IMAGE_SIZE), (245, 245, 245))
        paste_x = (_IMAGE_SIZE - image.width) // 2
        paste_y = (_IMAGE_SIZE - image.height) // 2
        framed.paste(image, (paste_x, paste_y))
    canvas.paste(framed, (x, y))


def _label_lines(entry: dict[str, Any]) -> list[str]:
    face_count = entry["face_count"]
    single_face = entry["single_face"]
    single_text = "single-face" if float(single_face) == 1.0 else "not single-face"
    return [
        f"id {_short_sample_id(entry['sample_id'])}  label {entry['label']}",
        f"cos {entry['latent_cosine']:.3f}  pred-pres {entry['source_prediction_preserved']:.3f}",
        f"faces {face_count:g}  {single_text}",
    ]


def _short_sample_id(sample_id: str) -> str:
    if len(sample_id) <= 28:
        return sample_id
    return "..." + sample_id[-25:]


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
