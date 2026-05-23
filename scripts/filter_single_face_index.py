#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Protocol


SUPPORTED_DETECTOR = "insightface_buffalo_l"


class FaceCountDetector(Protocol):
    def count_faces(self, image_path: Path) -> int:
        ...


class InsightFaceBuffaloLDetector:
    def __init__(self, device: str) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for insightface_buffalo_l filtering") from exc
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("insightface is required for insightface_buffalo_l filtering") from exc

        self._cv2 = cv2
        ctx_id = insightface_ctx_id(device)
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=(224, 224))

    def count_faces(self, image_path: Path) -> int:
        if not image_path.is_file():
            raise FileNotFoundError(f"source image does not exist: {image_path}")
        image = self._cv2.imread(str(image_path), self._cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"source image could not be decoded: {image_path}")
        return len(self.app.get(image))


def insightface_ctx_id(device: str) -> int:
    normalized = device.strip().lower()
    if normalized == "cpu":
        return -1
    if normalized == "cuda":
        return 0
    if normalized.startswith("cuda:"):
        suffix = normalized.split(":", 1)[1]
        try:
            ctx_id = int(suffix)
        except ValueError as exc:
            raise ValueError(f"invalid CUDA device {device!r}") from exc
        if ctx_id < 0:
            raise ValueError(f"invalid CUDA device {device!r}")
        return ctx_id
    raise ValueError(f"unsupported device for insightface_buffalo_l: {device!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_index(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            if "image_path" not in row:
                raise ValueError(f"{path}:{line_no}: missing required field 'image_path'")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: source index contains no rows")
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def manifest_path_for(output_index: Path) -> Path:
    return output_index.with_name(f"{output_index.stem}_manifest.json")


def validate_detector_name(detector_name: str) -> None:
    if detector_name != SUPPORTED_DETECTOR:
        raise ValueError(f"unsupported detector {detector_name!r}; supported detectors: {SUPPORTED_DETECTOR}")


def build_detector(detector_name: str, device: str) -> FaceCountDetector:
    validate_detector_name(detector_name)
    return InsightFaceBuffaloLDetector(device=device)


def validated_face_count(count: Any, image_path: Path) -> int:
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise ValueError(f"detector returned non-integer face count for {image_path}: {count!r}")
    parsed = int(count)
    if parsed < 0:
        raise ValueError(f"detector returned negative face count for {image_path}: {count!r}")
    return parsed


def filter_single_face_index(
    *,
    source_index: Path,
    output_index: Path,
    detector_name: str,
    device: str,
    detector_factory: Callable[[str, str], FaceCountDetector] | None = None,
) -> Path:
    validate_detector_name(detector_name)
    rows = read_jsonl_index(source_index)
    factory = detector_factory or build_detector
    detector = factory(detector_name, device)

    single_face_rows: list[dict[str, Any]] = []
    num_zero_face = 0
    num_multi_face = 0

    for row in rows:
        image_path = Path(str(row["image_path"]))
        face_count = validated_face_count(detector.count_faces(image_path), image_path)
        if face_count == 0:
            num_zero_face += 1
        elif face_count == 1:
            single_face_rows.append(row)
        else:
            num_multi_face += 1

    write_jsonl(single_face_rows, output_index)
    manifest_path = manifest_path_for(output_index)
    manifest = {
        "source_index": str(source_index),
        "source_index_sha256": sha256_file(source_index),
        "output_index": str(output_index),
        "output_index_sha256": sha256_file(output_index),
        "detector": detector_name,
        "device": device,
        "num_source": len(rows),
        "num_single_face": len(single_face_rows),
        "num_zero_face": num_zero_face,
        "num_multi_face": num_multi_face,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a JSONL image index to exactly single-face samples.")
    parser.add_argument("--source-index", required=True, type=Path)
    parser.add_argument("--output-index", required=True, type=Path)
    parser.add_argument("--detector", required=True, choices=[SUPPORTED_DETECTOR])
    parser.add_argument("--device", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        filter_single_face_index(
            source_index=args.source_index,
            output_index=args.output_index,
            detector_name=args.detector,
            device=args.device,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
