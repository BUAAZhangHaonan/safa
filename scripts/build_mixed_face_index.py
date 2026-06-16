#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from safa.data.index_schema import IMAGE_EXTENSIONS, IndexRecord, read_index, write_index  # noqa: E402


DEFAULT_CELEBAHQ_ROOT = Path("/home/k100/Datasets/Face/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img")
DEFAULT_FFHQ_ROOT = Path("/home/k100/Datasets/Face/FFHQ-1024")
DEFAULT_AFFECTNET_TRAIN_INDEX = Path("data/index/train_balanced_medium.jsonl")
DEFAULT_AFFECTNET_VAL_INDEX = Path("data/index/val_single_face.jsonl")
DEFAULT_TRAIN_OUT = Path("data/index/train_face_mixed_e14.jsonl")
DEFAULT_VAL_OUT = Path("data/index/val_face_mixed_e14.jsonl")
DEFAULT_EXPECTED_CELEBAHQ_COUNT = 30000
DEFAULT_EXPECTED_FFHQ_COUNT = 70000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_sample_ids(records: Iterable[IndexRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _scan_images(root: Path, dataset_name: str, allowed_extensions: set[str]) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"{dataset_name} root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"{dataset_name} root is not a directory: {root}")
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_extensions
    )
    if not images:
        allowed = ", ".join(sorted(allowed_extensions))
        raise ValueError(f"{dataset_name} root contains no supported images ({allowed}): {root}")
    return images


def _ensure_expected_count(image_paths: list[Path], dataset_name: str, expected_count: int) -> None:
    actual_count = len(image_paths)
    if actual_count != expected_count:
        raise ValueError(f"{dataset_name} expected {expected_count} images but found {actual_count}")


def _generic_records(
    *,
    image_paths: list[Path],
    root: Path,
    sample_id_prefix: str,
    dataset_version: str,
) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    for index, image_path in enumerate(image_paths):
        records.append(
            IndexRecord.from_mapping(
                {
                    "sample_id": f"{sample_id_prefix}_{index:06d}",
                    "image_path": str(image_path),
                    "label": 0,
                    "split": "train",
                    "dataset_root": str(root),
                    "dataset_version": dataset_version,
                }
            )
        )
    return records


def _ensure_unique_sample_ids(records: Iterable[IndexRecord]) -> None:
    counts = Counter(record.sample_id for record in records)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:10])
        suffix = "" if len(duplicates) <= 10 else f" ... (+{len(duplicates) - 10} more)"
        raise ValueError(f"duplicate sample_id values across mixed index: {preview}{suffix}")


def _ensure_unique_image_paths(records: Iterable[IndexRecord]) -> None:
    normalized_paths = [str(Path(record.image_path).resolve()) for record in records]
    counts = Counter(normalized_paths)
    duplicates = sorted(image_path for image_path, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:10])
        suffix = "" if len(duplicates) <= 10 else f" ... (+{len(duplicates) - 10} more)"
        raise ValueError(f"duplicate image_path values across mixed index: {preview}{suffix}")


def manifest_path_for(train_out: Path) -> Path:
    return train_out.with_name(f"{train_out.stem}_manifest.json")


def _write_manifest(
    *,
    affectnet_train_index: Path,
    affectnet_val_index: Path,
    celebahq_root: Path,
    ffhq_root: Path,
    train_out: Path,
    val_out: Path,
    train_records: list[IndexRecord],
    val_records: list[IndexRecord],
    train_dataset_counts: dict[str, int],
    val_dataset_counts: dict[str, int],
) -> Path:
    manifest_path = manifest_path_for(train_out)
    manifest = {
        "schema_version": 1,
        "affectnet_train_index": str(affectnet_train_index),
        "affectnet_train_index_sha256": sha256_file(affectnet_train_index),
        "affectnet_val_index": str(affectnet_val_index),
        "affectnet_val_index_sha256": sha256_file(affectnet_val_index),
        "celebahq_root": str(celebahq_root),
        "ffhq_root": str(ffhq_root),
        "train_index": str(train_out),
        "train_index_sha256": sha256_file(train_out),
        "val_index": str(val_out),
        "val_index_sha256": sha256_file(val_out),
        "train_ordered_sample_id_sha256": sha256_sample_ids(train_records),
        "val_ordered_sample_id_sha256": sha256_sample_ids(val_records),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "train_dataset_counts": dict(sorted(train_dataset_counts.items())),
        "val_dataset_counts": dict(sorted(val_dataset_counts.items())),
        "generic_face_label": 0,
        "generic_face_split": "train",
        "output_order_rule": "AffectNet train rows first, then sorted CelebA-HQ images, then sorted FFHQ images; val rows are AffectNet val rows.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def build_mixed_face_index(
    *,
    affectnet_train_index: Path,
    affectnet_val_index: Path,
    celebahq_root: Path,
    ffhq_root: Path,
    train_out: Path,
    val_out: Path,
    expected_celebahq_count: int = DEFAULT_EXPECTED_CELEBAHQ_COUNT,
    expected_ffhq_count: int = DEFAULT_EXPECTED_FFHQ_COUNT,
) -> Path:
    affectnet_train = read_index(affectnet_train_index)
    affectnet_val = read_index(affectnet_val_index)

    celebahq_paths = _scan_images(celebahq_root, "CelebA-HQ", IMAGE_EXTENSIONS)
    ffhq_paths = _scan_images(ffhq_root, "FFHQ", {".png"})
    _ensure_expected_count(celebahq_paths, "CelebA-HQ", expected_celebahq_count)
    _ensure_expected_count(ffhq_paths, "FFHQ", expected_ffhq_count)

    celebahq_records = _generic_records(
        image_paths=celebahq_paths,
        root=celebahq_root,
        sample_id_prefix="celebahq",
        dataset_version="celebahq-1024",
    )
    ffhq_records = _generic_records(
        image_paths=ffhq_paths,
        root=ffhq_root,
        sample_id_prefix="ffhq",
        dataset_version="ffhq-1024",
    )

    train_records = [*affectnet_train, *celebahq_records, *ffhq_records]
    val_records = list(affectnet_val)
    _ensure_unique_sample_ids([*train_records, *val_records])
    _ensure_unique_image_paths([*train_records, *val_records])

    write_index(train_records, train_out)
    write_index(val_records, val_out)
    return _write_manifest(
        affectnet_train_index=affectnet_train_index,
        affectnet_val_index=affectnet_val_index,
        celebahq_root=celebahq_root,
        ffhq_root=ffhq_root,
        train_out=train_out,
        val_out=val_out,
        train_records=train_records,
        val_records=val_records,
        train_dataset_counts={
            "affectnet": len(affectnet_train),
            "celebahq": len(celebahq_records),
            "ffhq": len(ffhq_records),
        },
        val_dataset_counts={"affectnet": len(affectnet_val)},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the E14 AffectNet + CelebA-HQ + FFHQ mixed face JSONL index.")
    parser.add_argument("--affectnet-train-index", type=Path, default=DEFAULT_AFFECTNET_TRAIN_INDEX)
    parser.add_argument("--affectnet-val-index", type=Path, default=DEFAULT_AFFECTNET_VAL_INDEX)
    parser.add_argument("--celebahq-root", type=Path, default=DEFAULT_CELEBAHQ_ROOT)
    parser.add_argument("--ffhq-root", type=Path, default=DEFAULT_FFHQ_ROOT)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--val-out", type=Path, default=DEFAULT_VAL_OUT)
    parser.add_argument("--expected-celebahq-count", type=int, default=DEFAULT_EXPECTED_CELEBAHQ_COUNT)
    parser.add_argument("--expected-ffhq-count", type=int, default=DEFAULT_EXPECTED_FFHQ_COUNT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest_path = build_mixed_face_index(
            affectnet_train_index=args.affectnet_train_index,
            affectnet_val_index=args.affectnet_val_index,
            celebahq_root=args.celebahq_root,
            ffhq_root=args.ffhq_root,
            train_out=args.train_out,
            val_out=args.val_out,
            expected_celebahq_count=args.expected_celebahq_count,
            expected_ffhq_count=args.expected_ffhq_count,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"wrote {manifest['num_train']} train records to {args.train_out}")
    print(f"wrote {manifest['num_val']} val records to {args.val_out}")
    print(f"train_dataset_counts={manifest['train_dataset_counts']}")
    print(f"val_dataset_counts={manifest['val_dataset_counts']}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
