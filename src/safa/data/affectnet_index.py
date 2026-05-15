from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

from safa.data.index_schema import IMAGE_EXTENSIONS, IndexRecord, VALID_LABELS


CSV_PATH_FIELDS = ("subDirectory_filePath", "subDirectory_file_path", "image_path", "path", "filepath", "file")
CSV_LABEL_FIELDS = ("expression", "label", "emotion", "class")
CSV_SPLIT_FIELDS = ("split", "partition", "set")
CSV_CANDIDATES = ("training.csv", "validation.csv", "train.csv", "val.csv", "valid.csv")
CLASS_NAME_TO_LABEL = {
    "neutral": 0,
    "happy": 1,
    "happiness": 1,
    "sad": 2,
    "sadness": 2,
    "surprise": 3,
    "fear": 4,
    "disgust": 5,
    "anger": 6,
    "angry": 6,
    "contempt": 7,
}


def build_affectnet_index(
    root: Path,
    default_split: str,
    dataset_version: str,
    limit: int | None = None,
) -> list[IndexRecord]:
    if not root.is_dir():
        raise FileNotFoundError(f"AffectNet root does not exist or is not a directory: {root}")
    csv_files = [root / name for name in CSV_CANDIDATES if (root / name).is_file()]
    if csv_files:
        records = _records_from_csvs(root, csv_files, default_split, dataset_version)
    else:
        records = _records_from_folders(root, default_split, dataset_version)
    records = sorted(records, key=lambda item: item.sample_id)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided")
        records = records[:limit]
    _validate_collection(records)
    return records


def _records_from_csvs(
    root: Path,
    csv_files: list[Path],
    default_split: str,
    dataset_version: str,
) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    errors: list[str] = []
    for csv_path in csv_files:
        split_from_name = _split_from_csv_name(csv_path.name, default_split)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {csv_path}")
            path_field = _first_present(reader.fieldnames, CSV_PATH_FIELDS)
            label_field = _first_present(reader.fieldnames, CSV_LABEL_FIELDS)
            split_field = _first_present(reader.fieldnames, CSV_SPLIT_FIELDS, required=False)
            if path_field is None or label_field is None:
                raise ValueError(
                    f"CSV {csv_path} must contain one image path field from {CSV_PATH_FIELDS} "
                    f"and one label field from {CSV_LABEL_FIELDS}; got {reader.fieldnames}"
                )
            for row_no, row in enumerate(reader, start=2):
                try:
                    rel_path = str(row[path_field]).strip()
                    label = _parse_label(row[label_field])
                    split = str(row[split_field]).strip() if split_field else split_from_name
                    image_path = Path(rel_path)
                    if not image_path.is_absolute():
                        image_path = root / image_path
                    sample_id = _sample_id(split, image_path, root)
                    records.append(
                        IndexRecord.from_mapping(
                            {
                                "sample_id": sample_id,
                                "image_path": str(image_path),
                                "label": label,
                                "split": split,
                                "dataset_root": str(root),
                                "dataset_version": dataset_version,
                            }
                        )
                    )
                except Exception as exc:
                    errors.append(f"{csv_path}:{row_no}: {exc}")
    if errors:
        raise ValueError("AffectNet CSV validation failed:\n" + "\n".join(errors[:50]))
    return records


def _records_from_folders(root: Path, default_split: str, dataset_version: str) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    errors: list[str] = []
    split_dirs = _discover_split_dirs(root, default_split)
    for split, split_dir in split_dirs:
        for class_dir in sorted([path for path in split_dir.iterdir() if path.is_dir()]):
            try:
                label = _parse_label(class_dir.name)
            except Exception as exc:
                errors.append(f"{class_dir}: {exc}")
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    try:
                        records.append(
                            IndexRecord.from_mapping(
                                {
                                    "sample_id": _sample_id(split, image_path, root),
                                    "image_path": str(image_path),
                                    "label": label,
                                    "split": split,
                                    "dataset_root": str(root),
                                    "dataset_version": dataset_version,
                                }
                            )
                        )
                    except Exception as exc:
                        errors.append(f"{image_path}: {exc}")
    if errors:
        raise ValueError("AffectNet folder validation failed:\n" + "\n".join(errors[:50]))
    return records


def _discover_split_dirs(root: Path, default_split: str) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for split in ("train", "training", "val", "valid", "validation", "test"):
        path = root / split
        if path.is_dir():
            normalized = "val" if split in {"valid", "validation"} else ("train" if split == "training" else split)
            candidates.append((normalized, path))
    if candidates:
        return candidates
    return [(default_split, root)]


def _first_present(fields: list[str], candidates: tuple[str, ...], required: bool = True) -> str | None:
    lowered = {field.lower(): field for field in fields}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise ValueError(f"None of the required fields are present: {candidates}")
    return None


def _parse_label(raw: object) -> int:
    text = str(raw).strip()
    if text.lower() in CLASS_NAME_TO_LABEL:
        return CLASS_NAME_TO_LABEL[text.lower()]
    label = int(float(text))
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid 8-class AffectNet label: {raw}")
    return label


def _split_from_csv_name(name: str, default_split: str) -> str:
    lowered = name.lower()
    if "train" in lowered:
        return "train"
    if "val" in lowered or "valid" in lowered:
        return "val"
    return default_split


def _sample_id(split: str, image_path: Path, root: Path) -> str:
    try:
        rel = image_path.relative_to(root)
    except ValueError:
        rel = image_path
    return f"{split}:{rel.as_posix()}"


def _validate_collection(records: list[IndexRecord]) -> None:
    if not records:
        raise ValueError("No AffectNet samples found")
    counts = Counter(record.label for record in records)
    missing = sorted(VALID_LABELS.difference(counts))
    if missing:
        raise ValueError(f"Index is missing labels: {missing}; counts={dict(sorted(counts.items()))}")
    duplicate_count = len(records) - len({record.sample_id for record in records})
    if duplicate_count:
        raise ValueError(f"Duplicate sample_id values found: {duplicate_count}")

