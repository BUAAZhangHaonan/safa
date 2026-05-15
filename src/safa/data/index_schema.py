from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable


VALID_LABELS = set(range(8))
REQUIRED_FIELDS = {
    "sample_id",
    "image_path",
    "label",
    "split",
    "dataset_root",
    "dataset_version",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class IndexRecord:
    sample_id: str
    image_path: str
    label: int
    split: str
    dataset_root: str
    dataset_version: str

    @classmethod
    def from_mapping(cls, data: dict) -> "IndexRecord":
        missing = REQUIRED_FIELDS.difference(data)
        if missing:
            raise ValueError(f"Index record missing fields: {sorted(missing)}")
        label = int(data["label"])
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid AffectNet 8-class label {label} for sample {data['sample_id']}")
        image_path = Path(str(data["image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Image path does not exist for sample {data['sample_id']}: {image_path}")
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension for sample {data['sample_id']}: {image_path}")
        split = str(data["split"]).strip()
        if not split:
            raise ValueError(f"Empty split for sample {data['sample_id']}")
        return cls(
            sample_id=str(data["sample_id"]),
            image_path=str(image_path),
            label=label,
            split=split,
            dataset_root=str(data["dataset_root"]),
            dataset_version=str(data["dataset_version"]),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "sample_id": self.sample_id,
                "image_path": self.image_path,
                "label": self.label,
                "split": self.split,
                "dataset_root": self.dataset_root,
                "dataset_version": self.dataset_version,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


def read_index(path: Path) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(IndexRecord.from_mapping(json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"Invalid index line {line_no} in {path}: {exc}") from exc
    if not records:
        raise ValueError(f"Index contains no records: {path}")
    return records


def write_index(records: Iterable[IndexRecord], path: Path) -> None:
    materialized = list(records)
    if not materialized:
        raise ValueError("Refusing to write an empty index")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in materialized:
            handle.write(record.to_json() + "\n")
