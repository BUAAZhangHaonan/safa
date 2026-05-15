from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

from safa.data.index_schema import IndexRecord, read_index


def load_rgb_image_strict(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to decode image {image_path}: {exc}") from exc


class AffectNetRecords:
    def __init__(self, index_path: str | Path, transform: Callable | None = None):
        self.index_path = Path(index_path)
        self.records = read_index(self.index_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = load_rgb_image_strict(record.image_path)
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "label": record.label, "sample_id": record.sample_id, "record": record}


def split_records(records: list[IndexRecord], split: str) -> list[IndexRecord]:
    selected = [record for record in records if record.split == split]
    if not selected:
        raise ValueError(f"No records found for split={split}")
    return selected

