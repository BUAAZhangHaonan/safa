from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from PIL import Image

from safa.data.dataset import AffectNetRecords, load_rgb_image_strict
from safa.data.index_schema import IndexRecord, write_index
from safa.cli.smoke import _balanced_smoke_records


class DatasetSmokeTests(unittest.TestCase):
    def test_load_rgb_image_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gray.jpg"
            Image.new("L", (8, 8)).save(path)
            image = load_rgb_image_strict(path)
            self.assertEqual(image.mode, "RGB")

    def test_dataset_reads_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "0.jpg"
            Image.new("RGB", (8, 8)).save(image_path)
            index_path = root / "index.jsonl"
            write_index(
                [
                    IndexRecord.from_mapping(
                        {
                            "sample_id": "train:0.jpg",
                            "image_path": str(image_path),
                            "label": 0,
                            "split": "train",
                            "dataset_root": str(root),
                            "dataset_version": "unit",
                        }
                    )
                ],
                index_path,
            )
            item = AffectNetRecords(index_path)[0]
            self.assertEqual(item["label"], 0)
            self.assertEqual(item["image"].mode, "RGB")

    def test_smoke_records_are_balanced(self) -> None:
        records = [
            IndexRecord(
                sample_id=f"val:{label}:{idx}",
                image_path="/tmp/unused.jpg",
                label=label,
                split="val",
                dataset_root="/tmp",
                dataset_version="unit",
            )
            for label in range(8)
            for idx in range(3)
        ]
        selected = _balanced_smoke_records(records, 16)
        counts = {label: sum(1 for record in selected if record.label == label) for label in range(8)}
        self.assertEqual(counts, {label: 2 for label in range(8)})
        with self.assertRaises(ValueError):
            _balanced_smoke_records(records, 10)


if __name__ == "__main__":
    unittest.main()
