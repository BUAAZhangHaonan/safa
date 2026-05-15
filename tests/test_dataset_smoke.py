from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from PIL import Image

from safa.data.dataset import AffectNetRecords, load_rgb_image_strict
from safa.data.index_schema import IndexRecord, write_index


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


if __name__ == "__main__":
    unittest.main()

