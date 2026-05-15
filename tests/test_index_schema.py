from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from PIL import Image

from safa.data.affectnet_index import build_affectnet_index
from safa.data.index_schema import IndexRecord, read_index, write_index


class IndexSchemaTests(unittest.TestCase):
    def test_record_requires_existing_image_and_valid_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "0.jpg"
            Image.new("RGB", (8, 8)).save(image_path)
            record = IndexRecord.from_mapping(
                {
                    "sample_id": "train:0.jpg",
                    "image_path": str(image_path),
                    "label": 0,
                    "split": "train",
                    "dataset_root": str(root),
                    "dataset_version": "unit",
                }
            )
            self.assertEqual(record.label, 0)
            with self.assertRaises(ValueError):
                IndexRecord.from_mapping({**json.loads(record.to_json()), "label": 9})

    def test_write_and_read_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "0.jpg"
            Image.new("RGB", (8, 8)).save(image_path)
            out = root / "index.jsonl"
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
                out,
            )
            self.assertEqual(read_index(out)[0].sample_id, "train:0.jpg")


class AffectNetIndexTests(unittest.TestCase):
    def test_build_index_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            rows = []
            for label in range(8):
                path = root / "images" / f"{label}.jpg"
                Image.new("RGB", (8, 8)).save(path)
                rows.append(f"images/{label}.jpg,{label}\n")
            (root / "training.csv").write_text("subDirectory_filePath,expression\n" + "".join(rows), encoding="utf-8")
            records = build_affectnet_index(root, default_split="train", dataset_version="unit")
            self.assertEqual(len(records), 8)
            self.assertEqual({record.label for record in records}, set(range(8)))

    def test_affectnet8_policy_filters_official_non_eight_class_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "Manually_Annotated_Images"
            image_root.mkdir()
            rows = []
            for label in range(11):
                path = image_root / f"{label}.jpg"
                Image.new("RGB", (8, 8)).save(path)
                rows.append(f"{label}.jpg,{label}\n")
            (root / "training.csv").write_text("subDirectory_filePath,expression\n" + "".join(rows), encoding="utf-8")
            records = build_affectnet_index(
                root,
                default_split="train",
                dataset_version="unit",
                label_policy="affectnet8",
                csv_image_prefix="Manually_Annotated_Images",
            )
            self.assertEqual(len(records), 8)
            self.assertEqual({record.label for record in records}, set(range(8)))

    def test_missing_label_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "0").mkdir()
            Image.new("RGB", (8, 8)).save(root / "0" / "a.jpg")
            with self.assertRaises(ValueError):
                build_affectnet_index(root, default_split="train", dataset_version="unit")


if __name__ == "__main__":
    unittest.main()
