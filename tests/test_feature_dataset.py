from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from PIL import Image

from safa.data.index_schema import IndexRecord, write_index
from safa.utils.hashing import sha256_file


class FeatureDatasetTests(unittest.TestCase):
    def _write_cache(self, root: Path, *, features, labels: list) -> tuple[Path, Path, Path]:
        import torch

        image_path = root / "sample.jpg"
        Image.new("RGB", (8, 8)).save(image_path)
        index_path = root / "index.jsonl"
        write_index(
            [
                IndexRecord(
                    sample_id="sample-1",
                    image_path=str(image_path),
                    label=0,
                    split="train",
                    dataset_root=str(root),
                    dataset_version="unit",
                )
            ],
            index_path,
        )
        checkpoint_path = root / "best.pt"
        checkpoint_path.write_text("checkpoint", encoding="utf-8")
        cache_dir = root / "features"
        cache_dir.mkdir()
        shard_path = cache_dir / "features.pt"
        torch.save({"features": features, "sample_ids": ["sample-1"], "labels": labels}, shard_path)
        manifest = {
            "dataset": "AffectNet",
            "index_path": str(index_path),
            "index_sha256": sha256_file(index_path),
            "encoder_checkpoint": str(checkpoint_path),
            "encoder_checkpoint_sha256": sha256_file(checkpoint_path),
            "num_samples": 1,
            "feature_dim": int(features.shape[1]),
            "l2_normalized": True,
            "dtype": str(features.dtype).replace("torch.", ""),
            "shard": "features.pt",
            "shard_sha256": sha256_file(shard_path),
            "sample_ids": ["sample-1"],
            "labels": labels,
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return index_path, cache_dir, checkpoint_path

    def _write_many_to_many_fixture(
        self,
        root: Path,
        *,
        source_records: list[tuple[str, int, tuple[int, int, int]]],
        target_records: list[tuple[str, int, tuple[int, int, int]]],
        features,
    ) -> tuple[Path, Path, Path, Path, dict[str, tuple[int, int, int]]]:
        import torch

        def write_records(
            entries: list[tuple[str, int, tuple[int, int, int]]],
            image_root: Path,
            index_path: Path,
        ) -> tuple[list[IndexRecord], dict[str, tuple[int, int, int]]]:
            image_root.mkdir()
            records = []
            colors = {}
            for sample_id, label, color in entries:
                image_path = image_root / f"{sample_id}.png"
                Image.new("RGB", (8, 8), color).save(image_path)
                records.append(
                    IndexRecord(
                        sample_id=sample_id,
                        image_path=str(image_path),
                        label=label,
                        split="train",
                        dataset_root=str(root),
                        dataset_version="unit",
                    )
                )
                colors[sample_id] = color
            write_index(records, index_path)
            return records, colors

        source_index_path = root / "source_index.jsonl"
        target_index_path = root / "target_index.jsonl"
        source, _ = write_records(source_records, root / "source_images", source_index_path)
        _, target_colors = write_records(target_records, root / "target_images", target_index_path)

        checkpoint_path = root / "best.pt"
        checkpoint_path.write_text("checkpoint", encoding="utf-8")
        cache_dir = root / "source_features"
        cache_dir.mkdir()
        shard_path = cache_dir / "features.pt"
        sample_ids = [record.sample_id for record in source]
        labels = [record.label for record in source]
        torch.save({"features": features, "sample_ids": sample_ids, "labels": labels}, shard_path)
        manifest = {
            "dataset": "AffectNet",
            "index_path": str(source_index_path),
            "index_sha256": sha256_file(source_index_path),
            "encoder_checkpoint": str(checkpoint_path),
            "encoder_checkpoint_sha256": sha256_file(checkpoint_path),
            "num_samples": len(source),
            "feature_dim": int(features.shape[1]),
            "l2_normalized": True,
            "dtype": str(features.dtype).replace("torch.", ""),
            "shard": "features.pt",
            "shard_sha256": sha256_file(shard_path),
            "sample_ids": sample_ids,
            "labels": labels,
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return source_index_path, target_index_path, cache_dir, checkpoint_path, target_colors

    def test_feature_dataset_rejects_non_float32_features(self) -> None:
        import torch

        from safa.data.feature_dataset import FeatureAlignedAffectNet

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = torch.zeros(1, 128, dtype=torch.int64)
            features[0, 0] = 1
            index_path, cache_dir, checkpoint_path = self._write_cache(root, features=features, labels=[0])
            with self.assertRaises(ValueError):
                FeatureAlignedAffectNet(index_path, cache_dir, checkpoint_path, transform=None)

    def test_feature_dataset_rejects_non_int_labels(self) -> None:
        import torch

        from safa.data.feature_dataset import FeatureAlignedAffectNet

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = torch.zeros(1, 128)
            features[0, 0] = 1.0
            index_path, cache_dir, checkpoint_path = self._write_cache(root, features=features, labels=["0"])
            with self.assertRaises(ValueError):
                FeatureAlignedAffectNet(index_path, cache_dir, checkpoint_path, transform=None)

    def test_many_to_many_dataset_uses_target_image_and_source_feature(self) -> None:
        import torch

        from safa.data.feature_dataset import ManyToManyFeatureAlignedAffectNet

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = torch.eye(3, 4, dtype=torch.float32)
            source_index_path, target_index_path, cache_dir, checkpoint_path, target_colors = (
                self._write_many_to_many_fixture(
                    root,
                    source_records=[
                        ("sample-0-a", 0, (11, 22, 33)),
                        ("sample-0-b", 0, (44, 55, 66)),
                        ("sample-1-a", 1, (77, 88, 99)),
                    ],
                    target_records=[
                        ("sample-0-a", 0, (101, 102, 103)),
                        ("sample-0-b", 0, (111, 112, 113)),
                        ("sample-1-a", 1, (121, 122, 123)),
                        ("sample-1-b", 1, (131, 132, 133)),
                    ],
                    features=features,
                )
            )

            dataset = ManyToManyFeatureAlignedAffectNet(
                source_index_path=source_index_path,
                target_index_path=target_index_path,
                source_feature_dir=cache_dir,
                e0_checkpoint=checkpoint_path,
                transform=None,
            )

            item = dataset[0]
            again = dataset[0]

            self.assertEqual(item["source_sample_id"], "sample-0-a")
            self.assertNotEqual(item["source_sample_id"], item["target_sample_id"])
            self.assertEqual(item["label"], 0)
            self.assertEqual(item["source_label"], item["label"])
            self.assertEqual(item["target_label"], item["label"])
            self.assertEqual(item["image"].getpixel((0, 0)), target_colors[item["target_sample_id"]])
            torch.testing.assert_close(item["z"], features[0])
            self.assertEqual(item["sample_id"], again["sample_id"])
            self.assertIn(item["source_sample_id"], item["sample_id"])
            self.assertIn(item["target_sample_id"], item["sample_id"])

    def test_many_to_many_dataset_selects_planned_cyclic_targets(self) -> None:
        import torch

        from safa.data.feature_dataset import ManyToManyFeatureAlignedAffectNet

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_index_path, target_index_path, cache_dir, checkpoint_path, _ = self._write_many_to_many_fixture(
                root,
                source_records=[("source-0", 0, (11, 22, 33))],
                target_records=[
                    ("source-0", 0, (101, 102, 103)),
                    ("cyclic-target-a", 0, (111, 112, 113)),
                    ("cyclic-target-b", 0, (121, 122, 123)),
                ],
                features=torch.eye(1, 4, dtype=torch.float32),
            )

            source_sample_id = "source-0"
            source_index = 0
            label_zero_targets = [source_sample_id, "cyclic-target-a", "cyclic-target-b"]
            eligible_targets = [target for target in label_zero_targets if target != source_sample_id]

            def expected_target_id(*, pairing_seed: int, pair_round: int) -> str:
                target_position = (source_index + pairing_seed + pair_round) % len(eligible_targets)
                return eligible_targets[target_position]

            dataset = ManyToManyFeatureAlignedAffectNet(
                source_index_path=source_index_path,
                target_index_path=target_index_path,
                source_feature_dir=cache_dir,
                e0_checkpoint=checkpoint_path,
                pairing_seed=0,
                pairs_per_source=2,
                transform=None,
            )

            first_pair = dataset[0]
            second_pair = dataset[1]

            self.assertEqual(len(dataset), 2)
            self.assertEqual(first_pair["source_sample_id"], "source-0")
            self.assertEqual(second_pair["source_sample_id"], "source-0")
            self.assertEqual(first_pair["target_sample_id"], expected_target_id(pairing_seed=0, pair_round=0))
            self.assertEqual(second_pair["target_sample_id"], expected_target_id(pairing_seed=0, pair_round=1))
            self.assertNotEqual(first_pair["target_sample_id"], second_pair["target_sample_id"])
            self.assertEqual(first_pair["pair_id"], "source-0__to__cyclic-target-a__round0")
            self.assertEqual(second_pair["pair_id"], "source-0__to__cyclic-target-b__round1")

            shifted_dataset = ManyToManyFeatureAlignedAffectNet(
                source_index_path=source_index_path,
                target_index_path=target_index_path,
                source_feature_dir=cache_dir,
                e0_checkpoint=checkpoint_path,
                pairing_seed=1,
                pairs_per_source=2,
                transform=None,
            )

            self.assertEqual(
                [shifted_dataset[0]["target_sample_id"], shifted_dataset[1]["target_sample_id"]],
                [
                    expected_target_id(pairing_seed=1, pair_round=0),
                    expected_target_id(pairing_seed=1, pair_round=1),
                ],
            )
            self.assertNotEqual(shifted_dataset[0]["target_sample_id"], shifted_dataset[1]["target_sample_id"])
            self.assertEqual(shifted_dataset[1]["pair_id"], "source-0__to__cyclic-target-a__round1")

    def test_many_to_many_dataset_rejects_label_bucket_with_only_same_sample_id(self) -> None:
        import torch

        from safa.data.feature_dataset import ManyToManyFeatureAlignedAffectNet

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_index_path, target_index_path, cache_dir, checkpoint_path, _ = self._write_many_to_many_fixture(
                root,
                source_records=[("sample-0-a", 0, (11, 22, 33))],
                target_records=[("sample-0-a", 0, (101, 102, 103))],
                features=torch.eye(1, 4, dtype=torch.float32),
            )

            message = r"(label 0|label=0).*(sample_id|eligible target|target bucket)"
            message += r"|(sample_id|eligible target|target bucket).*(label 0|label=0)"
            with self.assertRaisesRegex(ValueError, message):
                dataset = ManyToManyFeatureAlignedAffectNet(
                    source_index_path=source_index_path,
                    target_index_path=target_index_path,
                    source_feature_dir=cache_dir,
                    e0_checkpoint=checkpoint_path,
                    transform=None,
                )
                _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
