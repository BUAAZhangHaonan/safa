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


if __name__ == "__main__":
    unittest.main()
