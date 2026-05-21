from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from safa.data.feature_cache import FeatureCacheManifest, load_feature_cache, load_manifest
from safa.utils.hashing import sha256_file


class FeatureCacheManifestTests(unittest.TestCase):
    def _manifest_payload(self, *, feature_dim: int = 128, num_samples: int = 1, shard_sha256: str = "c") -> dict:
        return {
            "dataset": "AffectNet",
            "index_path": "index.jsonl",
            "index_sha256": "a",
            "encoder_checkpoint": "best.pt",
            "encoder_checkpoint_sha256": "b",
            "num_samples": num_samples,
            "feature_dim": feature_dim,
            "l2_normalized": True,
            "dtype": "float32",
            "shard": "features.pt",
            "shard_sha256": shard_sha256,
        }

    def test_manifest_accepts_positive_feature_dim_and_requires_l2_features(self) -> None:
        data = self._manifest_payload(feature_dim=128)
        self.assertEqual(FeatureCacheManifest.from_mapping(data).feature_dim, 128)
        with self.assertRaises(ValueError):
            FeatureCacheManifest.from_mapping({**data, "feature_dim": 0})
        with self.assertRaises(ValueError):
            FeatureCacheManifest.from_mapping({**data, "feature_dim": -1})
        with self.assertRaises(ValueError):
            FeatureCacheManifest.from_mapping({**data, "l2_normalized": False})

    def test_load_feature_cache_uses_manifest_feature_dim(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path = root / "index.jsonl"
            checkpoint_path = root / "best.pt"
            index_path.write_text("index", encoding="utf-8")
            checkpoint_path.write_text("checkpoint", encoding="utf-8")

            features = torch.zeros(1, 128)
            features[0, 0] = 1.0
            shard_path = root / "features.pt"
            torch.save({"features": features, "sample_ids": ["sample-1"], "labels": [0]}, shard_path)

            manifest = self._manifest_payload(feature_dim=128, shard_sha256=sha256_file(shard_path))
            manifest["index_sha256"] = sha256_file(index_path)
            manifest["encoder_checkpoint_sha256"] = sha256_file(checkpoint_path)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            payload, loaded_manifest = load_feature_cache(root, index_path, checkpoint_path)

        self.assertEqual(tuple(payload["features"].shape), (1, 128))
        self.assertEqual(loaded_manifest.feature_dim, 128)

    def test_load_feature_cache_rejects_feature_shape_mismatch(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path = root / "index.jsonl"
            checkpoint_path = root / "best.pt"
            index_path.write_text("index", encoding="utf-8")
            checkpoint_path.write_text("checkpoint", encoding="utf-8")

            features = torch.zeros(1, 127)
            features[0, 0] = 1.0
            shard_path = root / "features.pt"
            torch.save({"features": features, "sample_ids": ["sample-1"], "labels": [0]}, shard_path)

            manifest = self._manifest_payload(feature_dim=128, shard_sha256=sha256_file(shard_path))
            manifest["index_sha256"] = sha256_file(index_path)
            manifest["encoder_checkpoint_sha256"] = sha256_file(checkpoint_path)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_feature_cache(root, index_path, checkpoint_path)

    def test_manifest_requires_feature_dim(self) -> None:
        data = {
            "dataset": "AffectNet",
            "index_path": "index.jsonl",
            "index_sha256": "a",
            "encoder_checkpoint": "best.pt",
            "encoder_checkpoint_sha256": "b",
            "num_samples": 1,
            "l2_normalized": True,
            "dtype": "float32",
            "shard": "features.pt",
            "shard_sha256": "c",
        }
        with self.assertRaises(ValueError):
            FeatureCacheManifest.from_mapping(data)

    def test_load_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            payload = self._manifest_payload(feature_dim=128)
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_manifest(tmp).dataset, "AffectNet")


if __name__ == "__main__":
    unittest.main()
