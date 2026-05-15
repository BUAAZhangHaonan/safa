from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from safa.data.feature_cache import FeatureCacheManifest, load_manifest


class FeatureCacheManifestTests(unittest.TestCase):
    def test_manifest_requires_512_l2_features(self) -> None:
        data = {
            "dataset": "AffectNet",
            "index_path": "index.jsonl",
            "index_sha256": "a",
            "encoder_checkpoint": "best.pt",
            "encoder_checkpoint_sha256": "b",
            "num_samples": 1,
            "feature_dim": 512,
            "l2_normalized": True,
            "dtype": "float32",
            "shard": "features.pt",
            "shard_sha256": "c",
        }
        self.assertEqual(FeatureCacheManifest.from_mapping(data).feature_dim, 512)
        with self.assertRaises(ValueError):
            FeatureCacheManifest.from_mapping({**data, "feature_dim": 256})
        with self.assertRaises(ValueError):
            FeatureCacheManifest.from_mapping({**data, "l2_normalized": False})

    def test_load_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            payload = {
                "dataset": "AffectNet",
                "index_path": "index.jsonl",
                "index_sha256": "a",
                "encoder_checkpoint": "best.pt",
                "encoder_checkpoint_sha256": "b",
                "num_samples": 1,
                "feature_dim": 512,
                "l2_normalized": True,
                "dtype": "float32",
                "shard": "features.pt",
                "shard_sha256": "c",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_manifest(tmp).dataset, "AffectNet")


if __name__ == "__main__":
    unittest.main()

