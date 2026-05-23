from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from safa.data.feature_cache import FeatureCacheManifest, load_feature_cache, load_manifest
from safa.utils.hashing import sha256_file


class FeatureCacheManifestTests(unittest.TestCase):
    def _manifest_payload(
        self,
        *,
        feature_dim: int = 128,
        num_samples: int = 1,
        shard_sha256: str = "c",
        sample_ids: list[str] | None = None,
        labels: list[int] | None = None,
    ) -> dict:
        if sample_ids is None:
            sample_ids = [f"sample-{idx + 1}" for idx in range(num_samples)]
        if labels is None:
            labels = [idx % 8 for idx in range(num_samples)]
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
            "sample_ids": sample_ids,
            "labels": labels,
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

    def test_manifest_rejects_wrong_field_types_without_coercion(self) -> None:
        data = self._manifest_payload(feature_dim=128)
        invalid_values = {
            "dataset": 123,
            "index_path": Path("index.jsonl"),
            "index_sha256": 123,
            "encoder_checkpoint": 123,
            "encoder_checkpoint_sha256": 123,
            "num_samples": "1",
            "feature_dim": "128",
            "l2_normalized": "false",
            "dtype": 123,
            "shard": 123,
            "shard_sha256": 123,
            "sample_ids": ("sample-1",),
            "labels": (0,),
        }
        for field, value in invalid_values.items():
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    FeatureCacheManifest.from_mapping({**data, field: value})

    def test_manifest_requires_sample_ids_and_labels(self) -> None:
        data = self._manifest_payload(feature_dim=128)
        for field in ("sample_ids", "labels"):
            with self.subTest(field=field):
                payload = dict(data)
                payload.pop(field)
                with self.assertRaises(ValueError):
                    FeatureCacheManifest.from_mapping(payload)

    def test_manifest_rejects_sample_metadata_length_mismatch(self) -> None:
        data = self._manifest_payload(feature_dim=128, num_samples=2)
        invalid_values = {
            "sample_ids": ["sample-1"],
            "labels": [0],
        }
        for field, value in invalid_values.items():
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    FeatureCacheManifest.from_mapping({**data, field: value})

    def test_manifest_rejects_sample_metadata_item_types_and_duplicate_ids(self) -> None:
        data = self._manifest_payload(feature_dim=128, num_samples=2)
        invalid_payloads = [
            {**data, "sample_ids": ["sample-1", 2]},
            {**data, "labels": [0, "1"]},
            {**data, "labels": [0, True]},
            {**data, "sample_ids": ["sample-1", "sample-1"]},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    FeatureCacheManifest.from_mapping(payload)

    def test_manifest_constructor_validates_sample_metadata(self) -> None:
        data = self._manifest_payload(feature_dim=128, num_samples=2)
        with self.assertRaises(ValueError):
            FeatureCacheManifest(**{**data, "sample_ids": ["sample-1"]})
        with self.assertRaises(ValueError):
            FeatureCacheManifest(**{**data, "labels": [0, "1"]})
        with self.assertRaises(ValueError):
            FeatureCacheManifest(**{**data, "sample_ids": ["sample-1", "sample-1"]})

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
        self.assertEqual(loaded_manifest.sample_ids, ["sample-1"])
        self.assertEqual(loaded_manifest.labels, [0])

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

    def test_load_feature_cache_rejects_dtype_mismatch(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path = root / "index.jsonl"
            checkpoint_path = root / "best.pt"
            index_path.write_text("index", encoding="utf-8")
            checkpoint_path.write_text("checkpoint", encoding="utf-8")

            features = torch.zeros(1, 128, dtype=torch.float64)
            features[0, 0] = 1.0
            shard_path = root / "features.pt"
            torch.save({"features": features, "sample_ids": ["sample-1"], "labels": [0]}, shard_path)

            manifest = self._manifest_payload(feature_dim=128, shard_sha256=sha256_file(shard_path))
            manifest["index_sha256"] = sha256_file(index_path)
            manifest["encoder_checkpoint_sha256"] = sha256_file(checkpoint_path)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_feature_cache(root, index_path, checkpoint_path)

    def test_load_feature_cache_rejects_sample_id_and_label_length_mismatch(self) -> None:
        import torch

        for payload_override in ({"sample_ids": []}, {"labels": []}):
            with self.subTest(payload_override=payload_override):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    index_path = root / "index.jsonl"
                    checkpoint_path = root / "best.pt"
                    index_path.write_text("index", encoding="utf-8")
                    checkpoint_path.write_text("checkpoint", encoding="utf-8")

                    features = torch.zeros(1, 128)
                    features[0, 0] = 1.0
                    payload = {"features": features, "sample_ids": ["sample-1"], "labels": [0]}
                    payload.update(payload_override)
                    shard_path = root / "features.pt"
                    torch.save(payload, shard_path)

                    manifest = self._manifest_payload(feature_dim=128, shard_sha256=sha256_file(shard_path))
                    manifest["index_sha256"] = sha256_file(index_path)
                    manifest["encoder_checkpoint_sha256"] = sha256_file(checkpoint_path)
                    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

                    with self.assertRaises(ValueError):
                        load_feature_cache(root, index_path, checkpoint_path)

    def test_load_feature_cache_rejects_manifest_sample_ids_mismatch(self) -> None:
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

            manifest = self._manifest_payload(
                feature_dim=128,
                shard_sha256=sha256_file(shard_path),
                sample_ids=["sample-2"],
                labels=[0],
            )
            manifest["index_sha256"] = sha256_file(index_path)
            manifest["encoder_checkpoint_sha256"] = sha256_file(checkpoint_path)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_feature_cache(root, index_path, checkpoint_path)

    def test_load_feature_cache_rejects_manifest_labels_mismatch(self) -> None:
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

            manifest = self._manifest_payload(
                feature_dim=128,
                shard_sha256=sha256_file(shard_path),
                sample_ids=["sample-1"],
                labels=[1],
            )
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

    def test_cache_e0_from_config_records_collected_sample_metadata(self) -> None:
        import torch

        from safa.training import cache_e0 as cache_module

        class FakeImages:
            def to(self, device, non_blocking: bool = False):
                return self

        class FakeModel:
            def to(self, device):
                return self

            def eval(self) -> None:
                return None

            def __call__(self, images):
                return {"embedding": torch.eye(2, dtype=torch.float32)}

        def fake_loader(dataset, **kwargs):
            return [
                {
                    "image": FakeImages(),
                    "sample_id": ["sample-a", "sample-b"],
                    "label": torch.tensor([3, 5]),
                }
            ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path = root / "index.jsonl"
            checkpoint_path = root / "best.pt"
            out_dir = root / "features"
            index_path.write_text("index", encoding="utf-8")
            checkpoint_path.write_text("checkpoint", encoding="utf-8")
            config = {
                "seed": 1,
                "device": "cpu",
                "index": str(index_path),
                "checkpoint": str(checkpoint_path),
                "out_dir": str(out_dir),
                "image_size": 64,
                "batch_size": 2,
                "num_workers": 0,
            }

            with patch.object(cache_module, "set_seed"), patch.object(
                cache_module, "require_cuda_device", return_value=torch.device("cpu")
            ), patch.object(cache_module, "load_e0_checkpoint", return_value=(FakeModel(), {})), patch.object(
                cache_module, "freeze_e0"
            ), patch.object(
                cache_module, "AffectNetRecords", return_value=object()
            ), patch.object(
                cache_module, "eval_transform", return_value=object()
            ), patch(
                "torch.utils.data.DataLoader", side_effect=fake_loader
            ), patch(
                "tqdm.tqdm", side_effect=lambda iterable, **kwargs: iterable
            ):
                result = cache_module.cache_e0_from_config(config)

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            shard = torch.load(out_dir / "features.pt", map_location="cpu", weights_only=False)

        self.assertEqual(result["sample_ids"], ["sample-a", "sample-b"])
        self.assertEqual(result["labels"], [3, 5])
        self.assertEqual(manifest["sample_ids"], ["sample-a", "sample-b"])
        self.assertEqual(manifest["labels"], [3, 5])
        self.assertEqual(shard["sample_ids"], ["sample-a", "sample-b"])
        self.assertEqual(shard["labels"], [3, 5])


if __name__ == "__main__":
    unittest.main()
