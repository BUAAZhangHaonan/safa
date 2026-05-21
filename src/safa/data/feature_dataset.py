from __future__ import annotations

from pathlib import Path

from safa.data.dataset import load_rgb_image_strict
from safa.data.feature_cache import load_feature_cache
from safa.data.index_schema import read_index


class FeatureAlignedAffectNet:
    def __init__(self, index_path: str | Path, feature_dir: str | Path, e0_checkpoint: str | Path, transform):
        import torch

        self.records = read_index(Path(index_path))
        payload, manifest = load_feature_cache(feature_dir, index_path, e0_checkpoint)
        sample_ids = list(payload["sample_ids"])
        if sample_ids != [record.sample_id for record in self.records]:
            raise ValueError("Feature cache sample_id order does not match index order")
        features = payload["features"]
        if features.dtype != torch.float32:
            raise ValueError(f"Feature cache tensor must be float32 for training data, got {features.dtype}")
        labels = list(payload["labels"])
        invalid_labels = [item for item in labels if type(item) is not int]
        if invalid_labels:
            raise ValueError(f"Feature cache labels must be int values, got {type(invalid_labels[0]).__name__}")
        self.features = features
        self.labels = labels
        self.manifest = manifest
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = load_rgb_image_strict(record.image_path)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        if label != record.label:
            raise ValueError(f"Feature label mismatch for {record.sample_id}: feature={label}, index={record.label}")
        return {
            "image": image,
            "z": self.features[index],
            "label": label,
            "sample_id": record.sample_id,
        }
