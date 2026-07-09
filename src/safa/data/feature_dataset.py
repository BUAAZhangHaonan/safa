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


class ManyToManyFeatureAlignedAffectNet:
    def __init__(
        self,
        index_path: str | Path | None = None,
        feature_dir: str | Path | None = None,
        e0_checkpoint: str | Path | None = None,
        target_index_path: str | Path | None = None,
        pairing_seed: int = 1,
        pairs_per_source: int = 1,
        transform=None,
        *,
        source_index_path: str | Path | None = None,
        source_feature_dir: str | Path | None = None,
    ):
        import torch

        if index_path is not None and source_index_path is not None and Path(index_path) != Path(source_index_path):
            raise ValueError("Conflicting source index paths: index_path and source_index_path differ")
        if feature_dir is not None and source_feature_dir is not None and Path(feature_dir) != Path(source_feature_dir):
            raise ValueError("Conflicting source feature dirs: feature_dir and source_feature_dir differ")
        if index_path is None:
            index_path = source_index_path
        if feature_dir is None:
            feature_dir = source_feature_dir
        if index_path is None:
            raise ValueError("source index_path is required")
        if feature_dir is None:
            raise ValueError("source feature_dir is required")
        if e0_checkpoint is None:
            raise ValueError("e0_checkpoint is required")
        if target_index_path is None:
            raise ValueError("target_index_path is required")
        if type(pairing_seed) is not int:
            raise ValueError(f"pairing_seed must be int, got {type(pairing_seed).__name__}")
        if type(pairs_per_source) is not int:
            raise ValueError(f"pairs_per_source must be int, got {type(pairs_per_source).__name__}")
        if pairs_per_source <= 0:
            raise ValueError(f"pairs_per_source must be positive, got {pairs_per_source}")

        self.records = read_index(Path(index_path))
        self.source_records = self.records
        self.target_records = read_index(Path(target_index_path))
        payload, manifest = load_feature_cache(feature_dir, index_path, e0_checkpoint)
        sample_ids = list(payload["sample_ids"])
        if sample_ids != [record.sample_id for record in self.source_records]:
            raise ValueError("Feature cache sample_id order does not match index order")
        features = payload["features"]
        if features.dtype != torch.float32:
            raise ValueError(f"Feature cache tensor must be float32 for training data, got {features.dtype}")
        labels = list(payload["labels"])
        invalid_labels = [item for item in labels if type(item) is not int]
        if invalid_labels:
            raise ValueError(f"Feature cache labels must be int values, got {type(invalid_labels[0]).__name__}")
        for label, record in zip(labels, self.source_records, strict=True):
            if label != record.label:
                raise ValueError(f"Feature label mismatch for {record.sample_id}: feature={label}, index={record.label}")

        self.features = features
        self.labels = labels
        self.manifest = manifest
        self.transform = transform
        self.pairing_seed = pairing_seed
        self.pairs_per_source = pairs_per_source
        self.target_buckets = self._build_target_buckets(self.target_records)
        self._validate_target_buckets()

    def __len__(self) -> int:
        return len(self.source_records) * self.pairs_per_source

    def __getitem__(self, index: int):
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(f"Many-to-many feature dataset index out of range: {index}")

        source_index = index // self.pairs_per_source
        pair_round = index % self.pairs_per_source
        source_record = self.source_records[source_index]
        source_label = self.labels[source_index]
        if source_label != source_record.label:
            raise ValueError(
                f"Feature label mismatch for {source_record.sample_id}: "
                f"feature={source_label}, index={source_record.label}"
            )
        target_record = self._select_target(source_record, source_index, pair_round)
        target_label = target_record.label

        image = load_rgb_image_strict(target_record.image_path)
        if self.transform is not None:
            image = self.transform(image)

        pair_id = f"{source_record.sample_id}__to__{target_record.sample_id}__round{pair_round}"
        return {
            "image": image,
            "z": self.features[source_index],
            "label": source_label,
            "sample_id": pair_id,
            "source_sample_id": source_record.sample_id,
            "target_sample_id": target_record.sample_id,
            "source_label": source_label,
            "target_label": target_label,
            "pair_id": pair_id,
        }

    @staticmethod
    def _build_target_buckets(records):
        buckets = {}
        for record in records:
            buckets.setdefault(record.label, []).append(record)
        return buckets

    def _validate_target_buckets(self) -> None:
        for source_record in self.source_records:
            self._eligible_targets(source_record)

    def _eligible_targets(self, source_record):
        bucket = self.target_buckets.get(source_record.label, [])
        eligible = [record for record in bucket if record.sample_id != source_record.sample_id]
        if not eligible:
            raise ValueError(
                f"No eligible target in target bucket for label {source_record.label} "
                f"excluding source sample_id {source_record.sample_id}"
            )
        return eligible

    def _select_target(self, source_record, source_index: int, pair_round: int):
        eligible_targets = self._eligible_targets(source_record)
        target_position = (
            source_index + self.pairing_seed + pair_round
        ) % len(eligible_targets)
        return eligible_targets[target_position]
