from __future__ import annotations

from pathlib import Path

from safa.data.dataset import load_rgb_image_strict
from safa.data.feature_cache import load_feature_cache
from safa.data.index_schema import read_index


LEGACY_CYCLIC_PAIRING = "legacy_cyclic"
BALANCED_EPOCH_CYCLE_PAIRING = "balanced_epoch_cycle"
PAIRING_STRATEGIES = {LEGACY_CYCLIC_PAIRING, BALANCED_EPOCH_CYCLE_PAIRING}


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
        pairing_strategy: str = LEGACY_CYCLIC_PAIRING,
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
        if type(pairing_strategy) is not str or pairing_strategy not in PAIRING_STRATEGIES:
            raise ValueError(
                f"pairing_strategy must be one of {sorted(PAIRING_STRATEGIES)}, "
                f"got {pairing_strategy!r}"
            )

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
        self.pairing_strategy = pairing_strategy
        self.target_buckets = self._build_target_buckets(self.target_records)
        self._validate_target_buckets()
        if self.pairing_strategy == BALANCED_EPOCH_CYCLE_PAIRING:
            self._initialize_balanced_pairing()

    def __len__(self) -> int:
        return len(self.source_records) * self.pairs_per_source

    def __getitem__(self, index: int | tuple[int, int]):
        pairing_epoch = 0
        if self.pairing_strategy == BALANCED_EPOCH_CYCLE_PAIRING:
            pairing_epoch, index = self._parse_balanced_index(index)

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
        if self.pairing_strategy == BALANCED_EPOCH_CYCLE_PAIRING:
            target_record = self._select_balanced_target(
                source_record,
                source_index,
                pair_round,
                pairing_epoch,
            )
        else:
            target_record = self._select_target(source_record, source_index, pair_round)
        target_label = target_record.label

        image = load_rgb_image_strict(target_record.image_path)
        if self.transform is not None:
            image = self.transform(image)

        pair_id = f"{source_record.sample_id}__to__{target_record.sample_id}__round{pair_round}"
        item = {
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
        if self.pairing_strategy == BALANCED_EPOCH_CYCLE_PAIRING:
            pair_id = f"{pair_id}__epoch{pairing_epoch}"
            item.update(
                {
                    "sample_id": pair_id,
                    "prior_sample_id": target_record.sample_id,
                    "pair_id": pair_id,
                    "pairing_epoch": pairing_epoch,
                    "pair_round": pair_round,
                    "pairing_strategy": self.pairing_strategy,
                }
            )
        return item

    @staticmethod
    def _parse_balanced_index(index: int | tuple[int, int]) -> tuple[int, int]:
        if type(index) is int:
            return 0, index
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError("balanced_epoch_cycle index must be int or an (epoch, index) tuple")
        epoch, flat_index = index
        if type(epoch) is not int or epoch < 0:
            raise ValueError(f"balanced_epoch_cycle epoch must be a non-negative int, got {epoch!r}")
        if type(flat_index) is not int:
            raise TypeError(f"balanced_epoch_cycle sample index must be int, got {type(flat_index).__name__}")
        return epoch, flat_index

    @staticmethod
    def _build_target_buckets(records):
        buckets = {}
        for record in records:
            buckets.setdefault(record.label, []).append(record)
        return buckets

    def _initialize_balanced_pairing(self) -> None:
        self._validate_unique_sample_ids(self.source_records, "source")
        self._validate_unique_sample_ids(self.target_records, "target")

        self._balanced_source_buckets = {}
        self._balanced_source_ranks = {}
        for source_index, record in enumerate(self.source_records):
            bucket = self._balanced_source_buckets.setdefault(record.label, [])
            self._balanced_source_ranks[source_index] = len(bucket)
            bucket.append((source_index, record))

        self._balanced_target_by_id = {
            label: {record.sample_id: record for record in records}
            for label, records in self.target_buckets.items()
        }
        self._balanced_equal_set_labels = {
            label
            for label, source_bucket in self._balanced_source_buckets.items()
            if {record.sample_id for _, record in source_bucket}
            == set(self._balanced_target_by_id.get(label, {}))
        }
        self._balanced_assignment_cache = {}

    @staticmethod
    def _validate_unique_sample_ids(records, index_role: str) -> None:
        sample_ids = [record.sample_id for record in records]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(
                f"balanced_epoch_cycle requires unique sample_id values in the {index_role} index"
            )

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

    def _select_balanced_target(
        self,
        source_record,
        source_index: int,
        pair_round: int,
        pairing_epoch: int,
    ):
        label = source_record.label
        source_rank = self._balanced_source_ranks[source_index]
        cycle = pairing_epoch * self.pairs_per_source + pair_round

        if label in self._balanced_equal_set_labels:
            source_bucket = self._balanced_source_buckets[label]
            shift = 1 + (self.pairing_seed + cycle) % (len(source_bucket) - 1)
            prior_source_rank = (source_rank + shift) % len(source_bucket)
            prior_sample_id = source_bucket[prior_source_rank][1].sample_id
            return self._balanced_target_by_id[label][prior_sample_id]

        assignment = self._balanced_assignment(label, pair_round, pairing_epoch)
        return self.target_buckets[label][assignment[source_rank]]

    def _balanced_assignment(self, label: int, pair_round: int, pairing_epoch: int):
        cache_key = (label, pair_round)
        cached = self._balanced_assignment_cache.get(cache_key)
        if cached is not None and cached[0] == pairing_epoch:
            return cached[1]

        source_bucket = self._balanced_source_buckets[label]
        target_bucket = self.target_buckets[label]
        source_count = len(source_bucket)
        target_count = len(target_bucket)
        cycle = pairing_epoch * self.pairs_per_source + pair_round
        phase = (self.pairing_seed + cycle) % target_count
        rotated_targets = [(phase + offset) % target_count for offset in range(target_count)]
        complete_cycles, remainder = divmod(source_count, target_count)
        assignment = rotated_targets * complete_cycles + rotated_targets[:remainder]

        for source_rank, (_, source_record) in enumerate(source_bucket):
            target_index = assignment[source_rank]
            if source_record.sample_id != target_bucket[target_index].sample_id:
                continue
            if source_count == 1:
                assignment[source_rank] = next(
                    candidate
                    for candidate in rotated_targets
                    if source_record.sample_id != target_bucket[candidate].sample_id
                )
                continue

            for distance in range(1, source_count):
                swap_rank = (source_rank + distance) % source_count
                swap_source = source_bucket[swap_rank][1]
                swap_target_index = assignment[swap_rank]
                if (
                    source_record.sample_id != target_bucket[swap_target_index].sample_id
                    and swap_source.sample_id != target_bucket[target_index].sample_id
                ):
                    assignment[source_rank], assignment[swap_rank] = (
                        assignment[swap_rank],
                        assignment[source_rank],
                    )
                    break
            else:
                raise ValueError(
                    f"No self-free balanced assignment for label {label} at epoch {pairing_epoch}, "
                    f"round {pair_round}"
                )

        result = tuple(assignment)
        self._balanced_assignment_cache[cache_key] = (pairing_epoch, result)
        return result
