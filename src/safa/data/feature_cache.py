from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

from safa.utils.hashing import sha256_file


@dataclass(frozen=True)
class FeatureCacheManifest:
    dataset: str
    index_path: str
    index_sha256: str
    encoder_checkpoint: str
    encoder_checkpoint_sha256: str
    num_samples: int
    feature_dim: int
    l2_normalized: bool
    dtype: str
    shard: str
    shard_sha256: str
    sample_ids: list[str]
    labels: list[int]

    def __post_init__(self) -> None:
        if type(self.num_samples) is not int:
            raise ValueError(
                f"Feature cache manifest field num_samples must be int, got {type(self.num_samples).__name__}"
            )
        if type(self.feature_dim) is not int:
            raise ValueError(
                f"Feature cache manifest field feature_dim must be int, got {type(self.feature_dim).__name__}"
            )
        if type(self.l2_normalized) is not bool:
            raise ValueError(
                f"Feature cache manifest field l2_normalized must be bool, got {type(self.l2_normalized).__name__}"
            )
        if self.feature_dim <= 0:
            raise ValueError(f"Feature cache feature_dim must be positive, got {self.feature_dim}")
        if not self.l2_normalized:
            raise ValueError("Feature cache manifest must declare l2_normalized=true")
        if self.num_samples <= 0:
            raise ValueError("Feature cache manifest num_samples must be positive")
        sample_ids = _validate_str_list(self.sample_ids, "Feature cache manifest field sample_ids")
        labels = _validate_int_list(self.labels, "Feature cache manifest field labels")
        if len(sample_ids) != self.num_samples:
            raise ValueError(
                f"Feature cache manifest sample_ids length mismatch: got {len(sample_ids)}, expected {self.num_samples}"
            )
        if len(labels) != self.num_samples:
            raise ValueError(
                f"Feature cache manifest labels length mismatch: got {len(labels)}, expected {self.num_samples}"
            )
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Feature cache manifest sample_ids must be unique")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "labels", labels)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "FeatureCacheManifest":
        if not isinstance(data, dict):
            raise ValueError("Feature cache manifest must be a mapping")
        required = {
            "dataset",
            "index_path",
            "index_sha256",
            "encoder_checkpoint",
            "encoder_checkpoint_sha256",
            "num_samples",
            "feature_dim",
            "l2_normalized",
            "dtype",
            "shard",
            "shard_sha256",
            "sample_ids",
            "labels",
        }
        missing = required.difference(data)
        if missing:
            raise ValueError(f"Feature cache manifest missing fields: {sorted(missing)}")
        return cls(
            dataset=_require_str(data, "dataset"),
            index_path=_require_str(data, "index_path"),
            index_sha256=_require_str(data, "index_sha256"),
            encoder_checkpoint=_require_str(data, "encoder_checkpoint"),
            encoder_checkpoint_sha256=_require_str(data, "encoder_checkpoint_sha256"),
            num_samples=_require_int(data, "num_samples"),
            feature_dim=_require_int(data, "feature_dim"),
            l2_normalized=_require_bool(data, "l2_normalized"),
            dtype=_require_str(data, "dtype"),
            shard=_require_str(data, "shard"),
            shard_sha256=_require_str(data, "shard_sha256"),
            sample_ids=_require_str_list(data, "sample_ids"),
            labels=_require_int_list(data, "labels"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "index_path": self.index_path,
            "index_sha256": self.index_sha256,
            "encoder_checkpoint": self.encoder_checkpoint,
            "encoder_checkpoint_sha256": self.encoder_checkpoint_sha256,
            "num_samples": self.num_samples,
            "feature_dim": self.feature_dim,
            "l2_normalized": self.l2_normalized,
            "dtype": self.dtype,
            "shard": self.shard,
            "shard_sha256": self.shard_sha256,
            "sample_ids": list(self.sample_ids),
            "labels": list(self.labels),
        }


def load_manifest(cache_dir: str | Path) -> FeatureCacheManifest:
    path = Path(cache_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache manifest does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return FeatureCacheManifest.from_mapping(data)


def _require_str(data: dict[str, Any], field: str) -> str:
    value = data[field]
    if type(value) is not str:
        raise ValueError(f"Feature cache manifest field {field} must be str, got {type(value).__name__}")
    return value


def _require_int(data: dict[str, Any], field: str) -> int:
    value = data[field]
    if type(value) is not int:
        raise ValueError(f"Feature cache manifest field {field} must be int, got {type(value).__name__}")
    return value


def _require_bool(data: dict[str, Any], field: str) -> bool:
    value = data[field]
    if type(value) is not bool:
        raise ValueError(f"Feature cache manifest field {field} must be bool, got {type(value).__name__}")
    return value


def _validate_str_list(value: Any, context: str) -> list[str]:
    if type(value) is not list:
        raise ValueError(f"{context} must be list, got {type(value).__name__}")
    for index, item in enumerate(value):
        if type(item) is not str:
            raise ValueError(f"{context}[{index}] must be str, got {type(item).__name__}")
    return list(value)


def _validate_int_list(value: Any, context: str) -> list[int]:
    if type(value) is not list:
        raise ValueError(f"{context} must be list, got {type(value).__name__}")
    for index, item in enumerate(value):
        if type(item) is not int:
            raise ValueError(f"{context}[{index}] must be int, got {type(item).__name__}")
    return list(value)


def _require_str_list(data: dict[str, Any], field: str) -> list[str]:
    return _validate_str_list(data[field], f"Feature cache manifest field {field}")


def _require_int_list(data: dict[str, Any], field: str) -> list[int]:
    return _validate_int_list(data[field], f"Feature cache manifest field {field}")


def _require_shard_str_list(payload: dict[str, Any], field: str) -> list[str]:
    return _validate_str_list(payload[field], f"Feature shard {field}")


def _require_shard_int_list(payload: dict[str, Any], field: str) -> list[int]:
    return _validate_int_list(payload[field], f"Feature shard {field}")


def load_feature_cache(cache_dir: str | Path, index_path: str | Path, checkpoint_path: str | Path):
    import hashlib
    import io

    import torch

    cache_path = Path(cache_dir)
    manifest = load_manifest(cache_path)

    if manifest.index_sha256 != sha256_file(index_path):
        raise ValueError("Feature cache index_sha256 does not match the requested index")
    if manifest.encoder_checkpoint_sha256 != sha256_file(checkpoint_path):
        raise ValueError("Feature cache encoder_checkpoint_sha256 does not match the requested checkpoint")

    # Read shard file once into memory, verify hash, then load from the same bytes.
    # This eliminates the TOCTOU race where the file could change between hash
    # verification and torch.load.
    shard_path = cache_path / manifest.shard
    shard_bytes = shard_path.read_bytes()
    if manifest.shard_sha256 != hashlib.sha256(shard_bytes).hexdigest():
        raise ValueError("Feature cache shard_sha256 does not match shard contents")
    payload = torch.load(io.BytesIO(shard_bytes), map_location="cpu", weights_only=False)

    required = {"features", "sample_ids", "labels"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Feature shard missing keys: {sorted(missing)}")
    features = payload["features"]
    if not isinstance(features, torch.Tensor):
        raise ValueError(f"Feature shard features must be a torch.Tensor, got {type(features).__name__}")
    if tuple(features.shape) != (manifest.num_samples, manifest.feature_dim):
        raise ValueError(f"Feature shape mismatch: got {tuple(features.shape)}, expected {(manifest.num_samples, manifest.feature_dim)}")
    feature_dtype = str(features.dtype).replace("torch.", "")
    if feature_dtype != manifest.dtype:
        raise ValueError(f"Feature dtype mismatch: got {feature_dtype}, expected {manifest.dtype}")
    sample_ids = _require_shard_str_list(payload, "sample_ids")
    labels = _require_shard_int_list(payload, "labels")
    if len(sample_ids) != manifest.num_samples:
        raise ValueError(f"Feature sample_ids length mismatch: got {len(sample_ids)}, expected {manifest.num_samples}")
    if len(labels) != manifest.num_samples:
        raise ValueError(f"Feature labels length mismatch: got {len(labels)}, expected {manifest.num_samples}")
    if sample_ids != manifest.sample_ids:
        raise ValueError("Feature shard sample_ids do not match manifest sample_ids")
    if labels != manifest.labels:
        raise ValueError("Feature shard labels do not match manifest labels")
    norms = features.float().norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4):
        raise ValueError("Cached features are not L2-normalized")
    return payload, manifest


def write_manifest(cache_dir: str | Path, manifest: FeatureCacheManifest) -> None:
    path = Path(cache_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
