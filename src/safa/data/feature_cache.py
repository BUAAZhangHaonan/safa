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
        }
        missing = required.difference(data)
        if missing:
            raise ValueError(f"Feature cache manifest missing fields: {sorted(missing)}")
        manifest = cls(
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
        )
        if manifest.feature_dim <= 0:
            raise ValueError(f"Feature cache feature_dim must be positive, got {manifest.feature_dim}")
        if not manifest.l2_normalized:
            raise ValueError("Feature cache manifest must declare l2_normalized=true")
        if manifest.num_samples <= 0:
            raise ValueError("Feature cache manifest num_samples must be positive")
        return manifest

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
    sample_ids = payload["sample_ids"]
    labels = payload["labels"]
    if len(sample_ids) != manifest.num_samples:
        raise ValueError(f"Feature sample_ids length mismatch: got {len(sample_ids)}, expected {manifest.num_samples}")
    if len(labels) != manifest.num_samples:
        raise ValueError(f"Feature labels length mismatch: got {len(labels)}, expected {manifest.num_samples}")
    norms = features.float().norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4):
        raise ValueError("Cached features are not L2-normalized")
    return payload, manifest


def write_manifest(cache_dir: str | Path, manifest: FeatureCacheManifest) -> None:
    path = Path(cache_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
