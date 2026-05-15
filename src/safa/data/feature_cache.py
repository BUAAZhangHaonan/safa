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
            dataset=str(data["dataset"]),
            index_path=str(data["index_path"]),
            index_sha256=str(data["index_sha256"]),
            encoder_checkpoint=str(data["encoder_checkpoint"]),
            encoder_checkpoint_sha256=str(data["encoder_checkpoint_sha256"]),
            num_samples=int(data["num_samples"]),
            feature_dim=int(data["feature_dim"]),
            l2_normalized=bool(data["l2_normalized"]),
            dtype=str(data["dtype"]),
            shard=str(data["shard"]),
            shard_sha256=str(data["shard_sha256"]),
        )
        if manifest.feature_dim != 512:
            raise ValueError(f"Feature cache must use 512-d embeddings, got {manifest.feature_dim}")
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


def validate_manifest(cache_dir: str | Path, index_path: str | Path, checkpoint_path: str | Path) -> FeatureCacheManifest:
    cache_path = Path(cache_dir)
    manifest = load_manifest(cache_path)
    if manifest.index_sha256 != sha256_file(index_path):
        raise ValueError("Feature cache index_sha256 does not match the requested index")
    if manifest.encoder_checkpoint_sha256 != sha256_file(checkpoint_path):
        raise ValueError("Feature cache encoder_checkpoint_sha256 does not match the requested checkpoint")
    shard_path = cache_path / manifest.shard
    if manifest.shard_sha256 != sha256_file(shard_path):
        raise ValueError("Feature cache shard_sha256 does not match shard contents")
    return manifest


def load_feature_cache(cache_dir: str | Path, index_path: str | Path, checkpoint_path: str | Path):
    import torch

    manifest = validate_manifest(cache_dir, index_path, checkpoint_path)
    payload = torch.load(Path(cache_dir) / manifest.shard, map_location="cpu")
    required = {"features", "sample_ids", "labels"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Feature shard missing keys: {sorted(missing)}")
    features = payload["features"]
    if tuple(features.shape) != (manifest.num_samples, manifest.feature_dim):
        raise ValueError(f"Feature shape mismatch: got {tuple(features.shape)}, expected {(manifest.num_samples, manifest.feature_dim)}")
    norms = features.float().norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4):
        raise ValueError("Cached features are not L2-normalized")
    return payload, manifest


def write_manifest(cache_dir: str | Path, manifest: FeatureCacheManifest) -> None:
    path = Path(cache_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

