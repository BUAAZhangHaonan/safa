from __future__ import annotations

from pathlib import Path

from safa.data.dataset import AffectNetRecords
from safa.data.feature_cache import FeatureCacheManifest, write_manifest
from safa.models.e0 import freeze_e0, load_e0_checkpoint
from safa.training.transforms import eval_transform
from safa.utils.device import assert_finite_tensor, require_cuda_device
from safa.utils.hashing import sha256_file
from safa.utils.seed import set_seed


def cache_e0_from_config(config: dict) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    set_seed(int(config["seed"]))
    device = require_cuda_device(str(config["device"]))
    index_path = Path(config["index"])
    checkpoint_path = Path(config["checkpoint"])
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_e0_checkpoint(checkpoint_path, device=str(device))
    model.to(device)
    freeze_e0(model)
    dataset = AffectNetRecords(index_path, transform=eval_transform(int(config["image_size"])))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
    )
    features = []
    sample_ids: list[str] = []
    labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="cache_e0"):
            images = batch["image"].to(device, non_blocking=True)
            output = model(images)
            embedding = output["embedding"].detach().cpu().float()
            assert_finite_tensor("cached_embedding", embedding)
            norms = embedding.norm(dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4):
                raise RuntimeError("E0 emitted non-normalized embeddings")
            features.append(embedding)
            sample_ids.extend(list(batch["sample_id"]))
            labels.extend([int(item) for item in batch["label"]])
    tensor = torch.cat(features, dim=0)
    shard = "features.pt"
    shard_path = out_dir / shard
    torch.save({"features": tensor, "sample_ids": sample_ids, "labels": labels}, shard_path)
    manifest = FeatureCacheManifest(
        dataset="AffectNet",
        index_path=str(index_path),
        index_sha256=sha256_file(index_path),
        encoder_checkpoint=str(checkpoint_path),
        encoder_checkpoint_sha256=sha256_file(checkpoint_path),
        num_samples=int(tensor.shape[0]),
        feature_dim=int(tensor.shape[1]),
        l2_normalized=True,
        dtype=str(tensor.dtype).replace("torch.", ""),
        shard=shard,
        shard_sha256=sha256_file(shard_path),
        sample_ids=sample_ids,
        labels=labels,
    )
    write_manifest(out_dir, manifest)
    return manifest.to_json_dict()
