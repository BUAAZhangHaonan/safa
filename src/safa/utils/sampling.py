from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
import hashlib


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    payload = f"{int(base_seed)}\0{str(sample_id)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63)


def make_x_init_for_sample_ids(sample_ids: Iterable[str], base_seed: int, image_size: int, device, dtype) -> torch.Tensor:
    import torch

    ids = [str(sample_id) for sample_id in sample_ids]
    size = int(image_size)
    if size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if not ids:
        return torch.empty((0, 3, size, size), device="cpu", dtype=torch.float32).to(device=device, dtype=dtype)

    samples = []
    for sample_id in ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_sample_seed(int(base_seed), sample_id))
        samples.append(torch.randn((3, size, size), generator=generator, device="cpu", dtype=torch.float32))
    return torch.stack(samples, dim=0).to(device=device, dtype=dtype)


def sampling_base_seed_from_config(config: Mapping[str, Any]) -> int:
    if "sampling_seed" in config and config["sampling_seed"] is not None:
        return int(config["sampling_seed"])
    if "seed" in config and config["seed"] is not None:
        return int(config["seed"])
    raise KeyError("Config requires sampling_seed or seed for stable sample initialization")
