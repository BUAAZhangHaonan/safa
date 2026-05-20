"""Distributed training utilities shared across training loops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    device: object
    backend: str


def init_distributed(config: dict) -> DistributedContext:
    """Initialize distributed training context from config and env vars."""
    import os
    import torch
    import torch.distributed as dist

    from safa.utils.device import require_cuda_device

    world_size_raw = os.environ.get("WORLD_SIZE")
    rank_raw = os.environ.get("RANK")
    local_rank_raw = os.environ.get("LOCAL_RANK")
    world_size = int(world_size_raw) if world_size_raw else 1
    if world_size > 1:
        if rank_raw is None or local_rank_raw is None:
            raise RuntimeError(
                "DDP requires RANK and LOCAL_RANK when WORLD_SIZE > 1"
            )
        rank = int(rank_raw)
        local_rank = int(local_rank_raw)
        backend = str(config.get("distributed", {}).get("backend", "nccl"))
        if backend not in {"nccl", "gloo"}:
            raise ValueError(f"Unsupported DDP backend: {backend}")
        device = require_cuda_device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        if not dist.is_initialized():
            if backend == "nccl":
                dist.init_process_group(backend=backend, device_id=device)
            else:
                dist.init_process_group(backend=backend)
        return DistributedContext(
            enabled=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            is_main=rank == 0,
            device=device,
            backend=backend,
        )
    device = require_cuda_device(str(config["device"]))
    if device.index is not None:
        torch.cuda.set_device(device)
    return DistributedContext(
        enabled=False,
        rank=0,
        local_rank=device.index or 0,
        world_size=1,
        is_main=True,
        device=device,
        backend="single",
    )


def barrier(distributed: DistributedContext) -> None:
    if not distributed.enabled:
        return
    import torch.distributed as dist

    if distributed.backend == "nccl":
        dist.barrier(device_ids=[distributed.local_rank])
    else:
        dist.barrier()


def cleanup_distributed(distributed: DistributedContext) -> None:
    if not distributed.enabled:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def reduce_train_metrics(
    train_loss_sum: float, seen: int, device, distributed: DistributedContext
) -> dict:
    """Reduce (sum, count) across ranks and return {loss, samples}."""
    import torch

    values = torch.tensor(
        [train_loss_sum, float(seen)], device=device, dtype=torch.float64
    )
    if distributed.enabled:
        import torch.distributed as dist

        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_loss, total_seen = values.tolist()
    return {"loss": total_loss / max(total_seen, 1), "samples": int(total_seen)}


def broadcast_early_stop(should_stop: bool, device, distributed: DistributedContext) -> bool:
    """Broadcast early-stop decision from rank 0 to all ranks."""
    if not distributed.enabled:
        return should_stop
    import torch
    import torch.distributed as dist

    flag = torch.tensor([1.0 if should_stop else 0.0], device=device, dtype=torch.float64)
    dist.broadcast(flag, src=0)
    return bool(flag.item() > 0.5)
