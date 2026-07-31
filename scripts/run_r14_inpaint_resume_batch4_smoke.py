#!/usr/bin/env python3
"""One real DDP update to measure the locked R14 two-GPU batch=4 peak."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from safa.data.r14_spatial import R14SpatialEvalDataset
from safa.models.generator import build_generator
from safa.models.meanflow_sit import encode_inpaint_training_latents
from safa.training.g_loop import (
    _generator_config_from_train_config,
    _r14_resume_contract,
    _validate_r14_resume_checkpoint,
    _validate_r14_training_state_load,
    _validate_train_g_config,
)
from safa.training.latent_codec import build_latent_codec_from_train_config
from safa.training.transforms import r14_joint_transform
from safa.utils.ema import ExponentialMovingAverage


MIB = 1024**2
NCCL_ENV = {"NCCL_IB_DISABLE": "1", "NCCL_P2P_DISABLE": "0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _optimizer_steps(optimizer: torch.optim.Optimizer) -> set[int]:
    steps: set[int] = set()
    for state in optimizer.state.values():
        value = state.get("step")
        if value is None:
            raise RuntimeError("restored optimizer parameter state lacks step")
        numeric = float(value.detach().item()) if torch.is_tensor(value) else float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise RuntimeError(f"optimizer step must be a finite integer, got {numeric!r}")
        steps.add(int(numeric))
    if not steps:
        raise RuntimeError("restored optimizer state is empty")
    return steps


class _SmokeTrainingStep(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module) -> None:
        super().__init__()
        self.generator = generator

    def forward(
        self,
        target_latent: torch.Tensor,
        source_z: torch.Tensor,
        context_latent: torch.Tensor,
        latent_mask: torch.Tensor,
        seed: int,
    ):
        rng = torch.Generator(device=target_latent.device).manual_seed(seed)
        return self.generator.flow_matching_loss(
            target_latent,
            source_z,
            generator=rng,
            context_latent=context_latent,
            latent_mask=latent_mask,
        )


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size != 2 or local_rank not in range(2) or rank not in range(2):
        raise RuntimeError("R14 resume batch4 smoke requires exactly two torchrun ranks")
    for name, expected in NCCL_ENV.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"{name} must be {expected!r}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    output_exists = torch.tensor(
        [1 if args.output_dir.exists() else 0], device=device, dtype=torch.int64
    )
    dist.all_reduce(output_exists, op=dist.ReduceOp.MAX)
    if int(output_exists.item()) != 0:
        raise FileExistsError(f"refusing to reuse smoke output: {args.output_dir}")
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    config = dict(_mapping(yaml.safe_load(args.config.read_text(encoding="utf-8")), "config"))
    _validate_train_g_config(config)
    if config.get("global_batch_size") != 8 or config.get("per_device_batch_size") != 4:
        raise RuntimeError("R14 resume smoke requires global batch 8 and per-rank batch 4")
    contract = _r14_resume_contract(config)
    if contract is None:
        raise RuntimeError("R14 resume smoke requires the registered resume contract")
    if (
        contract["source_global_step"] != 2432
        or contract["additional_optimizer_steps"] != 128
        or contract["target_global_step"] != 2560
    ):
        raise RuntimeError("R14 resume smoke received the wrong source/target step contract")

    dataset = R14SpatialEvalDataset(
        args.manifest,
        Path(str(config["eval_index"])),
        Path(str(config["eval_features"])),
        Path(str(config["e0_checkpoint"])),
        r14_joint_transform(256, horizontal_flip_probability=0.0),
    )
    if len(dataset) != 8:
        raise RuntimeError(f"batch4 smoke manifest must contain exactly 8 samples, got {len(dataset)}")
    indices = list(range(rank * 4, (rank + 1) * 4))
    loader = DataLoader(Subset(dataset, indices), batch_size=4, shuffle=False, num_workers=0)
    if len(loader) != 1:
        raise RuntimeError("each smoke rank must receive one full batch of four")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint_path = Path(str(config["resume_from"]))
    checkpoint = _mapping(
        torch.load(checkpoint_path, map_location=device, weights_only=True),
        "source checkpoint",
    )
    _validate_r14_resume_checkpoint(checkpoint, str(checkpoint_path), contract)
    generator_config = _generator_config_from_train_config(config)
    generator = build_generator(generator_config.to_dict()).to(device).train()
    incompatible = generator.load_state_dict(
        _mapping(checkpoint.get("model_state_dict"), "source raw model state"),
        strict=False,
    )
    _validate_r14_training_state_load(
        incompatible.missing_keys, incompatible.unexpected_keys
    )
    ema = ExponentialMovingAverage(generator, decay=float(config["ema"]["decay"]))
    ema.load_state_dict(
        _mapping(checkpoint.get("ema_model_state_dict"), "source EMA model state")
    )
    codec = build_latent_codec_from_train_config(config, device)
    if codec is None:
        raise RuntimeError("R14 resume smoke requires the frozen latent codec")
    codec.vae.eval().requires_grad_(False)

    training_step = DistributedDataParallel(
        _SmokeTrainingStep(generator),
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in training_step.module.generator.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    optimizer.load_state_dict(
        dict(_mapping(checkpoint.get("optimizer_state_dict"), "source optimizer state"))
    )
    if _optimizer_steps(optimizer) != {2432}:
        raise RuntimeError("R14 resume smoke did not restore AdamW step 2432")

    batch = next(iter(loader))
    source_z = batch["source_z"].to(device=device, dtype=torch.float32)
    original = batch["image"].to(device=device, dtype=torch.float32)
    context = batch["context_image"].to(device=device, dtype=torch.float32)
    pixel_mask = batch["face_mask"].to(device=device, dtype=torch.bool)
    expanded = pixel_mask.expand_as(context)
    context_clean = bool(torch.equal(context[expanded], torch.zeros_like(context[expanded])))
    source_z_valid = bool(
        torch.isfinite(source_z).all().item()
        and torch.allclose(
            source_z.norm(dim=1),
            torch.ones(source_z.shape[0], device=device),
            atol=1e-5,
            rtol=0.0,
        )
    )
    target_latent, context_latent, latent_mask = encode_inpaint_training_latents(
        codec, original, context, pixel_mask
    )
    training_step.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = training_step(
        target_latent,
        source_z,
        context_latent,
        latent_mask,
        int(config["seed"]),
    )
    finite_loss = bool(torch.isfinite(loss).item()) and all(
        not (torch.is_tensor(value) and torch.is_floating_point(value))
        or bool(torch.isfinite(value).all().item())
        for value in metrics.values()
    )
    if not finite_loss:
        raise FloatingPointError("R14 batch4 smoke loss or metric is non-finite")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in training_step.module.generator.parameters()
        if parameter.grad is not None
    ]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    if not finite_gradients:
        raise FloatingPointError("R14 batch4 smoke gradients are missing or non-finite")
    if "grad_clip_norm" in config:
        torch.nn.utils.clip_grad_norm_(
            training_step.module.generator.parameters(),
            float(config["grad_clip_norm"]),
            error_if_nonfinite=True,
        )
    optimizer.step()
    ema.update(training_step.module.generator)
    if _optimizer_steps(optimizer) != {2433}:
        raise RuntimeError("R14 batch4 smoke AdamW state did not advance to step 2433")
    torch.cuda.synchronize(device)

    flags = torch.tensor(
        [
            int(context_clean),
            int(source_z_valid),
            int(finite_loss),
            int(finite_gradients),
            int(not any(parameter.requires_grad for parameter in codec.vae.parameters())),
        ],
        device=device,
        dtype=torch.int64,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    loss_total = loss.detach().to(dtype=torch.float64)
    dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
    local_memory = torch.tensor(
        [
            torch.cuda.max_memory_allocated(device) / MIB,
            torch.cuda.max_memory_reserved(device) / MIB,
            torch.cuda.memory_allocated(device) / MIB,
            torch.cuda.memory_reserved(device) / MIB,
        ],
        device=device,
        dtype=torch.float64,
    )
    gathered = [torch.zeros_like(local_memory) for _ in range(world_size)]
    dist.all_gather(gathered, local_memory)
    per_rank = [
        {
            "rank": index,
            "peak_allocated_mib": float(values[0].item()),
            "peak_reserved_mib": float(values[1].item()),
            "final_allocated_mib": float(values[2].item()),
            "final_reserved_mib": float(values[3].item()),
        }
        for index, values in enumerate(gathered)
    ]
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r14_resume_batch4_memory_smoke_v1",
        "source_checkpoint": str(checkpoint_path),
        "source_global_step": 2432,
        "smoke_optimizer_step": 2433,
        "formal_target_global_step": 2560,
        "world_size": 2,
        "per_device_batch_size": 4,
        "global_batch_size": 8,
        "sample_count": 8,
        "ddp_backward_and_adamw_step": True,
        "ema_loaded_and_updated": True,
        "source_face_pixels_enter_context_encoder": not bool(flags[0].item()),
        "source_z_finite_l2_normalized": bool(flags[1].item()),
        "loss_finite": bool(flags[2].item()),
        "gradients_finite": bool(flags[3].item()),
        "vae_frozen": bool(flags[4].item()),
        "loss_mean_across_ranks": float((loss_total / world_size).item()),
        "nccl_environment": NCCL_ENV,
        "per_rank_memory": per_rank,
        "max_peak_allocated_mib": max(row["peak_allocated_mib"] for row in per_rank),
        "max_peak_reserved_mib": max(row["peak_reserved_mib"] for row in per_rank),
    }
    required = (
        not summary["source_face_pixels_enter_context_encoder"]
        and summary["source_z_finite_l2_normalized"]
        and summary["loss_finite"]
        and summary["gradients_finite"]
        and summary["vae_frozen"]
    )
    if not required:
        raise RuntimeError(f"R14 batch4 smoke correctness gate failed: {summary}")
    if rank == 0:
        with (args.output_dir / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps(summary, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
