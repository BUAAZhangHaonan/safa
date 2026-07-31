#!/usr/bin/env python3
"""Eight-sample correctness gate before the only R14 training arm."""
from __future__ import annotations

import argparse
import json
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
from safa.models.meanflow_sit import (
    assemble_inpainted_pixels,
    encode_inpaint_training_latents,
    encode_masked_context_latent,
)
from safa.training.latent_codec import build_latent_codec_from_train_config
from safa.training.transforms import r14_joint_transform
from safa.utils.sampling import make_x_init_for_sample_ids


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


def _load_e15_inpaint(config: Mapping[str, object], device: torch.device):
    checkpoint = _mapping(
        torch.load(Path(str(config["resume_from"])), map_location="cpu", weights_only=True, mmap=True),
        "E15 checkpoint",
    )
    model_config = dict(_mapping(checkpoint.get("model_config"), "E15 model_config"))
    model_config["model_type"] = "meanflow_sit_inpaint"
    model_config["sit_pretrained_path"] = ""
    model_config["sit_pretrained_state_key"] = ""
    generator = build_generator(model_config)
    _load_legacy_e15_into_inpaint(
        generator, _mapping(checkpoint.get("ema_model_state_dict"), "E15 EMA state")
    )
    generator = generator.to(device).train()
    codec = build_latent_codec_from_train_config(dict(config), device)
    if codec is None:
        raise RuntimeError("R14 smoke requires the frozen latent codec")
    codec.vae.eval().requires_grad_(False)
    return generator, codec


def _load_legacy_e15_into_inpaint(generator, state: Mapping[str, object]) -> None:
    """Allow only the new zero-init context projection absent from legacy E15."""
    incompatible = generator.load_state_dict(state, strict=False)
    expected_missing = {
        "vector_field.context_embedder.weight",
        "vector_field.context_embedder.bias",
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "legacy E15/inpaint state topology differs beyond the registered context projection: "
            f"missing={incompatible.missing_keys!r} unexpected={incompatible.unexpected_keys!r}"
        )
    parameters = dict(generator.named_parameters())
    for name in expected_missing:
        value = parameters.get(name)
        if value is None or not torch.equal(value.detach(), torch.zeros_like(value.detach())):
            raise RuntimeError(f"new R14 parameter must remain exact zero-init after E15 load: {name}")


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
    if world_size != 4 or local_rank not in range(4) or rank not in range(4):
        raise RuntimeError("R14 smoke8 requires exactly four torchrun ranks")
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
    config = _mapping(yaml.safe_load(args.config.read_text(encoding="utf-8")), "config")
    if config.get("global_batch_size") != 8 or config.get("per_device_batch_size") != 2:
        raise RuntimeError("R14 smoke8 requires global batch 8 and per-rank batch 2")
    dataset = R14SpatialEvalDataset(
        args.manifest,
        Path(str(config["eval_index"])),
        Path(str(config["eval_features"])),
        Path(str(config["e0_checkpoint"])),
        r14_joint_transform(256, horizontal_flip_probability=0.0),
    )
    if len(dataset) != 8:
        raise RuntimeError(f"smoke8 must contain exactly 8 samples, got {len(dataset)}")
    indices = list(range(rank * 2, (rank + 1) * 2))
    loader = DataLoader(Subset(dataset, indices), batch_size=2, shuffle=False, num_workers=0)
    if len(loader) != 1:
        raise RuntimeError("each R14 smoke rank must receive exactly one full batch")
    generator, codec = _load_e15_inpaint(config, device)
    training_step = DistributedDataParallel(
        _SmokeTrainingStep(generator),
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    batch = next(iter(loader))
    sample_ids = [str(value) for value in batch["sample_id"]]
    source_z = batch["source_z"].to(device=device, dtype=torch.float32)
    original = batch["image"].to(device=device, dtype=torch.float32)
    context = batch["context_image"].to(device=device, dtype=torch.float32)
    pixel_mask = batch["face_mask"].to(device=device, dtype=torch.bool)
    source_z_finite = bool(
        torch.isfinite(source_z).all().item()
        and torch.allclose(
            source_z.norm(dim=1),
            torch.ones(source_z.shape[0], device=device),
            atol=1e-5,
            rtol=0.0,
        )
    )
    expanded = pixel_mask.expand_as(context)
    context_clean = bool(torch.equal(context[expanded], torch.zeros_like(context[expanded])))
    context_latent, latent_mask = encode_masked_context_latent(codec, context, pixel_mask)
    x_a = make_x_init_for_sample_ids(
        sample_ids, int(config["sampling_seed"]), 32, device, source_z.dtype, channels=4
    )
    x_b = make_x_init_for_sample_ids(
        sample_ids, int(config["sampling_seed"]), 32, device, source_z.dtype, channels=4
    )
    deterministic = bool(torch.equal(x_a, x_b))
    runtime_generator = training_step.module.generator
    runtime_generator.eval()
    with torch.no_grad():
        latent_a = runtime_generator.sample(
            source_z, x_init=x_a, context_latent=context_latent, latent_mask=latent_mask
        )
        latent_b = runtime_generator.sample(
            source_z, x_init=x_b, context_latent=context_latent, latent_mask=latent_mask
        )
        deterministic &= bool(torch.equal(latent_a, latent_b))
        decoded = codec.decode(latent_a)
        assembled = assemble_inpainted_pixels(original, decoded, pixel_mask)
    outside_exact = bool(torch.equal(assembled[~expanded], original[~expanded]))
    training_step.train()
    training_step.zero_grad(set_to_none=True)
    target_latent, train_context_latent, train_latent_mask = encode_inpaint_training_latents(
        codec, original, context, pixel_mask
    )
    loss, metrics = training_step(
        target_latent,
        source_z,
        train_context_latent,
        train_latent_mask,
        int(config["seed"]),
    )
    loss_finite = bool(torch.isfinite(loss).item())
    metrics_finite = all(
        not (torch.is_tensor(value) and torch.is_floating_point(value))
        or bool(torch.isfinite(value).all().item())
        for value in metrics.values()
    )
    if not loss_finite or not metrics_finite:
        raise FloatingPointError("R14 smoke masked loss or metric is non-finite")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in runtime_generator.parameters()
        if parameter.grad is not None
    ]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    vae_frozen = not any(parameter.requires_grad for parameter in codec.vae.parameters())
    flags = torch.tensor(
        [
            int(context_clean),
            int(outside_exact),
            int(deterministic),
            int(source_z_finite),
            int(loss_finite and metrics_finite),
            int(finite_gradients),
            int(vae_frozen),
        ],
        device=device,
        dtype=torch.int64,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    loss_total = loss.detach().to(dtype=torch.float64)
    dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
    training_step.zero_grad(set_to_none=True)
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r14_inpaint_smoke8_v1",
        "sample_count": 8,
        "world_size": world_size,
        "batch_size_per_rank": 2,
        "ddp_backward": True,
        "source_face_pixels_enter_context_encoder": not bool(flags[0].item()),
        "outside_mask_bit_exact": bool(flags[1].item()),
        "same_seed_noise_deterministic": bool(flags[2].item()),
        "source_z_finite_l2_normalized": bool(flags[3].item()),
        "masked_loss_finite": bool(flags[4].item()),
        "masked_gradients_finite": bool(flags[5].item()),
        "vae_frozen": bool(flags[6].item()),
        "masked_loss_mean_across_ranks": float((loss_total / world_size).item()),
    }
    required = (
        not summary["source_face_pixels_enter_context_encoder"]
        and summary["outside_mask_bit_exact"]
        and summary["same_seed_noise_deterministic"]
        and summary["source_z_finite_l2_normalized"]
        and summary["masked_loss_finite"]
        and summary["masked_gradients_finite"]
        and summary["vae_frozen"]
    )
    if not required:
        raise RuntimeError(f"R14 smoke8 correctness gate failed: {summary}")
    if rank == 0:
        with (args.output_dir / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps(summary, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
