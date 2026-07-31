#!/usr/bin/env python3
"""Generate the locked R14 regular32 candidate and matched-native pairs."""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset

from safa.data.r14_spatial import R14SpatialEvalDataset
from safa.models.generator import build_generator
from safa.models.meanflow_sit import (
    assemble_inpainted_pixels,
    encode_masked_context_latent,
)
from safa.training.latent_codec import build_latent_codec_from_train_config
from safa.training.transforms import r14_joint_transform
from safa.utils.sampling import make_x_init_for_sample_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _save_tensor_png(image: torch.Tensor, path: Path) -> None:
    if image.shape != (3, 256, 256) or not torch.isfinite(image).all().item():
        raise RuntimeError(f"invalid output image tensor for {path}: {tuple(image.shape)}")
    array = image.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(array, mode="RGB").save(path, format="PNG", compress_level=6)


def _mask_bbox(mask: torch.Tensor) -> list[int]:
    spatial = mask.detach().to(device="cpu", dtype=torch.bool).squeeze(0)
    positions = torch.nonzero(spatial, as_tuple=False)
    if positions.numel() == 0:
        raise RuntimeError("R14 eval mask is empty")
    y1, x1 = positions.min(dim=0).values.tolist()
    y2, x2 = (positions.max(dim=0).values + 1).tolist()
    if not (0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256):
        raise RuntimeError("R14 eval mask bbox is invalid")
    return [int(x1), int(y1), int(x2), int(y2)]


def _load_runtime(config: Mapping[str, object], checkpoint_path: Path, device: torch.device):
    checkpoint = _mapping(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True),
        "EMA export",
    )
    if checkpoint.get("contract_type") != "safa_r14_inpaint_ema_export_v1":
        raise RuntimeError("generation requires the versioned R14 EMA export")
    model_config = dict(_mapping(checkpoint.get("model_config"), "EMA model_config"))
    if model_config.get("model_type") != "meanflow_sit_inpaint":
        raise RuntimeError("EMA export is not meanflow_sit_inpaint")
    model_config["sit_pretrained_path"] = ""
    model_config["sit_pretrained_state_key"] = ""
    generator = build_generator(model_config)
    incompatible = generator.load_state_dict(
        _mapping(checkpoint.get("ema_model_state_dict"), "EMA state"), strict=False
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "trained R14 EMA topology differs: "
            f"missing={incompatible.missing_keys!r} unexpected={incompatible.unexpected_keys!r}"
        )
    parameters = dict(generator.named_parameters())
    for name in (
        "vector_field.context_embedder.weight",
        "vector_field.context_embedder.bias",
    ):
        value = parameters.get(name)
        if value is None or not torch.isfinite(value).all().item():
            raise RuntimeError(f"trained R14 EMA context parameter is missing or non-finite: {name}")
    generator = generator.to(device).eval().requires_grad_(False)
    codec = build_latent_codec_from_train_config(dict(config), device)
    if codec is None:
        raise RuntimeError("R14 generation requires the frozen latent codec")
    codec.vae.eval().requires_grad_(False)
    return generator, codec


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size != 4 or local_rank not in range(4) or rank not in range(4):
        raise RuntimeError("R14 regular32 generation requires exactly four torchrun ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    output_exists = torch.tensor([1 if args.output_dir.exists() else 0], device=device, dtype=torch.int64)
    dist.all_reduce(output_exists, op=dist.ReduceOp.MAX)
    if int(output_exists.item()) != 0:
        raise FileExistsError(f"refusing to reuse generation output: {args.output_dir}")
    if rank == 0:
        for name in ("source_images", "native_images", "generated_images"):
            (args.output_dir / name).mkdir(parents=True, exist_ok=False)
    dist.barrier()
    config = _mapping(yaml.safe_load(args.config.read_text(encoding="utf-8")), "config")
    dataset = R14SpatialEvalDataset(
        args.manifest,
        Path(str(config["eval_index"])),
        Path(str(config["eval_features"])),
        Path(str(config["e0_checkpoint"])),
        r14_joint_transform(256, horizontal_flip_probability=0.0),
    )
    if len(dataset) != 32:
        raise RuntimeError(f"regular32 must contain exactly 32 samples, got {len(dataset)}")
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(Subset(dataset, indices), batch_size=2, shuffle=False, num_workers=0)
    generator, codec = _load_runtime(config, args.checkpoint, device)
    rows: list[Mapping[str, object]] = []
    for batch in loader:
        sample_ids = [str(value) for value in batch["sample_id"]]
        source_z = batch["source_z"].to(device=device, dtype=torch.float32)
        original = batch["image"].to(device=device, dtype=torch.float32)
        context = batch["context_image"].to(device=device, dtype=torch.float32)
        pixel_mask = batch["face_mask"].to(device=device, dtype=torch.bool)
        if not torch.isfinite(source_z).all().item() or not torch.allclose(source_z.norm(dim=1), torch.ones(source_z.shape[0], device=device), atol=1e-5, rtol=0.0):
            raise RuntimeError("R14 source_z must be finite and L2 normalized")
        context_latent, latent_mask = encode_masked_context_latent(codec, context, pixel_mask)
        x_init = make_x_init_for_sample_ids(
            sample_ids,
            int(config["sampling_seed"]),
            int(config["image_size"]),
            device,
            source_z.dtype,
            channels=4,
        )
        null_z = generator.make_null_condition(batch_size=len(sample_ids), device=device, dtype=source_z.dtype)
        with torch.no_grad():
            candidate_latent = generator.sample(
                source_z,
                x_init=x_init,
                context_latent=context_latent,
                latent_mask=latent_mask,
            )
            native_latent = generator.sample(
                null_z,
                x_init=x_init,
                context_latent=context_latent,
                latent_mask=latent_mask,
            )
            candidate_decoded = codec.decode(candidate_latent)
            native_decoded = codec.decode(native_latent)
            candidate = assemble_inpainted_pixels(original, candidate_decoded, pixel_mask)
            native = assemble_inpainted_pixels(original, native_decoded, pixel_mask)
        outside = ~pixel_mask.expand_as(original)
        candidate_exact = torch.equal(candidate[outside], original[outside])
        native_exact = torch.equal(native[outside], original[outside])
        if not candidate_exact or not native_exact:
            raise RuntimeError("R14 outside-mask pixel equality failed")
        row_cursor = len(rows)
        for offset, sample_id in enumerate(sample_ids):
            safe_name = f"{indices[row_cursor + offset]:04d}.png"
            source_path = (args.output_dir / "source_images" / safe_name).resolve()
            native_path = (args.output_dir / "native_images" / safe_name).resolve()
            generated_path = (args.output_dir / "generated_images" / safe_name).resolve()
            _save_tensor_png(original[offset], source_path)
            _save_tensor_png(native[offset], native_path)
            _save_tensor_png(candidate[offset], generated_path)
            rows.append(
                {
                    "sample_id": sample_id,
                    "source": str(source_path),
                    "native": str(native_path),
                    "generated": str(generated_path),
                    "bbox_xyxy_256": _mask_bbox(pixel_mask[offset]),
                    "outside_mask_candidate_bit_exact": True,
                    "outside_mask_native_bit_exact": True,
                    "sampling_seed": int(config["sampling_seed"]),
                }
            )
    _write_jsonl(args.output_dir / f"per_sample.rank{rank}.jsonl", rows)
    dist.barrier()
    if rank == 0:
        all_rows: list[Mapping[str, object]] = []
        for other_rank in range(world_size):
            path = args.output_dir / f"per_sample.rank{other_rank}.jsonl"
            all_rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        order = {
            json.loads(line)["sample_id"]: index
            for index, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines())
        }
        if len(all_rows) != 32 or set(order) != {row["sample_id"] for row in all_rows}:
            raise RuntimeError("distributed generation rows do not match regular32")
        all_rows.sort(key=lambda row: order[str(row["sample_id"])])
        _write_jsonl(args.output_dir / "per_sample.jsonl", all_rows)
        _write_json(
            args.output_dir / "completion.json",
            {
                "schema_version": 1,
                "contract_type": "safa_r14_inpaint_generation_v1",
                "sample_count": 32,
                "candidate_count": 32,
                "native_count": 32,
                "outside_mask_bit_exact_all": True,
                "world_size": 4,
                "batch_size_per_rank": 2,
                "matched_native_condition": "learned_null_condition",
                "candidate_native_share_x_init_context_mask": True,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
