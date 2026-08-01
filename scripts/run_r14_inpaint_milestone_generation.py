#!/usr/bin/env python3
"""Generate one R14 EMA milestone on exactly two distributed ranks."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Subset

from safa.data.r14_spatial import R14SpatialEvalDataset
from safa.models.meanflow_sit import assemble_inpainted_pixels, encode_masked_context_latent
from safa.training.transforms import r14_joint_transform
from safa.utils.sampling import make_x_init_for_sample_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_r14_inpaint_generation as generation_core  # noqa: E402


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size != 2 or local_rank not in (0, 1) or rank not in (0, 1):
        raise RuntimeError("R14 milestone generation requires exactly two torchrun ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    output_exists = torch.tensor([int(args.output_dir.exists())], device=device, dtype=torch.int64)
    dist.all_reduce(output_exists, op=dist.ReduceOp.MAX)
    if int(output_exists.item()) != 0:
        raise FileExistsError(f"refusing to reuse generation output: {args.output_dir}")
    if rank == 0:
        for name in ("source_images", "native_images", "generated_images"):
            (args.output_dir / name).mkdir(parents=True, exist_ok=False)
    dist.barrier()
    config = _mapping(yaml.safe_load(args.config.read_text(encoding="utf-8")), "config")
    if config.get("sampling_seed") != 1337:
        raise RuntimeError("milestone generation requires sampling_seed=1337")
    dataset = R14SpatialEvalDataset(
        args.manifest,
        Path(str(config["eval_index"])),
        Path(str(config["eval_features"])),
        Path(str(config["e0_checkpoint"])),
        r14_joint_transform(256, horizontal_flip_probability=0.0),
    )
    if len(dataset) != 32:
        raise RuntimeError(f"milestone generation requires exactly 32 samples, got {len(dataset)}")
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(Subset(dataset, indices), batch_size=2, shuffle=False, num_workers=0)
    generator, codec = generation_core._load_runtime(config, args.checkpoint, device)
    rows: list[Mapping[str, object]] = []
    for batch in loader:
        sample_ids = [str(value) for value in batch["sample_id"]]
        source_z = batch["source_z"].to(device=device, dtype=torch.float32)
        original = batch["image"].to(device=device, dtype=torch.float32)
        context = batch["context_image"].to(device=device, dtype=torch.float32)
        pixel_mask = batch["face_mask"].to(device=device, dtype=torch.bool)
        if not torch.isfinite(source_z).all().item() or not torch.allclose(
            source_z.norm(dim=1), torch.ones(source_z.shape[0], device=device), atol=1e-5, rtol=0.0
        ):
            raise RuntimeError("R14 source_z must be finite and L2 normalized")
        context_latent, latent_mask = encode_masked_context_latent(codec, context, pixel_mask)
        x_init = make_x_init_for_sample_ids(
            sample_ids, 1337, int(config["image_size"]), device, source_z.dtype, channels=4
        )
        null_z = generator.make_null_condition(batch_size=len(sample_ids), device=device, dtype=source_z.dtype)
        with torch.no_grad():
            candidate_latent = generator.sample(
                source_z, x_init=x_init, context_latent=context_latent, latent_mask=latent_mask
            )
            native_latent = generator.sample(
                null_z, x_init=x_init, context_latent=context_latent, latent_mask=latent_mask
            )
            candidate_decoded = codec.decode(candidate_latent)
            native_decoded = codec.decode(native_latent)
            candidate = assemble_inpainted_pixels(original, candidate_decoded, pixel_mask)
            native = assemble_inpainted_pixels(original, native_decoded, pixel_mask)
        outside = ~pixel_mask.expand_as(original)
        if not torch.equal(candidate[outside], original[outside]) or not torch.equal(native[outside], original[outside]):
            raise RuntimeError("R14 outside-mask pixel equality failed")
        row_cursor = len(rows)
        for offset, sample_id in enumerate(sample_ids):
            global_index = indices[row_cursor + offset]
            name = f"{global_index:04d}.png"
            source_path = (args.output_dir / "source_images" / name).resolve()
            native_path = (args.output_dir / "native_images" / name).resolve()
            generated_path = (args.output_dir / "generated_images" / name).resolve()
            generation_core._save_tensor_png(original[offset], source_path)
            generation_core._save_tensor_png(native[offset], native_path)
            generation_core._save_tensor_png(candidate[offset], generated_path)
            rows.append(
                {
                    "sample_id": sample_id,
                    "source": str(source_path),
                    "native": str(native_path),
                    "generated": str(generated_path),
                    "bbox_xyxy_256": generation_core._mask_bbox(pixel_mask[offset]),
                    "outside_mask_candidate_bit_exact": True,
                    "outside_mask_native_bit_exact": True,
                    "sampling_seed": 1337,
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
            raise RuntimeError("distributed generation rows do not match manifest")
        all_rows.sort(key=lambda row: order[str(row["sample_id"])])
        _write_jsonl(args.output_dir / "per_sample.jsonl", all_rows)
        export_payload = _mapping(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True), "EMA export"
        )
        metrics = _mapping(export_payload.get("metrics"), "EMA metrics")
        _write_json(
            args.output_dir / "completion.json",
            {
                "schema_version": 1,
                "contract_type": "safa_r14_inpaint_milestone_generation_v1",
                "sample_count": 32,
                "candidate_count": 32,
                "native_count": 32,
                "global_step": metrics.get("global_step"),
                "outside_mask_bit_exact_all": True,
                "world_size": 2,
                "batch_size_per_rank": 2,
                "matched_native_condition": "learned_null_condition",
                "candidate_native_share_x_init_context_mask": True,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
