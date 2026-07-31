#!/usr/bin/env python3
"""Eight-sample correctness gate before the only R14 training arm."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

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


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse smoke output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = _mapping(yaml.safe_load(args.config.read_text(encoding="utf-8")), "config")
    dataset = R14SpatialEvalDataset(
        args.manifest,
        Path(str(config["eval_index"])),
        Path(str(config["eval_features"])),
        Path(str(config["e0_checkpoint"])),
        r14_joint_transform(256, horizontal_flip_probability=0.0),
    )
    if len(dataset) != 8:
        raise RuntimeError(f"smoke8 must contain exactly 8 samples, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    device = torch.device("cuda", 0)
    generator, codec = _load_e15_inpaint(config, device)
    all_context_clean = True
    all_outside_exact = True
    all_deterministic = True
    all_source_z_finite = True
    loss_value: float | None = None
    finite_gradients = False
    for batch_index, batch in enumerate(loader):
        sample_ids = [str(value) for value in batch["sample_id"]]
        source_z = batch["source_z"].to(device=device, dtype=torch.float32)
        original = batch["image"].to(device=device, dtype=torch.float32)
        context = batch["context_image"].to(device=device, dtype=torch.float32)
        pixel_mask = batch["face_mask"].to(device=device, dtype=torch.bool)
        all_source_z_finite &= bool(
            torch.isfinite(source_z).all().item()
            and torch.allclose(source_z.norm(dim=1), torch.ones(source_z.shape[0], device=device), atol=1e-5, rtol=0.0)
        )
        expanded = pixel_mask.expand_as(context)
        all_context_clean &= bool(torch.equal(context[expanded], torch.zeros_like(context[expanded])))
        context_latent, latent_mask = encode_masked_context_latent(codec, context, pixel_mask)
        x_a = make_x_init_for_sample_ids(sample_ids, int(config["sampling_seed"]), 32, device, source_z.dtype, channels=4)
        x_b = make_x_init_for_sample_ids(sample_ids, int(config["sampling_seed"]), 32, device, source_z.dtype, channels=4)
        all_deterministic &= bool(torch.equal(x_a, x_b))
        generator.eval()
        with torch.no_grad():
            latent_a = generator.sample(source_z, x_init=x_a, context_latent=context_latent, latent_mask=latent_mask)
            latent_b = generator.sample(source_z, x_init=x_b, context_latent=context_latent, latent_mask=latent_mask)
            all_deterministic &= bool(torch.equal(latent_a, latent_b))
            decoded = codec.decode(latent_a)
            assembled = assemble_inpainted_pixels(original, decoded, pixel_mask)
        all_outside_exact &= bool(torch.equal(assembled[~expanded], original[~expanded]))
        if batch_index == 0:
            generator.train()
            generator.zero_grad(set_to_none=True)
            target_latent, train_context_latent, train_latent_mask = encode_inpaint_training_latents(
                codec, original, context, pixel_mask
            )
            rng = torch.Generator(device=device).manual_seed(int(config["seed"]))
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if bool(config["amp"])
                else nullcontext()
            )
            with amp_context:
                loss, metrics = generator.flow_matching_loss(
                    target_latent,
                    source_z,
                    generator=rng,
                    context_latent=train_context_latent,
                    latent_mask=train_latent_mask,
                )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("R14 smoke masked loss is non-finite")
            for value in metrics.values():
                if torch.is_tensor(value) and torch.is_floating_point(value) and not torch.isfinite(value).all().item():
                    raise FloatingPointError("R14 smoke loss metric is non-finite")
            loss.backward()
            gradients = [parameter.grad for parameter in generator.parameters() if parameter.grad is not None]
            finite_gradients = bool(gradients) and all(torch.isfinite(gradient).all().item() for gradient in gradients)
            loss_value = float(loss.detach().item())
            generator.zero_grad(set_to_none=True)
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r14_inpaint_smoke8_v1",
        "sample_count": 8,
        "source_face_pixels_enter_context_encoder": not all_context_clean,
        "outside_mask_bit_exact": all_outside_exact,
        "same_seed_noise_deterministic": all_deterministic,
        "source_z_finite_l2_normalized": all_source_z_finite,
        "masked_loss_finite": loss_value is not None and torch.isfinite(torch.tensor(loss_value)).item(),
        "masked_gradients_finite": finite_gradients,
        "vae_frozen": not any(parameter.requires_grad for parameter in codec.vae.parameters()),
        "masked_loss": loss_value,
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
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
