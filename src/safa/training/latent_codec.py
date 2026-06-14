from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from safa.models.generator import GENERATOR_MODEL_TYPE_MEANFLOW_SIT, FlowGeneratorConfig


@dataclass(frozen=True)
class LatentCodecConfig:
    source: str
    scaling_factor: float


class LatentCodec:
    def __init__(self, vae, config: LatentCodecConfig):
        self.vae = vae
        self.config = config
        self.scaling_factor = float(config.scaling_factor)
        self.vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self._validate_images(images)
        posterior = self.vae.encode(images.mul(2.0).sub(1.0))
        latent_dist = getattr(posterior, "latent_dist", None)
        if latent_dist is None and isinstance(posterior, dict):
            latent_dist = posterior.get("latent_dist")
        if latent_dist is None or not hasattr(latent_dist, "sample"):
            raise RuntimeError("VAE encode output must expose latent_dist.sample()")
        latents = latent_dist.sample()
        return latents * self.scaling_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self._validate_latents(latents)
        decoded = self.vae.decode(latents / self.scaling_factor)
        sample = getattr(decoded, "sample", None)
        if sample is None and isinstance(decoded, dict):
            sample = decoded.get("sample")
        if not isinstance(sample, torch.Tensor):
            raise RuntimeError("VAE decode output must expose a tensor sample")
        return sample.add(1.0).mul(0.5).clamp(0.0, 1.0)

    @staticmethod
    def _validate_images(images: torch.Tensor) -> None:
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {type(images).__name__}")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape [B,3,H,W] for VAE encode, got {tuple(images.shape)}")
        if not torch.is_floating_point(images):
            raise TypeError(f"images must be floating point for VAE encode, got {images.dtype}")

    @staticmethod
    def _validate_latents(latents: torch.Tensor) -> None:
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"latents must be a torch.Tensor, got {type(latents).__name__}")
        if latents.ndim != 4 or latents.shape[1] != 4:
            raise ValueError(f"latents must have shape [B,4,H,W] for VAE decode, got {tuple(latents.shape)}")
        if not torch.is_floating_point(latents):
            raise TypeError(f"latents must be floating point for VAE decode, got {latents.dtype}")


def latent_training_enabled(config: dict[str, Any]) -> bool:
    value = config.get("latent_training", False)
    if not isinstance(value, bool):
        raise ValueError(f"train_g config.latent_training must be true or false, got {value!r}")
    return bool(value)


def latent_codec_config_from_train_config(config: dict[str, Any]) -> LatentCodecConfig:
    source = _latent_vae_source(config)
    scaling_factor = _latent_scaling_factor(config)
    return LatentCodecConfig(source=source, scaling_factor=scaling_factor)


def validate_latent_training_config(config: dict[str, Any], generator_config: FlowGeneratorConfig) -> None:
    if not latent_training_enabled(config):
        return
    if generator_config.model_type != GENERATOR_MODEL_TYPE_MEANFLOW_SIT:
        raise ValueError("latent_training requires generator.model_type == 'meanflow_sit'")
    if generator_config.sit_input_channels != 4:
        raise ValueError("latent_training requires generator.sit_input_channels == 4")
    if generator_config.sit_data_space != "latent":
        raise ValueError("latent_training requires generator.sit_data_space == 'latent'")
    image_size = _positive_int(config, "image_size", "train_g config")
    pixel_image_size = _positive_int(config, "pixel_image_size", "train_g config")
    if pixel_image_size != image_size * 8:
        raise ValueError(
            "latent_training requires pixel_image_size == image_size * 8 for SD VAE latents, "
            f"got pixel_image_size={pixel_image_size} image_size={image_size}"
        )
    _latent_vae_source(config)
    _latent_scaling_factor(config)


def build_latent_codec_from_train_config(config: dict[str, Any], device) -> LatentCodec | None:
    if not latent_training_enabled(config):
        return None
    codec_config = latent_codec_config_from_train_config(config)
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise RuntimeError("latent_training requires diffusers with AutoencoderKL; install diffusers before starting this run") from exc
    try:
        vae = AutoencoderKL.from_pretrained(codec_config.source)
    except Exception as exc:
        raise RuntimeError(f"failed to load VAE for latent_training from {codec_config.source!r}") from exc
    vae = vae.to(device)
    return LatentCodec(vae, codec_config)


def _latent_vae_source(config: dict[str, Any]) -> str:
    vae_path = str(config.get("vae_path", "") or "")
    vae_model = str(config.get("vae_model", "") or "")
    if vae_path:
        return str(Path(vae_path))
    if vae_model:
        return vae_model
    raise ValueError("latent_training requires vae_model or vae_path")


def _latent_scaling_factor(config: dict[str, Any]) -> float:
    if "vae_scaling_factor" not in config:
        raise ValueError("latent_training requires vae_scaling_factor")
    value = config["vae_scaling_factor"]
    if isinstance(value, bool):
        raise ValueError(f"vae_scaling_factor must be positive, got {value!r}")
    scaling_factor = float(value)
    if not scaling_factor > 0.0:
        raise ValueError(f"vae_scaling_factor must be positive, got {value!r}")
    return scaling_factor


def _positive_int(config: dict[str, Any], field: str, context: str) -> int:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context}.{field} must be a positive integer, got {value!r}")
    return int(value)
