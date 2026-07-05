from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from safa.models.sit_diffusion import build_sit_diffusion_generator


CONSISTENCY_PREDICTION_TYPE_X0 = "x0"
CONSISTENCY_TARGET_ANALYTIC_X0 = "analytic_x0"


def build_latent_consistency_generator(config):
    schedule_config = replace(config, diffusion_train_timesteps=config.consistency_train_timesteps)
    base = build_sit_diffusion_generator(schedule_config)
    base.config = config

    def forward(z):
        return base.sample(z, steps=base.config.sample_steps)

    def flow_matching_loss(x_0, z, generator=None):
        base._validate_z(z)
        expected = (config.sit_input_channels, base.image_size, base.image_size)
        if x_0.ndim != 4 or tuple(x_0.shape[1:]) != expected:
            raise ValueError(f"x_0 must have shape [B,{expected[0]},{expected[1]},{expected[2]}], got {tuple(x_0.shape)}")
        x_data = base._data_to_model_space(x_0)
        min_gap = int(config.consistency_min_step_gap)
        timesteps_t = torch.randint(
            min_gap,
            int(config.consistency_train_timesteps),
            (x_data.shape[0],),
            device=x_data.device,
            generator=generator,
            dtype=torch.long,
        )
        max_s_exclusive = (timesteps_t - min_gap + 1).clamp_min(1)
        random_unit = torch.rand(x_data.shape[0], device=x_data.device, generator=generator)
        timesteps_s = torch.floor(random_unit * max_s_exclusive.to(dtype=random_unit.dtype)).to(dtype=torch.long)
        noise = torch.randn(x_data.shape, device=x_data.device, dtype=x_data.dtype, generator=generator)
        x_t = _q_sample(base, x_data, noise, timesteps_t)
        x_s = _q_sample(base, x_data, noise, timesteps_s)
        del x_s
        predicted_x0 = base.denoiser(x_t, base._normalize_timesteps(timesteps_t, dtype=x_t.dtype), z)
        if config.consistency_target != CONSISTENCY_TARGET_ANALYTIC_X0:
            raise RuntimeError(f"Unsupported consistency_target {config.consistency_target!r}")
        target_x0 = x_data.detach()
        loss = F.mse_loss(predicted_x0, target_x0)
        if base.null_condition is not None:
            loss = loss + 0.0 * base.null_condition.embedding.sum()
        return loss, {
            "flow_matching_mse": loss.detach(),
            "consistency_mse": loss.detach(),
            "consistency_prediction_type": config.consistency_prediction_type,
            "consistency_target": config.consistency_target,
            "consistency_timestep_t_mean": timesteps_t.detach().to(dtype=x_data.dtype).mean(),
            "consistency_timestep_s_mean": timesteps_s.detach().to(dtype=x_data.dtype).mean(),
            "predicted_x0_abs_mean": predicted_x0.detach().abs().mean(),
            "target_x0_abs_mean": target_x0.detach().abs().mean(),
            "latent_consistency_attention_backend": base.attention_backend,
            "latent_consistency_attention_backend_requested": base.requested_attention_backend,
        }

    def sample(z, steps: int | None = None, checkpoint_steps: bool = False, *, x_init=None, clamp_output: bool = True):
        del checkpoint_steps
        base._validate_z(z)
        steps = int(steps or config.sample_steps)
        if steps <= 0:
            raise ValueError(f"sample steps must be positive, got {steps}")
        if x_init is None:
            x = torch.randn(
                z.shape[0],
                config.sit_input_channels,
                base.image_size,
                base.image_size,
                device=z.device,
                dtype=z.dtype,
            )
        else:
            base._validate_x_init(x_init, z)
            x = x_init
        timesteps = consistency_timesteps(steps).to(device=z.device)
        for index, timestep in enumerate(timesteps):
            next_timestep = timesteps[index + 1] if index + 1 < len(timesteps) else None
            t_batch = torch.full((z.shape[0],), int(timestep.item()), device=z.device, dtype=torch.long)
            x = _consistency_step(base, x, z, t_batch, next_timestep)
        return base._model_to_data_space(x, clamp_output=clamp_output)

    def consistency_timesteps(steps: int):
        steps = int(steps)
        if steps <= 0:
            raise ValueError(f"sample steps must be positive, got {steps}")
        max_timestep = int(config.consistency_train_timesteps) - 1
        return torch.linspace(max_timestep, 0, steps, dtype=torch.float64).round().to(dtype=torch.long)

    def load_pretrained(checkpoint_path: str | Path | None, *, state_key: str | None = None, strict: bool = False, allow_missing: bool = False):
        return base._sit_diffusion_load_pretrained(checkpoint_path, state_key=state_key, strict=strict, allow_missing=allow_missing)

    def normalize_timesteps(timesteps, *, dtype):
        max_timestep = max(int(config.consistency_train_timesteps) - 1, 1)
        return timesteps.to(dtype=dtype) / float(max_timestep)

    base.forward = forward
    base.flow_matching_loss = flow_matching_loss
    base.sample = sample
    base.consistency_timesteps = consistency_timesteps
    base._sit_diffusion_load_pretrained = base.load_pretrained
    base.load_pretrained = load_pretrained
    base._normalize_timesteps = normalize_timesteps
    return base


def _q_sample(generator, x_data, noise, timesteps):
    sqrt_alpha_bar = generator._extract(generator.sqrt_alphas_cumprod, timesteps, x_data)
    sqrt_one_minus_alpha_bar = generator._extract(generator.sqrt_one_minus_alphas_cumprod, timesteps, x_data)
    return sqrt_alpha_bar * x_data + sqrt_one_minus_alpha_bar * noise


def _consistency_step(generator, x_t, z, timesteps, next_timestep):
    alpha_bar_t = generator._extract(generator.alphas_cumprod, timesteps, x_t)
    pred_x0 = generator.denoiser(x_t, generator._normalize_timesteps(timesteps, dtype=x_t.dtype), z)
    if next_timestep is None:
        return pred_x0
    eps = (x_t - alpha_bar_t.sqrt() * pred_x0) / (1.0 - alpha_bar_t).sqrt().clamp_min(1.0e-12)
    next_timesteps = torch.full_like(timesteps, int(next_timestep.item()))
    alpha_bar_prev = generator._extract(generator.alphas_cumprod, next_timesteps, x_t)
    return alpha_bar_prev.sqrt() * pred_x0 + (1.0 - alpha_bar_prev).sqrt() * eps
