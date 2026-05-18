from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FlowGeneratorConfig:
    embedding_dim: int = 512
    image_size: int = 224
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    time_embedding_dim: int = 128
    condition_dim: int = 512
    sample_steps: int = 32
    train_cycle_steps: int = 8
    sampler: str = "heun"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FlowGeneratorConfig":
        return cls(
            embedding_dim=int(payload.get("embedding_dim", 512)),
            image_size=int(payload.get("image_size", 224)),
            base_channels=int(payload.get("base_channels", 32)),
            channel_multipliers=tuple(int(item) for item in payload.get("channel_multipliers", (1, 2, 4, 4))),
            time_embedding_dim=int(payload.get("time_embedding_dim", 128)),
            condition_dim=int(payload.get("condition_dim", 512)),
            sample_steps=int(payload.get("sample_steps", 32)),
            train_cycle_steps=int(payload.get("train_cycle_steps", 8)),
            sampler=str(payload.get("sampler", "heun")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "conditional_flow_matching",
            "embedding_dim": self.embedding_dim,
            "image_size": self.image_size,
            "base_channels": self.base_channels,
            "channel_multipliers": list(self.channel_multipliers),
            "time_embedding_dim": self.time_embedding_dim,
            "condition_dim": self.condition_dim,
            "sample_steps": self.sample_steps,
            "train_cycle_steps": self.train_cycle_steps,
            "sampler": self.sampler,
        }


class ConditionalFlowGenerator:
    def __new__(cls, config: FlowGeneratorConfig | dict[str, Any] | None = None, **kwargs):
        import math
        import torch
        from torch import nn
        import torch.nn.functional as F

        cfg_payload = {}
        if isinstance(config, FlowGeneratorConfig):
            cfg = config
        else:
            if config is not None:
                cfg_payload.update(config)
            cfg_payload.update(kwargs)
            cfg = FlowGeneratorConfig.from_dict(cfg_payload)
        _validate_config(cfg)

        def sinusoidal_embedding(timesteps, dim: int):
            if timesteps.ndim != 1:
                raise ValueError(f"t must have shape [B], got {tuple(timesteps.shape)}")
            half = dim // 2
            frequencies = torch.exp(
                torch.arange(half, device=timesteps.device, dtype=timesteps.dtype)
                * (-math.log(10000.0) / max(half - 1, 1))
            )
            args = timesteps[:, None] * frequencies[None, :]
            embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
            if dim % 2 == 1:
                embedding = F.pad(embedding, (0, 1))
            return embedding

        class FiLMResidualBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
                super().__init__()
                groups_in = _groups_for(in_channels)
                groups_out = _groups_for(out_channels)
                self.in_norm = nn.GroupNorm(groups_in, in_channels)
                self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                self.out_norm = nn.GroupNorm(groups_out, out_channels)
                self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                self.condition = nn.Linear(condition_dim, out_channels * 2)
                self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

            def forward(self, x, condition):
                hidden = self.in_conv(F.silu(self.in_norm(x)))
                scale_shift = self.condition(condition).view(condition.shape[0], -1, 1, 1)
                scale, shift = scale_shift.chunk(2, dim=1)
                hidden = self.out_norm(hidden)
                hidden = hidden * (1.0 + scale) + shift
                hidden = self.out_conv(F.silu(hidden))
                return hidden + self.skip(x)

        class VectorFieldUNet(nn.Module):
            def __init__(self):
                super().__init__()
                channels = [cfg.base_channels * item for item in cfg.channel_multipliers]
                self.input = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)
                self.time_mlp = nn.Sequential(
                    nn.Linear(cfg.time_embedding_dim, cfg.condition_dim),
                    nn.SiLU(),
                    nn.Linear(cfg.condition_dim, cfg.condition_dim),
                )
                self.z_mlp = nn.Sequential(
                    nn.Linear(cfg.embedding_dim, cfg.condition_dim),
                    nn.SiLU(),
                    nn.Linear(cfg.condition_dim, cfg.condition_dim),
                )
                self.down_blocks = nn.ModuleList()
                self.downsamplers = nn.ModuleList()
                current = channels[0]
                for next_channels in channels:
                    self.down_blocks.append(FiLMResidualBlock(current, next_channels, cfg.condition_dim))
                    self.downsamplers.append(nn.Conv2d(next_channels, next_channels, kernel_size=4, stride=2, padding=1))
                    current = next_channels
                self.mid = FiLMResidualBlock(current, current, cfg.condition_dim)
                self.up_blocks = nn.ModuleList()
                self.upsamplers = nn.ModuleList()
                for skip_channels in reversed(channels):
                    self.upsamplers.append(nn.ConvTranspose2d(current, skip_channels, kernel_size=4, stride=2, padding=1))
                    self.up_blocks.append(FiLMResidualBlock(skip_channels + skip_channels, skip_channels, cfg.condition_dim))
                    current = skip_channels
                self.output = nn.Sequential(
                    nn.GroupNorm(_groups_for(current), current),
                    nn.SiLU(),
                    nn.Conv2d(current, 3, kernel_size=3, padding=1),
                )

            def forward(self, x_t, t, z):
                if x_t.ndim != 4 or x_t.shape[1:] != (3, cfg.image_size, cfg.image_size):
                    raise ValueError(f"x_t must have shape [B,3,{cfg.image_size},{cfg.image_size}], got {tuple(x_t.shape)}")
                if z.ndim != 2 or z.shape[1] != cfg.embedding_dim:
                    raise ValueError(f"z must have shape [B,{cfg.embedding_dim}], got {tuple(z.shape)}")
                if t.ndim != 1 or t.shape[0] != z.shape[0]:
                    raise ValueError(f"t must have shape [B], got {tuple(t.shape)} for batch {z.shape[0]}")
                condition = self.time_mlp(sinusoidal_embedding(t, cfg.time_embedding_dim)) + self.z_mlp(z)
                hidden = self.input(x_t)
                skips = []
                for block, downsample in zip(self.down_blocks, self.downsamplers):
                    hidden = block(hidden, condition)
                    skips.append(hidden)
                    hidden = downsample(hidden)
                hidden = self.mid(hidden, condition)
                for upsample, block, skip in zip(self.upsamplers, self.up_blocks, reversed(skips)):
                    hidden = upsample(hidden)
                    if hidden.shape[-2:] != skip.shape[-2:]:
                        raise RuntimeError(f"U-Net skip shape mismatch: up={tuple(hidden.shape)} skip={tuple(skip.shape)}")
                    hidden = torch.cat([hidden, skip], dim=1)
                    hidden = block(hidden, condition)
                return self.output(hidden)

        class _ConditionalFlowGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = cfg
                self.embedding_dim = cfg.embedding_dim
                self.image_size = cfg.image_size
                self.vector_field = VectorFieldUNet()

            def forward(self, z):
                return self.sample(z, steps=self.config.sample_steps)

            def _single_step(self, x, z, step_index, total_steps):
                dt = 1.0 / float(total_steps)
                t = torch.full((z.shape[0],), step_index / float(total_steps), device=z.device, dtype=z.dtype)
                velocity = self.vector_field(x, t, z)
                if self.config.sampler == "euler":
                    return x + dt * velocity
                elif self.config.sampler == "heun":
                    proposal = x + dt * velocity
                    next_t = torch.full((z.shape[0],), (step_index + 1) / float(total_steps), device=z.device, dtype=z.dtype)
                    next_velocity = self.vector_field(proposal, next_t, z)
                    return x + 0.5 * dt * (velocity + next_velocity)
                else:
                    raise ValueError(f"Unsupported sampler: {self.config.sampler}")

            def sample(self, z, steps: int | None = None, checkpoint_steps: bool = False):
                self._validate_z(z)
                steps = int(steps or self.config.sample_steps)
                if steps <= 0:
                    raise ValueError(f"sample steps must be positive, got {steps}")
                x = torch.randn(z.shape[0], 3, self.image_size, self.image_size, device=z.device, dtype=z.dtype)
                for index in range(steps):
                    if checkpoint_steps:
                        x = torch.utils.checkpoint.checkpoint(
                            self._single_step, x, z, index, steps,
                            use_reentrant=False,
                        )
                    else:
                        x = self._single_step(x, z, index, steps)
                max_abs = x.abs().max().item()
                if max_abs > 5.0:
                    print(f"WARNING: ODE solver divergence detected, max_abs={max_abs:.2f}, step={index}/{steps}")
                return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)

            def flow_matching_loss(self, x_1, z, generator=None):
                self._validate_z(z)
                if x_1.ndim != 4 or x_1.shape[1:] != (3, self.image_size, self.image_size):
                    raise ValueError(f"x_1 must have shape [B,3,{self.image_size},{self.image_size}], got {tuple(x_1.shape)}")
                x_1_flow = x_1.mul(2.0).sub(1.0)
                x_0 = torch.randn(x_1_flow.shape, device=x_1_flow.device, dtype=x_1_flow.dtype, generator=generator)
                t = torch.rand(x_1_flow.shape[0], device=x_1_flow.device, dtype=x_1_flow.dtype, generator=generator)
                view_t = t.view(-1, 1, 1, 1)
                x_t = (1.0 - view_t) * x_0 + view_t * x_1_flow
                target_velocity = x_1_flow - x_0
                predicted_velocity = self.vector_field(x_t, t, z)
                loss = F.mse_loss(predicted_velocity, target_velocity)
                return loss, {
                    "flow_matching_mse": loss.detach(),
                    "target_velocity_abs_mean": target_velocity.detach().abs().mean(),
                    "predicted_velocity_abs_mean": predicted_velocity.detach().abs().mean(),
                }

            def _validate_z(self, z):
                if z.ndim != 2 or z.shape[1] != self.embedding_dim:
                    raise ValueError(f"G expects z with shape [B,{self.embedding_dim}], got {tuple(z.shape)}")

        return _ConditionalFlowGenerator()


def build_generator(config: dict[str, Any] | FlowGeneratorConfig | None = None, **kwargs):
    payload: dict[str, Any] = {}
    if isinstance(config, FlowGeneratorConfig):
        return ConditionalFlowGenerator(config)
    if config is not None:
        payload.update(config)
    payload.update(kwargs)
    model_type = payload.pop("model_type", "conditional_flow_matching")
    if model_type != "conditional_flow_matching":
        raise ValueError(f"Unsupported generator model_type: {model_type}")
    return ConditionalFlowGenerator(payload)


def _groups_for(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    raise ValueError(f"Cannot choose GroupNorm groups for channels={channels}")


def _validate_config(config: FlowGeneratorConfig) -> None:
    if config.embedding_dim != 512:
        raise ValueError(f"Generator expects 512-d z, got {config.embedding_dim}")
    if config.image_size != 224:
        raise ValueError(f"Generator image_size is fixed at 224, got {config.image_size}")
    if config.base_channels <= 0:
        raise ValueError(f"base_channels must be positive, got {config.base_channels}")
    if not config.channel_multipliers:
        raise ValueError("channel_multipliers must not be empty")
    if any(item <= 0 for item in config.channel_multipliers):
        raise ValueError(f"channel_multipliers must be positive, got {config.channel_multipliers}")
    if config.time_embedding_dim <= 0:
        raise ValueError(f"time_embedding_dim must be positive, got {config.time_embedding_dim}")
    if config.condition_dim <= 0:
        raise ValueError(f"condition_dim must be positive, got {config.condition_dim}")
    if config.sample_steps <= 0:
        raise ValueError(f"sample_steps must be positive, got {config.sample_steps}")
    if config.train_cycle_steps <= 0:
        raise ValueError(f"train_cycle_steps must be positive, got {config.train_cycle_steps}")
    if config.sampler not in {"euler", "heun"}:
        raise ValueError(f"sampler must be euler or heun, got {config.sampler}")
