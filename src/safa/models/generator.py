from __future__ import annotations

from dataclasses import dataclass
from typing import Any


GENERATOR_MODEL_TYPE_FLOW = "conditional_flow_matching"
GENERATOR_MODEL_TYPE_MEANFLOW = "meanflow"
GENERATOR_MODEL_TYPE_MEANFLOW_SIT = "meanflow_sit"
GENERATOR_MODEL_TYPE_SIT_DIFFUSION = "sit_diffusion"
GENERATOR_MODEL_TYPE_DDIM = "ddim"
GENERATOR_MODEL_TYPES = (
    GENERATOR_MODEL_TYPE_FLOW,
    GENERATOR_MODEL_TYPE_MEANFLOW,
    GENERATOR_MODEL_TYPE_MEANFLOW_SIT,
    GENERATOR_MODEL_TYPE_SIT_DIFFUSION,
    GENERATOR_MODEL_TYPE_DDIM,
)
MEANFLOW_JVP_MODE_TORCH_FUNC = "torch_func"
MEANFLOW_JVP_MODE_FIRST_ORDER = "first_order"
MEANFLOW_JVP_MODES = (MEANFLOW_JVP_MODE_TORCH_FUNC, MEANFLOW_JVP_MODE_FIRST_ORDER)
SIT_DATA_SPACE_PIXEL = "pixel"
SIT_DATA_SPACE_LATENT = "latent"
SIT_DATA_SPACES = (SIT_DATA_SPACE_PIXEL, SIT_DATA_SPACE_LATENT)
DDIM_BETA_SCHEDULE_LINEAR = "linear"
DDIM_BETA_SCHEDULE_COSINE = "cosine"
DDIM_BETA_SCHEDULES = (DDIM_BETA_SCHEDULE_LINEAR, DDIM_BETA_SCHEDULE_COSINE)
DIFFUSION_PREDICTION_TYPE_EPSILON = "epsilon"
DIFFUSION_PREDICTION_TYPES = (DIFFUSION_PREDICTION_TYPE_EPSILON,)
SIT_ATTENTION_BACKEND_AUTO = "auto"
SIT_ATTENTION_BACKEND_NATIVE = "native"
SIT_ATTENTION_BACKEND_SDPA = "sdpa"
SIT_ATTENTION_BACKEND_FA2 = "fa2"
SIT_ATTENTION_BACKEND_FA4 = "fa4"
SIT_ATTENTION_BACKENDS = (
    SIT_ATTENTION_BACKEND_AUTO,
    SIT_ATTENTION_BACKEND_NATIVE,
    SIT_ATTENTION_BACKEND_SDPA,
    SIT_ATTENTION_BACKEND_FA2,
    SIT_ATTENTION_BACKEND_FA4,
)

GENERATOR_CHECKPOINT_MODEL_CONFIG_FIELDS = (
    "model_type",
    "embedding_dim",
    "image_size",
    "base_channels",
    "channel_multipliers",
    "time_embedding_dim",
    "condition_dim",
    "sample_steps",
    "train_cycle_steps",
    "sampler",
)

FLOW_GENERATOR_REQUIRED_CONFIG_FIELDS = (
    "embedding_dim",
    "image_size",
    "base_channels",
    "channel_multipliers",
    "condition_dim",
    "sample_steps",
    "train_cycle_steps",
    "sampler",
)


@dataclass(frozen=True)
class FlowGeneratorConfig:
    model_type: str = GENERATOR_MODEL_TYPE_FLOW
    embedding_dim: int = 512
    image_size: int = 224
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    time_embedding_dim: int = 128
    condition_dim: int = 512
    sample_steps: int = 32
    train_cycle_steps: int = 8
    cycle_steps_schedule: tuple[int, ...] = ()
    sampler: str = "heun"
    learned_null_condition: bool = False
    meanflow_ratio: float = 0.75
    meanflow_ratio_r_not_equal_t: float = -1.0
    meanflow_adaptive_weighting: bool = True
    meanflow_norm_p: float = 0.75
    meanflow_norm_eps: float = 1.0e-3
    meanflow_jvp_mode: str = MEANFLOW_JVP_MODE_TORCH_FUNC
    ddim_num_train_timesteps: int = 1000
    ddim_beta_schedule: str = DDIM_BETA_SCHEDULE_LINEAR
    ddim_beta_start: float = 1.0e-4
    ddim_beta_end: float = 2.0e-2
    ddim_eta: float = 0.0
    diffusion_train_timesteps: int = 1000
    diffusion_beta_schedule: str = DDIM_BETA_SCHEDULE_LINEAR
    diffusion_prediction_type: str = DIFFUSION_PREDICTION_TYPE_EPSILON
    sit_input_channels: int = 3
    sit_patch_size: int = 16
    sit_hidden_size: int = 768
    sit_depth: int = 12
    sit_num_heads: int = 12
    sit_mlp_ratio: float = 4.0
    sit_time_embedding_dim: int = 256
    sit_pretrained_path: str = ""
    sit_pretrained_state_key: str = ""
    sit_data_space: str = SIT_DATA_SPACE_PIXEL
    attention_backend: str = SIT_ATTENTION_BACKEND_AUTO

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FlowGeneratorConfig":
        missing = [field for field in FLOW_GENERATOR_REQUIRED_CONFIG_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"FlowGeneratorConfig.from_dict missing required fields: {missing}")
        return cls(
            model_type=str(payload.get("model_type", GENERATOR_MODEL_TYPE_FLOW)),
            embedding_dim=int(payload["embedding_dim"]),
            image_size=int(payload["image_size"]),
            base_channels=int(payload["base_channels"]),
            channel_multipliers=tuple(int(item) for item in payload["channel_multipliers"]),
            time_embedding_dim=int(payload.get("time_embedding_dim", 128)),
            condition_dim=int(payload["condition_dim"]),
            sample_steps=int(payload["sample_steps"]),
            train_cycle_steps=int(payload["train_cycle_steps"]),
            cycle_steps_schedule=tuple(int(s) for s in payload["cycle_steps_schedule"]) if "cycle_steps_schedule" in payload and payload["cycle_steps_schedule"] else (),
            sampler=str(payload["sampler"]),
            learned_null_condition=bool(payload.get("learned_null_condition", False)),
            meanflow_ratio=float(payload.get("meanflow_ratio", 0.75)),
            meanflow_ratio_r_not_equal_t=float(payload.get("meanflow_ratio_r_not_equal_t", -1.0)),
            meanflow_adaptive_weighting=bool(payload.get("meanflow_adaptive_weighting", True)),
            meanflow_norm_p=float(payload.get("meanflow_norm_p", 0.75)),
            meanflow_norm_eps=float(payload.get("meanflow_norm_eps", 1.0e-3)),
            meanflow_jvp_mode=str(payload.get("meanflow_jvp_mode", MEANFLOW_JVP_MODE_TORCH_FUNC)),
            ddim_num_train_timesteps=int(payload.get("ddim_num_train_timesteps", 1000)),
            ddim_beta_schedule=str(payload.get("ddim_beta_schedule", DDIM_BETA_SCHEDULE_LINEAR)),
            ddim_beta_start=float(payload.get("ddim_beta_start", 1.0e-4)),
            ddim_beta_end=float(payload.get("ddim_beta_end", 2.0e-2)),
            ddim_eta=float(payload.get("ddim_eta", 0.0)),
            diffusion_train_timesteps=int(payload.get("diffusion_train_timesteps", 1000)),
            diffusion_beta_schedule=str(payload.get("diffusion_beta_schedule", DDIM_BETA_SCHEDULE_LINEAR)),
            diffusion_prediction_type=str(payload.get("diffusion_prediction_type", DIFFUSION_PREDICTION_TYPE_EPSILON)),
            sit_input_channels=int(payload.get("sit_input_channels", 3)),
            sit_patch_size=int(payload.get("sit_patch_size", 16)),
            sit_hidden_size=int(payload.get("sit_hidden_size", 768)),
            sit_depth=int(payload.get("sit_depth", 12)),
            sit_num_heads=int(payload.get("sit_num_heads", 12)),
            sit_mlp_ratio=float(payload.get("sit_mlp_ratio", 4.0)),
            sit_time_embedding_dim=int(payload.get("sit_time_embedding_dim", 256)),
            sit_pretrained_path=str(payload.get("sit_pretrained_path", "")),
            sit_pretrained_state_key=str(payload.get("sit_pretrained_state_key", "")),
            sit_data_space=str(payload.get("sit_data_space", SIT_DATA_SPACE_PIXEL)),
            attention_backend=str(payload.get("attention_backend", SIT_ATTENTION_BACKEND_AUTO)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_type": self.model_type,
            "embedding_dim": self.embedding_dim,
            "image_size": self.image_size,
            "base_channels": self.base_channels,
            "channel_multipliers": list(self.channel_multipliers),
            "time_embedding_dim": self.time_embedding_dim,
            "condition_dim": self.condition_dim,
            "sample_steps": self.sample_steps,
            "train_cycle_steps": self.train_cycle_steps,
            "sampler": self.sampler,
            **({"cycle_steps_schedule": list(self.cycle_steps_schedule)} if self.cycle_steps_schedule else {}),
        }
        if self.learned_null_condition:
            payload["learned_null_condition"] = True
        if self.model_type in {GENERATOR_MODEL_TYPE_MEANFLOW, GENERATOR_MODEL_TYPE_MEANFLOW_SIT}:
            payload.update(
                {
                    "meanflow_ratio": self.meanflow_ratio,
                    "meanflow_adaptive_weighting": self.meanflow_adaptive_weighting,
                    "meanflow_norm_p": self.meanflow_norm_p,
                    "meanflow_norm_eps": self.meanflow_norm_eps,
                    "meanflow_jvp_mode": self.meanflow_jvp_mode,
                }
            )
        if self.model_type == GENERATOR_MODEL_TYPE_MEANFLOW_SIT:
            payload["meanflow_ratio_r_not_equal_t"] = (
                self.meanflow_ratio_r_not_equal_t if self.meanflow_ratio_r_not_equal_t >= 0.0 else 1.0 - self.meanflow_ratio
            )
            payload.update(
                {
                    "sit_input_channels": self.sit_input_channels,
                    "sit_patch_size": self.sit_patch_size,
                    "sit_hidden_size": self.sit_hidden_size,
                    "sit_depth": self.sit_depth,
                    "sit_num_heads": self.sit_num_heads,
                    "sit_mlp_ratio": self.sit_mlp_ratio,
                    "sit_time_embedding_dim": self.sit_time_embedding_dim,
                }
            )
            payload["sit_data_space"] = self.sit_data_space
            payload["attention_backend"] = self.attention_backend
            if self.sit_pretrained_path:
                payload["sit_pretrained_path"] = self.sit_pretrained_path
            if self.sit_pretrained_state_key:
                payload["sit_pretrained_state_key"] = self.sit_pretrained_state_key
        if self.model_type == GENERATOR_MODEL_TYPE_SIT_DIFFUSION:
            payload.update(
                {
                    "sit_input_channels": self.sit_input_channels,
                    "sit_patch_size": self.sit_patch_size,
                    "sit_hidden_size": self.sit_hidden_size,
                    "sit_depth": self.sit_depth,
                    "sit_num_heads": self.sit_num_heads,
                    "sit_mlp_ratio": self.sit_mlp_ratio,
                    "sit_time_embedding_dim": self.sit_time_embedding_dim,
                    "sit_data_space": self.sit_data_space,
                    "attention_backend": self.attention_backend,
                    "diffusion_train_timesteps": self.diffusion_train_timesteps,
                    "diffusion_beta_schedule": self.diffusion_beta_schedule,
                    "diffusion_prediction_type": self.diffusion_prediction_type,
                    "ddim_beta_start": self.ddim_beta_start,
                    "ddim_beta_end": self.ddim_beta_end,
                    "ddim_eta": self.ddim_eta,
                }
            )
            if self.sit_pretrained_path:
                payload["sit_pretrained_path"] = self.sit_pretrained_path
            if self.sit_pretrained_state_key:
                payload["sit_pretrained_state_key"] = self.sit_pretrained_state_key
        if self.model_type == GENERATOR_MODEL_TYPE_DDIM:
            payload.update(
                {
                    "ddim_num_train_timesteps": self.ddim_num_train_timesteps,
                    "ddim_beta_schedule": self.ddim_beta_schedule,
                    "ddim_beta_start": self.ddim_beta_start,
                    "ddim_beta_end": self.ddim_beta_end,
                    "ddim_eta": self.ddim_eta,
                }
            )
        return payload


def require_generator_model_config(payload: dict[str, Any], checkpoint_path: str) -> dict[str, Any]:
    model_config = payload.get("model_config") if isinstance(payload, dict) else None
    if not isinstance(model_config, dict):
        raise ValueError(f"Generator checkpoint missing model_config: {checkpoint_path}")
    missing = [field for field in GENERATOR_CHECKPOINT_MODEL_CONFIG_FIELDS if field not in model_config]
    if missing:
        raise ValueError(f"Generator checkpoint model_config missing fields {missing}: {checkpoint_path}")
    return dict(model_config)


class ConditionalFlowGenerator:
    def __new__(cls, config: FlowGeneratorConfig | dict[str, Any] | None = None, **kwargs):
        import math
        import torch
        from torch import nn
        import torch.nn.functional as F

        from safa.models.conditioning import LearnedNullCondition

        cfg_payload = {}
        if isinstance(config, FlowGeneratorConfig):
            cfg = config
        elif config is None and not kwargs:
            cfg = FlowGeneratorConfig()
        else:
            if config is not None:
                cfg_payload.update(config)
            cfg_payload.update(kwargs)
            cfg = FlowGeneratorConfig.from_dict(cfg_payload)
        _validate_config(cfg)
        if cfg.model_type != GENERATOR_MODEL_TYPE_FLOW:
            raise ValueError(f"ConditionalFlowGenerator requires model_type={GENERATOR_MODEL_TYPE_FLOW!r}, got {cfg.model_type!r}")

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
                self.null_condition = LearnedNullCondition(cfg.embedding_dim) if cfg.learned_null_condition else None

            def forward(self, z):
                return self.sample(z, steps=self.config.sample_steps)

            def _single_step(self, x, z, step_index, total_steps):
                dt = 1.0 / float(total_steps)
                t = torch.full((z.shape[0],), step_index / float(total_steps), device=z.device, dtype=z.dtype)
                velocity = self.vector_field(x, t, z)
                if self.config.sampler == "euler":
                    return x + dt * velocity
                elif self.config.sampler == "heun":
                    # Euler for the final step: t=1.0 is outside training support
                    # (trained with t ~ Uniform[0,1)), so vector field at t=1.0
                    # is an uncontrolled extrapolation.
                    if step_index == total_steps - 1:
                        return x + dt * velocity
                    proposal = x + dt * velocity
                    next_t = torch.full((z.shape[0],), (step_index + 1) / float(total_steps), device=z.device, dtype=z.dtype)
                    next_velocity = self.vector_field(proposal, next_t, z)
                    return x + 0.5 * dt * (velocity + next_velocity)
                else:
                    raise ValueError(f"Unsupported sampler: {self.config.sampler}")

            def sample(self, z, steps: int | None = None, checkpoint_steps: bool = False, *, x_init=None, clamp_output: bool = True):
                self._validate_z(z)
                steps = int(steps or self.config.sample_steps)
                if steps <= 0:
                    raise ValueError(f"sample steps must be positive, got {steps}")
                if x_init is None:
                    x = torch.randn(z.shape[0], 3, self.image_size, self.image_size, device=z.device, dtype=z.dtype)
                else:
                    self._validate_x_init(x_init, z)
                    x = x_init
                divergence_step = None
                for index in range(steps):
                    if checkpoint_steps:
                        x = torch.utils.checkpoint.checkpoint(
                            self._single_step, x, z, index, steps,
                            use_reentrant=False,
                        )
                    else:
                        x = self._single_step(x, z, index, steps)
                    if divergence_step is None:
                        max_abs = x.abs().max().item()
                        if max_abs > 7.0:
                            divergence_step = index
                            print(f"WARNING: ODE solver divergence at step {index}/{steps}, max_abs={max_abs:.2f}")
                if divergence_step is not None and divergence_step < steps - 1:
                    print(f"WARNING: ODE solver diverged at step {divergence_step}, subsequent {steps - 1 - divergence_step} steps operated on diverged values")
                if clamp_output:
                    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
                return (x + 1.0) * 0.5

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

            def make_null_condition(self, *, batch_size: int, device, dtype):
                if self.null_condition is None:
                    raise RuntimeError("learned_null_condition is disabled for this generator")
                return self.null_condition(batch_size=batch_size, device=device, dtype=dtype)

            def _validate_z(self, z):
                if z.ndim != 2 or z.shape[1] != self.embedding_dim:
                    raise ValueError(f"G expects z with shape [B,{self.embedding_dim}], got {tuple(z.shape)}")

            def _validate_x_init(self, x_init, z):
                if not isinstance(x_init, torch.Tensor):
                    raise TypeError(f"x_init must be a torch.Tensor, got {type(x_init).__name__}")
                expected_shape = (z.shape[0], 3, self.image_size, self.image_size)
                if tuple(x_init.shape) != expected_shape:
                    raise ValueError(f"x_init must have shape {expected_shape}, got {tuple(x_init.shape)}")
                if x_init.device != z.device:
                    raise ValueError(f"x_init device must match z device {z.device}, got {x_init.device}")
                if x_init.dtype != z.dtype:
                    raise TypeError(f"x_init dtype must match z dtype {z.dtype}, got {x_init.dtype}")

        return _ConditionalFlowGenerator()


class MeanFlowGenerator:
    def __new__(cls, config: FlowGeneratorConfig | dict[str, Any] | None = None, **kwargs):
        import math
        import warnings

        import torch
        from torch import nn
        import torch.nn.functional as F

        from safa.models.conditioning import LearnedNullCondition

        cfg_payload = {}
        if isinstance(config, FlowGeneratorConfig):
            cfg = config
        elif config is None and not kwargs:
            cfg = FlowGeneratorConfig(model_type=GENERATOR_MODEL_TYPE_MEANFLOW, sampler="meanflow", sample_steps=1, train_cycle_steps=1)
        else:
            if config is not None:
                cfg_payload.update(config)
            cfg_payload.update(kwargs)
            cfg_payload.setdefault("model_type", GENERATOR_MODEL_TYPE_MEANFLOW)
            cfg_payload.setdefault("sampler", "meanflow")
            cfg = FlowGeneratorConfig.from_dict(cfg_payload)
        _validate_config(cfg)
        if cfg.model_type != GENERATOR_MODEL_TYPE_MEANFLOW:
            raise ValueError(f"MeanFlowGenerator requires model_type={GENERATOR_MODEL_TYPE_MEANFLOW!r}, got {cfg.model_type!r}")

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

        class MeanFlowUNet(nn.Module):
            def __init__(self):
                super().__init__()
                channels = [cfg.base_channels * item for item in cfg.channel_multipliers]
                self.input = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)
                self.time_mlp = nn.Sequential(
                    nn.Linear(cfg.time_embedding_dim, cfg.condition_dim),
                    nn.SiLU(),
                    nn.Linear(cfg.condition_dim, cfg.condition_dim),
                )
                self.horizon_mlp = nn.Sequential(
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

            def forward(self, x_t, t, r, z):
                if x_t.ndim != 4 or x_t.shape[1:] != (3, cfg.image_size, cfg.image_size):
                    raise ValueError(f"x_t must have shape [B,3,{cfg.image_size},{cfg.image_size}], got {tuple(x_t.shape)}")
                if z.ndim != 2 or z.shape[1] != cfg.embedding_dim:
                    raise ValueError(f"z must have shape [B,{cfg.embedding_dim}], got {tuple(z.shape)}")
                if t.ndim != 1 or t.shape[0] != z.shape[0]:
                    raise ValueError(f"t must have shape [B], got {tuple(t.shape)} for batch {z.shape[0]}")
                if r.ndim != 1 or r.shape[0] != z.shape[0]:
                    raise ValueError(f"r must have shape [B], got {tuple(r.shape)} for batch {z.shape[0]}")
                horizon = (t - r).clamp_min(0.0)
                condition = (
                    self.time_mlp(sinusoidal_embedding(t, cfg.time_embedding_dim))
                    + self.horizon_mlp(sinusoidal_embedding(horizon, cfg.time_embedding_dim))
                    + self.z_mlp(z)
                )
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

        class _MeanFlowGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = cfg
                self.embedding_dim = cfg.embedding_dim
                self.image_size = cfg.image_size
                self.vector_field = MeanFlowUNet()
                self.null_condition = LearnedNullCondition(cfg.embedding_dim) if cfg.learned_null_condition else None

            def forward(self, z):
                return self.sample(z, steps=1)

            def sample(self, z, steps: int | None = None, checkpoint_steps: bool = False, *, x_init=None, clamp_output: bool = True):
                del checkpoint_steps
                self._validate_z(z)
                requested_steps = 1 if steps is None else int(steps)
                if requested_steps <= 0:
                    raise ValueError(f"sample steps must be positive, got {requested_steps}")
                if requested_steps != 1:
                    warnings.warn("MeanFlowGenerator is 1-NFE; ignoring requested sample steps != 1", RuntimeWarning, stacklevel=2)
                if x_init is None:
                    x = torch.randn(z.shape[0], 3, self.image_size, self.image_size, device=z.device, dtype=z.dtype)
                else:
                    self._validate_x_init(x_init, z)
                    x = x_init
                t = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
                r = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
                velocity = self.vector_field(x, t, r, z)
                x = x - velocity
                if clamp_output:
                    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
                return (x + 1.0) * 0.5

            def flow_matching_loss(self, x_1, z, generator=None):
                self._validate_z(z)
                if x_1.ndim != 4 or x_1.shape[1:] != (3, self.image_size, self.image_size):
                    raise ValueError(f"x_1 must have shape [B,3,{self.image_size},{self.image_size}], got {tuple(x_1.shape)}")
                x_1_flow = x_1.mul(2.0).sub(1.0)
                x_0 = torch.randn(x_1_flow.shape, device=x_1_flow.device, dtype=x_1_flow.dtype, generator=generator)
                t, r = self._sample_t_r(x_1_flow.shape[0], device=x_1_flow.device, dtype=x_1_flow.dtype, generator=generator)
                view_t = t.view(-1, 1, 1, 1)
                x_t = (1.0 - view_t) * x_1_flow + view_t * x_0
                target_velocity = x_0 - x_1_flow
                predicted_velocity = self.vector_field(x_t, t, r, z)
                meanflow_target = self._meanflow_target(x_t, t, r, z, target_velocity)
                error = predicted_velocity - meanflow_target.detach()
                raw_mse = error.square().mean()
                loss = self._weighted_loss(error, meanflow_target)
                return loss, {
                    "flow_matching_mse": loss.detach(),
                    "meanflow_raw_mse": raw_mse.detach(),
                    "meanflow_ratio": x_1_flow.new_tensor(self.config.meanflow_ratio),
                    "meanflow_t_mean": t.detach().mean(),
                    "meanflow_r_mean": r.detach().mean(),
                    "meanflow_h_mean": (t - r).detach().mean(),
                    "meanflow_jvp_mode": self.config.meanflow_jvp_mode,
                    "meanflow_adaptive_weighting": x_1_flow.new_tensor(float(self.config.meanflow_adaptive_weighting)),
                    "target_velocity_abs_mean": target_velocity.detach().abs().mean(),
                    "predicted_velocity_abs_mean": predicted_velocity.detach().abs().mean(),
                }

            def make_null_condition(self, *, batch_size: int, device, dtype):
                if self.null_condition is None:
                    raise RuntimeError("learned_null_condition is disabled for this generator")
                return self.null_condition(batch_size=batch_size, device=device, dtype=dtype)

            def _sample_t_r(self, batch_size: int, *, device, dtype, generator=None):
                t = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
                r = t * torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
                if self.config.meanflow_ratio > 0.0:
                    same_mask = torch.rand(batch_size, device=device, generator=generator) < self.config.meanflow_ratio
                    r = torch.where(same_mask, t, r)
                return t, r

            def _meanflow_target(self, x_t, t, r, z, target_velocity):
                horizon = (t - r).view(-1, 1, 1, 1)
                if self.config.meanflow_jvp_mode == MEANFLOW_JVP_MODE_FIRST_ORDER:
                    # This is an explicit approximation mode for ablations only. The default
                    # torch_func path below uses a JVP through the network inputs.
                    return target_velocity
                from torch.func import jvp

                def field_fn(x_arg, t_arg, r_arg):
                    return self.vector_field(x_arg, t_arg, r_arg, z)

                _, dudt = jvp(
                    field_fn,
                    (x_t, t, r),
                    (
                        target_velocity,
                        torch.ones_like(t),
                        torch.zeros_like(r),
                    ),
                )
                return target_velocity - horizon * dudt

            def _weighted_loss(self, error, meanflow_target):
                per_sample = error.flatten(1).square().mean(dim=1)
                if not self.config.meanflow_adaptive_weighting:
                    return per_sample.mean()
                target_norm = meanflow_target.detach().flatten(1).norm(p=2, dim=1)
                target_norm = target_norm / math.sqrt(float(meanflow_target[0].numel()))
                weights = (target_norm + self.config.meanflow_norm_eps).pow(-self.config.meanflow_norm_p)
                weights = weights / weights.mean().clamp_min(self.config.meanflow_norm_eps)
                return (per_sample * weights).mean()

            def _validate_z(self, z):
                if z.ndim != 2 or z.shape[1] != self.embedding_dim:
                    raise ValueError(f"G expects z with shape [B,{self.embedding_dim}], got {tuple(z.shape)}")

            def _validate_x_init(self, x_init, z):
                if not isinstance(x_init, torch.Tensor):
                    raise TypeError(f"x_init must be a torch.Tensor, got {type(x_init).__name__}")
                expected_shape = (z.shape[0], 3, self.image_size, self.image_size)
                if tuple(x_init.shape) != expected_shape:
                    raise ValueError(f"x_init must have shape {expected_shape}, got {tuple(x_init.shape)}")
                if x_init.device != z.device:
                    raise ValueError(f"x_init device must match z device {z.device}, got {x_init.device}")
                if x_init.dtype != z.dtype:
                    raise TypeError(f"x_init dtype must match z dtype {z.dtype}, got {x_init.dtype}")

        return _MeanFlowGenerator()


class MeanFlowSiTGenerator:
    def __new__(cls, config: FlowGeneratorConfig | dict[str, Any] | None = None, **kwargs):
        from safa.models.meanflow_sit import build_meanflow_sit_generator

        cfg_payload = {}
        if isinstance(config, FlowGeneratorConfig):
            cfg = config
        elif config is None and not kwargs:
            cfg = FlowGeneratorConfig(model_type=GENERATOR_MODEL_TYPE_MEANFLOW_SIT, sampler="meanflow", sample_steps=1, train_cycle_steps=1)
        else:
            if config is not None:
                cfg_payload.update(config)
            cfg_payload.update(kwargs)
            cfg_payload.setdefault("model_type", GENERATOR_MODEL_TYPE_MEANFLOW_SIT)
            cfg_payload.setdefault("sampler", "meanflow")
            cfg = FlowGeneratorConfig.from_dict(cfg_payload)
        _validate_config(cfg)
        if cfg.model_type != GENERATOR_MODEL_TYPE_MEANFLOW_SIT:
            raise ValueError(f"MeanFlowSiTGenerator requires model_type={GENERATOR_MODEL_TYPE_MEANFLOW_SIT!r}, got {cfg.model_type!r}")
        generator = build_meanflow_sit_generator(cfg)
        if cfg.sit_pretrained_path:
            generator.pretrained_load_report = generator.load_pretrained(
                cfg.sit_pretrained_path,
                state_key=cfg.sit_pretrained_state_key or None,
                strict=False,
                allow_missing=False,
            )
        return generator


class SiTDiffusionGenerator:
    def __new__(cls, config: FlowGeneratorConfig | dict[str, Any] | None = None, **kwargs):
        from safa.models.sit_diffusion import build_sit_diffusion_generator

        cfg_payload = {}
        if isinstance(config, FlowGeneratorConfig):
            cfg = config
        elif config is None and not kwargs:
            cfg = FlowGeneratorConfig(model_type=GENERATOR_MODEL_TYPE_SIT_DIFFUSION, sampler="ddim", sample_steps=16, train_cycle_steps=16)
        else:
            if config is not None:
                cfg_payload.update(config)
            cfg_payload.update(kwargs)
            cfg_payload.setdefault("model_type", GENERATOR_MODEL_TYPE_SIT_DIFFUSION)
            cfg_payload.setdefault("sampler", "ddim")
            cfg = FlowGeneratorConfig.from_dict(cfg_payload)
        _validate_config(cfg)
        if cfg.model_type != GENERATOR_MODEL_TYPE_SIT_DIFFUSION:
            raise ValueError(f"SiTDiffusionGenerator requires model_type={GENERATOR_MODEL_TYPE_SIT_DIFFUSION!r}, got {cfg.model_type!r}")
        generator = build_sit_diffusion_generator(cfg)
        if cfg.sit_pretrained_path:
            generator.pretrained_load_report = generator.load_pretrained(
                cfg.sit_pretrained_path,
                state_key=cfg.sit_pretrained_state_key or None,
                strict=False,
                allow_missing=False,
            )
        return generator


class DDIMGenerator:
    def __new__(cls, config: FlowGeneratorConfig | dict[str, Any] | None = None, **kwargs):
        import math

        import torch
        from torch import nn
        import torch.nn.functional as F

        from safa.models.conditioning import LearnedNullCondition

        cfg_payload = {}
        if isinstance(config, FlowGeneratorConfig):
            cfg = config
        elif config is None and not kwargs:
            cfg = FlowGeneratorConfig(model_type=GENERATOR_MODEL_TYPE_DDIM, sampler="ddim", sample_steps=1, train_cycle_steps=1)
        else:
            if config is not None:
                cfg_payload.update(config)
            cfg_payload.update(kwargs)
            cfg_payload.setdefault("model_type", GENERATOR_MODEL_TYPE_DDIM)
            cfg_payload.setdefault("sampler", "ddim")
            cfg = FlowGeneratorConfig.from_dict(cfg_payload)
        _validate_config(cfg)
        if cfg.model_type != GENERATOR_MODEL_TYPE_DDIM:
            raise ValueError(f"DDIMGenerator requires model_type={GENERATOR_MODEL_TYPE_DDIM!r}, got {cfg.model_type!r}")

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

        def make_beta_schedule():
            if cfg.ddim_beta_schedule == DDIM_BETA_SCHEDULE_LINEAR:
                return torch.linspace(
                    cfg.ddim_beta_start,
                    cfg.ddim_beta_end,
                    cfg.ddim_num_train_timesteps,
                    dtype=torch.float32,
                )
            steps = cfg.ddim_num_train_timesteps + 1
            t = torch.linspace(0, cfg.ddim_num_train_timesteps, steps, dtype=torch.float64) / cfg.ddim_num_train_timesteps
            alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi * 0.5).pow(2)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(1.0e-8, 0.999).to(dtype=torch.float32)

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

        class DDIMDenoiserUNet(nn.Module):
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

        class _DDIMGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = cfg
                self.embedding_dim = cfg.embedding_dim
                self.image_size = cfg.image_size
                self.denoiser = DDIMDenoiserUNet()
                self.null_condition = LearnedNullCondition(cfg.embedding_dim) if cfg.learned_null_condition else None
                betas = make_beta_schedule()
                alphas = 1.0 - betas
                alphas_cumprod = torch.cumprod(alphas, dim=0)
                self.register_buffer("betas", betas, persistent=True)
                self.register_buffer("alphas", alphas, persistent=True)
                self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=True)
                self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=True)
                self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod), persistent=True)

            def forward(self, z):
                return self.sample(z, steps=self.config.sample_steps)

            def denoising_loss(self, x_0, z, generator=None):
                self._validate_z(z)
                if x_0.ndim != 4 or x_0.shape[1:] != (3, self.image_size, self.image_size):
                    raise ValueError(f"x_0 must have shape [B,3,{self.image_size},{self.image_size}], got {tuple(x_0.shape)}")
                x_0_diffusion = x_0.mul(2.0).sub(1.0)
                timesteps = torch.randint(
                    0,
                    self.config.ddim_num_train_timesteps,
                    (x_0_diffusion.shape[0],),
                    device=x_0_diffusion.device,
                    generator=generator,
                    dtype=torch.long,
                )
                noise = torch.randn(x_0_diffusion.shape, device=x_0_diffusion.device, dtype=x_0_diffusion.dtype, generator=generator)
                sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, x_0_diffusion)
                sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_0_diffusion)
                x_t = sqrt_alpha_bar * x_0_diffusion + sqrt_one_minus_alpha_bar * noise
                predicted_noise = self.denoiser(x_t, self._normalize_timesteps(timesteps, dtype=x_t.dtype), z)
                loss = F.mse_loss(predicted_noise, noise)
                return loss, {
                    "flow_matching_mse": loss.detach(),
                    "ddim_denoising_mse": loss.detach(),
                    "ddim_timestep_mean": timesteps.detach().to(dtype=x_0_diffusion.dtype).mean(),
                    "target_noise_abs_mean": noise.detach().abs().mean(),
                    "predicted_noise_abs_mean": predicted_noise.detach().abs().mean(),
                }

            def flow_matching_loss(self, x_0, z, generator=None):
                return self.denoising_loss(x_0, z, generator=generator)

            def sample(self, z, steps: int | None = None, checkpoint_steps: bool = False, *, x_init=None, clamp_output: bool = True):
                del checkpoint_steps
                self._validate_z(z)
                steps = int(steps or self.config.sample_steps)
                if steps <= 0:
                    raise ValueError(f"sample steps must be positive, got {steps}")
                if x_init is None:
                    x = torch.randn(z.shape[0], 3, self.image_size, self.image_size, device=z.device, dtype=z.dtype)
                else:
                    self._validate_x_init(x_init, z)
                    x = x_init
                timesteps = self.ddim_timesteps(steps).to(device=z.device)
                for index, timestep in enumerate(timesteps):
                    next_timestep = timesteps[index + 1] if index + 1 < len(timesteps) else None
                    t_batch = torch.full((z.shape[0],), int(timestep.item()), device=z.device, dtype=torch.long)
                    x = self._ddim_step(x, z, t_batch, next_timestep)
                if clamp_output:
                    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
                return (x + 1.0) * 0.5

            def ddim_timesteps(self, steps: int):
                steps = int(steps)
                if steps <= 0:
                    raise ValueError(f"sample steps must be positive, got {steps}")
                max_timestep = self.config.ddim_num_train_timesteps - 1
                return torch.linspace(max_timestep, 0, steps, dtype=torch.float64).round().to(dtype=torch.long)

            def make_null_condition(self, *, batch_size: int, device, dtype):
                if self.null_condition is None:
                    raise RuntimeError("learned_null_condition is disabled for this generator")
                return self.null_condition(batch_size=batch_size, device=device, dtype=dtype)

            def _ddim_step(self, x_t, z, timesteps, next_timestep):
                alpha_bar_t = self._extract(self.alphas_cumprod, timesteps, x_t)
                eps = self.denoiser(x_t, self._normalize_timesteps(timesteps, dtype=x_t.dtype), z)
                pred_x0 = (x_t - (1.0 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt().clamp_min(1.0e-12)
                if next_timestep is None:
                    return pred_x0
                next_timesteps = torch.full_like(timesteps, int(next_timestep.item()))
                alpha_bar_prev = self._extract(self.alphas_cumprod, next_timesteps, x_t)
                return alpha_bar_prev.sqrt() * pred_x0 + (1.0 - alpha_bar_prev).sqrt() * eps

            def _extract(self, values, timesteps, target):
                gathered = values.to(device=timesteps.device, dtype=target.dtype).gather(0, timesteps)
                return gathered.view(-1, 1, 1, 1)

            def _normalize_timesteps(self, timesteps, *, dtype):
                max_timestep = max(self.config.ddim_num_train_timesteps - 1, 1)
                return timesteps.to(dtype=dtype) / float(max_timestep)

            def _validate_z(self, z):
                if z.ndim != 2 or z.shape[1] != self.embedding_dim:
                    raise ValueError(f"G expects z with shape [B,{self.embedding_dim}], got {tuple(z.shape)}")

            def _validate_x_init(self, x_init, z):
                if not isinstance(x_init, torch.Tensor):
                    raise TypeError(f"x_init must be a torch.Tensor, got {type(x_init).__name__}")
                expected_shape = (z.shape[0], 3, self.image_size, self.image_size)
                if tuple(x_init.shape) != expected_shape:
                    raise ValueError(f"x_init must have shape {expected_shape}, got {tuple(x_init.shape)}")
                if x_init.device != z.device:
                    raise ValueError(f"x_init device must match z device {z.device}, got {x_init.device}")
                if x_init.dtype != z.dtype:
                    raise TypeError(f"x_init dtype must match z dtype {z.dtype}, got {x_init.dtype}")

        return _DDIMGenerator()


def build_generator(config: dict[str, Any] | FlowGeneratorConfig | None = None, **kwargs):
    payload: dict[str, Any] = {}
    if isinstance(config, FlowGeneratorConfig):
        if config.model_type == GENERATOR_MODEL_TYPE_FLOW:
            return ConditionalFlowGenerator(config)
        if config.model_type == GENERATOR_MODEL_TYPE_MEANFLOW:
            return MeanFlowGenerator(config)
        if config.model_type == GENERATOR_MODEL_TYPE_MEANFLOW_SIT:
            return MeanFlowSiTGenerator(config)
        if config.model_type == GENERATOR_MODEL_TYPE_SIT_DIFFUSION:
            return SiTDiffusionGenerator(config)
        if config.model_type == GENERATOR_MODEL_TYPE_DDIM:
            return DDIMGenerator(config)
        raise ValueError(f"Unsupported generator model_type: {config.model_type}")
    if config is None and not kwargs:
        return ConditionalFlowGenerator()
    if config is not None:
        payload.update(config)
    payload.update(kwargs)
    model_type = payload.get("model_type", GENERATOR_MODEL_TYPE_FLOW)
    if model_type == GENERATOR_MODEL_TYPE_FLOW:
        return ConditionalFlowGenerator(payload)
    if model_type == GENERATOR_MODEL_TYPE_MEANFLOW:
        return MeanFlowGenerator(payload)
    if model_type == GENERATOR_MODEL_TYPE_MEANFLOW_SIT:
        return MeanFlowSiTGenerator(payload)
    if model_type == GENERATOR_MODEL_TYPE_SIT_DIFFUSION:
        return SiTDiffusionGenerator(payload)
    if model_type == GENERATOR_MODEL_TYPE_DDIM:
        return DDIMGenerator(payload)
    raise ValueError(f"Unsupported generator model_type: {model_type}")


def _groups_for(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    raise ValueError(f"Cannot choose GroupNorm groups for channels={channels}")


def _validate_config(config: FlowGeneratorConfig) -> None:
    if config.model_type not in GENERATOR_MODEL_TYPES:
        allowed = ", ".join(GENERATOR_MODEL_TYPES)
        raise ValueError(f"model_type must be one of {allowed}, got {config.model_type!r}")
    if config.embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {config.embedding_dim}")
    if config.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {config.image_size}")
    if config.base_channels <= 0:
        raise ValueError(f"base_channels must be positive, got {config.base_channels}")
    if not config.channel_multipliers:
        raise ValueError("channel_multipliers must not be empty")
    if any(item <= 0 for item in config.channel_multipliers):
        raise ValueError(f"channel_multipliers must be positive, got {config.channel_multipliers}")
    downsample_factor = 2 ** len(config.channel_multipliers)
    if config.image_size % downsample_factor != 0:
        raise ValueError(
            f"image_size must be divisible by {downsample_factor} for "
            f"{len(config.channel_multipliers)} downsampling stages, got {config.image_size}"
        )
    if config.time_embedding_dim <= 0:
        raise ValueError(f"time_embedding_dim must be positive, got {config.time_embedding_dim}")
    if config.condition_dim <= 0:
        raise ValueError(f"condition_dim must be positive, got {config.condition_dim}")
    if config.sample_steps <= 0:
        raise ValueError(f"sample_steps must be positive, got {config.sample_steps}")
    if config.train_cycle_steps <= 0:
        raise ValueError(f"train_cycle_steps must be positive, got {config.train_cycle_steps}")
    if config.model_type == GENERATOR_MODEL_TYPE_FLOW and config.sampler not in {"euler", "heun"}:
        raise ValueError(f"sampler must be euler or heun, got {config.sampler}")
    if config.model_type in {GENERATOR_MODEL_TYPE_MEANFLOW, GENERATOR_MODEL_TYPE_MEANFLOW_SIT}:
        if config.sampler != "meanflow":
            raise ValueError(f"meanflow sampler must be 'meanflow', got {config.sampler!r}")
        if config.sample_steps != 1:
            raise ValueError(f"meanflow sample_steps must be 1, got {config.sample_steps}")
        if config.train_cycle_steps != 1:
            raise ValueError(f"meanflow train_cycle_steps must be 1, got {config.train_cycle_steps}")
        if config.meanflow_ratio < 0.0 or config.meanflow_ratio > 1.0:
            raise ValueError(f"meanflow_ratio must be in [0, 1], got {config.meanflow_ratio}")
        if config.model_type == GENERATOR_MODEL_TYPE_MEANFLOW_SIT and config.meanflow_ratio_r_not_equal_t != -1.0:
            if config.meanflow_ratio_r_not_equal_t < 0.0 or config.meanflow_ratio_r_not_equal_t > 1.0:
                raise ValueError(f"meanflow_ratio_r_not_equal_t must be in [0, 1], got {config.meanflow_ratio_r_not_equal_t}")
        if config.meanflow_norm_p < 0.0:
            raise ValueError(f"meanflow_norm_p must be non-negative, got {config.meanflow_norm_p}")
        if config.meanflow_norm_eps <= 0.0:
            raise ValueError(f"meanflow_norm_eps must be positive, got {config.meanflow_norm_eps}")
        if config.meanflow_jvp_mode not in MEANFLOW_JVP_MODES:
            allowed = ", ".join(MEANFLOW_JVP_MODES)
            raise ValueError(f"meanflow_jvp_mode must be one of {allowed}, got {config.meanflow_jvp_mode!r}")
    if config.model_type == GENERATOR_MODEL_TYPE_SIT_DIFFUSION:
        if config.sampler != "ddim":
            raise ValueError(f"sit_diffusion sampler must be 'ddim', got {config.sampler!r}")
        if config.diffusion_train_timesteps <= 1:
            raise ValueError(f"diffusion_train_timesteps must be greater than 1, got {config.diffusion_train_timesteps}")
        if config.diffusion_beta_schedule not in DDIM_BETA_SCHEDULES:
            allowed = ", ".join(DDIM_BETA_SCHEDULES)
            raise ValueError(f"diffusion_beta_schedule must be one of {allowed}, got {config.diffusion_beta_schedule!r}")
        if config.diffusion_prediction_type not in DIFFUSION_PREDICTION_TYPES:
            allowed = ", ".join(DIFFUSION_PREDICTION_TYPES)
            raise ValueError(f"diffusion_prediction_type must be one of {allowed}, got {config.diffusion_prediction_type!r}")
        if config.ddim_eta != 0.0:
            raise ValueError(f"sit_diffusion currently supports deterministic DDIM only, got ddim_eta={config.ddim_eta}")
    if config.model_type in {GENERATOR_MODEL_TYPE_MEANFLOW_SIT, GENERATOR_MODEL_TYPE_SIT_DIFFUSION}:
        if config.sit_input_channels <= 0:
            raise ValueError(f"sit_input_channels must be positive, got {config.sit_input_channels}")
        if config.sit_patch_size <= 0:
            raise ValueError(f"sit_patch_size must be positive, got {config.sit_patch_size}")
        if config.image_size % config.sit_patch_size != 0:
            raise ValueError(
                f"image_size must be divisible by sit_patch_size={config.sit_patch_size}, got {config.image_size}"
            )
        if config.sit_hidden_size <= 0:
            raise ValueError(f"sit_hidden_size must be positive, got {config.sit_hidden_size}")
        if config.sit_hidden_size % 4 != 0:
            raise ValueError(f"sit_hidden_size must be divisible by 4, got {config.sit_hidden_size}")
        if config.sit_depth <= 0:
            raise ValueError(f"sit_depth must be positive, got {config.sit_depth}")
        if config.sit_num_heads <= 0:
            raise ValueError(f"sit_num_heads must be positive, got {config.sit_num_heads}")
        if config.sit_hidden_size % config.sit_num_heads != 0:
            raise ValueError(
                f"sit_hidden_size must be divisible by sit_num_heads, got {config.sit_hidden_size} and {config.sit_num_heads}"
            )
        if config.sit_mlp_ratio <= 0.0:
            raise ValueError(f"sit_mlp_ratio must be positive, got {config.sit_mlp_ratio}")
        if config.sit_time_embedding_dim <= 0:
            raise ValueError(f"sit_time_embedding_dim must be positive, got {config.sit_time_embedding_dim}")
        if config.sit_data_space not in SIT_DATA_SPACES:
            raise ValueError(f"sit_data_space must be one of {SIT_DATA_SPACES}, got {config.sit_data_space!r}")
        if config.attention_backend not in SIT_ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend must be one of {SIT_ATTENTION_BACKENDS}, got {config.attention_backend!r}")
    if config.model_type == GENERATOR_MODEL_TYPE_DDIM:
        if config.sampler != "ddim":
            raise ValueError(f"ddim sampler must be 'ddim', got {config.sampler!r}")
        if config.ddim_num_train_timesteps <= 1:
            raise ValueError(f"ddim_num_train_timesteps must be greater than 1, got {config.ddim_num_train_timesteps}")
        if config.ddim_beta_schedule not in DDIM_BETA_SCHEDULES:
            allowed = ", ".join(DDIM_BETA_SCHEDULES)
            raise ValueError(f"ddim_beta_schedule must be one of {allowed}, got {config.ddim_beta_schedule!r}")
        if config.ddim_beta_start <= 0.0 or config.ddim_beta_start >= 1.0:
            raise ValueError(f"ddim_beta_start must be in (0, 1), got {config.ddim_beta_start}")
        if config.ddim_beta_end <= 0.0 or config.ddim_beta_end >= 1.0:
            raise ValueError(f"ddim_beta_end must be in (0, 1), got {config.ddim_beta_end}")
        if config.ddim_beta_schedule == DDIM_BETA_SCHEDULE_LINEAR and config.ddim_beta_start >= config.ddim_beta_end:
            raise ValueError(
                "linear DDIM beta schedule requires ddim_beta_start < ddim_beta_end, "
                f"got {config.ddim_beta_start} >= {config.ddim_beta_end}"
            )
        if config.ddim_eta != 0.0:
            raise ValueError(f"ddim_eta must be 0.0 for deterministic DDIM sampling, got {config.ddim_eta}")
    if not isinstance(config.learned_null_condition, bool):
        raise ValueError(f"learned_null_condition must be a bool, got {config.learned_null_condition!r}")
    if not isinstance(config.meanflow_adaptive_weighting, bool):
        raise ValueError(f"meanflow_adaptive_weighting must be a bool, got {config.meanflow_adaptive_weighting!r}")
