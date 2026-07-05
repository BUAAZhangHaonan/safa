from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from safa.models.conditioning import LearnedNullCondition
from safa.models.generator import (
    DDIM_BETA_SCHEDULE_COSINE,
    DDIM_BETA_SCHEDULE_LINEAR,
    SIT_DATA_SPACE_LATENT,
    SIT_DATA_SPACE_PIXEL,
)
from safa.models.meanflow_sit import (
    ATTENTION_BACKEND_FA2,
    ATTENTION_BACKEND_FA4,
    ATTENTION_BACKEND_NATIVE,
    ATTENTION_BACKEND_SDPA,
    _build_2d_sincos_pos_embed,
    _extract_state_dict,
    _first_tensor,
    _is_forward_ad_enabled,
    _load_fa2_attention_func,
    _load_fa4_attention_func,
    _looks_like_full_generator_state,
    _prepare_pretrained_state_dict,
    _sinusoidal_embedding,
    _strip_module_prefix,
    resolve_meanflow_sit_attention_backend,
)


DIFFUSION_PREDICTION_TYPE_EPSILON = "epsilon"


def build_sit_diffusion_generator(config):
    class TimestepEmbedder(nn.Module):
        def __init__(self, hidden_size: int, frequency_embedding_size: int):
            super().__init__()
            self.frequency_embedding_size = frequency_embedding_size
            self.mlp = nn.Sequential(
                nn.Linear(frequency_embedding_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        def forward(self, t):
            if t.ndim != 1:
                raise ValueError(f"t must have shape [B], got {tuple(t.shape)}")
            embedding = _sinusoidal_embedding(t, self.frequency_embedding_size)
            return self.mlp(embedding.to(dtype=t.dtype))

    class SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            hidden_size = config.sit_hidden_size
            self.num_heads = config.sit_num_heads
            self.head_dim = hidden_size // self.num_heads
            self.scale = self.head_dim**-0.5
            self.requested_attention_backend = config.attention_backend
            self.resolved_attention_backend = resolve_meanflow_sit_attention_backend(self.requested_attention_backend)
            self.qkv = nn.Linear(hidden_size, hidden_size * 3)
            self.proj = nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            batch_size, num_tokens, hidden_size = x.shape
            qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            x = self._attention(q, k, v)
            x = x.transpose(1, 2).reshape(batch_size, num_tokens, hidden_size)
            return self.proj(x)

        def _attention(self, q, k, v):
            backend = self.resolved_attention_backend
            if backend != ATTENTION_BACKEND_NATIVE and _is_forward_ad_enabled():
                return self._native_attention(q, k, v)
            if backend == ATTENTION_BACKEND_NATIVE:
                return self._native_attention(q, k, v)
            if backend == ATTENTION_BACKEND_SDPA:
                return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=self.scale)
            if backend == ATTENTION_BACKEND_FA2:
                func, _ = _load_fa2_attention_func()
                if func is None:
                    raise RuntimeError("FA2 attention backend was resolved but the standard flash_attn API is missing")
                return self._flash_attention(func, q, k, v)
            if backend == ATTENTION_BACKEND_FA4:
                func, _ = _load_fa4_attention_func()
                if func is None:
                    raise RuntimeError("FA4 attention backend was resolved but flash_attn.cute.flash_attn_func is missing")
                return self._flash_attention(func, q, k, v)
            raise RuntimeError(f"Unsupported SiT diffusion attention backend {backend!r}")

        def _native_attention(self, q, k, v):
            attention = (q @ k.transpose(-2, -1)) * self.scale
            attention = attention.softmax(dim=-1)
            return attention @ v

        def _flash_attention(self, func, q, k, v):
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            result = func(q, k, v, softmax_scale=self.scale, causal=False)
            out = _first_tensor(result)
            if not isinstance(out, torch.Tensor):
                raise RuntimeError(f"Flash attention backend returned {type(out).__name__}, expected Tensor")
            return out.transpose(1, 2)

    class SiTBlock(nn.Module):
        def __init__(self):
            super().__init__()
            hidden_size = config.sit_hidden_size
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1.0e-6)
            self.attn = SelfAttention()
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1.0e-6)
            mlp_hidden = int(hidden_size * config.sit_mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden, hidden_size),
            )
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

        def forward(self, x, condition):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=-1)
            attn_input = _modulate(self.norm1(x), shift_msa, scale_msa)
            x = x + gate_msa.unsqueeze(1) * self.attn(attn_input)
            mlp_input = _modulate(self.norm2(x), shift_mlp, scale_mlp)
            return x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)

    class FinalLayer(nn.Module):
        def __init__(self):
            super().__init__()
            hidden_size = config.sit_hidden_size
            patch_dim = config.sit_patch_size * config.sit_patch_size * config.sit_input_channels
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1.0e-6)
            self.linear = nn.Linear(hidden_size, patch_dim)
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

        def forward(self, x, condition):
            shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
            return self.linear(_modulate(self.norm_final(x), shift, scale))

    class SiTDiffusionBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            hidden_size = config.sit_hidden_size
            self.x_embedder = nn.Conv2d(
                config.sit_input_channels,
                hidden_size,
                kernel_size=config.sit_patch_size,
                stride=config.sit_patch_size,
            )
            grid_size = config.image_size // config.sit_patch_size
            pos_embed = _build_2d_sincos_pos_embed(hidden_size, grid_size)
            self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)
            self.t_embedder = TimestepEmbedder(hidden_size, config.sit_time_embedding_dim)
            self.z_embedder = nn.Sequential(
                nn.Linear(config.embedding_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.blocks = nn.ModuleList([SiTBlock() for _ in range(config.sit_depth)])
            self.requested_attention_backend = config.attention_backend
            self.resolved_attention_backend = self.blocks[0].attn.resolved_attention_backend
            self.final_layer = FinalLayer()
            self._initialize_weights()

        def forward(self, x, t, z):
            self._validate_inputs(x, t, z)
            hidden = self.x_embedder(x).flatten(2).transpose(1, 2)
            hidden = hidden + self.pos_embed.to(device=hidden.device, dtype=hidden.dtype)
            condition = self.t_embedder(t) + self.z_embedder(z)
            for block in self.blocks:
                hidden = block(hidden, condition)
            patches = self.final_layer(hidden, condition)
            return self._unpatchify(patches)

        def _unpatchify(self, patches):
            batch_size = patches.shape[0]
            patch_size = config.sit_patch_size
            channels = config.sit_input_channels
            grid_size = config.image_size // patch_size
            patches = patches.reshape(batch_size, grid_size, grid_size, patch_size, patch_size, channels)
            patches = torch.einsum("nhwpqc->nchpwq", patches)
            return patches.reshape(batch_size, channels, config.image_size, config.image_size)

        def _validate_inputs(self, x, t, z):
            expected_x = (config.sit_input_channels, config.image_size, config.image_size)
            if x.ndim != 4 or tuple(x.shape[1:]) != expected_x:
                raise ValueError(f"x_t must have shape [B,{expected_x[0]},{expected_x[1]},{expected_x[2]}], got {tuple(x.shape)}")
            if z.ndim != 2 or z.shape[1] != config.embedding_dim:
                raise ValueError(f"z must have shape [B,{config.embedding_dim}], got {tuple(z.shape)}")
            if t.ndim != 1 or t.shape[0] != z.shape[0]:
                raise ValueError(f"t must have shape [B], got {tuple(t.shape)} for batch {z.shape[0]}")
            if x.shape[0] != z.shape[0]:
                raise ValueError(f"x_t batch {x.shape[0]} must match z batch {z.shape[0]}")

        def _initialize_weights(self):
            def init(module):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            self.apply(init)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
            nn.init.zeros_(self.final_layer.linear.weight)
            nn.init.zeros_(self.final_layer.linear.bias)

    class _SiTDiffusionGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.embedding_dim = config.embedding_dim
            self.image_size = config.image_size
            self.denoiser = SiTDiffusionBackbone()
            self.null_condition = LearnedNullCondition(config.embedding_dim) if config.learned_null_condition else None
            self.attention_backend = self.denoiser.resolved_attention_backend
            self.requested_attention_backend = self.denoiser.requested_attention_backend
            betas = _make_beta_schedule(config)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.register_buffer("betas", betas, persistent=True)
            self.register_buffer("alphas", alphas, persistent=True)
            self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=True)
            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=True)
            self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod), persistent=True)

        def forward(self, z):
            return self.sample(z, steps=self.config.sample_steps)

        def flow_matching_loss(self, x_0, z, generator=None):
            self._validate_z(z)
            if x_0.ndim != 4 or tuple(x_0.shape[1:]) != (config.sit_input_channels, self.image_size, self.image_size):
                raise ValueError(
                    f"x_0 must have shape [B,{config.sit_input_channels},{self.image_size},{self.image_size}], got {tuple(x_0.shape)}"
                )
            x_data = self._data_to_model_space(x_0)
            timesteps = torch.randint(
                0,
                self.config.diffusion_train_timesteps,
                (x_data.shape[0],),
                device=x_data.device,
                generator=generator,
                dtype=torch.long,
            )
            noise = torch.randn(x_data.shape, device=x_data.device, dtype=x_data.dtype, generator=generator)
            sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, x_data)
            sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_data)
            x_t = sqrt_alpha_bar * x_data + sqrt_one_minus_alpha_bar * noise
            predicted_noise = self.denoiser(x_t, self._normalize_timesteps(timesteps, dtype=x_t.dtype), z)
            loss = F.mse_loss(predicted_noise, noise)
            if self.null_condition is not None:
                loss = loss + 0.0 * self.null_condition.embedding.sum()
            return loss, {
                "flow_matching_mse": loss.detach(),
                "diffusion_mse": loss.detach(),
                "diffusion_prediction_type": self.config.diffusion_prediction_type,
                "diffusion_timestep_mean": timesteps.detach().to(dtype=x_data.dtype).mean(),
                "target_noise_abs_mean": noise.detach().abs().mean(),
                "predicted_noise_abs_mean": predicted_noise.detach().abs().mean(),
                "sit_diffusion_attention_backend": self.attention_backend,
                "sit_diffusion_attention_backend_requested": self.requested_attention_backend,
            }

        def sample(self, z, steps: int | None = None, checkpoint_steps: bool = False, *, x_init=None, clamp_output: bool = True):
            del checkpoint_steps
            self._validate_z(z)
            steps = int(steps or self.config.sample_steps)
            if steps <= 0:
                raise ValueError(f"sample steps must be positive, got {steps}")
            if x_init is None:
                x = torch.randn(
                    z.shape[0],
                    config.sit_input_channels,
                    self.image_size,
                    self.image_size,
                    device=z.device,
                    dtype=z.dtype,
                )
            else:
                self._validate_x_init(x_init, z)
                x = x_init
            timesteps = self.ddim_timesteps(steps).to(device=z.device)
            for index, timestep in enumerate(timesteps):
                next_timestep = timesteps[index + 1] if index + 1 < len(timesteps) else None
                t_batch = torch.full((z.shape[0],), int(timestep.item()), device=z.device, dtype=torch.long)
                x = self._ddim_step(x, z, t_batch, next_timestep)
            return self._model_to_data_space(x, clamp_output=clamp_output)

        def ddim_timesteps(self, steps: int):
            steps = int(steps)
            if steps <= 0:
                raise ValueError(f"sample steps must be positive, got {steps}")
            max_timestep = self.config.diffusion_train_timesteps - 1
            return torch.linspace(max_timestep, 0, steps, dtype=torch.float64).round().to(dtype=torch.long)

        def make_null_condition(self, *, batch_size: int, device, dtype):
            if self.null_condition is None:
                raise RuntimeError("learned_null_condition is disabled for this generator")
            return self.null_condition(batch_size=batch_size, device=device, dtype=dtype)

        def load_pretrained(self, checkpoint_path: str | Path | None, *, state_key: str | None = None, strict: bool = False, allow_missing: bool = False):
            path = Path(checkpoint_path) if checkpoint_path else None
            if path is None or not path.is_file():
                if allow_missing:
                    return {
                        "loaded": False,
                        "missing_file": True,
                        "path": "" if path is None else str(path),
                        "source_format": "missing_file",
                        "loaded_keys": [],
                        "missing_keys": [],
                        "unexpected_keys": [],
                        "mismatched_keys": [],
                        "skipped_keys": [],
                    }
                raise FileNotFoundError("" if path is None else str(path))
            payload = torch.load(path, map_location="cpu")
            state_dict = _strip_module_prefix(_extract_state_dict(payload, state_key))
            target = self if _looks_like_full_generator_state(state_dict) else self.denoiser
            prepared_state, prepare_report = _prepare_pretrained_state_dict(state_dict, target.state_dict())
            incompatible = target.load_state_dict(prepared_state, strict=strict)
            unexpected_keys = list(prepare_report["unexpected_keys"]) + list(incompatible.unexpected_keys)
            return {
                "loaded": bool(prepared_state),
                "missing_file": False,
                "path": str(path),
                "source_format": prepare_report["source_format"],
                "loaded_keys": sorted(prepared_state),
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": prepare_report["mismatched_keys"],
                "skipped_keys": prepare_report["skipped_keys"],
            }

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
            max_timestep = max(self.config.diffusion_train_timesteps - 1, 1)
            return timesteps.to(dtype=dtype) / float(max_timestep)

        def _data_to_model_space(self, x):
            if config.sit_data_space == SIT_DATA_SPACE_PIXEL:
                return x.mul(2.0).sub(1.0)
            if config.sit_data_space == SIT_DATA_SPACE_LATENT:
                return x
            raise RuntimeError(f"Unsupported sit_data_space {config.sit_data_space!r}")

        def _model_to_data_space(self, x, *, clamp_output: bool):
            if config.sit_data_space == SIT_DATA_SPACE_PIXEL:
                return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0) if clamp_output else (x + 1.0) * 0.5
            if config.sit_data_space == SIT_DATA_SPACE_LATENT:
                return x
            raise RuntimeError(f"Unsupported sit_data_space {config.sit_data_space!r}")

        def _validate_z(self, z):
            if z.ndim != 2 or z.shape[1] != self.embedding_dim:
                raise ValueError(f"G expects z with shape [B,{self.embedding_dim}], got {tuple(z.shape)}")

        def _validate_x_init(self, x_init, z):
            if not isinstance(x_init, torch.Tensor):
                raise TypeError(f"x_init must be a torch.Tensor, got {type(x_init).__name__}")
            expected_shape = (z.shape[0], config.sit_input_channels, self.image_size, self.image_size)
            if tuple(x_init.shape) != expected_shape:
                raise ValueError(f"x_init must have shape {expected_shape}, got {tuple(x_init.shape)}")
            if x_init.device != z.device:
                raise ValueError(f"x_init device must match z device {z.device}, got {x_init.device}")
            if x_init.dtype != z.dtype:
                raise TypeError(f"x_init dtype must match z dtype {z.dtype}, got {x_init.dtype}")

    return _SiTDiffusionGenerator()


def _make_beta_schedule(config):
    if config.diffusion_beta_schedule == DDIM_BETA_SCHEDULE_LINEAR:
        return torch.linspace(
            config.ddim_beta_start,
            config.ddim_beta_end,
            config.diffusion_train_timesteps,
            dtype=torch.float32,
        )
    if config.diffusion_beta_schedule == DDIM_BETA_SCHEDULE_COSINE:
        steps = config.diffusion_train_timesteps + 1
        t = torch.linspace(0, config.diffusion_train_timesteps, steps, dtype=torch.float64) / config.diffusion_train_timesteps
        alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi * 0.5).pow(2)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1.0e-8, 0.999).to(dtype=torch.float32)
    raise ValueError(f"Unsupported diffusion_beta_schedule {config.diffusion_beta_schedule!r}")


def _modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
