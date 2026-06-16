from __future__ import annotations

from functools import lru_cache
import importlib
import math
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from safa.models.conditioning import LearnedNullCondition


ATTENTION_BACKEND_AUTO = "auto"
ATTENTION_BACKEND_NATIVE = "native"
ATTENTION_BACKEND_SDPA = "sdpa"
ATTENTION_BACKEND_FA2 = "fa2"
ATTENTION_BACKEND_FA4 = "fa4"
ATTENTION_BACKENDS = (
    ATTENTION_BACKEND_AUTO,
    ATTENTION_BACKEND_NATIVE,
    ATTENTION_BACKEND_SDPA,
    ATTENTION_BACKEND_FA2,
    ATTENTION_BACKEND_FA4,
)
ATTENTION_BACKEND_PRIORITY = (ATTENTION_BACKEND_FA4, ATTENTION_BACKEND_FA2, ATTENTION_BACKEND_SDPA, ATTENTION_BACKEND_NATIVE)


def resolve_meanflow_sit_attention_backend(requested_backend: str = ATTENTION_BACKEND_AUTO) -> str:
    requested = _normalize_attention_backend(requested_backend)
    if requested == ATTENTION_BACKEND_AUTO:
        for backend in ATTENTION_BACKEND_PRIORITY:
            if _is_attention_backend_available(backend):
                return backend
        return ATTENTION_BACKEND_NATIVE
    if not _is_attention_backend_available(requested):
        raise RuntimeError(f"MeanFlow-SiT attention backend {requested!r} is not available: {_attention_backend_unavailable_reason(requested)}")
    return requested


def describe_meanflow_sit_attention_backends() -> dict[str, dict[str, str | bool]]:
    return {
        backend: {
            "available": _is_attention_backend_available(backend),
            "reason": _attention_backend_unavailable_reason(backend),
        }
        for backend in ATTENTION_BACKEND_PRIORITY
    }


def clear_meanflow_sit_attention_backend_cache() -> None:
    _attention_backend_probe.cache_clear()


def _normalize_attention_backend(value: str) -> str:
    backend = str(value).lower()
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(f"attention_backend must be one of {ATTENTION_BACKENDS}, got {value!r}")
    return backend


def _is_attention_backend_available(backend: str) -> bool:
    return _attention_backend_probe(_normalize_attention_backend(backend))[0]


def _attention_backend_unavailable_reason(backend: str) -> str:
    return _attention_backend_probe(_normalize_attention_backend(backend))[1]


@lru_cache(maxsize=None)
def _attention_backend_probe(backend: str) -> tuple[bool, str]:
    if backend == ATTENTION_BACKEND_NATIVE:
        return True, "available"
    if backend == ATTENTION_BACKEND_SDPA:
        if hasattr(F, "scaled_dot_product_attention"):
            return True, "available"
        return False, "torch.nn.functional.scaled_dot_product_attention is missing"
    if backend == ATTENTION_BACKEND_FA2:
        fn, reason = _load_fa2_attention_func()
        if fn is None:
            return False, reason
        return _probe_flash_attention_func(fn, backend)
    if backend == ATTENTION_BACKEND_FA4:
        fn, reason = _load_fa4_attention_func()
        if fn is None:
            return False, reason
        return _probe_flash_attention_func(fn, backend)
    return False, f"unsupported backend {backend!r}"


def _load_fa2_attention_func():
    candidates = (
        ("flash_attn", "flash_attn_func"),
        ("flash_attn.flash_attn_interface", "flash_attn_func"),
    )
    errors: list[str] = []
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, attr_name)
        except Exception as exc:
            errors.append(f"{module_name}.{attr_name}: {type(exc).__name__}: {exc}")
            continue
        return func, "available"
    return None, "; ".join(errors) if errors else "standard flash_attn API is missing"


def _load_fa4_attention_func():
    try:
        module = importlib.import_module("flash_attn.cute")
        return getattr(module, "flash_attn_func"), "available"
    except Exception as exc:
        return None, f"flash_attn.cute.flash_attn_func: {type(exc).__name__}: {exc}"


def _probe_flash_attention_func(func, backend: str) -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    try:
        device = torch.device("cuda")
        q = torch.randn(1, 16, 2, 32, device=device, dtype=torch.float16, requires_grad=True)
        k = torch.randn(1, 16, 2, 32, device=device, dtype=torch.float16, requires_grad=True)
        v = torch.randn(1, 16, 2, 32, device=device, dtype=torch.float16, requires_grad=True)
        result = func(q, k, v, softmax_scale=32**-0.5, causal=False)
        out = _first_tensor(result)
        if not isinstance(out, torch.Tensor):
            return False, f"{backend} returned {type(out).__name__}, expected Tensor"
        out.float().sum().backward()
    except Exception as exc:
        return False, f"{backend} forward/backward probe failed: {type(exc).__name__}: {exc}"
    return True, "available"


def _first_tensor(result):
    return result[0] if isinstance(result, tuple) else result


def build_meanflow_sit_generator(config):
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
            self.requested_attention_backend = _normalize_attention_backend(config.attention_backend)
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
            raise RuntimeError(f"Unsupported MeanFlow-SiT attention backend {backend!r}")

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
            attn_output = self.attn(attn_input)
            x = x + gate_msa.unsqueeze(1) * attn_output
            mlp_input = _modulate(self.norm2(x), shift_mlp, scale_mlp)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
            return x

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

    class MeanFlowSiTBackbone(nn.Module):
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
            self.r_embedder = TimestepEmbedder(hidden_size, config.sit_time_embedding_dim)
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

        def forward(self, x, r, t, z):
            self._validate_inputs(x, r, t, z)
            hidden = self.x_embedder(x).flatten(2).transpose(1, 2)
            hidden = hidden + self.pos_embed.to(device=hidden.device, dtype=hidden.dtype)
            horizon = (t - r).clamp_min(0.0)
            condition = self.t_embedder(t) + self.r_embedder(horizon) + self.z_embedder(z)
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

        def _validate_inputs(self, x, r, t, z):
            expected_x = (config.sit_input_channels, config.image_size, config.image_size)
            if x.ndim != 4 or tuple(x.shape[1:]) != expected_x:
                raise ValueError(f"x_t must have shape [B,{expected_x[0]},{expected_x[1]},{expected_x[2]}], got {tuple(x.shape)}")
            if z.ndim != 2 or z.shape[1] != config.embedding_dim:
                raise ValueError(f"z must have shape [B,{config.embedding_dim}], got {tuple(z.shape)}")
            if t.ndim != 1 or t.shape[0] != z.shape[0]:
                raise ValueError(f"t must have shape [B], got {tuple(t.shape)} for batch {z.shape[0]}")
            if r.ndim != 1 or r.shape[0] != z.shape[0]:
                raise ValueError(f"r must have shape [B], got {tuple(r.shape)} for batch {z.shape[0]}")

        def _initialize_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            nn.init.xavier_uniform_(self.x_embedder.weight.view(self.x_embedder.weight.shape[0], -1))
            nn.init.constant_(self.x_embedder.bias, 0)
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    class _MeanFlowSiTGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.embedding_dim = config.embedding_dim
            self.image_size = config.image_size
            self.vector_field = MeanFlowSiTBackbone()
            self.null_condition = LearnedNullCondition(config.embedding_dim) if config.learned_null_condition else None
            self.requested_attention_backend = self.vector_field.requested_attention_backend
            self.pretrained_load_report: dict[str, Any] | None = None

        @property
        def attention_backend(self) -> str:
            return self.vector_field.resolved_attention_backend

        def forward(self, z):
            return self.sample(z, steps=1)

        def sample(self, z, steps: int | None = None, checkpoint_steps: bool = False, *, x_init=None, clamp_output: bool = True):
            del checkpoint_steps
            self._validate_z(z)
            requested_steps = 1 if steps is None else int(steps)
            if requested_steps <= 0:
                raise ValueError(f"sample steps must be positive, got {requested_steps}")
            if requested_steps != 1:
                warnings.warn("MeanFlowSiTGenerator is 1-NFE; ignoring requested sample steps != 1", RuntimeWarning, stacklevel=2)
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
            r = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
            t = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
            mean_velocity = self.vector_field(x, r, t, z)
            x = x - mean_velocity
            return self._model_to_data_space(x, clamp_output=clamp_output)

        def flow_matching_loss(self, x_1, z, generator=None):
            self._validate_z(z)
            if x_1.ndim != 4 or tuple(x_1.shape[1:]) != (config.sit_input_channels, self.image_size, self.image_size):
                raise ValueError(
                    f"x_1 must have shape [B,{config.sit_input_channels},{self.image_size},{self.image_size}], got {tuple(x_1.shape)}"
                )
            x_data = self._data_to_model_space(x_1)
            eps = torch.randn(x_data.shape, device=x_data.device, dtype=x_data.dtype, generator=generator)
            r, t = self._sample_t_r(x_data.shape[0], device=x_data.device, dtype=x_data.dtype, generator=generator)
            view_t = t.view(-1, 1, 1, 1)
            z_t = (1.0 - view_t) * x_data + view_t * eps
            target_velocity = eps - x_data
            predicted_velocity = self.vector_field(z_t, r, t, z)
            meanflow_target = self._meanflow_target(z_t, r, t, z, target_velocity)
            error = predicted_velocity - meanflow_target.detach()
            raw_mse = error.square().mean()
            loss = self._weighted_loss(error)
            return loss, {
                "flow_matching_mse": raw_mse.detach(),
                "meanflow_raw_mse": raw_mse.detach(),
                "meanflow_backbone": "sit",
                "meanflow_ratio": x_data.new_tensor(config.meanflow_ratio),
                "meanflow_ratio_r_not_equal_t": x_data.new_tensor(self._ratio_r_not_equal_t()),
                "meanflow_t_mean": t.detach().mean(),
                "meanflow_r_mean": r.detach().mean(),
                "meanflow_h_mean": (t - r).detach().mean(),
                "meanflow_jvp_mode": config.meanflow_jvp_mode,
                "meanflow_attention_backend": self.attention_backend,
                "meanflow_attention_backend_requested": self.requested_attention_backend,
                "meanflow_adaptive_weighting": x_data.new_tensor(float(config.meanflow_adaptive_weighting)),
                "target_velocity_abs_mean": target_velocity.detach().abs().mean(),
                "predicted_velocity_abs_mean": predicted_velocity.detach().abs().mean(),
            }

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
            target = self if _looks_like_full_generator_state(state_dict) else self.vector_field
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

        def _sample_t_r(self, batch_size: int, *, device, dtype, generator=None):
            samples = torch.rand(batch_size, 2, device=device, dtype=dtype, generator=generator)
            sorted_samples, _ = torch.sort(samples, dim=1)
            r = sorted_samples[:, 0]
            t = sorted_samples[:, 1]
            ratio_not_equal = self._ratio_r_not_equal_t()
            if ratio_not_equal < 1.0:
                equal_fraction = 1.0 - ratio_not_equal
                equal_mask = torch.rand(batch_size, device=device, generator=generator) < equal_fraction
                r = torch.where(equal_mask, t, r)
            return r, t

        def _ratio_r_not_equal_t(self):
            if config.meanflow_ratio_r_not_equal_t >= 0.0:
                return config.meanflow_ratio_r_not_equal_t
            return 1.0 - config.meanflow_ratio

        def _meanflow_target(self, z_t, r, t, z, target_velocity):
            horizon = (t - r).view(-1, 1, 1, 1)
            if config.meanflow_jvp_mode == "first_order":
                return target_velocity
            from torch.func import jvp

            def field_fn(x_arg, r_arg, t_arg, z_arg):
                return self.vector_field(x_arg, r_arg, t_arg, z_arg)

            z_t_jvp = _jvp_safe_tensor(z_t)
            r_jvp = _jvp_safe_tensor(r)
            t_jvp = _jvp_safe_tensor(t)
            z_jvp = _jvp_safe_tensor(z)
            velocity_jvp = _jvp_safe_tensor(target_velocity)
            _, dudt = jvp(
                field_fn,
                (z_t_jvp, r_jvp, t_jvp, z_jvp),
                (
                    velocity_jvp,
                    torch.zeros_like(r_jvp),
                    torch.ones_like(t_jvp),
                    torch.zeros_like(z_jvp),
                ),
            )
            return target_velocity - horizon * dudt

        def _weighted_loss(self, error):
            per_sample = error.flatten(1).square().mean(dim=1)
            if not config.meanflow_adaptive_weighting:
                return per_sample.mean()
            weights = (per_sample.detach() + config.meanflow_norm_eps).pow(-config.meanflow_norm_p)
            weights = weights / weights.mean().clamp_min(config.meanflow_norm_eps)
            return (per_sample * weights).mean()

        def _data_to_model_space(self, x):
            if config.sit_data_space == "pixel":
                return x.mul(2.0).sub(1.0)
            if config.sit_data_space == "latent":
                return x
            raise RuntimeError(f"Unsupported sit_data_space {config.sit_data_space!r}")

        def _model_to_data_space(self, x, *, clamp_output: bool):
            if config.sit_data_space == "pixel":
                return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0) if clamp_output else (x + 1.0) * 0.5
            if config.sit_data_space == "latent":
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

    return _MeanFlowSiTGenerator()


def _jvp_safe_tensor(tensor):
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _is_forward_ad_enabled() -> bool:
    return int(torch.autograd.forward_ad._current_level) >= 0


def _modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _sinusoidal_embedding(t, dim: int):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / max(half, 1)
    )
    args = t[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _build_2d_sincos_pos_embed(embed_dim: int, grid_size: int):
    if embed_dim % 4 != 0:
        raise ValueError(f"sit_hidden_size must be divisible by 4 for sin-cos position embedding, got {embed_dim}")
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    emb_h = _sincos_from_grid(embed_dim // 2, grid[1].reshape(-1))
    emb_w = _sincos_from_grid(embed_dim // 2, grid[0].reshape(-1))
    return torch.cat([emb_h, emb_w], dim=1)


def _sincos_from_grid(embed_dim: int, positions):
    half = embed_dim // 2
    omega = torch.arange(half, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(half, 1)))
    out = positions[:, None] * omega[None]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _extract_state_dict(payload, state_key: str | None):
    if state_key:
        if not isinstance(payload, dict) or state_key not in payload:
            raise KeyError(f"checkpoint missing state_key {state_key!r}")
        state_dict = payload[state_key]
    elif isinstance(payload, dict):
        for key in ("ema", "model", "model_state_dict", "state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                state_dict = value
                break
        else:
            state_dict = payload
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint state_dict must be a dict, got {type(state_dict).__name__}")
    return state_dict


def _strip_module_prefix(state_dict: dict[str, Any]):
    if not all(isinstance(key, str) for key in state_dict):
        return state_dict
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _looks_like_full_generator_state(state_dict: dict[str, Any]):
    return any(isinstance(key, str) and (key.startswith("vector_field.") or key.startswith("null_condition.")) for key in state_dict)


def _prepare_pretrained_state_dict(state_dict: dict[str, Any], target_state: dict[str, Any]):
    source_format = "zhuyu_meanflow_sit" if _looks_like_zhuyu_state_dict(state_dict) else "safa"
    prepared: dict[str, Any] = {}
    unexpected_keys: list[str] = []
    mismatched_keys: list[dict[str, Any]] = []
    skipped_keys: list[dict[str, Any]] = []
    for source_key, value in state_dict.items():
        if not isinstance(source_key, str):
            skipped_keys.append({"source_key": repr(source_key), "target_key": "", "reason": "non_string_key"})
            continue
        target_key = _map_pretrained_key(source_key, source_format)
        if target_key not in target_state:
            unexpected_keys.append(source_key)
            continue
        target_value = target_state[target_key]
        if not isinstance(value, torch.Tensor) or not isinstance(target_value, torch.Tensor):
            skipped_keys.append({"source_key": source_key, "target_key": target_key, "reason": "non_tensor_value"})
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            mismatched_keys.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "source_shape": list(value.shape),
                    "target_shape": list(target_value.shape),
                    "reason": "shape_mismatch",
                }
            )
            continue
        prepared[target_key] = value
    return prepared, {
        "source_format": source_format,
        "unexpected_keys": unexpected_keys,
        "mismatched_keys": mismatched_keys,
        "skipped_keys": skipped_keys,
    }


def _looks_like_zhuyu_state_dict(state_dict: dict[str, Any]) -> bool:
    return any(
        isinstance(key, str)
        and (
            key.startswith("x_embedder.proj.")
            or key.startswith("y_embedder.embedding_table.")
            or ".mlp.fc1." in key
            or ".mlp.fc2." in key
        )
        for key in state_dict
    )


def _map_pretrained_key(key: str, source_format: str) -> str:
    if source_format != "zhuyu_meanflow_sit":
        return key
    if key.startswith("x_embedder.proj."):
        return "x_embedder." + key.removeprefix("x_embedder.proj.")
    if key == "y_embedder.embedding_table.weight":
        return "z_embedder.0.weight"
    key = key.replace(".mlp.fc1.", ".mlp.0.")
    key = key.replace(".mlp.fc2.", ".mlp.2.")
    return key
