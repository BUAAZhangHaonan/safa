"""IP-Adapter cross-attention + wrap helpers for SAFA MeanFlow-SiT.

This module adds decoupled cross-attention adapters to a MeanFlow-SiT backbone
without modifying the original meanflow_sit.py source. We attach the adapter
as a sub-module of each targeted SiTBlock and monkey-patch the backbone's
forward to call the adapter after self-attention. State-dict keys remain
compatible with the original backbone (ip_adapter keys are simply extra).

Stays as a separate file to honor the "do not mess with the original repo"
constraint: only a single dispatch line is added to g_loop.py.
"""

import types
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class IPAdapterCrossAttention(nn.Module):
    """Decoupled cross-attention that takes z0 as Key/Value source.

    Standard IP-Adapter design (Ye et al. 2308.06721):
        Q from latent tokens (hidden), K/V from projected z0.
        Output is residual, gated by a learnable scalar (zero-init for stable start).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        z_dim: int = 512,
        num_z_tokens: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_z_tokens = num_z_tokens
        self.scale = self.head_dim ** -0.5

        # Project scalar z0 (z_dim) to M tokens of hidden_size.
        self.z_proj = nn.Linear(z_dim, num_z_tokens * hidden_size, bias=True)
        self.norm_z = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1.0e-6)

        # Standard QKV projections (K/V operate on z tokens, Q on latent).
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=True)

        # Small-init gate (0.1): ensures adapter params receive non-zero gradient at step 0.
        self.gate = nn.Parameter(torch.full((1,), 0.1))

        # Init weights: xavier for projections, zero for output bias and gate.
        for layer in (self.z_proj, self.to_q, self.to_k, self.to_v, self.to_out):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, N, hidden_size] latent tokens (Q source).
            z: [B, z_dim] z0 embedding.
        Returns:
            [B, N, hidden_size] residual contribution (multiplied by gate).
        """
        B = x.shape[0]
        H = self.hidden_size

        z_tokens = self.z_proj(z).reshape(B, self.num_z_tokens, H)
        z_tokens = self.norm_z(z_tokens)

        q = self.to_q(x)
        k = self.to_k(z_tokens)
        v = self.to_v(z_tokens)

        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, self.num_z_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, self.num_z_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Native attention (not F.scaled_dot_product_attention) — MeanFlow's JVP
        # mode triggers forward AD, which SDPA's efficient backend does not support.
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_logits.softmax(dim=-1)
        out = attn_weights @ v

        out = out.transpose(1, 2).reshape(B, -1, H)
        out = self.to_out(out)
        return self.gate * out


def _block_forward_peft(block, x, condition, z, modulate_fn):
    """Replicates SiTBlock.forward and inserts ip_adapter after self-attn.

    Args:
        block: original SiTBlock instance (with self-attn, mlp, adaLN_modulation).
        x: [B, N, hidden] latent tokens.
        condition: [B, hidden] global condition (t + r + z embedding).
        z: [B, z_dim] z0 embedding for the cross-attention.
        modulate_fn: meanflow_sit._modulate helper (shift, scale modulation).
    """
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        block.adaLN_modulation(condition).chunk(6, dim=-1)
    )
    attn_input = modulate_fn(block.norm1(x), shift_msa, scale_msa)
    attn_output = block.attn(attn_input)
    x = x + gate_msa.unsqueeze(1) * attn_output

    # IP-Adapter cross-attention injection (zero-init gate, identity at step 0).
    ip_adapter = getattr(block, "ip_adapter", None)
    if ip_adapter is not None:
        x = x + ip_adapter(x, z)

    mlp_input = modulate_fn(block.norm2(x), shift_mlp, scale_mlp)
    x = x + gate_mlp.unsqueeze(1) * block.mlp(mlp_input)
    return x


def wrap_backbone_with_ip_adapter(
    backbone: nn.Module,
    ip_adapter_layers: Iterable[int],
    num_z_tokens: int = 4,
    *,
    hidden_size: int | None = None,
    z_dim: int | None = None,
    num_heads: int | None = None,
) -> nn.Module:
    """Attach IP-Adapter cross-attention to selected SiTBlocks and patch forward.

    Args:
        backbone: MeanFlowSiTBackbone instance (created by build_meanflow_sit_generator).
        ip_adapter_layers: which block indices (0-based) receive an adapter.
        num_z_tokens: how many z tokens to project from z0 (IP-Adapter default 4).
        hidden_size / z_dim / num_heads: optional explicit values; if None, inferred
            from backbone sub-modules (works for MeanFlowSiTBackbone).
    """
    from safa.models.meanflow_sit import _modulate  # module-level helper

    # Infer dimensions from backbone sub-modules if not provided.
    if hidden_size is None:
        # x_embedder is Conv2d(in_channels, hidden_size, ...).
        hidden_size = backbone.x_embedder.out_channels
    if z_dim is None:
        # z_embedder[0] is Linear(z_dim, hidden_size).
        z_dim = backbone.z_embedder[0].in_features
    if num_heads is None:
        # Heuristic: SiT uses head_dim=64 by default, so num_heads = hidden_size // 64.
        num_heads = max(1, hidden_size // 64)
    target_layers = set(int(i) for i in ip_adapter_layers)

    # Determine backbone device/dtype for adapter placement.
    backbone_device = next(backbone.parameters()).device
    backbone_dtype = next(backbone.parameters()).dtype

    # Attach ip_adapter to each block (None for non-target layers).
    for i, block in enumerate(backbone.blocks):
        if i in target_layers:
            adapter = IPAdapterCrossAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                z_dim=z_dim,
                num_z_tokens=num_z_tokens,
            ).to(device=backbone_device, dtype=backbone_dtype)
            block.ip_adapter = adapter  # nn.Module.__setattr__ registers it
        else:
            # Ensure attribute exists (for forward dispatch) but is None.
            block.ip_adapter = None

    # Patch backbone.forward to call PEFT block forward (closure captures _modulate).
    def peft_forward(self, x, r, t, z):
        self._validate_inputs(x, r, t, z)
        hidden = self.x_embedder(x).flatten(2).transpose(1, 2)
        hidden = hidden + self.pos_embed.to(device=hidden.device, dtype=hidden.dtype)
        horizon = (t - r).clamp_min(0.0)
        condition = self.t_embedder(t) + self.r_embedder(horizon) + self.z_embedder(z)
        for i, block in enumerate(self.blocks):
            hidden = _block_forward_peft(block, hidden, condition, z, _modulate)
        patches = self.final_layer(hidden, condition)
        return self._unpatchify(patches)

    backbone.forward = types.MethodType(peft_forward, backbone)

    # Freeze base, unfreeze ip_adapter.
    for name, param in backbone.named_parameters():
        if "ip_adapter" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return backbone


def collect_ip_adapter_params(backbone: nn.Module) -> list[nn.Parameter]:
    """Return all trainable ip_adapter parameters for the optimizer."""
    return [p for n, p in backbone.named_parameters() if "ip_adapter" in n and p.requires_grad]


def adapter_output_norm(backbone: nn.Module, sample_inputs: dict) -> torch.Tensor:
    """L_cond: average ||ip_adapter(x, z)||_2 across all adapters.

    Used as the conditioning regularizer (expert L_cond). Defined as the
    mean L2 norm of the residual contribution (post gate) across adapters,
    averaged over batch and adapters.
    """
    # Run forward with hooks to capture adapter outputs.
    adapter_outputs: list[torch.Tensor] = []

    def make_hook(adapter):
        def hook(module, inputs, output):
            adapter_outputs.append(output)

        return hook

    handles = []
    for block in backbone.blocks:
        ip_adapter = getattr(block, "ip_adapter", None)
        if ip_adapter is not None:
            handles.append(ip_adapter.register_forward_hook(make_hook(ip_adapter)))

    # Forward to populate hook outputs. Use the same inputs as the main forward.
    try:
        with torch.no_grad():
            backbone(
                sample_inputs["x"],
                sample_inputs["r"],
                sample_inputs["t"],
                sample_inputs["z"],
            )
    finally:
        for h in handles:
            h.remove()

    if not adapter_outputs:
        return torch.tensor(0.0)

    norms = torch.stack([o.norm(dim=-1).mean() for o in adapter_outputs])
    return norms.mean()
"""Condition-MLP adapter additions (appended to ip_adapter.py).

Simple MLP that produces a delta added to the condition vector before the
SiTBlocks. Theoretical ceiling is higher than IP-Adapter because we modify
the condition main path (FiLM) rather than only the residual after self-attn.
"""

import types
from typing import Iterable

import torch
import torch.nn as nn


class ConditionMLPAdapter(nn.Module):
    """Simple MLP that produces a delta added to the condition vector.

    Injected at: condition = t_embed(t) + r_embed(horizon) + z_embed(z) + adapter(z)
    """

    def __init__(self, z_dim: int = 512, hidden_size: int = 768, mlp_dim: int = 768):
        super().__init__()
        self.norm = nn.LayerNorm(z_dim, elementwise_affine=False, eps=1e-6)
        self.fc1 = nn.Linear(z_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.fc3 = nn.Linear(mlp_dim, hidden_size)
        self.act = nn.GELU()
        # Hidden layers: xavier. Last layer: small normal (NOT zero — zero would
        # block gradient just like gate=0 did in v1/v2 IP-Adapter).
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.fc3.weight, std=0.01)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.norm(z)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        return self.fc3(h)  # [B, hidden_size]


def wrap_backbone_with_condition_mlp(
    backbone: nn.Module,
    *,
    z_dim: int | None = None,
    hidden_size: int | None = None,
    mlp_dim: int | None = None,
) -> nn.Module:
    """Attach ConditionMLPAdapter and patch backbone forward to inject delta into condition."""
    if getattr(backbone, "_cond_mlp_wrapped", False):
        return backbone
    from safa.models.meanflow_sit import _modulate

    if hidden_size is None:
        hidden_size = backbone.x_embedder.out_channels
    if z_dim is None:
        z_dim = backbone.z_embedder[0].in_features
    if mlp_dim is None:
        mlp_dim = hidden_size

    adapter = ConditionMLPAdapter(z_dim=z_dim, hidden_size=hidden_size, mlp_dim=mlp_dim)
    device = next(backbone.parameters()).device
    dtype = next(backbone.parameters()).dtype
    adapter = adapter.to(device=device, dtype=dtype)
    backbone.cond_mlp_adapter = adapter

    def mlp_forward(self, x, r, t, z):
        self._validate_inputs(x, r, t, z)
        hidden = self.x_embedder(x).flatten(2).transpose(1, 2)
        hidden = hidden + self.pos_embed.to(device=hidden.device, dtype=hidden.dtype)
        horizon = (t - r).clamp_min(0.0)
        condition = self.t_embedder(t) + self.r_embedder(horizon) + self.z_embedder(z)
        # === MLP adapter injection (condition main path) ===
        condition = condition + self.cond_mlp_adapter(z)
        for block in self.blocks:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                block.adaLN_modulation(condition).chunk(6, dim=-1)
            )
            attn_input = _modulate(block.norm1(hidden), shift_msa, scale_msa)
            attn_output = block.attn(attn_input)
            hidden = hidden + gate_msa.unsqueeze(1) * attn_output
            mlp_input = _modulate(block.norm2(hidden), shift_mlp, scale_mlp)
            hidden = hidden + gate_mlp.unsqueeze(1) * block.mlp(mlp_input)
        patches = self.final_layer(hidden, condition)
        return self._unpatchify(patches)

    backbone.forward = types.MethodType(mlp_forward, backbone)
    backbone._cond_mlp_wrapped = True

    # Freeze base, leave adapter trainable.
    for name, param in backbone.named_parameters():
        if "cond_mlp_adapter" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    return backbone


def collect_cond_mlp_params(backbone: nn.Module) -> list[nn.Parameter]:
    """Trainable params of the ConditionMLPAdapter."""
    return [
        p
        for n, p in backbone.named_parameters()
        if "cond_mlp_adapter" in n and p.requires_grad
    ]
