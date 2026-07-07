"""PEFT LoRA + Gated Low-Rank Residual + Generic Embedding Bank for SAFA MeanFlow-SiT.

Expert final plan (2026-07-07):
- LoRA on adaLN_modulation Linear (native interface adaptation, not IP-Adapter).
  MeanFlow-SiT uses adaLN FiLM modulation; IP-Adapter (token cross-attn) is the
  wrong interface. We wrap each SiTBlock.adaLN_modulation[-1] Linear (and the
  FinalLayer.adaLN_modulation[-1] Linear) with base + scaling * B(A(x)).
- Gated low-rank residual on the condition vector:
      u = t_emb + r_emb + null_embed + generic_emb + gate * B_proj(A_proj(z))
  gate is a scalar parameter initialized to 0, so at step 0 the residual is 0
  and the backbone behaves identically to the frozen teacher with null cond.
- Generic embedding bank: 16 learned generic face embeddings (NOT text prompts).
  Each forward samples one random embedding per batch element. This is the
  expert's #6 final recommendation: avoid text encoder, use learned IDs.
- z is bypassed in the generic main loop (p_drop=1.0): forward with z=zeros,
  so delta_z = gate * B_proj(A_proj(0)) is constant (but gate regularizer
  still applies via gamma * ||B_proj(A_proj(z=0))||^2).
- z is injected only in the SAFA sparse loop (step ratio 12:1): forward with
  real z, delta_z = gate * B_proj(A_proj(z)) carries the identity signal.

Idempotent: wrap_backbone_with_peft_lora() can be called multiple times safely.
"""

from __future__ import annotations

import types
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Wraps an nn.Linear's weight/bias with a low-rank additive branch.

    Implementation detail: we do NOT keep the original Linear as a sub-module
    (which would rename state-dict keys to ``base.weight``). Instead, we lift the
    weight/bias Parameters onto ``self`` so the keys stay
    ``<prefix>.weight`` / ``<prefix>.bias`` — exactly matching the original
    Linear. The LoRA branch adds ``<prefix>.lora_a.weight`` / ``<prefix>.lora_b.weight``
    which are absent from the e15 checkpoint and loaded via strict=False.

    Output: ``F.linear(x, weight, bias) + scaling * lora_b(lora_a(x))``.
    At init: lora_b = 0 -> LoRA output = 0 -> identical to base.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear base, got {type(base_linear).__name__}")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)
        self.has_bias = base_linear.bias is not None
        # Lift weight/bias onto self (preserves state-dict keys).
        self.weight = nn.Parameter(base_linear.weight.detach().clone())
        if self.has_bias:
            self.bias = nn.Parameter(base_linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)
        # LoRA branches.
        self.lora_a = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias if self.has_bias else None)
        return base_out + self.scaling * self.lora_b(self.lora_a(x))


class GatedLowRankResidual(nn.Module):
    """delta = gate * B_proj(A_proj(z)).

    gate is a learnable scalar initialized to 0 (step 0 delta = 0).
    A_proj: z_dim -> rank (xavier), B_proj: rank -> hidden_size (xavier).
    """

    def __init__(self, z_dim: int, hidden_size: int, rank: int = 8):
        super().__init__()
        self.rank = int(rank)
        self.A_proj = nn.Linear(int(z_dim), self.rank, bias=False)
        self.B_proj = nn.Linear(self.rank, int(hidden_size), bias=False)
        nn.init.xavier_uniform_(self.A_proj.weight)
        nn.init.xavier_uniform_(self.B_proj.weight)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, z_dim] -> [B, hidden_size]
        return self.gate * self.B_proj(self.A_proj(z))


class GenericEmbeddingBank(nn.Module):
    """N learned generic face embeddings (replaces text prompt bank).

    Forward samples one random embedding index per batch element (uniform).
    Returns [B, hidden_size].
    """

    def __init__(self, num_embeddings: int = 16, hidden_size: int = 768):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.hidden_size = int(hidden_size)
        self.embeddings = nn.Embedding(self.num_embeddings, self.hidden_size)
        nn.init.normal_(self.embeddings.weight, std=0.02)

    def forward(self, batch_size: int, *, device, dtype) -> torch.Tensor:
        idx = torch.randint(0, self.num_embeddings, (batch_size,), device=device)
        return self.embeddings(idx).to(dtype=dtype)


# ---------------------------------------------------------------------------
# Backdrop wrap
# ---------------------------------------------------------------------------


def wrap_backbone_with_peft_lora(
    backbone: nn.Module,
    *,
    lora_rank: int = 8,
    lora_alpha: float = 1.0,
    z_dim: int | None = None,
    hidden_size: int | None = None,
    num_generic_embeddings: int = 16,
) -> nn.Module:
    """Attach LoRA + gated low-rank residual + generic bank to a MeanFlow-SiT backbone.

    Idempotent: detects ``_peft_lora_wrapped`` and returns immediately on re-entry.

    Side effects on the backbone:
    1. Each ``block.adaLN_modulation[-1]`` (the nn.Linear inside Sequential) is
       replaced by a LoRALinear wrapping the original Linear. The original
       weight/bias tensors are preserved inside LoRALinear.base.
    2. ``backbone.final_layer.adaLN_modulation[-1]`` is also wrapped (FiLM
       modulation on the final patch projection is condition-driven too).
    3. New sub-modules are attached:
       - ``backbone.gated_low_rank_z`` (GatedLowRankResidual)
       - ``backbone.generic_bank`` (GenericEmbeddingBank)
       - ``backbone._peft_lora_null_proj`` (nn.Linear(hidden_size, hidden_size, bias=False),
         identity-ish init, used to project the generator-level null_condition
         embedding into the backbone condition space; we don't rely on the
         backbone's own z_embedder because the expert plan disables it).
    4. ``backbone.forward`` is monkey-patched with the PEFT-LoRA forward.
    5. Base parameters are frozen; only LoRA / gated / generic / null_proj
       parameters have requires_grad=True.

    The original ``z_embedder`` is left in place (state-dict compatible) but is
    NOT called by the patched forward.
    """
    if getattr(backbone, "_peft_lora_wrapped", False):
        return backbone

    from safa.models.meanflow_sit import _modulate  # module-level helper

    # Infer dims from backbone sub-modules.
    if hidden_size is None:
        hidden_size = int(backbone.x_embedder.out_channels)
    if z_dim is None:
        z_dim = int(backbone.z_embedder[0].in_features)

    device = next(backbone.parameters()).device
    dtype = next(backbone.parameters()).dtype

    # 1. Attach gated low-rank + generic bank.
    backbone.gated_low_rank_z = GatedLowRankResidual(z_dim, hidden_size, rank=lora_rank).to(device=device, dtype=dtype)
    backbone.generic_bank = GenericEmbeddingBank(num_generic_embeddings, hidden_size).to(device=device, dtype=dtype)

    # 2. Wrap each block's adaLN_modulation[-1] Linear with LoRALinear.
    for block in backbone.blocks:
        mod_seq = block.adaLN_modulation
        if not isinstance(mod_seq, nn.Sequential) or len(mod_seq) == 0:
            raise RuntimeError(f"SiTBlock.adaLN_modulation is not a non-empty Sequential: {type(mod_seq)}")
        base_linear = mod_seq[-1]
        if not isinstance(base_linear, nn.Linear):
            raise RuntimeError(f"SiTBlock.adaLN_modulation[-1] is not nn.Linear: {type(base_linear)}")
        lora_wrapped = LoRALinear(base_linear, rank=lora_rank, alpha=lora_alpha).to(device=device, dtype=dtype)
        mod_seq[-1] = lora_wrapped

    # 2b. Wrap FinalLayer.adaLN_modulation[-1] too.
    final_seq = backbone.final_layer.adaLN_modulation
    if isinstance(final_seq, nn.Sequential) and len(final_seq) > 0 and isinstance(final_seq[-1], nn.Linear):
        final_base = final_seq[-1]
        final_lora = LoRALinear(final_base, rank=lora_rank, alpha=lora_alpha).to(device=device, dtype=dtype)
        final_seq[-1] = final_lora

    # 3. Patched forward.
    def peft_lora_forward(self, x, r, t, z):
        """PEFT-LoRA forward.

        condition = t_embedder(t) + r_embedder(horizon) + null_proj(null_embed)
                    + generic_bank(B) + gate * B_proj(A_proj(z))

        The original z_embedder is bypassed; z signal enters only through the
        gated low-rank residual on the condition vector. When the caller passes
        z=zeros (generic main loop), the residual is constant w.r.t. z but the
        gate regularizer (gamma * ||residual||^2) still applies via the loss.
        """
        self._validate_inputs(x, r, t, z)
        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        hidden = self.x_embedder(x).flatten(2).transpose(1, 2)
        hidden = hidden + self.pos_embed.to(device=device, dtype=dtype)

        horizon = (t - r).clamp_min(0.0)
        t_emb = self.t_embedder(t)
        r_emb = self.r_embedder(horizon)

        # generic bank: random embedding per sample
        generic_emb = self.generic_bank(B, device=device, dtype=dtype)

        # gated low-rank residual from z (z=zeros in generic step -> still a
        # function of A_proj(0) and B_proj, regularized by L_cond).
        delta_z = self.gated_low_rank_z(z)

        # null condition: the generator-level null_condition embedding is a
        # z_dim-sized vector; we project it to hidden_size via a fresh linear
        # so that the condition has a learnable "null" anchor independent of z.
        # If the generator has no null_condition, fall back to zero.
        null_embed_attr = getattr(self, "_peft_lora_null_embed", None)
        if null_embed_attr is not None:
            null_embed_b = null_embed_attr.unsqueeze(0).expand(B, -1).to(dtype=dtype)
        else:
            null_embed_b = torch.zeros(B, hidden_size, device=device, dtype=dtype)

        condition = t_emb + r_emb + null_embed_b + generic_emb + delta_z

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

    backbone.forward = types.MethodType(peft_lora_forward, backbone)

    # 4. Add a learnable null embedding (hidden_size, ) at backbone level. This
    #    is a fresh parameter, not present in original state-dict (loaded into
    #    the new key ``_peft_lora_null_embed``).
    with torch.no_grad():
        null_embed_param = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))
        nn.init.normal_(null_embed_param, std=0.02)
    backbone._peft_lora_null_embed = null_embed_param

    backbone._peft_lora_wrapped = True

    # 5. Freeze base, unfreeze adapter params.
    for name, param in backbone.named_parameters():
        is_lora = ("lora_a" in name) or ("lora_b" in name)
        is_gated = name.startswith("gated_low_rank_z.")
        is_generic = name.startswith("generic_bank.")
        is_null = name == "_peft_lora_null_embed"
        param.requires_grad_(is_lora or is_gated or is_generic or is_null)

    return backbone


# ---------------------------------------------------------------------------
# Helpers exposed to the runner
# ---------------------------------------------------------------------------


def collect_peft_lora_params(backbone: nn.Module) -> list[nn.Parameter]:
    """All trainable adapter parameters (for explicit optimizer construction)."""
    return [p for _, p in backbone.named_parameters() if p.requires_grad]


def adapter_param_summary(backbone: nn.Module) -> dict[str, int]:
    """Return trainable / frozen / total parameter counts for logging."""
    trainable = 0
    frozen = 0
    for _, p in backbone.named_parameters():
        n = int(p.numel())
        if p.requires_grad:
            trainable += n
        else:
            frozen += n
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def compute_l_cond(backbone: nn.Module, z_sample: torch.Tensor) -> torch.Tensor:
    """L_cond = ||B_proj(A_proj(z))||^2 + lambda_g * gate^2.

    The expert plan defines L_cond as a regularizer on the adapter energy.
    We return the raw ||delta||^2 (caller scales by gamma); the gate term
    lambda_g * gate^2 is added separately by the runner so that lambda_g is
    configured at the runner level.
    """
    delta = backbone.gated_low_rank_z(z_sample)  # [B, hidden]
    return delta.square().sum(dim=-1).mean()
