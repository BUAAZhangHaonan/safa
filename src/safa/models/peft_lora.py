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

    Phase 0.3 (b): when ``num_embeddings == 1`` the bank degenerates to a
    single *shared* embedding (forward always returns the same vector). The
    caller may pass ``init_from_null_embed`` (a hidden_size tensor) to copy
    initial weights from the null embedding instead of random init.
    """

    def __init__(
        self,
        num_embeddings: int = 16,
        hidden_size: int = 768,
        init_from_null_embed: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.hidden_size = int(hidden_size)
        self.embeddings = nn.Embedding(self.num_embeddings, self.hidden_size)
        if init_from_null_embed is not None and self.num_embeddings >= 1:
            with torch.no_grad():
                init_vec = init_from_null_embed.detach().clone().to(
                    dtype=self.embeddings.weight.dtype
                )
                if init_vec.shape[-1] != self.hidden_size:
                    raise ValueError(
                        f"init_from_null_embed last-dim {init_vec.shape[-1]} != hidden_size {self.hidden_size}"
                    )
                # Broadcast the source (shape [hidden_size]) across all bank slots
                # so single_shared and N-bank both init to the null anchor.
                self.embeddings.weight.copy_(init_vec.unsqueeze(0).expand(self.num_embeddings, -1))
        else:
            nn.init.normal_(self.embeddings.weight, std=0.02)

    def forward(self, batch_size: int, *, device, dtype) -> torch.Tensor:
        if self.num_embeddings <= 1:
            # single shared: always index 0
            idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
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
    enable_lora: bool = True,
    enable_gated_low_rank: bool = True,
    enable_generic_bank: bool = True,
    lora_blocks: str = "all",
    generic_mode: str = "bank",
    freeze_null_embed: bool = False,
) -> nn.Module:
    """Attach LoRA + gated low-rank residual + generic bank to a MeanFlow-SiT backbone.

    Idempotent: detects ``_peft_lora_wrapped`` and returns immediately on re-entry.

    Phase 0.2 ablation knobs (expert 2026-07-07 strict order):
    - ``enable_lora`` (default True): when False, block adaLN_modulation[-1]
      and FinalLayer.adaLN_modulation[-1] keep their original nn.Linear; no
      LoRA params exist. Used for the "bank-only" arm.
    - ``enable_gated_low_rank`` (default True): when False, the
      GatedLowRankResidual is still attached (for state-dict shape compat) but
      A_proj, B_proj and gate are zeroed and frozen, so delta_z == 0. Used for
      the LoRA-only arms.
    - ``enable_generic_bank`` (default True): when False, the
      GenericEmbeddingBank is attached but its embeddings are zeroed and
      frozen, so generic_emb == 0. Used for the LoRA-only arms.
    - ``lora_blocks``: "all" wraps every block; "last_third" wraps only the
      last len(blocks)//3 blocks (still wraps FinalLayer). Used for the
      LoRA-only part-blocks arm (T-LoRA high-timestep-overfit analogy).

    Phase 0.3 (a)/(b) knobs (expert 2026-07-07 strict (a)+(b) parallel test):
    - ``generic_mode`` (default "bank"): one of "bank" / "null" / "single_shared".
      * "bank": original Phase 0 behaviour, ``num_generic_embeddings`` (e.g. 16)
        learned embeddings, random-sample one per batch element.
      * "null" (Phase 0.3 a): the generic bank is NOT created
        (``backbone.generic_bank = None``). The generic step condition is just
        ``t_emb + r_emb + null_embed (frozen) + delta_z`` — no learnable generic
        embedding is added. ``_peft_lora_null_embed`` is frozen in this mode so
        the only generic-step learnable contributions are LoRA + delta_z.
      * "single_shared" (Phase 0.3 b): the bank degenerates to ONE shared
        embedding (always index 0), initialized by copying
        ``_peft_lora_null_embed``. At step 0 the generic contribution is the
        null anchor itself; if the expert's "gate=0" intuition holds, the
        embedding should drift very little and behave close to (a).
    - ``freeze_null_embed`` (default False): when True, the learnable null
      anchor ``_peft_lora_null_embed`` is frozen (requires_grad=False). The
      Phase 0.3 (a) arm turns this on so that the ONLY generic-step learnable
      contributions are LoRA + delta_z. Phase 0.3 (b) leaves it trainable
      (matching Phase 0 baseline) so the only difference vs Phase 0 is the
      bank cardinality (1 vs 16) and init (null vs random).

    Side effects on the backbone:
    1. Each selected ``block.adaLN_modulation[-1]`` (the nn.Linear inside
       Sequential) is replaced by a LoRALinear wrapping the original Linear.
       The original weight/bias tensors are preserved inside LoRALinear.
    2. ``backbone.final_layer.adaLN_modulation[-1]`` is also wrapped when
       ``enable_lora`` is True (FiLM modulation on the final patch projection
       is condition-driven too).
    3. New sub-modules are attached:
       - ``backbone.gated_low_rank_z`` (GatedLowRankResidual)
       - ``backbone.generic_bank`` (GenericEmbeddingBank)
       - ``backbone._peft_lora_null_embed`` (Parameter, hidden_size)
    4. ``backbone.forward`` is monkey-patched with the PEFT-LoRA forward.
    5. Base parameters are frozen; only LoRA / gated / generic / null_embed
       parameters have requires_grad=True (and only for components whose
       ``enable_*`` flag is True).

    The original ``z_embedder`` is left in place (state-dict compatible) but is
    NOT called by the patched forward.
    """
    if getattr(backbone, "_peft_lora_wrapped", False):
        return backbone

    from safa.models.meanflow_sit import _modulate  # module-level helper

    # Validate lora_blocks.
    if lora_blocks not in ("all", "last_third"):
        raise ValueError(f"lora_blocks must be 'all' or 'last_third', got {lora_blocks!r}")

    # Validate generic_mode (Phase 0.3 a/b knob).
    if generic_mode not in ("bank", "null", "single_shared"):
        raise ValueError(
            f"generic_mode must be 'bank' / 'null' / 'single_shared', got {generic_mode!r}"
        )

    # Infer dims from backbone sub-modules.
    if hidden_size is None:
        hidden_size = int(backbone.x_embedder.out_channels)
    if z_dim is None:
        z_dim = int(backbone.z_embedder[0].in_features)

    device = next(backbone.parameters()).device
    dtype = next(backbone.parameters()).dtype

    # 1. Attach gated low-rank residual (always, for state-dict shape compat).
    backbone.gated_low_rank_z = GatedLowRankResidual(z_dim, hidden_size, rank=lora_rank).to(device=device, dtype=dtype)

    # 1b. Create the learnable null anchor FIRST so single_shared bank can
    #     initialize from it. The original code created this at the very end;
    #     we move it here to support Phase 0.3 (b) single_shared init.
    with torch.no_grad():
        null_embed_param = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))
        nn.init.normal_(null_embed_param, std=0.02)
    backbone._peft_lora_null_embed = null_embed_param

    # 1c. Build the generic bank based on generic_mode (Phase 0.3 a/b).
    if generic_mode == "null":
        # (a) main arm: NO bank. generic step condition = t_emb + r_emb +
        #     null_embed (frozen) + delta_z. We still attach a GenericEmbeddingBank
        #     module with num_embeddings=1 and zeroed/frozen weights so that
        #     downstream code (state-dict shape, frozen-step counting) is
        #     uniform, but the patched forward detects generic_mode and skips
        #     adding its output to the condition.
        backbone.generic_bank = GenericEmbeddingBank(1, hidden_size).to(device=device, dtype=dtype)
        with torch.no_grad():
            backbone.generic_bank.embeddings.weight.zero_()
        backbone._peft_generic_mode = "null"
    elif generic_mode == "single_shared":
        # (b) validation arm: bank of size 1, init = null_embed_param.
        backbone.generic_bank = GenericEmbeddingBank(
            1,
            hidden_size,
            init_from_null_embed=backbone._peft_lora_null_embed.detach().clone(),
        ).to(device=device, dtype=dtype)
        backbone._peft_generic_mode = "single_shared"
    else:
        # "bank" (Phase 0/0.1/0.2): N learned embeddings, random init.
        backbone.generic_bank = GenericEmbeddingBank(num_generic_embeddings, hidden_size).to(device=device, dtype=dtype)
        backbone._peft_generic_mode = "bank"

    # 2. Wrap selected blocks' adaLN_modulation[-1] Linear with LoRALinear.
    if enable_lora:
        n_blocks = len(backbone.blocks)
        if lora_blocks == "last_third":
            # Last third (rounded down). For 12 blocks -> indices 8,9,10,11.
            start_idx = n_blocks - max(1, n_blocks // 3)
            wrap_indices = set(range(start_idx, n_blocks))
        else:
            wrap_indices = set(range(n_blocks))

        for i, block in enumerate(backbone.blocks):
            if i not in wrap_indices:
                continue
            mod_seq = block.adaLN_modulation
            if not isinstance(mod_seq, nn.Sequential) or len(mod_seq) == 0:
                raise RuntimeError(f"SiTBlock.adaLN_modulation is not a non-empty Sequential: {type(mod_seq)}")
            base_linear = mod_seq[-1]
            if not isinstance(base_linear, nn.Linear):
                raise RuntimeError(f"SiTBlock.adaLN_modulation[-1] is not nn.Linear: {type(base_linear)}")
            lora_wrapped = LoRALinear(base_linear, rank=lora_rank, alpha=lora_alpha).to(device=device, dtype=dtype)
            mod_seq[-1] = lora_wrapped

        # 2b. Wrap FinalLayer.adaLN_modulation[-1] too (when LoRA enabled).
        final_seq = backbone.final_layer.adaLN_modulation
        if isinstance(final_seq, nn.Sequential) and len(final_seq) > 0 and isinstance(final_seq[-1], nn.Linear):
            final_base = final_seq[-1]
            final_lora = LoRALinear(final_base, rank=lora_rank, alpha=lora_alpha).to(device=device, dtype=dtype)
            final_seq[-1] = final_lora

    # 2c. Disable gated_low_rank / generic_bank by zeroing+freezing if requested.
    if not enable_gated_low_rank:
        with torch.no_grad():
            backbone.gated_low_rank_z.A_proj.weight.zero_()
            backbone.gated_low_rank_z.B_proj.weight.zero_()
            backbone.gated_low_rank_z.gate.zero_()
    if not enable_generic_bank:
        with torch.no_grad():
            backbone.generic_bank.embeddings.weight.zero_()

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

        # generic bank contribution depends on generic_mode (Phase 0.3 a/b):
        # - "bank" / "single_shared": sample 1 embedding per batch element
        #   (single_shared always returns the same index-0 vector).
        # - "null": skip the bank entirely so generic_emb == 0; the generic
        #   step condition is t_emb + r_emb + null_embed (frozen) + delta_z.
        generic_mode_attr = getattr(self, "_peft_generic_mode", "bank")
        if generic_mode_attr == "null":
            generic_emb = torch.zeros(B, hidden_size, device=device, dtype=dtype)
        else:
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

    backbone._peft_lora_wrapped = True
    # Record ablation flags on the backbone for downstream introspection
    # (logging / param counting / debugging).
    backbone._peft_lora_config = {
        "enable_lora": bool(enable_lora),
        "enable_gated_low_rank": bool(enable_gated_low_rank),
        "enable_generic_bank": bool(enable_generic_bank),
        "lora_blocks": str(lora_blocks),
        "lora_rank": int(lora_rank),
        "lora_alpha": float(lora_alpha),
        "num_generic_embeddings": int(num_generic_embeddings),
        "generic_mode": str(generic_mode),
        "freeze_null_embed": bool(freeze_null_embed),
    }

    # 5. Freeze base, unfreeze adapter params based on enabled flags.
    #    Phase 0.3 (a) "null" arm: null_embed is frozen (freeze_null_embed=True
    #    AND generic_mode=="null"); generic_bank is also frozen (zeroed). So
    #    the only generic-step learnable contributions are LoRA + delta_z.
    #    Phase 0.3 (b) "single_shared" arm: null_embed stays trainable (matching
    #    Phase 0 baseline) and the bank has 1 learnable embedding.
    for name, param in backbone.named_parameters():
        is_lora = ("lora_a" in name) or ("lora_b" in name)
        is_gated = name.startswith("gated_low_rank_z.")
        is_generic = name.startswith("generic_bank.")
        is_null = name == "_peft_lora_null_embed"
        train_lora = is_lora and enable_lora
        train_gated = is_gated and enable_gated_low_rank
        train_generic = is_generic and enable_generic_bank and (generic_mode != "null")
        train_null = is_null and not freeze_null_embed
        param.requires_grad_(train_lora or train_gated or train_generic or train_null)

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
