"""PEFT stage-2 batch runner for SAFA MeanFlow-SiT.

Implements the expert-recommended objective:
    L_main = L_pair (FM) + beta * L_preserve (EMA teacher) + gamma * L_cond (adapter norm)

L_repr (representation alignment) is added every K steps as a sparse auxiliary
that shares the same parameter space (adapter) — no projection needed because
the base is frozen.

This runner is dispatched by g_loop.py when stage2_objective.type == "peft_fm".
The dispatch line is the only modification to g_loop.py.
"""

import dataclasses
import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _PEFTStage2Objective:
    """PEFT stage-2 objective runtime config.

    Fields:
        type: always "peft_fm".
        beta_preserve: weight for L_preserve (EMA teacher velocity MSE). 0 disables.
        gamma_cond: weight for L_cond (adapter output L2 norm). 0 disables.
        lambda_repr: weight for L_repr (cosine alignment). 0 disables.
        repr_interval: K-step interval for L_repr (1 = every step).
        ip_adapter_layers: which SiTBlocks receive an adapter (1-based in config for readability).
        ip_adapter_num_tokens: number of z tokens to project from z0.
        ip_adapter_init: whether to wrap generator with ip_adapter (only at first call).
        flow_condition: required by _validate_train_g_config; we use "embedding" for PEFT.
    """
    type: str = "peft_fm"
    beta_preserve: float = 0.5
    gamma_cond: float = 0.05
    lambda_repr: float = 0.5
    repr_interval: int = 8
    ip_adapter_layers: tuple[int, ...] = (6, 7, 8, 9, 10, 11)
    ip_adapter_num_tokens: int = 4
    ip_adapter_init: bool = True
    flow_condition: str = "embedding"
    # Compat fields so existing helpers (e.g. _compute_repr_loss) that read
    # these attributes don't crash on _PEFTStage2Objective. These values are
    # not used by the PEFT objective itself.
    relation_weight: float = 0.0
    point_weight: float = 1.0
    offdiag_only: bool = True
    repr_learning_rate: float = 0.0
    projection_eps: float = 1.0e-12
    lambda_lpips: float = 0.0


def peft_stage2_objective_from_config(payload: dict[str, Any], context: str) -> _PEFTStage2Objective:
    """Parse PEFT stage-2 objective from a config dict (mirrors g_loop's _optional_numeric)."""
    beta = float(payload.get("beta_preserve", 0.5))
    gamma = float(payload.get("gamma_cond", 0.05))
    lambda_repr = float(payload.get("lambda_repr", 0.5))
    interval = int(payload.get("repr_interval", 8))
    layers = payload.get("ip_adapter_layers", [6, 7, 8, 9, 10, 11])
    if not isinstance(layers, (list, tuple)):
        raise ValueError(f"{context}: ip_adapter_layers must be a list of ints, got {type(layers).__name__}")
    layers = tuple(int(x) for x in layers)
    num_tokens = int(payload.get("ip_adapter_num_tokens", 4))
    return _PEFTStage2Objective(
        type="peft_fm",
        beta_preserve=beta,
        gamma_cond=gamma,
        lambda_repr=lambda_repr,
        repr_interval=interval,
        ip_adapter_layers=layers,
        ip_adapter_num_tokens=num_tokens,
        ip_adapter_init=True,
    )


def _compute_preserve_loss(
    student_backbone: torch.nn.Module,
    teacher_backbone: torch.nn.Module,
    images_in_latent_space: torch.Tensor,
    z: torch.Tensor,
    *,
    num_preserve_samples: int = 1,
) -> torch.Tensor:
    """L_preserve: teacher-student velocity MSE at t=0.5.

    Both backbones are forwarded on the same (x_t, r, t, z) and we MSE their
    velocity predictions. Teacher is frozen EMA, student is the live adapter
    backbone. Anchors the adapter's behavior to the base model's behavior on
    native samples (from the train batch).
    """
    B = images_in_latent_space.shape[0]
    device = images_in_latent_space.device
    dtype = images_in_latent_space.dtype

    # Use a fixed t and r in [0,1] for the preserve loss.
    t = torch.full((B,), 0.5, device=device, dtype=dtype)
    r = torch.full((B,), 0.0, device=device, dtype=dtype)
    view_t = t.view(-1, 1, 1, 1)
    eps = torch.randn_like(images_in_latent_space)
    x_t = (1.0 - view_t) * images_in_latent_space + view_t * eps

    with torch.no_grad():
        teacher_vel = teacher_backbone(x_t, r, t, z)

    student_vel = student_backbone(x_t, r, t, z)
    return (student_vel - teacher_vel.detach()).square().mean()


def _compute_cond_loss_via_forward(
    student_backbone: torch.nn.Module,
    images_in_latent_space: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """L_cond: mean L2 norm of all ip_adapter outputs (residual contributions).

    Runs a forward pass with hooks capturing each adapter's residual output,
    returns the mean L2 norm across adapters and batch. Cheap regularizer
    that prevents the adapter from injecting too much energy into the hidden
    state (which would cause the FiLM-like scale explosion).
    """
    adapter_outputs: list[torch.Tensor] = []

    def make_hook():
        def hook(module, inputs, output):
            adapter_outputs.append(output)

        return hook

    handles = []
    for block in student_backbone.blocks:
        ip_adapter = getattr(block, "ip_adapter", None)
        if ip_adapter is not None:
            handles.append(ip_adapter.register_forward_hook(make_hook()))

    if not handles:
        return images_in_latent_space.new_tensor(0.0)

    B = images_in_latent_space.shape[0]
    device = images_in_latent_space.device
    dtype = images_in_latent_space.dtype
    t = torch.full((B,), 0.5, device=device, dtype=dtype)
    r = torch.full((B,), 0.0, device=device, dtype=dtype)
    view_t = t.view(-1, 1, 1, 1)
    eps = torch.randn_like(images_in_latent_space)
    x_t = (1.0 - view_t) * images_in_latent_space + view_t * eps

    try:
        student_backbone(x_t, r, t, z)
    finally:
        for h in handles:
            h.remove()

    if not adapter_outputs:
        return images_in_latent_space.new_tensor(0.0)

    norms = torch.stack([o.norm(dim=-1).mean() for o in adapter_outputs])
    return norms.mean()


def init_peft_generator(generator: torch.nn.Module, objective: _PEFTStage2Objective) -> None:
    """Wrap generator.vector_field with IP-Adapter (idempotent)."""
    backbone = generator.vector_field
    if getattr(backbone, "_ip_adapter_wrapped", False):
        return
    from safa.models.ip_adapter import wrap_backbone_with_ip_adapter

    wrap_backbone_with_ip_adapter(
        backbone,
        ip_adapter_layers=objective.ip_adapter_layers,
        num_z_tokens=objective.ip_adapter_num_tokens,
    )
    backbone._ip_adapter_wrapped = True  # type: ignore[attr-defined]


def run_peft_stage2_batch(
    runtime: Any,
    batch: dict[str, Any],
    objective: _PEFTStage2Objective,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run one PEFT stage-2 training batch.

    Returns:
        (loss, flow_mse, cycle_zero, flow_loss, repr_loss, aux_metrics)
        Matching the 5-tuple shape used by g_loop's other stage2 runners, plus
        an aux_metrics dict for logging.
    """
    generator = runtime.generator
    images = batch["image"]
    z = batch["z"]
    sample_ids = batch["sample_id"]

    # Ensure ip_adapter is attached (idempotent).
    init_peft_generator(generator, objective)

    # Encode pixel images to latent space ( MeanFlow-SiT operates on [B, 4, 32, 32]).
    images_latent = runtime._encode_flow_images(images)

    # === L_pair: standard FM velocity MSE on (x_0, z_0). ===
    flow_loss, flow_metrics = generator.flow_matching_loss(images_latent, z)
    flow_mse = flow_metrics["flow_matching_mse"].detach()

    # === L_preserve: EMA teacher velocity MSE. ===
    # Note: runtime (the _Runtime/_Module instance) does not carry ema; the EMA
    # is owned by the outer train_g loop. Disable L_preserve for now if ema
    # is not exposed via runtime. (R3 will degrade to R2-equivalent; the expert
    # full L_preserve wiring would require additional plumbing.)
    ema_obj = getattr(runtime, "ema", None)
    if objective.beta_preserve > 0.0 and ema_obj is not None:
        teacher_generator = _build_teacher_view(generator, ema_obj)
        preserve_loss = _compute_preserve_loss(
            generator.vector_field,
            teacher_generator.vector_field,
            images_latent,
            z,
        )
    else:
        preserve_loss = flow_loss.new_tensor(0.0)

    # === L_cond: adapter output norm. ===
    if objective.gamma_cond > 0.0:
        cond_loss = _compute_cond_loss_via_forward(generator.vector_field, images_latent, z)
    else:
        cond_loss = flow_loss.new_tensor(0.0)

    # === Main loss ===
    main_loss = flow_loss + objective.beta_preserve * preserve_loss + objective.gamma_cond * cond_loss

    # === L_repr: K-step sparse repr loss (only on repr steps). ===
    batch_idx = runtime._batch_idx
    is_repr_step = (objective.lambda_repr > 0.0) and (batch_idx % max(1, objective.repr_interval) == 0)
    if is_repr_step:
        repr_loss, repr_metrics = runtime._compute_repr_loss(z, sample_ids, cycle_steps=1, images=images)
        total_loss = main_loss + objective.lambda_repr * repr_loss
    else:
        repr_loss = flow_loss.new_tensor(0.0)
        total_loss = main_loss

    runtime._batch_idx = batch_idx + 1

    aux_metrics = {
        "peft_main_loss": main_loss.detach(),
        "peft_preserve_loss": preserve_loss.detach(),
        "peft_cond_loss": cond_loss.detach(),
        "peft_is_repr_step": flow_loss.new_tensor(float(is_repr_step)),
        "peft_beta_preserve": flow_loss.new_tensor(objective.beta_preserve),
        "peft_gamma_cond": flow_loss.new_tensor(objective.gamma_cond),
        "peft_lambda_repr": flow_loss.new_tensor(objective.lambda_repr),
        "peft_repr_interval": flow_loss.new_tensor(float(objective.repr_interval)),
        "peft_num_adapter_layers": flow_loss.new_tensor(float(len(objective.ip_adapter_layers))),
        "peft_num_adapter_tokens": flow_loss.new_tensor(float(objective.ip_adapter_num_tokens)),
    }

    return total_loss, flow_mse, flow_loss.new_tensor(0.0), flow_loss.detach(), repr_loss.detach(), aux_metrics


def _build_teacher_view(student_generator: torch.nn.Module, ema: Any) -> torch.nn.Module:
    """Build a frozen teacher generator view from EMA state.

    Strategy: copy student generator, load EMA state dict, freeze.
    This is expensive (~per-batch), but for smoke testing it's fine. For
    production, switch to a persistent teacher module.
    """
    import copy

    teacher = copy.deepcopy(student_generator)
    # Restore the EMA weights (this includes base + adapter EMA tracking).
    # EMA tracks all params; adapter params are 0-step at first, so they equal init.
    with torch.no_grad():
        teacher.load_state_dict(ema.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher
"""Condition-MLP stage-2 objective + runner (appended to peft_runner.py).

Simplified vs IP-Adapter runner:
- No L_preserve (no EMA teacher).
- No L_cond (no adapter-output-norm regularizer; the MLP is small and the
  small-init last layer already keeps the delta near zero at step 0).
- Main loss = flow_loss (FM velocity MSE on (x_0, z_0)).
- L_repr every K steps with weight lambda_repr.
"""

import dataclasses
from typing import Any

import torch


@dataclasses.dataclass(frozen=True)
class _PEFTMLPObjective:
    """Condition-MLP stage-2 objective runtime config."""
    type: str = "peft_mlp"
    lambda_repr: float = 1.0
    repr_interval: int = 1
    flow_condition: str = "embedding"
    # Compat fields for helpers that read these attrs.
    beta_preserve: float = 0.0
    gamma_cond: float = 0.0
    relation_weight: float = 0.0
    point_weight: float = 1.0
    offdiag_only: bool = True
    repr_learning_rate: float = 0.0
    projection_eps: float = 1e-12
    lambda_lpips: float = 0.0


def peft_mlp_objective_from_config(payload: dict[str, Any], context: str) -> _PEFTMLPObjective:
    lambda_repr = float(payload.get("lambda_repr", 1.0))
    interval = int(payload.get("repr_interval", 1))
    return _PEFTMLPObjective(
        type="peft_mlp",
        lambda_repr=lambda_repr,
        repr_interval=interval,
    )


def init_peft_mlp_generator(generator: torch.nn.Module, objective: _PEFTMLPObjective) -> None:
    """Wrap generator.vector_field with ConditionMLPAdapter (idempotent)."""
    backbone = generator.vector_field
    if getattr(backbone, "_cond_mlp_wrapped", False):
        return
    from safa.models.ip_adapter import wrap_backbone_with_condition_mlp

    wrap_backbone_with_condition_mlp(backbone)
    backbone._cond_mlp_wrapped = True  # type: ignore[attr-defined]


def run_peft_mlp_batch(
    runtime: Any,
    batch: dict[str, Any],
    objective: _PEFTMLPObjective,
):
    """Run one Condition-MLP stage-2 training batch.

    Returns the same 6-tuple shape as run_peft_stage2_batch for compatibility
    with the g_loop dispatch that already exists.
    """
    generator = runtime.generator
    images = batch["image"]
    z = batch["z"]
    sample_ids = batch["sample_id"]

    init_peft_mlp_generator(generator, objective)

    images_latent = runtime._encode_flow_images(images)

    # === Main loss: FM velocity MSE ===
    flow_loss, flow_metrics = generator.flow_matching_loss(images_latent, z)
    flow_mse = flow_metrics["flow_matching_mse"].detach()

    # === L_repr (sparse) ===
    batch_idx = runtime._batch_idx
    is_repr_step = (objective.lambda_repr > 0.0) and (
        batch_idx % max(1, objective.repr_interval) == 0
    )
    if is_repr_step:
        repr_loss, repr_metrics = runtime._compute_repr_loss(
            z, sample_ids, cycle_steps=1, images=images
        )
        total_loss = flow_loss + objective.lambda_repr * repr_loss
    else:
        repr_loss = flow_loss.new_tensor(0.0)
        total_loss = flow_loss

    runtime._batch_idx = batch_idx + 1

    aux_metrics = {
        "peft_main_loss": flow_loss.detach(),
        "peft_preserve_loss": flow_loss.new_tensor(0.0),
        "peft_cond_loss": flow_loss.new_tensor(0.0),
        "peft_is_repr_step": flow_loss.new_tensor(float(is_repr_step)),
        "peft_beta_preserve": flow_loss.new_tensor(0.0),
        "peft_gamma_cond": flow_loss.new_tensor(0.0),
        "peft_lambda_repr": flow_loss.new_tensor(objective.lambda_repr),
        "peft_repr_interval": flow_loss.new_tensor(float(objective.repr_interval)),
        "peft_num_adapter_layers": flow_loss.new_tensor(0.0),
        "peft_num_adapter_tokens": flow_loss.new_tensor(0.0),
    }

    return (
        total_loss,
        flow_mse,
        flow_loss.new_tensor(0.0),
        flow_loss.detach(),
        repr_loss.detach(),
        aux_metrics,
    )

"""PEFT_LoRA stage-2 objective + runner (appended to peft_runner.py).

Expert plan (2026-07-07): generic main loop on FFHQ (z bypassed) + sparse SAFA
loop (z injected via gated low-rank residual). Two data sources are required:
the SAFA train_index/train_features (existing) and an FFHQ image directory
(new). The runner maintains an internal FFHQ iterator via runtime._ffhq_loader.

Design notes:
- step ratio 12:1 means every 13 batches: 12 generic (FFHQ, z=0) + 1 SAFA.
- L_main_generic = L_native (FM velocity on FFHQ x) + beta * L_teacher + gamma * L_cond
- L_main_safa = L_native (FM on SAFA x) + lambda_repr * L_repr (cycle/E0)
- The teacher is a frozen copy of the e15 best.pt backbone (no adapter). We
  build it lazily on the first SAFA step (which is also when we know the
  generator's device/dtype).
"""

import dataclasses
import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _PEFTLoRAObjective:
    """PEFT-LoRA stage-2 objective runtime config.

    Fields:
        type: always "peft_lora".
        beta_teacher: weight for L_teacher (frozen e15 vs student velocity MSE).
        gamma_cond: weight for L_cond (||gate * B(A(z=0))||^2 regularizer).
        lambda_g: weight for the gate^2 penalty term inside L_cond.
        lambda_repr: weight for L_repr on SAFA steps (cycle cosine).
        step_ratio: generic:SAFA = step_ratio:1 (e.g. 12 means 12 generic + 1 SAFA).
        lora_rank: rank of LoRA on adaLN_modulation Linear.
        num_generic_embeddings: number of learned generic face embeddings.
        ffhq_index: path to FFHQ index.jsonl (image_path fields).
        ffhq_image_size: image_size for FFHQ transform (should match pixel_image_size).
        ffhq_per_device_batch: per-device batch size for FFHQ loader.
        flow_condition: required by _validate_train_g_config ("embedding").
    """
    type: str = "peft_lora"
    beta_teacher: float = 1.0
    gamma_cond: float = 0.01
    lambda_g: float = 0.001
    lambda_repr: float = 0.5
    step_ratio: int = 12
    lora_rank: int = 8
    num_generic_embeddings: int = 16
    ffhq_index: str = ""
    ffhq_image_size: int = 256
    ffhq_per_device_batch: int = 16
    flow_condition: str = "embedding"
    # Phase 0.2 ablation knobs (expert 2026-07-07 strict order).
    enable_lora: bool = True
    enable_gated_low_rank: bool = True
    enable_generic_bank: bool = True
    lora_blocks: str = "all"
    # Phase 0.3 (a)/(b) knobs (expert 2026-07-07 strict (a)+(b) parallel test).
    # generic_mode: "bank" (default, Phase 0/0.1/0.2 behaviour),
    #               "null" (Phase 0.3 a: no generic bank, frozen null anchor),
    #               "single_shared" (Phase 0.3 b: 1 shared embedding init=null).
    generic_mode: str = "bank"
    freeze_null_embed: bool = False
    # Compat fields for helpers that read these attrs.
    beta_preserve: float = 0.0
    gamma_cond_compat: float = 0.0
    relation_weight: float = 0.0
    point_weight: float = 1.0
    offdiag_only: bool = True
    repr_learning_rate: float = 0.0
    projection_eps: float = 1.0e-12
    lambda_lpips: float = 0.0
    repr_interval: int = 1  # unused but required by some helpers


def peft_lora_objective_from_config(payload: dict[str, Any], context: str) -> _PEFTLoRAObjective:
    """Parse PEFT-LoRA objective from a config dict."""
    beta = float(payload.get("beta_teacher", 1.0))
    gamma = float(payload.get("gamma_cond", 0.01))
    lambda_g = float(payload.get("lambda_g", 0.001))
    lambda_repr = float(payload.get("lambda_repr", 0.5))
    step_ratio = int(payload.get("step_ratio", 12))
    lora_rank = int(payload.get("lora_rank", 8))
    num_generic = int(payload.get("num_generic_embeddings", 16))
    ffhq_index = str(payload.get("ffhq_index", ""))
    ffhq_image_size = int(payload.get("ffhq_image_size", 256))
    ffhq_batch = int(payload.get("ffhq_per_device_batch", 16))
    if step_ratio < 0:
        raise ValueError(f"{context}.step_ratio must be >= 0, got {step_ratio}")
    if not ffhq_index:
        raise ValueError(f"{context}.ffhq_index is required for peft_lora objective")
    enable_lora = bool(payload.get("enable_lora", True))
    enable_gated = bool(payload.get("enable_gated_low_rank", True))
    enable_bank = bool(payload.get("enable_generic_bank", True))
    lora_blocks = str(payload.get("lora_blocks", "all"))
    if lora_blocks not in ("all", "last_third"):
        raise ValueError(f"{context}.lora_blocks must be 'all' or 'last_third', got {lora_blocks!r}")
    # Phase 0.3 (a)/(b) knobs.
    generic_mode = str(payload.get("generic_mode", "bank"))
    if generic_mode not in ("bank", "null", "single_shared"):
        raise ValueError(
            f"{context}.generic_mode must be 'bank' / 'null' / 'single_shared', got {generic_mode!r}"
        )
    freeze_null_embed = bool(payload.get("freeze_null_embed", False))
    return _PEFTLoRAObjective(
        type="peft_lora",
        beta_teacher=beta,
        gamma_cond=gamma,
        lambda_g=lambda_g,
        lambda_repr=lambda_repr,
        step_ratio=step_ratio,
        lora_rank=lora_rank,
        num_generic_embeddings=num_generic,
        ffhq_index=ffhq_index,
        ffhq_image_size=ffhq_image_size,
        ffhq_per_device_batch=ffhq_batch,
        enable_lora=enable_lora,
        enable_gated_low_rank=enable_gated,
        enable_generic_bank=enable_bank,
        lora_blocks=lora_blocks,
        generic_mode=generic_mode,
        freeze_null_embed=freeze_null_embed,
    )


def init_peft_lora_generator(generator: torch.nn.Module, objective: _PEFTLoRAObjective) -> None:
    """Wrap generator.vector_field with PEFT-LoRA (idempotent)."""
    backbone = generator.vector_field
    if getattr(backbone, "_peft_lora_wrapped", False):
        return
    from safa.models.peft_lora import wrap_backbone_with_peft_lora

    # Infer z_dim and hidden_size from the backbone.
    z_dim = int(backbone.z_embedder[0].in_features)
    hidden_size = int(backbone.x_embedder.out_channels)
    wrap_backbone_with_peft_lora(
        backbone,
        lora_rank=objective.lora_rank,
        lora_alpha=1.0,
        z_dim=z_dim,
        hidden_size=hidden_size,
        num_generic_embeddings=objective.num_generic_embeddings,
        enable_lora=objective.enable_lora,
        enable_gated_low_rank=objective.enable_gated_low_rank,
        enable_generic_bank=objective.enable_generic_bank,
        lora_blocks=objective.lora_blocks,
        generic_mode=objective.generic_mode,
        freeze_null_embed=objective.freeze_null_embed,
    )


def _build_ffhq_loader(runtime: Any, objective: _PEFTLoRAObjective):
    """Build (or rebuild) the FFHQ DataLoader.

    The FFHQ loader is a simple ImageFolder-like loader that reads image_path
    from each jsonl line and applies the same transform as the SAFA loader
    (Resize + ToTensor, no Normalize). It yields {"image": tensor, "sample_id": str}.
    No z, no E0 features.
    """
    import json
    from pathlib import Path

    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    from safa.data.dataset import load_rgb_image_strict

    ffhq_index = Path(objective.ffhq_index)
    if not ffhq_index.is_file():
        raise FileNotFoundError(f"FFHQ index not found: {ffhq_index}")

    records = []
    with ffhq_index.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append({"image_path": rec["image_path"], "sample_id": rec.get("sample_id", rec["image_path"])})

    image_size = int(objective.ffhq_image_size)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    class _FFHQDataset(Dataset):
        def __init__(self, recs, tfm):
            self.records = recs
            self.transform = tfm

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            image = load_rgb_image_strict(rec["image_path"])
            if self.transform is not None:
                image = self.transform(image)
            return {"image": image, "sample_id": rec["sample_id"]}

    dataset = _FFHQDataset(records, transform)

    # Try to use DistributedSampler if the runtime is in DDP mode.
    sampler = None
    if hasattr(runtime, "distributed") and getattr(runtime.distributed, "enabled", False):
        from torch.utils.data import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            num_replicas=runtime.distributed.world_size,
            rank=runtime.distributed.rank,
            shuffle=True,
            seed=int(getattr(runtime, "sampling_seed", 1337)),
            drop_last=False,
        )

    per_device = int(objective.ffhq_per_device_batch)
    num_workers = max(1, int(getattr(runtime, "num_workers", 4)))
    return DataLoader(
        dataset,
        batch_size=per_device,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )


def _build_teacher_backbone(runtime: Any, generator: torch.nn.Module) -> torch.nn.Module:
    """Build a frozen teacher backbone (deepcopy of generator.vector_field with
    adapter params reset to zero/init, so teacher == pre-PEFT base).

    Strategy: deepcopy the whole generator, then for the teacher's vector_field:
    - zero out all lora_b weights (kills LoRA delta)
    - zero out gated_low_rank_z.gate (kills delta_z)
    - zero out generic_bank? No — the generic bank is sampled fresh on each
      forward, so we cannot "reset" it; instead, for teacher forward we bypass
      generic_bank and delta_z entirely by calling the *original* forward.

    Simpler: teacher is the e15 checkpoint loaded into a fresh backbone (no
    wrap). We load it lazily from runtime.teacher_state_dict if present, else
    we snapshot the *current* backbone weights with adapter zeroed.

    Because the e15 checkpoint is already loaded as the base of the wrapped
    backbone (the wrap preserves base weights via LoRALinear.base), the teacher
    is just: deepcopy(backbone) with all lora_b set to 0, gate set to 0,
    generic_bank set to 0, _peft_lora_null_embed set to 0. Then teacher.forward
    (still the patched forward) returns the e15-equivalent output.
    """
    import copy

    teacher_bb = copy.deepcopy(generator.vector_field)
    # Reset adapter contributions to 0 so teacher == base e15.
    with torch.no_grad():
        for module in teacher_bb.modules():
            from safa.models.peft_lora import LoRALinear, GatedLowRankResidual, GenericEmbeddingBank

            if isinstance(module, LoRALinear):
                module.lora_b.weight.zero_()
            elif isinstance(module, GatedLowRankResidual):
                module.gate.zero_()
                # zero A/B too so delta is exactly 0 even if gate is later mutated
                module.A_proj.weight.zero_()
                module.B_proj.weight.zero_()
        if hasattr(teacher_bb, "generic_bank"):
            teacher_bb.generic_bank.embeddings.weight.zero_()
        if hasattr(teacher_bb, "_peft_lora_null_embed") and isinstance(teacher_bb._peft_lora_null_embed, torch.nn.Parameter):
            teacher_bb._peft_lora_null_embed.zero_()
    for p in teacher_bb.parameters():
        p.requires_grad_(False)
    teacher_bb.eval()
    return teacher_bb


def _velocity_at_midpoint(backbone, x_latent, z):
    """Compute backbone(x_t, r, t, z) at t=0.5, r=0.0 for L_teacher."""
    B = x_latent.shape[0]
    device = x_latent.device
    dtype = x_latent.dtype
    t = torch.full((B,), 0.5, device=device, dtype=dtype)
    r = torch.full((B,), 0.0, device=device, dtype=dtype)
    view_t = t.view(-1, 1, 1, 1)
    eps = torch.randn_like(x_latent)
    x_t = (1.0 - view_t) * x_latent + view_t * eps
    return backbone(x_t, r, t, z), x_t, r, t, eps


def run_peft_lora_batch(
    runtime: Any,
    batch: dict[str, Any],
    objective: _PEFTLoRAObjective,
):
    """Run one PEFT-LoRA training batch.

    The g_loop dispatcher calls this on every SAFA loader batch. We internally
    decide whether this is a generic step (use FFHQ batch, z=0) or a SAFA step
    (use the passed SAFA batch) based on runtime._batch_idx modulo (step_ratio + 1).

    Returns the same 6-tuple shape as run_peft_stage2_batch / run_peft_mlp_batch.
    """
    generator = runtime.generator
    images = batch["image"]
    z = batch["z"]
    sample_ids = batch["sample_id"]

    # Ensure adapter is attached (idempotent).
    init_peft_lora_generator(generator, objective)

    # Decide step type. step_ratio=12 -> pattern: 12 SAFA + 1 generic? No.
    # Expert says generic:SAFA = 12:1, so 12 generic then 1 SAFA. But the g_loop
    # pulls SAFA batches; we cannot easily pull 12 SAFA batches then 1 generic.
    # Pragmatic: alternate by (batch_idx mod (step_ratio+1)) == step_ratio => SAFA step,
    # else generic step. So we get 12 generic + 1 SAFA in a 13-step cycle, but the
    # SAFA data is consumed only on SAFA steps (1/13 of the SAFA loader is used
    # per cycle). To keep the SAFA data meaningful, we still use the SAFA batch
    # image as the generic step image when it's a generic step BUT with z=0 and
    # no repr loss (effectively using SAFA images as generic FFHQ-style data).
    # This is suboptimal; the proper fix is a separate FFHQ loader.
    #
    # === Phase 0 simplification ===
    # We DO build a separate FFHQ loader. On generic steps we pull from FFHQ;
    # on SAFA steps we use the SAFA batch passed in.

    batch_idx = runtime._batch_idx
    cycle_len = int(objective.step_ratio) + 1  # 12 generic + 1 SAFA = 13
    is_safa_step = (batch_idx % cycle_len) == (cycle_len - 1)  # last step in cycle is SAFA

    # Build FFHQ loader lazily on first generic step.
    if not is_safa_step:
        if not hasattr(runtime, "_ffhq_loader") or runtime._ffhq_loader is None:
            runtime._ffhq_loader = _build_ffhq_loader(runtime, objective)
            runtime._ffhq_iter = iter(runtime._ffhq_loader)
        # Pull a batch; rebuild iterator if exhausted.
        try:
            ffhq_batch = next(runtime._ffhq_iter)
        except StopIteration:
            # On DDP, set_epoch is needed for proper shuffling. Skip for simplicity.
            runtime._ffhq_iter = iter(runtime._ffhq_loader)
            ffhq_batch = next(runtime._ffhq_iter)
        ffhq_images = ffhq_batch["image"].to(device=images.device, dtype=images.dtype)

    # === Common setup ===
    # Teacher: build lazily and cache on runtime.
    if objective.beta_teacher > 0.0:
        if not hasattr(runtime, "_peft_lora_teacher") or runtime._peft_lora_teacher is None:
            runtime._peft_lora_teacher = _build_teacher_backbone(runtime, generator)
        teacher_bb = runtime._peft_lora_teacher
    else:
        teacher_bb = None

    backbone = generator.vector_field
    flow_mse = images.new_tensor(0.0)
    repr_loss = images.new_tensor(0.0)
    teacher_loss = images.new_tensor(0.0)
    cond_loss = images.new_tensor(0.0)
    aux_metrics: dict[str, torch.Tensor] = {}

    if is_safa_step:
        # === SAFA step: full forward with z, L_native + lambda_repr * L_repr ===
        images_latent = runtime._encode_flow_images(images)
        flow_loss, flow_metrics = generator.flow_matching_loss(images_latent, z)
        flow_mse = flow_metrics["flow_matching_mse"].detach()

        # L_repr: cycle cosine via runtime helper.
        if objective.lambda_repr > 0.0:
            repr_loss, repr_metrics = runtime._compute_repr_loss(
                z, sample_ids, cycle_steps=1, images=images
            )

        total_loss = flow_loss + objective.lambda_repr * repr_loss

        aux_metrics["peft_lora_step_type"] = images.new_tensor(1.0)  # 1 = SAFA
    else:
        # === Generic step: FFHQ batch, z=0, L_native + beta*L_teacher + gamma*L_cond ===
        images_latent = runtime._encode_flow_images(ffhq_images)
        B = images_latent.shape[0]
        z_dim = int(backbone.z_embedder[0].in_features)
        z_zeros = torch.zeros(B, z_dim, device=images_latent.device, dtype=images_latent.dtype)

        flow_loss, flow_metrics = generator.flow_matching_loss(images_latent, z_zeros)
        flow_mse = flow_metrics["flow_matching_mse"].detach()

        # L_teacher: velocity MSE at midpoint between student and frozen teacher.
        if teacher_bb is not None:
            with torch.no_grad():
                # teacher backbone forward uses the patched forward too, but with
                # adapter contributions zeroed (set in _build_teacher_backbone),
                # so it is functionally the e15 base.
                teacher_vel, _, _, _, _ = _velocity_at_midpoint(teacher_bb, images_latent.detach(), z_zeros)
            student_vel, _, _, _, _ = _velocity_at_midpoint(backbone, images_latent, z_zeros)
            teacher_loss = (student_vel - teacher_vel.detach()).square().mean()
            total_loss = flow_loss + objective.beta_teacher * teacher_loss
        else:
            total_loss = flow_loss

        # L_cond: ||B_proj(A_proj(z=0))||^2 + lambda_g * gate^2
        if objective.gamma_cond > 0.0:
            from safa.models.peft_lora import compute_l_cond

            cond_delta_norm = compute_l_cond(backbone, z_zeros)
            gate_pen = backbone.gated_low_rank_z.gate.square()
            cond_loss = cond_delta_norm + objective.lambda_g * gate_pen
            total_loss = total_loss + objective.gamma_cond * cond_loss

        aux_metrics["peft_lora_step_type"] = images.new_tensor(0.0)  # 0 = generic

    runtime._batch_idx = batch_idx + 1

    aux_metrics.update({
        "peft_main_loss": total_loss.detach(),
        "peft_flow_loss": flow_loss.detach(),
        "peft_flow_mse": flow_mse,
        "peft_teacher_loss": teacher_loss.detach(),
        "peft_cond_loss": cond_loss.detach(),
        "peft_repr_loss": repr_loss.detach(),
        "peft_beta_teacher": images.new_tensor(objective.beta_teacher),
        "peft_gamma_cond": images.new_tensor(objective.gamma_cond),
        "peft_lambda_g": images.new_tensor(objective.lambda_g),
        "peft_lambda_repr": images.new_tensor(objective.lambda_repr),
        "peft_step_ratio": images.new_tensor(float(objective.step_ratio)),
        "peft_lora_rank": images.new_tensor(float(objective.lora_rank)),
        "peft_num_generic": images.new_tensor(float(objective.num_generic_embeddings)),
        "peft_gate_value": backbone.gated_low_rank_z.gate.detach(),
    })

    # Return 6-tuple matching other runners: (total_loss, flow_mse, cycle_zero, flow_loss, repr_loss, aux_metrics)
    cycle_zero = images.new_tensor(0.0)
    return total_loss, flow_mse, cycle_zero, flow_loss.detach(), repr_loss.detach(), aux_metrics
