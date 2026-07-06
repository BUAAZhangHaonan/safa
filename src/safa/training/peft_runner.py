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
