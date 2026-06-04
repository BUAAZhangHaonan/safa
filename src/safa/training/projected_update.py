from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ProjectionResult:
    dot_before: torch.Tensor
    dot_after: torch.Tensor
    fm_norm: torch.Tensor
    repr_norm: torch.Tensor
    projected_repr_norm: torch.Tensor
    projection_applied: bool
    projection_removed_norm: torch.Tensor
    repr_descent_inner_product: torch.Tensor
    fm_first_order_effect: torch.Tensor
    projected_gradients: list[torch.Tensor]


@dataclass(frozen=True)
class AdaptiveMarginAdjustment:
    previous_margin: float
    next_margin: float
    normalized_fm_loss: float
    baseline: float
    direction: str


@dataclass(frozen=True)
class TrustRegionScaleResult:
    trust_radius: float
    trust_scale: float
    trust_region_active: bool
    scaled_norm: float
    projected_norm: float
    scaled_gradients: list[torch.Tensor]


@dataclass(frozen=True)
class DualBudgetControlResult:
    previous_dual_value: float
    next_dual_value: float
    previous_trust_radius: float
    next_trust_radius: float
    direction: str
    fm_budget_violation: float


def project_gradient_onto_fm_feasible_cone(
    g_repr: list[torch.Tensor],
    g_fm: list[torch.Tensor],
    eps: float,
) -> ProjectionResult:
    _validate_gradient_lists(g_repr, g_fm)
    _validate_eps(eps)

    dot_before = _dot(g_repr, g_fm)
    fm_norm_squared = _squared_norm(g_fm)
    fm_norm = torch.sqrt(fm_norm_squared)
    repr_norm = torch.sqrt(_squared_norm(g_repr))
    eps_tensor = torch.as_tensor(eps, dtype=fm_norm.dtype, device=fm_norm.device)

    projection_applied = bool((dot_before < 0).item() and (fm_norm > eps_tensor).item())
    if projection_applied:
        coefficient = dot_before / fm_norm_squared
        projected_gradients = [repr_grad - coefficient * fm_grad for repr_grad, fm_grad in zip(g_repr, g_fm)]
    else:
        projected_gradients = [repr_grad.clone() for repr_grad in g_repr]

    dot_after = _dot(projected_gradients, g_fm)
    if projection_applied:
        zero = torch.zeros((), dtype=dot_after.dtype, device=dot_after.device)
        if not torch.allclose(dot_after, zero, rtol=1e-5, atol=1e-6):
            raise RuntimeError("Projected representation gradient is not orthogonal to FM gradient")

    projected_repr_norm = torch.sqrt(_squared_norm(projected_gradients))
    removed_gradients = [repr_grad - projected_grad for repr_grad, projected_grad in zip(g_repr, projected_gradients)]
    projection_removed_norm = torch.sqrt(_squared_norm(removed_gradients))
    repr_descent_inner_product = _dot(g_repr, projected_gradients)
    fm_first_order_effect = -dot_after
    return ProjectionResult(
        dot_before=dot_before,
        dot_after=dot_after,
        fm_norm=fm_norm,
        repr_norm=repr_norm,
        projected_repr_norm=projected_repr_norm,
        projection_applied=projection_applied,
        projection_removed_norm=projection_removed_norm,
        repr_descent_inner_product=repr_descent_inner_product,
        fm_first_order_effect=fm_first_order_effect,
        projected_gradients=projected_gradients,
    )


def apply_fm_anchor_trust_region_scaling(
    projected_gradients: list[torch.Tensor],
    g_fm: list[torch.Tensor],
    trust_radius: float,
    eps: float,
) -> TrustRegionScaleResult:
    _validate_gradient_lists(projected_gradients, g_fm)
    trust_radius = _validate_real_scalar(trust_radius, "trust_radius", min_value=0.0)
    _validate_eps(eps)

    projected_norm_tensor = torch.sqrt(_squared_norm(projected_gradients))
    fm_norm_tensor = torch.sqrt(_squared_norm(g_fm))
    eps_tensor = torch.as_tensor(eps, dtype=projected_norm_tensor.dtype, device=projected_norm_tensor.device)

    if bool((projected_norm_tensor <= eps_tensor).item()):
        scaled_gradients = [gradient.clone() for gradient in projected_gradients]
        trust_scale = 1.0
        scaled_norm_tensor = projected_norm_tensor
        trust_region_active = False
    else:
        max_projected_norm_tensor = fm_norm_tensor * trust_radius
        scale_tensor = torch.clamp(max_projected_norm_tensor / projected_norm_tensor, max=1.0)
        if not torch.isfinite(scale_tensor):
            raise FloatingPointError("Trust-region scale must be finite")
        trust_scale = float(scale_tensor.detach().cpu())
        scaled_gradients = [gradient * scale_tensor for gradient in projected_gradients]
        scaled_norm_tensor = torch.sqrt(_squared_norm(scaled_gradients))
        trust_region_active = trust_scale < 1.0
        if trust_region_active:
            max_allowed = max_projected_norm_tensor + eps_tensor
            within_roundoff = torch.allclose(
                scaled_norm_tensor,
                max_projected_norm_tensor,
                rtol=1e-5,
                atol=max(float(eps), 1e-6),
            )
            if bool((scaled_norm_tensor > max_allowed).item()) and not within_roundoff:
                raise RuntimeError("Trust-region scaling exceeded the requested FM-anchored radius")

    return TrustRegionScaleResult(
        trust_radius=trust_radius,
        trust_scale=trust_scale,
        trust_region_active=trust_region_active,
        scaled_norm=float(scaled_norm_tensor.detach().cpu()),
        projected_norm=float(projected_norm_tensor.detach().cpu()),
        scaled_gradients=scaled_gradients,
    )


def update_dual_budget_controller(
    current_dual_value: float,
    current_trust_radius: float,
    actual_fm_delta: float,
    fm_delta_target: float,
    dual_lr: float,
    trust_radius_min: float,
    trust_radius_max: float,
) -> DualBudgetControlResult:
    current_dual_value = _validate_real_scalar(current_dual_value, "current_dual_value", min_value=0.0)
    trust_radius_min = _validate_real_scalar(trust_radius_min, "trust_radius_min", min_value=0.0)
    trust_radius_max = _validate_real_scalar(trust_radius_max, "trust_radius_max", min_value=trust_radius_min)
    current_trust_radius = _validate_real_scalar(
        current_trust_radius,
        "current_trust_radius",
        min_value=trust_radius_min,
    )
    if current_trust_radius > trust_radius_max:
        raise ValueError("current_trust_radius must be <= trust_radius_max")
    actual_fm_delta = _validate_real_scalar(actual_fm_delta, "actual_fm_delta")
    fm_delta_target = _validate_real_scalar(fm_delta_target, "fm_delta_target", min_value=0.0)
    dual_lr = _validate_real_scalar(dual_lr, "dual_lr", min_value=0.0)

    fm_budget_violation = actual_fm_delta - fm_delta_target
    if fm_budget_violation > 0.0:
        direction = "tighten"
    elif fm_budget_violation < 0.0:
        direction = "loosen"
    else:
        direction = "hold"

    next_dual_value = max(0.0, current_dual_value + dual_lr * fm_budget_violation)
    proposed_trust_radius = current_trust_radius * math.exp(-dual_lr * fm_budget_violation)
    next_trust_radius = min(max(proposed_trust_radius, trust_radius_min), trust_radius_max)
    if not math.isfinite(next_trust_radius):
        raise FloatingPointError("next_trust_radius must be finite")

    return DualBudgetControlResult(
        previous_dual_value=current_dual_value,
        next_dual_value=next_dual_value,
        previous_trust_radius=current_trust_radius,
        next_trust_radius=next_trust_radius,
        direction=direction,
        fm_budget_violation=fm_budget_violation,
    )


def compute_adaptive_margin_adjustment(
    current_margin: float,
    normalized_fm_loss: float,
    baseline: float,
    step: float,
    min_margin: float,
    max_margin: float,
) -> AdaptiveMarginAdjustment:
    current_margin = _validate_real_scalar(current_margin, "current_margin", min_value=0.0)
    normalized_fm_loss = _validate_real_scalar(normalized_fm_loss, "normalized_fm_loss")
    baseline = _validate_real_scalar(baseline, "baseline")
    step = _validate_real_scalar(step, "step", min_value=0.0)
    min_margin = _validate_real_scalar(min_margin, "min_margin", min_value=0.0)
    max_margin = _validate_real_scalar(max_margin, "max_margin", min_value=min_margin)
    if current_margin < min_margin or current_margin > max_margin:
        raise ValueError("current_margin must lie within [min_margin, max_margin]")

    if normalized_fm_loss > baseline:
        next_margin = max(min_margin, current_margin - step)
        direction = "tighten"
    elif normalized_fm_loss < baseline:
        next_margin = min(max_margin, current_margin + step)
        direction = "loosen"
    else:
        next_margin = current_margin
        direction = "hold"
    return AdaptiveMarginAdjustment(
        previous_margin=current_margin,
        next_margin=next_margin,
        normalized_fm_loss=normalized_fm_loss,
        baseline=baseline,
        direction=direction,
    )


def _validate_gradient_lists(g_repr: list[torch.Tensor], g_fm: list[torch.Tensor]) -> None:
    if not isinstance(g_repr, list) or not isinstance(g_fm, list):
        raise TypeError("g_repr and g_fm must be list[Tensor]")
    if not g_repr or not g_fm:
        raise ValueError("g_repr and g_fm must be non-empty")
    if len(g_repr) != len(g_fm):
        raise ValueError("g_repr and g_fm must have the same length")
    for index, (repr_grad, fm_grad) in enumerate(zip(g_repr, g_fm)):
        if not isinstance(repr_grad, torch.Tensor):
            raise TypeError(f"g_repr[{index}] must be a torch.Tensor")
        if not isinstance(fm_grad, torch.Tensor):
            raise TypeError(f"g_fm[{index}] must be a torch.Tensor")
        if repr_grad.shape != fm_grad.shape:
            raise ValueError(f"g_repr[{index}] and g_fm[{index}] must have the same shape")
        if repr_grad.device != fm_grad.device:
            raise ValueError(f"g_repr[{index}] and g_fm[{index}] must be on the same device")
        if not repr_grad.is_floating_point() or not fm_grad.is_floating_point():
            raise TypeError("gradient tensors must be floating point")
        if not torch.isfinite(repr_grad).all() or not torch.isfinite(fm_grad).all():
            raise FloatingPointError("gradient tensors must be finite")


def _validate_eps(eps: float) -> None:
    if not isinstance(eps, (float, int)):
        raise TypeError("eps must be a real scalar")
    if not math.isfinite(float(eps)) or float(eps) < 0.0:
        raise ValueError("eps must be finite and non-negative")


def _validate_real_scalar(value: float, name: str, min_value: float | None = None) -> float:
    if not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if min_value is not None and scalar < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return scalar


def _dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> torch.Tensor:
    total = None
    for left_item, right_item in zip(left, right):
        item = (left_item * right_item).sum()
        total = item if total is None else total + item
    if total is None:
        raise RuntimeError("Cannot compute dot product for an empty gradient list")
    return total


def _squared_norm(gradients: list[torch.Tensor]) -> torch.Tensor:
    total = None
    for gradient in gradients:
        item = gradient.pow(2).sum()
        total = item if total is None else total + item
    if total is None:
        raise RuntimeError("Cannot compute norm for an empty gradient list")
    return total
