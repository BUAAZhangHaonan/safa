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
