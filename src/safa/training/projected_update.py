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
class CAGradResult:
    fm_weight: float
    cl_weight: float
    gradient_cosine: float
    combined_norm: float
    combined_gradients: list[torch.Tensor]


@dataclass(frozen=True)
class FMAnchoredCAGradResult:
    fm_weight: float
    cl_weight: float
    raw_fm_weight: float
    raw_cl_weight: float
    gradient_cosine: float
    combined_norm: float
    fm_descent_floor: float
    fm_descent_after_cagrad: float
    fm_descent_after_anchor: float
    anchor_active: bool
    combined_gradients: list[torch.Tensor]


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


def project_gradient_to_dot_lower_bound(
    g_repr: list[torch.Tensor],
    g_fm: list[torch.Tensor],
    lower_bound: float,
    eps: float,
) -> ProjectionResult:
    _validate_gradient_lists(g_repr, g_fm)
    if not isinstance(lower_bound, (float, int)):
        raise TypeError("lower_bound must be a real scalar")
    if not math.isfinite(float(lower_bound)):
        raise ValueError("lower_bound must be finite")
    _validate_eps(eps)

    dot_before = _dot(g_repr, g_fm)
    fm_norm_squared = _squared_norm(g_fm)
    fm_norm = torch.sqrt(fm_norm_squared)
    repr_norm = torch.sqrt(_squared_norm(g_repr))
    eps_tensor = torch.as_tensor(eps, dtype=fm_norm.dtype, device=fm_norm.device)
    lower_bound_tensor = torch.as_tensor(float(lower_bound), dtype=dot_before.dtype, device=dot_before.device)

    projection_applied = bool((dot_before < lower_bound_tensor).item() and (fm_norm > eps_tensor).item())
    if projection_applied:
        coefficient = (dot_before - lower_bound_tensor) / fm_norm_squared
        projected_gradients = [repr_grad - coefficient * fm_grad for repr_grad, fm_grad in zip(g_repr, g_fm)]
    else:
        projected_gradients = [repr_grad.clone() for repr_grad in g_repr]

    dot_after = _dot(projected_gradients, g_fm)
    if projection_applied and not torch.allclose(dot_after, lower_bound_tensor, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Projected representation gradient does not satisfy the FM dot-product lower bound")

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


def aggregate_two_task_cagrad(
    g_fm: list[torch.Tensor],
    g_cl: list[torch.Tensor],
    c: float,
    eps: float,
) -> CAGradResult:
    _validate_gradient_lists(g_fm, g_cl)
    _validate_cagrad_c(c)
    _validate_positive_eps(eps)

    c_value = float(c)
    g0 = [0.5 * (fm_grad + cl_grad) for fm_grad, cl_grad in zip(g_fm, g_cl)]
    g0_norm = torch.sqrt(_squared_norm(g0))
    if bool((g0_norm <= float(eps)).item()):
        raise FloatingPointError("CAGrad received a near-zero mean task gradient")

    def weighted_gradient(alpha: float) -> list[torch.Tensor]:
        return [float(alpha) * fm_grad + (1.0 - float(alpha)) * cl_grad for fm_grad, cl_grad in zip(g_fm, g_cl)]

    direction = [fm_grad - cl_grad for fm_grad, cl_grad in zip(g_fm, g_cl)]

    def objective_derivative(alpha: float) -> torch.Tensor:
        weighted = weighted_gradient(alpha)
        weighted_norm = torch.sqrt(_squared_norm(weighted))
        if bool((weighted_norm <= float(eps)).item()):
            raise FloatingPointError("CAGrad simplex solve reached a near-zero weighted gradient")
        return _dot(direction, g0) + c_value * g0_norm * _dot(direction, weighted) / weighted_norm

    left_derivative = float(objective_derivative(0.0).detach().cpu())
    right_derivative = float(objective_derivative(1.0).detach().cpu())
    if left_derivative >= 0.0:
        fm_weight = 0.0
    elif right_derivative <= 0.0:
        fm_weight = 1.0
    else:
        low = 0.0
        high = 1.0
        for _ in range(80):
            middle = 0.5 * (low + high)
            middle_derivative = float(objective_derivative(middle).detach().cpu())
            if middle_derivative < 0.0:
                low = middle
            else:
                high = middle
        fm_weight = 0.5 * (low + high)

    cl_weight = 1.0 - fm_weight
    weighted = weighted_gradient(fm_weight)
    weighted_norm = torch.sqrt(_squared_norm(weighted))
    if bool((weighted_norm <= float(eps)).item()):
        raise FloatingPointError("CAGrad selected a near-zero weighted gradient")
    scale = c_value * g0_norm / weighted_norm
    combined = [g0_grad + scale * weighted_grad for g0_grad, weighted_grad in zip(g0, weighted)]
    combined_norm = torch.sqrt(_squared_norm(combined))
    return CAGradResult(
        fm_weight=float(fm_weight),
        cl_weight=float(cl_weight),
        gradient_cosine=float(_gradient_cosine(g_fm, g_cl, eps).detach().cpu()),
        combined_norm=float(combined_norm.detach().cpu()),
        combined_gradients=combined,
    )


def aggregate_two_task_fm_anchored_cagrad(
    g_fm: list[torch.Tensor],
    g_cl: list[torch.Tensor],
    c: float,
    fm_descent_floor_fraction: float,
    eps: float,
) -> FMAnchoredCAGradResult:
    _validate_gradient_lists(g_fm, g_cl)
    floor_fraction = _validate_fraction(fm_descent_floor_fraction, "fm_descent_floor_fraction")
    raw = aggregate_two_task_cagrad(g_fm, g_cl, c=c, eps=eps)
    fm_norm_squared = _squared_norm(g_fm)
    if bool((fm_norm_squared <= float(eps)).item()):
        raise FloatingPointError("FM-anchored CAGrad received a near-zero FM gradient")

    floor_tensor = floor_fraction * fm_norm_squared
    raw_descent_tensor = _dot(g_fm, raw.combined_gradients)
    if bool((raw_descent_tensor + _eps_tensor(raw_descent_tensor, eps) >= floor_tensor).item()):
        combined = [gradient.clone() for gradient in raw.combined_gradients]
        fm_weight = raw.fm_weight
        cl_weight = raw.cl_weight
        anchor_active = False
    else:
        fm_cl_dot = _dot(g_fm, g_cl)
        fm_weight = _closest_fm_weight_for_floor(
            current_fm_weight=raw.fm_weight,
            fm_norm_squared=float(fm_norm_squared.detach().cpu()),
            fm_cl_dot=float(fm_cl_dot.detach().cpu()),
            floor=float(floor_tensor.detach().cpu()),
            eps=float(eps),
        )
        cl_weight = 1.0 - fm_weight
        combined = [fm_weight * fm_grad + cl_weight * cl_grad for fm_grad, cl_grad in zip(g_fm, g_cl)]
        anchor_active = True

    anchored_descent_tensor = _dot(g_fm, combined)
    tolerance = _eps_tensor(anchored_descent_tensor, eps)
    if bool((anchored_descent_tensor + tolerance < floor_tensor).item()):
        raise RuntimeError("FM-anchored CAGrad did not satisfy the FM descent floor")
    combined_norm = torch.sqrt(_squared_norm(combined))
    return FMAnchoredCAGradResult(
        fm_weight=float(fm_weight),
        cl_weight=float(cl_weight),
        raw_fm_weight=float(raw.fm_weight),
        raw_cl_weight=float(raw.cl_weight),
        gradient_cosine=raw.gradient_cosine,
        combined_norm=float(combined_norm.detach().cpu()),
        fm_descent_floor=float(floor_tensor.detach().cpu()),
        fm_descent_after_cagrad=float(raw_descent_tensor.detach().cpu()),
        fm_descent_after_anchor=float(anchored_descent_tensor.detach().cpu()),
        anchor_active=anchor_active,
        combined_gradients=combined,
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


def _validate_positive_eps(eps: float) -> None:
    if not isinstance(eps, (float, int)):
        raise TypeError("eps must be a real scalar")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")


def _validate_cagrad_c(c: float) -> None:
    if not isinstance(c, (float, int)) or not math.isfinite(float(c)) or float(c) < 0.0 or float(c) >= 1.0:
        raise ValueError("c must be finite and in [0, 1)")


def _validate_fraction(value: float, name: str) -> float:
    scalar = _validate_nonnegative_scalar(value, name)
    if scalar > 1.0:
        raise ValueError(f"{name} must be <= 1.0")
    return scalar


def _validate_nonnegative_scalar(value: float, name: str) -> float:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _closest_fm_weight_for_floor(
    current_fm_weight: float,
    fm_norm_squared: float,
    fm_cl_dot: float,
    floor: float,
    eps: float,
) -> float:
    denominator = fm_norm_squared - fm_cl_dot
    if abs(denominator) <= eps:
        if fm_cl_dot + eps >= floor:
            return min(max(float(current_fm_weight), 0.0), 1.0)
        raise RuntimeError("No two-task convex CAGrad weight can satisfy the FM descent floor")
    threshold = (floor - fm_cl_dot) / denominator
    if denominator > 0.0:
        lower = min(max(threshold, 0.0), 1.0)
        upper = 1.0
    else:
        lower = 0.0
        upper = min(max(threshold, 0.0), 1.0)
    if lower > upper + eps:
        raise RuntimeError("FM descent floor is infeasible inside the two-task simplex")
    return min(max(float(current_fm_weight), lower), upper)


def _eps_tensor(reference: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.as_tensor(max(float(eps), 1.0e-6), dtype=reference.dtype, device=reference.device)


def _gradient_cosine(left: list[torch.Tensor], right: list[torch.Tensor], eps: float) -> torch.Tensor:
    left_norm = torch.sqrt(_squared_norm(left))
    right_norm = torch.sqrt(_squared_norm(right))
    denominator = left_norm * right_norm
    if bool((denominator <= float(eps)).item()):
        raise FloatingPointError("Cannot compute cosine for near-zero gradients")
    return _dot(left, right) / denominator


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
