from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

from safa.training.losses import normalize_for_e0


@dataclass(frozen=True)
class GuidanceResult:
    latent: torch.Tensor
    nfe: int
    diagnostics: dict[str, torch.Tensor | float | int | list[float]]


class CountedFlowMap:
    """Count vector-field evaluations made through a generator flow map."""

    def __init__(self, generator):
        self.generator = generator
        self.nfe = 0

    def __call__(self, x, z, *, t, r):
        self.nfe += 1
        return self.generator.flow_map(x, z, t=t, r=r)


def freeze_guidance_stack(generator, codec, e0) -> None:
    _freeze_module(generator)
    _freeze_module(codec.vae)
    _freeze_module(e0)


def assert_guidance_stack_frozen(generator, codec, e0) -> None:
    for name, module in (("generator", generator), ("codec.vae", codec.vae), ("e0", e0)):
        if module.training:
            raise RuntimeError(f"guidance requires {name} in evaluation mode")
        for parameter in module.parameters():
            if parameter.requires_grad:
                raise RuntimeError(f"guidance requires frozen {name} parameters")
            if parameter.grad is not None:
                raise RuntimeError(f"guidance found an unexpected {name} parameter gradient")


def symmetric_relative_l2(left, right, eps: float = 1.0e-8) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(f"relative L2 inputs must have the same shape, got {tuple(left.shape)} and {tuple(right.shape)}")
    if left.ndim < 2:
        raise ValueError(f"relative L2 inputs must have a batch dimension, got {tuple(left.shape)}")
    left_norm = left.flatten(1).norm(dim=1)
    right_norm = right.flatten(1).norm(dim=1)
    difference = (left - right).flatten(1).norm(dim=1)
    return 2.0 * difference / (left_norm + right_norm + eps)


def semigroup_probe(flow_map, x_init, condition, split_times) -> dict[str, Any]:
    splits = [float(value) for value in split_times]
    if not splits or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in splits):
        raise ValueError("split_times must be strictly increasing and within (0,1)")
    if any(left >= right for left, right in zip(splits, splits[1:])):
        raise ValueError("split_times must be strictly increasing and within (0,1)")

    initial_nfe = flow_map.nfe
    direct = flow_map(x_init, condition, t=1.0, r=0.0)
    split_endpoints: dict[float, torch.Tensor] = {}
    residuals: dict[float, torch.Tensor] = {}
    for split in splits:
        intermediate = flow_map(x_init, condition, t=1.0, r=split)
        endpoint = flow_map(intermediate, condition, t=split, r=0.0)
        split_endpoints[split] = endpoint
        residuals[split] = symmetric_relative_l2(direct, endpoint)
    return {
        "direct_endpoint": direct,
        "split_endpoints": split_endpoints,
        "residuals": residuals,
        "nfe": flow_map.nfe - initial_nfe,
    }


def sample_official_head_current_xt(
    *,
    flow_map: CountedFlowMap,
    codec,
    e0,
    x_init: torch.Tensor,
    transport_condition: torch.Tensor,
    target_z0: torch.Tensor,
    guided_times: Sequence[float],
    unguided_times: Sequence[float],
    sample_mode: Literal["flow_map1", "flow_map2"],
    optimization_mode: Literal["official_adam", "paper_normalized_direct_autograd"],
    num_optim_iters: int,
    step_size: float,
) -> GuidanceResult:
    guided, unguided = _validate_fmrg_arguments(
        flow_map=flow_map,
        codec=codec,
        e0=e0,
        guided_times=guided_times,
        unguided_times=unguided_times,
        step_size=step_size,
    )
    if sample_mode not in {"flow_map1", "flow_map2"}:
        raise ValueError(f"unsupported sample_mode {sample_mode!r}")
    if optimization_mode not in {"official_adam", "paper_normalized_direct_autograd"}:
        raise ValueError(f"unsupported optimization_mode {optimization_mode!r}")
    if isinstance(num_optim_iters, bool) or int(num_optim_iters) != num_optim_iters or num_optim_iters <= 0:
        raise ValueError(f"num_optim_iters must be a positive integer, got {num_optim_iters!r}")
    if optimization_mode == "paper_normalized_direct_autograd" and num_optim_iters != 1:
        raise ValueError("paper_normalized_direct_autograd requires num_optim_iters == 1")

    initial_nfe = flow_map.nfe
    state = _finite_tensor("x_init", x_init.detach())
    loss_history: list[float] = []
    learning_rates: list[float] = []

    for interval_index, (t, s) in enumerate(zip(guided, guided[1:])):
        before = state.detach()
        if optimization_mode == "official_adam":
            lr = step_size * (1.0 - interval_index / 4.0)
            learning_rates.append(lr)
            optimized = before.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([optimized], lr=lr)
            endpoint_velocity = None
            for _ in range(num_optim_iters):
                optimizer.zero_grad(set_to_none=True)
                endpoint = flow_map(optimized, transport_condition, t=t, r=0.0)
                if endpoint_velocity is None:
                    endpoint_velocity = (optimized - endpoint) / t
                loss = _representation_loss(endpoint, codec, e0, target_z0)
                loss_history.append(float(loss.detach()))
                loss.backward()
                gradient = _finite_tensor("representation gradient", optimized.grad)
                optimizer.step()
                _finite_tensor("optimized x_t", optimized)
                del gradient
            if endpoint_velocity is None:
                raise RuntimeError("official Adam completed without an endpoint evaluation")
            delta_xt = -(optimized.detach() - before)
            if sample_mode == "flow_map1":
                step_velocity = endpoint_velocity.detach()
            else:
                step_endpoint = flow_map(before, transport_condition, t=t, r=s)
                step_velocity = (before - step_endpoint) / (t - s)
        else:
            current = before.clone().requires_grad_(True)
            endpoint = flow_map(current, transport_condition, t=t, r=0.0)
            endpoint_velocity = (current - endpoint) / t
            if sample_mode == "flow_map1":
                step_velocity = endpoint_velocity
            else:
                step_endpoint = flow_map(current, transport_condition, t=t, r=s)
                step_velocity = (current - step_endpoint) / (t - s)
            loss = _representation_loss(endpoint, codec, e0, target_z0)
            loss_history.append(float(loss.detach()))
            gradient = torch.autograd.grad(loss, current, only_inputs=True)[0]
            gradient = _finite_tensor("representation gradient", gradient)
            delta_xt = step_size * normalize_per_sample_to_velocity_norm(gradient, step_velocity)

        state = before - (t - s) * (step_velocity.detach() + delta_xt)
        state = _finite_tensor("guided state", state.detach())

    state = _run_unguided_tail(flow_map, state, transport_condition, unguided)
    assert_guidance_stack_frozen(flow_map.generator, codec, e0)
    return GuidanceResult(
        latent=state,
        nfe=flow_map.nfe - initial_nfe,
        diagnostics={
            "guided_times": guided,
            "unguided_times": unguided,
            "sample_mode": sample_mode,
            "optimization_mode": optimization_mode,
            "num_optim_iters": num_optim_iters,
            "step_size": step_size,
            "uses_adam": optimization_mode == "official_adam",
            "adam_learning_rates": learning_rates,
            "loss_history": loss_history,
        },
    )


def sample_paper_algorithm_split(
    *,
    flow_map: CountedFlowMap,
    codec,
    e0,
    x_init: torch.Tensor,
    transport_condition: torch.Tensor,
    target_z0: torch.Tensor,
    guided_times: Sequence[float],
    unguided_times: Sequence[float],
    step_size: float,
) -> GuidanceResult:
    guided, unguided = _validate_fmrg_arguments(
        flow_map=flow_map,
        codec=codec,
        e0=e0,
        guided_times=guided_times,
        unguided_times=unguided_times,
        step_size=step_size,
    )
    initial_nfe = flow_map.nfe
    state = _finite_tensor("x_init", x_init.detach())
    loss_history: list[float] = []

    for t, s in zip(guided, guided[1:]):
        transported = flow_map(state, transport_condition, t=t, r=s).detach()
        step_velocity = (state - transported) / (t - s)
        split_state = transported.requires_grad_(True)
        endpoint = flow_map(split_state, transport_condition, t=s, r=0.0)
        loss = _representation_loss(endpoint, codec, e0, target_z0)
        loss_history.append(float(loss.detach()))
        gradient = torch.autograd.grad(loss, split_state, only_inputs=True)[0]
        gradient = _finite_tensor("representation gradient", gradient)
        correction = normalize_per_sample_to_velocity_norm(gradient, step_velocity)
        state = transported - (t - s) * step_size * correction
        state = _finite_tensor("guided state", state.detach())

    state = _run_unguided_tail(flow_map, state, transport_condition, unguided)
    assert_guidance_stack_frozen(flow_map.generator, codec, e0)
    return GuidanceResult(
        latent=state,
        nfe=flow_map.nfe - initial_nfe,
        diagnostics={
            "guided_times": guided,
            "unguided_times": unguided,
            "step_size": step_size,
            "loss_history": loss_history,
        },
    )


def normalize_per_sample_to_velocity_norm(gradient, velocity, eps: float = 1.0e-8):
    gradient_norm = gradient.flatten(1).norm(dim=1)
    velocity_norm = velocity.detach().flatten(1).norm(dim=1)
    scale = velocity_norm / gradient_norm.clamp_min(eps)
    view_shape = (gradient.shape[0],) + (1,) * (gradient.ndim - 1)
    normalized = gradient * scale.view(view_shape)
    return _finite_tensor("normalized representation gradient", normalized)


def project_fixed_radius(candidate, initial, eps: float = 1.0e-8):
    _validate_noise_pair(candidate, initial)
    candidate_norm = candidate.flatten(1).norm(dim=1)
    if torch.any(candidate_norm <= eps).item():
        raise ValueError("fixed-radius projection received a zero-norm candidate")
    initial_norm = initial.flatten(1).norm(dim=1)
    scale = initial_norm / candidate_norm
    projected = candidate * scale.view(candidate.shape[0], 1, 1, 1)
    return _finite_tensor("fixed-radius projection", projected)


def project_gaussian_typical_shell(candidate, *, delta, eps: float = 1.0e-8):
    delta_value = float(delta)
    if not math.isfinite(delta_value) or not 0.0 < delta_value < 1.0:
        raise ValueError(f"typical-shell projection requires 0 < delta < 1, got {delta!r}")
    _validate_noise_tensor("candidate", candidate)
    candidate_norm = candidate.flatten(1).norm(dim=1)
    if torch.any(candidate_norm <= eps).item():
        raise ValueError("typical-shell projection received a zero-norm candidate")
    dimension = math.prod(candidate.shape[1:])
    minimum = math.sqrt(dimension * (1.0 - delta_value))
    maximum = math.sqrt(dimension * (1.0 + delta_value))
    scale = torch.ones_like(candidate_norm)
    scale = torch.where(candidate_norm < minimum, minimum / candidate_norm, scale)
    scale = torch.where(candidate_norm > maximum, maximum / candidate_norm, scale)
    projected = candidate * scale.view(candidate.shape[0], 1, 1, 1)
    return _finite_tensor("typical-shell projection", projected)


def optimize_initial_noise(
    *,
    flow_map: CountedFlowMap,
    codec,
    e0,
    x_init: torch.Tensor,
    transport_condition: torch.Tensor,
    target_z0: torch.Tensor,
    num_updates: int,
    eta: float,
    projection: Literal["fixed_radius", "typical_shell"],
    typical_delta: float = 0.05,
) -> GuidanceResult:
    assert_guidance_stack_frozen(flow_map.generator, codec, e0)
    _validate_noise_tensor("x_init", x_init)
    if isinstance(num_updates, bool) or int(num_updates) != num_updates or num_updates < 0:
        raise ValueError(f"num_updates must be a non-negative integer, got {num_updates!r}")
    eta_value = float(eta)
    if eta_value not in {0.25, 0.5, 1.0, 2.0}:
        raise ValueError(f"eta must be one of {{0.25, 0.5, 1.0, 2.0}}, got {eta!r}")
    if projection not in {"fixed_radius", "typical_shell"}:
        raise ValueError(f"unsupported noise projection {projection!r}")
    if projection == "typical_shell":
        delta_value = float(typical_delta)
        if not math.isfinite(delta_value) or not 0.0 < delta_value < 1.0:
            raise ValueError(f"typical-shell projection requires 0 < delta < 1, got {typical_delta!r}")

    initial_nfe = flow_map.nfe
    initial = x_init.detach().clone()
    noise = initial.clone()
    loss_history: list[float] = []

    for _ in range(num_updates):
        active_noise = noise.detach().requires_grad_(True)
        endpoint = flow_map(active_noise, transport_condition, t=1.0, r=0.0)
        loss = _representation_loss(endpoint, codec, e0, target_z0)
        loss_history.append(float(loss.detach()))
        gradient = torch.autograd.grad(loss, active_noise, only_inputs=True)[0]
        gradient = _finite_tensor("representation gradient", gradient)
        gradient_norm = gradient.flatten(1).norm(dim=1)
        view_shape = (gradient.shape[0],) + (1,) * (gradient.ndim - 1)
        normalized = gradient / (gradient_norm + 1.0e-8).view(view_shape)
        candidate = active_noise.detach() - eta_value * normalized
        if projection == "fixed_radius":
            noise = project_fixed_radius(candidate, initial)
        else:
            noise = project_gaussian_typical_shell(candidate, delta=typical_delta)
        noise = _finite_tensor("projected noise", noise.detach())

    final_noise = noise.detach()
    final_endpoint = flow_map(final_noise, transport_condition, t=1.0, r=0.0)
    final_loss = _representation_loss(final_endpoint, codec, e0, target_z0)
    loss_history.append(float(final_loss.detach()))
    dimension = math.prod(initial.shape[1:])
    initial_norm = initial.flatten(1).norm(dim=1)
    final_norm = final_noise.flatten(1).norm(dim=1)
    assert_guidance_stack_frozen(flow_map.generator, codec, e0)
    return GuidanceResult(
        latent=final_endpoint.detach(),
        nfe=flow_map.nfe - initial_nfe,
        diagnostics={
            "projection": projection,
            "eta": eta_value,
            "num_updates": int(num_updates),
            "typical_delta": float(typical_delta),
            "initial_noise": initial,
            "final_noise": final_noise,
            "initial_norm": initial_norm,
            "final_norm": final_norm,
            "initial_norm_squared_per_dimension": initial_norm.square() / dimension,
            "final_norm_squared_per_dimension": final_norm.square() / dimension,
            "initial_final_cosine": F.cosine_similarity(initial.flatten(1), final_noise.flatten(1), dim=1),
            "update_norm": (final_noise - initial).flatten(1).norm(dim=1),
            "channel_mean": final_noise.mean(dim=(0, 2, 3)),
            "channel_std": final_noise.std(dim=(0, 2, 3), unbiased=False),
            "loss_history": loss_history,
        },
    )


def select_t_cut(candidate_reports, registered_thresholds) -> float | None:
    ordered = sorted(candidate_reports, key=lambda report: float(report["t_cut"]))
    for report in ordered:
        if all(_threshold_passes(report.get(field), requirement) for field, requirement in registered_thresholds.items()):
            return float(report["t_cut"])
    return None


def _validate_fmrg_arguments(*, flow_map, codec, e0, guided_times, unguided_times, step_size):
    assert_guidance_stack_frozen(flow_map.generator, codec, e0)
    if not math.isfinite(float(step_size)) or step_size <= 0.0:
        raise ValueError(f"step_size must be positive and finite, got {step_size!r}")
    guided = _validate_decreasing_schedule("guided_times", guided_times)
    unguided = _validate_decreasing_schedule("unguided_times", unguided_times)
    if not math.isclose(guided[0], 1.0) or not math.isclose(unguided[-1], 0.0):
        raise ValueError("guided_times must start at 1 and unguided_times must end at 0")
    if not math.isclose(guided[-1], unguided[0]):
        raise ValueError("guided_times and unguided_times must share the same t_cut")
    return guided, unguided


def _validate_decreasing_schedule(name, times):
    values = [float(value) for value in times]
    if len(values) < 2 or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"{name} must contain at least two finite values within [0,1]")
    if any(left <= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly decreasing")
    return values


def _representation_loss(endpoint, codec, e0, target_z0):
    _finite_tensor("endpoint lookahead", endpoint)
    image = codec.decode(endpoint)
    prediction = e0(normalize_for_e0(image))["embedding"]
    if prediction.shape != target_z0.shape:
        raise ValueError(
            f"E0 embedding and target_z0 must have the same shape, got {tuple(prediction.shape)} and {tuple(target_z0.shape)}"
        )
    loss = (1.0 - F.cosine_similarity(prediction, target_z0, dim=1)).mean()
    if not torch.isfinite(loss).item():
        raise FloatingPointError("representation loss is non-finite")
    return loss


def _run_unguided_tail(flow_map, state, condition, unguided_times):
    for t, r in zip(unguided_times, unguided_times[1:]):
        state = flow_map(state, condition, t=t, r=r).detach()
        _finite_tensor("unguided state", state)
    return state


def _finite_tensor(name, tensor):
    if tensor is None or not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"non-finite {name}")
    return tensor


def _validate_noise_pair(candidate, initial):
    _validate_noise_tensor("candidate", candidate)
    _validate_noise_tensor("initial", initial)
    if candidate.shape != initial.shape:
        raise ValueError(
            f"candidate and initial noise must have the same shape, got {tuple(candidate.shape)} and {tuple(initial.shape)}"
        )


def _validate_noise_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or not torch.is_floating_point(tensor):
        shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
        raise ValueError(f"{name} must be a floating tensor with shape [B,C,H,W], got {shape}")
    _finite_tensor(name, tensor)


def _freeze_module(module) -> None:
    module.eval()
    module.requires_grad_(False)
    for parameter in module.parameters():
        parameter.grad = None


def _threshold_passes(value, requirement) -> bool:
    if isinstance(requirement, dict):
        if value is None:
            return False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(numeric):
            return False
        if "min" in requirement and numeric < float(requirement["min"]):
            return False
        if "max" in requirement and numeric > float(requirement["max"]):
            return False
        if not set(requirement).issubset({"min", "max"}):
            raise ValueError(f"unsupported threshold keys: {sorted(set(requirement) - {'min', 'max'})}")
        return True
    return value == requirement
