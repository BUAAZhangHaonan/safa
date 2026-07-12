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
