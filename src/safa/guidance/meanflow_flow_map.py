from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


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


def select_t_cut(candidate_reports, registered_thresholds) -> float | None:
    ordered = sorted(candidate_reports, key=lambda report: float(report["t_cut"]))
    for report in ordered:
        if all(_threshold_passes(report.get(field), requirement) for field, requirement in registered_thresholds.items()):
            return float(report["t_cut"])
    return None


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
