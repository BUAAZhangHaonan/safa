#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import MISSING, asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from safa.training.projected_update import (
    AdaptiveMarginAdjustment,
    DualBudgetControlResult,
    ProjectionResult,
    TrustRegionScaleResult,
    apply_fm_anchor_trust_region_scaling,
    compute_adaptive_margin_adjustment,
    project_gradient_onto_fm_feasible_cone,
    update_dual_budget_controller,
)


@dataclass(frozen=True)
class ToyConfig:
    run_name: str
    output_dir: str
    device: str
    seed: int
    deltas_deg: list[float]
    methods: list[str]
    lambdas: list[float]
    soft_margins: list[float]
    steps: int
    batch_size: int
    eval_batch_size: int
    hidden_dim: int
    layers: int
    sigma: float
    k_classes: int
    sample_steps: int
    learning_rate: float
    fm_learning_rate: float
    repr_learning_rate: float
    weight_decay: float
    repr_relation_weight: float
    normalize_losses: bool
    calibration_batches: int
    eval_interval: int
    projection_eps: float
    adaptive_margin_mode: str | None = None
    adaptive_margin_target: float | None = None
    adaptive_margin_ema_beta: float | None = None
    adaptive_margin_step: float | None = None
    adaptive_margin_min: float | None = None
    adaptive_margin_max: float | None = None
    adaptive_margin_initial: float | None = None
    fm_delta_target: float | None = None
    line_search_max_backtracks: int | None = None
    line_search_contraction: float | None = None
    dual_lr: float | None = None
    trust_radius_initial: float | None = None
    trust_radius_min: float | None = None
    trust_radius_max: float | None = None


SUPPORTED_METHODS = {
    "fm_only",
    "repr_only",
    "weighted_sum",
    "projected_two_step",
    "pcgrad",
    "soft_margin_projected",
    "adaptive_margin_projected",
    "adaptive_trust_projected",
    "line_search_projected",
}
NORM_EPS = 1.0e-12


@dataclass
class AdaptiveMarginState:
    margin: float
    ema_fm_loss: float | None


@dataclass
class AdaptiveTrustState:
    margin_state: AdaptiveMarginState
    dual_value: float
    trust_radius: float


@dataclass(frozen=True)
class LineSearchAcceptanceResult:
    attempts: int
    alpha: float
    accepted: bool
    flow_delta: float
    repr_delta: float


def load_config(path: str | Path) -> ToyConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Toy config must be a JSON object")
    config_fields = {item.name for item in fields(ToyConfig)}
    required = {
        item.name
        for item in fields(ToyConfig)
        if item.default is MISSING and item.default_factory is MISSING
    }
    actual = set(payload)
    missing = sorted(required - actual)
    extra = sorted(actual - config_fields)
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    if extra:
        raise KeyError(f"Unexpected config keys: {extra}")
    config = ToyConfig(**payload)
    validate_config(config)
    return config


def validate_config(config: ToyConfig) -> None:
    if not config.run_name:
        raise ValueError("run_name must be non-empty")
    if not config.output_dir:
        raise ValueError("output_dir must be non-empty")
    if not config.deltas_deg:
        raise ValueError("deltas_deg must be non-empty")
    if not config.methods:
        raise ValueError("methods must be non-empty")
    invalid_methods = sorted(set(config.methods) - SUPPORTED_METHODS)
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {invalid_methods}")
    if not config.lambdas:
        raise ValueError("lambdas must be non-empty")
    if not config.soft_margins:
        raise ValueError("soft_margins must be non-empty")
    positive_fields = {
        "steps": config.steps,
        "batch_size": config.batch_size,
        "eval_batch_size": config.eval_batch_size,
        "hidden_dim": config.hidden_dim,
        "layers": config.layers,
        "k_classes": config.k_classes,
        "sample_steps": config.sample_steps,
        "calibration_batches": config.calibration_batches,
        "eval_interval": config.eval_interval,
    }
    for name, value in positive_fields.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.k_classes < 2:
        raise ValueError("k_classes must be at least 2")
    scalar_fields = {
        "sigma": config.sigma,
        "learning_rate": config.learning_rate,
        "fm_learning_rate": config.fm_learning_rate,
        "repr_learning_rate": config.repr_learning_rate,
        "weight_decay": config.weight_decay,
        "repr_relation_weight": config.repr_relation_weight,
        "projection_eps": config.projection_eps,
    }
    for name, value in scalar_fields.items():
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if config.sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if config.learning_rate <= 0.0 or config.fm_learning_rate <= 0.0 or config.repr_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if config.repr_relation_weight < 0.0:
        raise ValueError("repr_relation_weight must be non-negative")
    if config.projection_eps < 0.0:
        raise ValueError("projection_eps must be non-negative")
    for value in config.lambdas:
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("all lambdas must be positive finite numbers")
    for value in config.soft_margins:
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("all soft_margins must be finite non-negative numbers")
    _validate_adaptive_margin_config(config)
    _validate_adaptive_trust_config(config)
    _validate_line_search_config(config)


class ToyVectorField(nn.Module):
    def __init__(self, hidden_dim: int, layers: int) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        input_dim = 5
        for layer_index in range(layers):
            modules.append(nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim))
            modules.append(nn.SiLU())
        modules.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*modules)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if x_t.shape != z.shape or x_t.ndim != 2 or x_t.shape[1] != 2:
            raise ValueError("x_t and z must have shape [batch, 2]")
        if t.shape != (x_t.shape[0], 1):
            raise ValueError("t must have shape [batch, 1]")
        return self.net(torch.cat([x_t, t, z], dim=1))


def run_experiment_grid(config: ToyConfig) -> dict[str, Any]:
    validate_config(config)
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    experiments: list[dict[str, Any]] = []
    final_points: list[dict[str, Any]] = []
    for delta_deg in config.deltas_deg:
        for method in config.methods:
            for lambda_repr, soft_margin in _method_parameter_grid(config, method):
                result = run_single_experiment(
                    config,
                    delta_deg=float(delta_deg),
                    method=method,
                    lambda_repr=float(lambda_repr),
                    soft_margin=float(soft_margin),
                )
                experiments.append(result)
                final_points.append(result["final"])
                with metrics_path.open("a", encoding="utf-8") as handle:
                    for row in result["metrics"]:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "run_name": config.run_name,
        "num_experiments": len(experiments),
        "experiments": [
            {
                "delta_deg": result["delta_deg"],
                "method": result["method"],
                "lambda_repr": result["lambda_repr"],
                "soft_margin": result["soft_margin"],
                "initial": result["initial"],
                "final": result["final"],
            }
            for result in experiments
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _plot_curves(run_dir / "metrics.jsonl", run_dir / "curves.png")
    _plot_trajectory(final_points, run_dir / "trajectory.png")
    return summary


def run_single_experiment(
    config: ToyConfig,
    delta_deg: float,
    method: str,
    lambda_repr: float,
    soft_margin: float,
) -> dict[str, Any]:
    validate_config(config)
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    if method in {"adaptive_margin_projected", "adaptive_trust_projected"}:
        if lambda_repr != 1.0:
            raise ValueError(f"{method} uses fixed lambda_repr=1.0")
        if config.adaptive_margin_initial is None:
            raise ValueError(f"{method} requires adaptive_margin_initial")
        if not math.isclose(float(soft_margin), float(config.adaptive_margin_initial), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{method} soft_margin must equal adaptive_margin_initial")
    device = torch.device(config.device)
    seed_offset = _stable_seed_offset(method, delta_deg, lambda_repr, soft_margin)
    train_generator = torch.Generator(device=device).manual_seed(config.seed + seed_offset)
    eval_generator = torch.Generator(device=device).manual_seed(config.seed + 100_000 + seed_offset)
    model = ToyVectorField(hidden_dim=config.hidden_dim, layers=config.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    fm_optimizer = torch.optim.AdamW(model.parameters(), lr=config.fm_learning_rate, weight_decay=config.weight_decay)
    eval_batch = _sample_batch(config, delta_deg, config.eval_batch_size, device, eval_generator)
    flow_scale, repr_scale = _calibrate_loss_scales(config, model, delta_deg, device, train_generator)
    adaptive_state = _new_adaptive_margin_state(config) if method == "adaptive_margin_projected" else None
    adaptive_trust_state = _new_adaptive_trust_state(config) if method == "adaptive_trust_projected" else None
    initial = _evaluate_model(
        config,
        model,
        eval_batch,
        delta_deg,
        method,
        lambda_repr,
        soft_margin,
        step=0,
        train_stats=_initial_step_stats(method, soft_margin),
    )

    metrics = [initial]
    stat_window = _new_stat_window()
    for step in range(1, config.steps + 1):
        batch = _sample_batch(config, delta_deg, config.batch_size, device, train_generator)
        if method == "fm_only":
            step_stats = _step_fm_only(config, model, optimizer, batch, flow_scale)
        elif method == "repr_only":
            step_stats = _step_repr_only(config, model, optimizer, batch, repr_scale)
        elif method == "weighted_sum":
            step_stats = _step_weighted_sum(config, model, optimizer, batch, flow_scale, repr_scale, lambda_repr)
        elif method == "projected_two_step":
            step_stats = _step_projected_two_step(
                config,
                model,
                fm_optimizer,
                batch,
                flow_scale,
                repr_scale,
                lambda_repr,
                soft_margin=0.0,
            )
        elif method == "soft_margin_projected":
            step_stats = _step_projected_two_step(
                config,
                model,
                fm_optimizer,
                batch,
                flow_scale,
                repr_scale,
                lambda_repr,
                soft_margin=soft_margin,
            )
        elif method == "adaptive_margin_projected":
            step_stats = _step_projected_two_step(
                config,
                model,
                fm_optimizer,
                batch,
                flow_scale,
                repr_scale,
                lambda_repr,
                soft_margin=soft_margin,
                adaptive_state=adaptive_state,
            )
        elif method == "adaptive_trust_projected":
            step_stats = _step_projected_two_step(
                config,
                model,
                fm_optimizer,
                batch,
                flow_scale,
                repr_scale,
                lambda_repr,
                soft_margin=soft_margin,
                adaptive_state=adaptive_trust_state.margin_state,
                adaptive_trust_state=adaptive_trust_state,
            )
        elif method == "line_search_projected":
            step_stats = _step_line_search_projected(
                config,
                model,
                fm_optimizer,
                batch,
                flow_scale,
                repr_scale,
                lambda_repr,
                soft_margin=soft_margin,
            )
        elif method == "pcgrad":
            step_stats = _step_pcgrad(config, model, optimizer, batch, flow_scale, repr_scale, lambda_repr)
        else:
            raise RuntimeError(f"Unhandled method: {method}")
        _accumulate_stats(stat_window, step_stats)
        if step % config.eval_interval == 0 or step == config.steps:
            metrics.append(
                _evaluate_model(
                    config,
                    model,
                    eval_batch,
                    delta_deg,
                    method,
                    lambda_repr,
                    soft_margin,
                    step=step,
                    train_stats=_summarize_stat_window(stat_window),
                )
            )
            stat_window = _new_stat_window()

    final = metrics[-1]
    return {
        "delta_deg": delta_deg,
        "method": method,
        "lambda_repr": lambda_repr,
        "soft_margin": soft_margin,
        "initial": initial,
        "final": final,
        "metrics": metrics,
    }


def project_gradient_with_soft_margin(
    g_repr: list[torch.Tensor],
    g_fm: list[torch.Tensor],
    epsilon: float,
    eps: float,
) -> ProjectionResult:
    _validate_gradient_lists(g_repr, g_fm)
    if not isinstance(epsilon, (float, int)) or not math.isfinite(float(epsilon)) or float(epsilon) < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    if float(epsilon) == 0.0:
        return project_gradient_onto_fm_feasible_cone(g_repr, g_fm, eps=eps)

    dot_before = _dot(g_repr, g_fm)
    fm_norm_squared = _squared_norm(g_fm)
    fm_norm = torch.sqrt(fm_norm_squared)
    repr_norm = torch.sqrt(_squared_norm(g_repr))
    eps_tensor = torch.as_tensor(eps, dtype=fm_norm.dtype, device=fm_norm.device)
    lower_bound = -float(epsilon) * repr_norm * fm_norm
    projection_applied = bool((dot_before < lower_bound).item() and (fm_norm > eps_tensor).item())
    if projection_applied:
        coefficient = (dot_before - lower_bound) / fm_norm_squared
        projected_gradients = [repr_grad - coefficient * fm_grad for repr_grad, fm_grad in zip(g_repr, g_fm)]
    else:
        projected_gradients = [repr_grad.clone() for repr_grad in g_repr]
    dot_after = _dot(projected_gradients, g_fm)
    if projection_applied and not torch.allclose(dot_after, lower_bound, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Soft-margin projected gradient does not satisfy the requested FM budget")
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


def _method_parameter_grid(config: ToyConfig, method: str) -> list[tuple[float, float]]:
    if method in {"fm_only", "repr_only"}:
        return [(config.lambdas[0], config.soft_margins[0])]
    if method == "soft_margin_projected":
        return [(lambda_repr, margin) for lambda_repr in config.lambdas for margin in config.soft_margins]
    if method in {"adaptive_margin_projected", "adaptive_trust_projected"}:
        if config.adaptive_margin_initial is None:
            raise ValueError(f"{method} requires adaptive_margin_initial")
        return [(1.0, float(config.adaptive_margin_initial))]
    if method == "line_search_projected":
        return [(1.0, 0.0)]
    return [(lambda_repr, config.soft_margins[0]) for lambda_repr in config.lambdas]


def _sample_batch(
    config: ToyConfig,
    delta_deg: float,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    labels = torch.randint(0, config.k_classes, (batch_size,), generator=generator, device=device)
    z = _label_to_unit_vectors(labels, config.k_classes)
    x0 = torch.randn(batch_size, 2, generator=generator, device=device)
    noise = torch.randn(batch_size, 2, generator=generator, device=device)
    x1_center = _rotate(z, math.radians(delta_deg))
    x1 = x1_center + config.sigma * noise
    t = torch.rand(batch_size, 1, generator=generator, device=device)
    x_t = (1.0 - t) * x0 + t * x1
    target_velocity = x1 - x0
    return {
        "labels": labels,
        "z": z,
        "x0": x0,
        "x1": x1,
        "x1_center": x1_center,
        "t": t,
        "x_t": x_t,
        "target_velocity": target_velocity,
    }


def _label_to_unit_vectors(labels: torch.Tensor, k_classes: int) -> torch.Tensor:
    angles = 2.0 * math.pi * labels.to(torch.float32) / float(k_classes)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


def _rotate(vectors: torch.Tensor, angle_rad: float) -> torch.Tensor:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rotation = torch.tensor([[c, -s], [s, c]], dtype=vectors.dtype, device=vectors.device)
    return vectors @ rotation.T


def _compute_losses(config: ToyConfig, model: ToyVectorField, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    velocity_pred = model(batch["x_t"], batch["t"], batch["z"])
    flow_loss = (velocity_pred - batch["target_velocity"]).pow(2).mean()
    x_hat = _sample_model(config, model, batch["z"], batch["x0"])
    h = _normalize_nonzero(x_hat)
    point_loss = (1.0 - (h * batch["z"]).sum(dim=1)).mean()
    relation_loss = _gram_relation_loss(h, batch["z"])
    repr_loss = point_loss + config.repr_relation_weight * relation_loss
    return {
        "flow": flow_loss,
        "repr": repr_loss,
        "point": point_loss,
        "relation": relation_loss,
        "x_hat": x_hat,
        "h": h,
    }


def _sample_model(config: ToyConfig, model: ToyVectorField, z: torch.Tensor, x_init: torch.Tensor) -> torch.Tensor:
    x = x_init
    dt = 1.0 / float(config.sample_steps)
    for step in range(config.sample_steps):
        t = torch.full((z.shape[0], 1), float(step) / float(config.sample_steps), dtype=z.dtype, device=z.device)
        x = x + dt * model(x, t, z)
    return x


def _normalize_nonzero(values: torch.Tensor) -> torch.Tensor:
    norms = values.norm(dim=1, keepdim=True)
    if not torch.isfinite(values).all() or not torch.isfinite(norms).all():
        raise FloatingPointError("Cannot normalize non-finite toy representations")
    if bool((norms <= NORM_EPS).any().item()):
        raise FloatingPointError("Cannot normalize zero-norm toy representations")
    return values / norms


def _gram_relation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("pred and target embeddings must have matching [batch, dim] shape")
    if pred.shape[0] <= 1:
        raise ValueError("Gram relation loss requires batch size > 1")
    diff = pred @ pred.T - target @ target.T
    mask = ~torch.eye(pred.shape[0], dtype=torch.bool, device=pred.device)
    return diff[mask].pow(2).mean()


def _calibrate_loss_scales(
    config: ToyConfig,
    model: ToyVectorField,
    delta_deg: float,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not config.normalize_losses:
        one = torch.ones((), dtype=torch.float32, device=device)
        return one, one
    flow_values: list[torch.Tensor] = []
    repr_values: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(config.calibration_batches):
            batch = _sample_batch(config, delta_deg, config.batch_size, device, generator)
            losses = _compute_losses(config, model, batch)
            flow_values.append(losses["flow"].detach())
            repr_values.append(losses["repr"].detach())
    flow_scale = torch.stack(flow_values).mean()
    repr_scale = torch.stack(repr_values).mean()
    if not torch.isfinite(flow_scale) or not torch.isfinite(repr_scale) or flow_scale <= 0 or repr_scale <= 0:
        raise FloatingPointError("Initial toy loss scales must be positive and finite")
    return flow_scale.detach(), repr_scale.detach()


def _step_fm_only(
    config: ToyConfig,
    model: ToyVectorField,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    flow_scale: torch.Tensor,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    losses = _compute_losses(config, model, batch)
    (losses["flow"] / flow_scale).backward()
    optimizer.step()
    return _empty_step_stats()


def _step_repr_only(
    config: ToyConfig,
    model: ToyVectorField,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    repr_scale: torch.Tensor,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    losses = _compute_losses(config, model, batch)
    (losses["repr"] / repr_scale).backward()
    optimizer.step()
    return _empty_step_stats()


def _step_weighted_sum(
    config: ToyConfig,
    model: ToyVectorField,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    flow_scale: torch.Tensor,
    repr_scale: torch.Tensor,
    lambda_repr: float,
) -> dict[str, float]:
    losses = _compute_losses(config, model, batch)
    flow_objective = losses["flow"] / flow_scale
    repr_objective = lambda_repr * losses["repr"] / repr_scale
    g_fm, g_repr = _task_gradients(model, flow_objective, repr_objective)
    dot_before = _dot(g_repr, g_fm)
    optimizer.zero_grad(set_to_none=True)
    (flow_objective + repr_objective).backward()
    optimizer.step()
    return _stats_from_gradients(g_repr, g_fm, dot_before, dot_before, actual_fm_delta=0.0)


def _step_projected_two_step(
    config: ToyConfig,
    model: ToyVectorField,
    fm_optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    flow_scale: torch.Tensor,
    repr_scale: torch.Tensor,
    lambda_repr: float,
    soft_margin: float,
    adaptive_state: AdaptiveMarginState | None = None,
    adaptive_trust_state: AdaptiveTrustState | None = None,
) -> dict[str, float]:
    fm_optimizer.zero_grad(set_to_none=True)
    losses = _compute_losses(config, model, batch)
    (losses["flow"] / flow_scale).backward()
    fm_optimizer.step()

    post_losses = _compute_losses(config, model, batch)
    flow_objective = post_losses["flow"] / flow_scale
    repr_objective = lambda_repr * post_losses["repr"] / repr_scale
    flow_guard_value = float(flow_objective.detach().cpu())
    g_fm, g_repr = _task_gradients(model, flow_objective, repr_objective)
    effective_margin = soft_margin
    adaptive_normalized_fm_loss = 0.0
    adaptive_margin_baseline = 0.0
    adaptive_margin_direction = 0.0
    if adaptive_state is not None:
        adjustment = _update_adaptive_margin(config, adaptive_state, flow_guard_value)
        effective_margin = adjustment.next_margin
        adaptive_normalized_fm_loss = adjustment.normalized_fm_loss
        adaptive_margin_baseline = adjustment.baseline
        adaptive_margin_direction = _adaptive_margin_direction_value(adjustment.direction)
    if effective_margin == 0.0:
        projection = project_gradient_onto_fm_feasible_cone(g_repr, g_fm, eps=config.projection_eps)
    else:
        projection = project_gradient_with_soft_margin(
            g_repr,
            g_fm,
            epsilon=effective_margin,
            eps=config.projection_eps,
        )
    trust_result = None
    if adaptive_trust_state is not None:
        trust_result = apply_fm_anchor_trust_region_scaling(
            projected_gradients=projection.projected_gradients,
            g_fm=g_fm,
            trust_radius=adaptive_trust_state.trust_radius,
            eps=config.projection_eps,
        )
        step_gradients = trust_result.scaled_gradients
    else:
        step_gradients = projection.projected_gradients
    scaled_dot_after = _dot(step_gradients, g_fm)
    _manual_parameter_step(model, step_gradients, config.repr_learning_rate)
    after_losses = _compute_losses(config, model, batch)
    actual_fm_delta = float((after_losses["flow"] / flow_scale).detach().cpu()) - flow_guard_value
    dual_update = None
    if adaptive_trust_state is not None:
        if config.fm_delta_target is None:
            raise ValueError("adaptive_trust_projected requires fm_delta_target")
        if config.dual_lr is None:
            raise ValueError("adaptive_trust_projected requires dual_lr")
        if config.trust_radius_min is None or config.trust_radius_max is None:
            raise ValueError("adaptive_trust_projected requires trust_radius_min and trust_radius_max")
        dual_update = update_dual_budget_controller(
            current_dual_value=adaptive_trust_state.dual_value,
            current_trust_radius=adaptive_trust_state.trust_radius,
            actual_fm_delta=actual_fm_delta,
            fm_delta_target=config.fm_delta_target,
            dual_lr=config.dual_lr,
            trust_radius_min=config.trust_radius_min,
            trust_radius_max=config.trust_radius_max,
        )
        adaptive_trust_state.dual_value = dual_update.next_dual_value
        adaptive_trust_state.trust_radius = dual_update.next_trust_radius
    return _stats_from_projection(
        projection,
        actual_fm_delta=actual_fm_delta,
        adaptive_margin=effective_margin,
        adaptive_normalized_fm_loss=adaptive_normalized_fm_loss,
        adaptive_margin_baseline=adaptive_margin_baseline,
        adaptive_margin_direction=adaptive_margin_direction,
        trust_result=trust_result,
        dual_update=dual_update,
        fm_delta_target=0.0 if config.fm_delta_target is None else float(config.fm_delta_target),
        scaled_dot_after=scaled_dot_after,
        repr_learning_rate=config.repr_learning_rate,
    )


def _step_line_search_projected(
    config: ToyConfig,
    model: ToyVectorField,
    fm_optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    flow_scale: torch.Tensor,
    repr_scale: torch.Tensor,
    lambda_repr: float,
    soft_margin: float,
) -> dict[str, float]:
    if not math.isclose(float(soft_margin), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("line_search_projected requires soft_margins[0] to equal 0.0")
    fm_delta_target = _require_finite_scalar(config.fm_delta_target, "fm_delta_target", min_value=0.0)
    max_backtracks = _require_positive_int(config.line_search_max_backtracks, "line_search_max_backtracks")
    contraction = _require_open_unit_scalar(config.line_search_contraction, "line_search_contraction")

    fm_optimizer.zero_grad(set_to_none=True)
    losses = _compute_losses(config, model, batch)
    (losses["flow"] / flow_scale).backward()
    fm_optimizer.step()

    post_losses = _compute_losses(config, model, batch)
    flow_objective = post_losses["flow"] / flow_scale
    repr_objective = lambda_repr * post_losses["repr"] / repr_scale
    flow_before = float(flow_objective.detach().cpu())
    repr_before = float(repr_objective.detach().cpu())
    g_fm, g_repr = _task_gradients(model, flow_objective, repr_objective)
    projection = project_gradient_onto_fm_feasible_cone(g_repr, g_fm, eps=config.projection_eps)
    theta_half = _copy_parameters(model)

    def evaluate_candidate(alpha: float) -> tuple[float, float]:
        _restore_parameters(model, theta_half)
        _manual_parameter_step(model, projection.projected_gradients, config.repr_learning_rate * alpha)
        candidate_losses = _compute_losses(config, model, batch)
        return (
            float((candidate_losses["flow"] / flow_scale).detach().cpu()),
            float((lambda_repr * candidate_losses["repr"] / repr_scale).detach().cpu()),
        )

    line_search = _find_line_search_acceptance(
        normalized_flow_before=flow_before,
        normalized_repr_before=repr_before,
        fm_delta_target=fm_delta_target,
        max_backtracks=max_backtracks,
        contraction=contraction,
        evaluate_candidate=evaluate_candidate,
    )
    if not line_search.accepted:
        _restore_parameters(model, theta_half)
        raise RuntimeError(
            "line_search_projected failed to find an acceptable representation step "
            f"after {line_search.attempts} attempts; "
            f"last_flow_delta={line_search.flow_delta:.6g}, "
            f"last_repr_delta={line_search.repr_delta:.6g}"
        )

    stats = _stats_from_projection(
        projection,
        actual_fm_delta=line_search.flow_delta,
        fm_delta_target=fm_delta_target,
        scaled_dot_after=_dot(projection.projected_gradients, g_fm),
        repr_learning_rate=config.repr_learning_rate * line_search.alpha,
    )
    stats.update(
        {
            "line_search_attempts": float(line_search.attempts),
            "line_search_alpha": float(line_search.alpha),
            "line_search_accepted": 1.0,
            "line_search_flow_delta": float(line_search.flow_delta),
            "line_search_repr_delta": float(line_search.repr_delta),
        }
    )
    return stats


def _step_pcgrad(
    config: ToyConfig,
    model: ToyVectorField,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    flow_scale: torch.Tensor,
    repr_scale: torch.Tensor,
    lambda_repr: float,
) -> dict[str, float]:
    losses = _compute_losses(config, model, batch)
    flow_objective = losses["flow"] / flow_scale
    repr_objective = lambda_repr * losses["repr"] / repr_scale
    g_fm, g_repr = _task_gradients(model, flow_objective, repr_objective)
    dot_before = _dot(g_repr, g_fm)
    pc_fm = [grad.clone() for grad in g_fm]
    pc_repr = [grad.clone() for grad in g_repr]
    if bool((dot_before < 0).item()):
        fm_norm_sq = _squared_norm(g_fm)
        repr_norm_sq = _squared_norm(g_repr)
        if fm_norm_sq <= config.projection_eps or repr_norm_sq <= config.projection_eps:
            raise FloatingPointError("PCGrad received a near-zero task gradient")
        pc_fm = [fm_grad - dot_before / repr_norm_sq * repr_grad for fm_grad, repr_grad in zip(g_fm, g_repr)]
        pc_repr = [repr_grad - dot_before / fm_norm_sq * fm_grad for repr_grad, fm_grad in zip(g_repr, g_fm)]
    combined = [fm_grad + repr_grad for fm_grad, repr_grad in zip(pc_fm, pc_repr)]
    dot_after = _dot(pc_repr, pc_fm)
    _assign_gradients(model, combined)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return _stats_from_gradients(g_repr, g_fm, dot_before, dot_after, actual_fm_delta=0.0)


def _task_gradients(
    model: ToyVectorField,
    flow_objective: torch.Tensor,
    repr_objective: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    flow_objective.backward(retain_graph=True)
    g_fm = _copy_gradients(model)
    model.zero_grad(set_to_none=True)
    repr_objective.backward(retain_graph=True)
    g_repr = _copy_gradients(model)
    model.zero_grad(set_to_none=True)
    return g_fm, g_repr


def _copy_gradients(model: ToyVectorField) -> list[torch.Tensor]:
    gradients: list[torch.Tensor] = []
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("Toy model parameter is missing a task gradient")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Toy task gradient is non-finite")
        gradients.append(parameter.grad.detach().clone())
    return gradients


def _assign_gradients(model: ToyVectorField, gradients: list[torch.Tensor]) -> None:
    parameters = list(model.parameters())
    if len(parameters) != len(gradients):
        raise ValueError("Gradient list length does not match toy model parameters")
    for parameter, gradient in zip(parameters, gradients):
        if parameter.shape != gradient.shape:
            raise ValueError("Gradient shape does not match toy model parameter shape")
        parameter.grad = gradient.detach().clone()


def _manual_parameter_step(model: ToyVectorField, gradients: list[torch.Tensor], learning_rate: float) -> None:
    parameters = list(model.parameters())
    if len(parameters) != len(gradients):
        raise ValueError("Gradient list length does not match toy model parameters")
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            if parameter.shape != gradient.shape:
                raise ValueError("Gradient shape does not match toy model parameter shape")
            parameter.add_(gradient, alpha=-learning_rate)
    model.zero_grad(set_to_none=True)


def _copy_parameters(model: ToyVectorField) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def _restore_parameters(model: ToyVectorField, values: list[torch.Tensor]) -> None:
    parameters = list(model.parameters())
    if len(parameters) != len(values):
        raise ValueError("Parameter snapshot length does not match toy model parameters")
    with torch.no_grad():
        for parameter, value in zip(parameters, values):
            if parameter.shape != value.shape:
                raise ValueError("Parameter snapshot shape does not match toy model parameter shape")
            parameter.copy_(value)
    model.zero_grad(set_to_none=True)


def _find_line_search_acceptance(
    normalized_flow_before: float,
    normalized_repr_before: float,
    fm_delta_target: float,
    max_backtracks: int,
    contraction: float,
    evaluate_candidate: Callable[[float], tuple[float, float]],
) -> LineSearchAcceptanceResult:
    normalized_flow_before = _require_finite_scalar(normalized_flow_before, "normalized_flow_before")
    normalized_repr_before = _require_finite_scalar(normalized_repr_before, "normalized_repr_before")
    fm_delta_target = _require_finite_scalar(fm_delta_target, "fm_delta_target", min_value=0.0)
    max_backtracks = _require_positive_int(max_backtracks, "line_search_max_backtracks")
    contraction = _require_open_unit_scalar(contraction, "line_search_contraction")

    alpha = 1.0
    last_flow_delta = math.inf
    last_repr_delta = math.inf
    for attempt in range(1, max_backtracks + 1):
        normalized_flow_after, normalized_repr_after = evaluate_candidate(alpha)
        normalized_flow_after = _require_finite_scalar(normalized_flow_after, "normalized_flow_after")
        normalized_repr_after = _require_finite_scalar(normalized_repr_after, "normalized_repr_after")
        flow_delta = normalized_flow_after - normalized_flow_before
        repr_delta = normalized_repr_after - normalized_repr_before
        last_flow_delta = flow_delta
        last_repr_delta = repr_delta
        if flow_delta <= fm_delta_target and repr_delta < 0.0:
            return LineSearchAcceptanceResult(
                attempts=attempt,
                alpha=alpha,
                accepted=True,
                flow_delta=flow_delta,
                repr_delta=repr_delta,
            )
        alpha *= contraction

    return LineSearchAcceptanceResult(
        attempts=max_backtracks,
        alpha=alpha / contraction,
        accepted=False,
        flow_delta=last_flow_delta,
        repr_delta=last_repr_delta,
    )


def _evaluate_model(
    config: ToyConfig,
    model: ToyVectorField,
    batch: dict[str, torch.Tensor],
    delta_deg: float,
    method: str,
    lambda_repr: float,
    soft_margin: float,
    step: int,
    train_stats: dict[str, float] | None = None,
) -> dict[str, float | str]:
    with torch.no_grad():
        losses = _compute_losses(config, model, batch)
        x_hat = losses["x_hat"]
        h = losses["h"]
        valid_fm_loss = float(losses["flow"].detach().cpu())
        generation_mse = float((x_hat - batch["x1"]).pow(2).mean().detach().cpu())
        center_mse = float((x_hat - batch["x1_center"]).pow(2).mean().detach().cpu())
        repr_point = float(losses["point"].detach().cpu())
        repr_relation = float(losses["relation"].detach().cpu())
        repr_cosine = float((h * batch["z"]).sum(dim=1).mean().detach().cpu())
        info_nce = float(_info_nce_loss(h, batch["z"]).detach().cpu())
        mean_error = float((x_hat.mean(dim=0) - batch["x1"].mean(dim=0)).pow(2).sum().sqrt().detach().cpu())
        cov_error = float((_covariance(x_hat) - _covariance(batch["x1"])).pow(2).mean().sqrt().detach().cpu())
        mmd = float(_rbf_mmd(x_hat, batch["x1"]).detach().cpu())
    row: dict[str, float | str] = {
        "delta_deg": float(delta_deg),
        "method": method,
        "lambda_repr": float(lambda_repr),
        "soft_margin": float(soft_margin),
        "step": float(step),
        "valid_fm_loss": valid_fm_loss,
        "generation_mse_to_x1": generation_mse,
        "generation_mse_to_x1_center": center_mse,
        "repr_point_loss": repr_point,
        "repr_relation_loss": repr_relation,
        "repr_cosine_mean": repr_cosine,
        "info_nce_loss": info_nce,
        "mean_error": mean_error,
        "cov_error": cov_error,
        "mmd_rbf": mmd,
    }
    if train_stats is None:
        row.update(_empty_step_stats())
    else:
        row.update(train_stats)
    return row


def _info_nce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = pred @ target.T / 0.07
    labels = torch.arange(pred.shape[0], dtype=torch.long, device=pred.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def _covariance(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.T @ centered / float(values.shape[0] - 1)


def _rbf_mmd(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    gamma = torch.tensor(1.0, dtype=left.dtype, device=left.device)
    xx = torch.exp(-gamma * torch.cdist(left, left).pow(2)).mean()
    yy = torch.exp(-gamma * torch.cdist(right, right).pow(2)).mean()
    xy = torch.exp(-gamma * torch.cdist(left, right).pow(2)).mean()
    return xx + yy - 2.0 * xy


def _new_stat_window() -> dict[str, list[float]]:
    return {
        "conflict": [],
        "dot_before": [],
        "dot_after": [],
        "actual_fm_delta_after_repr_step": [],
        "projected_repr_norm_ratio": [],
        "adaptive_margin": [],
        "adaptive_normalized_fm_loss": [],
        "adaptive_margin_baseline": [],
        "adaptive_margin_direction": [],
        "trust_radius_used": [],
        "trust_radius_next": [],
        "trust_scale": [],
        "trust_region_active": [],
        "dual_value_used": [],
        "dual_value_next": [],
        "fm_budget_violation": [],
        "fm_delta_target": [],
        "scaled_dot_after": [],
        "repr_step_fm_first_order_effect": [],
        "line_search_attempts": [],
        "line_search_alpha": [],
        "line_search_accepted": [],
        "line_search_flow_delta": [],
        "line_search_repr_delta": [],
    }


def _empty_step_stats() -> dict[str, float]:
    return {
        "conflict_fraction": 0.0,
        "dot_before_mean": 0.0,
        "dot_after_mean": 0.0,
        "actual_fm_delta_after_repr_step": 0.0,
        "projected_repr_norm_ratio": 0.0,
        "adaptive_margin": 0.0,
        "adaptive_normalized_fm_loss": 0.0,
        "adaptive_margin_baseline": 0.0,
        "adaptive_margin_direction": 0.0,
        "trust_radius_used": 0.0,
        "trust_radius_next": 0.0,
        "trust_scale": 0.0,
        "trust_region_active": 0.0,
        "dual_value_used": 0.0,
        "dual_value_next": 0.0,
        "fm_budget_violation": 0.0,
        "fm_delta_target": 0.0,
        "scaled_dot_after_mean": 0.0,
        "repr_step_fm_first_order_effect": 0.0,
        "line_search_attempts": 0.0,
        "line_search_alpha": 0.0,
        "line_search_accepted": 0.0,
        "line_search_flow_delta": 0.0,
        "line_search_repr_delta": 0.0,
    }


def _accumulate_stats(window: dict[str, list[float]], step_stats: dict[str, float]) -> None:
    window["conflict"].append(float(step_stats["conflict_fraction"]))
    window["dot_before"].append(float(step_stats["dot_before_mean"]))
    window["dot_after"].append(float(step_stats["dot_after_mean"]))
    window["actual_fm_delta_after_repr_step"].append(float(step_stats["actual_fm_delta_after_repr_step"]))
    window["projected_repr_norm_ratio"].append(float(step_stats["projected_repr_norm_ratio"]))
    window["adaptive_margin"].append(float(step_stats["adaptive_margin"]))
    window["adaptive_normalized_fm_loss"].append(float(step_stats["adaptive_normalized_fm_loss"]))
    window["adaptive_margin_baseline"].append(float(step_stats["adaptive_margin_baseline"]))
    window["adaptive_margin_direction"].append(float(step_stats["adaptive_margin_direction"]))
    window["trust_radius_used"].append(float(step_stats["trust_radius_used"]))
    window["trust_radius_next"].append(float(step_stats["trust_radius_next"]))
    window["trust_scale"].append(float(step_stats["trust_scale"]))
    window["trust_region_active"].append(float(step_stats["trust_region_active"]))
    window["dual_value_used"].append(float(step_stats["dual_value_used"]))
    window["dual_value_next"].append(float(step_stats["dual_value_next"]))
    window["fm_budget_violation"].append(float(step_stats["fm_budget_violation"]))
    window["fm_delta_target"].append(float(step_stats["fm_delta_target"]))
    window["scaled_dot_after"].append(float(step_stats["scaled_dot_after_mean"]))
    window["repr_step_fm_first_order_effect"].append(float(step_stats["repr_step_fm_first_order_effect"]))
    window["line_search_attempts"].append(float(step_stats["line_search_attempts"]))
    window["line_search_alpha"].append(float(step_stats["line_search_alpha"]))
    window["line_search_accepted"].append(float(step_stats["line_search_accepted"]))
    window["line_search_flow_delta"].append(float(step_stats["line_search_flow_delta"]))
    window["line_search_repr_delta"].append(float(step_stats["line_search_repr_delta"]))


def _summarize_stat_window(window: dict[str, list[float]]) -> dict[str, float]:
    if not window["conflict"]:
        return _empty_step_stats()
    return {
        "conflict_fraction": _mean(window["conflict"]),
        "dot_before_mean": _mean(window["dot_before"]),
        "dot_after_mean": _mean(window["dot_after"]),
        "actual_fm_delta_after_repr_step": _mean(window["actual_fm_delta_after_repr_step"]),
        "projected_repr_norm_ratio": _mean(window["projected_repr_norm_ratio"]),
        "adaptive_margin": _mean(window["adaptive_margin"]),
        "adaptive_normalized_fm_loss": _mean(window["adaptive_normalized_fm_loss"]),
        "adaptive_margin_baseline": _mean(window["adaptive_margin_baseline"]),
        "adaptive_margin_direction": _mean(window["adaptive_margin_direction"]),
        "trust_radius_used": _mean(window["trust_radius_used"]),
        "trust_radius_next": _mean(window["trust_radius_next"]),
        "trust_scale": _mean(window["trust_scale"]),
        "trust_region_active": _mean(window["trust_region_active"]),
        "dual_value_used": _mean(window["dual_value_used"]),
        "dual_value_next": _mean(window["dual_value_next"]),
        "fm_budget_violation": _mean(window["fm_budget_violation"]),
        "fm_delta_target": _mean(window["fm_delta_target"]),
        "scaled_dot_after_mean": _mean(window["scaled_dot_after"]),
        "repr_step_fm_first_order_effect": _mean(window["repr_step_fm_first_order_effect"]),
        "line_search_attempts": _mean(window["line_search_attempts"]),
        "line_search_alpha": _mean(window["line_search_alpha"]),
        "line_search_accepted": _mean(window["line_search_accepted"]),
        "line_search_flow_delta": _mean(window["line_search_flow_delta"]),
        "line_search_repr_delta": _mean(window["line_search_repr_delta"]),
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty value list")
    return float(sum(values) / len(values))


def _stats_from_projection(
    projection: ProjectionResult,
    actual_fm_delta: float,
    adaptive_margin: float = 0.0,
    adaptive_normalized_fm_loss: float = 0.0,
    adaptive_margin_baseline: float = 0.0,
    adaptive_margin_direction: float = 0.0,
    trust_result: TrustRegionScaleResult | None = None,
    dual_update: DualBudgetControlResult | None = None,
    fm_delta_target: float = 0.0,
    scaled_dot_after: torch.Tensor | None = None,
    repr_learning_rate: float = 1.0,
) -> dict[str, float]:
    if trust_result is None:
        ratio = projection.projected_repr_norm / projection.repr_norm
    else:
        ratio = projection.repr_norm.new_tensor(trust_result.scaled_norm) / projection.repr_norm
    if scaled_dot_after is None:
        scaled_dot_after = projection.dot_after
    repr_step_fm_first_order_effect = -float(repr_learning_rate) * scaled_dot_after
    return {
        "conflict_fraction": 1.0 if projection.dot_before < 0 else 0.0,
        "dot_before_mean": float(projection.dot_before.detach().cpu()),
        "dot_after_mean": float(projection.dot_after.detach().cpu()),
        "actual_fm_delta_after_repr_step": float(actual_fm_delta),
        "projected_repr_norm_ratio": float(ratio.detach().cpu()),
        "adaptive_margin": float(adaptive_margin),
        "adaptive_normalized_fm_loss": float(adaptive_normalized_fm_loss),
        "adaptive_margin_baseline": float(adaptive_margin_baseline),
        "adaptive_margin_direction": float(adaptive_margin_direction),
        "trust_radius_used": 0.0 if trust_result is None else float(trust_result.trust_radius),
        "trust_radius_next": 0.0 if dual_update is None else float(dual_update.next_trust_radius),
        "trust_scale": 0.0 if trust_result is None else float(trust_result.trust_scale),
        "trust_region_active": 0.0
        if trust_result is None
        else float(1.0 if trust_result.trust_region_active else 0.0),
        "dual_value_used": 0.0 if dual_update is None else float(dual_update.previous_dual_value),
        "dual_value_next": 0.0 if dual_update is None else float(dual_update.next_dual_value),
        "fm_budget_violation": 0.0 if dual_update is None else float(dual_update.fm_budget_violation),
        "fm_delta_target": float(fm_delta_target),
        "scaled_dot_after_mean": float(scaled_dot_after.detach().cpu()),
        "repr_step_fm_first_order_effect": float(repr_step_fm_first_order_effect.detach().cpu()),
        "line_search_attempts": 0.0,
        "line_search_alpha": 0.0,
        "line_search_accepted": 0.0,
        "line_search_flow_delta": 0.0,
        "line_search_repr_delta": 0.0,
    }


def _stats_from_gradients(
    g_repr: list[torch.Tensor],
    g_fm: list[torch.Tensor],
    dot_before: torch.Tensor,
    dot_after: torch.Tensor,
    actual_fm_delta: float,
) -> dict[str, float]:
    repr_norm = torch.sqrt(_squared_norm(g_repr))
    projected_norm = repr_norm
    return {
        "conflict_fraction": 1.0 if dot_before < 0 else 0.0,
        "dot_before_mean": float(dot_before.detach().cpu()),
        "dot_after_mean": float(dot_after.detach().cpu()),
        "actual_fm_delta_after_repr_step": float(actual_fm_delta),
        "projected_repr_norm_ratio": float((projected_norm / repr_norm).detach().cpu()),
        "adaptive_margin": 0.0,
        "adaptive_normalized_fm_loss": 0.0,
        "adaptive_margin_baseline": 0.0,
        "adaptive_margin_direction": 0.0,
        "trust_radius_used": 0.0,
        "trust_radius_next": 0.0,
        "trust_scale": 0.0,
        "trust_region_active": 0.0,
        "dual_value_used": 0.0,
        "dual_value_next": 0.0,
        "fm_budget_violation": 0.0,
        "fm_delta_target": 0.0,
        "scaled_dot_after_mean": float(dot_after.detach().cpu()),
        "repr_step_fm_first_order_effect": float((-dot_after).detach().cpu()),
        "line_search_attempts": 0.0,
        "line_search_alpha": 0.0,
        "line_search_accepted": 0.0,
        "line_search_flow_delta": 0.0,
        "line_search_repr_delta": 0.0,
    }


def _initial_step_stats(method: str, soft_margin: float) -> dict[str, float]:
    stats = _empty_step_stats()
    if method in {"adaptive_margin_projected", "adaptive_trust_projected"}:
        stats["adaptive_margin"] = float(soft_margin)
    return stats


def _new_adaptive_margin_state(config: ToyConfig) -> AdaptiveMarginState:
    if config.adaptive_margin_initial is None:
        raise ValueError("adaptive_margin_projected requires adaptive_margin_initial")
    return AdaptiveMarginState(
        margin=float(config.adaptive_margin_initial),
        ema_fm_loss=None,
    )


def _new_adaptive_trust_state(config: ToyConfig) -> AdaptiveTrustState:
    if config.trust_radius_initial is None:
        raise ValueError("adaptive_trust_projected requires trust_radius_initial")
    return AdaptiveTrustState(
        margin_state=_new_adaptive_margin_state(config),
        dual_value=0.0,
        trust_radius=float(config.trust_radius_initial),
    )


def _update_adaptive_margin(
    config: ToyConfig,
    adaptive_state: AdaptiveMarginState,
    normalized_fm_loss: float,
) -> AdaptiveMarginAdjustment:
    if config.adaptive_margin_mode is None:
        raise ValueError("adaptive_margin_projected requires adaptive_margin_mode")
    baseline = _adaptive_margin_baseline(config, adaptive_state, normalized_fm_loss)
    if config.adaptive_margin_step is None:
        raise ValueError("adaptive_margin_projected requires adaptive_margin_step")
    if config.adaptive_margin_min is None or config.adaptive_margin_max is None:
        raise ValueError("adaptive_margin_projected requires adaptive_margin_min and adaptive_margin_max")
    adjustment = compute_adaptive_margin_adjustment(
        current_margin=adaptive_state.margin,
        normalized_fm_loss=normalized_fm_loss,
        baseline=baseline,
        step=config.adaptive_margin_step,
        min_margin=config.adaptive_margin_min,
        max_margin=config.adaptive_margin_max,
    )
    adaptive_state.margin = adjustment.next_margin
    if config.adaptive_margin_mode == "ema":
        adaptive_state.ema_fm_loss = _ema_update(
            adaptive_state.ema_fm_loss,
            normalized_fm_loss,
            config.adaptive_margin_ema_beta,
        )
    return adjustment


def _adaptive_margin_baseline(
    config: ToyConfig,
    adaptive_state: AdaptiveMarginState,
    normalized_fm_loss: float,
) -> float:
    if config.adaptive_margin_mode == "target":
        if config.adaptive_margin_target is None:
            raise ValueError("adaptive_margin_projected requires adaptive_margin_target in target mode")
        return float(config.adaptive_margin_target)
    if config.adaptive_margin_mode == "ema":
        if adaptive_state.ema_fm_loss is None:
            return float(normalized_fm_loss)
        return float(adaptive_state.ema_fm_loss)
    raise ValueError("adaptive_margin_mode must be one of {'target', 'ema'}")


def _adaptive_margin_direction_value(direction: str) -> float:
    if direction == "tighten":
        return -1.0
    if direction == "hold":
        return 0.0
    if direction == "loosen":
        return 1.0
    raise ValueError(f"Unsupported adaptive margin direction: {direction}")


def _ema_update(previous: float | None, value: float, beta: float | None) -> float:
    if beta is None:
        raise ValueError("adaptive_margin_projected requires adaptive_margin_ema_beta in ema mode")
    if previous is None:
        return float(value)
    return float(beta) * float(previous) + (1.0 - float(beta)) * float(value)


def _validate_adaptive_margin_config(config: ToyConfig) -> None:
    adaptive_methods = [method for method in ("adaptive_margin_projected", "adaptive_trust_projected") if method in config.methods]
    if not adaptive_methods:
        return
    if len(adaptive_methods) != 1:
        raise ValueError("adaptive margin methods require standalone configs")
    method = adaptive_methods[0]
    if config.methods != [method]:
        if method == "adaptive_trust_projected":
            raise ValueError("adaptive_trust_projected requires a standalone config")
        raise ValueError("adaptive_margin_projected must run in a dedicated config")
    if config.lambdas != [1.0]:
        if method == "adaptive_trust_projected":
            raise ValueError("adaptive_trust_projected requires lambdas to equal [1.0]")
        raise ValueError("adaptive_margin_projected uses fixed lambda_repr=1.0; set lambdas to [1.0]")
    if len(config.soft_margins) != 1:
        raise ValueError(f"{method} requires exactly one soft_margin placeholder")

    missing = [
        name
        for name, value in {
            "adaptive_margin_mode": config.adaptive_margin_mode,
            "adaptive_margin_step": config.adaptive_margin_step,
            "adaptive_margin_min": config.adaptive_margin_min,
            "adaptive_margin_max": config.adaptive_margin_max,
            "adaptive_margin_initial": config.adaptive_margin_initial,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"{method} requires explicit config fields: {missing}")

    if config.adaptive_margin_mode not in {"target", "ema"}:
        raise ValueError("adaptive_margin_mode must be 'target' or 'ema'")
    if config.adaptive_margin_mode == "target" and config.adaptive_margin_target is None:
        raise ValueError(f"{method} requires adaptive_margin_target in target mode")
    if config.adaptive_margin_mode == "ema" and config.adaptive_margin_ema_beta is None:
        raise ValueError(f"{method} requires adaptive_margin_ema_beta in ema mode")

    adaptive_margin_step = _require_finite_scalar(config.adaptive_margin_step, "adaptive_margin_step", min_value=0.0)
    adaptive_margin_min = _require_finite_scalar(config.adaptive_margin_min, "adaptive_margin_min", min_value=0.0)
    adaptive_margin_max = _require_finite_scalar(
        config.adaptive_margin_max,
        "adaptive_margin_max",
        min_value=adaptive_margin_min,
    )
    adaptive_margin_initial = _require_finite_scalar(
        config.adaptive_margin_initial,
        "adaptive_margin_initial",
        min_value=adaptive_margin_min,
    )
    if adaptive_margin_initial > adaptive_margin_max:
        raise ValueError("adaptive_margin_initial must be <= adaptive_margin_max")
    if not math.isclose(float(config.soft_margins[0]), adaptive_margin_initial, rel_tol=0.0, abs_tol=1e-12):
        if method == "adaptive_trust_projected":
            raise ValueError("adaptive_trust_projected requires soft_margins[0] to equal adaptive_margin_initial")
        raise ValueError("adaptive_margin_projected soft_margins[0] must equal adaptive_margin_initial")
    if config.adaptive_margin_mode == "target":
        _require_finite_scalar(config.adaptive_margin_target, "adaptive_margin_target", min_value=0.0)
    if config.adaptive_margin_mode == "ema":
        adaptive_margin_ema_beta = _require_finite_scalar(
            config.adaptive_margin_ema_beta,
            "adaptive_margin_ema_beta",
            min_value=0.0,
        )
        if adaptive_margin_ema_beta >= 1.0:
            raise ValueError("adaptive_margin_ema_beta must be < 1.0")
    if adaptive_margin_step > adaptive_margin_max - adaptive_margin_min and adaptive_margin_max > adaptive_margin_min:
        raise ValueError("adaptive_margin_step must not exceed the adaptive margin range")


def _validate_adaptive_trust_config(config: ToyConfig) -> None:
    if "adaptive_trust_projected" not in config.methods:
        return
    missing = [
        name
        for name, value in {
            "dual_lr": config.dual_lr,
            "trust_radius_initial": config.trust_radius_initial,
            "trust_radius_min": config.trust_radius_min,
            "trust_radius_max": config.trust_radius_max,
            "fm_delta_target": config.fm_delta_target,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"adaptive_trust_projected requires explicit config fields: {missing}")

    dual_lr = _require_finite_scalar(config.dual_lr, "dual_lr", min_value=0.0)
    if dual_lr <= 0.0:
        raise ValueError("dual_lr must be positive")
    _require_finite_scalar(config.fm_delta_target, "fm_delta_target", min_value=0.0)
    trust_radius_min = _require_finite_scalar(config.trust_radius_min, "trust_radius_min", min_value=0.0)
    trust_radius_max = _require_finite_scalar(config.trust_radius_max, "trust_radius_max", min_value=trust_radius_min)
    trust_radius_initial = _require_finite_scalar(
        config.trust_radius_initial,
        "trust_radius_initial",
        min_value=trust_radius_min,
    )
    if trust_radius_initial > trust_radius_max:
        raise ValueError("trust_radius_initial must lie within [trust_radius_min, trust_radius_max]")


def _validate_line_search_config(config: ToyConfig) -> None:
    if "line_search_projected" not in config.methods:
        return
    if config.methods != ["line_search_projected"]:
        raise ValueError("line_search_projected requires a standalone config")
    if config.lambdas != [1.0]:
        raise ValueError("line_search_projected requires lambdas to equal [1.0]")
    missing = [
        name
        for name, value in {
            "fm_delta_target": config.fm_delta_target,
            "line_search_max_backtracks": config.line_search_max_backtracks,
            "line_search_contraction": config.line_search_contraction,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"line_search_projected requires explicit config fields: {missing}")
    if config.soft_margins != [0.0]:
        raise ValueError("line_search_projected requires soft_margins to equal [0.0]")

    _require_finite_scalar(config.fm_delta_target, "fm_delta_target", min_value=0.0)
    _require_positive_int(config.line_search_max_backtracks, "line_search_max_backtracks")
    _require_open_unit_scalar(config.line_search_contraction, "line_search_contraction")


def _require_finite_scalar(value: float | None, name: str, min_value: float | None = None) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    scalar = float(value)
    if min_value is not None and scalar < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return scalar


def _require_positive_int(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_open_unit_scalar(value: float | None, name: str) -> float:
    scalar = _require_finite_scalar(value, name)
    if scalar <= 0.0 or scalar >= 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return scalar


def _validate_gradient_lists(g_repr: list[torch.Tensor], g_fm: list[torch.Tensor]) -> None:
    if not isinstance(g_repr, list) or not isinstance(g_fm, list):
        raise TypeError("g_repr and g_fm must be list[Tensor]")
    if not g_repr or not g_fm:
        raise ValueError("g_repr and g_fm must be non-empty")
    if len(g_repr) != len(g_fm):
        raise ValueError("g_repr and g_fm must have the same length")
    for index, (repr_grad, fm_grad) in enumerate(zip(g_repr, g_fm)):
        if repr_grad.shape != fm_grad.shape:
            raise ValueError(f"g_repr[{index}] and g_fm[{index}] must have the same shape")
        if repr_grad.device != fm_grad.device:
            raise ValueError(f"g_repr[{index}] and g_fm[{index}] must be on the same device")
        if not torch.isfinite(repr_grad).all() or not torch.isfinite(fm_grad).all():
            raise FloatingPointError("gradient tensors must be finite")


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


def _stable_seed_offset(method: str, delta_deg: float, lambda_repr: float, soft_margin: float) -> int:
    text = f"{method}|{delta_deg:.6f}|{lambda_repr:.6f}|{soft_margin:.6f}"
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 1_000_000
    return value


def _plot_curves(metrics_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Cannot plot curves without metrics rows")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    curve_keys = [
        ("repr_cosine_mean", "representation cosine"),
        ("valid_fm_loss", "validation FM loss"),
        ("generation_mse_to_x1", "generation MSE"),
        ("conflict_fraction", "conflict fraction"),
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"d{row['delta_deg']}_{row['method']}_l{row['lambda_repr']}_m{row['soft_margin']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    for axis, (metric_key, title) in zip(axes.flatten(), curve_keys):
        for name, group_rows in groups.items():
            group_rows = sorted(group_rows, key=lambda item: item["step"])
            axis.plot([item["step"] for item in group_rows], [item[metric_key] for item in group_rows], label=name, linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("step")
    axes[0, 0].legend(fontsize=5, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_trajectory(final_points: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not final_points:
        raise ValueError("Cannot plot trajectory without final experiment points")
    fig, axis = plt.subplots(figsize=(10, 5))
    labels = [f"{item['delta_deg']}:{item['method']}:{item['lambda_repr']}:{item['soft_margin']}" for item in final_points]
    cosine = [float(item["repr_cosine_mean"]) for item in final_points]
    fm_loss = [float(item["valid_fm_loss"]) for item in final_points]
    scatter = axis.scatter(fm_loss, cosine, c=[float(item["delta_deg"]) for item in final_points], cmap="viridis")
    for label, x_value, y_value in zip(labels, fm_loss, cosine):
        axis.annotate(label, (x_value, y_value), fontsize=5, alpha=0.7)
    axis.set_xlabel("validation FM loss")
    axis.set_ylabel("representation cosine")
    axis.set_title("Toy FM/CL trade-off")
    fig.colorbar(scatter, ax=axis, label="delta_deg")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SAFA toy FM+CL projected-update diagnostic.")
    parser.add_argument("--config", required=True, help="Path to an explicit toy experiment JSON config.")
    args = parser.parse_args()
    config = load_config(args.config)
    summary = run_experiment_grid(config)
    print(json.dumps({"run_name": summary["run_name"], "num_experiments": summary["num_experiments"]}, sort_keys=True))


if __name__ == "__main__":
    main()
