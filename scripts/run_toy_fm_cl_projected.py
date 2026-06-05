#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import MISSING, asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from safa.training.projected_update import ProjectionResult, project_gradient_onto_fm_feasible_cone


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
    fm_delta_target: float
    dual_lr: float
    dual_max: float
    primal_dual_warmup_steps: int
    cagrad_c: float | None = None
    uncertainty_log_var_lr: float | None = None
    uncertainty_log_var_init_fm: float | None = None
    uncertainty_log_var_init_cl: float | None = None


@dataclass(frozen=True)
class CAGradResult:
    fm_weight: float
    cl_weight: float
    gradient_cosine: float
    combined_norm: float
    combined_gradients: list[torch.Tensor]


SUPPORTED_METHODS = {
    "fm_only",
    "repr_only",
    "weighted_sum",
    "projected_two_step",
    "primal_dual_projected",
    "pcgrad",
    "soft_margin_projected",
    "cagrad",
    "uncertainty_weighted",
}
NORM_EPS = 1.0e-12


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
    if not isinstance(config.primal_dual_warmup_steps, int) or config.primal_dual_warmup_steps < 0:
        raise ValueError("primal_dual_warmup_steps must be a non-negative integer")
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
        "fm_delta_target": config.fm_delta_target,
        "dual_lr": config.dual_lr,
        "dual_max": config.dual_max,
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
    if config.dual_lr <= 0.0:
        raise ValueError("dual_lr must be positive")
    if config.dual_max <= 0.0:
        raise ValueError("dual_max must be positive")
    for value in config.lambdas:
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("all lambdas must be positive finite numbers")
    for value in config.soft_margins:
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("all soft_margins must be finite non-negative numbers")
    if "primal_dual_projected" in config.methods:
        if config.methods != ["primal_dual_projected"]:
            raise ValueError("primal_dual_projected must be standalone in methods")
        if len(config.lambdas) != 1 or float(config.lambdas[0]) != 1.0:
            raise ValueError("primal_dual_projected lambdas must be [1.0]")
    if "cagrad" in config.methods:
        if config.methods != ["cagrad"]:
            raise ValueError("cagrad must be standalone in methods")
        if len(config.lambdas) != 1 or float(config.lambdas[0]) != 1.0:
            raise ValueError("cagrad lambdas must be [1.0]")
        cagrad_c = _require_optional_scalar(config.cagrad_c, "cagrad requires cagrad_c", min_value=0.0)
        if cagrad_c >= 1.0:
            raise ValueError("cagrad_c must be < 1.0")
    if "uncertainty_weighted" in config.methods:
        if config.methods != ["uncertainty_weighted"]:
            raise ValueError("uncertainty_weighted must be standalone in methods")
        if len(config.lambdas) != 1 or float(config.lambdas[0]) != 1.0:
            raise ValueError("uncertainty_weighted lambdas must be [1.0]")
        _require_optional_scalar(
            config.uncertainty_log_var_lr,
            "uncertainty_weighted requires uncertainty_log_var_lr",
            min_value=0.0,
            strictly_positive=True,
        )
        _require_optional_scalar(
            config.uncertainty_log_var_init_fm,
            "uncertainty_weighted requires uncertainty_log_var_init_fm",
        )
        _require_optional_scalar(
            config.uncertainty_log_var_init_cl,
            "uncertainty_weighted requires uncertainty_log_var_init_cl",
        )


def _require_optional_scalar(
    value: float | None,
    message: str,
    *,
    min_value: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if value is None or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise ValueError(message)
    scalar = float(value)
    if strictly_positive and scalar <= 0.0:
        raise ValueError(message)
    if min_value is not None and scalar < min_value:
        raise ValueError(message)
    return scalar


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
    device = torch.device(config.device)
    seed_offset = _stable_seed_offset(method, delta_deg, lambda_repr, soft_margin)
    train_generator = torch.Generator(device=device).manual_seed(config.seed + seed_offset)
    eval_generator = torch.Generator(device=device).manual_seed(config.seed + 100_000 + seed_offset)
    model = ToyVectorField(hidden_dim=config.hidden_dim, layers=config.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    fm_optimizer = torch.optim.AdamW(model.parameters(), lr=config.fm_learning_rate, weight_decay=config.weight_decay)
    eval_batch = _sample_batch(config, delta_deg, config.eval_batch_size, device, eval_generator)
    flow_scale, repr_scale = _calibrate_loss_scales(config, model, delta_deg, device, train_generator)
    initial = _evaluate_model(config, model, eval_batch, delta_deg, method, lambda_repr, soft_margin, step=0)
    uncertainty_optimizer: torch.optim.Optimizer | None = None
    uncertainty_log_vars: tuple[torch.nn.Parameter, torch.nn.Parameter] | None = None
    if method == "uncertainty_weighted":
        fm_log_var = torch.nn.Parameter(
            torch.tensor(float(config.uncertainty_log_var_init_fm), dtype=torch.float32, device=device)
        )
        cl_log_var = torch.nn.Parameter(
            torch.tensor(float(config.uncertainty_log_var_init_cl), dtype=torch.float32, device=device)
        )
        uncertainty_log_vars = (fm_log_var, cl_log_var)
        uncertainty_optimizer = torch.optim.AdamW(
            list(uncertainty_log_vars),
            lr=float(config.uncertainty_log_var_lr),
            weight_decay=0.0,
        )

    metrics = [initial]
    stat_window = _new_stat_window()
    dual_value = 0.0
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
        elif method == "primal_dual_projected":
            step_stats = _step_primal_dual_projected(
                config,
                model,
                fm_optimizer,
                batch,
                flow_scale,
                repr_scale,
                lambda_repr,
                dual_value=dual_value,
                step=step,
            )
            dual_value = float(step_stats["dual_value_next"])
        elif method == "pcgrad":
            step_stats = _step_pcgrad(config, model, optimizer, batch, flow_scale, repr_scale, lambda_repr)
        elif method == "cagrad":
            step_stats = _step_cagrad(config, model, optimizer, batch, flow_scale, repr_scale)
        elif method == "uncertainty_weighted":
            if uncertainty_optimizer is None or uncertainty_log_vars is None:
                raise RuntimeError("uncertainty_weighted state was not initialized")
            step_stats = _step_uncertainty_weighted(
                config,
                model,
                optimizer,
                uncertainty_optimizer,
                uncertainty_log_vars,
                batch,
                flow_scale,
                repr_scale,
            )
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
    if method == "primal_dual_projected":
        return [(1.0, config.soft_margins[0])]
    if method == "soft_margin_projected":
        return [(lambda_repr, margin) for lambda_repr in config.lambdas for margin in config.soft_margins]
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
    if soft_margin == 0.0:
        projection = project_gradient_onto_fm_feasible_cone(g_repr, g_fm, eps=config.projection_eps)
    else:
        projection = project_gradient_with_soft_margin(
            g_repr,
            g_fm,
            epsilon=soft_margin,
            eps=config.projection_eps,
        )
    _manual_parameter_step(model, projection.projected_gradients, config.repr_learning_rate)
    after_losses = _compute_losses(config, model, batch)
    actual_fm_delta = float((after_losses["flow"] / flow_scale).detach().cpu()) - flow_guard_value
    return _stats_from_projection(projection, actual_fm_delta=actual_fm_delta)


def _step_primal_dual_projected(
    config: ToyConfig,
    model: ToyVectorField,
    fm_optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    flow_scale: torch.Tensor,
    repr_scale: torch.Tensor,
    lambda_repr: float,
    dual_value: float,
    step: int,
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
    projection = project_gradient_onto_fm_feasible_cone(g_repr, g_fm, eps=config.projection_eps)

    # This is dual step-control projected training, not a full augmented Lagrangian.
    dual_value_used = float(dual_value)
    effective_repr_lr = float(config.repr_learning_rate) / (1.0 + dual_value_used)
    _manual_parameter_step(model, projection.projected_gradients, effective_repr_lr)

    after_losses = _compute_losses(config, model, batch)
    normalized_flow_after = float((after_losses["flow"] / flow_scale).detach().cpu())
    actual_fm_delta = normalized_flow_after - flow_guard_value
    violation = actual_fm_delta - float(config.fm_delta_target)
    warmup_active = bool(config.primal_dual_warmup_steps > 0 and step <= config.primal_dual_warmup_steps)
    if warmup_active:
        dual_value_next = dual_value_used
    else:
        dual_value_next = min(max(dual_value_used + float(config.dual_lr) * violation, 0.0), float(config.dual_max))

    stats = _stats_from_projection(projection, actual_fm_delta=actual_fm_delta)
    stats.update(
        {
            "dual_value_used": dual_value_used,
            "dual_value_next": dual_value_next,
            "effective_repr_lr": effective_repr_lr,
            "primal_dual_violation": float(violation),
            "primal_dual_warmup_active": 1.0 if warmup_active else 0.0,
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


def _step_cagrad(
    config: ToyConfig,
    model: ToyVectorField,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    flow_scale: torch.Tensor,
    repr_scale: torch.Tensor,
) -> dict[str, float]:
    losses = _compute_losses(config, model, batch)
    flow_objective = losses["flow"] / flow_scale
    repr_objective = losses["repr"] / repr_scale
    flow_before = float(flow_objective.detach().cpu())
    g_fm, g_repr = _task_gradients(model, flow_objective, repr_objective)
    result = aggregate_two_task_cagrad(g_fm, g_repr, c=float(config.cagrad_c), eps=config.projection_eps)
    _assign_gradients(model, result.combined_gradients)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after_losses = _compute_losses(config, model, batch)
    actual_fm_delta = float((after_losses["flow"] / flow_scale).detach().cpu()) - flow_before
    stats = _stats_from_gradients(
        g_repr,
        g_fm,
        _dot(g_repr, g_fm),
        _dot(result.combined_gradients, result.combined_gradients),
        actual_fm_delta=actual_fm_delta,
        combined_gradients=result.combined_gradients,
    )
    stats.update(
        {
            "cagrad_fm_weight": result.fm_weight,
            "cagrad_cl_weight": result.cl_weight,
            "gradient_cosine_mean": result.gradient_cosine,
            "combined_grad_norm": result.combined_norm,
        }
    )
    return stats


def _step_uncertainty_weighted(
    config: ToyConfig,
    model: ToyVectorField,
    optimizer: torch.optim.Optimizer,
    uncertainty_optimizer: torch.optim.Optimizer,
    uncertainty_log_vars: tuple[torch.nn.Parameter, torch.nn.Parameter],
    batch: dict[str, torch.Tensor],
    flow_scale: torch.Tensor,
    repr_scale: torch.Tensor,
) -> dict[str, float]:
    fm_log_var, cl_log_var = uncertainty_log_vars
    losses = _compute_losses(config, model, batch)
    flow_objective = losses["flow"] / flow_scale
    repr_objective = losses["repr"] / repr_scale
    flow_before = float(flow_objective.detach().cpu())
    fm_weight = 0.5 * torch.exp(-fm_log_var)
    cl_weight = 0.5 * torch.exp(-cl_log_var)
    total = fm_weight * flow_objective + 0.5 * fm_log_var + cl_weight * repr_objective + 0.5 * cl_log_var
    g_fm, g_repr = _task_gradients(model, flow_objective, repr_objective)
    combined = [fm_weight.detach() * fm_grad + cl_weight.detach() * repr_grad for fm_grad, repr_grad in zip(g_fm, g_repr)]
    optimizer.zero_grad(set_to_none=True)
    uncertainty_optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    uncertainty_optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    uncertainty_optimizer.zero_grad(set_to_none=True)
    after_losses = _compute_losses(config, model, batch)
    actual_fm_delta = float((after_losses["flow"] / flow_scale).detach().cpu()) - flow_before
    stats = _stats_from_gradients(
        g_repr,
        g_fm,
        _dot(g_repr, g_fm),
        _dot(combined, combined),
        actual_fm_delta=actual_fm_delta,
        combined_gradients=combined,
    )
    stats.update(
        {
            "uncertainty_fm_log_var": float(fm_log_var.detach().cpu()),
            "uncertainty_cl_log_var": float(cl_log_var.detach().cpu()),
            "uncertainty_fm_weight": float(fm_weight.detach().cpu()),
            "uncertainty_cl_weight": float(cl_weight.detach().cpu()),
            "uncertainty_formula_total": float(total.detach().cpu()),
        }
    )
    return stats


def aggregate_two_task_cagrad(
    g_fm: list[torch.Tensor],
    g_cl: list[torch.Tensor],
    c: float,
    eps: float,
) -> CAGradResult:
    _validate_gradient_lists(g_fm, g_cl)
    if not isinstance(c, (float, int)) or not math.isfinite(float(c)) or float(c) < 0.0 or float(c) >= 1.0:
        raise ValueError("c must be finite and in [0, 1)")
    if not isinstance(eps, (float, int)) or not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    c_value = float(c)
    g0 = [0.5 * (fm_grad + cl_grad) for fm_grad, cl_grad in zip(g_fm, g_cl)]
    g0_norm = torch.sqrt(_squared_norm(g0))
    if bool((g0_norm <= eps).item()):
        raise FloatingPointError("CAGrad received a near-zero mean task gradient")

    def weighted_gradient(alpha: float) -> list[torch.Tensor]:
        return [float(alpha) * fm_grad + (1.0 - float(alpha)) * cl_grad for fm_grad, cl_grad in zip(g_fm, g_cl)]

    def objective_derivative(alpha: float) -> torch.Tensor:
        weighted = weighted_gradient(alpha)
        direction = [fm_grad - cl_grad for fm_grad, cl_grad in zip(g_fm, g_cl)]
        weighted_norm = torch.sqrt(_squared_norm(weighted))
        if bool((weighted_norm <= eps).item()):
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
    if bool((weighted_norm <= eps).item()):
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
        "dual_value_used": [],
        "dual_value_next": [],
        "effective_repr_lr": [],
        "primal_dual_violation": [],
        "primal_dual_warmup_active": [],
        "gradient_cosine": [],
        "combined_grad_norm": [],
        "cagrad_fm_weight": [],
        "cagrad_cl_weight": [],
        "uncertainty_fm_log_var": [],
        "uncertainty_cl_log_var": [],
        "uncertainty_fm_weight": [],
        "uncertainty_cl_weight": [],
        "uncertainty_formula_total": [],
    }


def _empty_step_stats() -> dict[str, float]:
    return {
        "conflict_fraction": 0.0,
        "dot_before_mean": 0.0,
        "dot_after_mean": 0.0,
        "actual_fm_delta_after_repr_step": 0.0,
        "projected_repr_norm_ratio": 0.0,
        "dual_value_used": 0.0,
        "dual_value_next": 0.0,
        "effective_repr_lr": 0.0,
        "primal_dual_violation": 0.0,
        "primal_dual_warmup_active": 0.0,
        "gradient_cosine_mean": 0.0,
        "combined_grad_norm": 0.0,
        "cagrad_fm_weight": 0.0,
        "cagrad_cl_weight": 0.0,
        "uncertainty_fm_log_var": 0.0,
        "uncertainty_cl_log_var": 0.0,
        "uncertainty_fm_weight": 0.0,
        "uncertainty_cl_weight": 0.0,
        "uncertainty_formula_total": 0.0,
    }


def _accumulate_stats(window: dict[str, list[float]], step_stats: dict[str, float]) -> None:
    window["conflict"].append(float(step_stats["conflict_fraction"]))
    window["dot_before"].append(float(step_stats["dot_before_mean"]))
    window["dot_after"].append(float(step_stats["dot_after_mean"]))
    window["actual_fm_delta_after_repr_step"].append(float(step_stats["actual_fm_delta_after_repr_step"]))
    window["projected_repr_norm_ratio"].append(float(step_stats["projected_repr_norm_ratio"]))
    window["dual_value_used"].append(float(step_stats["dual_value_used"]))
    window["dual_value_next"].append(float(step_stats["dual_value_next"]))
    window["effective_repr_lr"].append(float(step_stats["effective_repr_lr"]))
    window["primal_dual_violation"].append(float(step_stats["primal_dual_violation"]))
    window["primal_dual_warmup_active"].append(float(step_stats["primal_dual_warmup_active"]))
    window["gradient_cosine"].append(float(step_stats["gradient_cosine_mean"]))
    window["combined_grad_norm"].append(float(step_stats["combined_grad_norm"]))
    window["cagrad_fm_weight"].append(float(step_stats["cagrad_fm_weight"]))
    window["cagrad_cl_weight"].append(float(step_stats["cagrad_cl_weight"]))
    window["uncertainty_fm_log_var"].append(float(step_stats["uncertainty_fm_log_var"]))
    window["uncertainty_cl_log_var"].append(float(step_stats["uncertainty_cl_log_var"]))
    window["uncertainty_fm_weight"].append(float(step_stats["uncertainty_fm_weight"]))
    window["uncertainty_cl_weight"].append(float(step_stats["uncertainty_cl_weight"]))
    window["uncertainty_formula_total"].append(float(step_stats["uncertainty_formula_total"]))


def _summarize_stat_window(window: dict[str, list[float]]) -> dict[str, float]:
    if not window["conflict"]:
        return _empty_step_stats()
    return {
        "conflict_fraction": _mean(window["conflict"]),
        "dot_before_mean": _mean(window["dot_before"]),
        "dot_after_mean": _mean(window["dot_after"]),
        "actual_fm_delta_after_repr_step": _mean(window["actual_fm_delta_after_repr_step"]),
        "projected_repr_norm_ratio": _mean(window["projected_repr_norm_ratio"]),
        "dual_value_used": window["dual_value_used"][-1],
        "dual_value_next": window["dual_value_next"][-1],
        "effective_repr_lr": _mean(window["effective_repr_lr"]),
        "primal_dual_violation": _mean(window["primal_dual_violation"]),
        "primal_dual_warmup_active": _mean(window["primal_dual_warmup_active"]),
        "gradient_cosine_mean": _mean(window["gradient_cosine"]),
        "combined_grad_norm": _mean(window["combined_grad_norm"]),
        "cagrad_fm_weight": window["cagrad_fm_weight"][-1],
        "cagrad_cl_weight": window["cagrad_cl_weight"][-1],
        "uncertainty_fm_log_var": window["uncertainty_fm_log_var"][-1],
        "uncertainty_cl_log_var": window["uncertainty_cl_log_var"][-1],
        "uncertainty_fm_weight": window["uncertainty_fm_weight"][-1],
        "uncertainty_cl_weight": window["uncertainty_cl_weight"][-1],
        "uncertainty_formula_total": _mean(window["uncertainty_formula_total"]),
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty value list")
    return float(sum(values) / len(values))


def _stats_from_projection(projection: ProjectionResult, actual_fm_delta: float) -> dict[str, float]:
    ratio = projection.projected_repr_norm / projection.repr_norm
    return {
        "conflict_fraction": 1.0 if projection.dot_before < 0 else 0.0,
        "dot_before_mean": float(projection.dot_before.detach().cpu()),
        "dot_after_mean": float(projection.dot_after.detach().cpu()),
        "actual_fm_delta_after_repr_step": float(actual_fm_delta),
        "projected_repr_norm_ratio": float(ratio.detach().cpu()),
        "dual_value_used": 0.0,
        "dual_value_next": 0.0,
        "effective_repr_lr": 0.0,
        "primal_dual_violation": 0.0,
        "primal_dual_warmup_active": 0.0,
        "gradient_cosine_mean": 0.0,
        "combined_grad_norm": float(projection.projected_repr_norm.detach().cpu()),
        "cagrad_fm_weight": 0.0,
        "cagrad_cl_weight": 0.0,
        "uncertainty_fm_log_var": 0.0,
        "uncertainty_cl_log_var": 0.0,
        "uncertainty_fm_weight": 0.0,
        "uncertainty_cl_weight": 0.0,
        "uncertainty_formula_total": 0.0,
    }


def _stats_from_gradients(
    g_repr: list[torch.Tensor],
    g_fm: list[torch.Tensor],
    dot_before: torch.Tensor,
    dot_after: torch.Tensor,
    actual_fm_delta: float,
    combined_gradients: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    repr_norm = torch.sqrt(_squared_norm(g_repr))
    projected_norm = repr_norm
    combined = combined_gradients if combined_gradients is not None else [fm_grad + repr_grad for fm_grad, repr_grad in zip(g_fm, g_repr)]
    combined_norm = torch.sqrt(_squared_norm(combined))
    return {
        "conflict_fraction": 1.0 if dot_before < 0 else 0.0,
        "dot_before_mean": float(dot_before.detach().cpu()),
        "dot_after_mean": float(dot_after.detach().cpu()),
        "actual_fm_delta_after_repr_step": float(actual_fm_delta),
        "projected_repr_norm_ratio": float((projected_norm / repr_norm).detach().cpu()),
        "dual_value_used": 0.0,
        "dual_value_next": 0.0,
        "effective_repr_lr": 0.0,
        "primal_dual_violation": 0.0,
        "primal_dual_warmup_active": 0.0,
        "gradient_cosine_mean": float(_gradient_cosine(g_fm, g_repr, NORM_EPS).detach().cpu()),
        "combined_grad_norm": float(combined_norm.detach().cpu()),
        "cagrad_fm_weight": 0.0,
        "cagrad_cl_weight": 0.0,
        "uncertainty_fm_log_var": 0.0,
        "uncertainty_cl_log_var": 0.0,
        "uncertainty_fm_weight": 0.0,
        "uncertainty_cl_weight": 0.0,
        "uncertainty_formula_total": 0.0,
    }


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


def _gradient_cosine(left: list[torch.Tensor], right: list[torch.Tensor], eps: float) -> torch.Tensor:
    _validate_gradient_lists(left, right)
    left_norm = torch.sqrt(_squared_norm(left))
    right_norm = torch.sqrt(_squared_norm(right))
    if bool((left_norm <= eps).item() or (right_norm <= eps).item()):
        raise FloatingPointError("Cannot compute cosine for a near-zero gradient")
    return _dot(left, right) / (left_norm * right_norm)


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
