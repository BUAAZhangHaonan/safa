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
    diagnostics: dict[str, Any]


_LOCKED_GUIDANCE_INTERVALS = (
    ("I1", 1.0, 0.75),
    ("I2", 0.75, 0.5),
    ("I3", 0.5, 0.25),
)


@dataclass(frozen=True)
class _IntervalState:
    interval_id: str
    active: bool
    t: float
    s: float
    before: torch.Tensor
    transported: torch.Tensor
    corrected: torch.Tensor
    velocity: torch.Tensor
    gradient: torch.Tensor | None


class CountedFlowMap:
    """Count vector-field evaluations made through a generator flow map."""

    def __init__(self, generator, *, kind: str = "flow_map"):
        self.generator = generator
        self.nfe = 0
        self.kind = str(kind)
        self.trace: list[dict[str, float | str | list[float]]] = []

    def __call__(self, x, z, *, t, r):
        self.nfe += 1
        self.trace.append({"t": _trace_time(t), "r": _trace_time(r), "kind": self.kind})
        return self.generator.flow_map(x, z, t=t, r=r)


def freeze_guidance_stack(generator, codec, e0) -> None:
    _freeze_module(generator)
    _freeze_module(codec.vae)
    _freeze_module(e0)


def assert_guidance_stack_frozen(generator, codec, e0) -> None:
    for name, module in (
        ("generator", generator),
        ("codec.vae", codec.vae),
        ("e0", e0),
    ):
        if module.training:
            raise RuntimeError(f"guidance requires {name} in evaluation mode")
        for parameter in module.parameters():
            if parameter.requires_grad:
                raise RuntimeError(f"guidance requires frozen {name} parameters")
            if parameter.grad is not None:
                raise RuntimeError(
                    f"guidance found an unexpected {name} parameter gradient"
                )


def symmetric_relative_l2(left, right, eps: float = 1.0e-8) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(
            f"relative L2 inputs must have the same shape, got {tuple(left.shape)} and {tuple(right.shape)}"
        )
    if left.ndim < 2:
        raise ValueError(
            f"relative L2 inputs must have a batch dimension, got {tuple(left.shape)}"
        )
    left_norm = left.flatten(1).norm(dim=1)
    right_norm = right.flatten(1).norm(dim=1)
    difference = (left - right).flatten(1).norm(dim=1)
    return 2.0 * difference / (left_norm + right_norm + eps)


def semigroup_probe(flow_map, x_init, condition, split_times) -> dict[str, Any]:
    splits = [float(value) for value in split_times]
    if not splits or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in splits
    ):
        raise ValueError("split_times must be strictly increasing and within (0,1)")
    if any(left >= right for left, right in zip(splits, splits[1:])):
        raise ValueError("split_times must be strictly increasing and within (0,1)")

    initial_nfe = flow_map.nfe
    split_endpoints: dict[float, torch.Tensor] = {}
    residuals: dict[float, torch.Tensor] = {}
    with torch.no_grad():
        direct = _finite_tensor(
            "semigroup direct endpoint", flow_map(x_init, condition, t=1.0, r=0.0)
        ).detach()
        for split in splits:
            intermediate = _finite_tensor(
                "semigroup intermediate state",
                flow_map(x_init, condition, t=1.0, r=split),
            ).detach()
            endpoint = _finite_tensor(
                "semigroup split endpoint",
                flow_map(intermediate, condition, t=split, r=0.0),
            ).detach()
            residual = _finite_tensor(
                "semigroup residual", symmetric_relative_l2(direct, endpoint)
            ).detach()
            split_endpoints[split] = endpoint
            residuals[split] = residual
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
    active_guidance_intervals: Sequence[str] | None = None,
    collect_interval_diagnostics: bool = False,
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
    if optimization_mode == "official_adam" and len(guided) != 4:
        raise ValueError(
            "official_adam requires exactly four guided times for the locked three-interval schedule"
        )
    if (
        isinstance(num_optim_iters, bool)
        or int(num_optim_iters) != num_optim_iters
        or num_optim_iters <= 0
    ):
        raise ValueError(
            f"num_optim_iters must be a positive integer, got {num_optim_iters!r}"
        )
    if optimization_mode == "paper_normalized_direct_autograd" and num_optim_iters != 1:
        raise ValueError(
            "paper_normalized_direct_autograd requires num_optim_iters == 1"
        )
    active_interval_ids = _resolve_active_guidance_intervals(
        guided, active_guidance_intervals
    )
    r9_mode = active_guidance_intervals is not None or collect_interval_diagnostics
    supports_r9_intervals = (
        sample_mode == "flow_map2"
        and optimization_mode == "paper_normalized_direct_autograd"
    )
    if r9_mode and not supports_r9_intervals:
        raise ValueError(
            "R9 interval masking and diagnostics are supported only for flow_map2 with "
            "paper_normalized_direct_autograd"
        )
    if not isinstance(collect_interval_diagnostics, bool):
        raise ValueError("collect_interval_diagnostics must be a boolean")
    if collect_interval_diagnostics:
        _require_locked_guidance_schedule(guided)

    initial_nfe = flow_map.nfe
    state = _finite_tensor("x_init", x_init.detach())
    loss_history: list[float] = []
    learning_rates: list[float] = []
    guided_interval_count = len(guided) - 1
    active_interval_set = set(active_interval_ids)
    interval_states: list[_IntervalState] = []

    for interval_index, (t, s) in enumerate(zip(guided, guided[1:])):
        interval_id = f"I{interval_index + 1}"
        interval_active = (
            active_guidance_intervals is None or interval_id in active_interval_set
        )
        before = state.detach()
        if not interval_active:
            with torch.no_grad():
                transported = _finite_tensor(
                    "transport-only state",
                    flow_map(before, transport_condition, t=t, r=s),
                ).detach()
            step_velocity = _finite_tensor(
                "transport-only velocity",
                (before - transported) / (t - s),
            )
            state = transported
            if collect_interval_diagnostics:
                interval_states.append(
                    _IntervalState(
                        interval_id=interval_id,
                        active=False,
                        t=t,
                        s=s,
                        before=before,
                        transported=transported,
                        corrected=state,
                        velocity=step_velocity.detach(),
                        gradient=None,
                    )
                )
            continue
        if optimization_mode == "official_adam":
            lr = step_size * (1.0 - interval_index / guided_interval_count)
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
                raise RuntimeError(
                    "official Adam completed without an endpoint evaluation"
                )
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
            loss = _representation_loss(endpoint, codec, e0, target_z0)
            loss_history.append(float(loss.detach()))
            gradient = torch.autograd.grad(loss, current, only_inputs=True)[0]
            gradient = _finite_tensor("representation gradient", gradient)
            if sample_mode == "flow_map1":
                step_velocity = endpoint_velocity
            else:
                with torch.no_grad():
                    step_source = current.detach()
                    step_endpoint = flow_map(step_source, transport_condition, t=t, r=s)
                    step_velocity = (step_source - step_endpoint) / (t - s)
            delta_xt = step_size * normalize_per_sample_to_velocity_norm(
                gradient, step_velocity
            )

        state = before - (t - s) * (step_velocity.detach() + delta_xt)
        state = _finite_tensor("guided state", state.detach())
        if collect_interval_diagnostics:
            interval_states.append(
                _IntervalState(
                    interval_id=interval_id,
                    active=True,
                    t=t,
                    s=s,
                    before=before,
                    transported=_finite_tensor(
                        "transported state", step_endpoint.detach()
                    ),
                    corrected=state,
                    velocity=_finite_tensor("step velocity", step_velocity.detach()),
                    gradient=gradient.detach(),
                )
            )

    state = _run_unguided_tail(flow_map, state, transport_condition, unguided)
    algorithm_nfe = flow_map.nfe - initial_nfe
    diagnostics: dict[str, Any] = {
        "guided_times": guided,
        "unguided_times": unguided,
        "sample_mode": sample_mode,
        "optimization_mode": optimization_mode,
        "num_optim_iters": num_optim_iters,
        "step_size": step_size,
        "uses_adam": optimization_mode == "official_adam",
        "adam_learning_rates": learning_rates,
        "loss_history": loss_history,
    }
    if r9_mode:
        diagnostics.update(
            _r9_interval_diagnostics(
                generator=flow_map.generator,
                codec=codec,
                e0=e0,
                condition=transport_condition,
                target_z0=target_z0,
                active_interval_ids=active_interval_ids,
                algorithm_nfe=algorithm_nfe,
                collect=collect_interval_diagnostics,
                interval_states=interval_states,
            )
        )
    assert_guidance_stack_frozen(flow_map.generator, codec, e0)
    return GuidanceResult(
        latent=state,
        nfe=algorithm_nfe,
        diagnostics=diagnostics,
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
    active_guidance_intervals: Sequence[str] | None = None,
    collect_interval_diagnostics: bool = False,
) -> GuidanceResult:
    guided, unguided = _validate_fmrg_arguments(
        flow_map=flow_map,
        codec=codec,
        e0=e0,
        guided_times=guided_times,
        unguided_times=unguided_times,
        step_size=step_size,
    )
    active_interval_ids = _resolve_active_guidance_intervals(
        guided, active_guidance_intervals
    )
    if not isinstance(collect_interval_diagnostics, bool):
        raise ValueError("collect_interval_diagnostics must be a boolean")
    if collect_interval_diagnostics:
        _require_locked_guidance_schedule(guided)
    r9_mode = active_guidance_intervals is not None or collect_interval_diagnostics
    initial_nfe = flow_map.nfe
    state = _finite_tensor("x_init", x_init.detach())
    loss_history: list[float] = []
    active_interval_set = set(active_interval_ids)
    interval_states: list[_IntervalState] = []

    for interval_index, (t, s) in enumerate(zip(guided, guided[1:])):
        interval_id = f"I{interval_index + 1}"
        interval_active = (
            active_guidance_intervals is None or interval_id in active_interval_set
        )
        before = state.detach()
        transported = _finite_tensor(
            "transported state",
            flow_map(before, transport_condition, t=t, r=s),
        ).detach()
        step_velocity = _finite_tensor(
            "step velocity", (before - transported) / (t - s)
        )
        gradient = None
        if interval_active:
            split_state = transported.requires_grad_(True)
            endpoint = flow_map(split_state, transport_condition, t=s, r=0.0)
            loss = _representation_loss(endpoint, codec, e0, target_z0)
            loss_history.append(float(loss.detach()))
            gradient = torch.autograd.grad(loss, split_state, only_inputs=True)[0]
            gradient = _finite_tensor("representation gradient", gradient)
            correction = normalize_per_sample_to_velocity_norm(gradient, step_velocity)
            state = transported - (t - s) * step_size * correction
            state = _finite_tensor("guided state", state.detach())
        else:
            state = transported
        if collect_interval_diagnostics:
            interval_states.append(
                _IntervalState(
                    interval_id=interval_id,
                    active=interval_active,
                    t=t,
                    s=s,
                    before=before,
                    transported=transported,
                    corrected=state,
                    velocity=step_velocity.detach(),
                    gradient=None if gradient is None else gradient.detach(),
                )
            )

    state = _run_unguided_tail(flow_map, state, transport_condition, unguided)
    algorithm_nfe = flow_map.nfe - initial_nfe
    diagnostics: dict[str, Any] = {
        "guided_times": guided,
        "unguided_times": unguided,
        "step_size": step_size,
        "loss_history": loss_history,
    }
    if r9_mode:
        diagnostics.update(
            _r9_interval_diagnostics(
                generator=flow_map.generator,
                codec=codec,
                e0=e0,
                condition=transport_condition,
                target_z0=target_z0,
                active_interval_ids=active_interval_ids,
                algorithm_nfe=algorithm_nfe,
                collect=collect_interval_diagnostics,
                interval_states=interval_states,
            )
        )
    assert_guidance_stack_frozen(flow_map.generator, codec, e0)
    return GuidanceResult(
        latent=state,
        nfe=algorithm_nfe,
        diagnostics=diagnostics,
    )


def _r9_interval_diagnostics(
    *,
    generator,
    codec,
    e0,
    condition: torch.Tensor,
    target_z0: torch.Tensor,
    active_interval_ids: Sequence[str],
    algorithm_nfe: int,
    collect: bool,
    interval_states: Sequence[_IntervalState],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "active_guidance_intervals": list(active_interval_ids),
        "interval_diagnostics_enabled": collect,
        "interval_diagnostics": {},
        "algorithm_nfe": algorithm_nfe,
        "diagnostic_nfe": 0,
        "diagnostic_flow_map_trace": [],
    }
    if not collect:
        return diagnostics

    diagnostic_flow_map = CountedFlowMap(generator, kind="interval_diagnostic")
    interval_diagnostics: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for interval in interval_states:
            direct_endpoint = _finite_tensor(
                f"{interval.interval_id} diagnostic direct endpoint",
                diagnostic_flow_map(interval.before, condition, t=interval.t, r=0.0),
            ).detach()
            split_endpoint = _finite_tensor(
                f"{interval.interval_id} diagnostic split endpoint",
                diagnostic_flow_map(
                    interval.transported, condition, t=interval.s, r=0.0
                ),
            ).detach()
            if interval.active:
                corrected_endpoint = _finite_tensor(
                    f"{interval.interval_id} diagnostic corrected endpoint",
                    diagnostic_flow_map(
                        interval.corrected, condition, t=interval.s, r=0.0
                    ),
                ).detach()
            else:
                corrected_endpoint = split_endpoint

            loss_before = _representation_loss_per_sample(
                split_endpoint, codec, e0, target_z0
            )
            if interval.active:
                loss_after = _representation_loss_per_sample(
                    corrected_endpoint, codec, e0, target_z0
                )
            else:
                loss_after = loss_before
            velocity_norm = _per_sample_norm(interval.velocity)
            transport_norm = _per_sample_norm(interval.before - interval.transported)
            correction_norm = _per_sample_norm(
                interval.corrected - interval.transported
            )
            if interval.gradient is None:
                gradient_norm = torch.zeros_like(velocity_norm)
                gradient_velocity_cosine = torch.zeros_like(velocity_norm)
            else:
                gradient_norm = _per_sample_norm(interval.gradient)
                gradient_velocity_cosine = F.cosine_similarity(
                    interval.gradient.flatten(1),
                    interval.velocity.flatten(1),
                    dim=1,
                    eps=1.0e-8,
                )
            local_residual = symmetric_relative_l2(direct_endpoint, split_endpoint)
            ratio = correction_norm / transport_norm.clamp_min(1.0e-8)
            metrics = {
                "interval_id": interval.interval_id,
                "active": interval.active,
                "t": interval.t,
                "s": interval.s,
                "loss_before_correction": loss_before,
                "loss_after_correction": loss_after,
                "gradient_norm": gradient_norm,
                "velocity_norm": velocity_norm,
                "transport_norm": transport_norm,
                "correction_norm": correction_norm,
                "correction_transport_ratio": ratio,
                "gradient_velocity_cosine": gradient_velocity_cosine,
                "local_semigroup_residual": local_residual,
            }
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    _finite_tensor(f"{interval.interval_id} diagnostic {name}", value)
            interval_diagnostics[interval.interval_id] = metrics

    diagnostics["interval_diagnostics"] = interval_diagnostics
    diagnostics["diagnostic_nfe"] = diagnostic_flow_map.nfe
    diagnostics["diagnostic_flow_map_trace"] = list(diagnostic_flow_map.trace)
    return diagnostics


def _per_sample_norm(value: torch.Tensor) -> torch.Tensor:
    return _finite_tensor("per-sample norm", value.flatten(1).norm(dim=1))


def _resolve_active_guidance_intervals(
    guided_times: Sequence[float],
    active_guidance_intervals: Sequence[str] | None,
) -> tuple[str, ...]:
    default_ids = tuple(f"I{index + 1}" for index in range(len(guided_times) - 1))
    if active_guidance_intervals is None:
        return default_ids
    if isinstance(active_guidance_intervals, (str, bytes)):
        raise ValueError("active_guidance_intervals must be a sequence of interval IDs")
    requested = tuple(active_guidance_intervals)
    if any(not isinstance(interval_id, str) for interval_id in requested):
        raise ValueError("active_guidance_intervals must contain only interval IDs")
    if len(set(requested)) != len(requested):
        raise ValueError(
            "active_guidance_intervals must not contain duplicate interval IDs"
        )
    if not requested:
        return ()
    _require_locked_guidance_schedule(guided_times)
    known_ids = {interval_id for interval_id, _, _ in _LOCKED_GUIDANCE_INTERVALS}
    unknown_ids = sorted(set(requested) - known_ids)
    if unknown_ids:
        raise ValueError(f"unknown active guidance interval IDs: {unknown_ids}")
    return tuple(
        interval_id
        for interval_id, _, _ in _LOCKED_GUIDANCE_INTERVALS
        if interval_id in requested
    )


def _require_locked_guidance_schedule(guided_times: Sequence[float]) -> None:
    locked_times = [1.0, 0.75, 0.5, 0.25]
    if len(guided_times) != len(locked_times) or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        for actual, expected in zip(guided_times, locked_times)
    ):
        raise ValueError(
            "R9 interval masking and diagnostics require the locked schedule [1, 0.75, 0.5, 0.25]"
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
        raise ValueError(
            f"typical-shell projection requires 0 < delta < 1, got {delta!r}"
        )
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


def radial_shell_rfft_energy(noise: torch.Tensor) -> dict[str, Any]:
    """Summarize per-channel radial-shell energy without retaining noise.

    The orthonormal real FFT is evaluated in float64 on CPU. Half-spectrum
    coefficients use their Hermitian multiplicity, so the shell sum obeys
    Parseval and enabling this diagnostic cannot perturb CUDA sampling math.
    """

    _validate_noise_tensor("spectral snapshot noise", noise)
    if noise.ndim != 4:
        raise ValueError(
            "spectral snapshot noise must have shape [batch, channels, height, width]"
        )
    height, width = (int(noise.shape[-2]), int(noise.shape[-1]))
    if height < 2 or width < 2:
        raise ValueError("spectral snapshot spatial dimensions must both be at least 2")
    host = noise.detach().to(device="cpu", dtype=torch.float64)
    spectrum = torch.fft.rfft2(host, norm="ortho")
    power = spectrum.real.square() + spectrum.imag.square()

    x_index = (
        torch.fft.rfftfreq(width, d=1.0, dtype=torch.float64) * width
    ).round().to(torch.int64)
    y_index = (
        torch.fft.fftfreq(height, d=1.0, dtype=torch.float64) * height
    ).round().to(torch.int64)
    radius_squared_grid = y_index[:, None].square() + x_index[None, :].square()
    radius_squared = torch.unique(radius_squared_grid, sorted=True)

    multiplicity = torch.full((width // 2 + 1,), 2.0, dtype=torch.float64)
    multiplicity[0] = 1.0
    if width % 2 == 0:
        multiplicity[-1] = 1.0
    weighted_power = power * multiplicity.view(1, 1, 1, -1)

    shell_energies = []
    coefficient_counts = []
    for value in radius_squared:
        mask = radius_squared_grid == value
        shell_energies.append(
            (weighted_power * mask.view(1, 1, height, -1)).sum(dim=(-2, -1))
        )
        coefficient_counts.append(
            int((mask.to(torch.float64) * multiplicity.view(1, -1)).sum().item())
        )
    per_channel_shell_energy = torch.stack(shell_energies, dim=-1)
    per_channel_spatial_energy = host.square().sum(dim=(-2, -1))
    per_channel_spectral_energy = per_channel_shell_energy.sum(dim=-1)
    if not torch.allclose(
        per_channel_spatial_energy,
        per_channel_spectral_energy,
        rtol=1.0e-10,
        atol=1.0e-8,
    ):
        raise RuntimeError("orthonormal rFFT radial shells violate Parseval identity")

    high_frequency_min_radius = float(min(height, width)) / 4.0
    high_mask = radius_squared.to(torch.float64) >= high_frequency_min_radius**2
    per_channel_high_frequency_energy = per_channel_shell_energy[..., high_mask].sum(
        dim=-1
    )
    return {
        "norm": "ortho",
        "height": height,
        "width": width,
        "radius_squared": [int(value) for value in radius_squared.tolist()],
        "full_spectrum_coefficient_count": coefficient_counts,
        "high_frequency_min_radius": high_frequency_min_radius,
        "per_channel_shell_energy": per_channel_shell_energy,
        "per_channel_spatial_energy": per_channel_spatial_energy,
        "per_channel_high_frequency_energy": per_channel_high_frequency_energy,
    }


def _spectral_snapshot_steps(
    requested: Sequence[int] | None, *, num_updates: int
) -> tuple[int, ...]:
    if requested is None:
        return ()
    if isinstance(requested, (str, bytes)):
        raise ValueError("spectral_snapshot_steps must be an integer sequence")
    values = tuple(requested)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("spectral_snapshot_steps must contain only integers")
    if values != tuple(sorted(set(values))):
        raise ValueError("spectral_snapshot_steps must be strictly increasing and unique")
    if any(value < 0 or value > num_updates for value in values):
        raise ValueError(
            f"spectral_snapshot_steps must be within [0,{num_updates}]"
        )
    return values


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
    spectral_snapshot_steps: Sequence[int] | None = None,
) -> GuidanceResult:
    assert_guidance_stack_frozen(flow_map.generator, codec, e0)
    _validate_noise_tensor("x_init", x_init)
    if (
        isinstance(num_updates, bool)
        or int(num_updates) != num_updates
        or num_updates < 0
    ):
        raise ValueError(
            f"num_updates must be a non-negative integer, got {num_updates!r}"
        )
    eta_value = float(eta)
    if eta_value not in {0.25, 0.5, 1.0, 2.0}:
        raise ValueError(f"eta must be one of {{0.25, 0.5, 1.0, 2.0}}, got {eta!r}")
    if projection not in {"fixed_radius", "typical_shell"}:
        raise ValueError(f"unsupported noise projection {projection!r}")
    if projection == "typical_shell":
        delta_value = float(typical_delta)
        if not math.isfinite(delta_value) or not 0.0 < delta_value < 1.0:
            raise ValueError(
                f"typical-shell projection requires 0 < delta < 1, got {typical_delta!r}"
            )

    snapshot_steps = _spectral_snapshot_steps(
        spectral_snapshot_steps, num_updates=int(num_updates)
    )
    initial_nfe = flow_map.nfe
    initial = x_init.detach().clone()
    noise = initial.clone()
    loss_history: list[float] = []
    spectral_snapshots: list[dict[str, Any]] = []
    if 0 in snapshot_steps:
        spectral_snapshots.append({"step": 0, **radial_shell_rfft_energy(noise)})

    for update_index in range(num_updates):
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
        completed_updates = update_index + 1
        if completed_updates in snapshot_steps:
            spectral_snapshots.append(
                {"step": completed_updates, **radial_shell_rfft_energy(noise)}
            )

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
            "initial_final_cosine": F.cosine_similarity(
                initial.flatten(1), final_noise.flatten(1), dim=1
            ),
            "update_norm": (final_noise - initial).flatten(1).norm(dim=1),
            "channel_mean": final_noise.mean(dim=(2, 3)),
            "channel_std": final_noise.std(dim=(2, 3), unbiased=False),
            "loss_history": loss_history,
            **(
                {"spectral_snapshots": spectral_snapshots}
                if snapshot_steps
                else {}
            ),
        },
    )


def select_t_cut(candidate_reports, registered_thresholds) -> float | None:
    validated: list[tuple[float, Any]] = []
    seen: set[float] = set()
    for report in candidate_reports:
        try:
            t_cut = float(report["t_cut"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "every candidate report must contain a numeric t_cut"
            ) from exc
        if not math.isfinite(t_cut) or not 0.0 < t_cut < 1.0:
            raise ValueError(
                f"candidate t_cut must be finite and within (0,1), got {t_cut!r}"
            )
        if t_cut in seen:
            raise ValueError(f"duplicate t_cut candidate {t_cut}")
        seen.add(t_cut)
        validated.append((t_cut, report))
    for t_cut, report in sorted(validated, key=lambda item: item[0]):
        if all(
            _threshold_passes(report.get(field), requirement)
            for field, requirement in registered_thresholds.items()
        ):
            return t_cut
    return None


def _validate_fmrg_arguments(
    *, flow_map, codec, e0, guided_times, unguided_times, step_size
):
    assert_guidance_stack_frozen(flow_map.generator, codec, e0)
    if not math.isfinite(float(step_size)) or step_size <= 0.0:
        raise ValueError(f"step_size must be positive and finite, got {step_size!r}")
    guided = _validate_decreasing_schedule("guided_times", guided_times)
    unguided = _validate_decreasing_schedule("unguided_times", unguided_times)
    if not math.isclose(guided[0], 1.0) or not math.isclose(unguided[-1], 0.0):
        raise ValueError(
            "guided_times must start at 1 and unguided_times must end at 0"
        )
    if not math.isclose(guided[-1], unguided[0]):
        raise ValueError("guided_times and unguided_times must share the same t_cut")
    return guided, unguided


def _validate_decreasing_schedule(name, times):
    values = [float(value) for value in times]
    if len(values) < 2 or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
    ):
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


def _representation_loss_per_sample(endpoint, codec, e0, target_z0) -> torch.Tensor:
    _finite_tensor("diagnostic endpoint lookahead", endpoint)
    image = codec.decode(endpoint)
    prediction = e0(normalize_for_e0(image))["embedding"]
    if prediction.shape != target_z0.shape:
        raise ValueError(
            f"E0 embedding and target_z0 must have the same shape, got {tuple(prediction.shape)} and {tuple(target_z0.shape)}"
        )
    loss = 1.0 - F.cosine_similarity(prediction, target_z0, dim=1)
    return _finite_tensor("per-sample representation loss", loss).detach()


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
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 4
        or not torch.is_floating_point(tensor)
    ):
        shape = (
            tuple(tensor.shape)
            if isinstance(tensor, torch.Tensor)
            else type(tensor).__name__
        )
        raise ValueError(
            f"{name} must be a floating tensor with shape [B,C,H,W], got {shape}"
        )
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
            raise ValueError(
                f"unsupported threshold keys: {sorted(set(requirement) - {'min', 'max'})}"
            )
        return True
    return value == requirement


def _trace_time(value) -> float | list[float]:
    tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
    values = [float(item) for item in tensor.tolist()]
    return values[0] if len(values) == 1 else values
