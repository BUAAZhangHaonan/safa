from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from safa.guidance.meanflow_flow_map import (  # noqa: E402
    CountedFlowMap,
    assert_guidance_stack_frozen,
    freeze_guidance_stack,
    sample_official_head_current_xt,
    sample_paper_algorithm_split,
    optimize_initial_noise,
    project_fixed_radius,
    project_gaussian_typical_shell,
    select_t_cut,
    semigroup_probe,
    symmetric_relative_l2,
)
from safa.training.latent_codec import LatentCodec, LatentCodecConfig  # noqa: E402


class _ExponentialFlowGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def flow_map(self, x, z, *, t, r):
        del z
        t_tensor = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        r_tensor = torch.as_tensor(r, device=x.device, dtype=x.dtype)
        horizon = t_tensor - r_tensor
        if horizon.ndim == 1:
            horizon = horizon.view(-1, 1, 1, 1)
        return torch.exp(-horizon) * x * self.scale


class _IdentityCodec:
    def __init__(self) -> None:
        self.vae = nn.Linear(1, 1, bias=False)

    def decode(self, latent):
        return latent


class _IdentityEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, image):
        return {"embedding": image.flatten(1) * self.scale}


class _DecodeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def decode(self, latent):
        return SimpleNamespace(sample=latent * self.scale)


class _CodecWhoseParametersMustNotBeRead:
    def __init__(self) -> None:
        self.vae = nn.Linear(1, 1, bias=False)

    def parameters(self):
        raise AssertionError("inspect codec.vae.parameters(), not codec.parameters()")


class _TracedAffineFlowGenerator(nn.Module):
    def __init__(self, velocity: float = 0.5) -> None:
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(velocity))
        self.calls: list[tuple[float, float, torch.Tensor]] = []

    def flow_map(self, x, z, *, t, r):
        del z
        t_value = float(torch.as_tensor(t).flatten()[0])
        r_value = float(torch.as_tensor(r).flatten()[0])
        self.calls.append((t_value, r_value, x.detach().clone()))
        return x - (t_value - r_value) * self.velocity


class _NormalizedIdentityCodec:
    def __init__(self) -> None:
        self.vae = nn.Linear(1, 1, bias=False)

    def decode(self, latent):
        mean = latent.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = latent.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return latent * std + mean


class _NanGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        del ctx
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        del ctx
        return torch.full_like(gradient, float("nan"))


class _NanBackwardEncoder(nn.Module):
    def forward(self, image):
        return {"embedding": _NanGradient.apply(image.flatten(1))}


GUIDED_TIMES = [1.0, 0.75, 0.5, 0.25]
UNGUIDED_TIMES = [0.25, 0.125, 0.0]


def _guidance_inputs(batch_size: int = 2):
    values = torch.linspace(-1.0, 1.0, batch_size * 3 * 2 * 2)
    x_init = values.reshape(batch_size, 3, 2, 2)
    condition = torch.zeros(batch_size, 4)
    target = torch.flip(x_init.flatten(1), dims=(1,))
    return x_init, condition, target


def _frozen_guidance_stack():
    generator = _TracedAffineFlowGenerator()
    codec = _NormalizedIdentityCodec()
    e0 = _IdentityEncoder()
    freeze_guidance_stack(generator, codec, e0)
    return generator, codec, e0


def _run_official(
    *,
    sample_mode="flow_map1",
    optimization_mode="paper_normalized_direct_autograd",
    num_optim_iters=1,
    step_size=0.5,
    guided_times=GUIDED_TIMES,
    unguided_times=UNGUIDED_TIMES,
    e0=None,
):
    generator, codec, default_e0 = _frozen_guidance_stack()
    if e0 is not None:
        default_e0 = e0.eval().requires_grad_(False)
    x_init, condition, target = _guidance_inputs()
    counted = CountedFlowMap(generator)
    result = sample_official_head_current_xt(
        flow_map=counted,
        codec=codec,
        e0=default_e0,
        x_init=x_init,
        transport_condition=condition,
        target_z0=target,
        guided_times=guided_times,
        unguided_times=unguided_times,
        sample_mode=sample_mode,
        optimization_mode=optimization_mode,
        num_optim_iters=num_optim_iters,
        step_size=step_size,
    )
    return result, generator, codec, default_e0, x_init


def _run_paper(
    *,
    step_size=0.5,
    guided_times=GUIDED_TIMES,
    unguided_times=UNGUIDED_TIMES,
):
    generator, codec, e0 = _frozen_guidance_stack()
    x_init, condition, target = _guidance_inputs()
    counted = CountedFlowMap(generator)
    result = sample_paper_algorithm_split(
        flow_map=counted,
        codec=codec,
        e0=e0,
        x_init=x_init,
        transport_condition=condition,
        target_z0=target,
        guided_times=guided_times,
        unguided_times=unguided_times,
        step_size=step_size,
    )
    return result, generator


def test_freeze_guidance_stack_disables_parameter_gradients() -> None:
    generator = _ExponentialFlowGenerator()
    codec = _IdentityCodec()
    e0 = _IdentityEncoder()

    freeze_guidance_stack(generator, codec, e0)

    assert not generator.training
    assert not codec.vae.training
    assert not e0.training
    assert all(not parameter.requires_grad for parameter in generator.parameters())
    assert all(not parameter.requires_grad for parameter in codec.vae.parameters())
    assert all(not parameter.requires_grad for parameter in e0.parameters())
    assert_guidance_stack_frozen(generator, codec, e0)


def test_counted_flow_map_counts_one_nfe_per_call() -> None:
    counted = CountedFlowMap(_ExponentialFlowGenerator())
    x = torch.ones(4, 1, 2, 2)
    z = torch.zeros(4, 2)

    counted(x, z, t=1.0, r=0.5)
    counted(x, z, t=0.5, r=0.0)

    assert counted.nfe == 2


def test_semigroup_probe_returns_zero_for_exact_semigroup() -> None:
    counted = CountedFlowMap(_ExponentialFlowGenerator().eval().requires_grad_(False))
    report = semigroup_probe(counted, torch.randn(3, 1, 2, 2), torch.zeros(3, 2), [0.25, 0.5, 0.75])

    for residual in report["residuals"].values():
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-6)
    assert report["nfe"] == 7


def test_semigroup_probe_reports_each_requested_split() -> None:
    counted = CountedFlowMap(_ExponentialFlowGenerator().eval().requires_grad_(False))
    report = semigroup_probe(counted, torch.randn(2, 1, 2, 2), torch.zeros(2, 2), [0.2, 0.6])

    assert list(report["split_endpoints"]) == [0.2, 0.6]
    assert list(report["residuals"]) == [0.2, 0.6]
    assert tuple(report["direct_endpoint"].shape) == (2, 1, 2, 2)
    assert report["nfe"] == 5


@pytest.mark.parametrize("split_times", [[0.5, 0.25], [0.0, 0.5], [0.5, 1.0], [0.5, 0.5]])
def test_semigroup_probe_rejects_unsorted_or_boundary_split(split_times) -> None:
    counted = CountedFlowMap(_ExponentialFlowGenerator())
    with pytest.raises(ValueError, match=r"strictly increasing.*within \(0,1\)"):
        semigroup_probe(counted, torch.ones(1, 1, 1, 1), torch.zeros(1, 1), split_times)


def test_semigroup_relative_residual_is_finite_for_zero_endpoints() -> None:
    residual = symmetric_relative_l2(torch.zeros(2, 1, 2, 2), torch.zeros(2, 1, 2, 2))

    assert torch.equal(residual, torch.zeros(2))
    assert torch.isfinite(residual).all()


def test_t_cut_selection_sorts_candidates_and_chooses_smallest_full_pass() -> None:
    reports = [
        {"t_cut": 0.75, "median": 0.08, "p90": 0.16, "cosine": 0.97, "visual_pass": True},
        {"t_cut": 0.25, "median": 0.11, "p90": 0.18, "cosine": 0.98, "visual_pass": True},
        {"t_cut": 0.50, "median": 0.09, "p90": 0.19, "cosine": 0.96, "visual_pass": True},
    ]
    thresholds = {
        "median": {"max": 0.10},
        "p90": {"max": 0.20},
        "cosine": {"min": 0.95},
        "visual_pass": True,
    }

    assert select_t_cut(reports, thresholds) == 0.5


def test_t_cut_selection_returns_gate_failure_when_none_pass() -> None:
    reports = [{"t_cut": 0.25, "median": 0.2, "visual_pass": True}]
    thresholds = {"median": {"max": 0.1}, "visual_pass": True}

    assert select_t_cut(reports, thresholds) is None


def test_t_cut_selection_has_no_manual_tie_break_input() -> None:
    assert list(inspect.signature(select_t_cut).parameters) == ["candidate_reports", "registered_thresholds"]


def test_latent_codec_wrapper_freezes_vae_but_keeps_decode_input_gradient() -> None:
    codec = LatentCodec(_DecodeVAE(), LatentCodecConfig(source="fake", scaling_factor=1.0))
    generator = _ExponentialFlowGenerator()
    e0 = _IdentityEncoder()
    latent = torch.randn(2, 4, 2, 2, requires_grad=True)

    assert not isinstance(codec, nn.Module)
    freeze_guidance_stack(generator, codec, e0)
    decoded = codec.decode(latent)
    decoded.sum().backward()

    assert not codec.vae.training
    assert all(not parameter.requires_grad for parameter in codec.vae.parameters())
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert all(parameter.grad is None for parameter in codec.vae.parameters())


def test_assert_guidance_stack_checks_codec_vae_not_codec_parameters() -> None:
    generator = _ExponentialFlowGenerator().eval().requires_grad_(False)
    codec = _CodecWhoseParametersMustNotBeRead()
    codec.vae.eval().requires_grad_(False)
    e0 = _IdentityEncoder().eval().requires_grad_(False)

    assert_guidance_stack_frozen(generator, codec, e0)


def test_official_current_xt_takes_endpoint_gradient_at_xt_before_advance() -> None:
    _, generator, _, _, x_init = _run_official(sample_mode="flow_map2")

    endpoint_t, endpoint_r, endpoint_x = generator.calls[0]
    step_t, step_r, step_x = generator.calls[1]
    assert (endpoint_t, endpoint_r) == (1.0, 0.0)
    assert (step_t, step_r) == (1.0, 0.75)
    assert torch.equal(endpoint_x, x_init)
    assert torch.equal(step_x, x_init)


def test_official_current_xt_flow_map1_reuses_endpoint_velocity() -> None:
    _, generator, _, _, _ = _run_official(
        sample_mode="flow_map1",
        guided_times=[1.0, 0.5],
        unguided_times=[0.5, 0.0],
    )

    assert [(t, r) for t, r, _ in generator.calls] == [(1.0, 0.0), (0.5, 0.0)]


def test_official_current_xt_flow_map2_uses_distinct_endpoint_and_step_maps() -> None:
    _, generator, _, _, _ = _run_official(
        sample_mode="flow_map2",
        guided_times=[1.0, 0.5],
        unguided_times=[0.5, 0.0],
    )

    assert [(t, r) for t, r, _ in generator.calls] == [(1.0, 0.0), (1.0, 0.5), (0.5, 0.0)]


@pytest.mark.parametrize("optimization_mode", ["official_adam", "paper_normalized_direct_autograd"])
def test_official_current_xt_supports_adam_and_normalized_direct_modes(optimization_mode) -> None:
    result, _, _, _, _ = _run_official(optimization_mode=optimization_mode)

    assert torch.isfinite(result.latent).all()
    assert result.diagnostics["optimization_mode"] == optimization_mode


def test_safa_uniform_schedule_flow_map1_nopt1_is_five_nfe() -> None:
    result, _, _, _, _ = _run_official(sample_mode="flow_map1")

    assert result.nfe == 5


def test_safa_uniform_schedule_flow_map2_nopt1_is_eight_nfe() -> None:
    result, _, _, _, _ = _run_official(sample_mode="flow_map2")

    assert result.nfe == 8


def test_official_adam_uses_interval_decay_one_minus_i_over_four() -> None:
    result, _, _, _, _ = _run_official(optimization_mode="official_adam", step_size=4.0)

    assert result.diagnostics["adam_learning_rates"] == [4.0, 3.0, 2.0]


def test_official_adam_nopt_gt_one_refreshes_endpoint_at_updated_xt() -> None:
    _, generator, _, _, _ = _run_official(
        optimization_mode="official_adam",
        num_optim_iters=2,
        guided_times=[1.0, 0.5],
        unguided_times=[0.5, 0.0],
    )

    first = generator.calls[0]
    second = generator.calls[1]
    assert (first[0], first[1]) == (1.0, 0.0)
    assert (second[0], second[1]) == (1.0, 0.0)
    assert not torch.equal(first[2], second[2])


def test_normalized_mode_has_no_adam_state_or_lr_decay() -> None:
    result, _, _, _, _ = _run_official(optimization_mode="paper_normalized_direct_autograd")

    assert result.diagnostics["uses_adam"] is False
    assert result.diagnostics["adam_learning_rates"] == []


def test_official_current_xt_finishes_with_official_unguided_tail_order() -> None:
    _, generator, _, _, _ = _run_official(sample_mode="flow_map1")

    assert [(t, r) for t, r, _ in generator.calls[-2:]] == [(0.25, 0.125), (0.125, 0.0)]


def test_official_current_xt_leaves_generator_codec_and_e0_unchanged() -> None:
    result, generator, codec, e0, _ = _run_official(optimization_mode="official_adam")

    assert torch.equal(generator.velocity, torch.tensor(0.5))
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert all(parameter.grad is None for parameter in codec.vae.parameters())
    assert all(parameter.grad is None for parameter in e0.parameters())
    assert_guidance_stack_frozen(generator, codec, e0)
    assert torch.isfinite(result.latent).all()


def test_official_current_xt_fails_on_non_finite_gradient() -> None:
    with pytest.raises(FloatingPointError, match="non-finite representation gradient"):
        _run_official(e0=_NanBackwardEncoder())


def test_paper_split_transports_to_xs_before_endpoint_gradient() -> None:
    generator, codec, e0 = _frozen_guidance_stack()
    x_init, condition, target = _guidance_inputs()
    sample_paper_algorithm_split(
        flow_map=CountedFlowMap(generator),
        codec=codec,
        e0=e0,
        x_init=x_init,
        transport_condition=condition,
        target_z0=target,
        guided_times=[1.0, 0.5],
        unguided_times=[0.5, 0.0],
        step_size=0.5,
    )

    first_t, first_r, first_x = generator.calls[0]
    second_t, second_r, second_x = generator.calls[1]
    assert (first_t, first_r) == (1.0, 0.5)
    assert (second_t, second_r) == (0.5, 0.0)
    assert torch.equal(first_x, x_init)
    assert torch.equal(second_x, x_init - 0.25)


def test_paper_split_iterates_both_shared_unguided_tail_segments() -> None:
    result, generator = _run_paper()

    assert [(t, r) for t, r, _ in generator.calls[-2:]] == [(0.25, 0.125), (0.125, 0.0)]
    assert result.nfe == len(generator.calls)


def test_paper_split_nfe_matches_counted_calls() -> None:
    result, generator = _run_paper()

    assert result.nfe == 8
    assert result.nfe == len(generator.calls)


def test_b1_and_b2_receive_identical_guided_and_unguided_times() -> None:
    official, _, _, _, _ = _run_official()
    paper, _ = _run_paper()

    assert official.diagnostics["guided_times"] == paper.diagnostics["guided_times"] == GUIDED_TIMES
    assert official.diagnostics["unguided_times"] == paper.diagnostics["unguided_times"] == UNGUIDED_TIMES


def test_paper_split_normalizes_gradient_per_sample() -> None:
    _, generator = _run_paper(step_size=0.5)
    first_input = generator.calls[0][2]
    x_bar = first_input - 0.125
    next_interval_input = generator.calls[2][2]
    correction_norm = (x_bar - next_interval_input).flatten(1).norm(dim=1)
    velocity_norm = torch.full_like(correction_norm, 0.5 * math.sqrt(x_bar[0].numel()))

    assert torch.allclose(correction_norm, 0.25 * 0.5 * velocity_norm, atol=1.0e-6)


def test_paper_split_reduces_tiny_representation_loss() -> None:
    result, generator = _run_paper(step_size=1.0)
    x_init, _, target = _guidance_inputs()
    codec = _NormalizedIdentityCodec()
    e0 = _IdentityEncoder().eval().requires_grad_(False)
    from safa.training.losses import normalize_for_e0

    native = x_init - generator.velocity.detach()
    native_embedding = e0(normalize_for_e0(codec.decode(native)))["embedding"]
    final_embedding = e0(normalize_for_e0(codec.decode(result.latent)))["embedding"]
    native_loss = 1.0 - torch.cosine_similarity(native_embedding, target, dim=1).mean()
    final_loss = 1.0 - torch.cosine_similarity(final_embedding, target, dim=1).mean()

    assert final_loss < native_loss


@pytest.mark.parametrize("variant", ["official", "paper"])
def test_fmrg_variants_reject_non_decreasing_schedule(variant) -> None:
    with pytest.raises(ValueError, match="strictly decreasing"):
        if variant == "official":
            _run_official(guided_times=[1.0, 0.5, 0.6], unguided_times=[0.6, 0.0])
        else:
            _run_paper(guided_times=[1.0, 0.5, 0.6], unguided_times=[0.6, 0.0])


@pytest.mark.parametrize("variant", ["official", "paper"])
@pytest.mark.parametrize("step_size", [0.0, -0.1])
def test_fmrg_variants_reject_non_positive_step_size(variant, step_size) -> None:
    with pytest.raises(ValueError, match="step_size must be positive"):
        if variant == "official":
            _run_official(step_size=step_size)
        else:
            _run_paper(step_size=step_size)


def test_project_fixed_radius_restores_each_initial_norm() -> None:
    initial = torch.randn(3, 4, 2, 2)
    candidate = torch.randn_like(initial) * torch.tensor([0.5, 2.0, 4.0]).view(3, 1, 1, 1)

    projected = project_fixed_radius(candidate, initial)

    assert torch.allclose(projected.flatten(1).norm(dim=1), initial.flatten(1).norm(dim=1), atol=1.0e-6)


def test_project_fixed_radius_rejects_zero_candidate() -> None:
    with pytest.raises(ValueError, match="zero-norm candidate"):
        project_fixed_radius(torch.zeros(2, 4, 2, 2), torch.randn(2, 4, 2, 2))


def test_project_typical_shell_clamps_only_outside_radii() -> None:
    dimension = 4 * 2 * 2
    delta = 0.25
    minimum = math.sqrt(dimension * (1.0 - delta))
    maximum = math.sqrt(dimension * (1.0 + delta))
    directions = torch.randn(3, 4, 2, 2)
    directions = directions / directions.flatten(1).norm(dim=1).view(3, 1, 1, 1)
    requested_norms = torch.tensor([minimum / 2.0, math.sqrt(dimension), maximum * 2.0])
    candidate = directions * requested_norms.view(3, 1, 1, 1)

    projected = project_gaussian_typical_shell(candidate, delta=delta)
    projected_norms = projected.flatten(1).norm(dim=1)

    assert torch.allclose(projected_norms, torch.tensor([minimum, math.sqrt(dimension), maximum]), atol=1.0e-6)
    assert torch.equal(projected[1], candidate[1])


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_project_typical_shell_rejects_invalid_delta(delta) -> None:
    with pytest.raises(ValueError, match=r"0 < delta < 1"):
        project_gaussian_typical_shell(torch.randn(2, 4, 2, 2), delta=delta)


def _run_noise_oracle(*, num_updates=8, eta=0.25, projection="fixed_radius", x_init=None):
    generator, codec, e0 = _frozen_guidance_stack()
    default_x, condition, target = _guidance_inputs()
    if x_init is None:
        x_init = default_x
    counted = CountedFlowMap(generator)
    result = optimize_initial_noise(
        flow_map=counted,
        codec=codec,
        e0=e0,
        x_init=x_init,
        transport_condition=condition,
        target_z0=target,
        num_updates=num_updates,
        eta=eta,
        projection=projection,
    )
    return result, generator, codec, e0, target


def test_noise_oracle_reduces_tiny_representation_loss() -> None:
    result, generator, codec, e0, target = _run_noise_oracle()
    initial, _, _ = _guidance_inputs()
    from safa.training.losses import normalize_for_e0

    initial_endpoint = initial - generator.velocity.detach()
    initial_embedding = e0(normalize_for_e0(codec.decode(initial_endpoint)))["embedding"]
    final_embedding = e0(normalize_for_e0(codec.decode(result.latent)))["embedding"]
    initial_loss = 1.0 - torch.cosine_similarity(initial_embedding, target, dim=1).mean()
    final_loss = 1.0 - torch.cosine_similarity(final_embedding, target, dim=1).mean()

    assert final_loss < initial_loss


def test_noise_oracle_re_evaluates_projected_final_point() -> None:
    result, generator, _, _, _ = _run_noise_oracle(num_updates=2)

    assert len(generator.calls) == 3
    assert torch.equal(generator.calls[-1][2], result.diagnostics["final_noise"])
    assert len(result.diagnostics["loss_history"]) == 3


def test_noise_oracle_reports_updates_plus_one_nfe() -> None:
    result, generator, _, _, _ = _run_noise_oracle(num_updates=8)

    assert result.nfe == 9
    assert result.nfe == len(generator.calls)


def test_noise_oracle_does_not_change_frozen_weights() -> None:
    result, generator, codec, e0, _ = _run_noise_oracle()

    assert torch.equal(generator.velocity, torch.tensor(0.5))
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert all(parameter.grad is None for parameter in codec.vae.parameters())
    assert all(parameter.grad is None for parameter in e0.parameters())
    assert_guidance_stack_frozen(generator, codec, e0)
    assert torch.isfinite(result.latent).all()


def test_noise_oracle_rejects_non_finite_state() -> None:
    x_init, _, _ = _guidance_inputs()
    x_init[0, 0, 0, 0] = float("nan")

    with pytest.raises(FloatingPointError, match="non-finite x_init"):
        _run_noise_oracle(x_init=x_init)
