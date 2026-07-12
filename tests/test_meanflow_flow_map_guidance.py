from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from safa.guidance.meanflow_flow_map import (  # noqa: E402
    CountedFlowMap,
    assert_guidance_stack_frozen,
    freeze_guidance_stack,
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
