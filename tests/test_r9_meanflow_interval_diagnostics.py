from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from safa.guidance.meanflow_flow_map import (  # noqa: E402
    CountedFlowMap,
    freeze_guidance_stack,
    sample_official_head_current_xt,
    sample_paper_algorithm_split,
    symmetric_relative_l2,
)


GUIDED_TIMES = [1.0, 0.75, 0.5, 0.25]
UNGUIDED_TIMES = [0.25, 0.125, 0.0]
INTERVAL_IDS = ("I1", "I2", "I3")
INTERVAL_METRIC_KEYS = {
    "interval_id",
    "active",
    "t",
    "s",
    "loss_before_correction",
    "loss_after_correction",
    "gradient_norm",
    "velocity_norm",
    "transport_norm",
    "correction_norm",
    "correction_transport_ratio",
    "gradient_velocity_cosine",
    "local_semigroup_residual",
}


class _AuditedFlowGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.calls: list[tuple[float, float, bool, torch.Tensor]] = []

    def flow_map(self, x, z, *, t, r):
        del z
        t_value = float(torch.as_tensor(t).flatten()[0])
        r_value = float(torch.as_tensor(r).flatten()[0])
        self.calls.append(
            (t_value, r_value, torch.is_grad_enabled(), x.detach().clone())
        )
        horizon = t_value - r_value
        return x - horizon * self.scale * (x.square() + 0.25)


class _NormalizedIdentityCodec:
    def __init__(self) -> None:
        self.vae = nn.Linear(1, 1, bias=False)

    def decode(self, latent):
        mean = latent.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = latent.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return latent * std + mean


class _CountingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.forward_calls = 0

    def forward(self, image):
        self.forward_calls += 1
        return {"embedding": image.flatten(1) * self.scale}


def _inputs():
    x_init = torch.linspace(-0.8, 0.9, 24).reshape(2, 3, 2, 2)
    condition = torch.zeros(2, 4)
    target = torch.flip(x_init.flatten(1), dims=(1,))
    return x_init, condition, target


def _run(
    variant: str,
    *,
    active_guidance_intervals=None,
    collect_interval_diagnostics: bool = False,
    guided_times=GUIDED_TIMES,
    unguided_times=UNGUIDED_TIMES,
):
    generator = _AuditedFlowGenerator()
    codec = _NormalizedIdentityCodec()
    e0 = _CountingEncoder()
    freeze_guidance_stack(generator, codec, e0)
    x_init, condition, target = _inputs()
    counted = CountedFlowMap(generator, kind="algorithm")
    common = {
        "flow_map": counted,
        "codec": codec,
        "e0": e0,
        "x_init": x_init,
        "transport_condition": condition,
        "target_z0": target,
        "guided_times": guided_times,
        "unguided_times": unguided_times,
        "step_size": 0.25,
        "active_guidance_intervals": active_guidance_intervals,
        "collect_interval_diagnostics": collect_interval_diagnostics,
    }
    if variant == "flow_map2_normalized":
        result = sample_official_head_current_xt(
            **common,
            sample_mode="flow_map2",
            optimization_mode="paper_normalized_direct_autograd",
            num_optim_iters=1,
        )
    elif variant == "paper_split":
        result = sample_paper_algorithm_split(**common)
    else:
        raise AssertionError(f"unknown test variant {variant}")
    return SimpleNamespace(result=result, counted=counted, generator=generator, e0=e0)


@pytest.mark.parametrize("variant", ["flow_map2_normalized", "paper_split"])
@pytest.mark.parametrize(
    "active",
    [[], ["I1"], ["I2"], ["I3"], ["I1", "I3"], ["I1", "I2", "I3"]],
)
def test_interval_mask_has_exact_algorithm_nfe_and_no_inactive_e0(
    variant, active
) -> None:
    run = _run(variant, active_guidance_intervals=active)

    expected_nfe = 5 + len(active)
    assert run.result.nfe == expected_nfe
    assert run.result.diagnostics["algorithm_nfe"] == expected_nfe
    assert run.result.diagnostics["diagnostic_nfe"] == 0
    assert run.counted.nfe == expected_nfe
    assert len(run.counted.trace) == expected_nfe
    assert run.e0.forward_calls == len(active)
    assert len(run.result.diagnostics["loss_history"]) == len(active)


@pytest.mark.parametrize("variant", ["flow_map2_normalized", "paper_split"])
def test_interval_mask_uses_transport_only_for_inactive_intervals(variant) -> None:
    run = _run(variant, active_guidance_intervals=[])

    assert [(entry["t"], entry["r"]) for entry in run.counted.trace] == [
        (1.0, 0.75),
        (0.75, 0.5),
        (0.5, 0.25),
        (0.25, 0.125),
        (0.125, 0.0),
    ]
    assert all(entry["kind"] == "algorithm" for entry in run.counted.trace)


@pytest.mark.parametrize("variant", ["flow_map2_normalized", "paper_split"])
@pytest.mark.parametrize("active", [[], ["I2"], ["I1", "I2", "I3"]])
def test_diagnostics_toggle_is_bitwise_latent_invariant(variant, active) -> None:
    plain = _run(
        variant, active_guidance_intervals=active, collect_interval_diagnostics=False
    )
    diagnosed = _run(
        variant, active_guidance_intervals=active, collect_interval_diagnostics=True
    )

    assert torch.equal(plain.result.latent, diagnosed.result.latent)
    assert plain.result.nfe == diagnosed.result.nfe
    assert diagnosed.result.diagnostics["algorithm_nfe"] == diagnosed.result.nfe
    assert diagnosed.counted.nfe == diagnosed.result.nfe
    assert diagnosed.result.diagnostics["diagnostic_nfe"] == 6 + len(active)
    assert len(diagnosed.result.diagnostics["diagnostic_flow_map_trace"]) == 6 + len(
        active
    )
    assert all(
        entry["kind"] == "interval_diagnostic"
        for entry in diagnosed.result.diagnostics["diagnostic_flow_map_trace"]
    )


@pytest.mark.parametrize("variant", ["flow_map2_normalized", "paper_split"])
def test_explicit_all_active_mask_is_bitwise_r8_equivalent(variant) -> None:
    default = _run(variant)
    explicit = _run(variant, active_guidance_intervals=list(INTERVAL_IDS))

    assert torch.equal(default.result.latent, explicit.result.latent)
    assert default.result.nfe == explicit.result.nfe == 8
    assert (
        default.result.diagnostics["loss_history"]
        == explicit.result.diagnostics["loss_history"]
    )


@pytest.mark.parametrize("variant", ["flow_map2_normalized", "paper_split"])
def test_interval_diagnostics_have_complete_finite_schema(variant) -> None:
    active = ["I1", "I3"]
    run = _run(
        variant,
        active_guidance_intervals=active,
        collect_interval_diagnostics=True,
    )
    diagnostics = run.result.diagnostics["interval_diagnostics"]

    assert tuple(diagnostics) == INTERVAL_IDS
    for interval_id, metrics in diagnostics.items():
        assert set(metrics) == INTERVAL_METRIC_KEYS
        assert metrics["interval_id"] == interval_id
        assert metrics["active"] is (interval_id in active)
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                assert value.shape == (2,), key
                assert torch.isfinite(value).all(), key
        if interval_id not in active:
            assert torch.equal(metrics["gradient_norm"], torch.zeros(2))
            assert torch.equal(metrics["gradient_velocity_cosine"], torch.zeros(2))
            assert torch.equal(metrics["correction_norm"], torch.zeros(2))
            assert torch.equal(metrics["correction_transport_ratio"], torch.zeros(2))
            assert torch.equal(
                metrics["loss_before_correction"], metrics["loss_after_correction"]
            )


def test_both_algorithms_emit_the_same_interval_diagnostic_schema() -> None:
    official = _run(
        "flow_map2_normalized",
        active_guidance_intervals=["I2"],
        collect_interval_diagnostics=True,
    )
    paper = _run(
        "paper_split",
        active_guidance_intervals=["I2"],
        collect_interval_diagnostics=True,
    )

    official_rows = official.result.diagnostics["interval_diagnostics"]
    paper_rows = paper.result.diagnostics["interval_diagnostics"]
    assert tuple(official_rows) == tuple(paper_rows) == INTERVAL_IDS
    for interval_id in INTERVAL_IDS:
        assert set(official_rows[interval_id]) == set(paper_rows[interval_id])


@pytest.mark.parametrize("variant", ["flow_map2_normalized", "paper_split"])
def test_local_semigroup_residual_uses_read_only_direct_and_split_probes(
    variant,
) -> None:
    run = _run(
        variant,
        active_guidance_intervals=[],
        collect_interval_diagnostics=True,
    )
    algorithm_call_count = run.result.nfe
    diagnostic_calls = run.generator.calls[algorithm_call_count:]

    assert len(diagnostic_calls) == 6
    for interval_index, interval_id in enumerate(INTERVAL_IDS):
        direct_call = diagnostic_calls[2 * interval_index]
        split_call = diagnostic_calls[2 * interval_index + 1]
        assert direct_call[2] is False
        assert split_call[2] is False
        direct_endpoint = run.generator.flow_map(
            direct_call[3], torch.zeros(2, 4), t=direct_call[0], r=direct_call[1]
        )
        split_endpoint = run.generator.flow_map(
            split_call[3], torch.zeros(2, 4), t=split_call[0], r=split_call[1]
        )
        expected = symmetric_relative_l2(direct_endpoint, split_endpoint)
        actual = run.result.diagnostics["interval_diagnostics"][interval_id][
            "local_semigroup_residual"
        ]
        assert torch.equal(actual, expected)


def test_default_calls_preserve_r8_diagnostic_key_sets() -> None:
    official = _run("flow_map2_normalized")
    paper = _run("paper_split")

    assert set(official.result.diagnostics) == {
        "guided_times",
        "unguided_times",
        "sample_mode",
        "optimization_mode",
        "num_optim_iters",
        "step_size",
        "uses_adam",
        "adam_learning_rates",
        "loss_history",
    }
    assert set(paper.result.diagnostics) == {
        "guided_times",
        "unguided_times",
        "step_size",
        "loss_history",
    }


@pytest.mark.parametrize("variant", ["flow_map2_normalized", "paper_split"])
@pytest.mark.parametrize("active", [["I4"], ["I1", "I1"], "I1"])
def test_interval_mask_rejects_unknown_duplicate_or_scalar_ids(variant, active) -> None:
    with pytest.raises(ValueError):
        _run(variant, active_guidance_intervals=active)


@pytest.mark.parametrize("variant", ["flow_map2_normalized", "paper_split"])
def test_interval_mask_rejects_nonlocked_schedule_before_nfe(variant) -> None:
    with pytest.raises(ValueError, match="locked schedule"):
        _run(
            variant,
            active_guidance_intervals=["I1"],
            guided_times=[1.0, 0.5],
            unguided_times=[0.5, 0.0],
        )
