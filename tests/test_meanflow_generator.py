from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from safa.utils.hashing import sha256_file

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIVE_M_FM_PARAMETER_COUNT = 5_004_291


def _small_meanflow_config() -> dict:
    return {
        "model_type": "meanflow",
        "embedding_dim": 16,
        "image_size": 16,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 16,
        "sample_steps": 1,
        "train_cycle_steps": 1,
        "sampler": "meanflow",
        "learned_null_condition": True,
        "meanflow_ratio": 0.75,
        "meanflow_adaptive_weighting": True,
        "meanflow_norm_p": 0.75,
        "meanflow_norm_eps": 0.001,
        "meanflow_jvp_mode": "torch_func",
    }


def test_build_generator_supports_meanflow_and_preserves_model_type_roundtrip() -> None:
    from safa.models.generator import FlowGeneratorConfig, build_generator

    config = FlowGeneratorConfig.from_dict(_small_meanflow_config())

    assert config.model_type == "meanflow"
    assert config.to_dict()["model_type"] == "meanflow"
    generator = build_generator(config.to_dict())
    assert generator.config.model_type == "meanflow"
    assert generator.config.sample_steps == 1


def test_e9_meanflow_config_is_200_epoch_one_step_gpu6_safe_and_larger_than_5m() -> None:
    from safa.models.generator import build_generator
    from safa.training import g_loop

    path = REPO_ROOT / "configs" / "medium_v2" / "experiments" / "e9_meanflow_200ep.yaml"
    assert path.is_file()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["experiment_name"] == "e9_meanflow_200ep"
    assert config["device"] == "cuda:0"
    assert config["out_dir"] == "artifacts/checkpoints/g_medium_v2_meanflow_200ep"
    assert config["stages"]["stage2"]["epochs"] == 200
    assert config["generator"]["model_type"] == "meanflow"
    assert config["generator"]["sample_steps"] == 1
    assert config["generator"]["train_cycle_steps"] == 1
    assert config["generator"]["learned_null_condition"] is True
    assert config["stages"]["stage2"]["stage2_objective"]["flow_condition"] == "learned_null_condition"
    quality_eval = config["stages"]["stage2"]["quality_eval"]
    assert quality_eval["distribution_cuda_visible_devices"] == "6"
    assert quality_eval["distribution_device"] == "cuda:0"
    assert quality_eval["output_dir"] == "artifacts/eval/e9_meanflow_200ep/quality"

    g_loop._validate_train_g_config(config)
    generator_config = dict(config["generator"])
    generator_config["embedding_dim"] = config["embedding_dim"]
    generator_config["image_size"] = config["image_size"]
    generator = build_generator(generator_config)
    parameter_count = sum(parameter.numel() for parameter in generator.parameters())
    assert parameter_count > FIVE_M_FM_PARAMETER_COUNT


def test_meanflow_loss_uses_official_data_to_noise_path_and_velocity() -> None:
    from torch import nn

    from safa.models.generator import build_generator

    class RecorderField(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x_t = None
            self.t = None
            self.r = None
            self.z = None

        def forward(self, x_t, t, r, z):
            self.x_t = x_t.detach().clone()
            self.t = t.detach().clone()
            self.r = r.detach().clone()
            self.z = z.detach().clone()
            return torch.zeros_like(x_t)

    config = _small_meanflow_config()
    config["meanflow_jvp_mode"] = "first_order"
    config["meanflow_adaptive_weighting"] = False
    generator = build_generator(config)
    recorder = RecorderField()
    generator.vector_field = recorder

    def fixed_sample_t_r(batch_size: int, *, device, dtype, generator=None):
        del generator
        assert batch_size == 2
        return (
            torch.tensor([0.0, 1.0], device=device, dtype=dtype),
            torch.tensor([0.0, 0.0], device=device, dtype=dtype),
        )

    captured = {}

    def capture_weighted_loss(error, meanflow_target):
        captured["target"] = meanflow_target.detach().clone()
        return error.square().mean()

    generator._sample_t_r = fixed_sample_t_r
    generator._weighted_loss = capture_weighted_loss
    images = torch.stack(
        (
            torch.full((3, 16, 16), 0.75),
            torch.full((3, 16, 16), 0.25),
        )
    )
    z = torch.zeros(2, 16)
    seed = 17
    expected_data = images.mul(2.0).sub(1.0)
    expected_noise = torch.randn(
        expected_data.shape,
        generator=torch.Generator().manual_seed(seed),
        device=expected_data.device,
        dtype=expected_data.dtype,
    )

    generator.flow_matching_loss(images, z, generator=torch.Generator().manual_seed(seed))

    assert torch.allclose(recorder.x_t[0], expected_data[0])
    assert torch.allclose(recorder.x_t[1], expected_noise[1])
    assert torch.equal(recorder.t, torch.tensor([0.0, 1.0]))
    assert torch.equal(recorder.r, torch.tensor([0.0, 0.0]))
    assert torch.allclose(captured["target"], expected_noise - expected_data)


def test_meanflow_loss_returns_scalar_metrics_and_backpropagates() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_small_meanflow_config())
    images = torch.rand(2, 3, 16, 16)
    z = torch.randn(2, 16)

    loss, metrics = generator.flow_matching_loss(images, z)
    loss.backward()

    assert tuple(loss.shape) == ()
    assert torch.isfinite(loss)
    assert metrics["meanflow_jvp_mode"] == "torch_func"
    assert torch.isfinite(metrics["flow_matching_mse"])
    finite_grads = [
        parameter.grad.detach()
        for parameter in generator.parameters()
        if parameter.grad is not None and torch.isfinite(parameter.grad).all()
    ]
    assert finite_grads


def test_meanflow_sample_subtracts_one_step_mean_velocity_at_noise_endpoint() -> None:
    from torch import nn

    from safa.models.generator import build_generator

    class ConstantMeanVelocity(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x_t = None
            self.t = None
            self.r = None

        def forward(self, x_t, t, r, z):
            del z
            self.x_t = x_t.detach().clone()
            self.t = t.detach().clone()
            self.r = r.detach().clone()
            return torch.full_like(x_t, 0.25)

    generator = build_generator(_small_meanflow_config())
    field = ConstantMeanVelocity()
    generator.vector_field = field
    z = torch.zeros(1, 16)
    eps = torch.full((1, 3, 16, 16), 0.2)

    output = generator.sample(z, x_init=eps, clamp_output=False)

    assert torch.allclose(field.x_t, eps)
    assert torch.equal(field.t, torch.ones(1))
    assert torch.equal(field.r, torch.zeros(1))
    assert torch.allclose(output, (eps - 0.25 + 1.0) * 0.5)


def test_meanflow_sample_is_one_step_and_steps_none_matches_steps_one() -> None:
    from safa.models.generator import build_generator

    torch.manual_seed(123)
    generator = build_generator(_small_meanflow_config())
    z = torch.randn(2, 16)
    x_init = torch.randn(2, 3, 16, 16)

    default_steps = generator.sample(z, steps=None, x_init=x_init, clamp_output=False)
    one_step = generator.sample(z, steps=1, x_init=x_init, clamp_output=False)
    ignored_multi_step = generator.sample(z, steps=8, x_init=x_init, clamp_output=False)

    assert default_steps.shape == (2, 3, 16, 16)
    assert torch.allclose(default_steps, one_step)
    assert torch.allclose(default_steps, ignored_multi_step)


def test_meanflow_checkpoint_roundtrip_and_existing_flow_checkpoint_still_load() -> None:
    from safa.evaluation.runner import _load_generator
    from safa.models.generator import ConditionalFlowGenerator, build_generator

    meanflow = build_generator(_small_meanflow_config())
    flow_config = {
        "embedding_dim": 16,
        "image_size": 16,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 16,
        "sample_steps": 1,
        "train_cycle_steps": 1,
        "sampler": "euler",
    }
    flow = ConditionalFlowGenerator(flow_config)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        meanflow_path = tmp_path / "meanflow.pt"
        flow_path = tmp_path / "flow.pt"
        torch.save({"model_state_dict": meanflow.state_dict(), "model_config": meanflow.config.to_dict(), "metrics": {}}, meanflow_path)
        torch.save({"model_state_dict": flow.state_dict(), "model_config": flow.config.to_dict(), "metrics": {}}, flow_path)

        loaded_meanflow = _load_generator(
            str(meanflow_path),
            {
                "checkpoint_model": "raw",
                "checkpoint_sha256": sha256_file(meanflow_path),
            },
            "cpu",
        )
        loaded_flow = _load_generator(
            str(flow_path),
            {
                "checkpoint_model": "raw",
                "checkpoint_sha256": sha256_file(flow_path),
            },
            "cpu",
        )

    assert loaded_meanflow.config.model_type == "meanflow"
    assert loaded_flow.config.model_type == "conditional_flow_matching"
    assert tuple(loaded_meanflow(torch.randn(1, 16)).shape) == (1, 3, 16, 16)
    assert tuple(loaded_flow(torch.randn(1, 16)).shape) == (1, 3, 16, 16)


def test_generator_training_step_smoke_calls_meanflow_loss_with_learned_null_condition() -> None:
    from torch import nn

    from safa.models.generator import FlowGeneratorConfig, build_generator
    from safa.training.g_loop import _GeneratorTrainingStep, _LossWeightingRuntime, _Stage2ObjectiveRuntime

    config = FlowGeneratorConfig.from_dict(_small_meanflow_config())
    generator = build_generator(config.to_dict())
    objective = _Stage2ObjectiveRuntime(
        type="fm_only_probe",
        lambda_repr=0.0,
        point_weight=0.0,
        relation_weight=0.0,
        offdiag_only=True,
        flow_condition="learned_null_condition",
    )
    module = _GeneratorTrainingStep(
        generator,
        nn.Identity(),
        config,
        sampling_seed=123,
        loss_weighting=_LossWeightingRuntime(type="legacy"),
        stage2_objective=objective,
    )

    loss, flow_mse, secondary, flow_loss, secondary_loss = module(
        torch.rand(2, 3, 16, 16),
        torch.randn(2, 16),
        ["a", "b"],
        True,
        0.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(flow_mse)
    assert torch.equal(secondary, secondary_loss.detach())
    assert torch.equal(flow_loss.detach(), flow_mse)
    assert module.last_loss_metrics["flow_condition"] == "learned_null_condition"
    assert generator.null_condition.embedding.grad is not None
