from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _tiny_latent_consistency_config() -> dict:
    return {
        "model_type": "latent_consistency",
        "embedding_dim": 16,
        "image_size": 16,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 16,
        "sample_steps": 4,
        "train_cycle_steps": 4,
        "sampler": "consistency",
        "learned_null_condition": True,
        "sit_input_channels": 3,
        "sit_patch_size": 4,
        "sit_hidden_size": 32,
        "sit_depth": 2,
        "sit_num_heads": 4,
        "sit_mlp_ratio": 2.0,
        "sit_time_embedding_dim": 32,
        "consistency_train_timesteps": 32,
        "consistency_prediction_type": "x0",
        "consistency_target": "analytic_x0",
        "consistency_min_step_gap": 1,
    }


def test_flow_generator_config_supports_latent_consistency_roundtrip() -> None:
    from safa.models.generator import FlowGeneratorConfig

    config = FlowGeneratorConfig.from_dict(_tiny_latent_consistency_config())
    payload = config.to_dict()

    assert config.model_type == "latent_consistency"
    assert config.sampler == "consistency"
    assert config.sample_steps == 4
    assert config.consistency_train_timesteps == 32
    assert config.consistency_prediction_type == "x0"
    assert config.consistency_target == "analytic_x0"
    assert config.consistency_min_step_gap == 1
    assert payload["model_type"] == "latent_consistency"
    assert payload["sample_steps"] == 4
    assert payload["consistency_train_timesteps"] == 32
    assert payload["consistency_prediction_type"] == "x0"
    assert payload["consistency_target"] == "analytic_x0"
    assert payload["consistency_min_step_gap"] == 1
    assert payload["sit_data_space"] == "pixel"


def test_build_generator_supports_tiny_latent_consistency() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_latent_consistency_config())

    assert generator.config.model_type == "latent_consistency"
    assert generator.config.sample_steps == 4
    assert generator.config.train_cycle_steps == 4
    assert generator.config.sampler == "consistency"


def test_latent_consistency_normalizes_analytic_x0_timesteps_to_unit_interval() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_latent_consistency_config())
    timesteps = torch.tensor([0, 31], dtype=torch.long)

    normalized = generator._normalize_timesteps(timesteps, dtype=torch.float32)

    assert torch.allclose(normalized, torch.tensor([0.0, 1.0]), atol=1.0e-6)
    assert normalized[-1].item() > 0.99


def test_latent_consistency_loss_reports_flow_compatible_metrics() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_latent_consistency_config())
    z = generator.make_null_condition(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.rand(2, 3, 16, 16)

    loss, metrics = generator.flow_matching_loss(x, z)
    loss.backward()

    assert tuple(loss.shape) == ()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["flow_matching_mse"])
    assert torch.isfinite(metrics["consistency_mse"])
    assert metrics["consistency_prediction_type"] == "x0"
    assert metrics["consistency_target"] == "analytic_x0"
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_latent_consistency_sample_steps_1_and_4_are_finite_and_shape_correct() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_latent_consistency_config())
    z = torch.zeros(2, 16)
    x_init = torch.randn(2, 3, 16, 16)

    one_step = generator.sample(z, steps=1, x_init=x_init, clamp_output=False)
    four_step = generator.sample(z, steps=4, x_init=x_init, clamp_output=False)

    assert tuple(one_step.shape) == (2, 3, 16, 16)
    assert tuple(four_step.shape) == (2, 3, 16, 16)
    assert torch.isfinite(one_step).all()
    assert torch.isfinite(four_step).all()


def test_latent_consistency_latent_space_does_not_clamp_sample_output() -> None:
    from torch import nn

    from safa.models.generator import build_generator

    class ConstantX0(nn.Module):
        def forward(self, x, t, z):
            del t, z
            return torch.full_like(x, 2.0)

    config = _tiny_latent_consistency_config()
    config.update({"sit_data_space": "latent", "sit_input_channels": 4})
    generator = build_generator(config)
    generator.denoiser = ConstantX0()
    z = torch.zeros(2, 16)
    x_init = torch.randn(2, 4, 16, 16)

    sample = generator.sample(z, steps=1, x_init=x_init, clamp_output=True)

    assert tuple(sample.shape) == (2, 4, 16, 16)
    assert torch.isfinite(sample).all()
    assert torch.equal(sample, torch.full_like(sample, 2.0))


def test_latent_consistency_requires_consistency_sampler_but_not_one_sample_step() -> None:
    from safa.models.generator import FlowGeneratorConfig, build_generator

    config = _tiny_latent_consistency_config()
    config["sample_steps"] = 8
    assert FlowGeneratorConfig.from_dict(config).sample_steps == 8
    assert build_generator(config).config.sample_steps == 8

    config["sampler"] = "ddim"
    with pytest.raises(ValueError, match="latent_consistency sampler must be 'consistency'"):
        build_generator(config)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"consistency_train_timesteps": 1}, "consistency_train_timesteps must be greater than 1"),
        ({"consistency_prediction_type": "epsilon"}, "consistency_prediction_type must be one of"),
        ({"consistency_target": "teacher"}, "consistency_target must be one of"),
        ({"consistency_min_step_gap": 0}, "consistency_min_step_gap must be positive"),
        ({"consistency_min_step_gap": 32}, "consistency_min_step_gap must be less than consistency_train_timesteps"),
    ],
)
def test_latent_consistency_validates_config(updates, match) -> None:
    from safa.models.generator import build_generator

    config = _tiny_latent_consistency_config()
    config.update(updates)

    with pytest.raises(ValueError, match=match):
        build_generator(config)
