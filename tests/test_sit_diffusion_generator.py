from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _tiny_sit_diffusion_config() -> dict:
    return {
        "model_type": "sit_diffusion",
        "embedding_dim": 16,
        "image_size": 16,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 16,
        "sample_steps": 4,
        "train_cycle_steps": 4,
        "sampler": "ddim",
        "learned_null_condition": True,
        "sit_input_channels": 3,
        "sit_patch_size": 4,
        "sit_hidden_size": 32,
        "sit_depth": 2,
        "sit_num_heads": 4,
        "sit_mlp_ratio": 2.0,
        "sit_time_embedding_dim": 32,
        "diffusion_train_timesteps": 32,
        "diffusion_beta_schedule": "linear",
        "diffusion_prediction_type": "epsilon",
        "ddim_eta": 0.0,
    }


def test_flow_generator_config_supports_sit_diffusion_roundtrip() -> None:
    from safa.models.generator import FlowGeneratorConfig

    config = FlowGeneratorConfig.from_dict(_tiny_sit_diffusion_config())
    payload = config.to_dict()

    assert config.model_type == "sit_diffusion"
    assert config.sampler == "ddim"
    assert config.sample_steps == 4
    assert config.diffusion_train_timesteps == 32
    assert config.diffusion_beta_schedule == "linear"
    assert config.diffusion_prediction_type == "epsilon"
    assert payload["model_type"] == "sit_diffusion"
    assert payload["sample_steps"] == 4
    assert payload["diffusion_train_timesteps"] == 32
    assert payload["diffusion_prediction_type"] == "epsilon"
    assert payload["sit_data_space"] == "pixel"


def test_build_generator_supports_tiny_sit_diffusion_and_allows_multi_step_sampling() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_sit_diffusion_config())

    assert generator.config.model_type == "sit_diffusion"
    assert generator.config.sample_steps == 4
    assert generator.config.train_cycle_steps == 4
    assert generator.config.sampler == "ddim"


def test_sit_diffusion_loss_reports_flow_compatible_metrics() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_sit_diffusion_config())
    z = generator.make_null_condition(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.rand(2, 3, 16, 16)

    loss, metrics = generator.flow_matching_loss(x, z)
    loss.backward()

    assert tuple(loss.shape) == ()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["flow_matching_mse"])
    assert torch.isfinite(metrics["diffusion_mse"])
    assert metrics["diffusion_prediction_type"] == "epsilon"
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_sit_diffusion_sample_steps_1_and_4_are_finite_and_shape_correct() -> None:
    from safa.models.generator import build_generator

    generator = build_generator(_tiny_sit_diffusion_config())
    z = torch.zeros(2, 16)
    x_init = torch.randn(2, 3, 16, 16)

    one_step = generator.sample(z, steps=1, x_init=x_init, clamp_output=False)
    four_step = generator.sample(z, steps=4, x_init=x_init, clamp_output=False)

    assert tuple(one_step.shape) == (2, 3, 16, 16)
    assert tuple(four_step.shape) == (2, 3, 16, 16)
    assert torch.isfinite(one_step).all()
    assert torch.isfinite(four_step).all()


def test_sit_diffusion_latent_space_does_not_clamp_sample_output() -> None:
    from safa.models.generator import build_generator

    config = _tiny_sit_diffusion_config()
    config.update({"sit_data_space": "latent", "sit_input_channels": 4})
    generator = build_generator(config)
    z = torch.zeros(2, 16)
    x_init = torch.randn(2, 4, 16, 16) * 3.0

    sample = generator.sample(z, steps=1, x_init=x_init, clamp_output=True)

    assert tuple(sample.shape) == (2, 4, 16, 16)
    assert torch.isfinite(sample).all()
    detached = sample.detach()
    assert float(detached.min()) < 0.0 or float(detached.max()) > 1.0


def test_sit_diffusion_requires_ddim_sampler_but_not_one_sample_step() -> None:
    from safa.models.generator import FlowGeneratorConfig, build_generator

    config = _tiny_sit_diffusion_config()
    config["sample_steps"] = 8
    assert FlowGeneratorConfig.from_dict(config).sample_steps == 8
    assert build_generator(config).config.sample_steps == 8

    config["sampler"] = "meanflow"
    with pytest.raises(ValueError, match="sit_diffusion sampler must be 'ddim'"):
        build_generator(config)
